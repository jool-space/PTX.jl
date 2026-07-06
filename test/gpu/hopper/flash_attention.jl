# Hopper FlashAttention forward — simplified port of
# pyptx/examples/hopper/experimental/flash_attention_hopper.py.
#
# v0 scope (faithful structure, smaller tiles):
#   - BM = 64 query rows per CTA (pyptx uses 128 with top/bottom split)
#   - BN = 16, HD = 16 (single wgmma per QK^T and per PV)
#   - one warpgroup (128 threads) per CTA, one Q-slice (no top/bottom split)
#   - sequential K-loop (no K/V double-buffering)
#   - bf16 inputs, f32 acc, f32 output (skip the bf16 output conversion)
#
# Algorithm per CTA:
#   1. TMA-load Q tile (BM × HD), wait
#   2. For each KV block (kv_off = 0, BN, 2*BN, ..., N_seq):
#      a. TMA-load K tile (HD × BN), V tile (BN × HD), wait
#      b. wgmma m64n16k16 (Q × K_t) → qk_frag (8 f32 per lane)
#      c. qk_frag *= qk_scale
#      d. Online softmax: per-row max via local + warp shfl (width=4),
#         compute alpha = exp2(m_old - m_new), exp2(qk_frag - m_new),
#         sum via warp shfl, l := l*alpha + ra_sum, acc := acc*alpha
#      e. Convert qk_frag → bf16, store to sP (BM × BN K-fast in SMEM)
#      f. bar.sync (warpgroup-wide)
#      g. wgmma m64n16k16 (sP × V) → acc (accumulating, m64n16k16)
#      h. bar.sync, m := m_new
#   3. acc /= l (per row)
#   4. v2.f32 store to O (BM × HD)
#
# Memory layouts:
#   Q  : Julia (HD, BM) col-major → HD-fast (K-fast for QK^T wgmma A)
#   K  : Julia (HD, N_seq) col-major → HD-fast (K-fast for QK^T wgmma B; K_t-shaped)
#   V  : Julia (HD, N_seq) col-major → HD-fast (K-fast for PV wgmma B with K=BN)
#   O  : Julia (HD, BM) col-major → HD-fast (N-innermost for v2.f32 stores)
#
# Wait — for V: PV wgmma is m64n{HD}k{BN}. A=sP (M=BM, K=BN), B=V (K=BN, N=HD).
# B's K-fast means BN-fast — so within each N (HD-col), BN K-rows are contiguous.
# So Julia layout for V (per-tile): (BN, HD) col-major = BN-fast within each
# HD col. Globally V is (N_seq, HD), Julia (N_seq, HD) col-major → N_seq-fast.
# For each tile (rows kv_off..kv_off+BN-1), the BN values within each HD-col
# are contiguous. ✓

using PTX: layout_for_a, wgmma_descriptor, smem_addr_u32,
           tensor_map_tile_2d, CuTensorMap
using CUDACore
using Random

const FA_BM        = 64
const FA_BN        = 16
const FA_HD        = 16
const FA_THREADS   = 128
const FA_LOG2E     = Float32(1.4426950408889634)

# Per-iter TMA bytes:
#   K tile: HD × BN × 2 = 16 × 16 × 2 = 512  (K_t-shaped)
#   V tile: BN × HD × 2 = 16 × 16 × 2 = 512
const FA_KV_LOAD_BYTES = FA_HD * FA_BN * 2 + FA_BN * FA_HD * 2

# wgmma m64n16k16 output: 8 f32 frags per lane.
# pyptx _frag_positions(16) layout:
#   frag 0,1 -> (row_a, col_base + 0), (row_a, col_base + 1)
#   frag 2,3 -> (row_b, col_base + 0), (row_b, col_base + 1)
#   frag 4,5 -> (row_a, col_base + 8), (row_a, col_base + 9)
#   frag 6,7 -> (row_b, col_base + 8), (row_b, col_base + 9)
const FA_ROW_A_FRAGS = (1, 2, 5, 6)   # 1-indexed for Julia
const FA_ROW_B_FRAGS = (3, 4, 7, 8)

# width-4 butterfly reduce (4 lanes per group cover the 4 col-pairs of a row).
@inline function _fa_reduce_max_w4(v::Float32)
    membermask = UInt32(0xFFFFFFFF)
    # c-operand for shfl.sync.bfly with width=4: (32-4)<<8 | (4-1) = 0x1C03.
    c = UInt32(0x1C03)
    u = reinterpret(UInt32, v)
    u_par = ptx"shfl.sync.bfly.b32"(u, UInt32(1), c, membermask)
    v = ptx"max.f32"(v, reinterpret(Float32, u_par))
    u = reinterpret(UInt32, v)
    u_par = ptx"shfl.sync.bfly.b32"(u, UInt32(2), c, membermask)
    v = ptx"max.f32"(v, reinterpret(Float32, u_par))
    v
end

@inline function _fa_reduce_sum_w4(v::Float32)
    membermask = UInt32(0xFFFFFFFF)
    c = UInt32(0x1C03)
    u = reinterpret(UInt32, v)
    u_par = ptx"shfl.sync.bfly.b32"(u, UInt32(1), c, membermask)
    v = ptx"add.f32"(v, reinterpret(Float32, u_par))
    u = reinterpret(UInt32, v)
    u_par = ptx"shfl.sync.bfly.b32"(u, UInt32(2), c, membermask)
    v = ptx"add.f32"(v, reinterpret(Float32, u_par))
    v
end

# Pack two f32 → bf16 → packed UInt32 (bf16 hi:lo).
@inline function _fa_pack_bf16x2(lo::Float32, hi::Float32)
    bf_lo = ptx"cvt.rn.bf16.f32"(lo)
    bf_hi = ptx"cvt.rn.bf16.f32"(hi)
    # mov.b32 %r, {bf_lo, bf_hi} — pack two b16 into one b32.
    ptx"mov.b32"((bf_lo, bf_hi))
end

# B32 GMMA swizzle (CUTLASS Swizzle<1, 4, 3>): physical = logical XOR
# ((logical & 0x80) >> 3). Per pyptx/pyptx/smem.py apply_swizzle. Required
# whenever we MANUALLY store to a SMEM tile that wgmma reads through a B32
# descriptor — TMA loads handle the swizzle for free, but `st.shared.b32`
# does not. Three ALU ops; cheap enough to inline at every store site.
@inline _fa_swizzle_b32(logical::UInt32) =
    logical ⊻ ((logical & UInt32(0x80)) >> UInt32(3))

function _fa_kernel!(
        O::CuDeviceVector{Float32, 1},
        tma_Q::PTX.TMADescriptorPtr,
        tma_K::PTX.TMADescriptorPtr,
        tma_V::PTX.TMADescriptorPtr,
        N_seq::Int32,
        qk_scale_log2e::Float32)

    sQ    = CuStaticSharedArray(UInt16, FA_BM * FA_HD)    # 64 × 16 bf16 = 2048 B
    sK    = CuStaticSharedArray(UInt16, FA_HD * FA_BN)    # 16 × 16 bf16 = 512 B
    sV    = CuStaticSharedArray(UInt16, FA_BN * FA_HD)    # 16 × 16 bf16 = 512 B
    sP    = CuStaticSharedArray(UInt16, FA_BM * FA_BN)    # 64 × 16 bf16 = 2048 B
    bar_q = CuStaticSharedArray(UInt64, 1)
    bar_k = CuStaticSharedArray(UInt64, 1)

    q_ptr = pointer(sQ); k_ptr = pointer(sK); v_ptr = pointer(sV); p_ptr = pointer(sP)
    bq    = pointer(bar_q); bk = pointer(bar_k)
    q_addr = smem_addr_u32(q_ptr)
    k_addr = smem_addr_u32(k_ptr)
    v_addr = smem_addr_u32(v_ptr)
    p_addr = smem_addr_u32(p_ptr)

    tid     = ptx"mov.u32"(sreg"tid.x")
    cta_x   = ptx"mov.u32"(sreg"ctaid.x")
    q_row_base = cta_x * UInt32(FA_BM)

    # --- 1. Init mbarriers + load Q ---------------------------------------
    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(bq, UInt32(1))
        ptx"mbarrier.init.shared.b64"(bk, UInt32(1))
        ptx"fence.proxy.async.shared::cta"()
        ptx"mbarrier.arrive.expect_tx.shared.b64"(bq, UInt32(FA_BM * FA_HD * 2))
        ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
            q_ptr, tma_Q, Int32(0), reinterpret(Int32, q_row_base), bq)
    end
    ptx"bar.sync"(Val(0))

    while !ptx"mbarrier.test_wait.parity.shared.b64"(bq, UInt32(0))
    end

    # --- 2. Build wgmma descriptors ----------------------------------------
    # Q is (BM, HD=16) K-fast in SMEM → A K-major canonical, row_bytes = HD*2 = 32 → B32.
    # sP is (BM, BN=16) K-fast → A K-major, row_bytes = BN*2 = 32 → B32.
    # K is (HD, BN=16) K-fast → B K-major canonical (use layout_for_a with m=BN).
    # V is (BN, HD=16) K-fast → B K-major (use layout_for_a with m=HD).
    lQ = layout_for_a(dtype = :bf16, m = FA_BM, k = FA_HD)
    lP = layout_for_a(dtype = :bf16, m = FA_BM, k = FA_BN)
    lK = layout_for_a(dtype = :bf16, m = FA_BN, k = FA_HD)
    lV = layout_for_a(dtype = :bf16, m = FA_HD, k = FA_BN)
    q_desc = wgmma_descriptor(q_addr;
        leading_byte_offset = lQ.leading_byte_offset,
        stride_byte_offset  = lQ.stride_byte_offset, swizzle = lQ.layout_type)
    p_desc = wgmma_descriptor(p_addr;
        leading_byte_offset = lP.leading_byte_offset,
        stride_byte_offset  = lP.stride_byte_offset, swizzle = lP.layout_type)
    k_desc = wgmma_descriptor(k_addr;
        leading_byte_offset = lK.leading_byte_offset,
        stride_byte_offset  = lK.stride_byte_offset, swizzle = lK.layout_type)
    v_desc = wgmma_descriptor(v_addr;
        leading_byte_offset = lV.leading_byte_offset,
        stride_byte_offset  = lV.stride_byte_offset, swizzle = lV.layout_type)

    # --- 3. Online-softmax state per lane ---------------------------------
    m_a  = Float32(-1f30)
    m_b  = Float32(-1f30)
    l_a  = 0f0
    l_b  = 0f0
    acc  = ntuple(_ -> 0f0, Val(8))

    # --- 4. KV loop --------------------------------------------------------
    phase_k = UInt32(0)
    kv_off  = Int32(0)
    while kv_off < N_seq
        if tid == UInt32(0)
            ptx"fence.proxy.async.shared::cta"()
            ptx"mbarrier.arrive.expect_tx.shared.b64"(bk, UInt32(FA_KV_LOAD_BYTES))
            # K_t tile at coord (kv_off, 0) — innermost=HD, outer=BN of K_t.
            # K_t is (HD, N_seq) row-major in our K-fast convention.
            ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                k_ptr, tma_K, Int32(0), kv_off, bk)
            # V tile at coord (kv_off, 0) — V is (HD, N_seq) col-major-of-(BN, HD)
            # per K-tile. Innermost = BN, outer = HD.
            ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                v_ptr, tma_V, kv_off, Int32(0), bk)
        end
        ptx"bar.sync"(Val(0))

        while !ptx"mbarrier.test_wait.parity.shared.b64"(bk, phase_k)
        end

        # ----- 4a. QK^T = Q × K_t (m64n16k16) ----------------------------
        ptx"fence.proxy.async.shared::cta"()
        ptx"wgmma.fence.sync.aligned"()
        qk = ntuple(_ -> 0f0, Val(8))
        qk = ptx"wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16"(
            qk, q_desc, k_desc, false)
        ptx"wgmma.commit_group.sync.aligned"()
        ptx"wgmma.wait_group.sync.aligned"(Val(0))

        # ----- 4b. Scale by qk_scale*log2e (unrolled — keeps tuple stable) -
        qk = (qk[1] * qk_scale_log2e, qk[2] * qk_scale_log2e,
              qk[3] * qk_scale_log2e, qk[4] * qk_scale_log2e,
              qk[5] * qk_scale_log2e, qk[6] * qk_scale_log2e,
              qk[7] * qk_scale_log2e, qk[8] * qk_scale_log2e)

        # ----- 4c. Per-row max + warp-shfl reduce (width=4) --------------
        # ROW_A_FRAGS = (1, 2, 5, 6), ROW_B_FRAGS = (3, 4, 7, 8) — per-lane.
        row_a_new = max(qk[1], max(qk[2], max(qk[5], qk[6])))
        row_b_new = max(qk[3], max(qk[4], max(qk[7], qk[8])))
        row_a_new = _fa_reduce_max_w4(row_a_new)
        row_b_new = _fa_reduce_max_w4(row_b_new)

        m_a_new = max(m_a, row_a_new)
        m_b_new = max(m_b, row_b_new)

        # alpha = exp2(m_old - m_new); applied to old l + old acc.
        alpha_a = ptx"ex2.approx.f32"(m_a - m_a_new)
        alpha_b = ptx"ex2.approx.f32"(m_b - m_b_new)

        # qk_frag := exp2(qk_frag - m_new), m_new per row-half.
        qk = (ptx"ex2.approx.f32"(qk[1] - m_a_new),
              ptx"ex2.approx.f32"(qk[2] - m_a_new),
              ptx"ex2.approx.f32"(qk[3] - m_b_new),
              ptx"ex2.approx.f32"(qk[4] - m_b_new),
              ptx"ex2.approx.f32"(qk[5] - m_a_new),
              ptx"ex2.approx.f32"(qk[6] - m_a_new),
              ptx"ex2.approx.f32"(qk[7] - m_b_new),
              ptx"ex2.approx.f32"(qk[8] - m_b_new))

        # Per-row sum + warp shfl.
        ra_sum = qk[1] + qk[2] + qk[5] + qk[6]
        rb_sum = qk[3] + qk[4] + qk[7] + qk[8]
        ra_sum = _fa_reduce_sum_w4(ra_sum)
        rb_sum = _fa_reduce_sum_w4(rb_sum)

        l_a = l_a * alpha_a + ra_sum
        l_b = l_b * alpha_b + rb_sum

        # Rescale acc by alpha (row-wise).
        acc = (acc[1] * alpha_a, acc[2] * alpha_a,
               acc[3] * alpha_b, acc[4] * alpha_b,
               acc[5] * alpha_a, acc[6] * alpha_a,
               acc[7] * alpha_b, acc[8] * alpha_b)

        # ----- 4d. Convert qk to bf16, store to sP (K-fast: row * BN*2 + col*2)
        wid   = tid >> UInt32(5)
        lane  = tid & UInt32(31)
        row_a = (wid << UInt32(4)) + (lane >> UInt32(2))   # 0..63 (per-warp 16 rows × 4 warps)
        row_b = row_a + UInt32(8)
        col_b = (lane & UInt32(3)) << UInt32(1)            # 0,2,4,6 — col_base for first frag-pair

        # 4 packed-bf16x2 stores: (row_a/b, col_b + {0, 8}) for each of the 2 pairs.
        # Each store writes 2 consecutive cols in sP at the given (row, col).
        # sP is K-fast: byte offset = (row * BN + col) * 2.
        pack0 = _fa_pack_bf16x2(qk[1], qk[2])              # row_a, col_b   .. col_b+1
        pack1 = _fa_pack_bf16x2(qk[3], qk[4])              # row_b, col_b   .. col_b+1
        pack2 = _fa_pack_bf16x2(qk[5], qk[6])              # row_a, col_b+8 .. col_b+9
        pack3 = _fa_pack_bf16x2(qk[7], qk[8])              # row_b, col_b+8 .. col_b+9

        # Logical (un-swizzled) byte offsets into sP. wgmma reads through a
        # B32 swizzle, so we apply the same swizzle here to land each pair
        # at the matching physical address.
        log_a0 = (row_a * UInt32(FA_BN) + col_b) * UInt32(2)
        log_b0 = (row_b * UInt32(FA_BN) + col_b) * UInt32(2)
        log_a8 = (row_a * UInt32(FA_BN) + col_b + UInt32(8)) * UInt32(2)
        log_b8 = (row_b * UInt32(FA_BN) + col_b + UInt32(8)) * UInt32(2)
        ptx"st.shared.b32"(p_ptr + Int(_fa_swizzle_b32(log_a0)), pack0)
        ptx"st.shared.b32"(p_ptr + Int(_fa_swizzle_b32(log_b0)), pack1)
        ptx"st.shared.b32"(p_ptr + Int(_fa_swizzle_b32(log_a8)), pack2)
        ptx"st.shared.b32"(p_ptr + Int(_fa_swizzle_b32(log_b8)), pack3)

        ptx"bar.sync"(Val(0))   # all threads must see sP writes before PV

        # ----- 4e. PV = sP × V (m64n16k16, accumulating) ----------------
        ptx"fence.proxy.async.shared::cta"()
        ptx"wgmma.fence.sync.aligned"()
        acc = ptx"wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16"(
            acc, p_desc, v_desc, true)
        ptx"wgmma.commit_group.sync.aligned"()
        ptx"wgmma.wait_group.sync.aligned"(Val(0))
        ptx"bar.sync"(Val(0))

        # ----- 4f. Update state -----------------------------------------
        m_a = m_a_new
        m_b = m_b_new

        kv_off  += Int32(FA_BN)
        phase_k ⊻= UInt32(1)
    end

    # --- 5. Normalize acc /= l ---------------------------------------------
    inv_a = ptx"rcp.approx.f32"(l_a)
    inv_b = ptx"rcp.approx.f32"(l_b)
    acc   = (acc[1] * inv_a, acc[2] * inv_a,
             acc[3] * inv_b, acc[4] * inv_b,
             acc[5] * inv_a, acc[6] * inv_a,
             acc[7] * inv_b, acc[8] * inv_b)

    # --- 6. Store acc to O (HD-innermost, f32) -----------------------------
    wid   = tid >> UInt32(5)
    lane  = tid & UInt32(31)
    row_a = (wid << UInt32(4)) + (lane >> UInt32(2))
    row_b = row_a + UInt32(8)
    col_b = (lane & UInt32(3)) << UInt32(1)
    abs_row_a = row_a + q_row_base
    abs_row_b = row_b + q_row_base

    po = pointer(O)
    off_a0 = (abs_row_a * UInt32(FA_HD) + col_b) * UInt32(4)
    off_b0 = (abs_row_b * UInt32(FA_HD) + col_b) * UInt32(4)
    off_a8 = (abs_row_a * UInt32(FA_HD) + col_b + UInt32(8)) * UInt32(4)
    off_b8 = (abs_row_b * UInt32(FA_HD) + col_b + UInt32(8)) * UInt32(4)
    ptx"st.global.v2.f32"(po + Int(off_a0), (acc[1], acc[2]))
    ptx"st.global.v2.f32"(po + Int(off_b0), (acc[3], acc[4]))
    ptx"st.global.v2.f32"(po + Int(off_a8), (acc[5], acc[6]))
    ptx"st.global.v2.f32"(po + Int(off_b8), (acc[7], acc[8]))
    return nothing
end

# --- Cross-arch ptxas validation -----------------------------------------

@testset "flash_attention kernel compiles at sm_90a" begin
    types = Tuple{CuDeviceVector{Float32, 1},
                  PTX.TMADescriptorPtr,
                  PTX.TMADescriptorPtr,
                  PTX.TMADescriptorPtr,
                  Int32, Float32}
    @test ptxas_compiles(_fa_kernel!, types; cap = v"9.0", feature_set = :arch)

    ptx = emit_ptx(_fa_kernel!, types; cap = v"9.0", feature_set = :arch)
    @test occursin("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16", ptx)
    @test occursin("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes", ptx)
    @test occursin("shfl.sync.bfly.b32", ptx)
    @test occursin("ex2.approx.f32", ptx)
    @test occursin("rcp.approx.f32", ptx)
    @test occursin("st.shared.b32", ptx)
    # tier-1 vec ld/st canonicalizes float stores to the `.b32` bit spelling
    @test occursin("st.global.v2.b32", ptx)
    @test occursin("cvt.rn.bf16.f32", ptx)
end

# --- Runtime — H100 only -------------------------------------------------

if v"9.0" <= DEV_CAP < v"10.0"
    function _fa_cpu_ref(Q::Array{Float32, 2}, K::Array{Float32, 2},
                         V::Array{Float32, 2}, sm_scale::Float32)
        # Q (M, HD), K (N, HD), V (N, HD).
        M, HD = size(Q); N, _ = size(K)
        Qb = bf16_to_f32.(bf16_bits.(Q))
        Kb = bf16_to_f32.(bf16_bits.(K))
        Vb = bf16_to_f32.(bf16_bits.(V))
        # scores = Q × K^T (M × N)
        scores = Qb * Kb' .* sm_scale
        # softmax over N axis (per row)
        rmax = maximum(scores; dims = 2)
        ex   = exp.(scores .- rmax)
        rsum = sum(ex; dims = 2)
        P    = ex ./ rsum
        return P * Vb
    end

    function _fa_pack_K_fast_Q(Q::Array{Float32, 2})
        # Q (M, HD) → Julia (HD, M) col-major (HD-fast).
        M, HD = size(Q)
        out = Array{UInt16}(undef, HD, M)
        for m in 1:M, h in 1:HD
            out[h, m] = bf16_bits(Q[m, h])
        end
        out
    end

    function _fa_pack_HD_fast(X::Array{Float32, 2})
        # X (N, HD) → Julia (HD, N) col-major (HD-fast).
        N, HD = size(X)
        out = Array{UInt16}(undef, HD, N)
        for n in 1:N, h in 1:HD
            out[h, n] = bf16_bits(X[n, h])
        end
        out
    end

    function _fa_pack_BN_fast(X::Array{Float32, 2})
        # X (N, HD) → Julia (N, HD) col-major (N-fast = BN-fast within tile).
        N, HD = size(X)
        out = Array{UInt16}(undef, N, HD)
        for n in 1:N, h in 1:HD
            out[n, h] = bf16_bits(X[n, h])
        end
        out
    end

    function _run_fa(M_q::Int, N_seq::Int; rtol = 5e-2, atol = 5e-2)
        @assert M_q % FA_BM == 0 "M_q must be divisible by BM=$FA_BM"
        @assert N_seq % FA_BN == 0 "N_seq must be divisible by BN=$FA_BN"
        HD = FA_HD
        sm_scale = 1f0 / sqrt(Float32(HD))

        rng = MersenneTwister(M_q * 131 + N_seq + HD * 7)
        Q = Float32.(randn(rng, M_q, HD)) .* 0.3f0
        K = Float32.(randn(rng, N_seq, HD)) .* 0.3f0
        V = Float32.(randn(rng, N_seq, HD)) .* 0.3f0

        Q_d = CuArray(_fa_pack_K_fast_Q(Q))    # (HD, M_q) col-major
        K_d = CuArray(_fa_pack_HD_fast(K))     # (HD, N_seq) col-major
        V_d = CuArray(_fa_pack_BN_fast(V))     # (N_seq, HD) col-major

        # TMA descriptors. All innermost = HD (or BN for V).
        # Q: per-tile (BM, HD); descriptor rows=M_q, cols=HD, innermost=HD.
        tmap_Q = tensor_map_tile_2d(:bf16, pointer(Q_d), M_q, HD, FA_BM, HD;
                                    swizzle = :B32)
        # K: per-tile (BN, HD) reading from (N_seq, HD)? No — K is (HD, N_seq)
        # in memory, per tile we want (BN=16, HD=16). Descriptor:
        # rows=N_seq, cols=HD, innermost=HD, box=(BN, HD).
        tmap_K = tensor_map_tile_2d(:bf16, pointer(K_d), N_seq, HD, FA_BN, HD;
                                    swizzle = :B32)
        # V: (N_seq, HD) col-major. Descriptor:
        # rows=HD, cols=N_seq, innermost=N_seq → can't be 16-wide. Re-think.
        # Actually V is laid out as (N_seq, HD) Julia col-major → N_seq-fast.
        # Per tile we want (BN=16, HD=16). Descriptor:
        # rows=HD, cols=N_seq, innermost=N_seq, box=(HD=16, BN=16).
        # Coord = (kv_off, 0) — kv_off in N_seq axis, 0 in HD axis.
        tmap_V = tensor_map_tile_2d(:bf16, pointer(V_d), HD, N_seq, HD, FA_BN;
                                    swizzle = :B32)

        q_const = upload_tma_descriptor(tmap_Q)
        k_const = upload_tma_descriptor(tmap_K)
        v_const = upload_tma_descriptor(tmap_V)

        O_d = CUDACore.zeros(Float32, M_q * HD)
        @cuda threads = FA_THREADS blocks = (M_q ÷ FA_BM, 1, 1) _fa_kernel!(
            O_d, q_const.ptr, k_const.ptr, v_const.ptr, Int32(N_seq), sm_scale * FA_LOG2E)
        CUDACore.synchronize()

        O_packed = reshape(Array(O_d), HD, M_q)
        O_got = Array{Float32}(undef, M_q, HD)
        for m in 1:M_q, h in 1:HD
            O_got[m, h] = O_packed[h, m]
        end

        O_ref = _fa_cpu_ref(Q, K, V, sm_scale)
        ≈(O_got, O_ref; rtol = rtol, atol = atol)
    end

    @testset "FA forward M_q=$M_q N_seq=$N_seq HD=$FA_HD" for (M_q, N_seq) in [
            (64, 16),
            (64, 64),
            (128, 128),
        ]
        @test _run_fa(M_q, N_seq)
    end
end
