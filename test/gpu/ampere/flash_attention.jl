# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.0
# Portable mma.sync FlashAttention forward (bf16, head_dim 64/128) —
# FlashAttention-2 algorithm on the m16n8k16 warp-MMA interface, the one
# tensor-core interface every sm_80+ part shares (wgmma is sm_90a-only,
# tcgen05 is datacenter-Blackwell-only). This is the repo's portable
# floor and the attention kernel that runs in EVERY gpu CI lane; the
# per-arch ceilings are gpu/hopper/flash_attention.jl (wgmma) and
# gpu/blackwell/flash_attention_defs.jl (tcgen05).
#
# Structure:
#   * grid (seqlen/64, batch*heads); each CTA owns 64 query rows of one
#     head. 4 warps; warp w owns rows [16w, 16w+16) of the CTA tile.
#     Fragment coordinates per lane (m16n8k16): gid = lane>>2 is the row
#     within the 16-row warp tile (rows gid and gid+8), col_lo = 2*(lane&3)
#     the column pair — same layout as gpu/ampere/gemm_pipelined.jl.
#   * K tiles (64 × HD) double-buffered through SMEM via cp.async: the
#     K(kt+1) copy flies under tile kt's compute. Two bar.syncs per tile.
#   * S = Q·Kᵀ per warp as 8 m16n8k16 accumulator tiles; Q fragments are
#     loaded to registers ONCE (Q is never re-read after the prologue).
#   * Online softmax entirely in registers: a row lives in one warp quad
#     (lanes 4r..4r+3), so row max/sum are width-4 butterflies —
#     warp_reduce(op, v, Val(4)) — no shared memory, no block barrier.
#     m/l carry in two registers per lane (rows gid, gid+8).
#   * P stays in registers: the m16n8 accumulator fragment layout IS the
#     m16n8k16 A fragment layout, so exp'd S packs pairwise (bf16x2)
#     straight into PV's A operand — the FA2 register-reuse trick.
#   * V is staged TRANSPOSED into SMEM (Vᵀ: one b32 per (d, seq-pair),
#     seq-contiguous) by plain v4 global loads + halfword repack, so
#     PV's B fragments are single b32 loads of two adjacent k values.
#     This is the correct-by-construction stand-in for ldmatrix.trans;
#     swapping the staging for ldmatrix + cp.async on V is the known
#     perf follow-up and changes only the staging block and B reads.
#   * Output rows normalize by rcp.approx(l) and store packed bf16x2.
#
# Tensors are passed 2D as (batch*heads*seqlen, head_dim) bf16 row-major
# (the blackwell kernel's convention). seqlen must be a multiple of 64.
# CAUSAL masks col > row within each head.

using Random
using PTX.Utils: @unrolled
using PTX.Warps: warp_reduce
using PTX: bf16x2_pack

const FAM_BM      = 64                  # CTA query rows
const FAM_BN      = 64                  # KV tile size
const FAM_WARPS   = 4
const FAM_THREADS = 32 * FAM_WARPS
const FAM_LOG2E   = 1.4426950408889634

# SMEM (bytes): Q tile, 2 K stages, Vᵀ tile.
fam_q_bytes(HD)    = FAM_BM * HD * 2
fam_k_bytes(HD)    = FAM_BN * HD * 2
fam_smem_bytes(HD) = fam_q_bytes(HD) + 2 * fam_k_bytes(HD) + HD * FAM_BN * 2

function fam_kernel!(
        O::CuDeviceVector{UInt16},
        Q::CuDeviceVector{UInt16},
        K::CuDeviceVector{UInt16},
        V::CuDeviceVector{UInt16},
        seqlen::UInt32,
        qk2::Float32,                   # sm_scale * log2(e)
        ::Val{HD}, ::Val{CAUSAL}) where {HD, CAUSAL}

    D8  = HD ÷ 8                        # 16-byte chunks per tile row
    D16 = HD ÷ 16                       # k-chunks over head_dim

    smem = @inbounds CuDynamicSharedArray(UInt8, fam_smem_bytes(HD))
    smem_base = pointer(smem)
    q_base  = smem_base
    k_base0 = smem_base + fam_q_bytes(HD)
    k_base1 = k_base0 + fam_k_bytes(HD)
    # Vᵀ as a b32 array: element (d, kp) at d*(BN÷2) + kp packs
    # V[2kp, d] (low half) and V[2kp+1, d] (high half).
    vt = @inbounds CuDynamicSharedArray(UInt32, HD * (FAM_BN ÷ 2),
                                        fam_q_bytes(HD) + 2 * fam_k_bytes(HD))
    vt_ptr = pointer(vt)

    qb = ptx"mov.u32"(sreg"ctaid.x")    # query block within the sequence
    bh = ptx"mov.u32"(sreg"ctaid.y")    # batch*head slice

    tid    = ptx"mov.u32"(sreg"tid.x")
    warp   = Int(tid >> UInt32(5))
    lane   = Int(tid & UInt32(31))
    gid    = lane >> 2
    col_lo = (lane & 3) << 1

    seq = Int(seqlen)
    head_row0 = Int(bh) * seq           # first global row of this head
    q_row0    = Int(qb) * FAM_BM        # first query row within the head
    pq = pointer(Q); pk = pointer(K); pv = pointer(V)
    po32 = reinterpret(Core.LLVMPtr{UInt32, 1}, pointer(O))

    # ── prologue: Q + K(0) staged in one cp.async group ─────────────────
    # 16 B per thread per issue; a 64×HD bf16 tile is 8·HD chunks →
    # HD/16 issues of 128 threads. Chunk c → row c ÷ D8, column 8·(c % D8).
    @unrolled 8 for i in 0:(D16 - 1)
        c = Int(tid) + i * FAM_THREADS
        row = c ÷ D8
        col = (c % D8) * 8
        ptx"cp.async.cg.shared.global"(
            q_base + (row * HD + col) * 2,
            pq + ((head_row0 + q_row0 + row) * HD + col) * 2, Val(16))
        ptx"cp.async.cg.shared.global"(
            k_base0 + (row * HD + col) * 2,
            pk + ((head_row0 + row) * HD + col) * 2, Val(16))
    end
    ptx"cp.async.commit_group"()

    # ── carried state ────────────────────────────────────────────────────
    @unrolled 16 for j in 1:D8
        oacc_j = (0f0, 0f0, 0f0, 0f0)
    end
    m_lo = -Inf32; m_hi = -Inf32
    l_lo = 0f0;    l_hi = 0f0
    wrow = warp << 4                              # warp's row base in the tile
    qrow_lo = Float32(q_row0 + wrow + gid)        # within-head row ids for
    qrow_hi = qrow_lo + 8f0                       # the causal compare

    # ── Q fragments to registers, once ──────────────────────────────────
    # m16n8k16 A fragment (row.col): (row gid, k pair), (+8 rows),
    # (k +8), (both) — gemm_pipelined.jl's layout.
    ptx"cp.async.wait_all"()
    ptx"bar.sync"(Val(0))
    @unrolled 8 for kc in 0:(D16 - 1)
        qoff = q_base + ((wrow + gid) * HD + kc * 16 + col_lo) * 2
        qf_kc = (ptx"ld.shared.b32"(qoff),
                 ptx"ld.shared.b32"(qoff + 8 * HD * 2),
                 ptx"ld.shared.b32"(qoff + 16),
                 ptx"ld.shared.b32"(qoff + 8 * HD * 2 + 16))
    end

    n_tiles = CAUSAL ? (qb + UInt32(1)) : (seqlen ÷ UInt32(FAM_BN))
    fmax = ptx"max.f32"
    fadd = ptx"add.f32"

    kt = UInt32(0)
    while kt < n_tiles
        # K(kt)'s copy (prologue or last iteration's prefetch) must have
        # landed, and every warp must be past tile kt-1's reads of the
        # stage K(kt+1) will overwrite below.
        ptx"cp.async.wait_all"()
        ptx"bar.sync"(Val(0))
        stage_k = (kt & UInt32(1)) == UInt32(0) ? k_base0 : k_base1

        # prefetch K(kt+1) into the other stage; the copy flies under
        # this tile's compute and drains at the next loop head
        if kt + UInt32(1) < n_tiles
            nxt = ((kt + UInt32(1)) & UInt32(1)) == UInt32(0) ? k_base0 : k_base1
            krow0 = head_row0 + Int(kt + UInt32(1)) * FAM_BN
            @unrolled 8 for i in 0:(D16 - 1)
                c = Int(tid) + i * FAM_THREADS
                row = c ÷ D8
                col = (c % D8) * 8
                ptx"cp.async.cg.shared.global"(
                    nxt + (row * HD + col) * 2,
                    pk + ((krow0 + row) * HD + col) * 2, Val(16))
            end
            ptx"cp.async.commit_group"()
        end

        # ── S = Q·Kᵀ: 8 accumulator tiles, contraction over HD ──────────
        @unrolled for j in 1:8
            sacc_j = (0f0, 0f0, 0f0, 0f0)
        end
        @unrolled 8 for kc in 0:(D16 - 1)
            @unrolled for j in 1:8
                boff = stage_k + (((j - 1) * 8 + gid) * HD +
                                  kc * 16 + col_lo) * 2
                sacc_j = ptx"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"(
                    qf_kc,
                    (ptx"ld.shared.b32"(boff), ptx"ld.shared.b32"(boff + 16)),
                    sacc_j)
            end
        end

        # ── causal mask on the diagonal tile (col > row → -Inf) ─────────
        if CAUSAL && kt == qb
            kcol0 = Float32(Int(kt) * FAM_BN + col_lo)
            @unrolled for j in 1:8
                c0 = kcol0 + Float32(8 * (j - 1))
                sacc_j = (ifelse(c0       > qrow_lo, -Inf32, sacc_j[1]),
                          ifelse(c0 + 1f0 > qrow_lo, -Inf32, sacc_j[2]),
                          ifelse(c0       > qrow_hi, -Inf32, sacc_j[3]),
                          ifelse(c0 + 1f0 > qrow_hi, -Inf32, sacc_j[4]))
            end
        end

        # ── online softmax (per lane: rows gid and gid+8) ───────────────
        tmax_lo = -Inf32; tmax_hi = -Inf32
        @unrolled for j in 1:8
            tmax_lo = fmax(tmax_lo, fmax(sacc_j[1], sacc_j[2]))
            tmax_hi = fmax(tmax_hi, fmax(sacc_j[3], sacc_j[4]))
        end
        tmax_lo = warp_reduce(fmax, tmax_lo, Val(4))
        tmax_hi = warp_reduce(fmax, tmax_hi, Val(4))
        m_new_lo = fmax(m_lo, tmax_lo)
        m_new_hi = fmax(m_hi, tmax_hi)
        alpha_lo = ptx"ex2.approx.f32"((m_lo - m_new_lo) * qk2)
        alpha_hi = ptx"ex2.approx.f32"((m_hi - m_new_hi) * qk2)
        m_lo = m_new_lo; m_hi = m_new_hi
        mneg_lo = -(m_lo * qk2)
        mneg_hi = -(m_hi * qk2)

        # P = exp2((s − m)·qk2) in the accumulator layout; fold row sums
        tsum_lo = 0f0; tsum_hi = 0f0
        @unrolled for j in 1:8
            sacc_j = (ptx"ex2.approx.f32"(ptx"fma.rn.f32"(sacc_j[1], qk2, mneg_lo)),
                      ptx"ex2.approx.f32"(ptx"fma.rn.f32"(sacc_j[2], qk2, mneg_lo)),
                      ptx"ex2.approx.f32"(ptx"fma.rn.f32"(sacc_j[3], qk2, mneg_hi)),
                      ptx"ex2.approx.f32"(ptx"fma.rn.f32"(sacc_j[4], qk2, mneg_hi)))
            tsum_lo = fadd(tsum_lo, fadd(sacc_j[1], sacc_j[2]))
            tsum_hi = fadd(tsum_hi, fadd(sacc_j[3], sacc_j[4]))
        end
        tsum_lo = warp_reduce(fadd, tsum_lo, Val(4))
        tsum_hi = warp_reduce(fadd, tsum_hi, Val(4))
        l_lo = ptx"fma.rn.f32"(l_lo, alpha_lo, tsum_lo)
        l_hi = ptx"fma.rn.f32"(l_hi, alpha_hi, tsum_hi)

        # rescale O by alpha (elements 1,2 = row gid; 3,4 = row gid+8)
        @unrolled 16 for j in 1:D8
            oacc_j = (oacc_j[1] * alpha_lo, oacc_j[2] * alpha_lo,
                      oacc_j[3] * alpha_hi, oacc_j[4] * alpha_hi)
        end

        # ── stage Vᵀ: v4 loads of two adjacent seq rows, halfword repack.
        # The loop-head barrier already ordered PV(kt-1)'s reads before
        # these writes.
        vrow0 = head_row0 + Int(kt) * FAM_BN
        @unrolled 4 for i in 0:(D16 ÷ 2 - 1)
            idx = Int(tid) + i * FAM_THREADS
            kp  = idx ÷ D8
            dc  = (idx % D8) * 8
            va = ptx"ld.global.v4.b32"(pv + ((vrow0 + 2kp) * HD + dc) * 2)
            vb = ptx"ld.global.v4.b32"(pv + ((vrow0 + 2kp + 1) * HD + dc) * 2)
            @unrolled for e in 1:4
                # va/vb b32s pack (d, d+1); split and re-pair by seq
                lo = (va[e] & UInt32(0xffff)) | (vb[e] << 16)
                hi = (va[e] >> 16) | (vb[e] & UInt32(0xffff0000))
                d0 = dc + 2 * (e - 1)
                @inbounds vt[d0 * (FAM_BN ÷ 2) + kp + 1] = lo
                @inbounds vt[(d0 + 1) * (FAM_BN ÷ 2) + kp + 1] = hi
            end
        end
        ptx"bar.sync"(Val(0))

        # ── O += P·V: A from the exp'd accumulators (k-chunk t = S tiles
        # ja, jb), B fragments straight from Vᵀ ──────────────────────────
        @unrolled for (t, ja, jb) in ((1, 1, 2), (2, 3, 4), (3, 5, 6), (4, 7, 8))
            af_t = (bf16x2_pack(sacc_ja[1], sacc_ja[2]),
                    bf16x2_pack(sacc_ja[3], sacc_ja[4]),
                    bf16x2_pack(sacc_jb[1], sacc_jb[2]),
                    bf16x2_pack(sacc_jb[3], sacc_jb[4]))
            @unrolled 16 for j in 1:D8
                voff = vt_ptr + (((j - 1) * 8 + gid) * (FAM_BN ÷ 2) +
                                 (t - 1) * 8 + (col_lo >> 1)) * 4
                oacc_j = ptx"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"(
                    af_t,
                    (ptx"ld.shared.b32"(voff), ptx"ld.shared.b32"(voff + 16)),
                    oacc_j)
            end
        end

        kt += UInt32(1)
    end

    # ── epilogue: normalize by l, pack bf16, store ───────────────────────
    inv_lo = ptx"rcp.approx.f32"(l_lo)
    inv_hi = ptx"rcp.approx.f32"(l_hi)
    orow_lo = head_row0 + q_row0 + wrow + gid
    orow_hi = orow_lo + 8
    @unrolled 16 for j in 1:D8
        colb = (j - 1) * 8 + col_lo
        ob_lo = bf16x2_pack(oacc_j[1] * inv_lo, oacc_j[2] * inv_lo)
        ob_hi = bf16x2_pack(oacc_j[3] * inv_hi, oacc_j[4] * inv_hi)
        ptx"st.global.b32"(po32 + (orow_lo * HD + colb) * 2, ob_lo)
        ptx"st.global.b32"(po32 + (orow_hi * HD + colb) * 2, ob_hi)
    end
    return nothing
end

# ── host side ───────────────────────────────────────────────────────────

# f32 reference with bf16-quantized inputs, per (batch*head) slice.
function fam_cpu_ref(Q::Matrix{Float32}, K::Matrix{Float32},
                      V::Matrix{Float32}, sm_scale::Float32, causal::Bool)
    Qb = bf16_to_f32.(bf16_bits.(Q))
    Kb = bf16_to_f32.(bf16_bits.(K))
    Vb = bf16_to_f32.(bf16_bits.(V))
    scores = (Qb * Kb') .* sm_scale
    if causal
        S = size(scores, 1)
        for r in 1:S, c in (r + 1):S
            scores[r, c] = -Inf32
        end
    end
    rmax = maximum(scores; dims = 2)
    ex = exp.(scores .- rmax)
    return (ex ./ sum(ex; dims = 2)) * Vb
end

# (rows, HD) f32 → bf16 bits, HD-fast row-major.
function fam_pack(X::Matrix{Float32})
    rows, hd = size(X)
    out = Array{UInt16}(undef, hd, rows)
    for r in 1:rows, h in 1:hd
        @inbounds out[h, r] = bf16_bits(X[r, h])
    end
    out
end

function run_fam(B, H, S, HD; causal = false, atol = 5e-2)
    @assert S % FAM_BM == 0
    bh = B * H
    total_rows = bh * S
    sm_scale = 1.0f0 / sqrt(Float32(HD))

    rng = MersenneTwister(B * 7919 + H * 131 + S + HD)
    Q = Float32.(randn(rng, total_rows, HD)) .* 0.5f0
    K = Float32.(randn(rng, total_rows, HD)) .* 0.5f0
    V = Float32.(randn(rng, total_rows, HD))

    Q_d = CuArray(vec(fam_pack(Q)))
    K_d = CuArray(vec(fam_pack(K)))
    V_d = CuArray(vec(fam_pack(V)))
    O_d = CUDACore.zeros(UInt16, total_rows * HD)

    args = (O_d, Q_d, K_d, V_d, UInt32(S), sm_scale * Float32(FAM_LOG2E),
            Val(HD), Val(causal))
    shmem = fam_smem_bytes(HD)
    kern = @cuda launch=false fam_kernel!(args...)
    attrs = CUDACore.attributes(kern.fun)
    attrs[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shmem
    kern(args...; blocks = (S ÷ FAM_BM, bh), threads = FAM_THREADS,
         shmem = shmem)
    CUDACore.synchronize()

    O_packed = reshape(Array(O_d), HD, total_rows)
    maxdiff = 0.0f0
    for i in 1:bh
        r = ((i - 1) * S + 1):(i * S)
        O_got = permutedims(bf16_to_f32.(O_packed[:, r]))
        O_ref = fam_cpu_ref(Q[r, :], K[r, :], V[r, :], sm_scale, causal)
        maxdiff = max(maxdiff, maximum(abs.(O_got - O_ref)))
    end
    @info "flash_attention_mma" B H S HD causal maxdiff
    maxdiff < atol
end

@testset "FA mma.sync B=$B H=$H S=$S D=$HD causal=$c" for (B, H, S, HD, c) in [
        (1, 1, 64,   128, false),   # single tile, single CTA
        (1, 1, 256,  128, false),
        (1, 2, 512,  128, false),
        (2, 4, 1024, 128, false),
        (1, 2, 512,  128, true),    # causal: diagonal-tile masking
        (1, 2, 512,  64,  false),   # head_dim 64
        (1, 1, 256,  64,  true),
    ]
    @test run_fam(B, H, S, HD; causal = c)
end
