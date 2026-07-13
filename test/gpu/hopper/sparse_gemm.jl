# TEST_TARGET: requires=toolkit evidence=mixed target=sm_90a
# Structured-sparse Hopper GEMM (`wgmma.mma_async.sp`) — ported from
# cutlass/examples/62_hopper_sparse_gemm (NVIDIA CUTLASS, BSD-3-Clause).
#
# The CUTLASS example runs a 2:4 structured-sparse GEMM: matrix A carries
# 50% zeros at 4-element granularity, is compressed offline to Mx(K/2)
# plus a metadata tensor (CUTLASS's "E"), and the mainloop issues sparse
# GMMA that reconstructs positions from the metadata on the fly — 2× the
# K-depth per instruction for the same operand bytes. What's ported:
#
#   - The compress step (host-side here, as in CUTLASS's device
#     transform): dense 64×K bf16 A with exactly 2 non-zeros per 4-wide
#     chunk → packed 64×(K/2) values + per-lane 32-bit metadata words.
#   - The metadata layout of PTX 9.3 Figure 176 (m64nNk32 f16/bf16,
#     sparsity selector 0): within each warp's 16-row slab, lane
#     l = 4q + h (h ∈ {0,1}) supplies rows q (bits 15:0) and q+8
#     (bits 31:16) for K-halves cols 16h..16h+15; one nibble per 4-wide
#     chunk, low 2 bits = index of the first stored non-zero, high 2
#     bits = the second. (0b0000/0b0101/0b1010/0b1111 — equal indices —
#     are invalid.) Lanes with (l & 3) ≥ 2 are ignored at selector 0.
#   - The mainloop: TMA-load the PACKED A tile (64×16 bf16 — byte-wise
#     identical to a dense k16 tile, so descriptor + swizzle math reuse
#     layout_for_a(m=64, k=16)) and the DENSE B tile (32×8, K-fast,
#     64-byte rows → first B64-swizzle operand in this corpus), then
#     wgmma.mma_async.sp.sync.aligned.m64n8k32.f32.bf16.bf16 with the
#     per-lane metadata word loaded from a device array (CUTLASS's
#     reordered E tensor, likewise precomputed).
#
# Not ported: CUTLASS's device-side compressor (host compress keeps the
# corpus test self-contained) and alpha/beta epilogue scaling.

using PTX: layout_for_a, wgmma_descriptor, smem_addr_u32, tensor_map_tile_2d
using CUDACore
using Random

const SPG_BM = 64
const SPG_BN = 8
const SPG_BK = 32                    # logical (dense) K per wgmma.sp
const SPG_BKP = SPG_BK ÷ 2           # packed K per A tile
const SPG_THREADS = 128
const SPG_LOAD_BYTES = SPG_BM * SPG_BKP * 2 + SPG_BK * SPG_BN * 2

function _spg_kernel!(
        D::CuDeviceVector{Float32, 1},
        meta::CuDeviceVector{UInt32, 1},   # [k_iter * 128 + tid]
        tma_A::PTX.TMADescriptorPtr,       # packed A, 64 × (K/2), K-fast
        tma_B::PTX.TMADescriptorPtr,       # dense B, 32-row K-fast tiles
        K::Int32)

    smem_A = CuStaticSharedArray(UInt16, SPG_BM * SPG_BKP)
    smem_B = CuStaticSharedArray(UInt16, SPG_BK * SPG_BN)
    mbar   = CuStaticSharedArray(UInt64, 1)

    a_ptr  = pointer(smem_A)
    b_ptr  = pointer(smem_B)
    mb_ptr = pointer(mbar)
    a_addr = smem_addr_u32(a_ptr)
    b_addr = smem_addr_u32(b_ptr)

    tid = ptx"mov.u32"(sreg"tid.x")

    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mb_ptr, UInt32(1))
        ptx"fence.proxy.async.shared::cta"()
    end
    ptx"bar.sync"(Val(0))

    # A descriptor covers the PACKED tile — canonical K-major layout of a
    # dense 64×16 bf16 tile (32-B rows, B32). B is a 32×8 K-fast tile:
    # 64-B rows → B64 family.
    la = layout_for_a(dtype = :bf16, m = SPG_BM, k = SPG_BKP)
    lb = layout_for_a(dtype = :bf16, m = SPG_BN, k = SPG_BK)
    a_desc = wgmma_descriptor(a_addr;
        leading_byte_offset = la.leading_byte_offset,
        stride_byte_offset  = la.stride_byte_offset,
        swizzle             = la.layout_type)
    b_desc = wgmma_descriptor(b_addr;
        leading_byte_offset = lb.leading_byte_offset,
        stride_byte_offset  = lb.stride_byte_offset,
        swizzle             = lb.layout_type)

    d = ntuple(_ -> 0f0, Val(4))
    num_k_tiles = K >> Int32(5)          # K / 32 (logical)

    @inbounds for k_iter in Int32(0):(num_k_tiles - Int32(1))
        if tid == UInt32(0)
            ptx"mbarrier.arrive.expect_tx.shared.b64"(mb_ptr, UInt32(SPG_LOAD_BYTES))
            # Packed A advances 16 packed columns per logical 32-K tile.
            ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                a_ptr, tma_A, k_iter * Int32(SPG_BKP), Int32(0), mb_ptr)
            ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                b_ptr, tma_B, k_iter * Int32(SPG_BK), Int32(0), mb_ptr)
        end
        phase = UInt32(k_iter & Int32(1))
        while !ptx"mbarrier.test_wait.parity.shared.b64"(mb_ptr, phase)
        end

        # Per-lane metadata word for this K-tile (CUTLASS's reordered E
        # tensor equivalent — precomputed, one .b32 per lane per tile).
        m_word = meta[Int(k_iter) * SPG_THREADS + Int(tid) + 1]

        ptx"fence.proxy.async.shared::cta"()
        ptx"wgmma.fence.sync.aligned"()
        d = ptx"wgmma.mma_async.sp.sync.aligned.m64n8k32.f32.bf16.bf16"(
            d, a_desc, b_desc, m_word, Val(0), true)
        ptx"wgmma.commit_group.sync.aligned"()
        ptx"wgmma.wait_group.sync.aligned"(Val(0))
        ptx"bar.sync"(Val(0))            # drain before re-arming the buffer
    end

    wid      = tid >> UInt32(5)
    lane     = tid & UInt32(31)
    frag_row = (wid << UInt32(4)) + (lane >> UInt32(2))
    frag_col = (lane & UInt32(3)) << UInt32(1)
    pd       = pointer(D)
    off_a    = (frag_row * UInt32(SPG_BN) + frag_col) * UInt32(4)
    off_b    = ((frag_row + UInt32(8)) * UInt32(SPG_BN) + frag_col) * UInt32(4)
    ptx"st.global.v2.f32"(pd + Int(off_a), (d[1], d[2]))
    ptx"st.global.v2.f32"(pd + Int(off_b), (d[3], d[4]))
    return nothing
end

# ── Host-side 2:4 compress + Figure-176 metadata packing ───────────────

# Generate a dense (M, K) f32 matrix with exactly two non-zeros per
# 4-wide K-chunk, returning (dense, chunk index pairs).
function spg_random_24(rng, M::Int, K::Int)
    dense = zeros(Float32, M, K)
    idx   = Array{NTuple{2, Int}}(undef, M, K ÷ 4)   # 0-based, i0 < i1
    for m in 1:M, c in 1:(K ÷ 4)
        i1 = rand(rng, 0:3)
        i0 = rand(rng, 0:3)
        i0 == i1 && (i1 = (i1 + 1) % 4; i0 > i1 && ((i0, i1) = (i1, i0)))
        i0 > i1 && ((i0, i1) = (i1, i0))
        idx[m, c] = (i0, i1)
        dense[m, 4 * (c - 1) + i0 + 1] = randn(rng, Float32) * 0.1f0
        dense[m, 4 * (c - 1) + i1 + 1] = randn(rng, Float32) * 0.1f0
    end
    return dense, idx
end

# Packed values: chunk order, two per chunk → (K/2, M) K-fast bf16 bits.
function spg_pack_A(dense::Array{Float32, 2}, idx)
    M, K = size(dense)
    packed = Array{UInt16}(undef, K ÷ 2, M)
    for m in 1:M, c in 1:(K ÷ 4)
        i0, i1 = idx[m, c]
        packed[2c - 1, m] = bf16_bits(dense[m, 4 * (c - 1) + i0 + 1])
        packed[2c,     m] = bf16_bits(dense[m, 4 * (c - 1) + i1 + 1])
    end
    return packed
end

# Metadata words per Figure 176, sparsity selector 0: lane l = 32w + lw
# contributes iff (lw & 3) < 2; q = lw >> 2 picks the row pair
# (16w + q, 16w + q + 8), h = lw & 1 picks the 16-column K-half.
function spg_metadata(idx, k_tile::Int)      # k_tile: 0-based logical tile
    meta = zeros(UInt32, SPG_THREADS)
    chunk0 = k_tile * (SPG_BK ÷ 4)           # first chunk of this tile
    for w in 0:3, lw in 0:31
        (lw & 3) < 2 || continue
        q = lw >> 2
        h = lw & 1
        word = UInt32(0)
        for (half, r) in ((0, 16w + q), (16, 16w + q + 8))
            for c in 0:3
                i0, i1 = idx[r + 1, chunk0 + 4h + c + 1]
                word |= (UInt32(i0) | UInt32(i1) << 2) << (4c + half)
            end
        end
        meta[32w + lw + 1] = word
    end
    return meta
end

# ── Cross-arch ptxas validation ────────────────────────────────────────

# Sweep the sparse-wgmma wrapper family across dtypes/shapes/selectors
# (compile-only, mirrors wgmma_sweep.jl's pattern).
const SPG_SWEEP = [
    # (dtype_d, a, b, N, K, sel, d_jltype, d_per_lane)
    (:f32, :bf16, :bf16, 8,   32, 0, Float32, 4),
    (:f32, :bf16, :bf16, 64,  32, 1, Float32, 32),
    (:f32, :f16,  :f16,  16,  32, 0, Float32, 8),
    (:f16, :f16,  :f16,  64,  32, 1, UInt32,  16),
    (:f32, :tf32, :tf32, 8,   16, 0, Float32, 4),
    (:f32, :e4m3, :e4m3, 8,   64, 0, Float32, 4),
    (:f16, :e5m2, :e5m2, 64,  64, 0, UInt32,  16),
    (:s32, :s8,   :s8,   8,   64, 0, Int32,   4),
    (:s32, :u8,   :s8,   128, 64, 0, Int32,   64),
]

_spg_kern_name(d_ty, a_ty, b_ty, n, k, sel) =
    Symbol("_spg_kern_", d_ty, "_", a_ty, "_", b_ty, "_n", n, "k", k, "s", sel)

for case in SPG_SWEEP
    d_ty, a_ty, b_ty, n, k, sel, d_jltype, d_per_lane = case
    mods = (:mma_async, :sp, :sync, :aligned,
            Symbol("m64n", n, "k", k), d_ty, a_ty, b_ty)
    op = PTX.Operation{:wgmma, mods}()
    fname = _spg_kern_name(d_ty, a_ty, b_ty, n, k, sel)
    @eval function $(fname)(out::CuDeviceVector{$d_jltype, 1},
                            a_desc::UInt64, b_desc::UInt64, m_word::UInt32)
        d = ntuple(_ -> zero($d_jltype), Val($d_per_lane))
        d = $op(d, a_desc, b_desc, m_word, Val($sel), false)
        @inbounds out[1] = d[1]
        return nothing
    end
end

@testset "sparse wgmma sweep: $(c[1]).$(c[2]).$(c[3]) m64n$(c[4])k$(c[5]) sel=$(c[6])" for
        c in SPG_SWEEP
    d_ty, a_ty, b_ty, n, k, sel, d_jltype, _ = c
    fname = _spg_kern_name(d_ty, a_ty, b_ty, n, k, sel)
    types = Tuple{CuDeviceVector{d_jltype, 1}, UInt64, UInt64, UInt32}
    @test ptxas_compiles(getfield(@__MODULE__, fname), types;
                         cap = v"9.0", feature_set = :arch)
end

@testset "sparse GEMM kernel compiles at sm_90a" begin
    types = Tuple{CuDeviceVector{Float32, 1}, CuDeviceVector{UInt32, 1},
                  PTX.TMADescriptorPtr, PTX.TMADescriptorPtr, Int32}
    @test ptxas_compiles(_spg_kernel!, types; cap = v"9.0", feature_set = :arch)

    ptx = emit_ptx(_spg_kernel!, types; cap = v"9.0", feature_set = :arch)
    @test occursin("wgmma.mma_async.sp.sync.aligned.m64n8k32.f32.bf16.bf16", ptx)
    # sp-sel immediate is baked (fifth operand, before scale-d's predicate)
    @test occursin(r"wgmma\.mma_async\.sp[^;]*, 0, %p", ptx)
end

# ── Runtime — Hopper hardware ──────────────────────────────────────────

if test_runtime_supported(@__FILE__)
    function spg_run(rng, K_test::Int)
        A_dense, idx = spg_random_24(rng, SPG_BM, K_test)
        B_f32 = randn(rng, Float32, K_test, SPG_BN) .* 0.1f0

        A_packed = spg_pack_A(A_dense, idx)
        B_packed = Array{UInt16}(undef, K_test, SPG_BN)
        for k in 1:K_test, n in 1:SPG_BN
            B_packed[k, n] = bf16_bits(B_f32[k, n])
        end
        A_d = CuArray(A_packed)
        B_d = CuArray(B_packed)

        n_tiles = K_test ÷ SPG_BK
        meta = reduce(vcat, (spg_metadata(idx, t) for t in 0:(n_tiles - 1)))
        meta_d = CuArray(meta)

        # Packed A: (K/2)-fast, 16-col boxes (32-B rows → B32).
        # B: K-fast, 32-col boxes (64-B rows → B64).
        tmap_A = tensor_map_tile_2d(:bf16, pointer(A_d),
            SPG_BM, K_test ÷ 2, SPG_BM, SPG_BKP; swizzle = :B32)
        tmap_B = tensor_map_tile_2d(:bf16, pointer(B_d),
            SPG_BN, K_test, SPG_BN, SPG_BK; swizzle = :B64)
        A = upload_tma_descriptor(tmap_A)
        B = upload_tma_descriptor(tmap_B)

        D_dev = CUDACore.zeros(Float32, SPG_BM * SPG_BN)
        @cuda threads = SPG_THREADS _spg_kernel!(
            D_dev, meta_d, A.ptr, B.ptr, Int32(K_test))
        CUDACore.synchronize()

        D_packed = reshape(Array(D_dev), SPG_BN, SPG_BM)
        D_got = Array{Float32}(undef, SPG_BM, SPG_BN)
        for m in 1:SPG_BM, n in 1:SPG_BN
            D_got[m, n] = D_packed[n, m]
        end
        return D_got, bf16_gemm_ref(A_dense, B_f32)
    end

    @testset "2:4 sparse GEMM vs dense reference (K=32, single tile)" begin
        D_got, D_ref = spg_run(MersenneTwister(0x62), 32)
        @test isapprox(D_got, D_ref; rtol = 1e-3, atol = 1e-3)
    end

    @testset "2:4 sparse GEMM vs dense reference (K=128, 4 tiles)" begin
        D_got, D_ref = spg_run(MersenneTwister(0x2424), 128)
        @test isapprox(D_got, D_ref; rtol = 1e-3, atol = 1e-3)
    end
end
