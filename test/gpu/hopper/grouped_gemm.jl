# Uniform-shape grouped GEMM (Hopper) — algorithmic port of
# pyptx/examples/hopper/grouped_gemm.py.
#
# G matrix-multiply problems all sharing shape (M, K) x (K, N) -> (M, N), as
# in MoE inference where every expert has equal capacity.
#
# Memory layouts. Pyptx uses N-innermost B SMEM + `trans_b=1` to make wgmma
# read it transposed. PTX.jl's wgmma chain bakes `trans_b=0` (B is col-major
# = K-fast, the wgmma.row.col default for bf16), so we lay BOTH A and B out
# K-fast in memory. The algorithm is identical; only the host-side packing
# and B's TMA-coord convention differ from pyptx:
#
#   A: bf16 (G, M, K) → Julia (K, G*M) col-major (K innermost in memory).
#      TMA coord (innermost, outer) = (k_off, group*M + ctay*BM).
#   B: bf16 (G, K, N) → Julia (K, G*N) col-major (K innermost in memory).
#      TMA coord (innermost, outer) = (k_off, group*N + ctax*BN).
#   C: f32  (G, M, N) → Julia (N, G*M) col-major (N innermost in memory).
#      Per-lane v2.f32 stores write two consecutive f32 at frag_col, frag_col+1.
#
# Grid is (N/BN, M/BM, G) — every CTA owns one BM × BN output tile of one
# group. `ctaid.z` picks the group, with offsets folded into the TMA coords.
# Reuses the existing 2D TMA path — no 3D tensormaps needed.
#
# v0 scope: tile_k = 16 (single wgmma per K iter), tile_n = 8 (m64n8k16 wgmma).
# Multi-K and larger tile_n variants are natural follow-ups — the existing
# wgmma chain has the shapes; only SMEM allocation and descriptor-byte-offset
# math changes.

using PTX: layout_for_a, wgmma_descriptor, smem_addr_u32,
           tensor_map_tile_2d, CuTensorMap
using CUDACore
using Random

const GG_BM = 64
const GG_BN = 8
const GG_BK = 16            # tile_k = 16 (single wgmma per K iter for v0)
const GG_THREADS = 128

# Tx-bytes per mbarrier round: A tile + B tile.
const GG_LOAD_BYTES = GG_BM * GG_BK * 2 + GG_BK * GG_BN * 2

function _grouped_gemm_kernel!(
        C::CuDeviceVector{Float32, 1},
        tma_A::Core.LLVMPtr{UInt8, PTX.AS.Const},
        tma_B::Core.LLVMPtr{UInt8, PTX.AS.Const},
        M::Int32, N::Int32, K::Int32)

    smem_A = CuStaticSharedArray(UInt16, GG_BM * GG_BK)
    smem_B = CuStaticSharedArray(UInt16, GG_BK * GG_BN)
    mbar   = CuStaticSharedArray(UInt64, 1)

    a_ptr  = pointer(smem_A)
    b_ptr  = pointer(smem_B)
    mb_ptr = pointer(mbar)
    a_addr = smem_addr_u32(a_ptr)
    b_addr = smem_addr_u32(b_ptr)

    tid   = ptx"mov.u32"(sreg"tid.x")
    cta_x = ptx"mov.u32"(sreg"ctaid.x")   # N tile
    cta_y = ptx"mov.u32"(sreg"ctaid.y")   # M tile
    cta_z = ptx"mov.u32"(sreg"ctaid.z")   # group

    # Element-coord bases. Both A and B are K-fast in memory; TMA descriptors
    # have K innermost, M-or-N outer with group offset baked in:
    #   A: coord = (k_off, m_outer)  with m_outer = ctaz*M + ctay*BM
    #   B: coord = (k_off, n_outer)  with n_outer = ctaz*N + ctax*BN
    # C is (G*M, N) row-major in global memory — group g's rows start at
    # offset g*M, but the N axis is shared across groups. So output row uses
    # m_outer (group offset baked in via the M-stride) and output col uses
    # only the per-tile n_base (no group offset).
    m_outer = cta_y * UInt32(GG_BM) + cta_z * UInt32(M)
    n_base  = cta_x * UInt32(GG_BN)
    n_outer = n_base + cta_z * UInt32(N)

    # mbarrier init (visible to all threads via bar.sync before any poll).
    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mb_ptr, UInt32(1))
    end
    ptx"bar.sync"(Val(0))

    # Build wgmma descriptors. Both operands are K-fast in SMEM, so both use
    # the K-major canonical formula (`layout_for_a`). For BM=64, BK=16 A and
    # BN=8, BK=16 B: row width = 16 bf16 = 32 B → B32 swizzle, LBO=16, SBO=256.
    # The wgmma_descriptor builder doesn't distinguish A vs B operands — it
    # only packs (SMEM address, LBO, SBO, swizzle) into the 64-bit blob.
    la = layout_for_a(dtype = :bf16, m = GG_BM, k = GG_BK)
    lb = layout_for_a(dtype = :bf16, m = GG_BN, k = GG_BK)
    a_desc = wgmma_descriptor(a_addr;
        leading_byte_offset = la.leading_byte_offset,
        stride_byte_offset  = la.stride_byte_offset,
        swizzle             = la.layout_type)
    b_desc = wgmma_descriptor(b_addr;
        leading_byte_offset = lb.leading_byte_offset,
        stride_byte_offset  = lb.stride_byte_offset,
        swizzle             = lb.layout_type)

    # Accumulator: 4 f32 regs per lane × 128 lanes = 512 elements = 64×8.
    d = ntuple(_ -> 0f0, Val(4))

    # K-loop. Phase XORs across iters so test_wait.parity correctly tracks
    # the mbarrier flips (pyptx uses the same `phase ^= 1` trick).
    phase = UInt32(0)
    k_off = Int32(0)
    while k_off < K
        if tid == UInt32(0)
            ptx"fence.proxy.async.shared::cta"()
            ptx"mbarrier.arrive.expect_tx.shared.b64"(mb_ptr, UInt32(GG_LOAD_BYTES))
            # A tile at coord (innermost=k_off, outer=m_outer).
            ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                a_ptr, tma_A,
                k_off, reinterpret(Int32, m_outer),
                mb_ptr)
            # B tile at coord (innermost=k_off, outer=n_outer).
            ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                b_ptr, tma_B,
                k_off, reinterpret(Int32, n_outer),
                mb_ptr)
        end
        ptx"bar.sync"(Val(0))

        while !ptx"mbarrier.test_wait.parity.shared.b64"(mb_ptr, phase)
        end

        ptx"fence.proxy.async.shared::cta"()
        ptx"wgmma.fence.sync.aligned"()
        # scale_d=false on the very first wgmma overwrites the (already-zero)
        # accumulator; scale_d=true on later iters accumulates across K-tiles.
        # With d initialized to 0, false/true on iter 0 are numerically
        # equivalent — mirroring pyptx exactly anyway.
        scale_d = k_off != Int32(0)
        d = ptx"wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16"(
            d, a_desc, b_desc, scale_d)
        ptx"wgmma.commit_group.sync.aligned"()
        ptx"wgmma.wait_group.sync.aligned"(Val(0))

        k_off += Int32(GG_BK)
        phase ⊻= UInt32(1)
    end

    # Epilogue. m64n8k16 wgmma f32 output layout (PTX 9.2 §9.7.14.5.5):
    #   warp w ∈ 0..3 owns rows [16w, 16w+16) of D; within a warp, lane l:
    #     frag_row = 16w + (l >> 2)      (rows 0..7 in the warp's 16-row band)
    #     frag_col = 2 * (l & 3)         (col pair, 4 pairs across the n=8 cols)
    #   The 4 f32 frags per lane are at (frag_row, frag_col), (frag_row,
    #   frag_col+1), (frag_row+8, frag_col), (frag_row+8, frag_col+1).
    wid      = tid >> UInt32(5)
    lane     = tid & UInt32(31)
    frag_row = (wid << UInt32(4)) + (lane >> UInt32(2))
    frag_col = (lane & UInt32(3)) << UInt32(1)
    abs_row  = frag_row + m_outer     # group offset already in m_outer
    abs_col  = frag_col + n_base      # no group offset — shared N axis

    # C is f32 (G*M, N) row-major (N innermost). off = (row*N + col) * 4 bytes.
    pc    = pointer(C)
    off_a = (abs_row * UInt32(N) + abs_col) * UInt32(4)
    off_b = ((abs_row + UInt32(8)) * UInt32(N) + abs_col) * UInt32(4)
    ptx"st.global.v2.f32"(pc + Int(off_a), (d[1], d[2]))
    ptx"st.global.v2.f32"(pc + Int(off_b), (d[3], d[4]))
    return nothing
end

# --- Cross-arch ptxas validation (always runs) ---------------------------

@testset "grouped_gemm kernel compiles at sm_90a" begin
    types = Tuple{CuDeviceVector{Float32, 1},
                  Core.LLVMPtr{UInt8, PTX.AS.Const},
                  Core.LLVMPtr{UInt8, PTX.AS.Const},
                  Int32, Int32, Int32}
    @test ptxas_compiles(_grouped_gemm_kernel!, types;
                         cap = v"9.0", feature_set = :arch)

    ptx = emit_ptx(_grouped_gemm_kernel!, types;
                   cap = v"9.0", feature_set = :arch)
    @test occursin("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16", ptx)
    @test occursin("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes", ptx)
    @test occursin("mbarrier.test_wait.parity.shared.b64", ptx)
    @test occursin("st.global.v2.f32", ptx)
    # Z-grid plumbing: ctaid.z reads + the M / K multipliers folded in.
    @test occursin("%ctaid.z", ptx)
end

# --- Runtime path — Hopper only (wgmma is sm_90a) -----------------------

if v"9.0" <= DEV_CAP < v"10.0"
    # bf16 ↔ Float32 round-trip helpers (mirrors test/gpu/ampere/gemm_minimal.jl).
    bf16_bits(x::Float32) = UInt16((reinterpret(UInt32, x) + UInt32(0x8000)) >> 16)
    bf16_to_f32(b::UInt16) = reinterpret(Float32, UInt32(b) << 16)

    function _grouped_gemm_cpu_ref(A3::Array{Float32, 3}, B3::Array{Float32, 3})
        G, M, K = size(A3)
        _, _, N = size(B3)
        @assert size(B3) == (G, K, N)
        # Round f32 → bf16 → f32 to match what the kernel sees.
        Ab = bf16_to_f32.(bf16_bits.(A3))
        Bb = bf16_to_f32.(bf16_bits.(B3))
        C = zeros(Float32, G, M, N)
        for g in 1:G, m in 1:M, n in 1:N
            s = 0f0
            for k in 1:K
                s += Ab[g, m, k] * Bb[g, k, n]
            end
            C[g, m, n] = s
        end
        return C
    end

    function _pack_row_major_bf16(A3::Array{Float32, 3})
        # A3 is (G, M, K) f32 row-major in our test convention. Pack to bf16
        # and lay out in memory as (G*M, K) row-major with K innermost, which
        # in Julia col-major terms is `(K, G*M)`.
        G, M, K = size(A3)
        out = Array{UInt16}(undef, K, G * M)
        for g in 1:G, m in 1:M, k in 1:K
            out[k, (g - 1) * M + m] = bf16_bits(A3[g, m, k])
        end
        return out
    end

    function _pack_kfast_bf16_B(B3::Array{Float32, 3})
        # K-fast packing: per group g, B[g, k, n] is at Julia[k, g*N+n]
        # (col-major (K, G*N) → K innermost in memory, N-cols striped across
        # groups). Matches the TMA descriptor with innermost=K below.
        G, K, N = size(B3)
        out = Array{UInt16}(undef, K, G * N)
        for g in 1:G, k in 1:K, n in 1:N
            out[k, (g - 1) * N + n] = bf16_bits(B3[g, k, n])
        end
        return out
    end

    function _run_grouped_gemm(G::Int, M::Int, N::Int, K::Int; rtol = 1e-3, atol = 1e-3)
        @assert M % GG_BM == 0 "M must be divisible by BM=$GG_BM"
        @assert N % GG_BN == 0 "N must be divisible by BN=$GG_BN"
        @assert K % GG_BK == 0 "K must be divisible by BK=$GG_BK"

        rng = MersenneTwister(G * 1009 + M * 997 + N * 17 + K)
        A3 = Float32.(randn(rng, G, M, K)) .* 0.1f0
        B3 = Float32.(randn(rng, G, K, N)) .* 0.1f0

        A_packed = _pack_row_major_bf16(A3)    # (K, G*M) col-major, K-fast
        B_packed = _pack_kfast_bf16_B(B3)      # (K, G*N) col-major, K-fast
        A_d = CuArray(A_packed)
        B_d = CuArray(B_packed)

        # TMA descriptors. Both have K as innermost (matching K-fast memory).
        # tensor_map_tile_2d takes (rows, cols, box_rows, box_cols) with
        # innermost = cols.
        tmap_A = tensor_map_tile_2d(:bf16, pointer(A_d), G * M, K, GG_BM, GG_BK;
                                    swizzle = :B32)
        tmap_B = tensor_map_tile_2d(:bf16, pointer(B_d), G * N, K, GG_BN, GG_BK;
                                    swizzle = :B32)

        tmap_A_dev = CuArray{UInt8}(undef, 128); copyto!(tmap_A_dev, collect(tmap_A.data))
        tmap_B_dev = CuArray{UInt8}(undef, 128); copyto!(tmap_B_dev, collect(tmap_B.data))
        a_const = reinterpret(Core.LLVMPtr{UInt8, PTX.AS.Const}, UInt64(pointer(tmap_A_dev)))
        b_const = reinterpret(Core.LLVMPtr{UInt8, PTX.AS.Const}, UInt64(pointer(tmap_B_dev)))

        C_d = CUDACore.zeros(Float32, G * M * N)
        grid = (N ÷ GG_BN, M ÷ GG_BM, G)
        @cuda threads = GG_THREADS blocks = grid feature_set = :arch _grouped_gemm_kernel!(
            C_d, a_const, b_const, Int32(M), Int32(N), Int32(K))
        CUDACore.synchronize()

        # C is laid out as (G*M, N) row-major with N innermost → Julia (N, G*M).
        C_packed = reshape(Array(C_d), N, G * M)
        C_got = Array{Float32}(undef, G, M, N)
        for g in 1:G, m in 1:M, n in 1:N
            C_got[g, m, n] = C_packed[n, (g - 1) * M + m]
        end

        C_ref = _grouped_gemm_cpu_ref(A3, B3)
        ≈(C_got, C_ref; rtol = rtol, atol = atol)
    end

    @testset "grouped GEMM G=$G M=$M N=$N K=$K" for (G, M, N, K) in [
            (1, 64, 8, 32),
            (2, 64, 8, 32),
            (4, 64, 8, 64),
        ]
        @test _run_grouped_gemm(G, M, N, K)
    end
end
