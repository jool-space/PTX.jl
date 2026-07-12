# REQUIRES CC 8.0
#
# Ported from cutlass/examples/18_ampere_fp64_tensorop_affine2_gemm
# (NVIDIA CUTLASS, BSD-3-Clause). The original demonstrates two things:
# fp64 tensor cores (DMMA, new on GA100) and CUTLASS's Affine2 strided
# layouts. The layout half is CuTe machinery with no PTX surface; what's
# ported is the DMMA half: mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64.
#
# "FP64 matmul on tensor cores" is a real thing since sm_80: DMMA performs
# IEEE-754 fp64 FMAs on the tensor-core datapath (full rate only on
# datacenter silicon — GA100/GH100 class; consumer/workstation parts like
# this sm_89 box execute it correctly but at a token rate, so this is a
# correctness demo, not a perf one). Because every element is a full f64
# register, the fragments are tiny: one f64 A element, one f64 B element,
# two f64 accumulators per lane — a warp computes an 8×8 = (8×4)·(4×8) tile
# per instruction.
#
# Inputs (flat row-major flattened):
#   A    :: (M, K) f64, row-major
#   B_T  :: (N, K) f64, row-major  (B^T so K is contiguous)
#   D    :: (M, N) f64, row-major
#
# Grid:  (N/8, M/8), one warp per CTA.
#
# Per-lane fragment indices (PTX ISA 9.3 §9.7.15.5.2, m8n8k4 f64):
#   gid = tid >> 2 (0..7), tig = tid & 0x3 (0..3)
#   a0 = A[gid, tig]            (8×4, row = gid, col = tig)
#   b0 = B[tig, gid]            (4×8) — with B^T storage: B_T[gid, tig]
#   c/d: d[i] = D[gid, 2*tig + i]  for i in {0, 1}

using Random

const F64_BM, F64_BN, F64_BK = 8, 8, 4

function gemm_fp64_kernel!(
        D::CuDeviceVector{Float64},
        A::CuDeviceVector{Float64},
        B_T::CuDeviceVector{Float64},
        ::Val{M}, ::Val{N}, ::Val{K}) where {M, N, K}
    n_iters = K ÷ F64_BK

    m_base = Int(ptx"mov.u32"(sreg"ctaid.y")) * F64_BM
    n_base = Int(ptx"mov.u32"(sreg"ctaid.x")) * F64_BN

    tid = ptx"mov.u32"(sreg"tid.x")
    gid = Int(tid >> UInt32(2))
    tig = Int(tid & UInt32(0x3))

    a_row = m_base + gid
    b_row = n_base + gid          # row of B^T = column of B

    acc = (0.0, 0.0)

    @inbounds for ki in 0:n_iters-1
        k = ki * F64_BK + tig
        acc = ptx"mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"(
            (A[a_row * K + k + 1],), (B_T[b_row * K + k + 1],), acc)
    end

    d_col = n_base + 2 * tig
    @inbounds begin
        D[a_row * N + d_col + 1] = acc[1]
        D[a_row * N + d_col + 2] = acc[2]
    end
    return nothing
end

flatten_rowmajor(A::AbstractMatrix) = vec(permutedims(A))

function run_fp64_gemm(M::Int, N::Int, K::Int)
    @assert M % F64_BM == 0 && N % F64_BN == 0 && K % F64_BK == 0

    rng  = MersenneTwister(M * 7919 + N * 31 + K)
    A64  = randn(rng, Float64, M, K)
    BT64 = randn(rng, Float64, N, K)

    A_d  = CuArray(flatten_rowmajor(A64))
    BT_d = CuArray(flatten_rowmajor(BT64))
    D_d  = CUDACore.zeros(Float64, M * N)

    @cuda blocks=(N ÷ F64_BN, M ÷ F64_BM) threads=32 gemm_fp64_kernel!(
        D_d, A_d, BT_d, Val(M), Val(N), Val(K))
    CUDACore.synchronize()

    D = Matrix(reshape(Array(D_d), N, M)')
    D_ref = A64 * BT64'
    return D, D_ref
end

@testset "Ampere fp64 GEMM (m8n8k4 DMMA)" begin
    # DMMA is IEEE fp64 FMA — kernel and host reference differ only by
    # summation order, so the tolerance is essentially machine epsilon
    # scaled by K.
    for (M, N, K) in [(8, 8, 4), (8, 8, 64), (16, 16, 16),
                      (32, 32, 32), (64, 64, 64)]
        D, D_ref = run_fp64_gemm(M, N, K)
        @test all(abs.(D .- D_ref) .<= 1e-12 .+ 1e-12 .* abs.(D_ref))
    end
end
