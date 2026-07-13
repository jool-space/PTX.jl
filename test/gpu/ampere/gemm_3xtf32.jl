# TEST_TARGET: requires=gpu evidence=runtime target=sm_80
#
# Ported from cutlass/examples/27_ampere_3xtf32_fast_accurate_tensorop_gemm
# (NVIDIA CUTLASS, BSD-3-Clause). The original swaps CUTLASS's OpMultiplyAdd
# for OpMultiplyAddFastF32; what's ported is the trick itself, hand-rolled:
#
#   a × b = (a_big + a_small) × (b_big + b_small)
#         ≈ a_big×b_big + a_big×b_small + a_small×b_big     (3 mmas)
#   big   = cvt.rna.tf32(x)                                 (top 10 mantissa bits)
#   small = cvt.rna.tf32(x - big)                           (next ~11 bits)
#
# a_small×b_small is discarded (≈2^-22 relative — below f32 accumulation
# noise). The result is near-SGEMM accuracy at tensor-core speed. The file
# runs the same GEMM three ways and checks the error hierarchy against a
# Float64 reference:
#
#   1xTF32  — inputs rounded to one tf32 (fast, ~2^-11 relative error)
#   3xTF32  — the big/small decomposition above (~f32-level error)
#   FP32    — host BLAS sgemm baseline
#
# Same single-warp tiling as gemm_tf32.jl: one CTA per 16×8 output tile,
# k-tile of 8, direct global loads. Fragment layout: see gemm_tf32.jl.

using Random
using LinearAlgebra

const X3_BM, X3_BN, X3_BK = 16, 8, 8

# tf32 bit pattern → the f32 value it represents (low 13 bits are zero).
@inline tf32_val(t::UInt32) = reinterpret(Float32, t)

@inline function tf32_split(x::Float32)
    big   = ptx"cvt.rna.tf32.f32"(x)
    small = ptx"cvt.rna.tf32.f32"(x - tf32_val(big))
    return big, small
end

function gemm_3xtf32_kernel!(
        D::CuDeviceVector{Float32},
        A::CuDeviceVector{Float32},
        B_T::CuDeviceVector{Float32},
        ::Val{M}, ::Val{N}, ::Val{K}, ::Val{THREEX}) where {M, N, K, THREEX}
    n_iters = K ÷ X3_BK

    m_base = Int(ptx"mov.u32"(sreg"ctaid.y")) * X3_BM
    n_base = Int(ptx"mov.u32"(sreg"ctaid.x")) * X3_BN

    tid = ptx"mov.u32"(sreg"tid.x")
    gid = Int(tid >> UInt32(2))
    tig = Int(tid & UInt32(0x3))

    row_lo = m_base + gid
    row_hi = row_lo + 8
    n_col  = n_base + gid

    acc = (0f0, 0f0, 0f0, 0f0)

    @inbounds for ki in 0:n_iters-1
        k_base = ki * X3_BK

        a1b, a1s = tf32_split(A[row_lo * K + k_base + tig + 1])
        a2b, a2s = tf32_split(A[row_hi * K + k_base + tig + 1])
        a3b, a3s = tf32_split(A[row_lo * K + k_base + tig + 5])
        a4b, a4s = tf32_split(A[row_hi * K + k_base + tig + 5])

        b1b, b1s = tf32_split(B_T[n_col * K + k_base + tig + 1])
        b2b, b2s = tf32_split(B_T[n_col * K + k_base + tig + 5])

        a_big = (a1b, a2b, a3b, a4b)
        b_big = (b1b, b2b)

        if THREEX
            # Small corrections first, big product last — the big terms
            # dominate, so adding them at the end loses the least accuracy.
            acc = ptx"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"(
                a_big, (b1s, b2s), acc)
            acc = ptx"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"(
                (a1s, a2s, a3s, a4s), b_big, acc)
        end
        acc = ptx"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"(
            a_big, b_big, acc)
    end

    d_col = n_base + 2 * tig
    @inbounds begin
        D[row_lo * N + d_col + 1] = acc[1]
        D[row_lo * N + d_col + 2] = acc[2]
        D[row_hi * N + d_col + 1] = acc[3]
        D[row_hi * N + d_col + 2] = acc[4]
    end
    return nothing
end

flatten_rowmajor(A::AbstractMatrix{Float32}) = vec(permutedims(A))

# Launch the kernel in 1x or 3x mode and return D as an (M, N) Matrix.
function run_tf32_variant(A_d, BT_d, M, N, K; threex::Bool)
    D_d = CUDACore.zeros(Float32, M * N)
    @cuda blocks=(N ÷ X3_BN, M ÷ X3_BM) threads=32 gemm_3xtf32_kernel!(
        D_d, A_d, BT_d, Val(M), Val(N), Val(K), Val(threex))
    CUDACore.synchronize()
    return Matrix(reshape(Array(D_d), N, M)')
end

rel_l2(D, D_ref) = norm(Float64.(D) .- D_ref) / norm(D_ref)

@testset "Ampere 3xTF32 fast-accurate GEMM (error hierarchy vs FP64)" begin
    for (M, N, K) in [(32, 16, 64), (128, 64, 256)]
        rng    = MersenneTwister(M * 7919 + N * 31 + K)
        A_f32  = randn(rng, Float32, M, K)
        BT_f32 = randn(rng, Float32, N, K)

        A_d  = CuArray(flatten_rowmajor(A_f32))
        BT_d = CuArray(flatten_rowmajor(BT_f32))

        D_1x = run_tf32_variant(A_d, BT_d, M, N, K; threex = false)
        D_3x = run_tf32_variant(A_d, BT_d, M, N, K; threex = true)

        D_ref64 = Float64.(A_f32) * Float64.(BT_f32)'   # ground truth
        D_fp32  = A_f32 * BT_f32'                        # host SGEMM baseline

        e_1x   = rel_l2(D_1x, D_ref64)
        e_3x   = rel_l2(D_3x, D_ref64)
        e_fp32 = rel_l2(D_fp32, D_ref64)

        # tf32's 10-bit mantissa puts 1x at ~2^-11 relative; the big/small
        # split recovers ~21 mantissa bits, landing near f32 accumulation
        # noise. Thresholds sit an order of magnitude off the expected
        # values so RNG draws can't flake them.
        @test e_1x > 5f-5                # 1x really is low-precision
        @test e_3x < e_1x / 10           # the correction terms pay off
        @test e_3x < 1f-5                # near-fp32 territory
        @test e_fp32 < 1f-6              # sanity: baseline is what we think
    end
end
