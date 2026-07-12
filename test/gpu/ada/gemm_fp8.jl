# REQUIRES CC 8.9
#
# Ported from cutlass/examples/58_ada_fp8_gemm (NVIDIA CUTLASS,
# BSD-3-Clause). The original instantiates a CUTLASS fp8 GEMM with f32
# accumulation on Ada; what's ported is the end-to-end shape: e4m3 inputs,
# tensor-core mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
# (sm_89's fp8 path), f32 accumulate and output.
#
# Storage is typed: CuArray{Float8_E4M3FN} (Microfloats). PTX `.e4m3` is
# the OCP E4M3FN format — no Inf, NaN at 0x7F/0xFF — which is exactly
# Microfloats' Float8_E4M3FN, so host quantization/dequantization runs
# through the package (pure integer software conversion; no LLVM fp8
# type exists, so unlike bf16 there is no vectorizer crash to dodge).
#
# Same single-warp corpus tiling: one CTA per 16×8 output tile, k-tile of
# 32, direct global loads. Four consecutive e4m3 bytes along K load as one
# b32 fragment register (low byte = lowest k, little-endian match).
#
# Inputs (flat row-major flattened):
#   A    :: (M, K) e4m3, row-major
#   B_T  :: (N, K) e4m3, row-major  (B^T so K is contiguous)
#   D    :: (M, N) f32,  row-major
#
# Per-lane fragment indices (PTX ISA 9.3 §9.7.13.4, m16n8k32 8-bit):
#   gid = tid >> 2 (0..7), tig = tid & 0x3 (0..3)
#   A frag NTuple{4, UInt32} (4 e4m3 per reg):
#     a[1] = A[gid,   4tig .. 4tig+3]      a[2] = A[gid+8, 4tig .. 4tig+3]
#     a[3] = A[gid,   4tig+16 .. 4tig+19]  a[4] = A[gid+8, 4tig+16 .. 4tig+19]
#   B frag NTuple{2, UInt32}:  (indexing B_T[n, k])
#     b[1] = B_T[gid, 4tig .. 4tig+3]      b[2] = B_T[gid, 4tig+16 .. 4tig+19]
#   D frag NTuple{4, Float32}: same as every m16n8kX shape — see gemm.jl.

using Random
using Microfloats

const F8_BM, F8_BN, F8_BK = 16, 8, 32

function gemm_fp8_kernel!(
        D::CuDeviceVector{Float32},
        A::CuDeviceVector{Float8_E4M3FN},
        B_T::CuDeviceVector{Float8_E4M3FN},
        ::Val{M}, ::Val{N}, ::Val{K}) where {M, N, K}
    n_iters = K ÷ F8_BK

    m_base = Int(ptx"mov.u32"(sreg"ctaid.y")) * F8_BM
    n_base = Int(ptx"mov.u32"(sreg"ctaid.x")) * F8_BN

    tid = ptx"mov.u32"(sreg"tid.x")
    gid = Int(tid >> UInt32(2))
    tig = Int(tid & UInt32(0x3))
    kl  = 4 * tig                       # fragment k low: 0,4,8,12

    row_lo = m_base + gid
    row_hi = row_lo + 8
    n_col  = n_base + gid

    pa = pointer(A)
    pb = pointer(B_T)

    acc = (0f0, 0f0, 0f0, 0f0)

    @inbounds for ki in 0:n_iters-1
        k_base = ki * F8_BK

        # 1 byte per element — offsets are element offsets directly.
        a1 = ptx"ld.global.b32"(pa + row_lo * K + k_base + kl)
        a2 = ptx"ld.global.b32"(pa + row_hi * K + k_base + kl)
        a3 = ptx"ld.global.b32"(pa + row_lo * K + k_base + kl + 16)
        a4 = ptx"ld.global.b32"(pa + row_hi * K + k_base + kl + 16)

        b1 = ptx"ld.global.b32"(pb + n_col * K + k_base + kl)
        b2 = ptx"ld.global.b32"(pb + n_col * K + k_base + kl + 16)

        acc = ptx"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"(
            (a1, a2, a3, a4), (b1, b2), acc)
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

flatten_rowmajor(A::AbstractMatrix) = vec(permutedims(A))

function run_fp8_gemm(M::Int, N::Int, K::Int)
    @assert M % F8_BM == 0 && N % F8_BN == 0 && K % F8_BK == 0

    rng   = MersenneTwister(M * 7919 + N * 31 + K)
    # e4m3 keeps only 3 mantissa bits; modest magnitudes keep the values
    # well inside its dynamic range (max 448).
    A8    = Float8_E4M3FN.(0.5f0 .* randn(rng, Float32, M, K))
    BT8   = Float8_E4M3FN.(0.5f0 .* randn(rng, Float32, N, K))

    A_d  = CuArray(flatten_rowmajor(A8))
    BT_d = CuArray(flatten_rowmajor(BT8))
    D_d  = CUDACore.zeros(Float32, M * N)

    @cuda blocks=(N ÷ F8_BN, M ÷ F8_BM) threads=32 gemm_fp8_kernel!(
        D_d, A_d, BT_d, Val(M), Val(N), Val(K))
    CUDACore.synchronize()

    D = Matrix(reshape(Array(D_d), N, M)')
    D_ref = Float32.(A8) * Float32.(BT8)'
    return D, D_ref
end

@testset "Ada fp8 GEMM (m16n8k32 e4m3, f32 accumulate)" begin
    # e4m3×e4m3 products are exact in f32 (4-bit × 4-bit significands), so
    # against a host reference over the same quantized inputs the only
    # discrepancy *should* be f32 summation order (~1e-4 at K=256). Measured
    # on sm_89 it is ~100× that (max |Δ| ≈ 9e-3 at K=256, growing with K):
    # Ada's fp8 tensor cores accumulate the "f32" C/D operand at reduced
    # alignment precision rather than as true fp32 adds. This is the same
    # hardware trade CUTLASS surfaces on Hopper as "fast accumulation" —
    # on sm_89 it is simply how the datapath works. The atol term scales
    # with K to track it (~2× margin over measured worst case).
    for (M, N, K) in [(16, 8, 32), (16, 8, 128), (32, 16, 64),
                      (64, 32, 128), (128, 64, 256)]
        D, D_ref = run_fp8_gemm(M, N, K)
        @test all(abs.(D .- D_ref) .<= K * 8f-5 .+ 1f-3 .* abs.(D_ref))
    end
end
