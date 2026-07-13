# TEST_TARGET: requires=gpu evidence=runtime target=sm_80
#
# Ported from cutlass/examples/14_ampere_tf32_tensorop_gemm
# (NVIDIA CUTLASS, BSD-3-Clause). The original is a CUTLASS device-API
# template instantiation; what's ported is the concept: fp32 data
# accelerated by Ampere tensor cores with *no host-side format change* —
# the kernel loads plain f32 and converts to tf32 internally.
#
# Ampere (sm_80) tf32 GEMM via mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32,
# with in-kernel cvt.rna.tf32.f32 (tf32 = 8-bit exponent, 10-bit mantissa,
# stored in the low-19-of-32 bits — a valid f32 bit pattern with the low
# 13 mantissa bits zeroed).
#
# Same bare-bones single-warp design as gemm.jl — one CTA computes a 16×8
# output tile, one mma.sync per K-tile of width 8, direct global loads.
#
# Inputs (flat row-major flattened):
#   A    :: (M, K) f32, row-major
#   B_T  :: (N, K) f32, row-major  (B^T so K is contiguous)
#   D    :: (M, N) f32, row-major
#
# Grid:  (N/8, M/16)        — blockIdx.x picks N tile, blockIdx.y picks M tile
# Block: (32, 1, 1)         — one warp per CTA
#
# Per-lane fragment indices (PTX ISA 9.3 §9.7.13.4, m16n8k8 tf32):
#   gid = tid >> 2  (0..7),  tig = tid & 0x3  (0..3)
#
#   A frag  NTuple{4, UInt32} (one tf32 per reg):
#     a[1] = A[gid,   tig]      a[2] = A[gid+8, tig]
#     a[3] = A[gid,   tig+4]    a[4] = A[gid+8, tig+4]
#   B frag  NTuple{2, UInt32}:  (B is K×N col layout ⇒ index B_T[n, k])
#     b[1] = B_T[gid, tig]      b[2] = B_T[gid, tig+4]
#   D frag  NTuple{4, Float32}: same as every m16n8kX shape
#     d[1] = D[gid, 2tig]  d[2] = D[gid, 2tig+1]
#     d[3] = D[gid+8, 2tig]  d[4] = D[gid+8, 2tig+1]

using Random

# Host emulation of cvt.rna.tf32.f32: round-to-nearest-ties-away on the
# 13 dropped mantissa bits (add half-ULP, truncate). Sign-magnitude f32
# means the magnitude trick handles negatives; mantissa overflow carries
# into the exponent, which is exactly the correct rounding behavior.
tf32_quantize(x::Float32) =
    reinterpret(Float32, (reinterpret(UInt32, x) + 0x00001000) & 0xffffe000)

const TF_BM, TF_BN, TF_BK = 16, 8, 8

function gemm_tf32_kernel!(
        D::CuDeviceVector{Float32},
        A::CuDeviceVector{Float32},
        B_T::CuDeviceVector{Float32},
        ::Val{M}, ::Val{N}, ::Val{K}) where {M, N, K}
    n_iters = K ÷ TF_BK

    m_base = Int(ptx"mov.u32"(sreg"ctaid.y")) * TF_BM
    n_base = Int(ptx"mov.u32"(sreg"ctaid.x")) * TF_BN

    tid = ptx"mov.u32"(sreg"tid.x")
    gid = Int(tid >> UInt32(2))         # 0..7
    tig = Int(tid & UInt32(0x3))        # 0..3

    row_lo = m_base + gid
    row_hi = row_lo + 8
    n_col  = n_base + gid

    acc = (0f0, 0f0, 0f0, 0f0)

    @inbounds for ki in 0:n_iters-1
        k_base = ki * TF_BK

        # f32 loads (1-based flat row-major), converted to tf32 in-register.
        a1 = ptx"cvt.rna.tf32.f32"(A[row_lo * K + k_base + tig + 1])
        a2 = ptx"cvt.rna.tf32.f32"(A[row_hi * K + k_base + tig + 1])
        a3 = ptx"cvt.rna.tf32.f32"(A[row_lo * K + k_base + tig + 5])
        a4 = ptx"cvt.rna.tf32.f32"(A[row_hi * K + k_base + tig + 5])

        b1 = ptx"cvt.rna.tf32.f32"(B_T[n_col * K + k_base + tig + 1])
        b2 = ptx"cvt.rna.tf32.f32"(B_T[n_col * K + k_base + tig + 5])

        acc = ptx"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"(
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

# Row-major flatten of a Julia (col-major) matrix.
flatten_rowmajor(A::AbstractMatrix{Float32}) = vec(permutedims(A))

function run_tf32_gemm(M::Int, N::Int, K::Int)
    @assert M % TF_BM == 0 && N % TF_BN == 0 && K % TF_BK == 0

    rng    = MersenneTwister(M * 7919 + N * 31 + K)
    A_f32  = 0.1f0 .* randn(rng, Float32, M, K)
    BT_f32 = 0.1f0 .* randn(rng, Float32, N, K)

    A_d  = CuArray(flatten_rowmajor(A_f32))
    BT_d = CuArray(flatten_rowmajor(BT_f32))
    D_d  = CUDACore.zeros(Float32, M * N)

    @cuda blocks=(N ÷ TF_BN, M ÷ TF_BM) threads=32 gemm_tf32_kernel!(
        D_d, A_d, BT_d, Val(M), Val(N), Val(K))
    CUDACore.synchronize()

    D = Matrix(reshape(Array(D_d), N, M)')
    # Reference over the SAME tf32-quantized inputs the tensor cores saw,
    # so the only tolerance budget is f32 accumulation-order noise.
    D_ref = tf32_quantize.(A_f32) * tf32_quantize.(BT_f32)'
    return D, D_ref
end

@testset "Ampere tf32 GEMM (m16n8k8, fp32 in/out, in-kernel cvt.rna)" begin
    for (M, N, K) in [(16, 8, 8), (16, 8, 64), (32, 16, 32),
                      (64, 32, 64), (128, 64, 128)]
        D, D_ref = run_tf32_gemm(M, N, K)
        @test all(abs.(D .- D_ref) .<= 1f-4 .+ 1f-4 .* abs.(D_ref))
    end
end
