# TEST_TARGET: requires=gpu evidence=runtime target=sm_80
#
# Ported from pyptx/examples/ampere/gemm.py
# (https://github.com/patrick-toulme/pyptx).
# Copyright 2026 Patrick Toulmé. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Translated to Julia and adapted.
#
# Ampere (sm_80) bf16 GEMM via mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32.
#
# Bare-bones single-warp design — one CTA computes a 16×8 output tile, one
# mma.sync per K-tile of width 16. Direct global-memory loads, no SMEM staging,
# no cp.async prefetch. Demonstrates the Ampere tensor-core path end-to-end
# (the ergonomics of driving sm_80+ tensor cores from PTX.jl) — not a perf demo.
# A future high-perf variant layers cp.async + SMEM staging + ldmatrix on top.
#
# Inputs (flat row-major flattened):
#   A    :: (M, K) bf16  stored as UInt16, row-major
#   B_T  :: (N, K) bf16  stored as UInt16, row-major  (B^T so K is contiguous)
#   D    :: (M, N) f32   row-major
#
# Grid:  (N/8, M/16)        — blockIdx.x picks N tile, blockIdx.y picks M tile
# Block: (32, 1, 1)         — one warp per CTA
#
# Per-lane fragment indices (PTX ISA 9.2 §9.7.13.4, m16n8k16.row.col):
#   group     = tid >> 2          (0..7)   — 1 of 8 row positions
#   in_group  = tid & 0x3         (0..3)   — 1 of 4 col positions
#   col_lo    = 2 * in_group      (0,2,4,6)
#
#   A frag  NTuple{4, UInt32}: each reg packs 2 bf16
#     a[1] = pack(A[g,    col_lo + 0], A[g,    col_lo + 1])
#     a[2] = pack(A[g+8,  col_lo + 0], A[g+8,  col_lo + 1])
#     a[3] = pack(A[g,    col_lo + 8], A[g,    col_lo + 9])
#     a[4] = pack(A[g+8,  col_lo + 8], A[g+8,  col_lo + 9])
#   B frag  NTuple{2, UInt32}: B viewed as (N, K) row-major
#     b[1] = pack(B_T[g,  col_lo + 0], B_T[g,  col_lo + 1])
#     b[2] = pack(B_T[g,  col_lo + 8], B_T[g,  col_lo + 9])
#   D frag  NTuple{4, Float32}
#     d[1] = D[g,    col_lo + 0],   d[2] = D[g,    col_lo + 1]
#     d[3] = D[g+8,  col_lo + 0],   d[4] = D[g+8,  col_lo + 1]

using Random

bf16_bits(x::Float32)  = UInt16(reinterpret(UInt32, x) >> 16)
bf16_to_f32(b::UInt16) = reinterpret(Float32, UInt32(b) << 16)

# Pack a (rows × cols) Float32 matrix into row-major bf16 UInt16 flat storage.
function pack_bf16_rowmajor(A::AbstractMatrix{Float32})
    rows, cols = size(A)
    out = Vector{UInt16}(undef, rows * cols)
    @inbounds for i in 1:rows, j in 1:cols
        out[(i-1)*cols + j] = bf16_bits(A[i, j])
    end
    out
end

# Quantize Float32 to bf16 precision via round-trip through bf16 storage,
# so the host-side reference matmul accumulates over the same bits the kernel
# saw — no spurious tolerance budget burned on input quantization.
quantize_bf16(A) = bf16_to_f32.(bf16_bits.(A))

const BM, BN, BK = 16, 8, 16

function gemm_ampere_kernel!(
        D::CuDeviceVector{Float32},
        A::CuDeviceVector{UInt16},
        B_T::CuDeviceVector{UInt16},
        ::Val{M}, ::Val{N}, ::Val{K}) where {M, N, K}
    n_iters = K ÷ BK

    m_base = Int(ptx"mov.u32"(sreg"ctaid.y")) * BM
    n_base = Int(ptx"mov.u32"(sreg"ctaid.x")) * BN

    tid    = ptx"mov.u32"(sreg"tid.x")
    gid    = Int(tid >> UInt32(2))      # group_id   in [0, 8)
    tig    = Int(tid & UInt32(0x3))     # in_group   in [0, 4)
    col_lo = tig << 1                   # 2*in_group → 0,2,4,6

    row_lo = m_base + gid
    row_hi = row_lo + 8
    n_col  = n_base + gid

    pa = pointer(A)
    pb = pointer(B_T)
    pd = pointer(D)

    acc = (0f0, 0f0, 0f0, 0f0)

    @inbounds for ki in 0:n_iters-1
        k_base = ki * BK

        # A frag: 4 b32 regs at (drow, dcol) ∈ {(0,0),(8,0),(0,8),(8,8)}.
        # A is (M, K) row-major bf16; byte offset = (row*K + col_elem) * 2.
        a1 = ptx"ld.global.b32"(pa + (row_lo * K + (k_base + col_lo + 0)) * 2)
        a2 = ptx"ld.global.b32"(pa + (row_hi * K + (k_base + col_lo + 0)) * 2)
        a3 = ptx"ld.global.b32"(pa + (row_lo * K + (k_base + col_lo + 8)) * 2)
        a4 = ptx"ld.global.b32"(pa + (row_hi * K + (k_base + col_lo + 8)) * 2)

        # B frag: 2 b32 regs at dcol ∈ {0, 8}.
        # B_T is (N, K) row-major bf16; byte offset = (n*K + col_elem) * 2.
        b1 = ptx"ld.global.b32"(pb + (n_col * K + (k_base + col_lo + 0)) * 2)
        b2 = ptx"ld.global.b32"(pb + (n_col * K + (k_base + col_lo + 8)) * 2)

        acc = ptx"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"(
            (a1, a2, a3, a4), (b1, b2), acc)
    end

    # D frag store: 4 f32 regs at (drow, dcol) ∈ {(0,0),(0,1),(8,0),(8,1)}.
    # D is (M, N) f32 row-major; byte offset = (row*N + col_elem) * 4.
    d_col_base = n_base + col_lo
    @inbounds begin
        ptx"st.global.f32"(pd + (row_lo * N + (d_col_base + 0)) * 4, acc[1])
        ptx"st.global.f32"(pd + (row_lo * N + (d_col_base + 1)) * 4, acc[2])
        ptx"st.global.f32"(pd + (row_hi * N + (d_col_base + 0)) * 4, acc[3])
        ptx"st.global.f32"(pd + (row_hi * N + (d_col_base + 1)) * 4, acc[4])
    end
    return nothing
end

function run_ampere_gemm(M::Int, N::Int, K::Int)
    @assert M % BM == 0 "M=$M must be divisible by $BM"
    @assert N % BN == 0 "N=$N must be divisible by $BN"
    @assert K % BK == 0 "K=$K must be divisible by $BK"

    rng    = MersenneTwister(M * 7919 + N * 31 + K)
    A_f32  = quantize_bf16(0.1f0 .* randn(rng, Float32, M, K))
    BT_f32 = quantize_bf16(0.1f0 .* randn(rng, Float32, N, K))

    A_d   = CuArray(pack_bf16_rowmajor(A_f32))
    BT_d  = CuArray(pack_bf16_rowmajor(BT_f32))
    D_d   = CUDACore.zeros(Float32, M * N)

    @cuda blocks=(N ÷ BN, M ÷ BM) threads=32 gemm_ampere_kernel!(
        D_d, A_d, BT_d, Val(M), Val(N), Val(K))
    CUDACore.synchronize()

    # row-major flat → (M, N) Matrix
    D     = Matrix(reshape(Array(D_d), N, M)')
    D_ref = A_f32 * BT_f32'   # (M, K) * (K, N)
    return D, D_ref
end

@testset "Ampere bf16 GEMM (m16n8k16, multi-CTA, no SMEM)" begin
    # Five sizes mirroring the pyptx test sweep:
    #   1) single CTA, single K-iter
    #   2) single CTA, 4 K-iters
    #   3) 2x2 CTAs
    #   4) 4x4 CTAs
    #   5) 8x8 CTAs
    # numpy-allclose-style criterion: |d - r| ≤ atol + rtol·|r|. A pure
    # relative-error metric blows up wherever a D_ref element happens to
    # cancel near zero from random accumulation — metric artifact, not a
    # correctness failure.
    for (M, N, K) in [(16, 8, 16), (16, 8, 64), (32, 16, 32),
                      (64, 32, 64), (128, 64, 128)]
        D, D_ref = run_ampere_gemm(M, N, K)
        @test all(abs.(D .- D_ref) .<= 1f-2 .+ 1f-2 .* abs.(D_ref))
    end
end
