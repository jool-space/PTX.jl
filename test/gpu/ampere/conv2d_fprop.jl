# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.0
#
# Ported from cutlass/examples/16_ampere_tensorop_conv2dfprop
# (NVIDIA CUTLASS, BSD-3-Clause). The original instantiates CUTLASS's
# implicit-GEMM convolution; what's ported is the implicit-GEMM mapping
# itself, hand-rolled onto the same single-warp mma tiling the gemm.jl
# corpus files use:
#
#   GEMM M = N·P·Q   (output positions)     row m  ↔ (n, p, q)
#   GEMM N = Kf      (filters)              col    ↔ k
#   GEMM K = R·S·C   (reduction)            index  ↔ (r, s, c)
#
#   O[n,p,q,k] = Σ_{r,s,c} X[n, p·stride−pad+r, q·stride−pad+s, c] · W[k,r,s,c]
#
# The A-matrix is never materialized: each lane gathers its fragment
# elements straight from NHWC global memory, with out-of-bounds (padding)
# taps contributing zero. That elementwise gather *is* the "implicit" in
# implicit GEMM — im2col performed by index arithmetic in registers.
# W is KRSC, so a filter's reduction axis is contiguous and B fragments
# load as plain b32 pairs, exactly like B_T in gemm.jl.
#
# Storage is typed: CuArray{BFloat16} (Core.BFloat16) rather than UInt16
# bit-bags. Host-side f32→bf16 broadcast is avoided — on Julia 1.12.6/x86
# it dies with `LLVM ERROR: Cannot select v16bf16` (vectorizer forms bf16
# SIMD ops ISel can't match) — so packing goes through integer bit ops.
#
# Grid:  (Kf/8, M/16), one warp per CTA, m16n8k16 bf16 mma per 16-wide
# k-slice. R·S·C must be a multiple of 16 (C a multiple of 16 suffices).
# Fragment layout: PTX ISA 9.3 §9.7.13.4, m16n8k16.row.col — see gemm.jl.

using Random
using Microfloats: BFloat16

bf16_pack(lo::UInt16, hi::UInt16) = UInt32(lo) | (UInt32(hi) << 16)

# f32 → bf16 bits, round-to-nearest-even, pure integer ops (see header).
rne_bf16_bits(x::Float32) =
    (b = reinterpret(UInt32, x);
     UInt16((b + 0x7fff + ((b >> 16) & 0x1)) >> 16))
to_bf16(x::Array{Float32}) = collect(reinterpret(BFloat16, rne_bf16_bits.(x)))
quantize_bf16(x) = reinterpret.(Float32, UInt32.(rne_bf16_bits.(x)) .<< 16)

# One implicit-A element: X[n, h, w, c] for GEMM coords (row-decoded n/p/q,
# reduction index rsc), or zero when the tap lands in the padding halo.
@inline function x_tap(X::CuDeviceVector{BFloat16}, n, p, q, rsc, dims)
    r, sc = divrem(rsc, dims.S * dims.C)
    s, c  = divrem(sc, dims.C)
    h = p * dims.stride - dims.pad + r
    w = q * dims.stride - dims.pad + s
    (0 <= h < dims.H) & (0 <= w < dims.W) || return UInt16(0)
    return reinterpret(UInt16,
        @inbounds X[((n * dims.H + h) * dims.W + w) * dims.C + c + 1])
end

function conv2d_fprop_kernel!(
        O::CuDeviceVector{Float32},
        X::CuDeviceVector{BFloat16},
        Wf::CuDeviceVector{BFloat16},
        ::Val{DIMS}) where {DIMS}
    d   = DIMS
    RSC = d.R * d.S * d.C
    PQ  = d.P * d.Q

    m_base = Int(ptx"mov.u32"(sreg"ctaid.y")) * 16
    k_base = Int(ptx"mov.u32"(sreg"ctaid.x")) * 8

    tid = ptx"mov.u32"(sreg"tid.x")
    gid = Int(tid >> UInt32(2))         # 0..7
    tig = Int(tid & UInt32(0x3))        # 0..3
    cl  = 2 * tig                       # fragment column low: 0,2,4,6

    # Decode the two output positions this lane accumulates (rows m, m+8).
    m_lo = m_base + gid
    m_hi = m_lo + 8
    n_lo, pq_lo = divrem(m_lo, PQ); p_lo, q_lo = divrem(pq_lo, d.Q)
    n_hi, pq_hi = divrem(m_hi, PQ); p_hi, q_hi = divrem(pq_hi, d.Q)

    filt = k_base + gid                 # B-fragment filter index
    pw   = pointer(Wf)

    acc = (0f0, 0f0, 0f0, 0f0)

    @inbounds for kt in 0:(RSC ÷ 16 - 1)
        rsc = kt * 16 + cl

        # A frag: 4 regs, elements (row, rsc+{0,1}) and (row, rsc+8+{0,1}).
        a1 = bf16_pack(x_tap(X, n_lo, p_lo, q_lo, rsc,     DIMS),
                       x_tap(X, n_lo, p_lo, q_lo, rsc + 1, DIMS))
        a2 = bf16_pack(x_tap(X, n_hi, p_hi, q_hi, rsc,     DIMS),
                       x_tap(X, n_hi, p_hi, q_hi, rsc + 1, DIMS))
        a3 = bf16_pack(x_tap(X, n_lo, p_lo, q_lo, rsc + 8, DIMS),
                       x_tap(X, n_lo, p_lo, q_lo, rsc + 9, DIMS))
        a4 = bf16_pack(x_tap(X, n_hi, p_hi, q_hi, rsc + 8, DIMS),
                       x_tap(X, n_hi, p_hi, q_hi, rsc + 9, DIMS))

        # B frag: W is (Kf, RSC) row-major bf16 — contiguous b32 pairs.
        b1 = ptx"ld.global.b32"(pw + (filt * RSC + rsc) * 2)
        b2 = ptx"ld.global.b32"(pw + (filt * RSC + rsc + 8) * 2)

        acc = ptx"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"(
            (a1, a2, a3, a4), (b1, b2), acc)
    end

    # O is (N·P·Q, Kf) row-major f32.
    o_col = k_base + cl
    @inbounds begin
        O[m_lo * d.Kf + o_col + 1] = acc[1]
        O[m_lo * d.Kf + o_col + 2] = acc[2]
        O[m_hi * d.Kf + o_col + 1] = acc[3]
        O[m_hi * d.Kf + o_col + 2] = acc[4]
    end
    return nothing
end

# Direct-convolution host reference over bf16-quantized data, f32 accumulate.
function conv2d_ref(Xq::Array{Float32, 4}, Wq::Array{Float32, 4}, dims)
    O = zeros(Float32, dims.N * dims.P * dims.Q, dims.Kf)
    for n in 0:dims.N-1, p in 0:dims.P-1, q in 0:dims.Q-1, k in 0:dims.Kf-1
        s_acc = 0f0
        for r in 0:dims.R-1, s in 0:dims.S-1, c in 0:dims.C-1
            h = p * dims.stride - dims.pad + r
            w = q * dims.stride - dims.pad + s
            (0 <= h < dims.H && 0 <= w < dims.W) || continue
            s_acc += Xq[n+1, h+1, w+1, c+1] * Wq[k+1, r+1, s+1, c+1]
        end
        O[(n * dims.P + p) * dims.Q + q + 1, k+1] = s_acc
    end
    return O
end

# NHWC/KRSC flatten: Julia arrays are col-major, so permutedims reverses
# the axis order before vec() to get row-major (last-axis-contiguous) flat.
flatten_last_contig(A::Array{Float32, 4}) = vec(permutedims(A, (4, 3, 2, 1)))

function run_conv2d(dims)
    @assert (dims.N * dims.P * dims.Q) % 16 == 0
    @assert dims.Kf % 8 == 0
    @assert (dims.R * dims.S * dims.C) % 16 == 0

    rng = MersenneTwister(dims.H * 131 + dims.C * 17 + dims.Kf)
    X4  = 0.25f0 .* randn(rng, Float32, dims.N, dims.H, dims.W, dims.C)
    W4  = 0.25f0 .* randn(rng, Float32, dims.Kf, dims.R, dims.S, dims.C)

    X_d = CuArray(to_bf16(flatten_last_contig(X4)))
    W_d = CuArray(to_bf16(flatten_last_contig(W4)))
    O_d = CUDACore.zeros(Float32, dims.N * dims.P * dims.Q * dims.Kf)

    M = dims.N * dims.P * dims.Q
    @cuda blocks=(dims.Kf ÷ 8, M ÷ 16) threads=32 conv2d_fprop_kernel!(
        O_d, X_d, W_d, Val(dims))
    CUDACore.synchronize()

    O     = Matrix(reshape(Array(O_d), dims.Kf, M)')
    O_ref = conv2d_ref(quantize_bf16(X4), quantize_bf16(W4), dims)
    return O, O_ref
end

conv_dims(; N, H, W, C, Kf, R, S, pad, stride) =
    (; N, H, W, C, Kf, R, S, pad, stride,
       P = (H + 2pad - R) ÷ stride + 1,
       Q = (W + 2pad - S) ÷ stride + 1)

@testset "Ampere conv2d fprop (implicit GEMM, bf16 tensor cores)" begin
    configs = [
        # 3×3 same-pad, stride 1 — the bread-and-butter conv
        conv_dims(N = 1, H = 8, W = 8, C = 16, Kf = 8,  R = 3, S = 3, pad = 1, stride = 1),
        # strided 3×3 with batch — output halo taps cross image edges
        conv_dims(N = 2, H = 7, W = 7, C = 32, Kf = 16, R = 3, S = 3, pad = 1, stride = 2),
        # 1×1 pointwise — implicit GEMM degenerates to plain GEMM
        conv_dims(N = 1, H = 4, W = 4, C = 64, Kf = 8,  R = 1, S = 1, pad = 0, stride = 1),
    ]
    for dims in configs
        O, O_ref = run_conv2d(dims)
        @test all(abs.(O .- O_ref) .<= 1f-2 .+ 1f-2 .* abs.(O_ref))
    end
end
