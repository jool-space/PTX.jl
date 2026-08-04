# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=7.0
using Random
using PTX.Warps: warp_reduce

# Ported from pyptx/examples/hopper/layer_norm.py
# (https://github.com/patrick-toulme/pyptx).
# Copyright 2026 Patrick Toulmé. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Translated to Julia and adapted.

# Fused LayerNorm.
#
#   y[b, i] = (x[b, i] - mean_b) * rstd_b * w[i] + bias[i]
#
# where  mean_b = mean(x[b, :])
#        rstd_b = 1 / sqrt(var(x[b, :]) + eps)
#
# Same shape as rms_norm but accumulates both Σx (for mean) and Σx² (for
# variance via E[x²] - E[x]²). Two warp reductions per row instead of one.
# v4 path only — N must be divisible by block*4.

const LN_WARP_SIZE = UInt32(32)

# ptx"add.f32" (not Julia +): the kernel's reduction op is the plain PTX
# add, and warp_reduce applies op verbatim between shuffle rounds.
@inline _ln_warp_reduce_sum(v::Float32) =
    warp_reduce((a, b) -> ptx"add.f32"(a, b), v)

function _layer_norm_v4_kernel!(
        Y::CuDeviceVector{Float32},
        X::CuDeviceVector{Float32},
        W::CuDeviceVector{Float32},
        B::CuDeviceVector{Float32},
        ::Val{N}, ::Val{block}, ::Val{eps}) where {N, block, eps}
    num_warps = block ÷ Int(LN_WARP_SIZE)
    v4_iters  = N ÷ (block * 4)
    @assert v4_iters * block * 4 == N
    @assert v4_iters <= 16    # register-tile unroll bound (64 f32 per thread)

    # 2 partials per warp: [warp_id, 0] = Σx, [warp_id, 1] = Σx².
    partials = CuStaticSharedArray(Float32, (16, 2))   # max 16 warps (block ≤ 512)
    stats    = CuStaticSharedArray(Float32, 2)         # [Σx, Σx²]

    tid = ptx"mov.u32"(sreg"tid.x")
    row = ptx"mov.u32"(sreg"ctaid.x")
    row_byte_off = Int(row) * (N * 4)
    px = pointer(X) + row_byte_off
    py = pointer(Y) + row_byte_off
    pw = pointer(W)
    pb = pointer(B)
    lane    = tid & UInt32(0x1F)
    warp_id = tid >> UInt32(5)
    elem_base = Int(tid) * 4

    # --- pass 1: load + accumulate Σx, Σx²; x4_j stays register-resident
    sum_x  = 0f0
    sum_x2 = 0f0
    Base.@nexprs 16 j -> begin
        if j <= v4_iters
            off_j = (elem_base + (j - 1) * (block * 4)) * 4
            x4_j = ptx"ld.global.v4.f32"(px + off_j)
            sum_x  = ptx"add.f32"(sum_x, x4_j[1])
            sum_x  = ptx"add.f32"(sum_x, x4_j[2])
            sum_x  = ptx"add.f32"(sum_x, x4_j[3])
            sum_x  = ptx"add.f32"(sum_x, x4_j[4])
            sum_x2 = ptx"fma.rn.f32"(x4_j[1], x4_j[1], sum_x2)
            sum_x2 = ptx"fma.rn.f32"(x4_j[2], x4_j[2], sum_x2)
            sum_x2 = ptx"fma.rn.f32"(x4_j[3], x4_j[3], sum_x2)
            sum_x2 = ptx"fma.rn.f32"(x4_j[4], x4_j[4], sum_x2)
        end
    end

    sum_x  = _ln_warp_reduce_sum(sum_x)
    sum_x2 = _ln_warp_reduce_sum(sum_x2)

    if lane == UInt32(0)
        @inbounds partials[Int(warp_id) + 1, 1] = sum_x
        @inbounds partials[Int(warp_id) + 1, 2] = sum_x2
    end
    ptx"bar.sync"(Val(0))

    if tid == UInt32(0)
        bs  = 0f0
        bs2 = 0f0
        Base.@nexprs 16 i -> begin
            if i <= num_warps
                @inbounds bs  = ptx"add.f32"(bs,  partials[i, 1])
                @inbounds bs2 = ptx"add.f32"(bs2, partials[i, 2])
            end
        end
        @inbounds stats[1] = bs
        @inbounds stats[2] = bs2
    end
    ptx"bar.sync"(Val(0))

    @inbounds row_sum   = stats[1]
    @inbounds row_sum_2 = stats[2]
    inv_n   = Float32(1 / N)
    mean    = ptx"mul.f32"(row_sum, inv_n)
    ex2     = ptx"mul.f32"(row_sum_2, inv_n)
    mean_sq = ptx"mul.f32"(mean, mean)
    var     = ptx"add.f32"(ptx"sub.f32"(ex2, mean_sq), Float32(eps))
    rstd    = ptx"rsqrt.approx.f32"(var)

    # --- pass 2: y = (x - mean) * rstd * w + b on register-resident x4_j
    Base.@nexprs 16 j -> begin
        if j <= v4_iters
            w4 = ptx"ld.global.v4.f32"(pw + off_j)
            b4 = ptx"ld.global.v4.f32"(pb + off_j)
            d1 = ptx"sub.f32"(x4_j[1], mean)
            d2 = ptx"sub.f32"(x4_j[2], mean)
            d3 = ptx"sub.f32"(x4_j[3], mean)
            d4 = ptx"sub.f32"(x4_j[4], mean)
            n1 = ptx"mul.f32"(d1, rstd)
            n2 = ptx"mul.f32"(d2, rstd)
            n3 = ptx"mul.f32"(d3, rstd)
            n4 = ptx"mul.f32"(d4, rstd)
            y1 = ptx"fma.rn.f32"(n1, w4[1], b4[1])
            y2 = ptx"fma.rn.f32"(n2, w4[2], b4[2])
            y3 = ptx"fma.rn.f32"(n3, w4[3], b4[3])
            y4 = ptx"fma.rn.f32"(n4, w4[4], b4[4])
            ptx"st.global.v4.f32"(py + off_j, (y1, y2, y3, y4))
        end
    end
    return nothing
end

function layer_norm_ref(X::AbstractMatrix{Float32}, W::AbstractVector{Float32},
                        B::AbstractVector{Float32}, eps::Float32)
    out = similar(X)
    for b in axes(X, 1)
        x_row = Float64.(view(X, b, :))
        m     = sum(x_row) / length(x_row)
        v     = sum((x_row .- m) .^ 2) / length(x_row)
        rstd  = 1 / sqrt(v + eps)
        out[b, :] .= Float32.((x_row .- m) .* rstd .* Float64.(W) .+ Float64.(B))
    end
    out
end

@testset "layer_norm kernel matches CPU reference" begin
    eps = 1f-5
    for (B_, N, block) in [(4, 512, 128), (8, 1024, 256), (16, 2048, 128)]
        @assert N % (block * 4) == 0
        rng = MersenneTwister(B_ * 65537 + N)
        X = randn(rng, Float32, B_, N) .* 2f0 .- 1f0
        W = randn(rng, Float32, N) .* 0.1f0 .+ 1f0
        Bias = randn(rng, Float32, N) .* 0.1f0
        X_d = CuArray(vec(transpose(X)))
        W_d = CuArray(W)
        B_d = CuArray(Bias)
        Y_d = CUDACore.zeros(Float32, B_ * N)
        @cuda blocks = B_ threads = block _layer_norm_v4_kernel!(
            Y_d, X_d, W_d, B_d, Val(N), Val(block), Val(eps))
        CUDACore.synchronize()
        Y = transpose(reshape(Array(Y_d), N, B_))
        ref = layer_norm_ref(X, W, Bias, eps)
        @test isapprox(Y, ref; atol = 1f-4, rtol = 1f-3)
    end
end
