# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=7.0
using Random

# Ported from pyptx/examples/hopper/softmax.py
# (https://github.com/patrick-toulme/pyptx).
# Copyright 2026 Patrick Toulmé. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Translated to Julia and adapted.

# Fused row-wise softmax.
#
#   y[b, i] = exp(x[b, i] - row_max) / sum(exp(x[b, :] - row_max))
#
# Same outer structure as rms_norm/layer_norm: one CTA per row, two warp
# reductions (max, then sum) gated by SMEM partials + bar.sync. Pass 2
# folds exp(x - m) as ex2(fma(x, log2e, -m*log2e)) so the inner chain is
# one fma + one ex2.approx per element.
#
# Like pyptx, the row is register-resident across all three passes: pass 1
# keeps the loaded x chunks live, pass 2 turns them into exp values (also
# register-resident), pass 3 scales and stores — one DRAM read of X total.
# The unroll uses the `@nexprs 16` + compile-time-folded guard idiom (same
# as the warp-partial combines), so N ÷ (block*4) ≤ 16.
#
# v4 path only — N must be divisible by block*4. Despite the file living
# in pyptx/examples/hopper/, the kernel uses no Hopper-only ops (the
# floor is shfl.sync.bfly at sm_70); runs unchanged on Ada.

const SM_WARP_SIZE = UInt32(32)
const SM_LOG2E     = Float32(1.4426950408889634)

@inline function _sm_warp_reduce_max(v::Float32)
    full = UInt32(0xFFFFFFFF)
    seg  = UInt32(0x1F)
    Base.@nexprs 5 i -> begin
        offset = UInt32(1) << UInt32(5 - i)
        u      = reinterpret(UInt32, v)
        u_par  = ptx"shfl.sync.bfly.b32"(u, offset, seg, full)
        v      = ptx"max.f32"(v, reinterpret(Float32, u_par))
    end
    v
end

@inline function _sm_warp_reduce_sum(v::Float32)
    full = UInt32(0xFFFFFFFF)
    seg  = UInt32(0x1F)
    Base.@nexprs 5 i -> begin
        offset = UInt32(1) << UInt32(5 - i)
        u      = reinterpret(UInt32, v)
        u_par  = ptx"shfl.sync.bfly.b32"(u, offset, seg, full)
        v      = ptx"add.f32"(v, reinterpret(Float32, u_par))
    end
    v
end

function _softmax_v4_kernel!(
        Y::CuDeviceVector{Float32},
        X::CuDeviceVector{Float32},
        ::Val{N}, ::Val{block}) where {N, block}
    num_warps = block ÷ Int(SM_WARP_SIZE)
    v4_iters  = N ÷ (block * 4)
    @assert v4_iters * block * 4 == N
    @assert v4_iters <= 16    # register-tile unroll bound (64 f32 per thread)

    partials = CuStaticSharedArray(Float32, 16)   # max 16 warps (block ≤ 512)
    stats    = CuStaticSharedArray(Float32, 1)

    tid = ptx"mov.u32"(sreg"tid.x")
    row = ptx"mov.u32"(sreg"ctaid.x")
    row_byte_off = Int(row) * (N * 4)
    px = pointer(X) + row_byte_off
    py = pointer(Y) + row_byte_off
    lane    = tid & UInt32(0x1F)
    warp_id = tid >> UInt32(5)
    elem_base = Int(tid) * 4

    # --- pass 1: load row + per-thread max; x4_j stays register-resident
    row_max = -Inf32
    Base.@nexprs 16 j -> begin
        if j <= v4_iters
            off_j = (elem_base + (j - 1) * (block * 4)) * 4
            x4_j = ptx"ld.global.v4.f32"(px + off_j)
            row_max = ptx"max.f32"(row_max, x4_j[1])
            row_max = ptx"max.f32"(row_max, x4_j[2])
            row_max = ptx"max.f32"(row_max, x4_j[3])
            row_max = ptx"max.f32"(row_max, x4_j[4])
        end
    end

    # --- cross-warp combine: row_max -------------------------------------
    row_max = _sm_warp_reduce_max(row_max)
    if lane == UInt32(0)
        @inbounds partials[Int(warp_id) + 1] = row_max
    end
    ptx"bar.sync"(Val(0))
    if tid == UInt32(0)
        bm = -Inf32
        Base.@nexprs 16 i -> begin
            if i <= num_warps
                @inbounds bm = ptx"max.f32"(bm, partials[i])
            end
        end
        @inbounds stats[1] = bm
    end
    ptx"bar.sync"(Val(0))
    @inbounds row_max = stats[1]

    # Precompute -row_max * log2e once; pass 2/3 fold exp(x - row_max) as
    #   ex2(fma(x, log2e, neg_max_l2)) = ex2((x - row_max) * log2e)
    neg_max_l2 = ptx"mul.f32"(row_max, -SM_LOG2E)
    log2e      = SM_LOG2E

    # --- pass 2: exp(x - row_max) from register-resident x4_j; the e*_j
    # values stay register-resident in turn (x4_j dies here).
    row_sum = 0f0
    Base.@nexprs 16 j -> begin
        if j <= v4_iters
            e1_j = ptx"ex2.approx.f32"(ptx"fma.rn.f32"(x4_j[1], log2e, neg_max_l2))
            e2_j = ptx"ex2.approx.f32"(ptx"fma.rn.f32"(x4_j[2], log2e, neg_max_l2))
            e3_j = ptx"ex2.approx.f32"(ptx"fma.rn.f32"(x4_j[3], log2e, neg_max_l2))
            e4_j = ptx"ex2.approx.f32"(ptx"fma.rn.f32"(x4_j[4], log2e, neg_max_l2))
            row_sum = ptx"add.f32"(row_sum, e1_j)
            row_sum = ptx"add.f32"(row_sum, e2_j)
            row_sum = ptx"add.f32"(row_sum, e3_j)
            row_sum = ptx"add.f32"(row_sum, e4_j)
        end
    end

    # --- cross-warp combine: row_sum -------------------------------------
    row_sum = _sm_warp_reduce_sum(row_sum)
    if lane == UInt32(0)
        @inbounds partials[Int(warp_id) + 1] = row_sum
    end
    ptx"bar.sync"(Val(0))
    if tid == UInt32(0)
        bs = 0f0
        Base.@nexprs 16 i -> begin
            if i <= num_warps
                @inbounds bs = ptx"add.f32"(bs, partials[i])
            end
        end
        @inbounds stats[1] = bs
    end
    ptx"bar.sync"(Val(0))
    @inbounds row_sum = stats[1]
    inv_sum = ptx"rcp.approx.f32"(row_sum)

    # --- pass 3: scale the register-resident exp values, store -----------
    Base.@nexprs 16 j -> begin
        if j <= v4_iters
            y1 = ptx"mul.f32"(e1_j, inv_sum)
            y2 = ptx"mul.f32"(e2_j, inv_sum)
            y3 = ptx"mul.f32"(e3_j, inv_sum)
            y4 = ptx"mul.f32"(e4_j, inv_sum)
            ptx"st.global.v4.f32"(py + off_j, (y1, y2, y3, y4))
        end
    end
    return nothing
end

# CPU reference, Float64 internally for stability.
function softmax_ref(X::AbstractMatrix{Float32})
    out = similar(X)
    for b in axes(X, 1)
        x = Float64.(view(X, b, :))
        x .-= maximum(x)
        e  = exp.(x)
        out[b, :] .= Float32.(e ./ sum(e))
    end
    out
end

@testset "softmax kernel matches CPU reference" begin
    for (B, N, block) in [(4, 512, 128), (8, 1024, 256), (16, 2048, 128)]
        @assert N % (block * 4) == 0
        rng = MersenneTwister(B * 7919 + N)
        X = randn(rng, Float32, B, N)                  # B × N (column-major)
        X_d = CuArray(vec(transpose(X)))               # row-major flat
        Y_d = CUDACore.zeros(Float32, B * N)
        @cuda blocks = B threads = block _softmax_v4_kernel!(
            Y_d, X_d, Val(N), Val(block))
        CUDACore.synchronize()
        Y = transpose(reshape(Array(Y_d), N, B))
        @test isapprox(Y, softmax_ref(X); atol = 1f-4, rtol = 1f-3)
    end
end
