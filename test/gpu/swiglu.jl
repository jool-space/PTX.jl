# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=7.0
using Random

# Ported from pyptx/examples/hopper/swiglu.py
# (https://github.com/patrick-toulme/pyptx).
# Copyright 2026 Patrick Toulmé. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Translated to Julia and adapted.

# Fused SwiGLU activation.
#
#   out[i, j] = silu(gate[i, j]) * up[i, j]
#   silu(x)   = x * sigmoid(x)
#             = x / (1 + exp(-x))
#
# PTX has no `exp.f32`; we use the identity
#   exp(-x) = exp2(-x * log2(e))
# and the hardware `ex2.approx.f32` + `rcp.approx.f32` instructions.
# Both are ~3 ULP — within the 1e-4 abs / 1e-3 rel tolerance.
#
# Vectorized via `ld.global.v4.f32` / `st.global.v4.f32` so each memory
# transaction moves 16 bytes — needed to saturate HBM bandwidth on a
# bandwidth-bound kernel like this one.
#
# Layout: gate / up / out are flat (M*F)-length vectors interpreted as
# M rows × F columns, row-major. CTA m processes row m. Each thread
# loops over `v4_iters = F / (block*4)` strided v4 loads.

const SWIGLU_LOG2E = 1.4426950408889634f0

@inline function _silu_mul_lane(g::Float32, u::Float32)
    # silu(g) * u = g / (1 + exp(-g)) * u
    neg_g_log2 = ptx"mul.f32"(g, -SWIGLU_LOG2E)
    exp_neg    = ptx"ex2.approx.f32"(neg_g_log2)
    denom      = ptx"add.f32"(1f0, exp_neg)
    sigm       = ptx"rcp.approx.f32"(denom)
    silu_g     = ptx"mul.f32"(g, sigm)
    ptx"mul.f32"(silu_g, u)
end

function _swiglu_v4_kernel!(
        out::CuDeviceVector{Float32},
        gate::CuDeviceVector{Float32},
        up::CuDeviceVector{Float32},
        ::Val{F}, ::Val{block}) where {F, block}
    v4_iters     = F ÷ (block * 4)
    row_base     = ptx"mov.u32"(sreg"ctaid.x")
    tid          = ptx"mov.u32"(sreg"tid.x")
    row_byte_off = Int(row_base) * (F * 4)
    pg = pointer(gate) + row_byte_off
    pu = pointer(up)   + row_byte_off
    po = pointer(out)  + row_byte_off
    elem_base = Int(tid) * 4              # elements (f32 units), not bytes

    j = 0
    while j < v4_iters
        idx = elem_base + j * (block * 4)
        off = idx * 4                      # byte offset
        g = ptx"ld.global.v4.f32"(pg + off)
        u = ptx"ld.global.v4.f32"(pu + off)
        o = (_silu_mul_lane(g[1], u[1]),
             _silu_mul_lane(g[2], u[2]),
             _silu_mul_lane(g[3], u[3]),
             _silu_mul_lane(g[4], u[4]))
        ptx"st.global.v4.f32"(po + off, o)
        j += 1
    end
    return nothing
end

# CPU reference using Float64 to keep the reference clean.
function swiglu_ref(gate::AbstractVector{Float32}, up::AbstractVector{Float32})
    g64 = Float64.(gate)
    u64 = Float64.(up)
    Float32.(g64 .* (1.0 ./ (1.0 .+ exp.(-g64))) .* u64)
end

@testset "swiglu kernel matches CPU reference" begin
    for (M, F, block) in [(4, 512, 128), (32, 1024, 256), (8, 2048, 128)]
        @assert F % (block * 4) == 0
        rng = MersenneTwister(M * 65537 + F)
        gate = (randn(rng, Float32, M * F)) .* 0.5f0
        up   = (randn(rng, Float32, M * F)) .* 0.5f0

        gate_d = CuArray(gate)
        up_d   = CuArray(up)
        out_d  = CUDACore.zeros(Float32, M * F)

        @cuda blocks = M threads = block _swiglu_v4_kernel!(
            out_d, gate_d, up_d, Val(F), Val(block))
        CUDACore.synchronize()

        out = Array(out_d)
        ref = swiglu_ref(gate, up)

        diff = maximum(abs.(out .- ref))
        @test diff < 1f-4
        @test isapprox(out, ref; atol = 1f-4, rtol = 1f-3)
    end
end
