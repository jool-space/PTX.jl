# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=7.0
using Random
using PTX.Warps: warp_reduce

# Fused RMS-norm port from pyptx/examples/hopper/rms_norm.py.
#
#   y[b, i] = x[b, i] * w[i] / sqrt(mean(x[b, :]^2) + eps)
#
# Structure (Triton's _layer_norm_fwd_fused tutorial pattern):
#   1. each thread loads N/block strided values from its row, accumulates
#      a per-thread sum-of-squares;
#   2. butterfly shfl reduces sum-of-squares within each warp;
#   3. lane 0 of each warp writes its partial to SMEM;
#   4. bar.sync;
#   5. thread 0 sums all warp partials, stashes in SMEM;
#   6. bar.sync;
#   7. every thread reads the row sum, computes rstd = rsqrt(mean + eps);
#   8. every thread reuses its loaded x_vals and emits y = x * rstd * w[i].
#
# v4-only path. N (= F) must be divisible by block*4, and N ÷ (block*4) ≤ 16
# (the register-tile unroll bound; 16 v4-chunks = 64 f32 per thread).
#
# The row is register-resident: pass 1's loaded chunks stay live as SSA
# values and pass 2 consumes them — one DRAM read of X total, matching the
# pyptx original. The unroll uses the same `@nexprs 16` + compile-time-folded
# guard idiom as the warp-partial combine below.

const RMS_WARP_SIZE = UInt32(32)

# ptx"add.f32" (not Julia +): the kernel's reduction op is the plain PTX
# add, and warp_reduce applies op verbatim between shuffle rounds.
@inline _warp_reduce_sum(v::Float32) =
    warp_reduce((a, b) -> ptx"add.f32"(a, b), v)

function _rms_norm_v4_kernel!(
        Y::CuDeviceVector{Float32},
        X::CuDeviceVector{Float32},
        W::CuDeviceVector{Float32},
        ::Val{N}, ::Val{block}, ::Val{eps}) where {N, block, eps}
    num_warps = block ÷ Int(RMS_WARP_SIZE)
    v4_iters  = N ÷ (block * 4)
    @assert v4_iters * block * 4 == N
    @assert v4_iters <= 16

    partials = CuStaticSharedArray(Float32, num_warps)
    stats    = CuStaticSharedArray(Float32, 1)

    tid = ptx"mov.u32"(sreg"tid.x")
    row = ptx"mov.u32"(sreg"ctaid.x")
    row_byte_off = Int(row) * (N * 4)
    px = pointer(X) + row_byte_off
    py = pointer(Y) + row_byte_off
    pw = pointer(W)
    lane    = tid & UInt32(0x1F)
    warp_id = tid >> UInt32(5)
    elem_base = Int(tid) * 4

    # --- pass 1: load + sum-of-squares; x4_j chunks stay register-resident
    sum_sq = 0f0
    Base.@nexprs 16 j -> begin
        if j <= v4_iters
            off_j = (elem_base + (j - 1) * (block * 4)) * 4
            x4_j = ptx"ld.global.v4.f32"(px + off_j)
            sum_sq = ptx"fma.rn.f32"(x4_j[1], x4_j[1], sum_sq)
            sum_sq = ptx"fma.rn.f32"(x4_j[2], x4_j[2], sum_sq)
            sum_sq = ptx"fma.rn.f32"(x4_j[3], x4_j[3], sum_sq)
            sum_sq = ptx"fma.rn.f32"(x4_j[4], x4_j[4], sum_sq)
        end
    end

    # --- warp reduce ------------------------------------------------------
    sum_sq = _warp_reduce_sum(sum_sq)

    if lane == UInt32(0)
        @inbounds partials[Int(warp_id) + 1] = sum_sq
    end
    ptx"bar.sync"(Val(0))

    if tid == UInt32(0)
        block_sum = 0f0
        Base.@nexprs 16 i -> begin            # up to 16 warps (block ≤ 512)
            if i <= num_warps
                @inbounds block_sum = ptx"add.f32"(block_sum, partials[i])
            end
        end
        @inbounds stats[1] = block_sum
    end
    ptx"bar.sync"(Val(0))

    # --- compute rstd -----------------------------------------------------
    @inbounds row_sum = stats[1]
    inv_n   = Float32(1 / N)
    mean_sq = ptx"add.f32"(ptx"mul.f32"(row_sum, inv_n), Float32(eps))
    rstd    = ptx"rsqrt.approx.f32"(mean_sq)

    # --- pass 2: apply to the register-resident x4_j, store --------------
    Base.@nexprs 16 j -> begin
        if j <= v4_iters
            w4 = ptx"ld.global.v4.f32"(pw + off_j)
            y1 = ptx"mul.f32"(ptx"mul.f32"(x4_j[1], rstd), w4[1])
            y2 = ptx"mul.f32"(ptx"mul.f32"(x4_j[2], rstd), w4[2])
            y3 = ptx"mul.f32"(ptx"mul.f32"(x4_j[3], rstd), w4[3])
            y4 = ptx"mul.f32"(ptx"mul.f32"(x4_j[4], rstd), w4[4])
            ptx"st.global.v4.f32"(py + off_j, (y1, y2, y3, y4))
        end
    end
    return nothing
end

# CPU reference, Float64 internally for stability.
function rms_norm_ref(X::AbstractMatrix{Float32}, W::AbstractVector{Float32},
                      eps::Float32)
    out = similar(X)
    for b in axes(X, 1)
        x_row = Float64.(view(X, b, :))
        ms    = sum(abs2, x_row) / length(x_row)
        rstd  = 1 / sqrt(ms + eps)
        out[b, :] .= Float32.(x_row .* rstd .* Float64.(W))
    end
    out
end

@testset "rms_norm kernel matches CPU reference" begin
    eps = 1f-6
    for (B, N, block) in [(4, 512, 128), (8, 1024, 256), (16, 2048, 128)]
        @assert N % (block * 4) == 0
        rng = MersenneTwister(B * 7919 + N)
        X = randn(rng, Float32, B, N) .* 0.3f0           # B × N (column-major)
        W = randn(rng, Float32, N) .* 0.1f0 .+ 1f0
        # Flatten row-major: kernel expects element (b, i) at offset b*N + i.
        X_flat = vec(transpose(X))                       # B*N flat, row-major
        X_d = CuArray(X_flat)
        W_d = CuArray(W)
        Y_d = CUDACore.zeros(Float32, B * N)
        @cuda blocks = B threads = block _rms_norm_v4_kernel!(
            Y_d, X_d, W_d, Val(N), Val(block), Val(eps))
        CUDACore.synchronize()
        Y = transpose(reshape(Array(Y_d), N, B))         # B × N
        ref = rms_norm_ref(X, W, eps)
        @test isapprox(Y, ref; atol = 1f-4, rtol = 1f-3)
    end
end
