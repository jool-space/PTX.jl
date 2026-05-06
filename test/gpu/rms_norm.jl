using Random

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
# v0: v4-only path. N (= F) must be divisible by block*4.

const RMS_WARP_SIZE = UInt32(32)

# Butterfly warp reduction (sum). 5 shfl steps cover all 32 lanes; after
# the final step every lane holds the full-warp sum. UInt32 reinterpret
# bridges the b32-typed shfl with our Float32 accumulator.
@inline function _warp_reduce_sum(v::Float32)
    full = UInt32(0xFFFFFFFF)
    seg  = UInt32(0x1F)
    Base.@nexprs 5 i -> begin
        offset = UInt32(1) << UInt32(5 - i)        # 16, 8, 4, 2, 1
        u      = reinterpret(UInt32, v)
        u_par  = ptx"shfl.sync.bfly.b32"(u, offset, seg, full)
        v      = ptx"add.f32"(v, reinterpret(Float32, u_par))
    end
    v
end

function _rms_norm_v4_kernel!(
        Y::CuDeviceVector{Float32},
        X::CuDeviceVector{Float32},
        W::CuDeviceVector{Float32},
        ::Val{N}, ::Val{block}, ::Val{eps}) where {N, block, eps}
    num_warps = block ÷ Int(RMS_WARP_SIZE)
    v4_iters  = N ÷ (block * 4)
    @assert v4_iters * block * 4 == N

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

    # --- pass 1: load + sum-of-squares -----------------------------------
    # Stash the loaded values in a Vector so pass 2 doesn't reread DRAM.
    # Using a `MVector` would be cleanest but stack allocation needs to be
    # explicit; bare `NTuple` walked by `j` works since v4_iters is a Val.
    sum_sq = 0f0
    # Pre-allocate as a Ref{NTuple{4*v4_iters, Float32}}? Easier: just
    # iterate with @nexprs since v4_iters is a compile-time constant.
    # We unroll j so each x[1..4] becomes its own SSA value.
    x_buf = ntuple(_ -> 0f0, Val(4 * v4_iters))   # placeholder for layout

    # Manual unroll over j (v4_iters is small, typically 1..8).
    j = 0
    while j < v4_iters
        idx = elem_base + j * (block * 4)
        off = idx * 4
        x4 = ptx"ld.global.v4.f32"(px + off)
        # Stash in row-major flat storage. Need mutable; switch to MVector.
        # …but we can't mutate a tuple. Workaround: reload in pass 2.
        # OK for v0 (DRAM bandwidth-bound; rstd path is tiny), but a
        # follow-up should keep x in registers.
        sum_sq = ptx"fma.rn.f32"(x4[1], x4[1], sum_sq)
        sum_sq = ptx"fma.rn.f32"(x4[2], x4[2], sum_sq)
        sum_sq = ptx"fma.rn.f32"(x4[3], x4[3], sum_sq)
        sum_sq = ptx"fma.rn.f32"(x4[4], x4[4], sum_sq)
        j += 1
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

    # --- pass 2: re-load X (DRAM cache makes this cheap), apply, store ---
    j = 0
    while j < v4_iters
        idx = elem_base + j * (block * 4)
        off = idx * 4
        x4 = ptx"ld.global.v4.f32"(px + off)
        w4 = ptx"ld.global.v4.f32"(pw + off)
        y1 = ptx"mul.f32"(ptx"mul.f32"(x4[1], rstd), w4[1])
        y2 = ptx"mul.f32"(ptx"mul.f32"(x4[2], rstd), w4[2])
        y3 = ptx"mul.f32"(ptx"mul.f32"(x4[3], rstd), w4[3])
        y4 = ptx"mul.f32"(ptx"mul.f32"(x4[4], rstd), w4[4])
        ptx"st.global.v4.f32"(py + off, (y1, y2, y3, y4))
        j += 1
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
