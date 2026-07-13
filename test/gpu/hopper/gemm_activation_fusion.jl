# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc==9.0
# Hopper GEMM with fused activation epilogue — ported from
# cutlass/examples/113_hopper_gemm_activation_fusion (NVIDIA CUTLASS,
# BSD-3-Clause), the `113_hopper_gemm_fused_act.cu` variant.
#
# The CUTLASS example builds an `AccCastLinCombEltActScale` EVT epilogue:
#
#   D = scale · act(alpha · acc + beta · C)
#
# with SiLU as the headline activation (`#elif 1` selects
# `epilogue::thread::SiLu`; ReLU / Identity are the compile-time
# alternates) and a per-tensor quantization scale (`Quantize = true`).
# What's ported: the full epilogue dataflow — accumulator, C source loaded
# from global at epilogue time, alpha/beta linear combination, SiLU, scale,
# store — fused after the standard bf16 TMA → mbarrier → wgmma m64n8k16
# brick (gemm_warpgroup.jl).
#
#   silu(x) = x · sigmoid(x) = x / (1 + exp(-x))
#
# computed as ex2.approx(−x·log2e) + rcp.approx — the same ~3 ULP chain as
# test/gpu/swiglu.jl (which covers this example's *gated* variant's math,
# `113_hopper_gemm_fused_gated_act.cu`, as a standalone kernel).
#
# Deviations: A/B are bf16 (example: fp16 — same wgmma family), C and D are
# f32 (example: half C, quantized-half D — the cast nodes are dtype
# plumbing, not dataflow), the scale is a kernel scalar (example reads it
# from a device pointer), and the grouped-GEMM variants are not ported
# (grouped scheduling is covered by grouped_gemm.jl).

using PTX: wgmma_descriptor, smem_addr_u32, layout_for_a
using CUDACore
using Random

const ACT_BM = 64
const ACT_BN = 8
const ACT_BK = 16
const ACT_THREADS = 128                 # one warpgroup
const ACT_LOAD_BYTES = ACT_BM * ACT_BK * 2 + ACT_BK * ACT_BN * 2
const ACT_LOG2E = 1.4426950408889634f0

@inline function _act_silu(x::Float32)
    exp_neg = ptx"ex2.approx.f32"(x * -ACT_LOG2E)
    x * ptx"rcp.approx.f32"(1f0 + exp_neg)
end

function _act_gemm_kernel!(
        D::CuDeviceVector{Float32, 1},
        C::CuDeviceVector{Float32, 1},
        tma_A::PTX.TMADescriptorPtr,
        tma_B::PTX.TMADescriptorPtr,
        alpha::Float32,
        beta::Float32,
        scale::Float32)

    smem_A = CuStaticSharedArray(UInt16, ACT_BM * ACT_BK)
    smem_B = CuStaticSharedArray(UInt16, ACT_BK * ACT_BN)
    mbar   = CuStaticSharedArray(UInt64, 1)

    a_ptr  = pointer(smem_A)
    b_ptr  = pointer(smem_B)
    mb_ptr = pointer(mbar)
    a_addr = smem_addr_u32(a_ptr)
    b_addr = smem_addr_u32(b_ptr)

    tid = ptx"mov.u32"(sreg"tid.x")

    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mb_ptr, UInt32(1))
        ptx"fence.proxy.async.shared::cta"()
        ptx"mbarrier.arrive.expect_tx.shared.b64"(mb_ptr, UInt32(ACT_LOAD_BYTES))
        ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
            a_ptr, tma_A, Int32(0), Int32(0), mb_ptr)
        ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
            b_ptr, tma_B, Int32(0), Int32(0), mb_ptr)
    end
    ptx"bar.sync"(Val(0))

    while !ptx"mbarrier.test_wait.parity.shared.b64"(mb_ptr, UInt32(0))
    end

    ptx"fence.proxy.async.shared::cta"()
    ptx"wgmma.fence.sync.aligned"()

    la = layout_for_a(dtype = :bf16, m = ACT_BM, k = ACT_BK)
    lb = layout_for_a(dtype = :bf16, m = ACT_BN, k = ACT_BK)
    a_desc = wgmma_descriptor(a_addr;
        leading_byte_offset = la.leading_byte_offset,
        stride_byte_offset  = la.stride_byte_offset,
        swizzle             = la.layout_type)
    b_desc = wgmma_descriptor(b_addr;
        leading_byte_offset = lb.leading_byte_offset,
        stride_byte_offset  = lb.stride_byte_offset,
        swizzle             = lb.layout_type)

    d = ntuple(_ -> 0f0, Val(4))
    d = ptx"wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16"(
        d, a_desc, b_desc, false)

    ptx"wgmma.commit_group.sync.aligned"()
    ptx"wgmma.wait_group.sync.aligned"(Val(0))

    # ── Fused epilogue: D = scale · silu(alpha·acc + beta·C) ───────────
    # C is loaded per-lane at the same (row, col) offsets the store uses —
    # the EVT's C source, materialized at epilogue time, not the mainloop.
    wid      = tid >> UInt32(5)
    lane     = tid & UInt32(31)
    frag_row = (wid << UInt32(4)) + (lane >> UInt32(2))
    frag_col = (lane & UInt32(3)) << UInt32(1)
    off_a    = Int((frag_row * UInt32(ACT_BN) + frag_col) * UInt32(4))
    off_b    = Int(((frag_row + UInt32(8)) * UInt32(ACT_BN) + frag_col) * UInt32(4))

    c12 = ptx"ld.global.v2.f32"(pointer(C) + off_a)
    c34 = ptx"ld.global.v2.f32"(pointer(C) + off_b)

    y1 = scale * _act_silu(alpha * d[1] + beta * c12[1])
    y2 = scale * _act_silu(alpha * d[2] + beta * c12[2])
    y3 = scale * _act_silu(alpha * d[3] + beta * c34[1])
    y4 = scale * _act_silu(alpha * d[4] + beta * c34[2])

    pd = pointer(D)
    ptx"st.global.v2.f32"(pd + off_a, (y1, y2))
    ptx"st.global.v2.f32"(pd + off_b, (y3, y4))
    return nothing
end

# ── Cross-arch ptxas validation ────────────────────────────────────────

@testset "GEMM + SiLU activation fusion compiles at sm_90a" begin
    types = Tuple{CuDeviceVector{Float32, 1}, CuDeviceVector{Float32, 1},
                  PTX.TMADescriptorPtr, PTX.TMADescriptorPtr,
                  Float32, Float32, Float32}
    @test ptxas_compiles(_act_gemm_kernel!, types;
                         cap = v"9.0", feature_set = :arch)

    ptx = emit_ptx(_act_gemm_kernel!, types; cap = v"9.0", feature_set = :arch)
    @test occursin("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16", ptx)
    @test occursin("ex2.approx.f32", ptx)
    @test occursin("rcp.approx.f32", ptx)
    # tier-1 vec ld/st canonicalizes f32 vectors to the `.b32` bit spelling
    @test occursin(r"ld\.global\.v2\.(f|b)32", ptx)
    @test occursin(r"st\.global\.v2\.(f|b)32", ptx)
end

# ── Runtime — Hopper hardware ──────────────────────────────────────────

_act_silu_ref(x::Float32) = x / (1f0 + exp(-x))

if test_runtime_supported(@__FILE__)
    @testset "GEMM + SiLU fusion (random bf16, alpha/beta/scale)" begin
        rng = MersenneTwister(0xac71)
        A_f32 = Float32.(randn(rng, ACT_BM, ACT_BK)) .* 0.5f0
        B_f32 = Float32.(randn(rng, ACT_BK, ACT_BN)) .* 0.5f0
        C_f32 = Float32.(randn(rng, ACT_BM, ACT_BN)) .* 0.5f0
        alpha, beta = 1.25f0, 0.75f0
        scale = 0.625f0                  # example draws it from [0.5, 1.0)

        A_packed = Array{UInt16}(undef, ACT_BK, ACT_BM)
        B_packed = Array{UInt16}(undef, ACT_BK, ACT_BN)
        for m in 1:ACT_BM, k in 1:ACT_BK
            A_packed[k, m] = bf16_bits(A_f32[m, k])
        end
        for n in 1:ACT_BN, k in 1:ACT_BK
            B_packed[k, n] = bf16_bits(B_f32[k, n])
        end
        A_d = CuArray(A_packed)
        B_d = CuArray(B_packed)

        # C in the kernel's store layout: (BN, BM) col-major = N-innermost.
        C_packed = Array{Float32}(undef, ACT_BN, ACT_BM)
        for m in 1:ACT_BM, n in 1:ACT_BN
            C_packed[n, m] = C_f32[m, n]
        end
        C_d = CuArray(vec(C_packed))

        tmap_A = PTX.tensor_map_tile_2d(:bf16, pointer(A_d),
            ACT_BM, ACT_BK, ACT_BM, ACT_BK; swizzle = :B32)
        tmap_B = PTX.tensor_map_tile_2d(:bf16, pointer(B_d),
            ACT_BN, ACT_BK, ACT_BN, ACT_BK; swizzle = :B32)
        A = upload_tma_descriptor(tmap_A)
        B = upload_tma_descriptor(tmap_B)

        D = CUDACore.zeros(Float32, ACT_BM * ACT_BN)
        @cuda threads = ACT_THREADS _act_gemm_kernel!(
            D, C_d, A.ptr, B.ptr, alpha, beta, scale)
        CUDACore.synchronize()

        D_packed = reshape(Array(D), ACT_BN, ACT_BM)
        D_got = Array{Float32}(undef, ACT_BM, ACT_BN)
        for m in 1:ACT_BM, n in 1:ACT_BN
            D_got[m, n] = D_packed[n, m]
        end

        acc_ref = bf16_gemm_ref(A_f32, B_f32)
        D_ref = scale .* _act_silu_ref.(alpha .* acc_ref .+ beta .* C_f32)
        @test isapprox(D_got, D_ref; rtol = 1e-3, atol = 1e-3)
    end

    @testset "GEMM + SiLU fusion (beta = 0 ignores C)" begin
        # beta = 0 must make the C source inert — mirrors the example's
        # `beta != 0 ? block_C.get() : nullptr` verification split.
        rng = MersenneTwister(0xac72)
        A_f32 = Float32.(randn(rng, ACT_BM, ACT_BK)) .* 0.5f0
        B_f32 = Float32.(randn(rng, ACT_BK, ACT_BN)) .* 0.5f0
        alpha, scale = 2f0, 1f0

        A_packed = Array{UInt16}(undef, ACT_BK, ACT_BM)
        B_packed = Array{UInt16}(undef, ACT_BK, ACT_BN)
        for m in 1:ACT_BM, k in 1:ACT_BK
            A_packed[k, m] = bf16_bits(A_f32[m, k])
        end
        for n in 1:ACT_BN, k in 1:ACT_BK
            B_packed[k, n] = bf16_bits(B_f32[k, n])
        end
        A_d = CuArray(A_packed)
        B_d = CuArray(B_packed)
        C_d = CuArray(fill(1f9, ACT_BM * ACT_BN))   # poison: must not leak

        tmap_A = PTX.tensor_map_tile_2d(:bf16, pointer(A_d),
            ACT_BM, ACT_BK, ACT_BM, ACT_BK; swizzle = :B32)
        tmap_B = PTX.tensor_map_tile_2d(:bf16, pointer(B_d),
            ACT_BN, ACT_BK, ACT_BN, ACT_BK; swizzle = :B32)
        A = upload_tma_descriptor(tmap_A)
        B = upload_tma_descriptor(tmap_B)

        D = CUDACore.zeros(Float32, ACT_BM * ACT_BN)
        @cuda threads = ACT_THREADS _act_gemm_kernel!(
            D, C_d, A.ptr, B.ptr, alpha, 0f0, scale)
        CUDACore.synchronize()

        D_packed = reshape(Array(D), ACT_BN, ACT_BM)
        D_got = Array{Float32}(undef, ACT_BM, ACT_BN)
        for m in 1:ACT_BM, n in 1:ACT_BN
            D_got[m, n] = D_packed[n, m]
        end

        acc_ref = bf16_gemm_ref(A_f32, B_f32)
        D_ref = _act_silu_ref.(alpha .* acc_ref)
        @test isapprox(D_got, D_ref; rtol = 1e-3, atol = 1e-3)
    end
end
