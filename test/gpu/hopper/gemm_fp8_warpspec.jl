# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc==9.0
# FP8 warp-specialized Hopper GEMM — ported from
# cutlass/examples/54_hopper_fp8_warp_specialized_gemm (NVIDIA CUTLASS,
# BSD-3-Clause).
#
# The CUTLASS example instantiates a warp-specialized e4m3×e4m3→f32 GEMM
# through the CUTLASS 3.0 collective builders and showcases the FP8-GEMM
# fusion set: per-tensor scale factors for A/B/C/D and the abs-max value
# of D (the "delayed scaling" statistic FP8 training recipes need).
# What's ported:
#
#   - e4m3 A and B tiles, TMA-loaded (tensor-map dtype `:u8` — the driver
#     has no fp8 tensor-map code; a byte is a byte, swizzle math is
#     identical), f32 accumulation via
#     wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3.
#   - Warp-specialized mainloop: WG0 producer (TMA + mbarrier ring),
#     WG1 consumer (wgmma) — the same 2-stage pipeline as
#     gemm_pc_pipeline.jl, retiled for the 1-byte dtype (BK=32, so the
#     SMEM row stays 32 B → B32 swizzle, same family as bf16 BK=16).
#   - Fused epilogue: D = (scale_a·scale_b)·acc, plus abs-max(D) reduced
#     device-side — per-lane in-register max over the 4 f32 frags, then
#     atom.global.max.u32 on the f32 bit pattern (|x| ≥ 0, so u32 order
#     equals f32 order — CUTLASS's amax epilogue plays the same trick).
#
# Not ported: CTA rasterization/swizzle tuning (needs multi-tile grids —
# scheduling is gemm_highperf_hopper.jl territory), the C operand
# (beta = 0 here, so scale_c/scale_d degenerate into scale_a·scale_b).
#
# e4m3 quantization host-side goes through Microfloats' Float8_E4M3FN —
# PTX `.e4m3` is OCP E4M3FN (no Inf, NaN at 0x7F/0xFF), exactly that type
# (same pairing as test/gpu/ada/gemm_fp8.jl and cvt_fp8.jl).

using PTX: layout_for_a, wgmma_descriptor, smem_addr_u32, tensor_map_tile_2d,
           step_desc
using PTX.MBarriers: BarrierArray,
                     barrier_wait, barrier_arrive, barrier_arrive_expect_tx
using PTX.Pipelines: Pipeline, pipeline_init!, pipeline_cursor
using CUDACore
using Microfloats
using Random

const FP8WS_BM       = 64
const FP8WS_BN       = 8
const FP8WS_BK       = 32                       # e4m3 wgmma K-step
const FP8WS_N_STAGES = 2
const FP8WS_THREADS  = 256                      # 2 warpgroups

# Per-stage SMEM bytes (1-byte dtype).
const FP8WS_A_STAGE_BYTES = FP8WS_BM * FP8WS_BK # 2048
const FP8WS_B_STAGE_BYTES = FP8WS_BK * FP8WS_BN # 256
const FP8WS_LOAD_BYTES    = FP8WS_A_STAGE_BYTES + FP8WS_B_STAGE_BYTES

function _fp8_ws_gemm_kernel!(
        D::CuDeviceVector{Float32, 1},
        amax::CuDeviceVector{UInt32, 1},
        tma_A::PTX.TMADescriptorPtr,
        tma_B::PTX.TMADescriptorPtr,
        K::Int32,
        scale_ab::Float32)

    smem_A = CuStaticSharedArray(UInt8, FP8WS_N_STAGES * FP8WS_BM * FP8WS_BK)
    smem_B = CuStaticSharedArray(UInt8, FP8WS_N_STAGES * FP8WS_BK * FP8WS_BN)
    mbar_full  = CuStaticSharedArray(UInt64, FP8WS_N_STAGES)
    mbar_empty = CuStaticSharedArray(UInt64, FP8WS_N_STAGES)

    a_base_ptr  = pointer(smem_A)
    b_base_ptr  = pointer(smem_B)
    a_base_addr = smem_addr_u32(a_base_ptr)
    b_base_addr = smem_addr_u32(b_base_ptr)

    full  = BarrierArray{FP8WS_N_STAGES}(pointer(mbar_full))
    empty = BarrierArray{FP8WS_N_STAGES}(pointer(mbar_empty))

    tid     = ptx"mov.u32"(sreg"tid.x")
    wg_id   = tid >> UInt32(7)                  # 0 = producer, 1 = consumer
    lane128 = tid & UInt32(127)

    if tid == 0
        pipeline_init!(full, empty, Val(1), Val(false))
    end
    ptx"bar.sync"(Val(0))

    num_k_tiles = K >> Int32(5)                 # K / 32 = K / BK

    if wg_id == UInt32(0)
        # ── PRODUCER (WG 0): TMA loads into the stage ring ─────────────
        ptx"setmaxnreg.dec.sync.aligned.u32"(Val(24))

        @inbounds for k_iter in Int32(0):(num_k_tiles - Int32(1))
            stage, phase = pipeline_cursor(Pipeline{FP8WS_N_STAGES}, k_iter)
            barrier_wait(empty[stage], phase)

            if lane128 == UInt32(0)
                ptx"fence.proxy.async.shared::cta"()
                barrier_arrive_expect_tx(full[stage], FP8WS_LOAD_BYTES)
                a_stage_ptr = a_base_ptr + Int(stage) * FP8WS_A_STAGE_BYTES
                b_stage_ptr = b_base_ptr + Int(stage) * FP8WS_B_STAGE_BYTES
                # K-fast layout → TMA coords (k_off, m_or_n_off). The tensor
                # map counts elements = bytes for :u8.
                k_off = k_iter * Int32(FP8WS_BK)
                ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                    a_stage_ptr, tma_A, k_off, Int32(0), full[stage])
                ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                    b_stage_ptr, tma_B, k_off, Int32(0), full[stage])
            end
        end
    else
        # ── CONSUMER (WG 1): wgmma per stage, fused scaling epilogue ───
        ptx"setmaxnreg.inc.sync.aligned.u32"(Val(240))

        # K-major canonical layout for both operands (both tiles K-fast in
        # SMEM; fp8 wgmma has no trans bits, so K-fast B is the only option).
        # Row width = 32 e4m3 = 32 B → B32, same family as bf16 BK=16.
        la = layout_for_a(dtype = :e4m3, m = FP8WS_BM, k = FP8WS_BK)
        lb = layout_for_a(dtype = :e4m3, m = FP8WS_BN, k = FP8WS_BK)
        a_desc_base = wgmma_descriptor(a_base_addr;
            leading_byte_offset = la.leading_byte_offset,
            stride_byte_offset  = la.stride_byte_offset,
            swizzle             = la.layout_type)
        b_desc_base = wgmma_descriptor(b_base_addr;
            leading_byte_offset = lb.leading_byte_offset,
            stride_byte_offset  = lb.stride_byte_offset,
            swizzle             = lb.layout_type)

        d = ntuple(_ -> 0f0, Val(4))

        @inbounds for k_iter in Int32(0):(num_k_tiles - Int32(1))
            stage, phase = pipeline_cursor(Pipeline{FP8WS_N_STAGES}, k_iter)
            barrier_wait(full[stage], phase)

            a_desc = step_desc(a_desc_base, Int(stage) * FP8WS_A_STAGE_BYTES)
            b_desc = step_desc(b_desc_base, Int(stage) * FP8WS_B_STAGE_BYTES)
            ptx"fence.proxy.async.shared::cta"()
            ptx"wgmma.fence.sync.aligned"()
            d = ptx"wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3"(
                d, a_desc, b_desc, true)
            ptx"wgmma.commit_group.sync.aligned"()
            ptx"wgmma.wait_group.sync.aligned"(Val(0))

            if lane128 == UInt32(0)
                barrier_arrive(empty[stage])
            end
        end

        # ── Epilogue: dequant scale + store + fused abs-max ────────────
        # Frag layout is the m64nNk* f32 standard (same as gemm_pc_pipeline).
        s1 = d[1] * scale_ab
        s2 = d[2] * scale_ab
        s3 = d[3] * scale_ab
        s4 = d[4] * scale_ab

        wid      = lane128 >> UInt32(5)
        lane     = lane128 & UInt32(31)
        frag_row = (wid << UInt32(4)) + (lane >> UInt32(2))
        frag_col = (lane & UInt32(3)) << UInt32(1)
        pd       = pointer(D)
        off_a    = (frag_row * UInt32(FP8WS_BN) + frag_col) * UInt32(4)
        off_b    = ((frag_row + UInt32(8)) * UInt32(FP8WS_BN) + frag_col) * UInt32(4)
        ptx"st.global.v2.f32"(pd + Int(off_a), (s1, s2))
        ptx"st.global.v2.f32"(pd + Int(off_b), (s3, s4))

        # Per-lane |·| max over the 4 frags, then a u32 max-atomic on the
        # bit pattern. abs(f32) ≥ 0 → IEEE ordering == unsigned ordering.
        local_amax = max(max(abs(s1), abs(s2)), max(abs(s3), abs(s4)))
        ptx"atom.global.max.u32"(pointer(amax), reinterpret(UInt32, local_amax))
    end

    return nothing
end

# ── Cross-arch ptxas validation ────────────────────────────────────────

@testset "FP8 warp-specialized Hopper GEMM compiles at sm_90a" begin
    types = Tuple{CuDeviceVector{Float32, 1}, CuDeviceVector{UInt32, 1},
                  PTX.TMADescriptorPtr, PTX.TMADescriptorPtr, Int32, Float32}
    @test ptxas_compiles(_fp8_ws_gemm_kernel!, types;
                         cap = v"9.0", feature_set = :arch)

    ptx = emit_ptx(_fp8_ws_gemm_kernel!, types; cap = v"9.0", feature_set = :arch)
    @test occursin("wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3", ptx)
    @test occursin("setmaxnreg.dec.sync.aligned.u32",                     ptx)
    @test occursin("mbarrier.arrive.expect_tx.shared.b64",                ptx)
    @test occursin("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes", ptx)
    @test occursin("atom.global.max.u32",                                 ptx)
end

# ── Runtime — Hopper hardware ──────────────────────────────────────────

if test_runtime_supported(@__FILE__)
    @testset "FP8 warp-specialized GEMM (random e4m3, K=128, fused amax)" begin
        rng = MersenneTwister(0xf8f8)
        K_test = 128                            # 4 K-iters, 2 ring wraps
        A_f32 = randn(rng, Float32, FP8WS_BM, K_test) .* 0.25f0
        B_f32 = randn(rng, Float32, K_test, FP8WS_BN) .* 0.25f0

        # Quantize to e4m3 and keep BOTH the bits (device input) and the
        # dequantized f32 (host reference).
        A_q = Float8_E4M3FN.(A_f32)
        B_q = Float8_E4M3FN.(B_f32)
        A_deq = Float32.(A_q)
        B_deq = Float32.(B_q)

        # Pack K-fast: A (M, K) → (K, M) col-major; B (K, N) stays (K, N).
        A_packed = Array{UInt8}(undef, K_test, FP8WS_BM)
        B_packed = Array{UInt8}(undef, K_test, FP8WS_BN)
        for m in 1:FP8WS_BM, k in 1:K_test
            A_packed[k, m] = reinterpret(UInt8, A_q[m, k])
        end
        for k in 1:K_test, n in 1:FP8WS_BN
            B_packed[k, n] = reinterpret(UInt8, B_q[k, n])
        end
        A_d = CuArray(A_packed)
        B_d = CuArray(B_packed)

        # :u8 tensor maps (no fp8 dtype code in the driver enum). 32-byte
        # box rows → B32 swizzle, matching layout_for_a's pick in-kernel.
        tmap_A = tensor_map_tile_2d(:u8, pointer(A_d),
            FP8WS_BM, K_test, FP8WS_BM, FP8WS_BK; swizzle = :B32)
        tmap_B = tensor_map_tile_2d(:u8, pointer(B_d),
            FP8WS_BN, K_test, FP8WS_BN, FP8WS_BK; swizzle = :B32)
        A = upload_tma_descriptor(tmap_A)
        B = upload_tma_descriptor(tmap_B)

        # Per-tensor dequant scales, folded host-side like CUTLASS's
        # alpha = scale_a * scale_b.
        scale_ab = 0.5f0 * 1.5f0

        D_dev    = CUDACore.zeros(Float32, FP8WS_BM * FP8WS_BN)
        amax_dev = CUDACore.zeros(UInt32, 1)
        @cuda threads = FP8WS_THREADS _fp8_ws_gemm_kernel!(
            D_dev, amax_dev, A.ptr, B.ptr, Int32(K_test), scale_ab)
        CUDACore.synchronize()

        D_packed = reshape(Array(D_dev), FP8WS_BN, FP8WS_BM)
        D_got    = Array{Float32}(undef, FP8WS_BM, FP8WS_BN)
        for m in 1:FP8WS_BM, n in 1:FP8WS_BN
            D_got[m, n] = D_packed[n, m]
        end

        # Reference: f32 gemm over the dequantized e4m3 values. Individual
        # e4m3 products are exact in f32, but PTX ISA §9.7.16.5 only
        # guarantees fp8 wgmma accumulation at "higher than half but lower
        # than single precision" — the 1e-3 tolerance covers both the
        # reduced-precision accumulate and summation-order differences.
        D_ref = scale_ab .* (A_deq * B_deq)
        @test isapprox(D_got, D_ref; rtol = 1e-3, atol = 1e-3)

        # Fused amax must equal the exact max |D| of the values the device
        # itself produced (same f32s, host just re-reduces them).
        amax_got = reinterpret(Float32, Array(amax_dev)[1])
        @test amax_got == maximum(abs.(D_got))
    end
end
