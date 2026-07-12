# GEMM with L2 weight prefetch — ported from
# cutlass/examples/63_hopper_gemm_with_weight_prefetch (NVIDIA CUTLASS,
# BSD-3-Clause).
#
# The CUTLASS example targets low-batch LLM inference: while the mainloop
# waits on activations, a dedicated warp in the producer warpgroup walks
# the weight tensor's K-tiles and issues
# `cp.async.bulk.prefetch.tensor.2d.L2.global.tile` — no SMEM destination,
# no mbarrier, pure L2 warming — so the real TMA loads that follow hit L2
# instead of DRAM. `prefetch_ratio` picks what fraction of the K-tiles to
# prefetch. What's ported:
#
#   - The producer/consumer pipeline of gemm_pc_pipeline.jl, unchanged.
#   - A prefetch warp: warp 1 of the producer warpgroup (the TMA-issuing
#     warp is warp 0) walks `prefetch_k_tiles` K-tiles of the weight
#     operand A through the new L2-prefetch wrapper, concurrently with
#     warp 0's regular TMA issue loop. Fire-and-forget: no completion
#     mechanism exists, and none is needed — correctness must be
#     identical whether the prefetch covers 0%, 50% or 100% of K.
#
# Not ported: programmatic dependent launch (PDL) — CUTLASS overlaps the
# prefetch with the *previous* kernel's epilogue via griddepcontrol;
# that's a launch-config feature outside a single-kernel corpus test.
# `prefetch_ratio` is folded host-side into a tile count.
#
# The runtime testset runs the SAME input at ratios 0 / 0.5 / 1.0 and
# requires bit-identical outputs across all three (a prefetch that
# perturbs results is a miscompile, not a perf feature).

using PTX: layout_for_a, wgmma_descriptor, smem_addr_u32, tensor_map_tile_2d,
           step_desc
using PTX.MBarriers: BarrierArray,
                     barrier_wait, barrier_arrive, barrier_arrive_expect_tx
using PTX.Pipelines: Pipeline, pipeline_init!, pipeline_cursor
using CUDACore
using Random

const WPF_BM       = 64
const WPF_BN       = 8
const WPF_BK       = 16
const WPF_N_STAGES = 2
const WPF_THREADS  = 256                            # 2 warpgroups
const WPF_LOAD_BYTES    = WPF_BM * WPF_BK * 2 + WPF_BK * WPF_BN * 2
const WPF_A_STAGE_BYTES = WPF_BM * WPF_BK * 2
const WPF_B_STAGE_BYTES = WPF_BK * WPF_BN * 2

function _wpf_gemm_kernel!(
        C::CuDeviceVector{Float32, 1},
        tma_A::PTX.TMADescriptorPtr,
        tma_B::PTX.TMADescriptorPtr,
        K::Int32,
        prefetch_k_tiles::Int32)

    smem_A = CuStaticSharedArray(UInt16, WPF_N_STAGES * WPF_BM * WPF_BK)
    smem_B = CuStaticSharedArray(UInt16, WPF_N_STAGES * WPF_BK * WPF_BN)
    mbar_full  = CuStaticSharedArray(UInt64, WPF_N_STAGES)
    mbar_empty = CuStaticSharedArray(UInt64, WPF_N_STAGES)

    a_base_ptr  = pointer(smem_A)
    b_base_ptr  = pointer(smem_B)
    a_base_addr = smem_addr_u32(a_base_ptr)
    b_base_addr = smem_addr_u32(b_base_ptr)

    full  = BarrierArray{WPF_N_STAGES}(pointer(mbar_full))
    empty = BarrierArray{WPF_N_STAGES}(pointer(mbar_empty))

    tid     = ptx"mov.u32"(sreg"tid.x")
    wg_id   = tid >> UInt32(7)
    lane128 = tid & UInt32(127)

    if tid == 0
        pipeline_init!(full, empty, Val(1), Val(false))
    end
    ptx"bar.sync"(Val(0))

    num_k_tiles = K >> Int32(4)

    if wg_id == UInt32(0)
        # ── PRODUCER (WG 0) ────────────────────────────────────────────
        ptx"setmaxnreg.dec.sync.aligned.u32"(Val(24))

        # Prefetch warp = warp 1 (lane 32). Walks the weight (A) K-tiles
        # ahead of the consuming TMA loads, warming L2. Runs concurrently
        # with warp 0's issue loop below; no ordering between them exists
        # or is required — the prefetch has no observable effect other
        # than latency.
        if lane128 == UInt32(32)
            @inbounds for p_iter in Int32(0):(prefetch_k_tiles - Int32(1))
                ptx"cp.async.bulk.prefetch.tensor.2d.L2.global.tile"(
                    tma_A, p_iter * Int32(WPF_BK), Int32(0))
            end
        end

        @inbounds for k_iter in Int32(0):(num_k_tiles - Int32(1))
            stage, phase = pipeline_cursor(Pipeline{WPF_N_STAGES}, k_iter)
            barrier_wait(empty[stage], phase)

            if lane128 == UInt32(0)
                ptx"fence.proxy.async.shared::cta"()
                barrier_arrive_expect_tx(full[stage], WPF_LOAD_BYTES)
                a_stage_ptr = a_base_ptr + Int(stage) * WPF_A_STAGE_BYTES
                b_stage_ptr = b_base_ptr + Int(stage) * WPF_B_STAGE_BYTES
                k_off = k_iter * Int32(WPF_BK)
                ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                    a_stage_ptr, tma_A, k_off, Int32(0), full[stage])
                ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                    b_stage_ptr, tma_B, k_off, Int32(0), full[stage])
            end
        end
    else
        # ── CONSUMER (WG 1) — identical to gemm_pc_pipeline.jl ─────────
        ptx"setmaxnreg.inc.sync.aligned.u32"(Val(240))

        la = layout_for_a(dtype = :bf16, m = WPF_BM, k = WPF_BK)
        lb = layout_for_a(dtype = :bf16, m = WPF_BN, k = WPF_BK)
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
            stage, phase = pipeline_cursor(Pipeline{WPF_N_STAGES}, k_iter)
            barrier_wait(full[stage], phase)

            a_desc = step_desc(a_desc_base, Int(stage) * WPF_A_STAGE_BYTES)
            b_desc = step_desc(b_desc_base, Int(stage) * WPF_B_STAGE_BYTES)
            ptx"fence.proxy.async.shared::cta"()
            ptx"wgmma.fence.sync.aligned"()
            d = ptx"wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16"(
                d, a_desc, b_desc, true)
            ptx"wgmma.commit_group.sync.aligned"()
            ptx"wgmma.wait_group.sync.aligned"(Val(0))

            if lane128 == UInt32(0)
                barrier_arrive(empty[stage])
            end
        end

        wid      = lane128 >> UInt32(5)
        lane     = lane128 & UInt32(31)
        frag_row = (wid << UInt32(4)) + (lane >> UInt32(2))
        frag_col = (lane & UInt32(3)) << UInt32(1)
        pc       = pointer(C)
        off_a    = (frag_row * UInt32(WPF_BN) + frag_col) * UInt32(4)
        off_b    = ((frag_row + UInt32(8)) * UInt32(WPF_BN) + frag_col) * UInt32(4)
        ptx"st.global.v2.f32"(pc + Int(off_a), (d[1], d[2]))
        ptx"st.global.v2.f32"(pc + Int(off_b), (d[3], d[4]))
    end

    return nothing
end

# ── Cross-arch ptxas validation ────────────────────────────────────────

@testset "weight-prefetch Hopper GEMM compiles at sm_90a" begin
    types = Tuple{CuDeviceVector{Float32, 1},
                  PTX.TMADescriptorPtr, PTX.TMADescriptorPtr, Int32, Int32}
    @test ptxas_compiles(_wpf_gemm_kernel!, types;
                         cap = v"9.0", feature_set = :arch)

    ptx = emit_ptx(_wpf_gemm_kernel!, types; cap = v"9.0", feature_set = :arch)
    @test occursin("cp.async.bulk.prefetch.tensor.2d.L2.global.tile", ptx)
    @test occursin("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes", ptx)
    @test occursin("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16", ptx)
end

# ── Runtime — Hopper hardware ──────────────────────────────────────────

if v"9.0" <= DEV_CAP < v"10.0"
    @testset "weight-prefetch GEMM: ratios 0 / 0.5 / 1.0 agree bit-exactly" begin
        rng = MersenneTwister(0x63)
        K_test = 128                                # 8 K-iters
        A_f32 = randn(rng, Float32, WPF_BM, K_test) .* 0.1f0
        B_f32 = randn(rng, Float32, K_test, WPF_BN) .* 0.1f0

        A_packed = Array{UInt16}(undef, K_test, WPF_BM)
        B_packed = Array{UInt16}(undef, K_test, WPF_BN)
        for m in 1:WPF_BM, k in 1:K_test
            A_packed[k, m] = bf16_bits(A_f32[m, k])
        end
        for k in 1:K_test, n in 1:WPF_BN
            B_packed[k, n] = bf16_bits(B_f32[k, n])
        end
        A_d = CuArray(A_packed)
        B_d = CuArray(B_packed)

        tmap_A = tensor_map_tile_2d(:bf16, pointer(A_d),
            WPF_BM, K_test, WPF_BM, WPF_BK; swizzle = :B32)
        tmap_B = tensor_map_tile_2d(:bf16, pointer(B_d),
            WPF_BN, K_test, WPF_BN, WPF_BK; swizzle = :B32)
        A = upload_tma_descriptor(tmap_A)
        B = upload_tma_descriptor(tmap_B)

        num_k_tiles = K_test ÷ WPF_BK
        results = Vector{Vector{Float32}}()
        for ratio in (0.0, 0.5, 1.0)
            prefetch_tiles = Int32(ceil(ratio * num_k_tiles))
            C_d = CUDACore.zeros(Float32, WPF_BM * WPF_BN)
            @cuda threads = WPF_THREADS _wpf_gemm_kernel!(
                C_d, A.ptr, B.ptr, Int32(K_test), prefetch_tiles)
            CUDACore.synchronize()
            push!(results, Array(C_d))
        end

        # Prefetch is semantically invisible: all ratios bit-identical.
        @test results[2] == results[1]
        @test results[3] == results[1]

        C_packed = reshape(results[1], WPF_BN, WPF_BM)
        C_got    = Array{Float32}(undef, WPF_BM, WPF_BN)
        for m in 1:WPF_BM, n in 1:WPF_BN
            C_got[m, n] = C_packed[n, m]
        end
        C_ref = bf16_gemm_ref(A_f32, B_f32)
        @test isapprox(C_got, C_ref; rtol = 1e-3, atol = 1e-3)
    end
end
