# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc==9.0
# Producer-consumer pipelined Hopper GEMM — the canonical pyptx warp-
# specialized pattern, isolated from clusters / multicast / scheduling.
#
# Stepping stone between gemm_warpgroup.jl (single warpgroup, one K tile)
# and a full gemm_highperf_hopper port. Validates the brick that's hardest
# to debug in isolation: an N-stage SMEM ring driven by two warpgroups
# rendezvous-ing on per-stage mbarrier pairs.
#
# Shape:
#   - BM = 64, BN = 8, BK = 16 (single wgmma m64n8k16 per stage)
#   - N_STAGES = 2 (double-buffered)
#   - 256 threads = 2 warpgroups; WG0 = producer (tid 0..127),
#     WG1 = consumer (tid 128..255)
#   - K is a runtime arg; must be a multiple of BK
#   - Single CTA, single (M, N) tile — no scheduling
#
# mbarrier layout:
#   mbar_full[s]  — count = 1 (producer's arrive.expect_tx fires once per
#                  stage). Consumer waits on this to know the SMEM tile
#                  is loaded; flips phase on TMA completion.
#   mbar_empty[s] — count = 1 (consumer's arrive fires once per stage when
#                  wgmma is done). Producer waits on this to know the
#                  SMEM tile is free to overwrite.
#
# Phase tracking: each stage cycles through phase 0 / 1 / 0 / 1 / … as
# the ring wraps. For iter k, stage = k mod N_STAGES, expected phase =
# (k ÷ N_STAGES) & 1. Same expression for both producer and consumer
# (with appropriate mbarrier).
#
# Pipeline warm-up: thread 0 pre-arrives every `empty[s]` once at init
# time so the producer's first wait on empty succeeds without the consumer
# having started. After this, the loop bodies handle phase tracking
# unaided.

using PTX: layout_for_a, wgmma_descriptor, smem_addr_u32, tensor_map_tile_2d,
           step_desc
using PTX.MBarriers: BarrierArray,
                     barrier_wait, barrier_arrive, barrier_arrive_expect_tx
using PTX.Pipelines: Pipeline, pipeline_init!, pipeline_cursor
using CUDACore
using Random

const PC_BM       = 64
const PC_BN       = 8
const PC_BK       = 16
const PC_N_STAGES = 2
const PC_THREADS  = 256                                 # 2 warpgroups
const PC_LOAD_BYTES = PC_BM * PC_BK * 2 + PC_BK * PC_BN * 2

# Per-stage SMEM bytes for A and B.
const PC_A_STAGE_BYTES = PC_BM * PC_BK * 2              # 2048
const PC_B_STAGE_BYTES = PC_BK * PC_BN * 2              # 256

function _pc_gemm_kernel!(
        C::CuDeviceVector{Float32, 1},
        tma_A::PTX.TMADescriptorPtr,
        tma_B::PTX.TMADescriptorPtr,
        K::Int32)

    # Ring buffers — N_STAGES copies of each tile, contiguous in SMEM.
    smem_A = CuStaticSharedArray(UInt16, PC_N_STAGES * PC_BM * PC_BK)
    smem_B = CuStaticSharedArray(UInt16, PC_N_STAGES * PC_BK * PC_BN)
    mbar_full  = CuStaticSharedArray(UInt64, PC_N_STAGES)
    mbar_empty = CuStaticSharedArray(UInt64, PC_N_STAGES)

    a_base_ptr  = pointer(smem_A)
    b_base_ptr  = pointer(smem_B)
    a_base_addr = smem_addr_u32(a_base_ptr)
    b_base_addr = smem_addr_u32(b_base_ptr)

    full  = BarrierArray{PC_N_STAGES}(pointer(mbar_full))
    empty = BarrierArray{PC_N_STAGES}(pointer(mbar_empty))

    tid     = ptx"mov.u32"(sreg"tid.x")
    wg_id   = tid >> UInt32(7)              # 0 = producer, 1 = consumer
    lane128 = tid & UInt32(127)             # lane within warpgroup

    # ── Init mbarriers + pipeline warm-up ───────────────────────────
    # Thread 0 inits 2 × N_STAGES mbarriers and pre-arrives every
    # `empty[s]` once. After init, empty[s] is at phase=1, so the
    # producer's first wait (expected_phase=0) succeeds without waiting.
    if tid == 0
        pipeline_init!(full, empty, Val(1), Val(false))
    end
    ptx"bar.sync"(Val(0))                                   # CTA-wide

    num_k_tiles = K >> Int32(4)                             # K / 16 = K / BK

    if wg_id == UInt32(0)
        # ─────────────────────────────────────────────────────────────
        # PRODUCER (WG 0). Issues all TMA loads; signals full[stage].
        # Holds fewer registers — give them back to the consumer.
        # ─────────────────────────────────────────────────────────────
        ptx"setmaxnreg.dec.sync.aligned.u32"(Val(24))

        @inbounds for k_iter in Int32(0):(num_k_tiles - Int32(1))
            stage, phase = pipeline_cursor(Pipeline{PC_N_STAGES}, k_iter)

            # Wait for consumer to free this stage's SMEM buffer.
            barrier_wait(empty[stage], phase)

            # Lane 0 issues the TMA pair + arms `full[stage]` for completion.
            if lane128 == UInt32(0)
                ptx"fence.proxy.async.shared::cta"()
                barrier_arrive_expect_tx(full[stage], PC_LOAD_BYTES)
                a_stage_ptr = a_base_ptr + Int(stage) * PC_A_STAGE_BYTES
                b_stage_ptr = b_base_ptr + Int(stage) * PC_B_STAGE_BYTES
                # TMA coords: K-fast layout → (k_off, m_or_n_off). m_off / n_off
                # are 0 here (single tile in M/N); only K advances.
                k_off = k_iter * Int32(PC_BK)
                ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                    a_stage_ptr, tma_A, k_off, Int32(0), full[stage])
                ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                    b_stage_ptr, tma_B, k_off, Int32(0), full[stage])
            end
        end
    else
        # ─────────────────────────────────────────────────────────────
        # CONSUMER (WG 1). Waits on full[stage], runs wgmma, releases
        # empty[stage]. Owns the f32 accumulator + epilogue.
        # ─────────────────────────────────────────────────────────────
        ptx"setmaxnreg.inc.sync.aligned.u32"(Val(240))

        # Build base descriptors once (per-stage offsets fold in via
        # the descriptor byte-offset trick at the call site).
        la = layout_for_a(dtype = :bf16, m = PC_BM, k = PC_BK)
        lb = layout_for_a(dtype = :bf16, m = PC_BN, k = PC_BK)
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
            stage, phase = pipeline_cursor(Pipeline{PC_N_STAGES}, k_iter)

            # Wait for producer's TMA to land for this stage.
            barrier_wait(full[stage], phase)

            # Per-stage SMEM offset on the wgmma operand descriptors:
            # step_desc handles the bytes → 16-byte-unit conversion.
            a_desc = step_desc(a_desc_base, Int(stage) * PC_A_STAGE_BYTES)
            b_desc = step_desc(b_desc_base, Int(stage) * PC_B_STAGE_BYTES)
            ptx"fence.proxy.async.shared::cta"()
            ptx"wgmma.fence.sync.aligned"()
            d = ptx"wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16"(
                d, a_desc, b_desc, true)            # accumulate
            ptx"wgmma.commit_group.sync.aligned"()
            ptx"wgmma.wait_group.sync.aligned"(Val(0))

            # Signal stage's SMEM is free for the next round.
            if lane128 == UInt32(0)
                barrier_arrive(empty[stage])
            end
        end

        # ── Epilogue. m64n8k16 wgmma f32 frag layout (same as gemm_warpgroup):
        #    warp w ∈ 0..3, lane l ∈ 0..31 →
        #      frag_row = 16w + (l >> 2),  frag_col = 2 * (l & 3)
        #    4 f32 frags per lane at (frag_row, frag_col), (frag_row, frag_col+1),
        #    (frag_row+8, frag_col), (frag_row+8, frag_col+1).
        # Only the consumer warpgroup runs this — `lane128` is 0..127.
        wid      = lane128 >> UInt32(5)
        lane     = lane128 & UInt32(31)
        frag_row = (wid << UInt32(4)) + (lane >> UInt32(2))
        frag_col = (lane & UInt32(3)) << UInt32(1)
        pc       = pointer(C)
        off_a    = (frag_row * UInt32(PC_BN) + frag_col) * UInt32(4)
        off_b    = ((frag_row + UInt32(8)) * UInt32(PC_BN) + frag_col) * UInt32(4)
        ptx"st.global.v2.f32"(pc + Int(off_a), (d[1], d[2]))
        ptx"st.global.v2.f32"(pc + Int(off_b), (d[3], d[4]))
    end

    return nothing
end

# ── Cross-arch ptxas validation ────────────────────────────────────────

@testset "producer-consumer Hopper GEMM compiles at sm_90a" begin
    types = Tuple{CuDeviceVector{Float32, 1},
                  PTX.TMADescriptorPtr, PTX.TMADescriptorPtr, Int32}
    @test ptxas_compiles(_pc_gemm_kernel!, types;
                         cap = v"9.0", feature_set = :arch)

    ptx = emit_ptx(_pc_gemm_kernel!, types; cap = v"9.0", feature_set = :arch)
    @test occursin("setmaxnreg.inc.sync.aligned.u32",  ptx)
    @test occursin("setmaxnreg.dec.sync.aligned.u32",  ptx)
    @test occursin("mbarrier.init.shared.b64",          ptx)
    @test occursin("mbarrier.arrive.expect_tx.shared.b64", ptx)
    @test occursin("mbarrier.arrive.shared.b64",        ptx)
    @test occursin("mbarrier.test_wait.parity.shared.b64", ptx)
    @test occursin("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16", ptx)
end

# ── Runtime — Hopper hardware ──────────────────────────────────────────

if test_runtime_supported(@__FILE__)
    @testset "producer-consumer Hopper GEMM (random bf16, K=64)" begin
        # K = 64 → 4 K-iters, 2 wraparounds through the 2-stage ring.
        # Exercises the phase-tracking path (first 2 iters phase=0, last 2 iters phase=1).
        rng = MersenneTwister(0xc0ffee)
        K_test = 64
        A_f32 = randn(rng, Float32, PC_BM, K_test) .* 0.1f0
        B_f32 = randn(rng, Float32, K_test, PC_BN) .* 0.1f0

        # Pack to bf16 with K-fast convention (matches gemm_warpgroup.jl).
        # A: (M, K) → Julia (K, M) col-major (K innermost).
        # B: (K, N) → Julia (K, N) col-major (K innermost).
        A_packed = Array{UInt16}(undef, K_test, PC_BM)
        B_packed = Array{UInt16}(undef, K_test, PC_BN)
        for m in 1:PC_BM, k in 1:K_test
            A_packed[k, m] = bf16_bits(A_f32[m, k])
        end
        for k in 1:K_test, n in 1:PC_BN
            B_packed[k, n] = bf16_bits(B_f32[k, n])
        end
        A_d = CuArray(A_packed)
        B_d = CuArray(B_packed)

        # TMA: tile = (BM, BK) for A; (BN, BK) for B. K-fast, row-width
        # 32 B → B32 swizzle.
        tmap_A = tensor_map_tile_2d(:bf16, pointer(A_d),
            PC_BM, K_test, PC_BM, PC_BK; swizzle = :B32)
        tmap_B = tensor_map_tile_2d(:bf16, pointer(B_d),
            PC_BN, K_test, PC_BN, PC_BK; swizzle = :B32)
        A = upload_tma_descriptor(tmap_A)
        B = upload_tma_descriptor(tmap_B)

        C_d = CUDACore.zeros(Float32, PC_BM * PC_BN)
        @cuda threads = PC_THREADS _pc_gemm_kernel!(
            C_d, A.ptr, B.ptr, Int32(K_test))
        CUDACore.synchronize()

        # Output is (BN, BM) N-fast in memory (each lane v2.f32 stores two
        # consecutive N values at one M row); reshape back to (M, N).
        C_packed = reshape(Array(C_d), PC_BN, PC_BM)
        C_got    = Array{Float32}(undef, PC_BM, PC_BN)
        for m in 1:PC_BM, n in 1:PC_BN
            C_got[m, n] = C_packed[n, m]
        end

        C_ref = bf16_gemm_ref(A_f32, B_f32)
        @test isapprox(C_got, C_ref; rtol = 1e-3, atol = 1e-3)
    end

    @testset "producer-consumer Hopper GEMM (K=128, more ring wraps)" begin
        rng = MersenneTwister(0xb0bb1e)
        K_test = 128
        A_f32 = randn(rng, Float32, PC_BM, K_test) .* 0.1f0
        B_f32 = randn(rng, Float32, K_test, PC_BN) .* 0.1f0

        A_packed = Array{UInt16}(undef, K_test, PC_BM)
        B_packed = Array{UInt16}(undef, K_test, PC_BN)
        for m in 1:PC_BM, k in 1:K_test
            A_packed[k, m] = bf16_bits(A_f32[m, k])
        end
        for k in 1:K_test, n in 1:PC_BN
            B_packed[k, n] = bf16_bits(B_f32[k, n])
        end
        A_d = CuArray(A_packed)
        B_d = CuArray(B_packed)

        tmap_A = tensor_map_tile_2d(:bf16, pointer(A_d),
            PC_BM, K_test, PC_BM, PC_BK; swizzle = :B32)
        tmap_B = tensor_map_tile_2d(:bf16, pointer(B_d),
            PC_BN, K_test, PC_BN, PC_BK; swizzle = :B32)
        A = upload_tma_descriptor(tmap_A)
        B = upload_tma_descriptor(tmap_B)

        C_d = CUDACore.zeros(Float32, PC_BM * PC_BN)
        @cuda threads = PC_THREADS _pc_gemm_kernel!(
            C_d, A.ptr, B.ptr, Int32(K_test))
        CUDACore.synchronize()

        C_packed = reshape(Array(C_d), PC_BN, PC_BM)
        C_got    = Array{Float32}(undef, PC_BM, PC_BN)
        for m in 1:PC_BM, n in 1:PC_BN
            C_got[m, n] = C_packed[n, m]
        end

        C_ref = bf16_gemm_ref(A_f32, B_f32)
        @test isapprox(C_got, C_ref; rtol = 1e-3, atol = 1e-3)
    end
end
