# Producer + 2 consumers (M-split) pipelined Hopper GEMM.
#
# Extension of gemm_pc_pipeline.jl: same producer-consumer rendezvous,
# same N-stage SMEM ring, but the consumer side splits into TWO
# warpgroups that each handle half of the M-tile. This is the
# 1 producer + 2 consumer warpgroup pattern used by
# pyptx/examples/hopper/gemm_highperf_hopper.py — the missing bricks
# beyond gemm_pc_pipeline are:
#
#   (a) `mbar_empty[s].count = 2` and a 2-arrive pre-warm at init
#       (consumer0 + consumer1 each release the buffer once per round)
#   (b) per-consumer A-descriptor M-offset: consumer0 reads A rows
#       0..B_WG_M-1, consumer1 reads rows B_WG_M..BM-1
#   (c) the m64n16k16 wgmma frag count is 8 (not 4 as in m64n8k16)
#       and the epilogue handles two col-pair groups per lane
#
# Shape:
#   - BM = 128, BN = 16, BK = 16, N_STAGES = 2
#   - 384 threads = 3 warpgroups; WG0 = producer, WG1/WG2 = consumers
#   - Each consumer runs m64n16k16 over its half of A
#   - Single CTA, single (M, N) tile, K via runtime arg (multiple of BK)
#
# Output C is (BN, BM) col-major (N-innermost), matching the convention
# in gemm_warpgroup.jl / gemm_pc_pipeline.jl. Per consumer the writes
# land at consumer-base offset (consumer_id × B_WG_M rows).

using PTX: layout_for_a, wgmma_descriptor, smem_addr_u32, tensor_map_tile_2d,
           step_desc
using PTX.MBarriers: BarrierArray, Pipeline, pipeline_init!, pipeline_cursor,
                     barrier_wait, barrier_arrive, barrier_arrive_expect_tx
using CUDACore
using Random

const PCS_BM         = 128
const PCS_BN         = 16
const PCS_BK         = 16
const PCS_N_STAGES   = 2
const PCS_NUM_CONSUMERS = 2
const PCS_B_WG_M     = PCS_BM ÷ PCS_NUM_CONSUMERS   # 64 rows per consumer
const PCS_THREADS    = 128 * (1 + PCS_NUM_CONSUMERS)   # 384

const PCS_LOAD_BYTES = PCS_BM * PCS_BK * 2 + PCS_BK * PCS_BN * 2

const PCS_A_STAGE_BYTES = PCS_BM * PCS_BK * 2     # 4096
const PCS_B_STAGE_BYTES = PCS_BK * PCS_BN * 2     # 512

# Per-consumer A-offset within one stage's BM×BK A tile (in bytes).
const PCS_A_CONSUMER_BYTES = PCS_B_WG_M * PCS_BK * 2     # 2048

function _pcs_gemm_kernel!(
        C::CuDeviceVector{Float32, 1},
        tma_A::PTX.TMADescriptorPtr,
        tma_B::PTX.TMADescriptorPtr,
        K::Int32)

    smem_A = CuStaticSharedArray(UInt16, PCS_N_STAGES * PCS_BM * PCS_BK)
    smem_B = CuStaticSharedArray(UInt16, PCS_N_STAGES * PCS_BK * PCS_BN)
    mbar_full  = CuStaticSharedArray(UInt64, PCS_N_STAGES)
    mbar_empty = CuStaticSharedArray(UInt64, PCS_N_STAGES)

    a_base_ptr = pointer(smem_A)
    b_base_ptr = pointer(smem_B)

    full  = BarrierArray{PCS_N_STAGES}(pointer(mbar_full))
    empty = BarrierArray{PCS_N_STAGES}(pointer(mbar_empty))

    tid     = ptx"mov.u32"(sreg"tid.x")
    wg_id   = tid >> UInt32(7)              # 0 = producer, 1/2 = consumers
    lane128 = tid & UInt32(127)

    # ── Init: thread 0 inits all mbarriers and pre-arrives empty[s]
    # twice per stage (one per consumer). After init, empty[s].phase = 1
    # (count=2, two arrives → flip), so the producer's first wait
    # (expected_phase=0) succeeds without consumer participation.
    if tid == 0
        pipeline_init!(full, empty, Val(PCS_NUM_CONSUMERS), Val(false))
    end
    ptx"bar.sync"(Val(0))

    num_k_tiles = K >> Int32(4)             # K / BK

    if wg_id == UInt32(0)
        # ─────────────────────── PRODUCER (WG 0) ─────────────────────
        ptx"setmaxnreg.dec.sync.aligned.u32"(Val(24))

        @inbounds for k_iter in Int32(0):(num_k_tiles - Int32(1))
            stage, phase = pipeline_cursor(Pipeline{PCS_N_STAGES}, k_iter)
            barrier_wait(empty[stage], phase)

            if lane128 == UInt32(0)
                ptx"fence.proxy.async.shared::cta"()
                barrier_arrive_expect_tx(full[stage], PCS_LOAD_BYTES)
                a_stage_ptr = a_base_ptr + Int(stage) * PCS_A_STAGE_BYTES
                b_stage_ptr = b_base_ptr + Int(stage) * PCS_B_STAGE_BYTES
                k_off = k_iter * Int32(PCS_BK)
                ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                    a_stage_ptr, tma_A, k_off, Int32(0), full[stage])
                ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                    b_stage_ptr, tma_B, k_off, Int32(0), full[stage])
            end
        end
    else
        # ─────────────────── CONSUMER (WG 1 or WG 2) ─────────────────
        ptx"setmaxnreg.inc.sync.aligned.u32"(Val(240))
        consumer_id = wg_id - UInt32(1)                  # 0 or 1

        # Each consumer's A-descriptor points to its M-slice of stage 0.
        # `Int(consumer_id)` keeps the host-side bytecount math static.
        a_slice_ptr  = a_base_ptr + Int(consumer_id) * PCS_A_CONSUMER_BYTES
        a_slice_addr = smem_addr_u32(a_slice_ptr)
        b_addr       = smem_addr_u32(b_base_ptr)

        # m64n16k16 wgmma: A side is m64 × k16 (one consumer's slice),
        # B side is k16 × n16 (shared between consumers).
        la = layout_for_a(dtype = :bf16, m = PCS_B_WG_M, k = PCS_BK)
        lb = layout_for_a(dtype = :bf16, m = PCS_BN,     k = PCS_BK)
        a_desc_base = wgmma_descriptor(a_slice_addr;
            leading_byte_offset = la.leading_byte_offset,
            stride_byte_offset  = la.stride_byte_offset,
            swizzle             = la.layout_type)
        b_desc_base = wgmma_descriptor(b_addr;
            leading_byte_offset = lb.leading_byte_offset,
            stride_byte_offset  = lb.stride_byte_offset,
            swizzle             = lb.layout_type)

        d = ntuple(_ -> 0f0, Val(8))         # m64n16k16 → 8 f32 frags/lane

        @inbounds for k_iter in Int32(0):(num_k_tiles - Int32(1))
            stage, phase = pipeline_cursor(Pipeline{PCS_N_STAGES}, k_iter)
            barrier_wait(full[stage], phase)

            a_desc = step_desc(a_desc_base, Int(stage) * PCS_A_STAGE_BYTES)
            b_desc = step_desc(b_desc_base, Int(stage) * PCS_B_STAGE_BYTES)
            ptx"fence.proxy.async.shared::cta"()
            ptx"wgmma.fence.sync.aligned"()
            d = ptx"wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16"(
                d, a_desc, b_desc, true)
            ptx"wgmma.commit_group.sync.aligned"()
            ptx"wgmma.wait_group.sync.aligned"(Val(0))

            if lane128 == UInt32(0)
                barrier_arrive(empty[stage])
            end
        end

        # ── Epilogue. m64n16k16 fragment layout (PTX 9.2 §9.7.14.5.5):
        #   warp w ∈ 0..3, lane l ∈ 0..31
        #   row_a = 16w + (l >> 2),  row_b = row_a + 8
        #   col_a = 2*(l & 3),       col_b = col_a + 8
        #   frags 1..4 cover (row_a/row_b, col_a/col_a+1)
        #   frags 5..8 cover (row_a/row_b, col_b/col_b+1)
        # Two v2.f32 stores per col-group per lane = 4 v2.f32 per lane.
        wid      = lane128 >> UInt32(5)
        lane     = lane128 & UInt32(31)
        frag_row = (wid << UInt32(4)) + (lane >> UInt32(2))
        frag_col = (lane & UInt32(3)) << UInt32(1)
        # Per-consumer base in C (N-innermost): consumer_id × B_WG_M rows.
        c_consumer_base = pointer(C) +
            Int(consumer_id) * PCS_B_WG_M * PCS_BN * sizeof(Float32)
        off_a1 = (frag_row * UInt32(PCS_BN) + frag_col) * UInt32(4)
        off_b1 = ((frag_row + UInt32(8)) * UInt32(PCS_BN) + frag_col) * UInt32(4)
        off_a2 = (frag_row * UInt32(PCS_BN) + frag_col + UInt32(8)) * UInt32(4)
        off_b2 = ((frag_row + UInt32(8)) * UInt32(PCS_BN) + frag_col + UInt32(8)) * UInt32(4)
        ptx"st.global.v2.f32"(c_consumer_base + Int(off_a1), (d[1], d[2]))
        ptx"st.global.v2.f32"(c_consumer_base + Int(off_b1), (d[3], d[4]))
        ptx"st.global.v2.f32"(c_consumer_base + Int(off_a2), (d[5], d[6]))
        ptx"st.global.v2.f32"(c_consumer_base + Int(off_b2), (d[7], d[8]))
    end

    return nothing
end

# ── Cross-arch ptxas validation ────────────────────────────────────────

@testset "producer-2-consumer Hopper GEMM compiles at sm_90a" begin
    types = Tuple{CuDeviceVector{Float32, 1},
                  PTX.TMADescriptorPtr, PTX.TMADescriptorPtr, Int32}
    @test ptxas_compiles(_pcs_gemm_kernel!, types;
                         cap = v"9.0", feature_set = :arch)

    ptx = emit_ptx(_pcs_gemm_kernel!, types; cap = v"9.0", feature_set = :arch)
    @test occursin("setmaxnreg.inc.sync.aligned.u32",  ptx)
    @test occursin("setmaxnreg.dec.sync.aligned.u32",  ptx)
    @test occursin("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16", ptx)
    # Two consumers + one producer pre-warm = 2 arrives at init + 2 per
    # iter; the kernel must emit mbarrier.arrive.shared.b64 somewhere.
    @test occursin("mbarrier.arrive.shared.b64",       ptx)
end

# ── Runtime — Hopper hardware ──────────────────────────────────────────

if v"9.0" <= DEV_CAP < v"10.0"
    @testset "producer-2-consumer Hopper GEMM (K=64, random bf16)" begin
        rng = MersenneTwister(0x5912)
        K_test = 64
        A_f32 = randn(rng, Float32, PCS_BM, K_test) .* 0.1f0
        B_f32 = randn(rng, Float32, K_test, PCS_BN) .* 0.1f0

        A_packed = Array{UInt16}(undef, K_test, PCS_BM)
        B_packed = Array{UInt16}(undef, K_test, PCS_BN)
        for m in 1:PCS_BM, k in 1:K_test
            A_packed[k, m] = bf16_bits(A_f32[m, k])
        end
        for k in 1:K_test, n in 1:PCS_BN
            B_packed[k, n] = bf16_bits(B_f32[k, n])
        end
        A_d = CuArray(A_packed)
        B_d = CuArray(B_packed)

        tmap_A = tensor_map_tile_2d(:bf16, pointer(A_d),
            PCS_BM, K_test, PCS_BM, PCS_BK; swizzle = :B32)
        tmap_B = tensor_map_tile_2d(:bf16, pointer(B_d),
            PCS_BN, K_test, PCS_BN, PCS_BK; swizzle = :B32)
        A = upload_tma_descriptor(tmap_A)
        B = upload_tma_descriptor(tmap_B)

        C_d = CUDACore.zeros(Float32, PCS_BM * PCS_BN)
        @cuda threads = PCS_THREADS feature_set = :arch _pcs_gemm_kernel!(
            C_d, A.ptr, B.ptr, Int32(K_test))
        CUDACore.synchronize()

        # Output is (BN, BM) N-fast — reshape and transpose.
        C_packed = reshape(Array(C_d), PCS_BN, PCS_BM)
        C_got    = Array{Float32}(undef, PCS_BM, PCS_BN)
        for m in 1:PCS_BM, n in 1:PCS_BN
            C_got[m, n] = C_packed[n, m]
        end

        C_ref = bf16_gemm_ref(A_f32, B_f32)
        @test isapprox(C_got, C_ref; rtol = 1e-3, atol = 1e-3)
    end
end
