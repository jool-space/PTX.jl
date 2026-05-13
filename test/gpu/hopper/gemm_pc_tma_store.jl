# Cluster GEMM with TMA store epilogue + bf16 output.
#
# Differs from gemm_pc_pipeline_cluster.jl in only one piece — the
# epilogue. Same producer/consumer pattern, same cluster + multicast B,
# same multi-tile grid. Output is now bf16 (not f32), and per-consumer
# TMA store replaces the per-lane v2.f32 global writes.
#
# Epilogue choreography (per consumer in each CTA):
#   1. Each lane converts its 8 f32 frags → 4 bf16x2 packed UInt32s
#      via `cvt.rn.bf16x2.f32`.
#   2. Linear `st.shared.b32` writes to a per-consumer SMEM region
#      (BN-innermost, B_WG_M rows × BN cols bf16). Per-lane positions
#      mirror the m64n16k16 wgmma fragment layout.
#   3. Named-barrier sync within the consumer warpgroup
#      (`bar.sync consumer_id+2, 128`): waits only for the 128 threads
#      of THIS consumer, doesn't pull in the producer or sister consumer.
#   4. Lane 0 issues a TMA store of the (B_WG_M × BN) bf16 tile to
#      global at coord = (n_off_global, m_off_global + consumer_id × B_WG_M).
#   5. `cp.async.bulk.commit_group` queues the store; the final
#      `cp.async.bulk.wait_group(0)` outside the K-loop drains all stores
#      before kernel exit.
#
# TMA store descriptor: 2D bf16, (N_total, M_total) col-major BN-fast,
# tile (BN, B_WG_M), swizzle = :NONE (matches our linear SMEM writes —
# no swizzle math at the consumer side).

using PTX: layout_for_a, wgmma_descriptor, smem_addr_u32, tensor_map_tile_2d,
           bf16x2_pack, step_desc
using PTX.MBarriers: BarrierArray,
                     barrier_wait, barrier_arrive_expect_tx, barrier_arrive_cluster
using PTX.Pipelines: Pipeline, pipeline_init!, pipeline_cursor
using CUDACore
using CUDACore: cluster_arrive, cluster_wait
using Random

const PCT_BM_CTA     = 128
const PCT_BN         = 16
const PCT_BK         = 16
const PCT_N_STAGES   = 2
const PCT_NUM_CONSUMERS = 2
const PCT_CLUSTERS   = 2
const PCT_B_WG_M     = PCT_BM_CTA ÷ PCT_NUM_CONSUMERS
const PCT_THREADS    = 128 * (1 + PCT_NUM_CONSUMERS)
const PCT_BM_CLUSTER = PCT_BM_CTA * PCT_CLUSTERS

const PCT_A_PER_CTA  = PCT_BM_CTA * PCT_BK * 2
const PCT_B_BYTES    = PCT_BK     * PCT_BN * 2
const PCT_LOAD_BYTES = PCT_A_PER_CTA + PCT_B_BYTES

const PCT_A_STAGE_BYTES = PCT_A_PER_CTA
const PCT_B_STAGE_BYTES = PCT_B_BYTES

const PCT_A_CONSUMER_BYTES = PCT_B_WG_M * PCT_BK * 2

const PCT_EMPTY_COUNT = PCT_NUM_CONSUMERS * PCT_CLUSTERS

# Per-consumer SMEM C-buffer (bf16): B_WG_M × BN bf16 = 64 × 16 × 2 = 2048 B.
const PCT_C_CONSUMER_BYTES = PCT_B_WG_M * PCT_BN * 2

function _pct_gemm_kernel!(
        C::CuDeviceVector{UInt16, 1},                      # bf16 output (stored as UInt16)
        tma_A::PTX.TMADescriptorPtr,
        tma_B::PTX.TMADescriptorPtr,
        tma_C::PTX.TMADescriptorPtr,
        M::Int32, N::Int32, K::Int32)

    smem_A = CuStaticSharedArray(UInt16, PCT_N_STAGES * PCT_BM_CTA * PCT_BK)
    smem_B = CuStaticSharedArray(UInt16, PCT_N_STAGES * PCT_BK    * PCT_BN)
    smem_C = CuStaticSharedArray(UInt16, PCT_BM_CTA   * PCT_BN)      # 4096 B
    mbar_full  = CuStaticSharedArray(UInt64, PCT_N_STAGES)
    mbar_empty = CuStaticSharedArray(UInt64, PCT_N_STAGES)

    a_base_ptr = pointer(smem_A)
    b_base_ptr = pointer(smem_B)
    c_base_ptr = pointer(smem_C)

    full  = BarrierArray{PCT_N_STAGES}(pointer(mbar_full))
    empty = BarrierArray{PCT_N_STAGES}(pointer(mbar_empty))

    tid      = ptx"mov.u32"(sreg"tid.x")
    cta_rank = ptx"mov.u32"(sreg"cluster_ctarank")
    ctaid_x  = ptx"mov.u32"(sreg"ctaid.x")
    ctaid_y  = ptx"mov.u32"(sreg"ctaid.y")
    wg_id    = tid >> UInt32(7)
    lane128  = tid & UInt32(127)

    tile_m_cluster = ctaid_y
    tile_n         = ctaid_x >> UInt32(1)
    m_off_global = tile_m_cluster * UInt32(PCT_BM_CLUSTER) + cta_rank * UInt32(PCT_BM_CTA)
    n_off_global = tile_n * UInt32(PCT_BN)

    # ── Init mbarriers + pipeline warm-up ───────────────────────────
    if tid == UInt32(0)
        pipeline_init!(full, empty, Val(PCT_EMPTY_COUNT), Val(true))
    end
    ptx"bar.sync"(Val(0))

    cluster_arrive()
    cluster_wait()

    num_k_tiles = K >> Int32(4)

    if wg_id == UInt32(0)
        # ─────────────────────── PRODUCER ────────────────────────────
        ptx"setmaxnreg.dec.sync.aligned.u32"(Val(24))

        @inbounds for k_iter in Int32(0):(num_k_tiles - Int32(1))
            stage, phase = pipeline_cursor(Pipeline{PCT_N_STAGES}, k_iter)
            barrier_wait(empty[stage], phase)

            if lane128 == UInt32(0)
                ptx"fence.proxy.async.shared::cta"()
                barrier_arrive_expect_tx(full[stage], PCT_LOAD_BYTES)
                a_stage_ptr = a_base_ptr + Int(stage) * PCT_A_STAGE_BYTES
                b_stage_ptr = b_base_ptr + Int(stage) * PCT_B_STAGE_BYTES
                k_off = k_iter * Int32(PCT_BK)
                ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                    a_stage_ptr, tma_A, k_off, reinterpret(Int32, m_off_global), full[stage])
                if cta_rank == UInt32(0)
                    ptx"cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"(
                        b_stage_ptr, tma_B, k_off,
                        reinterpret(Int32, n_off_global), full[stage], UInt16(0x3))
                end
            end
        end
    else
        # ─────────────────────── CONSUMER ────────────────────────────
        ptx"setmaxnreg.inc.sync.aligned.u32"(Val(240))
        consumer_id = wg_id - UInt32(1)

        a_slice_ptr  = a_base_ptr + Int(consumer_id) * PCT_A_CONSUMER_BYTES
        a_slice_addr = smem_addr_u32(a_slice_ptr)
        b_addr       = smem_addr_u32(b_base_ptr)

        la = layout_for_a(dtype = :bf16, m = PCT_B_WG_M, k = PCT_BK)
        lb = layout_for_a(dtype = :bf16, m = PCT_BN,     k = PCT_BK)
        a_desc_base = wgmma_descriptor(a_slice_addr;
            leading_byte_offset = la.leading_byte_offset,
            stride_byte_offset  = la.stride_byte_offset,
            swizzle             = la.layout_type)
        b_desc_base = wgmma_descriptor(b_addr;
            leading_byte_offset = lb.leading_byte_offset,
            stride_byte_offset  = lb.stride_byte_offset,
            swizzle             = lb.layout_type)

        d = ntuple(_ -> 0f0, Val(8))

        @inbounds for k_iter in Int32(0):(num_k_tiles - Int32(1))
            stage, phase = pipeline_cursor(Pipeline{PCT_N_STAGES}, k_iter)
            barrier_wait(full[stage], phase)

            a_desc = step_desc(a_desc_base, Int(stage) * PCT_A_STAGE_BYTES)
            b_desc = step_desc(b_desc_base, Int(stage) * PCT_B_STAGE_BYTES)
            ptx"fence.proxy.async.shared::cta"()
            ptx"wgmma.fence.sync.aligned"()
            d = ptx"wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16"(
                d, a_desc, b_desc, true)
            ptx"wgmma.commit_group.sync.aligned"()
            ptx"wgmma.wait_group.sync.aligned"(Val(0))

            if lane128 < UInt32(PCT_CLUSTERS)
                barrier_arrive_cluster(empty[stage], lane128)
            end
        end

        # ── Epilogue: f32 → bf16, write to per-consumer SMEM tile ────
        # Convert 8 f32 frags into 4 bf16x2 packed UInt32s. m64n16k16
        # frag layout per lane (w = lane128 >> 5, l = lane128 & 31):
        #   row_a = 16w + (l >> 2),  row_b = row_a + 8
        #   col_a = 2*(l & 3),       col_b = col_a + 8
        # Frags pair up as (d1,d2), (d3,d4), (d5,d6), (d7,d8) — each
        # pair lands at one (row, col, col+1) position.
        wid   = lane128 >> UInt32(5)
        lane  = lane128 & UInt32(31)
        row_a = (wid << UInt32(4)) + (lane >> UInt32(2))
        row_b = row_a + UInt32(8)
        col_a = (lane & UInt32(3)) << UInt32(1)
        col_b = col_a + UInt32(8)

        # PTX 9.2 §9.7.9.21: `cvt.rn.bf16x2.f32 d, a, b` puts `a` in the
        # UPPER half of `d` and `b` in the LOWER half. `bf16x2_pack(lo, hi)`
        # hides the arg swap so this reads in tuple-natural (lo, hi) order.
        p_a1 = bf16x2_pack(d[1], d[2])
        p_a2 = bf16x2_pack(d[3], d[4])
        p_b1 = bf16x2_pack(d[5], d[6])
        p_b2 = bf16x2_pack(d[7], d[8])

        # Per-consumer SMEM C-tile base (linear, BN-innermost).
        c_consumer_ptr = c_base_ptr + Int(consumer_id) * PCT_C_CONSUMER_BYTES
        # Byte offsets within the per-consumer (B_WG_M × BN) bf16 tile.
        off_a1 = (row_a * UInt32(PCT_BN) + col_a) * UInt32(2)
        off_b1 = (row_b * UInt32(PCT_BN) + col_a) * UInt32(2)
        off_a2 = (row_a * UInt32(PCT_BN) + col_b) * UInt32(2)
        off_b2 = (row_b * UInt32(PCT_BN) + col_b) * UInt32(2)
        ptx"st.shared.b32"(c_consumer_ptr + Int(off_a1), p_a1)
        ptx"st.shared.b32"(c_consumer_ptr + Int(off_b1), p_a2)
        ptx"st.shared.b32"(c_consumer_ptr + Int(off_a2), p_b1)
        ptx"st.shared.b32"(c_consumer_ptr + Int(off_b2), p_b2)

        # Wait for this consumer's 128 threads to finish the SMEM landing.
        # `bar.sync (consumer_id + 2), 128` — distinct named barrier per
        # consumer wg so the two consumers don't accidentally rendezvous.
        if consumer_id == UInt32(0)
            ptx"bar.sync"(Val(2), Val(128))
        else
            ptx"bar.sync"(Val(3), Val(128))
        end

        # Lane 0 of each consumer issues its TMA store. Coord =
        # (n_off_global, m_off_global + consumer_id × B_WG_M).
        if lane128 == UInt32(0)
            ptx"fence.proxy.async.shared::cta"()
            store_m = reinterpret(Int32,
                                  m_off_global + consumer_id * UInt32(PCT_B_WG_M))
            store_n = reinterpret(Int32, n_off_global)
            ptx"cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"(
                tma_C, store_n, store_m, c_consumer_ptr)
            ptx"cp.async.bulk.commit_group"()
        end
    end

    # Drain any in-flight TMA stores before kernel exit. Only consumers
    # issued stores; producer's wait_group is harmless (no group queued).
    ptx"cp.async.bulk.wait_group"(Val(0))
    return nothing
end

# ── Cross-arch ptxas validation ────────────────────────────────────────

@testset "cluster GEMM with TMA store compiles at sm_90a" begin
    types = Tuple{CuDeviceVector{UInt16, 1},
                  PTX.TMADescriptorPtr, PTX.TMADescriptorPtr, PTX.TMADescriptorPtr,
                  Int32, Int32, Int32}
    @test ptxas_compiles(_pct_gemm_kernel!, types;
                         cap = v"9.0", feature_set = :arch)

    ptx = emit_ptx(_pct_gemm_kernel!, types; cap = v"9.0", feature_set = :arch)
    # Epilogue packs two f32 frags per col-pair into one bf16x2 via
    # `bf16x2_pack(lo, hi)` (fused `cvt.rn.bf16x2.f32` under the hood).
    @test occursin("cvt.rn.bf16x2.f32",                        ptx)
    @test occursin("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group", ptx)
    @test occursin("cp.async.bulk.commit_group",               ptx)
    @test occursin("cp.async.bulk.wait_group",                 ptx)
    @test occursin("bar.sync",                                 ptx)
end

# ── Runtime — clusters-capable arch only ───────────────────────────────

if v"9.0" <= DEV_CAP < v"12.0"
    function _run_pct_case(rng, M_total, N_total, K_test)
        @assert M_total % PCT_BM_CLUSTER == 0
        @assert N_total % PCT_BN         == 0
        @assert K_test  % PCT_BK         == 0

        A_f32 = randn(rng, Float32, M_total, K_test) .* 0.1f0
        B_f32 = randn(rng, Float32, K_test, N_total) .* 0.1f0

        A_packed = Array{UInt16}(undef, K_test, M_total)
        B_packed = Array{UInt16}(undef, K_test, N_total)
        for m in 1:M_total, k in 1:K_test
            A_packed[k, m] = bf16_bits(A_f32[m, k])
        end
        for k in 1:K_test, n in 1:N_total
            B_packed[k, n] = bf16_bits(B_f32[k, n])
        end
        A_d = CuArray(A_packed)
        B_d = CuArray(B_packed)
        # bf16 output buffer (N-innermost = (N, M) col-major Julia layout).
        C_d = CUDACore.zeros(UInt16, M_total * N_total)

        tmap_A = tensor_map_tile_2d(:bf16, pointer(A_d),
            M_total, K_test, PCT_BM_CTA, PCT_BK; swizzle = :B32)
        tmap_B = tensor_map_tile_2d(:bf16, pointer(B_d),
            N_total, K_test, PCT_BN, PCT_BK; swizzle = :B32)
        # TMA store descriptor. C_d is N-innermost (col-major `(N, M)`
        # in Julia), so N is the `cols` (innermost) axis of the
        # descriptor and M is `rows`. The tile is (box_rows=B_WG_M
        # M-rows, box_cols=BN N-cols) — same orientation. Linear (no
        # swizzle) since our consumer writes to SMEM C-tile linearly.
        tmap_C = tensor_map_tile_2d(:bf16, pointer(C_d),
            M_total, N_total, PCT_B_WG_M, PCT_BN; swizzle = :NONE)
        A = upload_tma_descriptor(tmap_A)
        B = upload_tma_descriptor(tmap_B)
        Cd = upload_tma_descriptor(tmap_C)

        num_n_tiles    = N_total ÷ PCT_BN
        num_m_clusters = M_total ÷ PCT_BM_CLUSTER
        grid = (PCT_CLUSTERS * num_n_tiles, num_m_clusters, 1)
        @cuda blocks = grid threads = PCT_THREADS clustersize = (PCT_CLUSTERS, 1, 1) feature_set = :arch _pct_gemm_kernel!(
            C_d, A.ptr, B.ptr, Cd.ptr, Int32(M_total), Int32(N_total), Int32(K_test))
        CUDACore.synchronize()

        # C_d is bf16 (UInt16) in N-innermost layout: (N, M) Julia col-major.
        C_packed = reshape(Array(C_d), N_total, M_total)
        C_got    = Array{Float32}(undef, M_total, N_total)
        for m in 1:M_total, n in 1:N_total
            C_got[m, n] = bf16_to_f32(C_packed[n, m])
        end
        C_ref = bf16_gemm_ref(A_f32, B_f32)
        return C_got, C_ref
    end

    @testset "cluster GEMM + TMA store (M=256, N=16, K=64) bf16 out" begin
        # Tolerance loosened slightly vs the f32-out variant because the
        # accumulator is rounded to bf16 at the very end (single rounding
        # vs the per-iter accumulation rounding hidden in the ref).
        C_got, C_ref = _run_pct_case(MersenneTwister(0x7ea), PCT_BM_CLUSTER, PCT_BN, 64)
        @test isapprox(C_got, C_ref; rtol = 5e-3, atol = 5e-3)
    end

    @testset "cluster GEMM + TMA store (multi-tile: M=512, N=32, K=64)" begin
        C_got, C_ref = _run_pct_case(MersenneTwister(0x9ea), 512, 32, 64)
        @test isapprox(C_got, C_ref; rtol = 5e-3, atol = 5e-3)
    end
end
