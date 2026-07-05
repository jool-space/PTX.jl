# 2-CTA cluster GEMM: producer + 2 consumers per CTA, multicast B-load
# across the cluster. Builds on gemm_pc_pipeline_split.jl by wrapping
# its 1-producer + 2-consumer M-split shape inside a 2-CTA cluster
# (CLUSTER_M=2, CLUSTER_N=1). The B tile is shared across both CTAs
# via multicast; A is M-split (each CTA loads its half).
#
# Composes the entire warp-spec stack short of the multi-tile iteration
# + Hilbert scheduler + TMA store epilogue:
#
#   - 2-CTA cluster (clustersize=(2,1,1), blocks=(2,1,1))
#   - Per CTA: 1 producer + 2 consumers (384 threads)
#   - Per CTA: handles BM_CTA=128 rows; total cluster tile is 256 × BN
#   - Per stage: full[s] mbar count=1, expected_tx = A_PER_CTA + B_BYTES
#               (multicast B delivers tx-bytes to each receiver's mbar)
#               empty[s] mbar count = NUM_CONSUMERS × CLUSTERS = 4
#
# mbar choreography:
#
#   Init (thread 0 per CTA):
#     - init full[s].count = 1, empty[s].count = 4 for each stage
#     - pre-arrive empty[s] four times to flip phase → 1
#     - fence.mbarrier_init.release.cluster makes init visible cluster-wide
#
#   Cluster barrier: cluster.sync (CUDACore intrinsic; chain-default's
#   asmcall would lose the convergent attribute and the rendezvous would
#   deadlock — see commit e5d6d0e).
#
#   Producer loop (per CTA, lane 0 of WG0 issues):
#     - wait empty[s] expected_phase
#     - arrive_expect_tx(full[s], LOAD_TX_BYTES)  [LOCAL mbar]
#     - local A load (shared::cta) at coord (k_off, cta_rank × BM_CTA)
#     - if cta_rank == 0: multicast B load (shared::cluster, mask=0b11)
#       — delivers B to both CTAs' SMEM and arrives on both full[s]
#
#   Consumer loop (per CTA, all threads of WG1/WG2):
#     - wait full[s] expected_phase
#     - wgmma m64n16k16 over local M-slice + shared B
#     - lane 0/1 of consumer wg: mapa.shared::cluster.u32(empty[s], lane128)
#       + mbarrier.arrive.shared::cluster.b64  →  cross-cluster release.
#       4 arrives per stage land on each CTA's empty[s] (2 consumer wgs
#       × 2 CTAs); phase flips when count=4 is reached.
#
# Runtime gated to clusters-capable arch (`[9.0, 12.0)`).

using PTX: layout_for_a, wgmma_descriptor, smem_addr_u32, tensor_map_tile_2d,
           step_desc
using PTX.MBarriers: BarrierArray,
                     barrier_wait, barrier_arrive_expect_tx, barrier_arrive_cluster
using PTX.Pipelines: Pipeline, pipeline_init!, pipeline_cursor
using CUDACore
# barrier.cluster via PTX tier-2 wrappers (CUDACore's cluster_arrive/wait
# ccall-by-name silently traps on Julia ≤ 1.11 — see wrappers/barrier_cluster.jl)
using Random

const PCC_BM_CTA     = 128
const PCC_BN         = 16
const PCC_BK         = 16
const PCC_N_STAGES   = 2
const PCC_NUM_CONSUMERS = 2
const PCC_CLUSTERS   = 2
const PCC_B_WG_M     = PCC_BM_CTA ÷ PCC_NUM_CONSUMERS    # 64 rows per consumer
const PCC_THREADS    = 128 * (1 + PCC_NUM_CONSUMERS)     # 384 per CTA
const PCC_BM_CLUSTER = PCC_BM_CTA * PCC_CLUSTERS          # 256 rows across cluster

const PCC_A_PER_CTA  = PCC_BM_CTA * PCC_BK * 2            # 4096 bytes
const PCC_B_BYTES    = PCC_BK     * PCC_BN * 2            # 512  bytes
const PCC_LOAD_BYTES = PCC_A_PER_CTA + PCC_B_BYTES        # 4608

const PCC_A_STAGE_BYTES = PCC_A_PER_CTA
const PCC_B_STAGE_BYTES = PCC_B_BYTES

const PCC_A_CONSUMER_BYTES = PCC_B_WG_M * PCC_BK * 2      # 2048

const PCC_EMPTY_COUNT = PCC_NUM_CONSUMERS * PCC_CLUSTERS  # 4

function _pcc_gemm_kernel!(
        C::CuDeviceVector{Float32, 1},
        tma_A::PTX.TMADescriptorPtr,
        tma_B::PTX.TMADescriptorPtr,
        M::Int32, N::Int32, K::Int32)

    smem_A = CuStaticSharedArray(UInt16, PCC_N_STAGES * PCC_BM_CTA * PCC_BK)
    smem_B = CuStaticSharedArray(UInt16, PCC_N_STAGES * PCC_BK    * PCC_BN)
    mbar_full  = CuStaticSharedArray(UInt64, PCC_N_STAGES)
    mbar_empty = CuStaticSharedArray(UInt64, PCC_N_STAGES)

    a_base_ptr = pointer(smem_A)
    b_base_ptr = pointer(smem_B)

    full  = BarrierArray{PCC_N_STAGES}(pointer(mbar_full))
    empty = BarrierArray{PCC_N_STAGES}(pointer(mbar_empty))

    tid      = ptx"mov.u32"(sreg"tid.x")
    cta_rank = ptx"mov.u32"(sreg"cluster_ctarank")
    ctaid_x  = ptx"mov.u32"(sreg"ctaid.x")
    ctaid_y  = ptx"mov.u32"(sreg"ctaid.y")
    wg_id    = tid >> UInt32(7)
    lane128  = tid & UInt32(127)

    # Cluster tile indices. CLUSTERS=2 in the X dimension, so the grid
    # has CLUSTERS × num_n_tiles blocks along X; each pair maps to one
    # N tile. M tiles fold straight onto Y (one cluster per M tile).
    tile_m_cluster = ctaid_y                           # 0-based cluster M-tile
    tile_n         = ctaid_x >> UInt32(1)              # 0-based N-tile

    # Global offsets for this CTA's A-slice and the shared B-tile.
    m_off_global = tile_m_cluster * UInt32(PCC_BM_CLUSTER) + cta_rank * UInt32(PCC_BM_CTA)
    n_off_global = tile_n * UInt32(PCC_BN)

    # ── Init mbarriers + pipeline warm-up (per CTA) ─────────────────
    if tid == UInt32(0)
        # Cluster scope: full pre-arrive count = NUM_CONSUMERS × CLUSTERS,
        # cluster-release fence (vs CTA-scope fence in single-CTA case).
        pipeline_init!(full, empty, Val(PCC_EMPTY_COUNT), Val(true))
    end
    ptx"bar.sync"(Val(0))               # CTA-local sync after init

    # Cluster-wide sync: every CTA's mbarriers visible to every CTA
    # before any cluster traffic (multicast TMA, cross-CTA mbarrier
    # arrives) starts. CUDACore intrinsic carries `convergent` —
    # chain-default `barrier.cluster.{arrive,wait}` doesn't and would
    # deadlock here.
    ptx"barrier.cluster.arrive"()
    ptx"barrier.cluster.wait"()

    num_k_tiles = K >> Int32(4)

    if wg_id == UInt32(0)
        # ─────────────────────── PRODUCER (WG 0) ─────────────────────
        ptx"setmaxnreg.dec.sync.aligned.u32"(Val(24))

        @inbounds for k_iter in Int32(0):(num_k_tiles - Int32(1))
            stage, phase = pipeline_cursor(Pipeline{PCC_N_STAGES}, k_iter)
            barrier_wait(empty[stage], phase)

            if lane128 == UInt32(0)
                ptx"fence.proxy.async.shared::cta"()
                barrier_arrive_expect_tx(full[stage], PCC_LOAD_BYTES)

                a_stage_ptr = a_base_ptr + Int(stage) * PCC_A_STAGE_BYTES
                b_stage_ptr = b_base_ptr + Int(stage) * PCC_B_STAGE_BYTES
                k_off  = k_iter * Int32(PCC_BK)

                # Local A load — each CTA fetches its M-slice of this
                # cluster's (M, K) tile.
                ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                    a_stage_ptr, tma_A, k_off, reinterpret(Int32, m_off_global), full[stage])

                # Multicast B load — CTA 0 issues, both CTAs receive.
                if cta_rank == UInt32(0)
                    ptx"cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"(
                        b_stage_ptr, tma_B, k_off,
                        reinterpret(Int32, n_off_global), full[stage], UInt16(0x3))
                end
            end
        end
    else
        # ─────────────────── CONSUMER (WG 1 or WG 2) ─────────────────
        ptx"setmaxnreg.inc.sync.aligned.u32"(Val(240))
        consumer_id = wg_id - UInt32(1)

        a_slice_ptr  = a_base_ptr + Int(consumer_id) * PCC_A_CONSUMER_BYTES
        a_slice_addr = smem_addr_u32(a_slice_ptr)
        b_addr       = smem_addr_u32(b_base_ptr)

        la = layout_for_a(dtype = :bf16, m = PCC_B_WG_M, k = PCC_BK)
        lb = layout_for_a(dtype = :bf16, m = PCC_BN,     k = PCC_BK)
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
            stage, phase = pipeline_cursor(Pipeline{PCC_N_STAGES}, k_iter)
            barrier_wait(full[stage], phase)

            a_desc = step_desc(a_desc_base, Int(stage) * PCC_A_STAGE_BYTES)
            b_desc = step_desc(b_desc_base, Int(stage) * PCC_B_STAGE_BYTES)
            ptx"fence.proxy.async.shared::cta"()
            ptx"wgmma.fence.sync.aligned"()
            d = ptx"wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16"(
                d, a_desc, b_desc, true)
            ptx"wgmma.commit_group.sync.aligned"()
            ptx"wgmma.wait_group.sync.aligned"(Val(0))

            # Cross-cluster release. Lanes 0..CLUSTERS-1 each arrive on
            # a specific CTA's empty[s] via mapa: lane k arrives on
            # CTA k's empty[s]. 2 lanes × 2 consumer wgs = 4 arrives per
            # CTA per stage = empty[s].count.
            if lane128 < UInt32(PCC_CLUSTERS)
                barrier_arrive_cluster(empty[stage], lane128)
            end
        end

        # ── Epilogue ─────────────────────────────────────────────────
        # C is (N, M) col-major (N-innermost). Linear offset for
        # (global_m, global_n) = global_m * N + global_n.
        # Per-consumer base global row: cluster m-base + per-CTA m-base
        # + per-consumer m-base (consumer 0 → first B_WG_M rows of the
        # CTA, consumer 1 → next B_WG_M rows).
        wid      = lane128 >> UInt32(5)
        lane     = lane128 & UInt32(31)
        frag_row = (wid << UInt32(4)) + (lane >> UInt32(2))
        frag_col = (lane & UInt32(3)) << UInt32(1)

        global_m_base = m_off_global + consumer_id * UInt32(PCC_B_WG_M)
        N_u32         = reinterpret(UInt32, N)
        c_base        = pointer(C) +
            Int(global_m_base * N_u32 + n_off_global) * sizeof(Float32)

        off_a1 = (frag_row * N_u32 + frag_col) * UInt32(4)
        off_b1 = ((frag_row + UInt32(8)) * N_u32 + frag_col) * UInt32(4)
        off_a2 = (frag_row * N_u32 + frag_col + UInt32(8)) * UInt32(4)
        off_b2 = ((frag_row + UInt32(8)) * N_u32 + frag_col + UInt32(8)) * UInt32(4)
        ptx"st.global.v2.f32"(c_base + Int(off_a1), (d[1], d[2]))
        ptx"st.global.v2.f32"(c_base + Int(off_b1), (d[3], d[4]))
        ptx"st.global.v2.f32"(c_base + Int(off_a2), (d[5], d[6]))
        ptx"st.global.v2.f32"(c_base + Int(off_b2), (d[7], d[8]))
    end

    return nothing
end

# ── Cross-arch ptxas validation ────────────────────────────────────────

@testset "cluster GEMM (producer + 2 consumers + multicast B) compiles at sm_90a" begin
    types = Tuple{CuDeviceVector{Float32, 1},
                  PTX.TMADescriptorPtr, PTX.TMADescriptorPtr,
                  Int32, Int32, Int32}
    @test ptxas_compiles(_pcc_gemm_kernel!, types;
                         cap = v"9.0", feature_set = :arch)

    ptx = emit_ptx(_pcc_gemm_kernel!, types; cap = v"9.0", feature_set = :arch)
    @test occursin("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster", ptx)
    @test occursin("mapa.shared::cluster.u32",          ptx)
    @test occursin("mbarrier.arrive.shared::cluster.b64", ptx)
    @test occursin("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16", ptx)
end

# ── Runtime — Hopper only (wgmma is sm_90a; Blackwell uses tcgen05) ─────

if v"9.0" <= DEV_CAP < v"10.0"
    # Helper: run the kernel for a given (M, N, K) shape and compare
    # against the bf16-rounded reference. M must be a multiple of
    # BM_CLUSTER, N a multiple of BN, K a multiple of BK.
    function _run_pcc_case(rng, M_total, N_total, K_test)
        @assert M_total % PCC_BM_CLUSTER == 0
        @assert N_total % PCC_BN         == 0
        @assert K_test  % PCC_BK         == 0

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

        # TMA covers the full M/N range; per-tile coord-offset selects the slice.
        tmap_A = tensor_map_tile_2d(:bf16, pointer(A_d),
            M_total, K_test, PCC_BM_CTA, PCC_BK; swizzle = :B32)
        tmap_B = tensor_map_tile_2d(:bf16, pointer(B_d),
            N_total, K_test, PCC_BN, PCC_BK; swizzle = :B32)
        A = upload_tma_descriptor(tmap_A)
        B = upload_tma_descriptor(tmap_B)

        # Grid: (CLUSTERS × num_n_tiles, num_m_clusters, 1). Each cluster
        # handles one (tile_m_cluster, tile_n).
        num_n_tiles    = N_total ÷ PCC_BN
        num_m_clusters = M_total ÷ PCC_BM_CLUSTER
        grid = (PCC_CLUSTERS * num_n_tiles, num_m_clusters, 1)

        C_d = CUDACore.zeros(Float32, M_total * N_total)
        @cuda blocks = grid threads = PCC_THREADS clustersize = (PCC_CLUSTERS, 1, 1) _pcc_gemm_kernel!(
            C_d, A.ptr, B.ptr, Int32(M_total), Int32(N_total), Int32(K_test))
        CUDACore.synchronize()

        # C_d is N-innermost: (N_total, M_total) Julia col-major.
        C_packed = reshape(Array(C_d), N_total, M_total)
        C_got    = Array{Float32}(undef, M_total, N_total)
        for m in 1:M_total, n in 1:N_total
            C_got[m, n] = C_packed[n, m]
        end
        C_ref = bf16_gemm_ref(A_f32, B_f32)
        return C_got, C_ref
    end

    @testset "cluster GEMM (single cluster-tile: M=256, N=16, K=64)" begin
        # 1×1 grid of cluster-tiles — same shape as the pre-multi-tile version.
        C_got, C_ref = _run_pcc_case(MersenneTwister(0xc1d5), PCC_BM_CLUSTER, PCC_BN, 64)
        @test isapprox(C_got, C_ref; rtol = 1e-3, atol = 1e-3)
    end

    @testset "cluster GEMM (multi-tile: M=512, N=32, K=64) random bf16" begin
        # 2 M-clusters × 2 N-tiles = 4 cluster-tiles in the grid. Validates
        # the per-cluster offset math (TMA coords + global C-write address).
        C_got, C_ref = _run_pcc_case(MersenneTwister(0xd1d5), 512, 32, 64)
        @test isapprox(C_got, C_ref; rtol = 1e-3, atol = 1e-3)
    end
end
