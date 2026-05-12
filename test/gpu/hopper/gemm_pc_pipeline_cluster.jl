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

using PTX: layout_for_a, wgmma_descriptor, smem_addr_u32, tensor_map_tile_2d
using CUDACore
using CUDACore: cluster_arrive, cluster_wait
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

const PCC_A_DESC_STEP = UInt64(PCC_A_STAGE_BYTES >> 4)
const PCC_B_DESC_STEP = UInt64(PCC_B_STAGE_BYTES >> 4)
const PCC_A_CONSUMER_BYTES = PCC_B_WG_M * PCC_BK * 2      # 2048

const PCC_EMPTY_COUNT = PCC_NUM_CONSUMERS * PCC_CLUSTERS  # 4

function _pcc_gemm_kernel!(
        C::CuDeviceVector{Float32, 1},
        tma_A::PTX.TMADescriptorPtr,
        tma_B::PTX.TMADescriptorPtr,
        K::Int32)

    smem_A = CuStaticSharedArray(UInt16, PCC_N_STAGES * PCC_BM_CTA * PCC_BK)
    smem_B = CuStaticSharedArray(UInt16, PCC_N_STAGES * PCC_BK    * PCC_BN)
    mbar_full  = CuStaticSharedArray(UInt64, PCC_N_STAGES)
    mbar_empty = CuStaticSharedArray(UInt64, PCC_N_STAGES)

    a_base_ptr    = pointer(smem_A)
    b_base_ptr    = pointer(smem_B)
    mb_full_base  = pointer(mbar_full)
    mb_empty_base = pointer(mbar_empty)

    tid      = ptx"mov.u32"(sreg"tid.x")
    cta_rank = ptx"mov.u32"(sreg"cluster_ctarank")
    wg_id    = tid >> UInt32(7)
    lane128  = tid & UInt32(127)

    # ── Init mbarriers + pipeline warm-up (per CTA) ─────────────────
    if tid == UInt32(0)
        @inbounds for s in 0:(PCC_N_STAGES - 1)
            full_s  = mb_full_base  + s * sizeof(UInt64)
            empty_s = mb_empty_base + s * sizeof(UInt64)
            ptx"mbarrier.init.shared.b64"(full_s,  1)
            ptx"mbarrier.init.shared.b64"(empty_s, PCC_EMPTY_COUNT)
            # 4 local pre-arrives → phase flips to 1 (count satisfied).
            ptx"mbarrier.arrive.shared.b64"(empty_s)
            ptx"mbarrier.arrive.shared.b64"(empty_s)
            ptx"mbarrier.arrive.shared.b64"(empty_s)
            ptx"mbarrier.arrive.shared.b64"(empty_s)
        end
        ptx"fence.mbarrier_init.release.cluster"()
    end
    ptx"bar.sync"(Val(0))               # CTA-local sync after init

    # Cluster-wide sync: every CTA's mbarriers visible to every CTA
    # before any cluster traffic (multicast TMA, cross-CTA mbarrier
    # arrives) starts. CUDACore intrinsic carries `convergent` —
    # chain-default `barrier.cluster.{arrive,wait}` doesn't and would
    # deadlock here.
    cluster_arrive()
    cluster_wait()

    num_k_tiles = K >> Int32(4)

    if wg_id == UInt32(0)
        # ─────────────────────── PRODUCER (WG 0) ─────────────────────
        ptx"setmaxnreg.dec.sync.aligned.u32"(Val(24))

        @inbounds for k_iter in Int32(0):(num_k_tiles - Int32(1))
            stage = k_iter & Int32(PCC_N_STAGES - 1)
            phase = UInt32((k_iter >> Int32(1)) & Int32(1))
            full_s  = mb_full_base  + Int(stage) * sizeof(UInt64)
            empty_s = mb_empty_base + Int(stage) * sizeof(UInt64)

            while !ptx"mbarrier.test_wait.parity.shared.b64"(empty_s, phase)
            end

            if lane128 == UInt32(0)
                ptx"fence.proxy.async.shared::cta"()
                ptx"mbarrier.arrive.expect_tx.shared.b64"(full_s, PCC_LOAD_BYTES)

                a_stage_ptr = a_base_ptr + Int(stage) * PCC_A_STAGE_BYTES
                b_stage_ptr = b_base_ptr + Int(stage) * PCC_B_STAGE_BYTES
                k_off  = k_iter * Int32(PCC_BK)
                m_off  = reinterpret(Int32, cta_rank * UInt32(PCC_BM_CTA))

                # Local A load — each CTA fetches its M-slice.
                ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                    a_stage_ptr, tma_A, k_off, m_off, full_s)

                # Multicast B load — CTA 0 issues, both CTAs receive.
                if cta_rank == UInt32(0)
                    ptx"cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"(
                        b_stage_ptr, tma_B, k_off, Int32(0), full_s, UInt16(0x3))
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
            stage = k_iter & Int32(PCC_N_STAGES - 1)
            phase = UInt32((k_iter >> Int32(1)) & Int32(1))
            full_s  = mb_full_base  + Int(stage) * sizeof(UInt64)
            empty_s = mb_empty_base + Int(stage) * sizeof(UInt64)

            while !ptx"mbarrier.test_wait.parity.shared.b64"(full_s, phase)
            end

            a_desc = a_desc_base + UInt64(stage) * PCC_A_DESC_STEP
            b_desc = b_desc_base + UInt64(stage) * PCC_B_DESC_STEP
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
                remote_empty = ptx"mapa.shared::cluster.u32"(empty_s, lane128)
                ptx"mbarrier.arrive.shared::cluster.b64"(remote_empty)
            end
        end

        # ── Epilogue ─────────────────────────────────────────────────
        # Each CTA writes its 128×16 f32 output. Global C is laid out as
        # (BN, BM_CLUSTER) col-major → BN-innermost. Per-CTA base offset
        # in C: cta_rank * BM_CTA rows × BN cols × 4 bytes/f32. Within
        # a CTA, consumer_id selects the upper or lower half of the M-slice.
        wid      = lane128 >> UInt32(5)
        lane     = lane128 & UInt32(31)
        frag_row = (wid << UInt32(4)) + (lane >> UInt32(2))
        frag_col = (lane & UInt32(3)) << UInt32(1)

        cta_base_off = cta_rank * UInt32(PCC_BM_CTA) * UInt32(PCC_BN) * UInt32(4)
        consumer_off = consumer_id * UInt32(PCC_B_WG_M) * UInt32(PCC_BN) * UInt32(4)
        c_base       = pointer(C) + Int(cta_base_off) + Int(consumer_off)

        off_a1 = (frag_row * UInt32(PCC_BN) + frag_col) * UInt32(4)
        off_b1 = ((frag_row + UInt32(8)) * UInt32(PCC_BN) + frag_col) * UInt32(4)
        off_a2 = (frag_row * UInt32(PCC_BN) + frag_col + UInt32(8)) * UInt32(4)
        off_b2 = ((frag_row + UInt32(8)) * UInt32(PCC_BN) + frag_col + UInt32(8)) * UInt32(4)
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
                  PTX.TMADescriptorPtr, PTX.TMADescriptorPtr, Int32}
    @test ptxas_compiles(_pcc_gemm_kernel!, types;
                         cap = v"9.0", feature_set = :arch)

    ptx = emit_ptx(_pcc_gemm_kernel!, types; cap = v"9.0", feature_set = :arch)
    @test occursin("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster", ptx)
    @test occursin("mapa.shared::cluster.u32",          ptx)
    @test occursin("mbarrier.arrive.shared::cluster.b64", ptx)
    @test occursin("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16", ptx)
end

# ── Runtime — clusters-capable arch only ───────────────────────────────

if v"9.0" <= DEV_CAP < v"12.0"
    @testset "cluster GEMM (BM=256, BN=16, K=64) random bf16" begin
        rng = MersenneTwister(0xc1d5)
        K_test = 64
        M_total = PCC_BM_CLUSTER       # 256 = 2 CTAs × 128 rows
        A_f32 = randn(rng, Float32, M_total, K_test) .* 0.1f0
        B_f32 = randn(rng, Float32, K_test, PCC_BN)  .* 0.1f0

        # K-fast packing: A as (K, M_total), B as (K, BN), both col-major.
        A_packed = Array{UInt16}(undef, K_test, M_total)
        B_packed = Array{UInt16}(undef, K_test, PCC_BN)
        for m in 1:M_total, k in 1:K_test
            A_packed[k, m] = bf16_bits(A_f32[m, k])
        end
        for k in 1:K_test, n in 1:PCC_BN
            B_packed[k, n] = bf16_bits(B_f32[k, n])
        end
        A_d = CuArray(A_packed)
        B_d = CuArray(B_packed)

        # TMA descriptors. A covers the full M_total; per-CTA tile is
        # (BM_CTA, BK) at offset (k_off, cta_rank × BM_CTA).
        tmap_A = tensor_map_tile_2d(:bf16, pointer(A_d),
            M_total, K_test, PCC_BM_CTA, PCC_BK; swizzle = :B32)
        tmap_B = tensor_map_tile_2d(:bf16, pointer(B_d),
            PCC_BN, K_test, PCC_BN, PCC_BK; swizzle = :B32)
        A = upload_tma_descriptor(tmap_A)
        B = upload_tma_descriptor(tmap_B)

        C_d = CUDACore.zeros(Float32, M_total * PCC_BN)
        @cuda blocks = (2, 1, 1) threads = PCC_THREADS clustersize = (2, 1, 1) feature_set = :arch _pcc_gemm_kernel!(
            C_d, A.ptr, B.ptr, Int32(K_test))
        CUDACore.synchronize()

        # Output is (BN, BM_CLUSTER) BN-fast.
        C_packed = reshape(Array(C_d), PCC_BN, M_total)
        C_got    = Array{Float32}(undef, M_total, PCC_BN)
        for m in 1:M_total, n in 1:PCC_BN
            C_got[m, n] = C_packed[n, m]
        end

        C_ref = bf16_gemm_ref(A_f32, B_f32)
        @test isapprox(C_got, C_ref; rtol = 1e-3, atol = 1e-3)
    end
end
