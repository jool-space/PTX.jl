# REQUIRES CC 9.0
#
# Runtime cluster-launch round-trip for TMA multicast:
#
#   - `clustersize=(2, 1, 1)` — 2-CTA Hopper cluster
#   - Each CTA inits a local mbarrier with `count=1`, then a cluster barrier
#     pair (`barrier.cluster.arrive` / `barrier.cluster.wait` via CUDACore's
#     `cluster_arrive` / `cluster_wait` LLVM intrinsics) makes that init
#     visible across the cluster before any CTA reads the mbarrier.
#   - Each CTA arms `arrive.expect_tx(128)` on its own mbarrier.
#   - CTA rank 0 issues a single `cp.async.bulk.tensor.2d ... .multicast::cluster`
#     load with `mask = 0b11`; the hardware broadcasts the 128-byte tile to
#     both CTAs' SMEM at the same offset and arrives on both CTAs' mbarriers.
#   - Each CTA spins on `mbarrier.test_wait.parity`, then writes its 64-cell
#     SMEM tile to its half of the 128-element output vector. Both halves
#     must equal the input — failure on either half indicates the multicast
#     fan-out is broken.

using PTX: AS
using CUDACore: cluster_arrive, cluster_wait

function _tma_multicast_cluster_kernel!(out::CuDeviceVector{UInt16, 1},
                                        tma_src::Core.LLVMPtr{UInt8, AS.Const})
    smem = CuStaticSharedArray(UInt16, 64)       # 8 × 8 bf16 = 128 B
    mbar = CuStaticSharedArray(UInt64, 1)
    mb_ptr = pointer(mbar)
    s_ptr  = pointer(smem)

    tid = ptx"mov.u32"(sreg"tid.x")
    cta_rank = ptx"mov.u32"(sreg"cluster_ctarank")

    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mb_ptr, UInt32(1))
        ptx"fence.proxy.async.shared::cta"()
    end
    ptx"bar.sync"(Val(0))

    cluster_arrive()
    cluster_wait()

    if tid == UInt32(0)
        ptx"mbarrier.arrive.expect_tx.shared.b64"(mb_ptr, UInt32(128))
        if cta_rank == UInt32(0)
            ptx"cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"(
                s_ptr, tma_src, Int32(0), Int32(0), mb_ptr, UInt16(0x3))
        end
    end
    ptx"bar.sync"(Val(0))

    while !ptx"mbarrier.test_wait.parity.shared.b64"(mb_ptr, UInt32(0))
    end

    if tid < UInt32(64)
        @inbounds out[Int(cta_rank) * 64 + Int(tid) + 1] = smem[Int(tid) + 1]
    end
    return nothing
end

@testset "TMA multicast — ptxas sm_90a (cluster kernel)" begin
    types = Tuple{CuDeviceVector{UInt16, 1}, Core.LLVMPtr{UInt8, AS.Const}}
    @test ptxas_compiles(_tma_multicast_cluster_kernel!, types;
                         cap = v"9.0", feature_set = :arch)
end

# Runtime cluster-launch path — Hopper only. Blackwell (sm_100/sm_120/sm_121)
# either retired multi-CTA clusters or restricts size in ways that don't
# match this 2-CTA shape (GB10 rejects with ERROR_INVALID_CLUSTER_SIZE).
if v"9.0" <= DEV_CAP < v"10.0"
    @testset "TMA multicast cluster round-trip" begin
        input_vals = UInt16[(0x3f80 + i) for i in 0:63]
        src = CuArray(reshape(input_vals, 8, 8))

        tmap_host = PTX.tensor_map_tile_2d(:bf16, pointer(src), 8, 8, 8, 8;
                                           swizzle = :NONE)
        tmap_dev = CuArray{UInt8}(undef, 128)
        copyto!(tmap_dev, collect(tmap_host.data))

        out = CUDACore.zeros(UInt16, 128)
        src_const = reinterpret(Core.LLVMPtr{UInt8, AS.Const},
                                UInt64(pointer(tmap_dev)))

        @cuda threads = 128 clustersize = (2, 1, 1) _tma_multicast_cluster_kernel!(out, src_const)
        CUDACore.synchronize()

        result = Array(out)
        @test result[1:64] == input_vals
        @test result[65:128] == input_vals
    end
end
