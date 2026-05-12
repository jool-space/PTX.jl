# REQUIRES CC 9.0
#
# Compile-only smoke test for the TMA multicast load wrapper
# `cp.async.bulk.tensor.<N>d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster`.
#
# Multicast TMA distributes a single global read across the cluster CTAs whose
# bits are set in the u16 mask, then each CTA's local mbarrier arrives once
# the load completes. Gating item for `pyptx/examples/hopper/gemm_highperf_hopper.py`
# (2-CTA cluster, ~815 TFLOPS @ 8192³ bf16 on H100).
#
# A full runtime cluster-launch test requires CUDA.jl cluster-launch plumbing
# we don't yet exercise in this suite — covered here at the ptxas level only.
# Once the high-perf GEMM port lands, that exercises the runtime path
# end-to-end.

using PTX: AS

function _tma_multicast_kernel!(out::CuDeviceVector{UInt16, 1},
                                tma_src::Core.LLVMPtr{UInt8, AS.Const})
    smem = CuStaticSharedArray(UInt16, 64)       # 8 × 8 bf16
    mbar = CuStaticSharedArray(UInt64, 1)
    mb_ptr = pointer(mbar)
    s_ptr  = pointer(smem)

    tid = ptx"mov.u32"(sreg"tid.x")

    # Cluster-rank-0 CTA issues the multicast load; mask = 0b11 means both
    # CTAs in a CLUSTER_M=2 cluster receive the tile.
    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mb_ptr, UInt32(1))
        ptx"fence.proxy.async.shared::cta"()
        ptx"mbarrier.arrive.expect_tx.shared.b64"(mb_ptr, UInt32(128))
        ptx"cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"(
            s_ptr, tma_src, Int32(0), Int32(0), mb_ptr, UInt16(0x3))
    end
    ptx"bar.sync"(Val(0))

    while !ptx"mbarrier.test_wait.parity.shared.b64"(mb_ptr, UInt32(0))
    end

    if tid < UInt32(64)
        @inbounds out[Int(tid) + 1] = smem[Int(tid) + 1]
    end
    return nothing
end

@testset "TMA multicast load — ptxas sm_90a" begin
    types = Tuple{CuDeviceVector{UInt16, 1}, Core.LLVMPtr{UInt8, AS.Const}}
    @test ptxas_compiles(_tma_multicast_kernel!, types;
                         cap = v"9.0", feature_set = :arch)
end
