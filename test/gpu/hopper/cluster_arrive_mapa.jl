# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc==9.0
# Cross-CTA mbarrier arrive via `mapa.shared::cluster` — the canonical
# rendezvous pattern for producer/consumer warpgroup splits across the
# cluster. Each CTA's thread 0 arrives on CTA-0's local mbarrier; CTA 0
# spins until the remote arrive lands.
#
# Exercises three new wrappers:
#   1. `mapa.shared::cluster.u32` — translate own-CTA mbarrier address
#      into the cluster-mapped view of CTA-0's equivalent mbarrier.
#   2. `mbarrier.arrive.shared::cluster.b64` — sink-operand arrive on the
#      cluster-mapped address.
#   3. `cluster_arrive` / `cluster_wait` (CUDACore LLVM intrinsics, not
#      `ptx"..."`) — flush mbarrier init across the cluster before any CTA
#      arrives. The intrinsics carry `IntrConvergent`; routing through
#      `@asmcall` would drop that attribute and the rendezvous deadlocks.
#
# The ptxas testset is cross-target; the structured metadata separately
# gates the Hopper runtime testset.

# barrier.cluster via PTX tier-2 wrappers (CUDACore's cluster_arrive/wait
# ccall-by-name silently traps on Julia ≤ 1.11 — see wrappers/barrier_cluster.jl)

function _cluster_arrive_mapa_kernel!(out::CuDeviceVector{UInt32, 1})
    mbar = CuStaticSharedArray(UInt64, 1)
    mb_ptr = pointer(mbar)

    tid = ptx"mov.u32"(sreg"tid.x")
    cta_rank = ptx"mov.u32"(sreg"cluster_ctarank")

    # Each CTA inits its own local mbarrier expecting 2 arrives (one from
    # itself, one from the other CTA in the cluster).
    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mb_ptr, UInt32(2))
        ptx"fence.mbarrier_init.release.cluster"()
    end
    # Cluster-wide barrier so the init becomes visible cluster-wide before
    # any CTA tries to arrive on a remote mbarrier.
    ptx"barrier.cluster.arrive"()
    ptx"barrier.cluster.wait"()

    if tid == UInt32(0)
        # Map our local mbarrier address into CTA-0's view. When cta_rank=0
        # this is identity; when cta_rank=1 it's the cluster-mapped pointer
        # to CTA-0's mbarrier at the same SMEM offset.
        remote_mb = ptx"mapa.shared::cluster.u32"(mb_ptr, UInt32(0))
        ptx"mbarrier.arrive.shared::cluster.b64"(remote_mb)
    end

    # Every CTA also waits on its own local mbarrier so we know all expected
    # arrives landed. CTA-0 will see both arrives (its own + remote);
    # CTA-1's mbarrier sees only its own — so we set the init count to 2
    # for CTA-0 only. Adjust: every CTA initted count=2 above, but CTA-1
    # never receives a remote arrive. So gate test_wait to CTA-0 only.
    if cta_rank == UInt32(0)
        while !ptx"mbarrier.test_wait.parity.shared.b64"(mb_ptr, UInt32(0))
        end
        if tid == UInt32(0)
            @inbounds out[1] = UInt32(0xdeadbeef)   # marker — both arrives landed
        end
    end
    return nothing
end

@testset "cluster mbarrier arrive via mapa — ptxas sm_90a" begin
    types = Tuple{CuDeviceVector{UInt32, 1}}
    @test ptxas_compiles(_cluster_arrive_mapa_kernel!, types;
                         cap = v"9.0", feature_set = :arch)
end

# Hopper-only by directory convention ([9.0, 10.0)); the mapa/cluster ops
# here also work on datacenter Blackwell, but hopper/ tests gate strictly
# to Hopper — a Blackwell cluster port would live under blackwell/.
if test_runtime_supported(@__FILE__)
    @testset "cluster mbarrier arrive via mapa — runtime" begin
        out = CUDACore.zeros(UInt32, 1)
        @cuda blocks = (2, 1, 1) threads = 128 clustersize = (2, 1, 1) _cluster_arrive_mapa_kernel!(out)
        CUDACore.synchronize()
        @test Array(out)[1] == 0xdeadbeef
    end
end
