# Verb-style API over the raw `mbarrier.*` wrappers, plus the
# `BarrierArray{N}` storage type. Mirrors `<cuda/barrier>` in scope:
# init / arrive / arrive_expect_tx / wait / cluster-arrive, with the
# arrive variants covering producer (expect_tx) and consumer (plain
# arrive) sides.
#
# Sister module `PTX.Pipelines` builds the N-stage producer/consumer
# ring on top of these — mirrors `<cuda/pipeline>`.
#
# Contents are plain sm_90 (mbarrier, mapa, fence). Kernels that use
# them only become arch-specific via what other ops they pull in
# (e.g. wgmma → sm_90a, tcgen05.mma → sm_100a).

module MBarriers

using ..PTX: AS, @ptx_str
using Republic: @public

@public BarrierArray,
        barrier_init, barrier_arrive, barrier_arrive_expect_tx,
        barrier_wait, barrier_arrive_cluster

# Typed SMEM mbarrier array. `N` is the slot count; `base` points at
# the first of N consecutive 8-byte mbarriers. Indexing returns the
# raw shared-AS pointer expected by all mbarrier ops (and by TMA copies
# that take a completion mbarrier), so callers can pass `full[stage]`
# directly to wrappers without an unwrap step.
struct BarrierArray{N}
    base::Core.LLVMPtr{UInt64, AS.Shared}
end

@inline Base.getindex(ba::BarrierArray, stage::Integer) =
    ba.base + Int(stage) * sizeof(UInt64)

# ── mbarrier verb-style API ────────────────────────────────────────────

@inline barrier_init(mbar::Core.LLVMPtr{UInt64, AS.Shared}, count::Integer) =
    ptx"mbarrier.init.shared.b64"(mbar, UInt32(count))

@inline barrier_arrive(mbar::Core.LLVMPtr{UInt64, AS.Shared}) =
    ptx"mbarrier.arrive.shared.b64"(mbar)

@inline barrier_arrive_expect_tx(mbar::Core.LLVMPtr{UInt64, AS.Shared},
                                  tx::Integer) =
    ptx"mbarrier.arrive.expect_tx.shared.b64"(mbar, UInt32(tx))

# Spin on test_wait.parity until the named phase has been observed.
# Lowers to a `setp ; @!pred bra` loop — identical to the hand-rolled
# `while !test_wait_parity ... end` callers were writing.
@inline function barrier_wait(mbar::Core.LLVMPtr{UInt64, AS.Shared},
                              phase::Integer)
    while !ptx"mbarrier.test_wait.parity.shared.b64"(mbar, UInt32(phase))
    end
    nothing
end

# Cluster-scope arrive: translate the local mbarrier address through
# `mapa.shared::cluster` to the remote CTA's view, then arrive on it.
# The mbarrier itself lives in some CTA's SMEM; mapa rebases the SMEM
# offset to a cluster-mapped address so the remote CTA's hardware sees
# the arrival.
@inline function barrier_arrive_cluster(mbar::Core.LLVMPtr{UInt64, AS.Shared},
                                         remote_rank::Integer)
    remote = ptx"mapa.shared::cluster.u32"(mbar, UInt32(remote_rank))
    ptx"mbarrier.arrive.shared::cluster.b64"(remote)
    nothing
end

end # module MBarriers
