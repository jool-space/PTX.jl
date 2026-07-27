"""
Verb-style API over the raw `mbarrier.*` wrappers, plus the
[`BarrierArray`](@ref MBarriers.BarrierArray) storage type. Mirrors
`<cuda/barrier>` in scope: init / arrive / arrive_expect_tx / wait /
cluster-arrive, with the arrive variants covering producer (expect_tx)
and consumer (plain arrive) sides.

Sister module `PTX.Pipelines` builds the N-stage producer/consumer
ring on top of these — mirrors `<cuda/pipeline>`.

Contents are plain sm_90 (mbarrier, mapa, fence). Kernels that use
them only become arch-specific via what other ops they pull in
(e.g. wgmma → sm_90a, tcgen05.mma → sm_100a).
"""
module MBarriers

using ..PTX: AS, @ptx_str
using Republic: @public

@public BarrierArray,
        barrier_init, barrier_arrive, barrier_arrive_expect_tx,
        barrier_wait, barrier_try_wait, barrier_arrive_cluster

"""
    BarrierArray{N}(base::Core.LLVMPtr{UInt64, AS.Shared})

Typed SMEM mbarrier array. `N` is the slot count; `base` points at
the first of N consecutive 8-byte mbarriers. Indexing (0-based stage)
returns the raw shared-AS pointer expected by all mbarrier ops (and by
TMA copies that take a completion mbarrier), so callers can pass
`full[stage]` directly to wrappers without an unwrap step.
"""
struct BarrierArray{N}
    base::Core.LLVMPtr{UInt64, AS.Shared}
end

@inline Base.getindex(ba::BarrierArray, stage::Integer) =
    ba.base + Int(stage) * sizeof(UInt64)

# ── mbarrier verb-style API ────────────────────────────────────────────

"""
    barrier_init(mbar, count)

Initialize an SMEM mbarrier with an arrival `count` —
`mbarrier.init.shared.b64`.
"""
@inline barrier_init(mbar::Core.LLVMPtr{UInt64, AS.Shared}, count::Integer) =
    ptx"mbarrier.init.shared.b64"(mbar, UInt32(count))

"""
    barrier_arrive(mbar)

Consumer-side arrival — `mbarrier.arrive.shared.b64`. Returns the
`UInt64` state token.
"""
@inline barrier_arrive(mbar::Core.LLVMPtr{UInt64, AS.Shared}) =
    ptx"mbarrier.arrive.shared.b64"(mbar)

"""
    barrier_arrive_expect_tx(mbar, tx)

Producer-side arrival declaring `tx` expected transaction bytes —
`mbarrier.arrive.expect_tx.shared.b64`. Pair with TMA copies whose
completion mechanism is `mbarrier::complete_tx::bytes`.
"""
@inline barrier_arrive_expect_tx(mbar::Core.LLVMPtr{UInt64, AS.Shared},
                                  tx::Integer) =
    ptx"mbarrier.arrive.expect_tx.shared.b64"(mbar, UInt32(tx))

"""
    barrier_wait(mbar, phase)

Spin on `mbarrier.test_wait.parity` until the named phase has been
observed. Lowers to a `setp ; @!pred bra` loop — identical to the
hand-rolled `while !test_wait_parity ... end` callers were writing.

`test_wait` vs `try_wait` are NOT interchangeable: test_wait is a
non-suspending poll; try_wait may park the thread until a scheduler
wake. The Blackwell tcgen05 commit-drain idiom is specifically
`try_wait.parity`, and casually swapping the two — or the phase
literal — is a known hang footgun. Hence two distinct verbs
([`barrier_try_wait`](@ref) is the other), not one with a flag: when
porting, pick the one the source kernel uses, verbatim.
"""
@inline function barrier_wait(mbar::Core.LLVMPtr{UInt64, AS.Shared},
                              phase::Integer)
    while !ptx"mbarrier.test_wait.parity.shared.b64"(mbar, UInt32(phase))
    end
    nothing
end

"""
    barrier_try_wait(mbar, phase)

Spin on `mbarrier.try_wait.parity` (the potentially-suspending form)
until `phase` is observed — the tcgen05 `commit → mbarrier::arrive::one`
drain. Same loop shape as [`barrier_wait`](@ref); lowers byte-identically
to the hand-rolled `while !try_wait_parity ... end` the blackwell
kernels were writing.
"""
@inline function barrier_try_wait(mbar::Core.LLVMPtr{UInt64, AS.Shared},
                                  phase::Integer)
    while !ptx"mbarrier.try_wait.parity.shared.b64"(mbar, UInt32(phase))
    end
    nothing
end

"""
    barrier_arrive_cluster(mbar, remote_rank)

Cluster-scope arrive: translate the local mbarrier address through
`mapa.shared::cluster` to the remote CTA's view, then arrive on it.
The mbarrier itself lives in some CTA's SMEM; mapa rebases the SMEM
offset to a cluster-mapped address so the remote CTA's hardware sees
the arrival.
"""
@inline function barrier_arrive_cluster(mbar::Core.LLVMPtr{UInt64, AS.Shared},
                                         remote_rank::Integer)
    remote = ptx"mapa.shared::cluster.u32"(mbar, UInt32(remote_rank))
    ptx"mbarrier.arrive.shared::cluster.b64"(remote)
    nothing
end

end # module MBarriers
