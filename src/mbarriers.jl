# mbarrier-driven producer/consumer pipeline scaffolding for sm_90+
# kernels. Wraps three patterns that every Hopper warp-spec kernel
# repeats verbatim:
#
#   1. SMEM mbarrier array indexed by stage (byte arithmetic on the base
#      pointer).
#   2. The `stage = k & (N-1)`, `phase = (k >> log2 N) & 1` cursor.
#   3. The thread-0 init ritual: init(full, 1) + init(empty, count) +
#      `count` pre-arrives on empty + a CTA-or-cluster-scope fence.
#
# The verb-style API (`barrier_init`, `barrier_arrive`, `barrier_wait`,
# `barrier_arrive_expect_tx`, `barrier_arrive_cluster`) is a thin layer
# over the raw `ptx"..."` mbarrier wrappers — it exists so callers don't
# have to remember which of the seven mbarrier ops needs a phase, a state
# token, or an expect-tx count.
#
# Lives in a submodule because it's a composition layer above the
# `src/wrappers/*.jl` 1:1 PTX bindings, not a wrapper itself. The
# contents are plain sm_90 (mbarrier, mapa, fence) — kernels that use
# them only become arch-specific via what other ops they pull in
# (e.g. wgmma → sm_90a, tcgen05.mma → sm_100a).

module MBarriers

using ..PTX: AS, @ptx_str
using Republic: @public

@public BarrierArray, Pipeline,
        pipeline_stage, pipeline_phase, pipeline_cursor,
        pipeline_init!,
        barrier_init, barrier_arrive, barrier_arrive_expect_tx,
        barrier_wait, barrier_arrive_cluster

# Typed SMEM mbarrier array. `N` is the stage count; `base` points at
# the first of N consecutive 8-byte mbarriers. Indexing returns the
# raw shared-AS pointer expected by all mbarrier ops (and by TMA copies
# that take a completion mbarrier), so callers can pass `full[stage]`
# directly to wrappers without an unwrap step.
struct BarrierArray{N}
    base::Core.LLVMPtr{UInt64, AS.Shared}
end

@inline Base.getindex(ba::BarrierArray, stage::Integer) =
    ba.base + Int(stage) * sizeof(UInt64)

# Phantom type carrying the stage count for cursor arithmetic. Use
# `pipeline_cursor(Pipeline{N_STAGES}, k_iter)` to get `(stage, phase)`.
struct Pipeline{N_STAGES} end

@inline pipeline_stage(::Type{Pipeline{N}}, k_iter::Integer) where N =
    Int32(k_iter) & Int32(N - 1)

# log2(N) lifted to a compile-time shift via @generated. N must be a
# power of two; the construction is the standard ring-buffer phase trick.
@generated function pipeline_phase(::Type{Pipeline{N}}, k_iter::Integer) where N
    @assert ispow2(N) "Pipeline N_STAGES must be a power of two (got $N)"
    lg = trailing_zeros(N)
    :(UInt32((Int32(k_iter) >> Int32($lg)) & Int32(1)))
end

@inline pipeline_cursor(P::Type{Pipeline{N}}, k_iter::Integer) where N =
    (pipeline_stage(P, k_iter), pipeline_phase(P, k_iter))

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

# ── Pipeline init ──────────────────────────────────────────────────────
#
# Initializes the producer/consumer mbarrier pair for an N-stage ring
# and pre-arrives `empty[s]` once per consumer so the producer's first
# wait succeeds without a prior consumer arrival.
#
# - `full`  : mbarriers consumers wait on after the producer finishes a
#             stage. Init count = 1 (the producer's expect_tx arrive
#             flips the phase by itself).
# - `empty` : mbarriers the producer waits on after consumers finish a
#             stage. Init count = `EMPTY_COUNT` (one arrive per
#             releasing party per stage).
# - `EMPTY_COUNT` is the per-stage release count: NUM_CONSUMERS for a
#   single CTA, NUM_CONSUMERS × CLUSTERS for a clustered pipeline.
# - `CLUSTER=true` emits `fence.mbarrier_init.release.cluster` (visible
#   cluster-wide) instead of the CTA-scope `fence.proxy.async.shared::cta`.
#
# Caller responsibility:
#   - Invoke from a single thread (typically thread 0).
#   - Follow with `bar.sync(0)` to publish init within the CTA.
#   - If `CLUSTER=true`, follow with `cluster_arrive(); cluster_wait()`
#     (use the CUDACore intrinsics — see commit e5d6d0e).
#
# Body is fully unrolled across stages and pre-arrives via @generated;
# the emitted PTX matches the hand-rolled version byte-for-byte.
@generated function pipeline_init!(full::BarrierArray{N},
                                    empty::BarrierArray{N},
                                    ::Val{EMPTY_COUNT},
                                    ::Val{CLUSTER}) where {N, EMPTY_COUNT, CLUSTER}
    body = Expr(:block)
    # Inline so the unrolled mbarrier sequence lands directly in the
    # caller — otherwise LLVM keeps it as an extern call and the
    # `~{memory}` clobbers on each asm op don't compose with the
    # caller's `bar.sync`/`cluster_arrive`.
    push!(body.args, Expr(:meta, :inline))
    for s in 0:(N - 1)
        push!(body.args, :(barrier_init(full[$s], UInt32(1))))
        push!(body.args, :(barrier_init(empty[$s], UInt32($EMPTY_COUNT))))
        for _ in 1:EMPTY_COUNT
            push!(body.args, :(barrier_arrive(empty[$s])))
        end
    end
    push!(body.args, CLUSTER ? :(ptx"fence.mbarrier_init.release.cluster"()) :
                                :(ptx"fence.proxy.async.shared::cta"()))
    push!(body.args, :(nothing))
    body
end

end # module MBarriers
