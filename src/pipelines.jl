# N-stage producer/consumer ring built on top of `PTX.MBarriers`.
# Mirrors `<cuda/pipeline>` in scope: a phantom `Pipeline{N}` type, the
# `(stage, phase) = cursor(P, k_iter)` arithmetic, and the thread-0 init
# ritual that every Hopper warp-spec kernel repeats verbatim.
#
# Pure leaf module: depends on `PTX.MBarriers` for the verbs +
# `BarrierArray` storage type, but no method dispatches on a Barrier
# type — Pipelines just calls verb functions and treats `BarrierArray`
# as an indexable pointer.

module Pipelines

using ..PTX: @ptx_str
using ..MBarriers: BarrierArray, barrier_init, barrier_arrive
using Republic: @public

@public Pipeline,
        pipeline_stage, pipeline_phase, pipeline_cursor,
        pipeline_init!

# Phantom type carrying the stage count for cursor arithmetic. Use
# `pipeline_cursor(Pipeline{N_STAGES}, k_iter)` to get `(stage, phase)`.
struct Pipeline{N_STAGES} end

# Stage cursor: `k_iter mod N`. Compile-time-constant divisor → LLVM
# strength-reduces to `& (N-1)` for power-of-two N and to a multiply-
# and-shift otherwise.
@inline pipeline_stage(::Type{Pipeline{N}}, k_iter::Integer) where N =
    Int32(Int32(k_iter) % Int32(N))

# Phase parity: which round through this stage's mbarrier we're on.
# `(k_iter ÷ N) & 1`. For pow2 N this folds to a single shift; for
# non-pow2 N (e.g. N=3, the pyptx headline config) it lowers to a
# divide-by-constant which LLVM strength-reduces.
@inline pipeline_phase(::Type{Pipeline{N}}, k_iter::Integer) where N =
    UInt32((Int32(k_iter) ÷ Int32(N)) & Int32(1))

@inline pipeline_cursor(P::Type{Pipeline{N}}, k_iter::Integer) where N =
    (pipeline_stage(P, k_iter), pipeline_phase(P, k_iter))

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

end # module Pipelines
