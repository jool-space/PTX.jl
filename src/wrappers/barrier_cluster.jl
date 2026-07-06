# `barrier.cluster.{arrive,wait}` (PTX 9.2 §9.7.12.3), sm_90+ — first
# slice of the barrier-family tier-2 migration (the CTA-scope slice
# followed 2026-07-06; see wrappers/barrier.jl).
#
# Two reasons this family doesn't ride the chain default (asm):
#   - The intrinsics carry `convergent` from the registry; the chain's
#     sideeffect + ~{memory} asm does not, and a merged/duplicated
#     execution barrier is the worst instance of the collective-op
#     convergence hazard class.
#   - CUDACore's own cluster_arrive/cluster_wait use
#     `ccall("llvm.nvvm.barrier.cluster.*", llvmcall, ...)`, which
#     resolves the name against the IN-PROCESS LLVM's intrinsic table —
#     on Julia ≤ 1.11 (LLVM ≤ 16, which predates the cluster barrier
#     intrinsics) the call is demoted to a runtime trap, silently. The
#     tier-2 declare+call form passes unknown names through to the
#     external backend on every version (verified 2026-07-05).
#
# `.aligned` asserts all warps of all CTAs execute the same instruction;
# `.relaxed` drops the release/acquire ordering of arrive.
#
# Methods are written out literally (no name-building loop) so every
# intrinsic this file stands on is greppable — test/host/conformance.jl
# scans for `nvvm"..."` literals and requires a probe for each.

@inline (::Operation{:barrier, (:cluster, :arrive)})() =
    nvvm"barrier.cluster.arrive"()
@inline (::Operation{:barrier, (:cluster, :arrive, :relaxed)})() =
    nvvm"barrier.cluster.arrive.relaxed"()
@inline (::Operation{:barrier, (:cluster, :wait)})() =
    nvvm"barrier.cluster.wait"()

@inline (::Operation{:barrier, (:cluster, :arrive, :aligned)})() =
    nvvm"barrier.cluster.arrive.aligned"()
@inline (::Operation{:barrier, (:cluster, :arrive, :relaxed, :aligned)})() =
    nvvm"barrier.cluster.arrive.relaxed.aligned"()
@inline (::Operation{:barrier, (:cluster, :wait, :aligned)})() =
    nvvm"barrier.cluster.wait.aligned"()
