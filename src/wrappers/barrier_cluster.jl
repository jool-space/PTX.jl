# `barrier.cluster.{arrive,wait}` (PTX 9.2 §9.7.12.3), sm_90+ — the
# cluster-scope sibling of the CTA-scope tier-2 family in
# wrappers/barrier.jl.
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
#     external backend on every version.
#
# `.aligned` asserts all warps of all CTAs execute the same instruction;
# `.relaxed` drops the release/acquire ordering of arrive.
#
# Methods are written out literally (no name-building loop) so every
# intrinsic this file stands on is greppable — test/host/conformance.jl
# scans for `nvvm"..."` literals and requires a probe for each.

@inline optype"barrier.cluster.arrive"() =
    ceiled(nvvm"barrier.cluster.arrive", ptx"barrier.cluster.arrive")()
@inline optype"barrier.cluster.arrive.relaxed"() =
    ceiled(nvvm"barrier.cluster.arrive.relaxed",
           ptx"barrier.cluster.arrive.relaxed")()
@inline optype"barrier.cluster.wait"() =
    ceiled(nvvm"barrier.cluster.wait", ptx"barrier.cluster.wait")()

@inline optype"barrier.cluster.arrive.aligned"() =
    ceiled(nvvm"barrier.cluster.arrive.aligned",
           ptx"barrier.cluster.arrive.aligned")()
@inline optype"barrier.cluster.arrive.relaxed.aligned"() =
    ceiled(nvvm"barrier.cluster.arrive.relaxed.aligned",
           ptx"barrier.cluster.arrive.relaxed.aligned")()
@inline optype"barrier.cluster.wait.aligned"() =
    ceiled(nvvm"barrier.cluster.wait.aligned",
           ptx"barrier.cluster.wait.aligned")()
