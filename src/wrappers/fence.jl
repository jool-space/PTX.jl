# Memory-ordering fences. The three proxy/init fences below are migrated to
# tier-2 intrinsic lowering (DESIGN.md): they fence a specific memory proxy
# (async, mbarrier-init), which a core-IR `fence` cannot express — so they
# ride `llvm.nvvm.fence.*` intrinsics, not LLVM's `fence` instruction. The
# generic memory fences (`fence.sc.*`, `fence.acq_rel.*`) are the tier-1
# core-IR case (`fence <ordering> syncscope(...)`, a semantic translation)
# and stay on the asm tier via the inst.jl chain default for now.
# PTX 9.2 §9.7.12.4.
#
# Literal nvvm"..." spellings so the conformance scan finds them
# (test/host/conformance.jl).

# `fence.proxy.async;` — async-proxy ↔ generic-proxy ordering fence. The
# async proxy is the memory proxy used by TMA (cp.async.bulk*) and tcgen05;
# generic-proxy ops (regular load/store, mma) need this fence to observe
# async-proxy writes (and vice versa). sm_90+ only.
@inline (::Operation{:fence, (:proxy, :async)})() =
    nvvm"fence.proxy.async"()

# `fence.proxy.async.shared::cta;` — the shared-CTA-scoped async proxy fence
# the producer/consumer GEMM pipelines emit after staging into shared memory
# (used across the Hopper/Blackwell kernels). Intrinsic spells the space
# with an underscore (shared_cta); the emitted PTX is `shared::cta`.
@inline (::Operation{:fence, (:proxy, :async, Symbol("shared::cta"))})() =
    nvvm"fence.proxy.async.shared_cta"()

# `fence.mbarrier_init.release.cluster;` — release-fence after
# `mbarrier.init` so other CTAs in the cluster observe the initialized
# mbarrier state before reading/arriving on it. sm_90+ (cluster scope).
@inline (::Operation{:fence, (:mbarrier_init, :release, :cluster)})() =
    nvvm"fence.mbarrier_init.release.cluster"()
