# Memory-ordering fences, two lowering tiers. PTX 9.2 §9.7.12.4.
#
# The three proxy/init fences are tier-2 intrinsics: they fence
# a specific memory *proxy* (async, mbarrier-init), which a core-IR `fence`
# cannot express — so they ride `llvm.nvvm.fence.*` intrinsics. Literal
# nvvm"..." spellings so the conformance scan finds them
# (test/host/conformance.jl).
#
# The generic memory fences (`fence.sc.*`, `fence.acq_rel.*`) are TIER-1
# core IR: `fence <ordering> syncscope(...)`. This is a *semantic*
# translation of the PTX sem/scope pair, not a renaming — each mapping below
# was verified by trial compilation through the artifact llc 22.1.7
# (expected instruction asserted, not just acceptance):
#
#   PTX sem    LLVM ordering      PTX scope   LLVM syncscope
#   .sc        seq_cst            .cta        "block"
#   .acq_rel   acq_rel            .cluster    "cluster"   (sm_90+, ISel-enforced)
#                                 .gpu        "device"
#                                 .sys        (default — system scope)
#
# Non-WYSIWYG at the floor: below sm_70 / PTX 6.0 the backend legalizes to
# the pre-Volta `membar.{cta,gl,sys}` equivalents (the sc/acq_rel
# distinction collapses conservatively) — where the asm tier would have fed
# ptxas an instruction it rejects. Cluster scope below sm_90 fails ISel
# loudly ("Requires SM >= 90 and PTX >= 78"), never silently downgrades.
#
# Like the proxy fences, these are NOT convergent — duplicating an
# idempotent ordering fence is harmless. Unlike the asm tier's sideeffect
# call, a core-IR fence is a real ordering op the optimizer understands:
# it pins memory accesses across it (that's the point) but no longer pins
# unrelated code motion around it.

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

# --- Tensor-map proxy fences (PTX 8.3, sm_90+) -----------------------------
#
# `fence.proxy.tensormap::generic` is uni-directional.  Release publishes
# prior generic-proxy descriptor writes to the tensormap proxy and has no
# operands.  Acquire makes prior tensormap-proxy accesses visible to later
# generic-proxy accesses over exactly one 128-byte tensor-map descriptor:
#
#   fence.proxy.tensormap::generic.release.<scope>;
#   fence.proxy.tensormap::generic.acquire.<scope> [addr], 128;
#
# PTX 9.3 §9.7.14.4 requires generic addressing for `addr`, requires the
# address to resolve into global memory, and admits only the literal size
# 128.  The typed boundary therefore accepts AS.Generic plus `Val{128}`;
# callers holding an address-space-qualified global pointer must explicitly
# construct/reinterpret the corresponding generic address first.  The four
# scopes are baseline sm_90 features, not sm_90a architecture-specific.
#
# Proxy fences are ordering operations, not collective operations: no
# `convergent` attribute.  The acquire intrinsics carry `argmemonly` and the
# release intrinsics deliberately carry no permissive memory attribute, so
# optimized LLVM retains every fence and cannot move memory through it.

@inline function (::Operation{:fence,
        (:proxy, Symbol("tensormap::generic"), :acquire, :cta)})(
        addr::Core.LLVMPtr{T, AS.Generic}, ::Val{128}) where {T}
    nvvm"fence.proxy.tensormap_generic.acquire.cta"(addr, Val(128))
end

@inline function (::Operation{:fence,
        (:proxy, Symbol("tensormap::generic"), :acquire, :cluster)})(
        addr::Core.LLVMPtr{T, AS.Generic}, ::Val{128}) where {T}
    nvvm"fence.proxy.tensormap_generic.acquire.cluster"(addr, Val(128))
end

@inline function (::Operation{:fence,
        (:proxy, Symbol("tensormap::generic"), :acquire, :gpu)})(
        addr::Core.LLVMPtr{T, AS.Generic}, ::Val{128}) where {T}
    nvvm"fence.proxy.tensormap_generic.acquire.gpu"(addr, Val(128))
end

@inline function (::Operation{:fence,
        (:proxy, Symbol("tensormap::generic"), :acquire, :sys)})(
        addr::Core.LLVMPtr{T, AS.Generic}, ::Val{128}) where {T}
    nvvm"fence.proxy.tensormap_generic.acquire.sys"(addr, Val(128))
end

@inline (::Operation{:fence,
    (:proxy, Symbol("tensormap::generic"), :release, :cta)})() =
    nvvm"fence.proxy.tensormap_generic.release.cta"()

@inline (::Operation{:fence,
    (:proxy, Symbol("tensormap::generic"), :release, :cluster)})() =
    nvvm"fence.proxy.tensormap_generic.release.cluster"()

@inline (::Operation{:fence,
    (:proxy, Symbol("tensormap::generic"), :release, :gpu)})() =
    nvvm"fence.proxy.tensormap_generic.release.gpu"()

@inline (::Operation{:fence,
    (:proxy, Symbol("tensormap::generic"), :release, :sys)})() =
    nvvm"fence.proxy.tensormap_generic.release.sys"()

# --- generic memory fences (tier-1 core IR) ---------------------------------

# PTX scope modifier → LLVM syncscope clause ("" is the system default).
const _FENCE_SCOPES = (
    (:cta,     "syncscope(\"block\") "),
    (:cluster, "syncscope(\"cluster\") "),
    (:gpu,     "syncscope(\"device\") "),
    (:sys,     ""),
)

function _generic_fence_register(sem::Symbol, ordering::String,
                                 scope::Symbol, syncscope::String)
    ir = "fence $syncscope$ordering\nret void"
    @eval @inline (::Operation{:fence, ($(QuoteNode(sem)), $(QuoteNode(scope)))})() =
        Base.llvmcall($ir, Nothing, Tuple{})
    nothing
end

for (sem, ordering) in ((:sc, "seq_cst"), (:acq_rel, "acq_rel"))
    for (scope, syncscope) in _FENCE_SCOPES
        _generic_fence_register(sem, ordering, scope, syncscope)
    end
end

# --- Fabric proxy fences (PTX 9.3, sm_100+), asm tier -----------------------
# `fence.proxy.<to::from>.alias.<sem>.sys;` — uni-directional proxy ordering
# between the fabric proxy (used by `fabric.*` ops on the CFT/NVLink path)
# and the generic proxy. All three handle directions × {acquire, release}
# are enumerated. `.alias` and `.sys` are mandatory per the PTX 9.3 spec
# (§9.7.14.4); no scope/sem variation beyond what's shown. No NVVM
# intrinsics exist at 22.1.7, so these stay on the asm tier — like the
# other fences, NOT convergent (duplicating an ordering fence is harmless),
# but sideeffect + `~{memory}` so nothing moves across them.

for dir in (Symbol("generic::fabric"), Symbol("fabric::generic"),
            Symbol("fabric::fabric")),
    sem in (:acquire, :release)

    asm = "fence.proxy.$dir.alias.$sem.sys;"
    @eval @generated function (::Operation{:fence, (:proxy, $(QuoteNode(dir)),
                                                    :alias, $(QuoteNode(sem)), :sys)})()
        quote
            Base.@inline
            @asmcall($$asm, "~{memory}", true, Nothing, Tuple{})
            nothing
        end
    end
end
