# Memory-ordering fences. Pure no-arg / no-return forms; could route
# through the generic @asmcall fallback in inst.jl, but typed wrappers
# make the surface discoverable and pin the asm string at PTX.jl.
# PTX 9.2 §9.7.12.4.

# `fence.proxy.async;` — async-proxy ↔ generic-proxy ordering fence. The
# async proxy is the memory proxy used by TMA (cp.async.bulk*) and tcgen05;
# generic-proxy ops (regular load/store, mma) need this fence to observe
# async-proxy writes (and vice versa). sm_90+ only.
@generated function (::Operation{:fence, (:proxy, :async)})()
    quote
        Base.@inline
        @asmcall("fence.proxy.async;",
                 "~{memory}", true, Nothing,
                 Tuple{})
        nothing
    end
end

# `fence.mbarrier_init.release.cluster;` — release-fence after
# `mbarrier.init` so other CTAs in the cluster observe the initialized
# mbarrier state before reading/arriving on it. sm_90+ (cluster scope).
@generated function (::Operation{:fence, (:mbarrier_init, :release, :cluster)})()
    quote
        Base.@inline
        @asmcall("fence.mbarrier_init.release.cluster;",
                 "~{memory}", true, Nothing,
                 Tuple{})
        nothing
    end
end
