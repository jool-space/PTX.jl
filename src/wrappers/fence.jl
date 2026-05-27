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

# --- Fabric proxy fences (PTX 9.3, sm_100+) -------------------------------
# `fence.proxy.<to::from>.alias.<sem>.sys;` — uni-directional proxy ordering
# between the fabric proxy (used by `fabric.*` ops on the CFT/NVLink path)
# and the generic proxy. All three handle directions × {acquire, release}
# are enumerated. `.alias` and `.sys` are mandatory per the PTX 9.3 spec
# (§9.7.14.4); no scope/sem variation beyond what's shown.

@generated function (::Operation{:fence, (:proxy, Symbol("generic::fabric"), :alias, :acquire, :sys)})()
    quote
        Base.@inline
        @asmcall("fence.proxy.generic::fabric.alias.acquire.sys;",
                 "~{memory}", true, Nothing,
                 Tuple{})
        nothing
    end
end

@generated function (::Operation{:fence, (:proxy, Symbol("generic::fabric"), :alias, :release, :sys)})()
    quote
        Base.@inline
        @asmcall("fence.proxy.generic::fabric.alias.release.sys;",
                 "~{memory}", true, Nothing,
                 Tuple{})
        nothing
    end
end

@generated function (::Operation{:fence, (:proxy, Symbol("fabric::generic"), :alias, :acquire, :sys)})()
    quote
        Base.@inline
        @asmcall("fence.proxy.fabric::generic.alias.acquire.sys;",
                 "~{memory}", true, Nothing,
                 Tuple{})
        nothing
    end
end

@generated function (::Operation{:fence, (:proxy, Symbol("fabric::generic"), :alias, :release, :sys)})()
    quote
        Base.@inline
        @asmcall("fence.proxy.fabric::generic.alias.release.sys;",
                 "~{memory}", true, Nothing,
                 Tuple{})
        nothing
    end
end

@generated function (::Operation{:fence, (:proxy, Symbol("fabric::fabric"), :alias, :acquire, :sys)})()
    quote
        Base.@inline
        @asmcall("fence.proxy.fabric::fabric.alias.acquire.sys;",
                 "~{memory}", true, Nothing,
                 Tuple{})
        nothing
    end
end

@generated function (::Operation{:fence, (:proxy, Symbol("fabric::fabric"), :alias, :release, :sys)})()
    quote
        Base.@inline
        @asmcall("fence.proxy.fabric::fabric.alias.release.sys;",
                 "~{memory}", true, Nothing,
                 Tuple{})
        nothing
    end
end
