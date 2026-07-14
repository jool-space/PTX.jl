# PTX 9.3 §9.7.14.4: all tensor-map proxy fences are PTX 8.3, baseline
# sm_90 features.  This is offline compilation evidence only: proxy ordering
# is meaningful around real tensor-map mutation/TMA use, not as an isolated
# runtime value test.

function _tensormap_proxy_fences_all!(
        addr::Core.LLVMPtr{UInt8, PTX.AS.Generic})
    ptx"fence.proxy.tensormap::generic.release.cta"()
    ptx"fence.proxy.tensormap::generic.acquire.cta"(addr, Val(128))
    ptx"fence.proxy.tensormap::generic.release.cluster"()
    ptx"fence.proxy.tensormap::generic.acquire.cluster"(addr, Val(128))
    ptx"fence.proxy.tensormap::generic.release.gpu"()
    ptx"fence.proxy.tensormap::generic.acquire.gpu"(addr, Val(128))
    ptx"fence.proxy.tensormap::generic.release.sys"()
    ptx"fence.proxy.tensormap::generic.acquire.sys"(addr, Val(128))
    nothing
end

@testset "tensor-map proxy fences at baseline sm_90" begin
    types = Tuple{Core.LLVMPtr{UInt8, PTX.AS.Generic}}
    @test ptxas_compiles(_tensormap_proxy_fences_all!, types; cap = v"9.0")

    ptx = emit_ptx(_tensormap_proxy_fences_all!, types; cap = v"9.0")
    @test occursin(".target sm_90", ptx)
    for sem in (:acquire, :release), scope in (:cta, :cluster, :gpu, :sys)
        head = "fence.proxy.tensormap::generic.$sem.$scope"
        @test count(head, ptx) == 1
        if sem === :acquire
            @test occursin(Regex(replace(head, "." => "\\.") *
                                 " \\[%rd\\d+\\], (128|0x80);"), ptx)
        else
            @test occursin(head * ";", ptx)
        end
    end

    # Optimized LLVM must retain all eight ordering operations.  Their
    # registry contracts intentionally do not mark them convergent: proxy
    # fences are per-thread ordering operations, not collective rendezvous.
    llvm = emit_llvm(_tensormap_proxy_fences_all!, types; cap = v"9.0")
    for sem in (:acquire, :release), scope in (:cta, :cluster, :gpu, :sys)
        intr = "llvm.nvvm.fence.proxy.tensormap_generic.$sem.$scope"
        @test count(intr, llvm) >= 2 # one declaration plus one retained call
    end
    @test !occursin(r"call void @\"?llvm\.nvvm\.fence\.proxy\.tensormap_generic\.[^\n]+convergent",
                    llvm)
end
