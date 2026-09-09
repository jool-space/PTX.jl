include(joinpath(@__DIR__, "..", "ptx94_ga_defs.jl"))

function _ga_proxy_fences!()
    ptx"fence.proxy.alias.acquire.sys"()
    ptx"fence.proxy.alias.release.sys"()
    ptx"fence.proxy.async::generic.release.sync_restrict::shared::cluster::read.cluster"()
    nothing
end

@testset "GA readonly loads and proxy fences assemble" begin
    if _ptxas_isa() < v"9.4"
        @test_skip "PTX 9.4 assembler required"
    else
        types = Tuple{CuDeviceVector{UInt64,1}, CuDeviceVector{UInt64,1}}
        @test ptxas_compiles(_ga_readonly_loads!, types; cap = v"9.0")
        ptx = emit_ptx(_ga_readonly_loads!, types; cap = v"9.0")
        for space in ("", "global."), (kind, _) in _GA_LOAD_TYPES
            @test occursin("ld.$space$kind.proxy::readonly", ptx)
        end
        @test ptxas_compiles(_ga_proxy_fences!, Tuple{}; cap = v"9.0")
        fences = emit_ptx(_ga_proxy_fences!, Tuple{}; cap = v"9.0")
        for head in ("fence.proxy.alias.acquire.sys",
                     "fence.proxy.alias.release.sys",
                     "fence.proxy.async::generic.release.sync_restrict::shared::cluster::read.cluster")
            @test occursin(head, fences)
        end
        for kernel in (_ga_readonly_loads!, _ga_proxy_fences!)
            tt = kernel === _ga_readonly_loads! ? types : Tuple{}
            @test_throws ErrorException ptxas_compiles(kernel, tt; cap = v"8.0")
        end
    end
end
