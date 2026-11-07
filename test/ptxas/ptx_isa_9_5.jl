# Assembly evidence for the PTX ISA 9.5 forms PTX.jl binds; their rendering
# and lowering contracts live in host/ptx_isa_9_5.jl. Each form assembles at
# its ISA floor, and the floors that separate a family or `a` target from the
# baseline are pinned to ptxas rejections. Offline compilation only: none of
# these kernels runs.

const _PTX95_GLOBAL = Core.LLVMPtr{UInt8, PTX.AS.Global}
const _PTX95_SHARED = Core.LLVMPtr{UInt8, PTX.AS.Shared}
const _PTX95_GENERIC = Core.LLVMPtr{UInt8, PTX.AS.Generic}

function _ptx95_locality!(out::CuDeviceVector{UInt32, 1})
    Base.@inbounds out[1] = ptx"mov.u32"(ptx"%locality_domain_id")
    Base.@inbounds out[2] = ptx"mov.u32"(ptx"%locality_domain_supplemental_sm")
    nothing
end

function _ptx95_sync_restrict!()
    ptx"fabric.submit.commit_group"()
    ptx"fabric.wait.sync_restrict::read"(Val(0))
    ptx"fabric.wait.sync_restrict::read"(Val(3))
    ptx"cp.async.bulk.wait_group.sync_restrict::proxy::async"(Val(0))
    ptx"cp.async.bulk.wait_group.sync_restrict::proxy::async"(Val(3))
    nothing
end

function _ptx95_alias_fence!()
    ptx"fence.alias.sync_restrict::read.release.sys"()
    nothing
end

function _ptx95_sincos_2pi!(out::CuDeviceVector{Float32, 1}, x::Float32)
    Base.@inbounds out[1] = ptx"sin.approx.2pi.ftz.f32"(x)
    Base.@inbounds out[2] = ptx"cos.approx.2pi.ftz.f32"(x)
    nothing
end

function _ptx95_alloc_class_query!(out::CuDeviceVector{UInt32, 1},
                                   g::_PTX95_GLOBAL, s::_PTX95_SHARED,
                                   p::_PTX95_GENERIC)
    Base.@inbounds out[1] =
        ptx"tensormap.query.alloc_class_large.global.b1024"(g) % UInt32
    Base.@inbounds out[2] =
        ptx"tensormap.query.alloc_class_large.shared::cta.b1024"(s) % UInt32
    Base.@inbounds out[3] =
        ptx"tensormap.query.alloc_class_large.b1024"(p) % UInt32
    nothing
end

function _ptx95_alloc_class_replace!(g::_PTX95_GLOBAL, s::_PTX95_SHARED,
                                     addr::UInt64)
    ptx"tensormap.replace.tile.alloc_large.global.b1024"(g)
    ptx"tensormap.replace.tile.alloc_small.shared::cta.b1024"(s)
    ptx"tensormap.replace.tile.global_address.alloc_same.global.b1024.b64"(g, addr)
    ptx"tensormap.replace.tile.global_address.alloc_large.shared::cta.b1024.b64"(s, addr)
    nothing
end

const _PTX95_TENSORMAP_TYPES =
    Tuple{CuDeviceVector{UInt32, 1}, _PTX95_GLOBAL, _PTX95_SHARED, _PTX95_GENERIC}
const _PTX95_REPLACE_TYPES = Tuple{_PTX95_GLOBAL, _PTX95_SHARED, UInt64}

@testset "PTX 9.5 forms assemble" begin
    if _ptxas_isa() < v"9.5"
        @test_skip "PTX 9.5 assembler required"
    else
        @testset "locality special registers (sm_100)" begin
            types = Tuple{CuDeviceVector{UInt32, 1}}
            @test ptxas_compiles(_ptx95_locality!, types; cap = v"10.0")
            ptx = emit_ptx(_ptx95_locality!, types; cap = v"10.0")
            @test occursin(r"^\.version 9\.5$"m, ptx)
            @test occursin(r"mov\.u32\s+%r\d+, %locality_domain_id;", ptx)
            @test occursin(r"mov\.u32\s+%r\d+, %locality_domain_supplemental_sm;", ptx)
            @test ptxas_rejects(_ptx95_locality!, types; cap = v"9.0",
                                target = "sm_90")
        end

        @testset "fabric submit groups and the async-proxy bulk wait (sm_100)" begin
            @test ptxas_compiles(_ptx95_sync_restrict!, Tuple{}; cap = v"10.0")
            ptx = emit_ptx(_ptx95_sync_restrict!, Tuple{}; cap = v"10.0")
            @test count("fabric.submit.commit_group;", ptx) == 1
            for n in (0, 3)
                @test count("fabric.wait.sync_restrict::read $n;", ptx) == 1
                @test count("cp.async.bulk.wait_group.sync_restrict::proxy::async $n;",
                            ptx) == 1
            end
            @test ptxas_rejects(_ptx95_sync_restrict!, Tuple{}; cap = v"9.0",
                                target = "sm_90")
        end

        @testset "read-only alias release fence (sm_90)" begin
            @test ptxas_compiles(_ptx95_alias_fence!, Tuple{}; cap = v"9.0")
            ptx = emit_ptx(_ptx95_alias_fence!, Tuple{}; cap = v"9.0")
            @test count("fence.alias.sync_restrict::read.release.sys;", ptx) == 1
            @test ptxas_rejects(_ptx95_alias_fence!, Tuple{}; cap = v"8.0",
                                target = "sm_80")
        end

        @testset "sin/cos .approx.2pi (all targets)" begin
            types = Tuple{CuDeviceVector{Float32, 1}, Float32}
            @test ptxas_compiles(_ptx95_sincos_2pi!, types; cap = v"7.5")
            ptx = emit_ptx(_ptx95_sincos_2pi!, types; cap = v"7.5")
            @test occursin(r"sin\.approx\.2pi\.ftz\.f32\s+%[rf]\d+, %[rf]\d+;", ptx)
            @test occursin(r"cos\.approx\.2pi\.ftz\.f32\s+%[rf]\d+, %[rf]\d+;", ptx)
        end

        @testset "tensormap.query.alloc_class_large" begin
            for (cap, feature_set) in ((v"9.0", :arch), (v"10.0", :family),
                                       (v"11.0", :family), (v"12.0", :family))
                @test ptxas_compiles(_ptx95_alloc_class_query!,
                                     _PTX95_TENSORMAP_TYPES; cap, feature_set)
            end
            ptx = emit_ptx(_ptx95_alloc_class_query!, _PTX95_TENSORMAP_TYPES;
                           cap = v"10.0", feature_set = :family)
            for ss in (".global", ".shared::cta", "")
                @test occursin(Regex("tensormap\\.query\\.alloc_class_large" *
                                     replace(ss, "." => "\\.") *
                                     "\\.b1024\\s+%p\\d+, \\[%rd\\d+\\];"), ptx)
            end
            @test ptxas_rejects(_ptx95_alloc_class_query!, _PTX95_TENSORMAP_TYPES;
                                cap = v"9.0", target = "sm_90")
            @test ptxas_rejects(_ptx95_alloc_class_query!, _PTX95_TENSORMAP_TYPES;
                                cap = v"10.0", target = "sm_100")
        end

        @testset "tensormap.replace .alloc_class" begin
            for (cap, feature_set) in ((v"9.0", :arch), (v"10.0", :family),
                                       (v"12.0", :family))
                @test ptxas_compiles(_ptx95_alloc_class_replace!,
                                     _PTX95_REPLACE_TYPES; cap, feature_set)
            end
            ptx = emit_ptx(_ptx95_alloc_class_replace!, _PTX95_REPLACE_TYPES;
                           cap = v"9.0", feature_set = :arch)
            @test count("tensormap.replace.tile.alloc_large.global.b1024", ptx) == 1
            @test count("tensormap.replace.tile.alloc_small.shared::cta.b1024", ptx) == 1
            @test count("tensormap.replace.tile.global_address.alloc_same.global.b1024.b64",
                        ptx) == 1
            @test count("tensormap.replace.tile.global_address.alloc_large.shared::cta.b1024.b64",
                        ptx) == 1
            @test ptxas_rejects(_ptx95_alloc_class_replace!, _PTX95_REPLACE_TYPES;
                                cap = v"9.0", target = "sm_90")
        end
    end
end
