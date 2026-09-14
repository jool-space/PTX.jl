include(joinpath(@__DIR__, "..", "ptx94_bindings_defs.jl"))

# Assembler evidence for the PTX ISA 9.4 bindings that previously had only
# host-tier (spelling) coverage. Each form assembles at its admitting target
# and is rejected at a target below its floor.

function _bind_rejects(kernel, types, cap, feature_set, target)
    err = try
        ptxas_compiles(kernel, types; cap, feature_set)
        nothing
    catch caught
        caught
    end
    err isa ErrorException || return false
    msg = sprint(showerror, err)
    occursin("Failed to compile PTX code", msg) && occursin(target, msg)
end

@testset "9.4 cvt qualifiers assemble on the sm_107 family" begin
    @test length(_BIND_CVT94_FORMS) == 49
    @test count(m -> :rz in m && :ue5m3x2 ∉ m, _BIND_CVT94_FORMS) == 15
    @test count(m -> Symbol("scaled::n1::ue8m0") in m, _BIND_CVT94_FORMS) == 19
    @test count(m -> :pzo in m, _BIND_CVT94_FORMS) == 8
    @test count(m -> :ue5m3x2 in m, _BIND_CVT94_FORMS) == 8 + 3
    if _ptxas_isa() < v"9.4"
        @test_skip "PTX 9.4 assembler required"
    else
        for feature_set in (:family, :arch)
            @test ptxas_compiles(_bind_cvt94!, _BIND_CVT94_TYPES;
                                 cap = v"10.7", feature_set)
        end
        ptx = emit_ptx(_bind_cvt94!, _BIND_CVT94_TYPES;
                       cap = v"10.7", feature_set = :family)
        @test occursin(".target sm_107f", ptx)
        for mods in _BIND_CVT94_FORMS
            @test occursin(PTX.build_head(:cvt, mods) * " ", ptx)
        end
        # Every n1 form reads its scale factor from a .b8 register.
        @test count("cvt.u8.u16 cvt_scale, ", ptx) == 19
        @test _bind_rejects(_bind_cvt94!, _BIND_CVT94_TYPES, v"10.0",
                            :family, "sm_100f")
        @test _bind_rejects(_bind_cvt94!, _BIND_CVT94_TYPES, v"12.0",
                            :family, "sm_120f")
    end
end

@testset "mbarrier .multicast::cluster::32b assembles on sm_107f" begin
    schemas = _bind_mbarrier32_schemas()
    @test length(schemas) == 26
    @test sum(length(s.variants) for s in schemas) == 36
    if _ptxas_isa() < v"9.4"
        @test_skip "PTX 9.4 assembler required"
    else
        @test ptxas_compiles(_bind_mbarrier32!, _BIND_MBARRIER32_TYPES;
                             cap = v"10.7", feature_set = :family)
        ptx = emit_ptx(_bind_mbarrier32!, _BIND_MBARRIER32_TYPES;
                       cap = v"10.7", feature_set = :family)
        for s in schemas
            @test occursin(PTX.build_head(:mbarrier, s.ptxmods) * " ", ptx)
        end
        @test _bind_rejects(_bind_mbarrier32!, _BIND_MBARRIER32_TYPES,
                            v"10.0", :family, "sm_100f")
    end
end

@testset "tcgen05 exclusive allocation assembles on sm_100f and sm_107f" begin
    types = Tuple{Core.LLVMPtr{UInt32, PTX.AS.Shared}, UInt32}
    if _ptxas_isa() < v"9.4"
        @test_skip "PTX 9.4 assembler required"
    else
        for kernel in (_bind_tcgen05_alloc94_cg1!, _bind_tcgen05_alloc94_cg2!)
            # The ISA's target list omits sm_107f while its nCols range
            # (up to 576) requires it; the assembler admits the family.
            for (cap, feature_set) in ((v"10.0", :family), (v"10.7", :family),
                                       (v"10.7", :arch))
                @test ptxas_compiles(kernel, types; cap, feature_set)
            end
            ptx = emit_ptx(kernel, types; cap = v"10.7", feature_set = :family)
            @test occursin("tcgen05.alloc.exclusive.cta_group::", ptx)
            @test occursin("tcgen05.dealloc.exclusive.cta_group::", ptx)
            @test _bind_rejects(kernel, types, v"9.0", :arch, "sm_90a")
        end
    end
end

@testset "tcgen05 9.4 commit forms assemble at their floors" begin
    if _ptxas_isa() < v"9.4"
        @test_skip "PTX 9.4 assembler required"
    else
        t16 = Tuple{UInt32, UInt16}
        for kernel in (_bind_tcgen05_commit16_cg1!, _bind_tcgen05_commit16_cg2!)
            @test ptxas_compiles(kernel, t16; cap = v"10.0", feature_set = :family)
            @test ptxas_compiles(kernel, t16; cap = v"10.7", feature_set = :family)
            @test occursin("multicast::cluster::16b.b64",
                           emit_ptx(kernel, t16; cap = v"10.0", feature_set = :family))
        end
        t32 = Tuple{UInt32, UInt32}
        for kernel in (_bind_tcgen05_commit94_cg1!, _bind_tcgen05_commit94_cg2!)
            @test ptxas_compiles(kernel, t32; cap = v"10.7", feature_set = :family)
            @test ptxas_compiles(kernel, t32; cap = v"10.7", feature_set = :arch)
            ptx = emit_ptx(kernel, t32; cap = v"10.7", feature_set = :family)
            @test count("multicast::cluster::32b.b64", ptx) == 2
            @test count("sync_restrict::shared::read::mma::a", ptx) == 2
            @test _bind_rejects(kernel, t32, v"10.0", :family, "sm_100f")
        end
    end
end

@testset "prefetch .L1::32B.valid_addr and .add.noftz.f32 assemble on sm_90" begin
    if _ptxas_isa() < v"9.4"
        @test_skip "PTX 9.4 assembler required"
    else
        @test ptxas_compiles(_bind_sm90_chain!, _BIND_SM90_CHAIN_TYPES; cap = v"9.0")
        ptx = emit_ptx(_bind_sm90_chain!, _BIND_SM90_CHAIN_TYPES; cap = v"9.0")
        for head in ("prefetch.global.L1::32B.valid_addr", "prefetch.L1::32B.valid_addr",
                     "atom.global.add.noftz.f32", "atom.add.noftz.f32",
                     "red.global.add.noftz.f32", "red.add.noftz.f32")
            @test occursin(head * " ", ptx)
        end
        @test _bind_rejects(_bind_sm90_chain!, _BIND_SM90_CHAIN_TYPES,
                            v"8.0", :baseline, "sm_80")
    end
end

@testset "ldmatrix .m8n16 .s8.s4 assembles on sm_90a and the sm_100f+ families" begin
    if _ptxas_isa() < v"9.4"
        @test_skip "PTX 9.4 assembler required"
    else
        for (cap, feature_set) in ((v"9.0", :arch), (v"10.0", :family),
                                   (v"12.0", :family), (v"12.1", :arch))
            @test ptxas_compiles(_bind_ldmatrix_s8s4!, _BIND_LDMATRIX_TYPES;
                                 cap, feature_set)
        end
        ptx = emit_ptx(_bind_ldmatrix_s8s4!, _BIND_LDMATRIX_TYPES;
                       cap = v"9.0", feature_set = :arch)
        for count in ("x1", "x2", "x4"), space in ("shared", "shared::cta")
            @test occursin("ldmatrix.sync.aligned.m8n16.$count.$space.s8.s4 ", ptx)
        end
        # Baseline sm_90 is not an admitting target; only sm_90a is.
        @test _bind_rejects(_bind_ldmatrix_s8s4!, _BIND_LDMATRIX_TYPES,
                            v"9.0", :baseline, "sm_90")
        @test _bind_rejects(_bind_ldmatrix_s8s4!, _BIND_LDMATRIX_TYPES,
                            v"8.0", :baseline, "sm_80")
    end
end
