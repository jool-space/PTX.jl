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

function _ga_rejected_at(kernel, types, cap, feature_set, target)
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

@testset "tcgen05.ld.spcompress assembles on sm_107a only" begin
    types = Tuple{CuDeviceVector{UInt32, 1}, UInt32}
    @test length(PTX.wrapper_asm_forms(:tcgen05_ldspc)) == 72
    if _ptxas_isa() < v"9.4"
        @test_skip "PTX 9.4 assembler required"
    else
        @test ptxas_compiles(_ga_t5_ldspc!, types; cap = v"10.7", feature_set = :arch)
        ptx = emit_ptx(_ga_t5_ldspc!, types; cap = v"10.7", feature_set = :arch)
        @test occursin(".target sm_107a", ptx)
        for mods in PTX.wrapper_asm_forms(:tcgen05_ldspc)
            @test occursin(PTX.build_head(:tcgen05, mods) * " {", ptx)
        end
        @test occursin(r"tcgen05\.ld\.red\.spcompress\.sync\.aligned\.32x32b\.x4\.max\.sp::2:4\.abs\.NaN\.f32\.b2 \{%r\d+\}, \{%r\d+, %r\d+\}, %r\d+, \[%r\d+\];",
                       ptx)
        @test _ga_rejected_at(_ga_t5_ldspc!, types, v"10.7", :family, "sm_107f")
        @test _ga_rejected_at(_ga_t5_ldspc!, types, v"10.0", :arch, "sm_100a")
    end
end

@testset "spcompress and spdecompress assemble on sm_107a only" begin
    @test length(PTX.wrapper_asm_forms(:spcompress)) == 28
    @test length(PTX.wrapper_asm_forms(:spdecompress)) == 143
    if _ptxas_isa() < v"9.4"
        @test_skip "PTX 9.4 assembler required"
    else
        out = CuDeviceVector{UInt32, 1}
        for elem in (:b8, :b16)
            types = Tuple{out, Val{elem}, UInt32, UInt32}
            @test ptxas_compiles(_ga_spcompress!, types; cap = v"10.7", feature_set = :arch)
            ptx = emit_ptx(_ga_spcompress!, types; cap = v"10.7", feature_set = :arch)
            for mods in _ga_sp_forms(:spcompress, elem)
                @test occursin(PTX.build_head(:spcompress, mods) * " {", ptx)
            end
            @test _ga_rejected_at(_ga_spcompress!, types, v"10.7", :family, "sm_107f")
            @test _ga_rejected_at(_ga_spcompress!, types, v"10.0", :arch, "sm_100a")
        end
        # The ISA's own example spelling, exact operand shape.
        ptx = emit_ptx(_ga_spcompress!, Tuple{out, Val{:b8}, UInt32, UInt32};
                       cap = v"10.7", feature_set = :arch)
        @test occursin(r"spcompress\.b8\.b2\.sp::2:4\.x4 \{%r\d+\}, \{%r\d+, %r\d+, %r\d+, %r\d+\}, \{(%r\d+, ){7}%r\d+\}, %r\d+;",
                       ptx)

        for mods in PTX.wrapper_asm_forms(:spdecompress)
            types = Tuple{out, Val{mods}, UInt32}
            @test ptxas_compiles(_ga_spdecompress!, types; cap = v"10.7", feature_set = :arch)
            @test occursin(PTX.build_head(:spdecompress, mods) * " {",
                           emit_ptx(_ga_spdecompress!, types; cap = v"10.7", feature_set = :arch))
        end
        types = Tuple{out, Val{(:b8, :b2, Symbol("sp::2:4"), :x32)}, UInt32}
        @test _ga_rejected_at(_ga_spdecompress!, types, v"10.7", :family, "sm_107f")
        @test _ga_rejected_at(_ga_spdecompress!, types, v"10.0", :arch, "sm_100a")
        ptx = emit_ptx(_ga_spdecompress!, types; cap = v"10.7", feature_set = :arch)
        @test occursin(r"spdecompress\.b8\.b2\.sp::2:4\.x32 \{(%r\d+, ){31}%r\d+\}, \{%r\d+, %r\d+, %r\d+, %r\d+\}, \{(%r\d+, ){15}%r\d+\};",
                       ptx)
@testset "tcgen05 ti16, collector::b, and lut::b mma assemble on sm_107" begin
    if _ptxas_isa() < v"9.4"
        @test_skip "PTX 9.4 assembler required"
    else
        family = ((_ga_t5_ti16_cg1!, _GA_T5_TI16_CG1_TT),
                  (_ga_t5_ti16_cg2!, _GA_T5_TI16_CG2_TT),
                  (_ga_t5_collb_cg1!, _GA_T5_MX_TT),
                  (_ga_t5_lut_cg1!, _GA_T5_MX_TT))
        for (kernel, tt) in family, feature_set in (:family, :arch)
            @test ptxas_compiles(kernel, tt; cap = v"10.7", feature_set)
        end
        @test ptxas_compiles(_ga_t5_collb_avariant!, _GA_T5_MX_TT;
                             cap = v"10.7", feature_set = :arch)

        ptx = emit_ptx(_ga_t5_ti16_cg1!, _GA_T5_TI16_CG1_TT;
                       cap = v"10.7", feature_set = :family)
        @test occursin(".target sm_107f", ptx)
        # The operand schema per form: sp metadata bracketed after B, the
        # mask before enable-input-d, the ws zero-column mask last.
        @test occursin(r"tcgen05\.mma\.cta_group::1\.kind::ti16 \[%r\d+\], %rd\d+, %rd\d+, %r\d+, %p\d+;",
                       ptx)
        @test occursin(r"tcgen05\.mma\.cta_group::1\.kind::ti16\.collector::b::fill \[%r\d+\], %rd\d+, %rd\d+, %r\d+, \{%r\d+, %r\d+, %r\d+, %r\d+\}, %p\d+;",
                       ptx)
        @test occursin(r"tcgen05\.mma\.cta_group::1\.kind::ti16\.ashift\.collector::a::lastuse\.collector::b::lastuse \[%r\d+\], \[%r\d+\], %rd\d+, %r\d+, \{%r\d+, %r\d+, %r\d+, %r\d+\}, %p\d+;",
                       ptx)
        @test occursin(r"tcgen05\.mma\.sp\.cta_group::1\.kind::ti16\.collector::b::fill \[%r\d+\], %rd\d+, %rd\d+, \[%r\d+\], %r\d+, %p\d+;",
                       ptx)
        @test occursin(r"tcgen05\.mma\.ws\.cta_group::1\.kind::ti16\.collector::b1::fill \[%r\d+\], \[%r\d+\], %rd\d+, %r\d+, %p\d+, %rd\d+;",
                       ptx)
        @test occursin(r"tcgen05\.mma\.ws\.sp\.cta_group::1\.kind::ti16\.collector::b3::lastuse \[%r\d+\], %rd\d+, %rd\d+, \[%r\d+\], %r\d+, %p\d+;",
                       ptx)

        ptx = emit_ptx(_ga_t5_collb_cg1!, _GA_T5_MX_TT;
                       cap = v"10.7", feature_set = :family)
        # scale-input-d is a trailing immediate; the block-scale forms keep
        # the two scale addresses.
        @test occursin(r"tcgen05\.mma\.cta_group::1\.kind::f16\.collector::a::fill\.collector::b::use \[%r\d+\], %rd\d+, %rd\d+, %r\d+, \{%r\d+, %r\d+, %r\d+, %r\d+\}, %p\d+, 5;",
                       ptx)
        @test occursin(r"tcgen05\.mma\.cta_group::1\.kind::tf32\.ashift\.collector::a::lastuse\.collector::b::lastuse \[%r\d+\], \[%r\d+\], %rd\d+, %r\d+, %p\d+, 1;",
                       ptx)
        @test occursin(r"tcgen05\.mma\.sp\.cta_group::1\.kind::tf32\.ashift\.collector::b::use \[%r\d+\], \[%r\d+\], %rd\d+, \[%r\d+\], %r\d+, \{%r\d+, %r\d+, %r\d+, %r\d+\}, %p\d+, 15;",
                       ptx)
        @test occursin(r"tcgen05\.mma\.sp\.cta_group::1\.kind::mxf8f6f4\.block_scale\.block32\.collector::a::fill\.collector::b::use \[%r\d+\], %rd\d+, %rd\d+, \[%r\d+\], %r\d+, \[%r\d+\], \[%r\d+\], %p\d+;",
                       ptx)

        ptx = emit_ptx(_ga_t5_lut_cg1!, _GA_T5_MX_TT;
                       cap = v"10.7", feature_set = :family)
        @test occursin(r"tcgen05\.mma\.cta_group::1\.kind::f8f6f4\.decompress::lut::b \[%r\d+\], %rd\d+, %rd\d+, \[%r\d+\], %r\d+, %p\d+;",
                       ptx)
        @test occursin(r"tcgen05\.mma\.cta_group::1\.kind::mxf8f6f4\.block_scale\.decompress::lut::b\.block32\.collector::b::lastuse \[%r\d+\], \[%r\d+\], %rd\d+, \[%r\d+\], %r\d+, \[%r\d+\], \[%r\d+\], %p\d+;",
                       ptx)

        for (kernel, tt) in family,
                (cap, feature_set, target) in ((v"10.0", :family, "sm_100f"),
                                               (v"10.3", :arch, "sm_103a"),
                                               (v"12.0", :family, "sm_120f"))
            @test _ga_rejected_at(kernel, tt, cap, feature_set, target)
        end
        # The sparse mxf4 kinds refuse the family target even on sm_107.
        @test _ga_rejected_at(_ga_t5_collb_avariant!, _GA_T5_MX_TT, v"10.7",
                          :family, "sm_107f")
    end
end
