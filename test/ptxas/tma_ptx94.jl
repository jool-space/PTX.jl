include(joinpath(@__DIR__, "..", "tma_ptx94_defs.jl"))

# Assembler evidence for the PTX ISA 9.4 tensor-copy forms: every wrapped
# spelling assembles at its admitting target with the operand shape the
# ISA syntax block prescribes, and the family-gated spellings are rejected
# below their floor.

function _tma94_rejects(kernel, types, cap, feature_set, target)
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

_tma94_lines(ptx, needle) = [String(l) for l in eachline(IOBuffer(ptx))
                             if occursin(needle, l)]

function _tma94_callsite_attrs(llvm::AbstractString, needle::AbstractString)
    groups = Dict{String, String}()
    for line in eachline(IOBuffer(llvm))
        m = match(r"^attributes #([0-9]+) = \{([^}]*)\}", strip(line))
        m === nothing || (groups[m.captures[1]] = m.captures[2])
    end
    calls = [String(line) for line in eachline(IOBuffer(llvm))
             if occursin(" asm ", line) && occursin(needle, line)]
    attrs = String[]
    for call in calls
        m = match(r" #([0-9]+)(?:,|$)", strip(call))
        m === nothing && return calls, String[]
        push!(attrs, get(groups, m.captures[1], ""))
    end
    calls, attrs
end

const _TMA94_FAMILY_GROUPS = (
    (:cluster, "cp.async.bulk.tensor", 85),
    (:cluster_cg2, "cp.async.bulk.tensor", 5),
    (:cta, "cp.async.bulk.tensor", 40),
    (:store, "cp.async.bulk.tensor", 13),
    (:reduce, "cp.reduce.async.bulk.tensor", 104),
    (:prefetch, r"cp\.async\.bulk\.prefetch\.tensor|applypriority\.async\.bulk\.tensor", 36),
)

@testset "PTX ISA 9.4 tensor-copy forms assemble on the sm_107 family" begin
    for (group, needle, count) in _TMA94_FAMILY_GROUPS
        @test length(_TMA94_GROUPS[group]) == count
    end
    if _ptxas_isa() < v"9.4"
        @test_skip "PTX 9.4 assembler required"
    else
        for (group, needle, count) in _TMA94_FAMILY_GROUPS
            types = _tma94_sink_types(group)
            for feature_set in (:family, :arch)
                @test ptxas_compiles(_tma94_sink!, types; cap = v"10.7", feature_set)
            end
            ptx = emit_ptx(_tma94_sink!, types; cap = v"10.7", feature_set = :family)
            @test occursin(".target sm_107f", ptx)
            lines = [String(l) for l in eachline(IOBuffer(ptx))
                     if occursin(needle, l) && !occursin("//", l)]
            @test length(lines) == count
            for (op, mods, _, _) in _TMA94_GROUPS[group]
                @test occursin(PTX.build_head(op, mods) * " [", ptx)
            end
            for (cap, feature_set, target) in ((v"10.3", :family, "sm_103f"),
                                               (v"12.0", :family, "sm_120f"),
                                               (v"10.7", :baseline, "sm_107"))
                @test _tma94_rejects(_tma94_sink!, types, cap, feature_set, target)
            end
        end

        # Operand shapes, one per new grammar.
        ptx = emit_ptx(_tma94_sink!, _tma94_sink_types(:cluster);
                       cap = v"10.7", feature_set = :family)
        @test occursin(r"tile\.mbarrier::complete_tx::bytes\.multicast::cluster::32b\.override::global_address\.override::global_dim_stride \[%r\d+\], \[%rd\d+, %rd\d+, \{%rs\d+, %rs\d+\}, \{%r\d+\}, %rs\d+, \{%r\d+, %r\d+\}\], \[%r\d+\], %r\d+;",
                       ptx)
        @test occursin(r"per_16bytes::8\.multicast::cluster::32b \[%r\d+\], \[%rd\d+, \{%r\d+\}\], \[%r\d+\], %r\d+;",
                       ptx)
        ptx = emit_ptx(_tma94_sink!, _tma94_sink_types(:cta);
                       cap = v"10.7", feature_set = :family)
        @test occursin(r"1d\.shared::cta\.global\.tile\.mbarrier::complete_tx::bytes\.override::global_address\.override::global_dim \[%r\d+\], \[%rd\d+, %rd\d+, \{%rs\d+\}, \{%r\d+\}\], \[%r\d+\];",
                       ptx)
        ptx = emit_ptx(_tma94_sink!, _tma94_sink_types(:store);
                       cap = v"10.7", feature_set = :family)
        @test occursin(r"3d\.global\.shared::cta\.im2col_no_offs::w\.bulk_group \[%rd\d+, \{%r\d+, %r\d+, %r\d+\}\], \[%r\d+\];",
                       ptx)
        @test occursin(r"1d\.global\.shared::cta\.tile\.bulk_group\.override::global_address \[%rd\d+, %rd\d+, \{%r\d+\}\], \[%r\d+\];",
                       ptx)
        ptx = emit_ptx(_tma94_sink!, _tma94_sink_types(:prefetch);
                       cap = v"10.7", feature_set = :family)
        @test occursin(r"prefetch\.tensor\.3d\.L2\.global\.im2col\.L2::evict_last \[%rd\d+, \{%r\d+, %r\d+, %r\d+\}\], \{%rs\d+\};",
                       ptx)
        @test occursin(r"prefetch\.tensor\.3d\.L2\.global\.tile\.override::global_address\.override::global_dim_stride \[%rd\d+, %rd\d+, \{%rs\d+, %rs\d+, %rs\d+\}, \{%r\d+, %r\d+\}, %rs\d+, \{%r\d+, %r\d+, %r\d+\}\];",
                       ptx)
        @test occursin(r"applypriority\.async\.bulk\.tensor\.2d\.global\.bulk_group\.tile\.L2::evict_normal \[%rd\d+, \{%r\d+, %r\d+\}\];",
                       ptx)
        ptx = emit_ptx(_tma94_sink!, _tma94_sink_types(:reduce);
                       cap = v"10.7", feature_set = :family)
        @test occursin(r"cp\.reduce\.async\.bulk\.tensor\.2d\.global\.shared::cta\.add\.tile\.bulk_group\.override::global_address\.override::global_dim_stride \[%rd\d+, %rd\d+, \{%rs\d+, %rs\d+\}, \{%r\d+\}, %rs\d+, \{%r\d+, %r\d+\}\], \[%r\d+\];",
                       ptx)

        # Every copy call site keeps the convergent nomerge boundary.
        llvm = emit_llvm(_tma94_sink!, _tma94_sink_types(:cluster);
                         cap = v"10.7", feature_set = :family)
        calls, attrs = _tma94_callsite_attrs(llvm, "cp.async.bulk.tensor")
        @test length(calls) == 85
        @test length(attrs) == 85
        @test all(a -> occursin(r"\bconvergent\b", a) && occursin(r"\bnomerge\b", a),
                  attrs)
    end
end

@testset "PTX ISA 9.4 bulk chain spellings assemble on sm_107f" begin
    if _ptxas_isa() < v"9.4"
        @test_skip "PTX 9.4 assembler required"
    else
        @test ptxas_compiles(_tma94_chain!, _TMA94_CHAIN_TYPES;
                             cap = v"10.7", feature_set = :family)
        ptx = emit_ptx(_tma94_chain!, _TMA94_CHAIN_TYPES;
                       cap = v"10.7", feature_set = :family)
        for head in ("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b [",
                     "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.mbarrier::report::validity::per_element::ff.multicast::cluster::32b [",
                     "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.mbarrier::report::validity::per_16bytes::80000000 [",
                     "cp.async.bulk.prefetch.L2.global.L2::evict_last [",
                     "applypriority.async.bulk.global.bulk_group.L2::evict_normal [")
            @test occursin(head, ptx)
        end
        @test _tma94_rejects(_tma94_chain!, _TMA94_CHAIN_TYPES, v"10.0", :family,
                             "sm_100f")
    end
end

@testset "PTX ISA 9.4 spellings with lower floors assemble there" begin
    if _ptxas_isa() < v"9.4"
        @test_skip "PTX 9.4 assembler required"
    else
        @test length(_TMA94_GROUPS[:sm90]) == 72
        types = _tma94_sink_types(:sm90)
        @test ptxas_compiles(_tma94_sink!, types; cap = v"9.0")
        @test ptxas_compiles(_tma94_sink!, types; cap = v"10.7", feature_set = :family)
        ptx = emit_ptx(_tma94_sink!, types; cap = v"9.0")
        @test occursin(".target sm_90", ptx)
        @test occursin(r"tile\.mbarrier::complete_tx::bytes\.multicast::cluster::16b \[%r\d+\], \[%rd\d+, \{%r\d+\}\], \[%r\d+\], %rs\d+;",
                       ptx)
        @test occursin("3d.global.shared::cta.im2col_no_offs.bulk_group [", ptx)
        @test occursin("cp.reduce.async.bulk.tensor.5d.global.shared::cta.xor.im2col_no_offs.bulk_group [",
                       ptx)
        @test count("cp.reduce.async.bulk.tensor.", ptx) == 64
        @test _tma94_rejects(_tma94_sink!, types, v"8.0", :baseline, "sm_80")

        @test length(_TMA94_GROUPS[:sm100a]) == 5
        types = _tma94_sink_types(:sm100a)
        @test ptxas_compiles(_tma94_sink!, types; cap = v"10.0", feature_set = :arch)
        @test ptxas_compiles(_tma94_sink!, types; cap = v"10.7", feature_set = :family)
        ptx = emit_ptx(_tma94_sink!, types; cap = v"10.0", feature_set = :arch)
        @test count("cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster::16b [",
                    ptx) == 5
    end
end
