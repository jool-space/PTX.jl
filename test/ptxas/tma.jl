include(joinpath(@__DIR__, "..", "tma_defs.jl"))

# Exact-floor compiler evidence for the tensor-copy (TMA) wrapper surface:
# the PTX 9.3 §9.7.9.26.5.4 tile and base-im2col prefetches at their sm_90
# floor, and the PTX ISA 9.4 tensor-copy forms, which assemble at their
# admitting target with the operand shape the ISA syntax block prescribes
# and are rejected below their floor. Cache policy is a performance hint,
# so the prefetch legs intentionally stop at optimized LLVM, emitted PTX,
# and ptxas rather than claiming runtime semantics.

# Attribute groups of every inline-asm call site whose text matches
# `needle`. Single-route asm since the demotion: the call sites are inline
# asm (convergent_asm_ir), not llvm.nvvm.* intrinsic calls.
function _tma_callsite_attrs(llvm::AbstractString, needle)
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

# --- tile prefetch (PTX 9.3 §9.7.9.26.5.4) ------------------------------------

function _tma_tile_prefetch_surface!(
        tmap::PTX.TMADescriptorPtr, policy::UInt64)
    ptx"cp.async.bulk.prefetch.tensor.1d.L2.global.tile"(
        tmap, Int32(1))
    ptx"cp.async.bulk.prefetch.tensor.2d.L2.global.tile"(
        tmap, Int32(1), Int32(2))
    ptx"cp.async.bulk.prefetch.tensor.3d.L2.global.tile"(
        tmap, Int32(1), Int32(2), Int32(3))
    ptx"cp.async.bulk.prefetch.tensor.4d.L2.global.tile"(
        tmap, Int32(1), Int32(2), Int32(3), Int32(4))
    ptx"cp.async.bulk.prefetch.tensor.5d.L2.global.tile"(
        tmap, Int32(1), Int32(2), Int32(3), Int32(4), Int32(5))

    ptx"cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::cache_hint"(
        tmap, Int32(1), policy)
    ptx"cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint"(
        tmap, Int32(1), Int32(2), policy)
    ptx"cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint"(
        tmap, Int32(1), Int32(2), Int32(3), policy)
    ptx"cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::cache_hint"(
        tmap, Int32(1), Int32(2), Int32(3), Int32(4), policy)
    ptx"cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::cache_hint"(
        tmap, Int32(1), Int32(2), Int32(3), Int32(4), Int32(5), policy)
    return nothing
end

@testset "TMA tile prefetch at its sm_90 target floor" begin
    types = Tuple{PTX.TMADescriptorPtr, UInt64}
    @test ptxas_compiles(_tma_tile_prefetch_surface!, types; cap = v"9.0")

    ptx = emit_ptx(_tma_tile_prefetch_surface!, types; cap = v"9.0")
    # The instruction was introduced in PTX 8.0; the backend stamps the module
    # with the toolkit-negotiated ISA (the assembler ceiling — 13.3 → 9.3,
    # 13.4 → 9.4), independently of this instruction's floor.
    isa = _ptxas_isa()
    @test occursin(".version $(isa.major).$(isa.minor)", ptx)
    @test occursin(".target sm_90", ptx)
    lines = [String(line) for line in eachline(IOBuffer(ptx))
             if occursin("cp.async.bulk.prefetch.tensor", line)]
    @test length(lines) == 10
    for rank in 1:5
        ranked = filter(line -> occursin("tensor.$(rank)d.L2.global.tile", line),
                        lines)
        @test length(ranked) == 2
        @test count(line -> occursin(".L2::cache_hint", line), ranked) == 1
        @test all(line -> occursin(
            Regex("tensor\\.$(rank)d\\.L2\\.global\\.tile(?:\\.L2::cache_hint)? \\["),
            line), ranked)
    end

    # The optimized middle end must retain every weak-memory hint call.
    # The asm route keeps the convergent nomerge boundary the retired
    # intrinsic records imposed, even though PTX imposes no
    # warp-collective participation rule.
    llvm = emit_llvm(_tma_tile_prefetch_surface!, types; cap = v"9.0")
    calls, attrs = _tma_callsite_attrs(llvm, "cp.async.bulk.prefetch.tensor")
    @test length(calls) == 10
    @test length(attrs) == 10
    @test all(attr -> occursin(r"\bconvergent\b", attr), attrs)
    @test all(attr -> occursin(r"\bnomerge\b", attr), attrs)
end

# --- base-im2col prefetch (PTX 9.3 §9.7.9.26.5.4) -----------------------------

function _tma_im2col_prefetch_surface!(
        tmap::PTX.TMADescriptorPtr, policy::UInt64)
    ptx"cp.async.bulk.prefetch.tensor.3d.L2.global.im2col"(
        tmap, Int32(1), Int32(2), Int32(3), Int16(4))
    ptx"cp.async.bulk.prefetch.tensor.4d.L2.global.im2col"(
        tmap, Int32(1), Int32(2), Int32(3), Int32(4), Int16(5), Int16(6))
    ptx"cp.async.bulk.prefetch.tensor.5d.L2.global.im2col"(
        tmap, Int32(1), Int32(2), Int32(3), Int32(4), Int32(5),
        Int16(6), Int16(7), Int16(8))

    ptx"cp.async.bulk.prefetch.tensor.3d.L2.global.im2col.L2::cache_hint"(
        tmap, Int32(1), Int32(2), Int32(3), Int16(4), policy)
    ptx"cp.async.bulk.prefetch.tensor.4d.L2.global.im2col.L2::cache_hint"(
        tmap, Int32(1), Int32(2), Int32(3), Int32(4), Int16(5), Int16(6),
        policy)
    ptx"cp.async.bulk.prefetch.tensor.5d.L2.global.im2col.L2::cache_hint"(
        tmap, Int32(1), Int32(2), Int32(3), Int32(4), Int32(5),
        Int16(6), Int16(7), Int16(8), policy)
    return nothing
end

@testset "TMA base-im2col prefetch at its sm_90 target floor" begin
    types = Tuple{PTX.TMADescriptorPtr, UInt64}
    @test ptxas_compiles(_tma_im2col_prefetch_surface!, types; cap = v"9.0")

    ptx = emit_ptx(_tma_im2col_prefetch_surface!, types; cap = v"9.0")
    # Module version tracks the toolkit-negotiated ISA, not this
    # instruction's PTX 8.0 floor (see the tile prefetch leg above).
    isa = _ptxas_isa()
    @test occursin(".version $(isa.major).$(isa.minor)", ptx)
    @test occursin(".target sm_90", ptx)
    lines = [String(line) for line in eachline(IOBuffer(ptx))
             if occursin("cp.async.bulk.prefetch.tensor", line)]
    @test length(lines) == 6
    for rank in 3:5
        ranked = filter(
            line -> occursin("tensor.$(rank)d.L2.global.im2col", line), lines)
        @test length(ranked) == 2
        @test count(line -> occursin(".L2::cache_hint", line), ranked) == 1
        coords = "%r\\d+" * repeat(", %r\\d+", rank - 1)
        offsets = "%rs\\d+" * repeat(", %rs\\d+", rank - 3)
        plain = Regex("tensor\\.$(rank)d\\.L2\\.global\\.im2col " *
                      "\\[%rd\\d+, \\{$coords\\}\\], \\{$offsets\\};")
        hinted = Regex("tensor\\.$(rank)d\\.L2\\.global\\.im2col" *
                       "\\.L2::cache_hint \\[%rd\\d+, \\{$coords\\}\\], " *
                       "\\{$offsets\\}, %rd\\d+;")
        @test count(line -> occursin(plain, line), ranked) == 1
        @test count(line -> occursin(hinted, line), ranked) == 1
    end

    llvm = emit_llvm(_tma_im2col_prefetch_surface!, types; cap = v"9.0")
    calls, attrs = _tma_callsite_attrs(llvm, r"cp\.async\.bulk\.prefetch\.tensor\.\dd\.L2\.global\.im2col")
    @test length(calls) == 6
    @test length(attrs) == 6
    @test all(attr -> occursin(r"\bconvergent\b", attr), attrs)
    @test all(attr -> occursin(r"\bnomerge\b", attr), attrs)
end

@testset "TMA base-im2col prefetch fails below sm_90" begin
    # The asm route has no ISel gate (the retired intrinsic route failed
    # with "Cannot select" in the backend); the sm_90 floor is now
    # enforced by ptxas, which rejects the spelling for an sm_89 target.
    types = Tuple{PTX.TMADescriptorPtr, UInt64}
    ptx = emit_ptx(_tma_im2col_prefetch_surface!, types; cap = v"8.9")
    @test occursin("cp.async.bulk.prefetch.tensor.3d.L2.global.im2col", ptx)
    err = try
        ptxas_compiles(_tma_im2col_prefetch_surface!, types; cap = v"8.9")
        nothing
    catch e
        e
    end
    @test err !== nothing
    @test occursin("sm_90 or higher", sprint(showerror, err))
end

# --- PTX ISA 9.4 tensor-copy forms (§9.7.10.28, §9.7.10.19) --------------------

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
                @test ptxas_rejects(_tma94_sink!, types; cap, feature_set, target)
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
        calls, attrs = _tma_callsite_attrs(llvm, "cp.async.bulk.tensor")
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
        @test ptxas_rejects(_tma94_chain!, _TMA94_CHAIN_TYPES;
                            cap = v"10.0", feature_set = :family,
                            target = "sm_100f")
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
        @test ptxas_rejects(_tma94_sink!, types; cap = v"8.0", target = "sm_80")

        @test length(_TMA94_GROUPS[:sm100a]) == 5
        types = _tma94_sink_types(:sm100a)
        @test ptxas_compiles(_tma94_sink!, types; cap = v"10.0", feature_set = :arch)
        @test ptxas_compiles(_tma94_sink!, types; cap = v"10.7", feature_set = :family)
        ptx = emit_ptx(_tma94_sink!, types; cap = v"10.0", feature_set = :arch)
        @test count("cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster::16b [",
                    ptx) == 5
    end
end
