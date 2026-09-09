# Exact-floor compiler evidence for PTX 9.3 §9.7.9.26.5.4 tile prefetch.
# Cache policy is a performance hint, so this test intentionally stops at
# optimized LLVM, emitted PTX, and ptxas rather than claiming runtime semantics.

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

function _tma_prefetch_callsite_attrs(llvm::AbstractString)
    groups = Dict{String,String}()
    for line in eachline(IOBuffer(llvm))
        m = match(r"^attributes #([0-9]+) = \{([^}]*)\}", strip(line))
        m === nothing || (groups[m.captures[1]] = m.captures[2])
    end
    # Single-route asm since the demotion: the call sites are inline asm
    # (convergent_asm_ir), not llvm.nvvm.* intrinsic calls.
    calls = [String(line) for line in eachline(IOBuffer(llvm))
             if occursin(" asm ", line) &&
                occursin("cp.async.bulk.prefetch.tensor", line)]
    attrs = String[]
    for call in calls
        m = match(r" #([0-9]+)(?:,|$)", strip(call))
        m === nothing && return calls, String[]
        push!(attrs, get(groups, m.captures[1], ""))
    end
    calls, attrs
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
    calls, attrs = _tma_prefetch_callsite_attrs(llvm)
    @test length(calls) == 10
    @test length(attrs) == 10
    @test all(attr -> occursin(r"\bconvergent\b", attr), attrs)
    @test all(attr -> occursin(r"\bnomerge\b", attr), attrs)
end
