# Exact-floor compiler evidence for PTX 9.3 §9.7.9.26.5.4 base im2col
# prefetch. This weak, fire-and-forget hint has no runtime claim.

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

function _tma_im2col_prefetch_callsite_attrs(llvm::AbstractString)
    groups = Dict{String,String}()
    for line in eachline(IOBuffer(llvm))
        m = match(r"^attributes #([0-9]+) = \{([^}]*)\}", strip(line))
        m === nothing || (groups[m.captures[1]] = m.captures[2])
    end
    calls = [String(line) for line in eachline(IOBuffer(llvm))
             if occursin(" call ", line) &&
                occursin(
                    "llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col", line)]
    attrs = String[]
    for call in calls
        m = match(r" #([0-9]+)(?:,|$)", strip(call))
        m === nothing && return calls, String[]
        push!(attrs, get(groups, m.captures[1], ""))
    end
    calls, attrs
end

@testset "TMA base-im2col prefetch at its sm_90 target floor" begin
    types = Tuple{PTX.TMADescriptorPtr, UInt64}
    @test ptxas_compiles(_tma_im2col_prefetch_surface!, types; cap = v"9.0")

    ptx = emit_ptx(_tma_im2col_prefetch_surface!, types; cap = v"9.0")
    @test occursin(".version 9.3", ptx)
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
    calls, attrs = _tma_im2col_prefetch_callsite_attrs(llvm)
    @test length(calls) == 6
    @test length(attrs) == 6
    @test all(attr -> occursin(r"\bconvergent\b", attr), attrs)
    @test all(attr -> occursin(r"\bnomerge\b", attr), attrs)
end

@testset "TMA base-im2col prefetch fails below sm_90" begin
    types = Tuple{PTX.TMADescriptorPtr, UInt64}
    err = try
        emit_ptx(_tma_im2col_prefetch_surface!, types; cap = v"8.9")
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("Cannot select", sprint(showerror, err))
    @test occursin("prefetch.im2col", sprint(showerror, err))
end
