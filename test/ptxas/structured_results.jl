# Exhaustive offline compiler evidence for the 1,114-form structured-result
# ledger. CUDA 13's retained sm_75 floor covers general/f16 setp, lop3, and
# match; sm_90 covers bf16 setp and elect. No live GPU is used.

using LLVM.Interop: @asmcall

function _structured_raw_ptxas(source::AbstractString, target::AbstractString;
                               inspect_global::Bool = false)
    mktempdir() do dir
        ptx_path = joinpath(dir, "probe.ptx")
        cubin_path = joinpath(dir, "probe.cubin")
        write(ptx_path, source)
        cmd = `$(CUDACore.CUDA_Compiler.ptxas()) --gpu-name $target --output-file $cubin_path $ptx_path`
        log = IOBuffer()
        proc = run(pipeline(ignorestatus(cmd), stdout = log, stderr = log))
        diagnostics = String(take!(log))
        dump = if success(proc) && inspect_global
            readelf = Sys.which("readelf")
            readelf === nothing && error(
                "readelf is required by the ptxas constant-expression oracle")
            readelf_log = IOBuffer()
            readelf_proc = run(pipeline(
                ignorestatus(`$readelf -x .nv.global.init $cubin_path`),
                stdout = readelf_log, stderr = readelf_log))
            success(readelf_proc) || error(
                "readelf could not inspect ptxas global initializer:\n" *
                String(take!(readelf_log)))
            String(take!(readelf_log))
        else
            ""
        end
        success(proc), diagnostics, dump
    end
end

function _structured_constant_ptxas(expression::AbstractString)
    source = """.version 8.5
    .target sm_75
    .address_size 64
    .global .align 8 .u64 probe_value = $expression;
    """
    _structured_raw_ptxas(source, "sm_75"; inspect_global = true)
end

function _structured_setp_immediate_ptxas(dtype::AbstractString,
                                          literal::AbstractString)
    carrier = dtype == "bf16" ? "b16" : dtype
    source = """.version 8.5
    .target sm_90
    .address_size 64
    .visible .entry probe() {
        .reg .pred %p;
        .reg .$carrier %value;
        setp.eq.$dtype %p, %value, $literal;
        ret;
    }
    """
    _structured_raw_ptxas(source, "sm_90")
end

_structured_ptxas_partition(schema) =
    (schema.min_sm === nothing || schema.min_sm <= v"7.5") ? :sm75 :
    schema.min_sm == v"9.0" ? :sm90 :
    error("unpartitioned structured-result floor $(schema.min_sm) for " *
          PTX.build_head(schema.op, schema.mods))

_structured_ptxas_arg(kind) =
    kind === :pred ? :pred :
    kind === :imm8 ? :(Val(0x96)) :
    kind === :f16 ? :f16 :
    kind === :bf16 ? :bf16 :
    kind === :f32 ? :f32 :
    kind === :f64 ? :f64 :
    kind === :b16 ? :b16 :
    kind === :b32 ? :b32 :
    kind === :b64 ? :b64 :
    kind === :u16 ? :u16 :
    kind === :s16 ? :s16 :
    kind === :u32 ? :u32 :
    kind === :s32 ? :s32 :
    kind === :u64 ? :u64 :
    kind === :s64 ? :s64 :
    error("unknown structured ptxas operand kind $kind")

function _structured_ptxas_body(partition)
    body = Expr(:block)
    output_index = 0
    for schema in PTX.STRUCTURED_RESULT_SCHEMAS
        _structured_ptxas_partition(schema) === partition || continue
        args = _structured_ptxas_arg.(schema.operands)
        call = :(PTX.Operation{$(QuoteNode(schema.op)), $(schema.mods)}()(
                     $(args...)))
        result = gensym(:structured_result)
        push!(body.args, :(local $result = $call))
        for (lane, kind) in enumerate(schema.outputs)
            output_index += 1
            value = length(schema.outputs) == 1 ? result : :($result[$lane])
            stored = kind === :pred ? :($value ? UInt32(1) : UInt32(0)) : value
            push!(body.args, :(Base.@inbounds out[$output_index] = $stored))
        end
    end
    push!(body.args, :(return nothing))
    body
end

@generated function _structured_sm75!(
        out::CuDeviceVector{UInt32,1}, pred::Bool,
        b16::UInt16, b32::UInt32, b64::UInt64,
        u16::UInt16, s16::Int16, u32::UInt32, s32::Int32,
        u64::UInt64, s64::Int64,
        f16::Float16, bf16::UInt16, f32::Float32, f64::Float64)
    _structured_ptxas_body(:sm75)
end

@generated function _structured_sm90!(
        out::CuDeviceVector{UInt32,1}, pred::Bool,
        b16::UInt16, b32::UInt32, b64::UInt64,
        u16::UInt16, s16::Int16, u32::UInt32, s32::Int32,
        u64::UInt64, s64::Int64,
        f16::Float16, bf16::UInt16, f32::Float32, f64::Float64)
    _structured_ptxas_body(:sm90)
end

const _STRUCTURED_PTXAS_TYPES = Tuple{
    CuDeviceVector{UInt32,1}, Bool,
    UInt16, UInt32, UInt64,
    UInt16, Int16, UInt32, Int32, UInt64, Int64,
    Float16, UInt16, Float32, Float64,
}

@testset "all structured-result spellings assemble at retained/exact floors" begin
    partitions = (
        (_structured_sm75!, :sm75, 1001, 1501, v"7.5"),
        (_structured_sm90!, :sm90, 113, 170, v"9.0"),
    )
    for (kernel, partition, expected_forms, expected_outputs, cap) in partitions
        schemas = filter(s -> _structured_ptxas_partition(s) === partition,
                         PTX.STRUCTURED_RESULT_SCHEMAS)
        @test length(schemas) == expected_forms
        @test sum(length(s.outputs) for s in schemas) == expected_outputs
        @test ptxas_compiles(kernel, _STRUCTURED_PTXAS_TYPES; cap)
        emitted = emit_ptx(kernel, _STRUCTURED_PTXAS_TYPES; cap)
        @test occursin(".target sm_$(cap.major)$(cap.minor)", emitted)
        for head in unique(PTX.build_head(s.op, s.ptxmods) for s in schemas)
            @test occursin(head, emitted)
        end
        @test !occursin("setp.dual", emitted)
        @test !occursin(r"match\.(?:any|all)\.sync\.b(?:32|64)\.pred", emitted)
    end
end

# The ISA calls membermask a 32-bit integer, while its examples use both a
# literal and a register. Pin both through ptxas so the schema does not narrow
# this to one carrier interpretation during a later cleanup.
@noinline function _structured_match_masks!(out::CuDeviceVector{UInt32,1},
                                             value::UInt64,
                                             mask::UInt32)
    @inbounds out[1] = ptx"match.any.sync.b64"(value, mask)
    @inbounds out[2] = ptx"match.any.sync.b64"(value, Val(0xffffffff))
    return nothing
end

@testset "match membermask register and immediate both assemble" begin
    types = Tuple{CuDeviceVector{UInt32,1}, UInt64, UInt32}
    @test ptxas_compiles(_structured_match_masks!, types; cap = v"7.5")
    emitted = emit_ptx(_structured_match_masks!, types; cap = v"7.5")
    @test occursin(r"match\.any\.sync\.b64\s+%r[0-9]+,\s*%rd[0-9]+,\s*%r[0-9]+;",
                   emitted)
    @test occursin(r"match\.any\.sync\.b64\s+%r[0-9]+,\s*%rd[0-9]+,\s*(?:0xffffffff|4294967295);",
                   emitted)
end

@testset "ptxas pins ambiguous constant-expression semantics" begin
    # PTX 9.3 §4.5.5's prose and §4.5.6 Table 5 disagree about the type of a
    # bitwise result. Inspecting ptxas's global initializer proves the result
    # remains signed: its arithmetic right shift produces eight all-one bytes.
    ok, log, dump = _structured_constant_ptxas("((-1 & -1) >> 63)")
    @test ok
    @test isempty(log)
    @test occursin(r"ffffffff\s+ffffffff", lowercase(dump))

    # Constant-expression validation is eager rather than C-style
    # short-circuiting: an invalid RHS is diagnosed in both cases.
    for expression in ("(0 && (1 / 0))", "(1 || (1 / 0))")
        ok, log, dump = _structured_constant_ptxas(expression)
        @test !ok
        @test occursin("division by zero", log)
        @test isempty(dump)
    end
end

@testset "ptxas pins floating setp immediate boundary" begin
    # PTX §6.1 forbids automatic integer-to-float operand conversion. f32/f64
    # accept floating constants but reject integer constants; half/bfloat setp
    # accepts only compatible registers, even for a floating `0.0` literal.
    for (dtype, literal) in (("f32", "0.0"), ("f64", "0.0"))
        ok, log, dump = _structured_setp_immediate_ptxas(dtype, literal)
        @test ok
        @test isempty(log)
        @test isempty(dump)
    end
    for (dtype, literal) in (("f32", "0"), ("f64", "0"),
                             ("f16", "0.0"), ("bf16", "0.0"))
        ok, log, dump = _structured_setp_immediate_ptxas(dtype, literal)
        @test !ok
        @test occursin("Arguments mismatch", log)
        @test isempty(dump)
    end
end

# Exact sink syntax is a property of the PTX destination grammar, independent
# of PTX.jl's materialized tuple ABI. These probes cover every distinct policy:
# scalar sink, either side of an any-one pipe, d-only sink, and mandatory p.
@noinline function _structured_sink_syntax(a::UInt32, b::UInt32,
                                            h::Float16, packed::UInt32,
                                            pred::Bool)
    @asmcall("setp.eq.u32 _, \$0, \$1;", "r,r", true, Nothing,
             Tuple{UInt32,UInt32}, a, b)
    @asmcall("setp.eq.u32 _|\$0, \$1, \$2;", "=b,r,r", true, Bool,
             Tuple{UInt32,UInt32}, a, b)
    @asmcall("setp.eq.u32 \$0|_, \$1, \$2;", "=b,r,r", true, Bool,
             Tuple{UInt32,UInt32}, a, b)
    @asmcall("setp.eq.f16 _, \$0, \$1;", "h,h", true, Nothing,
             Tuple{Float16,Float16}, h, h)
    @asmcall("setp.eq.f16x2 _|\$0, \$1, \$2;", "=b,r,r", true, Bool,
             Tuple{UInt32,UInt32}, packed, packed)
    @asmcall("setp.eq.f16x2 \$0|_, \$1, \$2;", "=b,r,r", true, Bool,
             Tuple{UInt32,UInt32}, packed, packed)
    @asmcall("lop3.or.b32 _|\$0, \$1, \$2, \$3, 0x96, \$4;",
             "=b,r,r,r,b", true, Bool,
             Tuple{UInt32,UInt32,UInt32,Bool}, a, b, packed, pred)
    @asmcall("match.all.sync.b32 _, \$0, \$1;", "r,r,~{memory}", true,
             Nothing, Tuple{UInt32,UInt32}, a, b)
    @asmcall("match.all.sync.b32 _|\$0, \$1, \$2;",
             "=b,r,r,~{memory}", true, Bool,
             Tuple{UInt32,UInt32}, a, b)
    @asmcall("match.all.sync.b32 \$0|_, \$1, \$2;",
             "=r,r,r,~{memory}", true, UInt32,
             Tuple{UInt32,UInt32}, a, b)
    @asmcall("elect.sync _|\$0, \$1;", "=b,r,~{memory}", true, Bool,
             Tuple{UInt32}, b)
    return nothing
end

@testset "structured-result legal sink positions assemble" begin
    types = Tuple{UInt32,UInt32,Float16,UInt32,Bool}
    @test ptxas_compiles(_structured_sink_syntax, types; cap = v"9.0")
    emitted = emit_ptx(_structured_sink_syntax, types; cap = v"9.0")
    for spelling in ("setp.eq.u32 _", "setp.eq.u32 _|",
                     "setp.eq.f16 _", "setp.eq.f16x2 _|",
                     "lop3.or.b32 _|", "match.all.sync.b32 _",
                     "match.all.sync.b32 _|", "elect.sync _|")
        @test occursin(spelling, emitted)
    end
end

function _structured_callsite_attributes(llvm::AbstractString,
                                         needle::AbstractString)
    groups = Dict{String,String}()
    for line in eachline(IOBuffer(llvm))
        match_group = match(r"^attributes #([0-9]+) = \{([^}]*)\}",
                            strip(line))
        match_group === nothing ||
            (groups[match_group.captures[1]] = match_group.captures[2])
    end
    calls = [String(line) for line in eachline(IOBuffer(llvm))
             if occursin(" call ", line) && occursin(needle, line)]
    isempty(calls) && return String[]
    [begin
         attr = match(r" #([0-9]+)(?:,|$)", strip(call))
         attr === nothing ? "" : get(groups, attr.captures[1], "")
     end for call in calls]
end

@noinline function _structured_convergence_probe(value::UInt64,
                                                  mask::UInt32,
                                                  choose::Bool)
    if choose
        ptx"match.all.sync.b64.pred"(value, mask)
    else
        ptx"elect.sync"(mask)
    end
    return nothing
end

@testset "collective structured calls retain call-site convergence" begin
    types = Tuple{UInt64,UInt32,Bool}
    @test ptxas_compiles(_structured_convergence_probe, types; cap = v"9.0")
    llvm = emit_llvm(_structured_convergence_probe, types; cap = v"9.0")
    for needle in ("match.all.sync.b64", "elect.sync")
        attrs = _structured_callsite_attributes(llvm, needle)
        @test length(attrs) == 1
        @test occursin(r"\bconvergent\b", only(attrs))
        @test occursin(r"\bnomerge\b", only(attrs))
    end
end
