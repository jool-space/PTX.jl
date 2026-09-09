# Assembly coverage for the fixed-scalar-result ledger. PTX ISA 9.4 floors:
# popc/clz sm_20 (§§9.7.1.15-.16), dp2a/dp4a sm_61 (§§9.7.1.24-.25),
# cvt.pack sm_72 (sub-byte forms sm_75; §9.7.10.25), mixed-precision
# add/sub/fma sm_100 (§§9.7.5.1-.3), and packed 4x8 arithmetic on the
# {sm_107f, sm_120f} families (§§9.7.1.1-.2, .12-.14; sm_120f-only pre-9.4).
#
# CUDA 13 ptxas no longer accepts targets below sm_75, so older families
# assemble at that retained floor. The ledger is partitioned by target metadata
# rather than sampled: 38 retained-floor, 11 sm_90, 59 sm_100, and 18 sm_120f
# schemas, plus 36 PTX 9.4 packed mixed forms on sm_107f/sm_107a.
# Alternate x4 arithmetic adds 903 forms restricted to sm_100a/sm_103a.
# Exact historical floors remain pinned by the independent host oracle.

_scalar_ptxas_arg(kind) =
    kind === :f16  ? :f16 :
    kind === :bf16 ? :bf16 :
    kind === :f32  ? :f32 :
    kind === :u16  ? :u16 :
    kind === :s16  ? :s16 :
    kind === :u32  ? :u32 :
    kind === :s32  ? :s32 :
    kind === :u64  ? :u64 :
    kind === :s64  ? :s64 :
    kind === :b16  ? :u16 :
    kind === :b32  ? :u32 :
    kind === :b64  ? :u64 :
    error("unknown scalar ptxas operand kind $kind")

function _scalar_ptxas_partition(schema)
    schema.op === :set && return :set # Exhaustive partitions live in ptxas/set.jl.
    label = string(schema.op, ".", join(schema.mods, "."))
    if schema.ptx_version == v"9.4" && schema.feature_set === :arch &&
       schema.min_sm == v"10.0"
        return :alternate
    end
    if schema.ptx_version >= v"9.4" && schema.feature_set === :family &&
       schema.min_sm == v"10.7"
        return :sm107f
    end
    if schema.feature_set === :family
        # 10.7 is the packed-integer gate {sm_107f, sm_120f families} (PTX
        # ISA 9.4). These older forms assemble under sm_120f so that
        # compilers supporting pre-9.4 PTX retain full coverage.
        schema.min_sm in (v"10.7", v"12.0") || error(
            "$label has family feature_set but unpartitioned min_sm=" *
            repr(schema.min_sm) * "; add an exact ptxas target partition")
        return :sm120f
    end
    schema.feature_set === :baseline || error(
        "$label has unpartitioned feature_set=$(repr(schema.feature_set)), " *
        "min_sm=$(repr(schema.min_sm))")
    (schema.min_sm === nothing || schema.min_sm <= v"7.5") && return :sm75
    schema.min_sm == v"9.0" && return :sm90
    schema.min_sm == v"10.0" && return :sm100
    error("$label has unpartitioned baseline min_sm=$(repr(schema.min_sm)); " *
          "add an exact ptxas target partition")
end

function _scalar_ptxas_body(partition)
    body = Expr(:block)
    counts = Dict(Float32 => 0, UInt32 => 0, Int32 => 0,
                  UInt64 => 0, Int64 => 0)
    outputs = Dict(Float32 => :out_f32, UInt32 => :out_u32,
                   Int32 => :out_s32, UInt64 => :out_u64,
                   Int64 => :out_s64)
    for schema in PTX.SCALAR_RESULT_SCHEMAS
        _scalar_ptxas_partition(schema) === partition || continue
        counts[schema.rettype] += 1
        dst = outputs[schema.rettype]
        index = counts[schema.rettype]
        args = [_scalar_ptxas_arg(kind) for kind in schema.operands]
        op = schema.op
        mods = schema.mods
        call = :(PTX.Operation{$(QuoteNode(op)), $mods}()($(args...)))
        push!(body.args, :(Base.@inbounds $dst[$index] = $call))
    end
    push!(body.args, :(return nothing))
    body
end

@generated function _ptxas_sm75_scalar_results!(
        out_f32::CuDeviceVector{Float32,1},
        out_u32::CuDeviceVector{UInt32,1},
        out_s32::CuDeviceVector{Int32,1},
        out_u64::CuDeviceVector{UInt64,1},
        out_s64::CuDeviceVector{Int64,1},
        f16::Float16, bf16::BFloat16, f32::Float32,
        u16::UInt16, s16::Int16, u32::UInt32, s32::Int32,
        u64::UInt64, s64::Int64)
    _scalar_ptxas_body(:sm75)
end

@generated function _ptxas_sm90_scalar_results!(
        out_f32::CuDeviceVector{Float32,1},
        out_u32::CuDeviceVector{UInt32,1},
        out_s32::CuDeviceVector{Int32,1},
        out_u64::CuDeviceVector{UInt64,1},
        out_s64::CuDeviceVector{Int64,1},
        f16::Float16, bf16::BFloat16, f32::Float32,
        u16::UInt16, s16::Int16, u32::UInt32, s32::Int32,
        u64::UInt64, s64::Int64)
    _scalar_ptxas_body(:sm90)
end

@generated function _ptxas_sm100_scalar_results!(
        out_f32::CuDeviceVector{Float32,1},
        out_u32::CuDeviceVector{UInt32,1},
        out_s32::CuDeviceVector{Int32,1},
        out_u64::CuDeviceVector{UInt64,1},
        out_s64::CuDeviceVector{Int64,1},
        f16::Float16, bf16::BFloat16, f32::Float32,
        u16::UInt16, s16::Int16, u32::UInt32, s32::Int32,
        u64::UInt64, s64::Int64)
    _scalar_ptxas_body(:sm100)
end

@generated function _ptxas_sm120f_scalar_results!(
        out_f32::CuDeviceVector{Float32,1},
        out_u32::CuDeviceVector{UInt32,1},
        out_s32::CuDeviceVector{Int32,1},
        out_u64::CuDeviceVector{UInt64,1},
        out_s64::CuDeviceVector{Int64,1},
        f16::Float16, bf16::BFloat16, f32::Float32,
        u16::UInt16, s16::Int16, u32::UInt32, s32::Int32,
        u64::UInt64, s64::Int64)
    _scalar_ptxas_body(:sm120f)
end

@generated function _ptxas_sm107f_scalar_results!(
        out_f32::CuDeviceVector{Float32,1},
        out_u32::CuDeviceVector{UInt32,1},
        out_s32::CuDeviceVector{Int32,1},
        out_u64::CuDeviceVector{UInt64,1},
        out_s64::CuDeviceVector{Int64,1},
        f16::Float16, bf16::BFloat16, f32::Float32,
        u16::UInt16, s16::Int16, u32::UInt32, s32::Int32,
        u64::UInt64, s64::Int64)
    _scalar_ptxas_body(:sm107f)
end

@generated function _ptxas_alternate_scalar_results!(
        out_f32::CuDeviceVector{Float32,1},
        out_u32::CuDeviceVector{UInt32,1},
        out_s32::CuDeviceVector{Int32,1},
        out_u64::CuDeviceVector{UInt64,1},
        out_s64::CuDeviceVector{Int64,1},
        f16::Float16, bf16::BFloat16, f32::Float32,
        u16::UInt16, s16::Int16, u32::UInt32, s32::Int32,
        u64::UInt64, s64::Int64)
    _scalar_ptxas_body(:alternate)
end

const _SCALAR_ALL_TYPES = Tuple{
    CuDeviceVector{Float32,1}, CuDeviceVector{UInt32,1},
    CuDeviceVector{Int32,1}, CuDeviceVector{UInt64,1},
    CuDeviceVector{Int64,1}, Float16, BFloat16, Float32,
    UInt16, Int16, UInt32, Int32, UInt64, Int64,
}
@testset "fixed scalar-result forms assemble at retained/exact floors" begin
    partitions = (
        (_ptxas_sm75_scalar_results!, :sm75, 38, v"7.5", :baseline),
        (_ptxas_sm90_scalar_results!, :sm90, 11, v"9.0", :baseline),
        (_ptxas_sm100_scalar_results!, :sm100, 59, v"10.0", :baseline),
        (_ptxas_sm120f_scalar_results!, :sm120f, 18, v"12.0", :family),
    )
    for (kernel, partition, expected_count, cap, feature_set) in partitions
        schemas = filter(s -> _scalar_ptxas_partition(s) === partition,
                         PTX.SCALAR_RESULT_SCHEMAS)
        @test length(schemas) == expected_count
        @test ptxas_compiles(kernel, _SCALAR_ALL_TYPES; cap, feature_set)
        emitted = emit_ptx(kernel, _SCALAR_ALL_TYPES; cap, feature_set)
        for schema in schemas
            @test occursin(PTX.build_head(schema.op, schema.mods), emitted)
        end
    end
end

@testset "PTX 9.4 packed mixed forms assemble only on the sm_107 family" begin
    schemas = filter(s -> _scalar_ptxas_partition(s) === :sm107f,
                     PTX.SCALAR_RESULT_SCHEMAS)
    @test length(schemas) == 36
    @test count(s -> s.rettype === UInt64, schemas) == 28
    @test count(s -> s.rettype === UInt32, schemas) == 8
    if _ptxas_isa() < v"9.4"
        @test_skip "PTX 9.4 assembler required for sm_107"
    else
        for feature_set in (:family, :arch)
            @test ptxas_compiles(_ptxas_sm107f_scalar_results!, _SCALAR_ALL_TYPES;
                                 cap = v"10.7", feature_set)
            emitted = emit_ptx(_ptxas_sm107f_scalar_results!, _SCALAR_ALL_TYPES;
                               cap = v"10.7", feature_set)
            suffix = feature_set === :family ? "f" : "a"
            @test occursin(".target sm_107$suffix", emitted)
            for schema in schemas
                @test occursin(PTX.build_head(schema.op, schema.mods), emitted)
            end
        end
        for (cap, feature_set, target) in ((v"10.7", :baseline, "sm_107"),
                                           (v"10.3", :family, "sm_103f"),
                                           (v"12.0", :family, "sm_120f"))
            err = try
                ptxas_compiles(_ptxas_sm107f_scalar_results!, _SCALAR_ALL_TYPES;
                               cap, feature_set)
                nothing
            catch caught
                caught
            end
            @test err isa ErrorException
            @test occursin("Failed to compile PTX code", sprint(showerror, err))
            @test occursin("not supported on .target '$target'", sprint(showerror, err))
        end
    end
end

function _alternate_target_probe!(out::CuDeviceVector{UInt32,1},
                                  compact::UInt16, packed::UInt32)
    @inbounds begin
        out[1] = ptx"add.rn.e4m3x4.e2m1x4"(compact, packed)
        out[2] = ptx"sub.rn.satfinite.e5m2x4.e2m1p4x4"(packed, packed)
        out[3] = ptx"mul.rn.e5m2x4.e2m1x4.e2m1x4"(compact, compact)
        out[4] = ptx"fma.rn.e4m3x4.ue8m0x4.e3m2x4"(packed, packed, packed)
    end
    nothing
end

@testset "alternate packed forms assemble on exactly sm_100a and sm_103a" begin
    schemas = filter(s -> _scalar_ptxas_partition(s) === :alternate,
                     PTX.SCALAR_RESULT_SCHEMAS)
    @test length(schemas) == 903
    @test count(s -> s.provenance === :isa, schemas) == 896
    @test count(s -> s.provenance === :ptxas_compat, schemas) == 7
    @test count(s -> :b16 in s.operands, schemas) == 226
    if _ptxas_isa() < v"9.4"
        @test_skip "PTX 9.4 assembler required for alternate packed arithmetic"
    else
        for (cap, target) in ((v"10.0", "sm_100a"), (v"10.3", "sm_103a"))
            @test ptxas_compiles(_ptxas_alternate_scalar_results!, _SCALAR_ALL_TYPES;
                                 cap, feature_set = :arch)
            emitted = emit_ptx(_ptxas_alternate_scalar_results!, _SCALAR_ALL_TYPES;
                               cap, feature_set = :arch)
            @test occursin(".target $target", emitted)
            for schema in schemas
                @test occursin(PTX.build_head(schema.op, schema.mods), emitted)
            end
        end
        types = Tuple{CuDeviceVector{UInt32,1}, UInt16, UInt32}
        for (cap, feature_set, target) in (
            (v"10.0", :baseline, "sm_100"), (v"10.3", :baseline, "sm_103"),
            (v"10.0", :family, "sm_100f"), (v"10.3", :family, "sm_103f"),
            (v"10.7", :arch, "sm_107a"), (v"12.0", :arch, "sm_120a"),
        )
            err = try
                ptxas_compiles(_alternate_target_probe!, types; cap, feature_set)
                nothing
            catch caught
                caught
            end
            @test err isa ErrorException
            @test occursin("Failed to compile PTX code", sprint(showerror, err))
            @test occursin("not supported on .target '$target'", sprint(showerror, err))
        end
    end
end
