# Assembly coverage for the fixed-scalar-result ledger. PTX ISA 9.3 floors:
# popc/clz sm_20 (§§9.7.1.15-.16), dp2a/dp4a sm_61 (§§9.7.1.24-.25),
# cvt.pack sm_72 (sub-byte forms sm_75; §9.7.9.23), mixed-precision
# add/sub/fma sm_100 (§§9.7.5.1-.3), and packed 4x8 arithmetic on
# family-specific sm_120f (§§9.7.1.1-.2, .12-.14).
#
# CUDA 13 ptxas no longer accepts targets below sm_75, so older families
# assemble at that retained floor. The ledger is partitioned by target metadata
# rather than sampled: 38 retained-floor, 11 sm_90, 59 sm_100, and 18 sm_120f
# schemas. Exact historical floors remain pinned by the independent host oracle.

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
    kind === :b32  ? :u32 :
    kind === :b64  ? :u64 :
    error("unknown scalar ptxas operand kind $kind")

function _scalar_ptxas_partition(schema)
    label = string(schema.op, ".", join(schema.mods, "."))
    if schema.feature_set === :family
        schema.min_sm == v"12.0" || error(
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
        f16::Float16, bf16::UInt16, f32::Float32,
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
        f16::Float16, bf16::UInt16, f32::Float32,
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
        f16::Float16, bf16::UInt16, f32::Float32,
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
        f16::Float16, bf16::UInt16, f32::Float32,
        u16::UInt16, s16::Int16, u32::UInt32, s32::Int32,
        u64::UInt64, s64::Int64)
    _scalar_ptxas_body(:sm120f)
end

const _SCALAR_ALL_TYPES = Tuple{
    CuDeviceVector{Float32,1}, CuDeviceVector{UInt32,1},
    CuDeviceVector{Int32,1}, CuDeviceVector{UInt64,1},
    CuDeviceVector{Int64,1}, Float16, UInt16, Float32,
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
