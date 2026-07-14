# Closed PTX 9.3 result/operand grammar for the registered instruction
# families whose destination shape cannot be recovered from the terminal
# modifier convention.  This ledger deliberately includes the scalar siblings
# of the grouped forms: otherwise a typo could leave the reviewed island and
# silently re-enter the permissive scalar path.

struct StructuredResultSchema
    op::Symbol
    mods::Tuple{Vararg{Symbol}}       # PTX.jl API modifiers
    ptxmods::Tuple{Vararg{Symbol}}    # emitted PTX modifiers
    outputs::Tuple{Vararg{Symbol}}
    sinkable::Tuple{Vararg{Bool}}     # PTX `_` accepted at each output
    max_sinks::Int                    # `_ |_` is rejected by ptxas
    operands::Tuple{Vararg{Symbol}}
    ptx_version::VersionNumber
    min_sm::Union{Nothing,VersionNumber}
    feature_set::Symbol
    section::String
end

function _structured_result_schema!(schemas, op, mods, ptxmods, outputs,
                                    operands, ptx_version, min_sm, section;
                                    sinkable = ntuple(_ -> false,
                                                      length(outputs)))
    any(s -> s.op === op && s.mods == mods, schemas) &&
        error("duplicate structured-result form: ", op, ".", join(mods, "."))
    length(sinkable) == length(outputs) ||
        error("structured-result sink policy does not match output arity")
    max_sinks = any(sinkable) ? 1 : 0
    push!(schemas, StructuredResultSchema(op, mods, ptxmods, outputs,
                                          Tuple(sinkable), max_sinks, operands,
                                          ptx_version, min_sm, :baseline,
                                          section))
end

const _GENERAL_SETP_SECTION =
    "ptx/9-instruction-set/9.7.6.2-comparison-and-selection-instructions-setp.md"
const _HALF_SETP_SECTION =
    "ptx/9-instruction-set/9.7.7.2-half-precision-comparison-instructions-setp.md"
const _LOP3_SECTION =
    "ptx/9-instruction-set/9.7.8.6-logic-and-shift-instructions-lop3.md"
const _MATCH_SECTION =
    "ptx/9-instruction-set/9.7.14.11-parallel-synchronization-and-communication-instructions-match.sync.md"
const _ELECT_SECTION =
    "ptx/9-instruction-set/9.7.14.15-parallel-synchronization-and-communication-instructions-elect.sync.md"

const _SETP_BASIC_CMPS = (:eq, :ne, :lt, :le, :gt, :ge)
const _SETP_FLOAT_CMPS =
    (:eq, :ne, :lt, :le, :gt, :ge,
     :equ, :neu, :ltu, :leu, :gtu, :geu, :num, :nan)
const _SETP_BOOL_OPS = (nothing, :and, :or, :xor)

function _setp_mods(cmp::Symbol, boolop, ftz::Bool, dtype::Symbol)
    middle = boolop === nothing ? () : (boolop,)
    ftzmods = ftz ? (:ftz,) : ()
    (cmp, middle..., ftzmods..., dtype)
end

const STRUCTURED_RESULT_SCHEMAS = let schemas = StructuredResultSchema[]
    # PTX 9.3 §9.7.6.2. The same emitted spelling admits either `p` or
    # `p|q`; PTX.jl's leading :dual selector chooses the latter and is never
    # printed. CmpOp legality comes from the section's integer/bit/FP notes.
    general_types = (
        ((:b16, :b32, :b64), (:eq, :ne), false),
        ((:s16, :s32, :s64), _SETP_BASIC_CMPS, false),
        ((:u16, :u32, :u64),
         (:eq, :ne, :lt, :le, :gt, :ge, :lo, :ls, :hi, :hs), false),
        ((:f64,), _SETP_FLOAT_CMPS, false),
        ((:f32,), _SETP_FLOAT_CMPS, true),
    )
    for (dtypes, cmps, admits_ftz) in general_types, dtype in dtypes,
        cmp in cmps, boolop in _SETP_BOOL_OPS,
        ftz in (admits_ftz ? (false, true) : (false,))
        ptxmods = _setp_mods(cmp, boolop, ftz, dtype)
        operands = boolop === nothing ? (dtype, dtype) : (dtype, dtype, :pred)
        min_sm = dtype === :f64 ? v"1.3" : nothing
        _structured_result_schema!(schemas, :setp, ptxmods, ptxmods,
                                   (:pred,), operands, v"1.0", min_sm,
                                   _GENERAL_SETP_SECTION;
                                   sinkable = (true,))
        _structured_result_schema!(schemas, :setp, (:dual, ptxmods...),
                                   ptxmods, (:pred, :pred), operands,
                                   v"1.0", min_sm, _GENERAL_SETP_SECTION;
                                   sinkable = (true, true))
    end

    # PTX 9.3 §9.7.7.2. Scalar f16/bf16 has one predicate. Packed x2
    # compares always return lane-0|lane-1 predicates; unlike general dual
    # setp, the second predicate is not the complement of the first.
    for (dtype, input, packed, ptx_version, min_sm, admits_ftz) in (
        (:f16,    :f16, false, v"4.2", v"5.3", true),
        (:f16x2,  :b32, true,  v"4.2", v"5.3", true),
        (:bf16,   :bf16, false, v"7.8", v"9.0", false),
        (:bf16x2, :b32, true,  v"7.8", v"9.0", false),
    ), cmp in _SETP_FLOAT_CMPS, boolop in _SETP_BOOL_OPS,
        ftz in (admits_ftz ? (false, true) : (false,))
        ptxmods = _setp_mods(cmp, boolop, ftz, dtype)
        operands = boolop === nothing ? (input, input) : (input, input, :pred)
        outputs = packed ? (:pred, :pred) : (:pred,)
        _structured_result_schema!(schemas, :setp, ptxmods, ptxmods,
                                   outputs, operands, ptx_version, min_sm,
                                   _HALF_SETP_SECTION;
                                   # §9.2 permits the bit bucket for optional
                                   # destinations. CUDA 13 ptxas independently
                                   # pins scalar and both packed-half positions.
                                   sinkable = ntuple(_ -> true,
                                                     length(outputs)))
    end

    # PTX 9.3 §9.7.8.6. immLut is an integer constant in 0:255. BoolOp
    # forms add a predicate result and predicate input.
    _structured_result_schema!(schemas, :lop3, (:b32,), (:b32,), (:b32,),
                               (:b32, :b32, :b32, :imm8),
                               v"4.3", v"5.0", _LOP3_SECTION)
    for boolop in (:or, :and)
        mods = (boolop, :b32)
        _structured_result_schema!(schemas, :lop3, mods, mods,
                                   (:b32, :pred),
                                   (:b32, :b32, :b32, :imm8, :pred),
                                   v"8.2", v"7.0", _LOP3_SECTION;
                                   sinkable = (true, false))
    end

    # PTX 9.3 §9.7.14.15. Both destinations are mandatory; d is a b32
    # lane id and p identifies the elected thread.
    _structured_result_schema!(schemas, :elect, (:sync,), (:sync,),
                               (:b32, :pred), (:b32,),
                               v"8.0", v"9.0", _ELECT_SECTION;
                               sinkable = (true, false))

    # PTX 9.3 §9.7.14.11. d is always b32 even when a is b64. The trailing
    # :pred API selector chooses match.all's optional predicate destination.
    for dtype in (:b32, :b64)
        for mode in (:any, :all)
            mods = (mode, :sync, dtype)
            _structured_result_schema!(schemas, :match, mods, mods, (:b32,),
                                       (dtype, :b32), v"6.0", v"7.0",
                                       _MATCH_SECTION;
                                       # Only match.all's mode description
                                       # permits result discard; ptxas rejects
                                       # `match.any ... _`.
                                       sinkable = (mode === :all,))
            mode === :all || continue
            _structured_result_schema!(schemas, :match, (mods..., :pred),
                                       mods, (:b32, :pred), (dtype, :b32),
                                       v"6.0", v"7.0", _MATCH_SECTION;
                                       sinkable = (true, true))
        end
    end

    Tuple(schemas)
end

const _STRUCTURED_RESULT_SCHEMA_BY_FORM = Dict(
    (schema.op, schema.mods) => schema for schema in STRUCTURED_RESULT_SCHEMAS)
length(_STRUCTURED_RESULT_SCHEMA_BY_FORM) == length(STRUCTURED_RESULT_SCHEMAS) ||
    error("duplicate keys in STRUCTURED_RESULT_SCHEMAS")
length(STRUCTURED_RESULT_SCHEMAS) == 1114 ||
    error("structured-result ledger count changed; re-audit PTX 9.3 grammar")

structured_result_schema(op::Symbol, mods::Tuple{Vararg{Symbol}}) =
    get(_STRUCTURED_RESULT_SCHEMA_BY_FORM, (op, mods), nothing)

requires_structured_result_schema(op::Symbol) =
    op in (:setp, :lop3, :match, :elect)

function structured_result_schema_miss(op::Symbol,
                                       mods::Tuple{Vararg{Symbol}})
    spelling = isempty(mods) ? string(op) : string(op, ".", join(mods, "."))
    ArgumentError(
        "ptx\"$spelling\" is inside the closed PTX 9.3 result grammar " *
        "for setp/lop3/match/elect but does not match a reviewed form. " *
        "Refusing to guess destination grouping, operand carriers, or " *
        "modifier legality; the raw tier cannot declare an alternative " *
        "result ABI.")
end

_structured_output_type(kind::Symbol) =
    kind === :pred ? Bool :
    kind === :b32  ? UInt32 :
    error("unknown structured-result output kind: $kind")

function structured_result_type(schema::StructuredResultSchema)
    types = map(_structured_output_type, schema.outputs)
    length(types) == 1 ? only(types) : Core.apply_type(Tuple, types...)
end

_structured_integer_immediate(::Type{Val{V}}) where {V} =
    V isa Integer && !(V isa Bool)
_structured_integer_immediate(::Type) = false

function _structured_operand_accepts(kind::Symbol, ::Type{T}) where {T}
    if kind === :pred
        return T === Bool
    elseif kind === :imm8
        return T <: Val && _structured_integer_immediate(T) &&
               0 <= T.parameters[1] <= 255
    elseif kind === :f16
        return T === Float16 || T === UInt16
    elseif kind === :bf16
        return T === UInt16
    elseif kind === :f32
        return T === Float32 || T === UInt32
    elseif kind === :f64
        return T === Float64 || T === UInt64
    elseif kind === :b16
        return T in (UInt16, Int16, Float16) ||
               _structured_integer_immediate(T)
    elseif kind === :b32
        return T in (UInt32, Int32, Float32) ||
               _structured_integer_immediate(T)
    elseif kind === :b64
        return T in (UInt64, Int64, Float64) ||
               _structured_integer_immediate(T)
    elseif kind in (:u16, :s16)
        return T in (UInt16, Int16) || _structured_integer_immediate(T)
    elseif kind in (:u32, :s32)
        return T in (UInt32, Int32) || _structured_integer_immediate(T)
    elseif kind in (:u64, :s64)
        return T in (UInt64, Int64) || _structured_integer_immediate(T)
    end
    return false
end

function _structured_operand_description(kind::Symbol)
    kind === :pred && return "Bool predicate"
    kind === :imm8 && return "Val{N} with integer N in 0:255"
    kind === :f16 && return "Float16 or UInt16 bit carrier"
    kind === :bf16 && return "UInt16 (.b16 carrier for bf16)"
    kind === :f32 && return "Float32 or UInt32 bit carrier"
    kind === :f64 && return "Float64 or UInt64 bit carrier"
    string(kind, "-compatible scalar")
end

function validate_structured_result_args(schema::StructuredResultSchema,
                                         argtypes)
    spelling = string(schema.op, ".", join(schema.mods, "."))
    length(argtypes) == length(schema.operands) || throw(ArgumentError(
        "ptx\"$spelling\" is an audited result form with " *
        "$(length(schema.operands)) source operands, got $(length(argtypes)); " *
        "see $(schema.section)"))
    for (i, (kind, T)) in enumerate(zip(schema.operands, argtypes))
        _structured_operand_accepts(kind, T) && continue
        throw(ArgumentError(
            "ptx\"$spelling\" operand $i must use " *
            _structured_operand_description(kind) * ", got $T; see " *
            schema.section))
    end
    nothing
end
