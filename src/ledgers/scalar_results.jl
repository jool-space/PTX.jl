# Audited scalar forms whose destination ABI cannot be recovered from the
# terminal-modifier convention.  Keep this ledger closed: each entry fixes the
# complete modifier spelling, result carrier, operand carriers, PTX version,
# target floor/feature set, related ISA section, and whether its spelling comes
# from canonical grammar or an exact contradictory documented example verified
# with ptxas. Forms not present here continue through the generic scalar rule in
# types.jl.

struct ScalarResultSchema
    op::Symbol
    mods::Tuple{Vararg{Symbol}}
    rettype::Type
    operands::Tuple{Vararg{Symbol}}
    ptx_version::VersionNumber
    min_sm::Union{Nothing,VersionNumber}
    feature_set::Symbol
    provenance::Symbol
    section::String
end

function _scalar_result_schema!(schemas, op, mods, rettype, operands,
                                ptx_version, min_sm, feature_set, section;
                                provenance = :isa)
    feature_set in (:baseline, :family, :arch) ||
        error("invalid audited scalar-result feature set: ", feature_set)
    provenance in (:isa, :ptxas_compat) ||
        error("invalid audited scalar-result provenance: ", provenance)
    any(s -> s.op === op && s.mods == mods, schemas) &&
        error("duplicate audited scalar-result form: ", op, ".", join(mods, "."))
    push!(schemas, ScalarResultSchema(op, mods, rettype, operands,
                                     ptx_version, min_sm, feature_set,
                                     provenance, section))
end

# PTX ISA 9.3 §9.7.5.1-.3:
#   add/sub{.rnd}{.sat}.f32.{f16,bf16}
#   fma.rnd{.sat}.f32.{f16,bf16}
# All have an f32 destination even though the terminal token names the narrow
# multiplicand type. PTX 9.3's syntax places `.sat` before `.f32`, while its
# examples contain three contradictory post-type spellings. The canonical ISA
# grammar is recorded with `:isa` provenance; only those exact documented
# examples are retained as `:ptxas_compat`. Other ptxas-accepted permutations
# are deliberately not treated as legal PTX forms.
const _MIXED_ADD_SECTION =
    "ptx/9-instruction-set/9.7.5.1-mixed-precision-floating-point-instructions-add.md"
const _MIXED_SUB_SECTION =
    "ptx/9-instruction-set/9.7.5.2-mixed-precision-floating-point-instructions-sub.md"
const _MIXED_FMA_SECTION =
    "ptx/9-instruction-set/9.7.5.3-mixed-precision-floating-point-instructions-fma.md"

# PTX ISA 9.3 §9.7.1.15-.16: popc/clz always write a u32 destination;
# `.b32`/`.b64` name only the source width.
const _POPC_SECTION =
    "ptx/9-instruction-set/9.7.1.15-integer-arithmetic-instructions-popc.md"
const _CLZ_SECTION =
    "ptx/9-instruction-set/9.7.1.16-integer-arithmetic-instructions-clz.md"

# PTX ISA 9.3 §9.7.1.24-.25: dp2a/dp4a use a u32 accumulator/result only
# when *both* multiplicand types are u32; every mixed/signed form uses s32.
const _DP4A_SECTION =
    "ptx/9-instruction-set/9.7.1.24-integer-arithmetic-instructions-dp4a.md"
const _DP2A_SECTION =
    "ptx/9-instruction-set/9.7.1.25-integer-arithmetic-instructions-dp2a.md"

# PTX ISA 9.3 §9.7.1.3-.4: `.wide` names the multiplication mode. For 16-
# and 32-bit integer inputs, mul writes a twice-wide result and mad also takes
# its accumulator at that widened type. Both instructions exist on all targets.
const _MUL_SECTION =
    "ptx/9-instruction-set/9.7.1.3-integer-arithmetic-instructions-mul.md"
const _MAD_SECTION =
    "ptx/9-instruction-set/9.7.1.4-integer-arithmetic-instructions-mad.md"

# PTX ISA 9.3 §9.7.9.7: the six optimized-control modes retain the b32
# result and three b32 sources of base prmt, but their terminal tokens are
# mode selectors rather than result types.
const _PRMT_SECTION =
    "ptx/9-instruction-set/9.7.9.7-data-movement-and-conversion-instructions-prmt.md"

# Packed integer arithmetic uses a 32-bit carrier for every 2x16/4x8 result
# and source.  The sm_120f families require the Blackwell family feature set;
# recording only a numeric floor would incorrectly imply baseline sm_120.
const _PACKED_ADD_SECTION =
    "ptx/9-instruction-set/9.7.1.1-integer-arithmetic-instructions-add.md"
const _PACKED_SUB_SECTION =
    "ptx/9-instruction-set/9.7.1.2-integer-arithmetic-instructions-sub.md"
const _PACKED_NEG_SECTION =
    "ptx/9-instruction-set/9.7.1.12-integer-arithmetic-instructions-neg.md"
const _PACKED_MIN_SECTION =
    "ptx/9-instruction-set/9.7.1.13-integer-arithmetic-instructions-min.md"
const _PACKED_MAX_SECTION =
    "ptx/9-instruction-set/9.7.1.14-integer-arithmetic-instructions-max.md"

# PTX ISA 9.3 §9.7.9.23: every cvt.pack form writes u32.  The conversion
# type, source type, and optional cType are all input descriptors.
const _CVTPACK_SECTION =
    "ptx/9-instruction-set/9.7.9.23-data-movement-and-conversion-instructions-cvt.pack.md"

const SCALAR_RESULT_SCHEMAS = let schemas = ScalarResultSchema[]
    narrow_carrier = Dict(:f16 => :f16, :bf16 => :bf16)
    for (op, section) in ((:add, _MIXED_ADD_SECTION),
                          (:sub, _MIXED_SUB_SECTION))
        for atype in (:f16, :bf16), rnd in (nothing, :rn, :rz, :rm, :rp),
            prefix_sat in (false, true)
            prefix = rnd === nothing ? () : (rnd,)
            prefix_sat && (prefix = (prefix..., :sat))
            _scalar_result_schema!(schemas, op,
                                   (prefix..., :f32, atype),
                                   Float32, (narrow_carrier[atype], :f32),
                                   v"8.6", v"10.0", :baseline, section)
        end
    end
    for atype in (:f16, :bf16), rnd in (:rn, :rz, :rm, :rp),
        prefix_sat in (false, true)
        prefix = prefix_sat ? (rnd, :sat) : (rnd,)
        _scalar_result_schema!(schemas, :fma,
                               (prefix..., :f32, atype),
                               Float32,
                               (narrow_carrier[atype], narrow_carrier[atype], :f32),
                               v"8.6", v"10.0", :baseline,
                               _MIXED_FMA_SECTION)
    end
    for (op, mods, operands, section) in (
        (:add, (:rz, :f32, :bf16, :sat), (:bf16, :f32),
         _MIXED_ADD_SECTION),
        (:sub, (:rz, :f32, :f16, :sat), (:f16, :f32),
         _MIXED_SUB_SECTION),
        (:fma, (:rz, :sat, :f32, :f16, :sat),
         (:f16, :f16, :f32), _MIXED_FMA_SECTION),
    )
        _scalar_result_schema!(schemas, op, mods, Float32, operands,
                               v"8.6", v"10.0", :baseline, section;
                               provenance = :ptxas_compat)
    end

    for (op, section) in ((:popc, _POPC_SECTION), (:clz, _CLZ_SECTION)),
        width in (:b32, :b64)
        _scalar_result_schema!(schemas, op, (width,), UInt32, (width,),
                               v"2.0", v"2.0", :baseline, section)
    end

    for atype in (:u32, :s32), btype in (:u32, :s32)
        result = atype === :u32 && btype === :u32 ? UInt32 : Int32
        result_kind = result === UInt32 ? :u32 : :s32
        _scalar_result_schema!(schemas, :dp4a, (atype, btype), result,
                               (atype, btype, result_kind),
                               v"5.0", v"6.1", :baseline, _DP4A_SECTION)
        for mode in (:lo, :hi)
            _scalar_result_schema!(schemas, :dp2a, (mode, atype, btype), result,
                                   (atype, btype, result_kind),
                                   v"5.0", v"6.1", :baseline, _DP2A_SECTION)
        end
    end

    for (itype, result, result_kind) in (
        (:u16, UInt32, :u32), (:s16, Int32, :s32),
        (:u32, UInt64, :u64), (:s32, Int64, :s64),
    )
        _scalar_result_schema!(schemas, :mul, (:wide, itype), result,
                               (itype, itype), v"1.0", nothing, :baseline,
                               _MUL_SECTION)
        _scalar_result_schema!(schemas, :mad, (:wide, itype), result,
                               (itype, itype, result_kind),
                               v"1.0", nothing, :baseline, _MAD_SECTION)
    end

    for mode in (:f4e, :b4e, :rc8, :ecl, :ecr, :rc16)
        _scalar_result_schema!(schemas, :prmt, (:b32, mode), UInt32,
                               (:b32, :b32, :b32), v"2.0", v"2.0",
                               :baseline, _PRMT_SECTION)
    end

    # Canonical packed add/sub spellings. Alternate modifier placements remain
    # explicit entries too; none are inferred from the terminal token.
    for packed_type in (:u16x2, :s16x2)
        _scalar_result_schema!(schemas, :add, (packed_type,), UInt32,
                               (:b32, :b32), v"8.0", v"9.0", :baseline,
                               _PACKED_ADD_SECTION)
    end
    for packed_type in (:u16x2, :s16x2)
        _scalar_result_schema!(schemas, :add, (:sat, packed_type), UInt32,
                               (:b32, :b32), v"9.2", v"12.0", :family,
                               _PACKED_ADD_SECTION)
    end
    for packed_type in (:u8x4, :s8x4), sat in (false, true)
        mods = sat ? (:sat, packed_type) : (packed_type,)
        _scalar_result_schema!(schemas, :add, mods, UInt32, (:b32, :b32),
                               v"9.2", v"12.0", :family,
                               _PACKED_ADD_SECTION)
    end
    for packed_type in (:u8x4, :s8x4), sat in (false, true)
        mods = sat ? (:sat, packed_type) : (packed_type,)
        _scalar_result_schema!(schemas, :sub, mods, UInt32, (:b32, :b32),
                               v"9.2", v"12.0", :family,
                               _PACKED_SUB_SECTION)
    end
    # The ISA's add example places `.sat` after `.s8x4`, contradicting its
    # canonical prefix grammar. Retain that exact documented spelling only.
    _scalar_result_schema!(schemas, :add, (:s8x4, :sat), UInt32,
                           (:b32, :b32), v"9.2", v"12.0", :family,
                           _PACKED_ADD_SECTION;
                           provenance = :ptxas_compat)
    _scalar_result_schema!(schemas, :neg, (:s8x4,), UInt32, (:b32,),
                           v"9.2", v"12.0", :family, _PACKED_NEG_SECTION)

    for (op, section) in ((:min, _PACKED_MIN_SECTION),
                          (:max, _PACKED_MAX_SECTION))
        _scalar_result_schema!(schemas, op, (:u16x2,), UInt32, (:b32, :b32),
                               v"8.0", v"9.0", :baseline, section)
        for mods in ((:s16x2,), (:relu, :s16x2))
            _scalar_result_schema!(schemas, op, mods, UInt32, (:b32, :b32),
                                   v"8.0", v"9.0", :baseline, section)
        end
        _scalar_result_schema!(schemas, op, (:u8x4,), UInt32, (:b32, :b32),
                               v"9.2", v"12.0", :family, section)
        for mods in ((:s8x4,), (:relu, :s8x4))
            _scalar_result_schema!(schemas, op, mods, UInt32, (:b32, :b32),
                                   v"9.2", v"12.0", :family, section)
        end

        # Scalar `.relu.s32` has a conventional terminal result but belongs to
        # the same closed grammar island, preventing malformed relu orders from
        # falling through. It is normative PTX 8.0 syntax.
        _scalar_result_schema!(schemas, op, (:relu, :s32), Int32,
                               (:s32, :s32), v"8.0", v"9.0", :baseline,
                               section)
    end
    # Only min's ISA example contradicts the canonical prefix grammar.
    _scalar_result_schema!(schemas, :min, (:s16x2, :relu), UInt32,
                           (:b32, :b32), v"8.0", v"9.0", :baseline,
                           _PACKED_MIN_SECTION; provenance = :ptxas_compat)

    for convert_type in (:u16, :s16)
        _scalar_result_schema!(schemas, :cvt,
                               (:pack, :sat, convert_type, :s32),
                               UInt32, (:s32, :s32),
                               v"6.5", v"7.2", :baseline, _CVTPACK_SECTION)
    end
    for convert_type in (:u8, :s8, :u4, :s4, :u2, :s2)
        min_sm = convert_type in (:u8, :s8) ? v"7.2" : v"7.5"
        _scalar_result_schema!(schemas, :cvt,
                               (:pack, :sat, convert_type, :s32, :b32),
                               UInt32, (:s32, :s32, :b32),
                               v"6.5", min_sm, :baseline, _CVTPACK_SECTION)
    end

    # Deliberately a Vector, not Tuple(...): a several-hundred-element NTuple
    # constant makes every downstream generator/Dict build specialize on the
    # full tuple type. Those inference+codegen bombs tripled package
    # precompile time (measured 30s -> 12s converting the ledgers back).
    schemas
end

const _SCALAR_RESULT_SCHEMA_BY_FORM = Dict(
    (schema.op, schema.mods) => schema for schema in SCALAR_RESULT_SCHEMAS)
length(_SCALAR_RESULT_SCHEMA_BY_FORM) == length(SCALAR_RESULT_SCHEMAS) ||
    error("duplicate keys in SCALAR_RESULT_SCHEMAS")

schema(::ScalarLedger, op::Symbol, mods::Tuple{Vararg{Symbol}}) =
    get(_SCALAR_RESULT_SCHEMA_BY_FORM, (op, mods), nothing)

# The scalar island's routing predicate lives in `island_of`
# (src/ledgers/protocol.jl); this file owns the schemas it routes to.

result_type(schema::ScalarResultSchema) = schema.rettype

function miss(::ScalarLedger, op::Symbol,
              mods::Tuple{Vararg{Symbol}})
    spelling = isempty(mods) ? string(op) : string(op, ".", join(mods, "."))
    ArgumentError(
        "ptx\"$spelling\" is inside an audited scalar-result grammar " *
        "island but does not match a reviewed form. Refusing terminal-modifier " *
        "result inference; check modifier order and spelling. The raw tier " *
        "also rejects this miss because it cannot supply an explicit result ABI.")
end

function validate_scalar_result_args(schema::ScalarResultSchema, argtypes)
    spelling = string(schema.op, ".", join(schema.mods, "."))
    length(argtypes) == length(schema.operands) || throw(ArgumentError(
        "ptx\"$spelling\" is an audited scalar form with " *
        "$(length(schema.operands)) source operands, got $(length(argtypes)); " *
        "see $(schema.section)"))
    for (i, (kind, T)) in enumerate(zip(schema.operands, argtypes))
        operand_accepts(kind, T) && continue
        throw(ArgumentError(
            "ptx\"$spelling\" operand $i must use " *
            operand_description(kind) * ", got $T; see $(schema.section)"))
    end
    nothing
end

validate_ledger_args(::ScalarLedger, s::ScalarResultSchema, argtypes) =
    validate_scalar_result_args(s, argtypes)
