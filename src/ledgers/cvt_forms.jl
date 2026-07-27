# Ordinary `cvt` has a result ABI and a source ABI that point in opposite
# directions: the penultimate type token names the destination, while the
# terminal token names the source.  Direct Julia calls already carry their
# source types in the arguments.  The PTX transpiler instead has to recover
# those carriers from the instruction spelling before turning PTX constants
# into Julia expressions.
#
# This is deliberately an operand-ABI ledger, not a complete modifier-grammar
# validator.  Rounding, saturation, and target legality remain ptxas's job;
# the closed boundary here prevents an unreviewed destination/source pair from
# receiving a plausible but wrong Julia carrier.

const ORDINARY_CVT_SECTION =
    "ptx/9-instruction-set/9.7.9.22-data-movement-and-conversion-instructions-cvt.md"

struct OrdinaryCvtSourceSchema
    destination::Symbol
    source::Symbol
    operands::Tuple{Vararg{Symbol}}
    stochastic::Bool
    scaled::Bool
    vector_source::Bool
    section::String
end

# PTX ISA 9.3 §9.7.9.22 has twelve fundamental source types.  Alternate
# floating-point formats live in bit-size registers (§5.2.3), so their Julia
# carrier is a bit type rather than a numerically similar fundamental float.
# e2m1x2 is physically b8, but NVPTX has no i8 inline-asm constraint; PTX.jl's
# exact cvt wrappers bridge it through UInt16, hence the reviewed b16 carrier.
const ORDINARY_CVT_SOURCE_CARRIERS = (
    :u8 => :u8, :u16 => :u16, :u32 => :u32, :u64 => :u64,
    :s8 => :s8, :s16 => :s16, :s32 => :s32, :s64 => :s64,
    :f16 => :f16, :f32 => :f32, :f64 => :f64, :bf16 => :b16,
    :f16x2 => :b32, :bf16x2 => :b32,
    :e4m3x2 => :b16, :e5m2x2 => :b16,
    :e2m1x2 => :b16, :e2m3x2 => :b16, :e3m2x2 => :b16,
    :ue8m0x2 => :b16, :s2f6x2 => :b16,
)

const _ORDINARY_CVT_SOURCE_CARRIER = Dict(ORDINARY_CVT_SOURCE_CARRIERS)
length(_ORDINARY_CVT_SOURCE_CARRIER) == 21 ||
    error("ordinary cvt source-carrier ledger must contain 21 unique formats")

const _CVT_FUNDAMENTAL_TYPES =
    (:u8, :u16, :u32, :u64, :s8, :s16, :s32, :s64,
     :bf16, :f16, :f32, :f64)

const _CVT_PACK2_FROM_F32 =
    (:f16x2, :bf16x2, :e4m3x2, :e5m2x2, :e2m1x2,
     :e2m3x2, :e3m2x2, :ue8m0x2, :s2f6x2)

const _CVT_PACK4_FROM_F32 =
    (:e4m3x4, :e5m2x4, :e2m1x4, :e2m3x4, :e3m2x4)

const _CVT_SCALED_PAIRS = Set{Tuple{Symbol, Symbol}}((
    (:bf16x2, :e4m3x2), (:bf16x2, :e5m2x2),
    (:bf16x2, :e2m1x2), (:bf16x2, :e2m3x2),
    (:bf16x2, :e3m2x2), (:bf16x2, :s2f6x2),
    (:s2f6x2, :f32), (:s2f6x2, :bf16x2),
))

function _ordinary_cvt_special_pairs()
    pairs = Tuple{Symbol, Symbol}[]

    # Two f32 values pack into f16x2/bf16x2.
    append!(pairs, ((:f16x2, :f32), (:bf16x2, :f32)))
    push!(pairs, (:tf32, :f32))

    # FP8/FP4/FP6 x2 conversions: down-convert from f32 or either packed
    # 16-bit format, and up-convert into f16x2/bf16x2.
    for dst in (:e4m3x2, :e5m2x2), src in (:f32, :f16x2, :bf16x2)
        push!(pairs, (dst, src))
    end
    for dst in (:f16x2, :bf16x2), src in (:e4m3x2, :e5m2x2)
        push!(pairs, (dst, src))
    end
    for dst in (:e2m1x2,), src in (:f32, :f16x2, :bf16x2)
        push!(pairs, (dst, src))
    end
    for dst in (:f16x2, :bf16x2), src in (:e2m1x2,)
        push!(pairs, (dst, src))
    end
    for dst in (:e2m3x2, :e3m2x2), src in (:f32, :f16x2, :bf16x2)
        push!(pairs, (dst, src))
    end
    for dst in (:f16x2, :bf16x2), src in (:e2m3x2, :e3m2x2)
        push!(pairs, (dst, src))
    end

    append!(pairs, ((:ue8m0x2, :f32), (:ue8m0x2, :bf16x2),
                    (:bf16x2, :ue8m0x2),
                    (:s2f6x2, :f32), (:s2f6x2, :bf16x2),
                    (:bf16x2, :s2f6x2)))

    for dst in _CVT_PACK4_FROM_F32
        push!(pairs, (dst, :f32))
    end

    length(pairs) == 39 ||
        error("ordinary cvt special-pair ledger must contain 39 forms")
    length(Set(pairs)) == length(pairs) ||
        error("duplicate ordinary cvt special destination/source pair")
    # Deliberately a Vector, not Tuple(...): a several-hundred-element NTuple
    # constant makes every downstream generator/Dict build specialize on the
    # full tuple type. Those inference+codegen bombs tripled package
    # precompile time (measured 30s -> 12s converting the ledgers back).
    pairs
end

const _CVT_SPECIAL_PAIRS = _ordinary_cvt_special_pairs()

function _ordinary_cvt_schema!(schemas, destination, source, operands;
                               stochastic = false, scaled = false,
                               vector_source = false)
    push!(schemas, OrdinaryCvtSourceSchema(
        destination, source, Tuple(operands), stochastic, scaled,
        vector_source, ORDINARY_CVT_SECTION))
end

# 144 fundamental pairs plus 49 alternate-format variants.  The latter are
# 34 ordinary special pairs, two stochastic f16x2/bf16x2 forms, five
# stochastic x4 forms, and eight additional scaled variants.
const ORDINARY_CVT_SOURCE_SCHEMAS = let schemas = OrdinaryCvtSourceSchema[]
    for destination in _CVT_FUNDAMENTAL_TYPES,
        source in _CVT_FUNDAMENTAL_TYPES
        _ordinary_cvt_schema!(schemas, destination, source,
                              (_ORDINARY_CVT_SOURCE_CARRIER[source],))
    end

    for (destination, source) in _CVT_SPECIAL_PAIRS
        if destination in _CVT_PACK4_FROM_F32
            _ordinary_cvt_schema!(schemas, destination, source,
                                  (:f32, :b32);
                                  stochastic = true, vector_source = true)
            continue
        end

        operands = source === :f32 && destination in _CVT_PACK2_FROM_F32 ?
                   (:f32, :f32) :
                   (_ORDINARY_CVT_SOURCE_CARRIER[source],)
        _ordinary_cvt_schema!(schemas, destination, source, operands)

        if (destination, source) in _CVT_SCALED_PAIRS
            _ordinary_cvt_schema!(schemas, destination, source,
                                  (operands..., :b16); scaled = true)
        end
        if destination in (:f16x2, :bf16x2) && source === :f32
            _ordinary_cvt_schema!(schemas, destination, source,
                                  (:f32, :f32, :b32); stochastic = true)
        end
    end
    # Deliberately a Vector, not Tuple(...): a several-hundred-element NTuple
    # constant makes every downstream generator/Dict build specialize on the
    # full tuple type. Those inference+codegen bombs tripled package
    # precompile time (measured 30s -> 12s converting the ledgers back).
    schemas
end

length(ORDINARY_CVT_SOURCE_SCHEMAS) == 193 ||
    error("ordinary cvt source-ABI ledger must contain 193 schemas")

_ordinary_cvt_key(schema::OrdinaryCvtSourceSchema) =
    (schema.destination, schema.source, schema.stochastic, schema.scaled)

const _ORDINARY_CVT_SOURCE_SCHEMA_BY_KEY = Dict(
    _ordinary_cvt_key(schema) => schema
    for schema in ORDINARY_CVT_SOURCE_SCHEMAS)

length(_ORDINARY_CVT_SOURCE_SCHEMA_BY_KEY) ==
    length(ORDINARY_CVT_SOURCE_SCHEMAS) ||
    error("duplicate ordinary cvt source-ABI schema")

_cvt_modifier_spelling(mod::Symbol) = replace(String(mod), "__" => "::")
_is_cvt_scaled_modifier(mod::Symbol) =
    _cvt_modifier_spelling(mod) == "scaled::n2::ue8m0"

function ordinary_cvt_source_schema_miss(mods::Tuple{Vararg{Symbol}})
    spelling = isempty(mods) ? "cvt" : "cvt." * join(mods, ".")
    ArgumentError(
        "PTX transpiler: ptx\"$spelling\" does not match the reviewed " *
        "ordinary cvt source-operand ABI in PTX ISA 9.3 §9.7.9.22. " *
        "Refusing to guess immediate carriers; cvt.pack uses its separate " *
        "scalar-result schema.")
end

function ordinary_cvt_source_schema(mods::Tuple{Vararg{Symbol}})
    :pack in mods && return nothing
    length(mods) >= 2 || throw(ordinary_cvt_source_schema_miss(mods))
    destination, source = mods[end - 1], mods[end]
    prefix = mods[1:end - 2]
    stochastic_count = count(==(:rs), prefix)
    scaled_count = count(_is_cvt_scaled_modifier, prefix)
    stochastic_count <= 1 && scaled_count <= 1 ||
        throw(ordinary_cvt_source_schema_miss(mods))
    key = (destination, source, stochastic_count == 1, scaled_count == 1)
    schema = get(_ORDINARY_CVT_SOURCE_SCHEMA_BY_KEY, key, nothing)
    schema === nothing && throw(ordinary_cvt_source_schema_miss(mods))
    schema
end
