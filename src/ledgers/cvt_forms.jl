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
    "ptx/9-instruction-set/9.7.10.24-data-movement-and-conversion-instructions-cvt.md"

# `scaled` is the scale-factor operand kind: :none, :n1 (one ue8m0 in a b8
# register — PTX ISA 9.4 `.scaled::n1::ue8m0`), or :n2 (two packed ue8m0 in
# b16 — `.scaled::n2::ue8m0`).
struct OrdinaryCvtSourceSchema
    destination::Symbol
    source::Symbol
    operands::Tuple{Vararg{Symbol}}
    stochastic::Bool
    scaled::Symbol
    vector_source::Bool
    section::String
end

# PTX ISA 9.4 §9.7.10.24 has twelve fundamental source types.  Alternate
# floating-point formats live in bit-size registers (§5.2.3), so their Julia
# carrier is a bit type rather than a numerically similar fundamental float.
# e2m1x2 is physically b8, but NVPTX has no i8 inline-asm constraint; PTX.jl's
# exact cvt wrappers bridge it through UInt16, hence the reviewed b16 carrier.
# ue5m3x2 (PTX ISA 9.4, sm_107f) packs two ue5m3 into b16.
const ORDINARY_CVT_SOURCE_CARRIERS = (
    :u8 => :u8, :u16 => :u16, :u32 => :u32, :u64 => :u64,
    :s8 => :s8, :s16 => :s16, :s32 => :s32, :s64 => :s64,
    :f16 => :f16, :f32 => :f32, :f64 => :f64, :bf16 => :b16,
    :f16x2 => :b32, :bf16x2 => :b32,
    :e4m3x2 => :b16, :e5m2x2 => :b16,
    :e2m1x2 => :b16, :e2m3x2 => :b16, :e3m2x2 => :b16,
    :ue8m0x2 => :b16, :s2f6x2 => :b16, :ue5m3x2 => :b16,
)

const _ORDINARY_CVT_SOURCE_CARRIER = Dict(ORDINARY_CVT_SOURCE_CARRIERS)
length(_ORDINARY_CVT_SOURCE_CARRIER) == 22 ||
    error("ordinary cvt source-carrier ledger must contain 22 unique formats")

const _CVT_FUNDAMENTAL_TYPES =
    (:u8, :u16, :u32, :u64, :s8, :s16, :s32, :s64,
     :bf16, :f16, :f32, :f64)

const _CVT_PACK2_FROM_F32 =
    (:f16x2, :bf16x2, :e4m3x2, :e5m2x2, :e2m1x2,
     :e2m3x2, :e3m2x2, :ue8m0x2, :s2f6x2, :ue5m3x2)

const _CVT_PACK4_FROM_F32 =
    (:e4m3x4, :e5m2x4, :e2m1x4, :e2m3x4, :e3m2x4)

# `.scaled::n2::ue8m0` — two packed ue8m0 scale factors in one b16 operand.
const _CVT_SCALED_N2_PAIRS = Set{Tuple{Symbol, Symbol}}((
    (:bf16x2, :e4m3x2), (:bf16x2, :e5m2x2),
    (:bf16x2, :e2m1x2), (:bf16x2, :e2m3x2),
    (:bf16x2, :e3m2x2), (:bf16x2, :s2f6x2),
    (:s2f6x2, :f32), (:s2f6x2, :bf16x2),
    (:bf16x2, :ue5m3x2),                    # PTX ISA 9.4
))

# `.scaled::n1::ue8m0` (PTX ISA 9.4, sm_107f) — ONE ue8m0 scale factor in a
# b8 register, dividing the inputs before the down-convert. Physically b8;
# carried as b16 for the same i8-constraint reason as the e2m1x2 bridge.
const _CVT_SCALED_N1_PAIRS = Set{Tuple{Symbol, Symbol}}(
    (dst, src)
    for dst in (:e4m3x2, :e5m2x2, :e2m1x2, :e2m3x2, :e3m2x2, :ue5m3x2),
        src in (:f32, :f16x2, :bf16x2))

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

    # ue5m3x2 (PTX ISA 9.4): down-convert from f32 or either packed 16-bit
    # float, up-convert into both.
    for src in (:f32, :f16x2, :bf16x2)
        push!(pairs, (:ue5m3x2, src))
    end
    for dst in (:f16x2, :bf16x2)
        push!(pairs, (dst, :ue5m3x2))
    end

    for dst in _CVT_PACK4_FROM_F32
        push!(pairs, (dst, :f32))
    end

    length(pairs) == 44 ||
        error("ordinary cvt special-pair ledger must contain 44 forms")
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
                               stochastic = false, scaled = :none,
                               vector_source = false)
    push!(schemas, OrdinaryCvtSourceSchema(
        destination, source, Tuple(operands), stochastic, scaled,
        vector_source, ORDINARY_CVT_SECTION))
end

# 144 fundamental pairs plus 73 alternate-format variants.  The latter are
# 39 ordinary special pairs, two stochastic f16x2/bf16x2 forms, five
# stochastic x4 forms, nine n2-scaled variants, and eighteen n1-scaled
# variants (PTX ISA 9.4).
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

        if (destination, source) in _CVT_SCALED_N2_PAIRS
            _ordinary_cvt_schema!(schemas, destination, source,
                                  (operands..., :b16); scaled = :n2)
        end
        if (destination, source) in _CVT_SCALED_N1_PAIRS
            # The n1 scale factor is physically b8; carried as b16 (see the
            # _CVT_SCALED_N1_PAIRS note).
            _ordinary_cvt_schema!(schemas, destination, source,
                                  (operands..., :b16); scaled = :n1)
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

length(ORDINARY_CVT_SOURCE_SCHEMAS) == 217 ||
    error("ordinary cvt source-ABI ledger must contain 217 schemas")

_ordinary_cvt_key(schema::OrdinaryCvtSourceSchema) =
    (schema.destination, schema.source, schema.stochastic, schema.scaled)

const _ORDINARY_CVT_SOURCE_SCHEMA_BY_KEY = Dict(
    _ordinary_cvt_key(schema) => schema
    for schema in ORDINARY_CVT_SOURCE_SCHEMAS)

length(_ORDINARY_CVT_SOURCE_SCHEMA_BY_KEY) ==
    length(ORDINARY_CVT_SOURCE_SCHEMAS) ||
    error("duplicate ordinary cvt source-ABI schema")

_cvt_modifier_spelling(mod::Symbol) = replace(String(mod), "__" => "::")
function _cvt_scaled_kind(mod::Symbol)
    spelling = _cvt_modifier_spelling(mod)
    spelling == "scaled::n2::ue8m0" && return :n2
    spelling == "scaled::n1::ue8m0" && return :n1
    return nothing
end

# `op` is always :cvt for a claimed miss; the spelling is fixed.
function miss(::CvtLedger, op::Symbol, mods::Tuple{Vararg{Symbol}})
    spelling = isempty(mods) ? "cvt" : "cvt." * join(mods, ".")
    ArgumentError(
        "PTX transpiler: ptx\"$spelling\" does not match the reviewed " *
        "ordinary cvt source-operand ABI in PTX ISA 9.4 §9.7.10.24. " *
        "Refusing to guess immediate carriers; cvt.pack uses its separate " *
        "scalar-result schema.")
end

# Unlike the island schemas, this lookup throws its own miss for an invalid
# spelling: every non-pack cvt spelling is inside this ledger's domain, so a
# key miss is always a hard source-ABI error, never a fall-through (see the
# CvtLedger note in protocol.jl).
function schema(::CvtLedger, op::Symbol, mods::Tuple{Vararg{Symbol}})
    op === :cvt || return nothing
    :pack in mods && return nothing
    length(mods) >= 2 || throw(miss(CvtLedger(), op, mods))
    destination, source = mods[end - 1], mods[end]
    prefix = mods[1:end - 2]
    stochastic_count = count(==(:rs), prefix)
    scaled_kinds = [k for k in map(_cvt_scaled_kind, prefix) if k !== nothing]
    stochastic_count <= 1 && length(scaled_kinds) <= 1 ||
        throw(miss(CvtLedger(), op, mods))
    scaled = isempty(scaled_kinds) ? :none : only(scaled_kinds)
    key = (destination, source, stochastic_count == 1, scaled)
    found = get(_ORDINARY_CVT_SOURCE_SCHEMA_BY_KEY, key, nothing)
    found === nothing && throw(miss(CvtLedger(), op, mods))
    found
end
