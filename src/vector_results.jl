# Audited vector-result grammar islands.  The terminal modifier names one
# scalar lane, not the Julia result: every form below returns a homogeneous
# tuple and therefore needs one LLVM inline-asm output constraint per lane.
# Keep the core inventory closed and small; optional memory qualifiers are
# parsed independently so their cross product is not materialized here.

struct VectorTargetRequirement
    min_sm::Union{Nothing,VersionNumber}
    architectures::Tuple{Vararg{VersionNumber}}
    families::Tuple{Vararg{VersionNumber}}
    family_since::Union{Nothing,VersionNumber}
end

VectorTargetRequirement(min_sm::Union{Nothing,VersionNumber}) =
    VectorTargetRequirement(min_sm, (), (), nothing)

struct VectorResultCoreForm
    op::Symbol
    coremods::Tuple{Vararg{Symbol}}
    lanes::Int
    lane_kind::Symbol
    lane_type::Type
    ptx_version::VersionNumber
    target::VectorTargetRequirement
    provenance::Symbol
    section::String
end

# Some atom examples place `.vec.type` before `.op[.noftz]`, contrary to the
# syntax block.  CUDA 13.3 ptxas accepts the eight exact documented spellings
# below. No other alternate permutation is generalized from them; the ninth
# vector example (`.v4.b16x2.min.noftz`) is invalid and remains a pinned miss.
struct VectorResultCompatForm
    spellings::Tuple{Vararg{Tuple{Vararg{Symbol}}}}
    section::String
end

struct VectorResultSchema
    form::VectorResultCoreForm
    mods::Tuple{Vararg{Symbol}}
    operands::Tuple{Vararg{Symbol}}
    cache_hint::Bool
    provenance::Symbol
end

const _VECTOR_LD_SECTION =
    "ptx/9-instruction-set/9.7.9.8-data-movement-and-conversion-instructions-ld.md"
const _VECTOR_ATOM_SECTION =
    "ptx/9-instruction-set/9.7.14.5-parallel-synchronization-and-communication-instructions-atom.md"
const _VECTOR_MULTIMEM_SECTION =
    "ptx/9-instruction-set/9.7.9.15-data-movement-and-conversion-instructions-multimem.ld_reduce-multimem.st-multimem.red.md"

const _VECTOR_LANE_TYPES = Dict{Symbol,Type}(
    :b8 => UInt8, :u8 => UInt8, :s8 => Int8,
    :b16 => UInt16, :u16 => UInt16, :s16 => Int16,
    :b32 => UInt32, :u32 => UInt32, :s32 => Int32,
    :b64 => UInt64, :u64 => UInt64, :s64 => Int64,
    :f16 => Float16, :bf16 => UInt16, :f32 => Float32, :f64 => Float64,
    :f16x2 => UInt32, :bf16x2 => UInt32,
    # Scalar FP8 tokens intentionally remain local to this reviewed vector
    # island.  They must not become a generic terminal-result fallback.
    :e4m3 => UInt8, :e5m2 => UInt8,
    :e4m3x2 => UInt16, :e5m2x2 => UInt16,
    :e4m3x4 => UInt32, :e5m2x4 => UInt32,
)

function _vector_result_core!(forms, op, coremods, lanes, lane_kind,
                              ptx_version, target, section)
    haskey(_VECTOR_LANE_TYPES, lane_kind) ||
        error("unknown vector-result lane kind: ", lane_kind)
    any(f -> f.op === op && f.coremods == coremods, forms) &&
        error("duplicate vector-result core form: ", op, ".", join(coremods, "."))
    push!(forms, VectorResultCoreForm(op, coremods, lanes, lane_kind,
                                     _VECTOR_LANE_TYPES[lane_kind],
                                     ptx_version, target, :isa, section))
end

const _TARGET_ALL = VectorTargetRequirement(nothing)
const _TARGET_SM13 = VectorTargetRequirement(v"1.3")
const _TARGET_SM90 = VectorTargetRequirement(v"9.0")
const _TARGET_SM100 = VectorTargetRequirement(v"10.0")
const _TARGET_MULTIMEM_FP8 = VectorTargetRequirement(
    nothing, (v"10.0", v"11.0", v"12.0", v"12.1"),
    (v"10.0", v"11.0"), v"8.8")

const VECTOR_RESULT_CORE_FORMS = let forms = VectorResultCoreForm[]
    # ld: 14 v2 + 10 ordinary v4 + 4 wide v4 + 4 v8 = 32.
    for lane_kind in (:b8, :u8, :s8, :b16, :u16, :s16,
                      :b32, :u32, :s32, :b64, :u64, :s64, :f32, :f64)
        _vector_result_core!(forms, :ld, (:v2, lane_kind), 2, lane_kind,
                             v"1.0", lane_kind === :f64 ? _TARGET_SM13 : _TARGET_ALL,
                             _VECTOR_LD_SECTION)
    end
    for lane_kind in (:b8, :u8, :s8, :b16, :u16, :s16,
                      :b32, :u32, :s32, :f32)
        _vector_result_core!(forms, :ld, (:v4, lane_kind), 4, lane_kind,
                             v"1.0", _TARGET_ALL, _VECTOR_LD_SECTION)
    end
    for lane_kind in (:b64, :u64, :s64, :f64)
        _vector_result_core!(forms, :ld, (:v4, lane_kind), 4, lane_kind,
                             v"8.8", _TARGET_SM100, _VECTOR_LD_SECTION)
    end
    for lane_kind in (:b32, :u32, :s32, :f32)
        _vector_result_core!(forms, :ld, (:v8, lane_kind), 8, lane_kind,
                             v"8.8", _TARGET_SM100, _VECTOR_LD_SECTION)
    end

    # atom: 2 f32 + 18 half-word + 12 packed = 32.
    for lanes in (2, 4)
        vec = Symbol("v", lanes)
        _vector_result_core!(forms, :atom, (:add, vec, :f32), lanes, :f32,
                             v"8.1", _TARGET_SM90, _VECTOR_ATOM_SECTION)
    end
    for lanes in (2, 4, 8), lane_kind in (:f16, :bf16), atomop in (:add, :min, :max)
        vec = Symbol("v", lanes)
        _vector_result_core!(forms, :atom,
                             (atomop, :noftz, vec, lane_kind), lanes, lane_kind,
                             v"8.1", _TARGET_SM90, _VECTOR_ATOM_SECTION)
    end
    for lanes in (2, 4), lane_kind in (:f16x2, :bf16x2), atomop in (:add, :min, :max)
        vec = Symbol("v", lanes)
        _vector_result_core!(forms, :atom,
                             (atomop, :noftz, vec, lane_kind), lanes, lane_kind,
                             v"8.1", _TARGET_SM90, _VECTOR_ATOM_SECTION)
    end

    # multimem.ld_reduce: 74 base + 30 acc::f32 + 42 acc::f16 = 146.
    multimem_types = (
        2 => (:f16, :f16x2, :bf16, :bf16x2, :f32,
              :e4m3x2, :e4m3x4, :e5m2x2, :e5m2x4),
        4 => (:f16, :f16x2, :bf16, :bf16x2, :f32,
              :e4m3, :e4m3x2, :e4m3x4, :e5m2, :e5m2x2, :e5m2x4),
        8 => (:f16, :bf16, :e4m3, :e4m3x2, :e5m2, :e5m2x2),
    )
    fp8 = Set((:e4m3, :e4m3x2, :e4m3x4, :e5m2, :e5m2x2, :e5m2x4))
    narrow = Set((:f16, :f16x2, :bf16, :bf16x2))
    for (lanes, lane_kinds) in multimem_types, lane_kind in lane_kinds
        vec = Symbol("v", lanes)
        target = lane_kind in fp8 ? _TARGET_MULTIMEM_FP8 : _TARGET_SM90
        version = lane_kind in fp8 ? v"8.6" : v"8.1"
        ops = lane_kind === :f32 ? (:add,) : (:add, :min, :max)
        for reduceop in ops
            _vector_result_core!(forms, :multimem,
                                 (:ld_reduce, reduceop, vec, lane_kind),
                                 lanes, lane_kind, version, target,
                                 _VECTOR_MULTIMEM_SECTION)
        end
        if lane_kind in narrow
            for reduceop in (:add, :min, :max)
                _vector_result_core!(forms, :multimem,
                                     (:ld_reduce, reduceop, Symbol("acc::f32"),
                                      vec, lane_kind),
                                     lanes, lane_kind, v"8.2", _TARGET_SM90,
                                     _VECTOR_MULTIMEM_SECTION)
            end
        elseif lane_kind in fp8
            for reduceop in (:add, :min, :max)
                _vector_result_core!(forms, :multimem,
                                     (:ld_reduce, reduceop, Symbol("acc::f16"),
                                      vec, lane_kind),
                                     lanes, lane_kind, v"8.6", _TARGET_MULTIMEM_FP8,
                                     _VECTOR_MULTIMEM_SECTION)
            end
        end
    end
    length(forms) == 210 || error("vector-result core ledger drift: ", length(forms))
    # Deliberately a Vector, not Tuple(...): a several-hundred-element NTuple
    # constant makes every downstream generator/Dict build specialize on the
    # full tuple type. Those inference+codegen bombs tripled package
    # precompile time (measured 30s -> 12s converting the ledgers back).
    forms
end

const _VECTOR_RESULT_CORE_BY_FORM = Dict(
    (form.op, form.coremods) => form for form in VECTOR_RESULT_CORE_FORMS)

const VECTOR_RESULT_COMPAT_FORMS = (
    VectorResultCompatForm(((:global, :v8, :f16, :max, :noftz),), _VECTOR_ATOM_SECTION),
    VectorResultCompatForm(((:global, :v8, :bf16, :add, :noftz),), _VECTOR_ATOM_SECTION),
    VectorResultCompatForm(((:global, :v2, :f16, :add, :noftz),), _VECTOR_ATOM_SECTION),
    VectorResultCompatForm(((:global, :v2, :bf16, :add, :noftz),), _VECTOR_ATOM_SECTION),
    VectorResultCompatForm(((:global, :v2, :f16x2, :min, :noftz),), _VECTOR_ATOM_SECTION),
    VectorResultCompatForm(((:global, :v2, :bf16x2, :max, :noftz),), _VECTOR_ATOM_SECTION),
    VectorResultCompatForm(((:global, :v4, :f32, :add),), _VECTOR_ATOM_SECTION),
    VectorResultCompatForm(((:global, :v2, :f32, :add),), _VECTOR_ATOM_SECTION),
)

const _VECTOR_ATOM_COMPAT_SPELLINGS = let spellings = Dict{Tuple,Tuple}()
    for compat in VECTOR_RESULT_COMPAT_FORMS, mods in compat.spellings
        vec, lane_kind, atomop = mods[2], mods[3], mods[4]
        core = lane_kind === :f32 ? (atomop, vec, lane_kind) :
               (atomop, :noftz, vec, lane_kind)
        spellings[mods] = core
    end
    length(spellings) == 8 || error("atom compatibility spelling drift")
    spellings
end

const _VECTOR_MARKERS = Set((:v2, :v4, :v8))
const _VECTOR_STATE_SPACES = Set((
    :const, :global, :local, :param, Symbol("param::entry"), Symbol("param::func"),
    :shared, Symbol("shared::cta"), Symbol("shared::cluster"),
))
const _VECTOR_SCOPES = Set((:cta, :cluster, :gpu, :sys))
const _LD_CACHE_OPS = Set((:ca, :cg, :cs, :lu, :cv))
const _LD_L1_EVICTION = Set((Symbol("L1::evict_normal"), Symbol("L1::evict_unchanged"),
                             Symbol("L1::evict_first"), Symbol("L1::evict_last"),
                             Symbol("L1::no_allocate")))
const _LD_L2_EVICTION = Set((Symbol("L2::evict_normal"), Symbol("L2::evict_first"),
                             Symbol("L2::evict_last")))
const _LD_PREFETCH = Set((Symbol("L2::64B"), Symbol("L2::128B"), Symbol("L2::256B")))
const _CACHE_HINT = Symbol("L2::cache_hint")

_take_optional(tokens, i, allowed) =
    i <= length(tokens) && tokens[i] in allowed ? (i + 1, tokens[i]) : (i, nothing)

function _ld_prefix_info(prefix::Tuple, form::VectorResultCoreForm)
    i = 1
    semantic = :weak
    explicit_weak = false
    if i <= length(prefix) && prefix[i] in (:volatile, :relaxed, :acquire)
        semantic = prefix[i]
        i += 1
    elseif i <= length(prefix) && prefix[i] === :weak
        explicit_weak = true
        i += 1
    end
    scope = nothing
    if semantic in (:relaxed, :acquire)
        i, scope = _take_optional(prefix, i, _VECTOR_SCOPES)
        scope === nothing && return nothing
    end
    i, state = _take_optional(prefix, i, _VECTOR_STATE_SPACES)

    cache_op = l1 = l2 = cache_hint = prefetch = nothing
    if semantic === :volatile
        i, prefetch = _take_optional(prefix, i, _LD_PREFETCH)
    elseif semantic in (:relaxed, :acquire)
        i, l1 = _take_optional(prefix, i, _LD_L1_EVICTION)
        i, l2 = _take_optional(prefix, i, _LD_L2_EVICTION)
        if i <= length(prefix) && prefix[i] === _CACHE_HINT
            cache_hint = prefix[i]; i += 1
        end
        i, prefetch = _take_optional(prefix, i, _LD_PREFETCH)
    else
        # The two weak syntax alternatives differ only in cache-op versus
        # eviction qualifiers; parsing either in canonical order is closed.
        i, cache_op = _take_optional(prefix, i, _LD_CACHE_OPS)
        if cache_op === nothing
            i, l1 = _take_optional(prefix, i, _LD_L1_EVICTION)
            i, l2 = _take_optional(prefix, i, _LD_L2_EVICTION)
        end
        if i <= length(prefix) && prefix[i] === _CACHE_HINT
            cache_hint = prefix[i]; i += 1
        end
        i, prefetch = _take_optional(prefix, i, _LD_PREFETCH)
    end
    i == length(prefix) + 1 || return nothing

    # State-space restrictions stated by §9.7.9.8.
    generic = state === nothing
    semantic in (:relaxed, :acquire) &&
        !(generic || state in (:global, :shared, Symbol("shared::cta"),
                               Symbol("shared::cluster"))) && return nothing
    semantic === :volatile &&
        !(generic || state in (:global, :local, :shared, Symbol("shared::cta"),
                               Symbol("shared::cluster"))) && return nothing
    (cache_hint !== nothing || prefetch !== nothing) &&
        !(generic || state === :global) && return nothing
    wide = form.coremods[1] === :v8 ||
           (form.coremods[1] === :v4 && form.lane_kind in (:b64, :u64, :s64, :f64))
    wide && !(generic || state === :global) && return nothing
    l2 !== nothing && !wide && return nothing
    (; cache_hint = cache_hint !== nothing, wide, state, semantic,
       explicit_weak, scope, cache_op, l1, l2, prefetch)
end

function _atom_prefix_info(prefix::Tuple)
    i = 1
    sem = nothing
    if i <= length(prefix) && prefix[i] in (:relaxed, :acquire, :release, :acq_rel)
        sem = prefix[i]; i += 1
    end
    scope = nothing
    if i <= length(prefix) && prefix[i] in _VECTOR_SCOPES
        scope = prefix[i]; i += 1
    end
    state = nothing
    if i <= length(prefix) && prefix[i] === :global
        state = :global; i += 1
    end
    i == length(prefix) + 1 || return nothing
    (; sem, scope, state)
end

function _multimem_prefix_info(prefix::Tuple)
    # Prefix includes the mandatory ld_reduce selector.
    !isempty(prefix) && first(prefix) === :ld_reduce || return nothing
    i = 2
    sem = nothing
    scope = nothing
    if i <= length(prefix) && prefix[i] === :weak
        sem = :weak; i += 1
    elseif i <= length(prefix) && prefix[i] in (:relaxed, :acquire)
        sem = prefix[i]; i += 1
        i, scope = _take_optional(prefix, i, _VECTOR_SCOPES)
        scope === nothing && return nothing
    end
    state = nothing
    if i <= length(prefix) && prefix[i] === :global
        state = :global; i += 1
    end
    i == length(prefix) + 1 || return nothing
    (; sem, scope, state)
end

function _schema_from_core(form::VectorResultCoreForm,
                           mods::Tuple{Vararg{Symbol}}, prefix_info;
                           provenance::Symbol = :isa)
    cache_policy = hasproperty(prefix_info, :cache_hint) && prefix_info.cache_hint
    operands = form.op === :atom ? (:address, :vector) :
        form.op === :ld ? (:address,) :
        (:address,)
    VectorResultSchema(form, mods, operands, cache_policy, provenance)
end

function vector_result_schema(op::Symbol, mods::Tuple{Vararg{Symbol}})
    if op === :atom
        compat_core = get(_VECTOR_ATOM_COMPAT_SPELLINGS, mods, nothing)
        if compat_core !== nothing
            form = get(_VECTOR_RESULT_CORE_BY_FORM, (:atom, compat_core), nothing)
            form === nothing && error("compatibility form lacks canonical atom core")
            return VectorResultSchema(form, mods, (:address, :vector), false,
                                      :ptxas_compat)
        end
    end

    vecindex = findlast(mod -> mod in _VECTOR_MARKERS, mods)
    vecindex === nothing && return nothing
    vecindex < length(mods) || return nothing

    if op === :ld
        vecindex == length(mods) - 1 || return nothing
        core = (mods[vecindex], mods[end])
        form = get(_VECTOR_RESULT_CORE_BY_FORM, (:ld, core), nothing)
        form === nothing && return nothing
        info = _ld_prefix_info(mods[1:vecindex-1], form)
        info === nothing && return nothing
        return _schema_from_core(form, mods, info)
    elseif op === :atom
        # Canonical grammar is op.noftz?.cache_hint?.vec.type.  Split off the
        # fixed tail, then let the prefix parser consume sem/scope/global and
        # optional cache hint immediately before the vector token.
        j = vecindex - 1
        cache_hint = j >= 1 && mods[j] === _CACHE_HINT
        cache_hint && (j -= 1)
        noftz = j >= 1 && mods[j] === :noftz
        noftz && (j -= 1)
        j >= 1 || return nothing
        atomop = mods[j]
        core = noftz ? (atomop, :noftz, mods[vecindex], mods[end]) :
                        (atomop, mods[vecindex], mods[end])
        form = get(_VECTOR_RESULT_CORE_BY_FORM, (:atom, core), nothing)
        form === nothing && return nothing
        prefix = mods[1:j-1]
        info = _atom_prefix_info(prefix)
        info === nothing && return nothing
        info = merge(info, (; cache_hint))
        return _schema_from_core(form, mods, info)
    elseif op === :multimem
        # ld_reduce + sem/scope/state precede op/acc/vec/type.
        acc = vecindex >= 2 && mods[vecindex - 1] in
              (Symbol("acc::f32"), Symbol("acc::f16"))
        opindex = acc ? vecindex - 2 : vecindex - 1
        opindex >= 2 || return nothing
        core = acc ? (:ld_reduce, mods[opindex], mods[vecindex - 1],
                      mods[vecindex], mods[end]) :
                     (:ld_reduce, mods[opindex], mods[vecindex], mods[end])
        form = get(_VECTOR_RESULT_CORE_BY_FORM, (:multimem, core), nothing)
        form === nothing && return nothing
        info = _multimem_prefix_info(mods[1:opindex-1])
        info === nothing && return nothing
        return _schema_from_core(form, mods, info)
    end
    nothing
end

function requires_vector_result_schema(op::Symbol,
                                       mods::Tuple{Vararg{Symbol}})
    any(mod -> mod in _VECTOR_MARKERS, mods) || return false
    op in (:ld, :atom) && return true
    op === :multimem && :ld_reduce in mods && return true
    false
end

function vector_result_schema_miss(op::Symbol,
                                   mods::Tuple{Vararg{Symbol}})
    spelling = isempty(mods) ? string(op) : string(op, ".", join(mods, "."))
    ArgumentError(
        "ptx\"$spelling\" is inside an audited vector-result grammar " *
        "island but does not match a reviewed vector shape, type, operation, " *
        "or canonical modifier order. Refusing scalar terminal-modifier " *
        "inference. The raw tier also rejects this miss because it cannot " *
        "supply an explicit vector result ABI.")
end

_vector_integer_immediate(::Type{Val{V}}) where {V} = V isa Integer && !(V isa Bool)
_vector_integer_immediate(::Type) = false

_vector_address_type(::Type{<:Core.LLVMPtr}) = true
# `Address` is defined in address_operands.jl, included after this ledger;
# resolve it in the body (call time), not the signature (definition time).
_vector_address_type(::Type{T}) where {T} =
    T in (UInt32, Int32, UInt64, Int64) || _vector_integer_immediate(T) ||
    (T <: Address && is_ptx_address_integer_type(T.parameters[1]))

_vector_cache_policy_type(::Type{T}) where {T} =
    T in (UInt64, Int64) || _vector_integer_immediate(T)

function validate_vector_result_args(schema::VectorResultSchema, argtypes)
    spelling = string(schema.form.op, ".", join(schema.mods, "."))
    roles = vector_result_operand_roles(schema, length(argtypes))
    roles === nothing && throw(ArgumentError(
        "ptx\"$spelling\" is an audited vector form with " *
        string(length(schema.operands) + (schema.cache_hint ? 1 : 0)) *
        " source operands" *
        (schema.cache_hint ? " (including the required 64-bit cache-policy " *
         "operand for .L2::cache_hint)" : "") *
        ", got $(length(argtypes)); " *
        "see $(schema.form.section)"))
    for (i, (kind, T)) in enumerate(zip(roles, argtypes))
        valid = kind === :address ? _vector_address_type(T) :
                kind === :cache_policy ? _vector_cache_policy_type(T) :
                kind === :vector ? T === NTuple{schema.form.lanes,
                                                schema.form.lane_type} : false
        valid && continue
        expected = kind === :address ? "a 32/64-bit address or LLVM pointer" :
                   kind === :cache_policy ? "a 64-bit cache-policy carrier" :
                   "NTuple{$(schema.form.lanes),$(schema.form.lane_type)}"
        throw(ArgumentError(
            "ptx\"$spelling\" operand $i must use $expected, got $T; " *
            "see $(schema.form.section)"))
    end
    nothing
end

function vector_result_operand_roles(schema::VectorResultSchema, nargs::Int)
    # The ISA syntax marks both `.L2::cache_hint` and `cache_policy` optional,
    # and its prose states only policy => qualifier. In practice CUDA 12.9 and
    # 13.3 ptxas reject qualifier-without-policy for both scalar and vector
    # `ld`/`atom`. Keep this production lowering on the assemblable subset:
    # the qualifier and its 64-bit operand are a pair.
    expected = length(schema.operands) + (schema.cache_hint ? 1 : 0)
    nargs == expected || return nothing
    schema.cache_hint && return (schema.operands..., :cache_policy)
    schema.operands
end

vector_result_type(schema::VectorResultSchema) =
    NTuple{schema.form.lanes, schema.form.lane_type}

function validate_vector_result_mask(schema::VectorResultSchema, mask)
    mask isa Tuple && length(mask) == schema.form.lanes && all(x -> x isa Bool, mask) ||
        throw(ArgumentError("vector-result sink mask must be an NTuple{" *
                            string(schema.form.lanes) * ",Bool}, got " * repr(mask)))
    wide = schema.form.op === :ld &&
           (schema.form.coremods[1] === :v8 ||
            (schema.form.coremods[1] === :v4 &&
             schema.form.lane_kind in (:b64, :u64, :s64, :f64)))
    (wide || all(mask)) || throw(ArgumentError(
        "PTX sink destinations are supported only for ld.v8.{b32,u32,s32,f32} " *
        "and ld.v4.{b64,u64,s64,f64}; see $(schema.form.section)"))
    wide && !any(mask) && throw(ArgumentError(
        "an all-false vector-result sink mask is not lowered: PTX 9.3's " *
        "per-lane sink wording does not resolve whether an all-sink ld still " *
        "has the memory/synchronization effects assigned to ld by section " *
        "8.4, and CUDA 12.9/13.3 ptxas reject an all-`_` destination because " *
        "they cannot infer its element-register type. Keep at least one live " *
        "destination lane; see $(schema.form.section)"))
    Tuple(mask)
end
