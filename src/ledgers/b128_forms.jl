# Audited scalar `.b128` register grammar and Julia carrier policy.
#
# PTX has a real 128-bit bit register, but LLVM's NVPTX inline-assembly
# constraints expose only predicate, 16-, 32-, and 64-bit register classes.
# Use the ISA's own `mov.b128` pack/unpack operation to bridge every `.b128`
# register through one stable Julia carrier: two low-to-high UInt64 words.
# See PTX ISA 9.3:
#   ptx/9-instruction-set/9.7.9.4-data-movement-and-conversion-instructions-mov.md
#   ptx/9-instruction-set/9.7.9.8-data-movement-and-conversion-instructions-ld.md
#   ptx/9-instruction-set/9.7.9.9-data-movement-and-conversion-instructions-ld.global.nc.md
#   ptx/9-instruction-set/9.7.9.10-data-movement-and-conversion-instructions-ldu.md
#   ptx/9-instruction-set/9.7.9.11-data-movement-and-conversion-instructions-st.md
#   ptx/9-instruction-set/9.7.14.5-parallel-synchronization-and-communication-instructions-atom.md
#   ptx/9-instruction-set/9.7.14.19-parallel-synchronization-and-communication-instructions-clusterlaunchcontrol.query_cancel.md

"""
    B128

PTX.jl's Julia carrier for one PTX `.b128` register: `NTuple{2,UInt64}`,
low word first. LLVM's NVPTX inline-assembly constraints expose no 128-bit
register class, so every `.b128` value crosses the asm boundary as two
64-bit words bridged by the ISA's own `mov.b128` pack/unpack. Construct
with [`b128`](@ref).
"""
const B128 = NTuple{2,UInt64}

"""Construct PTX.jl's low-word-first carrier for one PTX `.b128` register."""
@inline b128(lo::UInt64, hi::UInt64)::B128 = (lo, hi)
@inline b128(x::B128)::B128 = x
@inline b128(x0::UInt32, x1::UInt32, x2::UInt32, x3::UInt32)::B128 =
    (UInt64(x0) | (UInt64(x1) << 32),
     UInt64(x2) | (UInt64(x3) << 32))

struct B128FormSchema
    op::Symbol
    mods::Tuple{Vararg{Symbol}}
    kind::Symbol
    result::Type
    operands::Tuple{Vararg{Symbol}}
    cache_policy::Bool
    section::String
end

const _B128_MOV_SECTION =
    "ptx/9-instruction-set/9.7.9.4-data-movement-and-conversion-instructions-mov.md"
const _B128_LD_SECTION =
    "ptx/9-instruction-set/9.7.9.8-data-movement-and-conversion-instructions-ld.md"
const _B128_LD_NC_SECTION =
    "ptx/9-instruction-set/9.7.9.9-data-movement-and-conversion-instructions-ld.global.nc.md"
const _B128_LDU_SECTION =
    "ptx/9-instruction-set/9.7.9.10-data-movement-and-conversion-instructions-ldu.md"
const _B128_ST_SECTION =
    "ptx/9-instruction-set/9.7.9.11-data-movement-and-conversion-instructions-st.md"
const _B128_ATOM_SECTION =
    "ptx/9-instruction-set/9.7.14.5-parallel-synchronization-and-communication-instructions-atom.md"
const _B128_QUERY_SECTION =
    "ptx/9-instruction-set/9.7.14.19-parallel-synchronization-and-communication-instructions-clusterlaunchcontrol.query_cancel.md"

const _B128_STATES_LD = Set((
    :const, :global, :local, :param, Symbol("param::entry"),
    Symbol("param::func"), :shared, Symbol("shared::cta"),
    Symbol("shared::cluster"),
))
const _B128_STATES_ST = Set((
    :global, :local, :param, Symbol("param::func"), :shared,
    Symbol("shared::cta"), Symbol("shared::cluster"),
))
const _B128_ATOM_STATES = Set((
    :global, :shared, Symbol("shared::cta"), Symbol("shared::cluster"),
))
const _B128_ST_CACHE_OPS = Set((:wb, :cg, :cs, :wt))

function _b128_normal_ld_info(prefix::Tuple)
    # Reuse the closed scalar-independent parser from the vector-result
    # ledger. A fake scalar core keeps its vector-only gates disabled.
    form = VectorResultCoreForm(:ld, (:scalar, :b128), 1, :b128, B128,
                                v"8.3", _TARGET_SM100, :isa,
                                _B128_LD_SECTION)
    _ld_prefix_info(prefix, form)
end

function _b128_ld_nc_info(prefix::Tuple)
    # global{.cop}.nc... and global.nc{.eviction...}...
    length(prefix) >= 2 && first(prefix) === :global || return nothing
    i = 2
    cache_op = i <= length(prefix) && prefix[i] in (:ca, :cg, :cs) ?
               prefix[i] : nothing
    cache_op === nothing || (i += 1)
    i <= length(prefix) && prefix[i] === :nc || return nothing
    i += 1
    l1 = l2 = cache_hint = prefetch = nothing
    if cache_op === nothing
        i, l1 = _take_optional(prefix, i, _LD_L1_EVICTION)
        i, l2 = _take_optional(prefix, i, _LD_L2_EVICTION)
        # Scalar b128 is not one of the two wide vector shapes for which the
        # ISA permits L2 eviction priority.
        l2 === nothing || return nothing
    end
    if i <= length(prefix) && prefix[i] === _CACHE_HINT
        cache_hint = prefix[i]; i += 1
    end
    i, prefetch = _take_optional(prefix, i, _LD_PREFETCH)
    i == length(prefix) + 1 || return nothing
    (; cache_hint = cache_hint !== nothing, cache_op, l1, prefetch)
end

function _b128_st_info(prefix::Tuple)
    i = 1
    semantic = :weak
    if i <= length(prefix) && prefix[i] in (:volatile, :relaxed, :release)
        semantic = prefix[i]; i += 1
    elseif i <= length(prefix) && prefix[i] === :weak
        i += 1
    end
    scope = nothing
    if semantic in (:relaxed, :release)
        i, scope = _take_optional(prefix, i, _VECTOR_SCOPES)
        scope === nothing && return nothing
    end
    i, state = _take_optional(prefix, i, _B128_STATES_ST)
    cache_op = l1 = l2 = cache_hint = nothing
    if semantic === :weak
        i, cache_op = _take_optional(prefix, i, _B128_ST_CACHE_OPS)
        if cache_op === nothing
            i, l1 = _take_optional(prefix, i, _LD_L1_EVICTION)
            i, l2 = _take_optional(prefix, i, _LD_L2_EVICTION)
            l2 === nothing || return nothing
        end
        if i <= length(prefix) && prefix[i] === _CACHE_HINT
            cache_hint = prefix[i]; i += 1
        end
    elseif semantic in (:relaxed, :release)
        i, l1 = _take_optional(prefix, i, _LD_L1_EVICTION)
        i, l2 = _take_optional(prefix, i, _LD_L2_EVICTION)
        l2 === nothing || return nothing
        if i <= length(prefix) && prefix[i] === _CACHE_HINT
            cache_hint = prefix[i]; i += 1
        end
    end
    i == length(prefix) + 1 || return nothing
    generic = state === nothing
    semantic in (:relaxed, :release) &&
        !(generic || state in (:global, :shared, Symbol("shared::cta"),
                               Symbol("shared::cluster"))) && return nothing
    semantic === :volatile &&
        !(generic || state in (:global, :local, :shared,
                               Symbol("shared::cta"),
                               Symbol("shared::cluster"))) && return nothing
    cache_hint !== nothing && !(generic || state === :global) && return nothing
    (; cache_hint = cache_hint !== nothing, semantic, scope, state,
       cache_op, l1)
end

function _b128_mmio_info(prefix::Tuple, load::Bool)
    length(prefix) in (3, 4) || return nothing
    prefix[1] === :mmio || return nothing
    prefix[2] in (load ? (:relaxed, :acquire) : (:relaxed, :release)) ||
        return nothing
    prefix[3] === :sys || return nothing
    length(prefix) == 3 || prefix[4] === :global || return nothing
    (; cache_hint = false)
end

function _b128_atom_info(prefix::Tuple)
    i = 1
    sem = nothing
    if i <= length(prefix) && prefix[i] in (:relaxed, :acquire, :release, :acq_rel)
        sem = prefix[i]; i += 1
    end
    scope = nothing
    if i <= length(prefix) && prefix[i] in _VECTOR_SCOPES
        scope = prefix[i]; i += 1
    end
    i, state = _take_optional(prefix, i, _B128_ATOM_STATES)
    i == length(prefix) + 1 || return nothing
    (; sem, scope, state)
end

function _parse_b128_form_schema(op::Symbol, mods::Tuple{Vararg{Symbol}})
    op === :mov && mods === (:b128,) &&
        return B128FormSchema(op, mods, :mov, B128, (:b128,), false,
                              _B128_MOV_SECTION)

    if op in (:ld, :ldu, :st, :atom)
        isempty(mods) || last(mods) === :b128 || return nothing
    end

    if op === :ld && !isempty(mods) && last(mods) === :b128
        prefix = mods[1:end-1]
        info = (:nc in prefix) ? _b128_ld_nc_info(prefix) :
               (!isempty(prefix) && first(prefix) === :mmio) ?
                   _b128_mmio_info(prefix, true) : _b128_normal_ld_info(prefix)
        info === nothing && return nothing
        section = :nc in prefix ? _B128_LD_NC_SECTION : _B128_LD_SECTION
        operands = getproperty(info, :cache_hint) ? (:address, :cache_policy) :
                                                   (:address,)
        return B128FormSchema(op, mods, :load, B128, operands,
                              getproperty(info, :cache_hint), section)
    elseif op === :ldu && mods in ((:b128,), (:global, :b128))
        return B128FormSchema(op, mods, :load, B128, (:address,), false,
                              _B128_LDU_SECTION)
    elseif op === :st && !isempty(mods) && last(mods) === :b128
        prefix = mods[1:end-1]
        info = !isempty(prefix) && first(prefix) === :mmio ?
               _b128_mmio_info(prefix, false) : _b128_st_info(prefix)
        info === nothing && return nothing
        operands = getproperty(info, :cache_hint) ?
                   (:address, :b128, :cache_policy) : (:address, :b128)
        return B128FormSchema(op, mods, :store, Nothing, operands,
                              getproperty(info, :cache_hint), _B128_ST_SECTION)
    elseif op === :atom && !isempty(mods) && last(mods) === :b128
        length(mods) >= 2 || return nothing
        opindex = length(mods) - 1
        cache_hint = mods[opindex] === _CACHE_HINT
        cache_hint && (opindex -= 1)
        atomop = mods[opindex]
        atomop in (:exch, :cas) || return nothing
        cache_hint && atomop !== :exch && return nothing
        info = _b128_atom_info(mods[1:opindex-1])
        info === nothing && return nothing
        cache_hint && !(info.state === nothing || info.state === :global) &&
            return nothing
        operands = atomop === :cas ? (:address, :b128, :b128) :
                   cache_hint ? (:address, :b128, :cache_policy) :
                                (:address, :b128)
        return B128FormSchema(op, mods, atomop, B128, operands, cache_hint,
                              _B128_ATOM_SECTION)
    elseif op === :clusterlaunchcontrol
        if mods === (:query_cancel, :is_canceled, :pred, :b128)
            return B128FormSchema(op, mods, :query_pred, Bool, (:b128,), false,
                                  _B128_QUERY_SECTION)
        elseif mods === (:query_cancel, :get_first_ctaid, :v4, :b32, :b128)
            return B128FormSchema(op, mods, :query_v4, NTuple{4,UInt32},
                                  (:b128,), false, _B128_QUERY_SECTION)
        elseif length(mods) == 4 && mods[1] === :query_cancel &&
               mods[2] in (Symbol("get_first_ctaid::x"),
                           Symbol("get_first_ctaid::y"),
                           Symbol("get_first_ctaid::z")) &&
               mods[3:4] === (:b32, :b128)
            return B128FormSchema(op, mods, :query_dim, UInt32, (:b128,), false,
                                  _B128_QUERY_SECTION)
        end
    end
    nothing
end

# Materialize the complete finite grammar island. The parser above validates
# modifier ordering and prose restrictions; this candidate expansion supplies
# every syntax branch and deliberately includes nearby invalid choices so the
# validator, rather than a hand-picked examples list, decides membership.
function _b128_candidate_keys()
    keys = Set{Tuple{Symbol,Tuple}}()
    add(op, mods) = push!(keys, (op, Tuple(mods)))
    add(:mov, (:b128,))

    states_ld = (nothing, _B128_STATES_LD...)
    states_st = (nothing, _B128_STATES_ST...)
    state_tuple(state) = state === nothing ? () : (state,)
    cache_hints = ((), (_CACHE_HINT,))
    prefetches = ((), ((x,) for x in _LD_PREFETCH)...)
    l1s = ((), ((x,) for x in _LD_L1_EVICTION)...)
    l2s = ((), ((x,) for x in _LD_L2_EVICTION)...)

    for weak in ((), (:weak,)), state in states_ld
        prefix = (weak..., state_tuple(state)...)
        for cacheop in ((), ((x,) for x in _LD_CACHE_OPS)...),
            hint in cache_hints, prefetch in prefetches
            add(:ld, (prefix..., cacheop..., hint..., prefetch..., :b128))
        end
        for l1 in l1s, l2 in l2s, hint in cache_hints, prefetch in prefetches
            add(:ld, (prefix..., l1..., l2..., hint..., prefetch..., :b128))
        end
    end
    for sem in (:volatile, :relaxed, :acquire), scope in (nothing, _VECTOR_SCOPES...),
        state in states_ld, l1 in l1s, hint in cache_hints,
        prefetch in prefetches
        sem === :volatile && (scope !== nothing || l1 != () || hint != ()) && continue
        sem === :volatile || scope !== nothing || continue
        prefix = sem === :volatile ? (sem, state_tuple(state)...) :
                 (sem, scope, state_tuple(state)..., l1..., hint...,
                  prefetch...)
        sem === :volatile && (prefix = (sem, state_tuple(state)..., prefetch...))
        add(:ld, (prefix..., :b128))
    end
    for sem in (:relaxed, :acquire, :release), state in (nothing, :global)
        add(:ld, (:mmio, sem, :sys, state_tuple(state)..., :b128))
    end

    # ld.global.nc has its cache operation before `.nc`; the alternative
    # eviction syntax starts immediately after `.nc`.
    for cacheop in ((), ((x,) for x in (:ca, :cg, :cs))...),
        hint in cache_hints, prefetch in prefetches
        add(:ld, (:global, cacheop..., :nc, hint..., prefetch..., :b128))
    end
    for l1 in l1s, l2 in l2s, hint in cache_hints, prefetch in prefetches
        add(:ld, (:global, :nc, l1..., l2..., hint..., prefetch..., :b128))
    end
    add(:ldu, (:b128,)); add(:ldu, (:global, :b128))

    for weak in ((), (:weak,)), state in states_st
        prefix = (weak..., state_tuple(state)...)
        for cacheop in ((), ((x,) for x in _B128_ST_CACHE_OPS)...), hint in cache_hints
            add(:st, (prefix..., cacheop..., hint..., :b128))
        end
        for l1 in l1s, l2 in l2s, hint in cache_hints
            add(:st, (prefix..., l1..., l2..., hint..., :b128))
        end
    end
    for sem in (:volatile, :relaxed, :release), scope in (nothing, _VECTOR_SCOPES...),
        state in states_st, l1 in l1s, hint in cache_hints
        sem === :volatile && (scope !== nothing || l1 != () || hint != ()) && continue
        sem === :volatile || scope !== nothing || continue
        prefix = sem === :volatile ? (sem, state_tuple(state)...) :
                 (sem, scope, state_tuple(state)..., l1..., hint...)
        add(:st, (prefix..., :b128))
    end
    for sem in (:relaxed, :release, :acquire), state in (nothing, :global)
        add(:st, (:mmio, sem, :sys, state_tuple(state)..., :b128))
    end

    for sem in (nothing, :relaxed, :acquire, :release, :acq_rel),
        scope in (nothing, _VECTOR_SCOPES...),
        state in (nothing, _B128_ATOM_STATES...), atomop in (:exch, :cas, :add),
        hint in ((), (_CACHE_HINT,))
        prefix = (sem === nothing ? () : (sem,))
        scope_part = scope === nothing ? () : (scope,)
        prefix = (prefix..., scope_part...,
                  state_tuple(state)..., atomop, hint..., :b128)
        add(:atom, prefix)
    end

    add(:clusterlaunchcontrol, (:query_cancel, :is_canceled, :pred, :b128))
    add(:clusterlaunchcontrol,
        (:query_cancel, :get_first_ctaid, :v4, :b32, :b128))
    for dim in (:x, :y, :z, :w)
        add(:clusterlaunchcontrol,
            (:query_cancel, Symbol("get_first_ctaid::", dim), :b32, :b128))
    end
    keys
end

const B128_FORM_SCHEMAS = let schemas = Dict{Tuple{Symbol,Tuple},B128FormSchema}()
    for (op, mods) in _b128_candidate_keys()
        schema = _parse_b128_form_schema(op, mods)
        schema === nothing || (schemas[(op, mods)] = schema)
    end
    isempty(schemas) && error("scalar-b128 form ledger is empty")
    schemas
end

schema(::B128Ledger, op::Symbol, mods::Tuple{Vararg{Symbol}}) =
    get(B128_FORM_SCHEMAS, (op, mods), nothing)

function claims(::B128Ledger, op::Symbol,
                mods::Tuple{Vararg{Symbol}})
    :b128 in mods || return false
    op in (:mov, :ld, :ldu, :st, :atom) && return true
    op === :clusterlaunchcontrol && :query_cancel in mods && return true
    false
end

function miss(::B128Ledger, op::Symbol, mods::Tuple{Vararg{Symbol}})
    spelling = isempty(mods) ? string(op) : string(op, ".", join(mods, "."))
    ArgumentError(
        "ptx\"$spelling\" is inside the audited scalar-b128 grammar island " *
        "but is not one of PTX 9.3's reviewed mov, scalar ld/ldu/st, " *
        "atom.{exch,cas}, or clusterlaunchcontrol.query_cancel forms. " *
        "Refusing scalar/void fallback inference; the raw tier cannot supply " *
        "an explicit 128-bit register ABI either.")
end

# Deliberately NOT unified with the vector/mbarrier address predicates: b128
# forms reject Val immediates and any-space LLVMPtr is accepted here, so the
# accept-set difference is semantic.
_b128_address_type(::Type{T}) where {T} =
    T <: Core.LLVMPtr || T in (UInt32, Int32, UInt64, Int64) ||
    (T <: Address && is_ptx_address_integer_type(T.parameters[1]))
_b128_cache_policy_type(::Type{T}) where {T} = T in (UInt64, Int64)

function validate_b128_form_args(schema::B128FormSchema, argtypes)
    length(argtypes) == length(schema.operands) || throw(ArgumentError(
        "ptx\"$(schema.op).$(join(schema.mods, '.'))\" requires " *
        "$(length(schema.operands)) operand(s), got $(length(argtypes)); " *
        "see $(schema.section)"))
    for (i, (kind, T)) in enumerate(zip(schema.operands, argtypes))
        valid = kind === :address ? _b128_address_type(T) :
                kind === :cache_policy ? _b128_cache_policy_type(T) :
                kind === :b128 ? T === B128 : false
        valid && continue
        expected = kind === :address ? "a 32/64-bit address carrier" :
                   kind === :cache_policy ? "a 64-bit cache-policy carrier" :
                   "PTX.B128 (NTuple{2,UInt64})"
        throw(ArgumentError(
            "ptx\"$(schema.op).$(join(schema.mods, '.'))\" operand $i must " *
            "use $expected, got $T; see $(schema.section)"))
    end
    nothing
end

validate_ledger_args(::B128Ledger, s::B128FormSchema, argtypes) =
    validate_b128_form_args(s, argtypes)
