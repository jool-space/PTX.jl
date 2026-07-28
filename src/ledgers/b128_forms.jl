# Audited scalar `.b128` register grammar and Julia carrier policy.
#
# PTX has a real 128-bit bit register, but LLVM's NVPTX inline-assembly
# constraints expose only predicate, 16-, 32-, and 64-bit register classes.
# Use the ISA's own `mov.b128` pack/unpack operation to bridge every `.b128`
# register through one stable Julia carrier: two low-to-high UInt64 words.
# See PTX ISA 9.3:
#   ptx/9-instruction-set/9.7.10.4-data-movement-and-conversion-instructions-mov.md
#   ptx/9-instruction-set/9.7.10.8-data-movement-and-conversion-instructions-ld.md
#   ptx/9-instruction-set/9.7.10.9-data-movement-and-conversion-instructions-ld.global.nc.md
#   ptx/9-instruction-set/9.7.10.10-data-movement-and-conversion-instructions-ldu.md
#   ptx/9-instruction-set/9.7.10.11-data-movement-and-conversion-instructions-st.md
#   ptx/9-instruction-set/9.7.15.5-parallel-synchronization-and-communication-instructions-atom.md
#   ptx/9-instruction-set/9.7.15.19-parallel-synchronization-and-communication-instructions-clusterlaunchcontrol.query_cancel.md

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
    "ptx/9-instruction-set/9.7.10.4-data-movement-and-conversion-instructions-mov.md"
const _B128_LD_SECTION =
    "ptx/9-instruction-set/9.7.10.8-data-movement-and-conversion-instructions-ld.md"
const _B128_LD_NC_SECTION =
    "ptx/9-instruction-set/9.7.10.9-data-movement-and-conversion-instructions-ld.global.nc.md"
const _B128_LDU_SECTION =
    "ptx/9-instruction-set/9.7.10.10-data-movement-and-conversion-instructions-ldu.md"
const _B128_ST_SECTION =
    "ptx/9-instruction-set/9.7.10.11-data-movement-and-conversion-instructions-st.md"
const _B128_ATOM_SECTION =
    "ptx/9-instruction-set/9.7.15.5-parallel-synchronization-and-communication-instructions-atom.md"
const _B128_QUERY_SECTION =
    "ptx/9-instruction-set/9.7.15.19-parallel-synchronization-and-communication-instructions-clusterlaunchcontrol.query_cancel.md"

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

# Materialize the complete finite grammar island directly from its audited
# acceptance rules. Membership used to be decided at every package load by
# running a candidate superset through a b128 grammar parser; that
# parse/validate sweep now lives verbatim in test/host/b128_forms.jl, whose
# roundtrip testset re-derives this table through the real grammar (modifier
# ordering plus prose restrictions) and asserts exact schema-for-schema
# equality — alongside that file's fully independent expected-key oracle and
# count pins. Editing the rules below without updating both test-side
# oracles is a pinned failure, not a silent drift.
function _b128_form_schemas()
    schemas = Dict{Tuple{Symbol,Tuple},B128FormSchema}()
    function add!(op::Symbol, mods::Vector{Symbol}, kind::Symbol,
                  result::Type, operands::Tuple{Vararg{Symbol}},
                  cache_policy::Bool, section::String)
        key = (op, (mods...,))
        schemas[key] = B128FormSchema(op, key[2], kind, result, operands,
                                      cache_policy, section)
        nothing
    end
    function mods_of(parts::Union{Nothing,Symbol}...)
        mods = Symbol[]
        for part in parts
            part === nothing || push!(mods, part)
        end
        push!(mods, :b128)
        mods
    end
    ld!(mods::Vector{Symbol}, hint::Bool) =
        add!(:ld, mods, :load, B128,
             hint ? (:address, :cache_policy) : (:address,), hint,
             :nc in mods ? _B128_LD_NC_SECTION : _B128_LD_SECTION)
    st!(mods::Vector{Symbol}, hint::Bool) =
        add!(:st, mods, :store, Nothing,
             hint ? (:address, :b128, :cache_policy) : (:address, :b128),
             hint, _B128_ST_SECTION)
    hint_mod(hint::Bool) = hint ? _CACHE_HINT : nothing
    withnothing(tokens) =
        Union{Nothing,Symbol}[nothing; sort!(collect(Symbol, tokens))]

    states_ld = withnothing(_B128_STATES_LD)
    states_st = withnothing(_B128_STATES_ST)
    states_mem = withnothing(_B128_ATOM_STATES)
    states_volatile = Union{Nothing,Symbol}[
        nothing, :global, :local, :shared, Symbol("shared::cta"),
        Symbol("shared::cluster")]
    scopes = sort!(collect(Symbol, _VECTOR_SCOPES))
    l1s = withnothing(_LD_L1_EVICTION)
    prefetches = withnothing(_LD_PREFETCH)
    ld_cache_ops = withnothing(_LD_CACHE_OPS)
    nc_cache_ops = Union{Nothing,Symbol}[nothing, :ca, :cg, :cs]
    st_cache_ops = withnothing(_B128_ST_CACHE_OPS)
    # Cache hints and prefetch sizes require the generic or global space.
    hintable(state) = state === nothing || state === :global

    add!(:mov, Symbol[:b128], :mov, B128, (:b128,), false, _B128_MOV_SECTION)

    # ld weak tier: the two syntax alternatives differ only in cache-op
    # versus L1-eviction qualifiers (scalar b128 admits no L2 eviction).
    for weak in (false, true), state in states_ld
        weakmod = weak ? :weak : nothing
        for hint in (false, true), prefetch in prefetches
            (hint || prefetch !== nothing) && !hintable(state) && continue
            for cop in ld_cache_ops
                ld!(mods_of(weakmod, state, cop, hint_mod(hint), prefetch),
                    hint)
            end
            for l1 in l1s
                ld!(mods_of(weakmod, state, l1, hint_mod(hint), prefetch),
                    hint)
            end
        end
    end
    # ld volatile / relaxed / acquire tiers, then MMIO.
    for state in states_volatile, prefetch in prefetches
        prefetch !== nothing && !hintable(state) && continue
        ld!(mods_of(:volatile, state, prefetch), false)
    end
    for sem in (:relaxed, :acquire), scope in scopes, state in states_mem,
        l1 in l1s, hint in (false, true), prefetch in prefetches
        (hint || prefetch !== nothing) && !hintable(state) && continue
        ld!(mods_of(sem, scope, state, l1, hint_mod(hint), prefetch), hint)
    end
    for sem in (:relaxed, :acquire), state in (nothing, :global)
        ld!(mods_of(:mmio, sem, :sys, state), false)
    end
    # ld.global.nc has its cache operation before `.nc`; the alternative
    # eviction syntax starts immediately after `.nc`.
    for hint in (false, true), prefetch in prefetches
        for cop in nc_cache_ops
            ld!(mods_of(:global, cop, :nc, hint_mod(hint), prefetch), hint)
        end
        for l1 in l1s
            ld!(mods_of(:global, :nc, l1, hint_mod(hint), prefetch), hint)
        end
    end
    add!(:ldu, Symbol[:b128], :load, B128, (:address,), false,
         _B128_LDU_SECTION)
    add!(:ldu, Symbol[:global, :b128], :load, B128, (:address,), false,
         _B128_LDU_SECTION)

    # st weak / volatile / relaxed / release tiers, then MMIO.
    for weak in (false, true), state in states_st
        weakmod = weak ? :weak : nothing
        for hint in (false, true)
            hint && !hintable(state) && continue
            for cop in st_cache_ops
                st!(mods_of(weakmod, state, cop, hint_mod(hint)), hint)
            end
            for l1 in l1s
                st!(mods_of(weakmod, state, l1, hint_mod(hint)), hint)
            end
        end
    end
    for state in states_volatile
        st!(mods_of(:volatile, state), false)
    end
    for sem in (:relaxed, :release), scope in scopes, state in states_mem,
        l1 in l1s, hint in (false, true)
        hint && !hintable(state) && continue
        st!(mods_of(sem, scope, state, l1, hint_mod(hint)), hint)
    end
    for sem in (:relaxed, :release), state in (nothing, :global)
        st!(mods_of(:mmio, sem, :sys, state), false)
    end

    # atom: Table 35 admits only exch and cas at .b128; the cache hint is an
    # exch-only qualifier and requires the generic or global space.
    for sem in Union{Nothing,Symbol}[nothing, :relaxed, :acquire, :release,
                                     :acq_rel],
        scope in Union{Nothing,Symbol}[nothing; scopes], state in states_mem
        for atomop in (:exch, :cas)
            add!(:atom, mods_of(sem, scope, state, atomop), atomop, B128,
                 atomop === :cas ? (:address, :b128, :b128) :
                                   (:address, :b128),
                 false, _B128_ATOM_SECTION)
        end
        hintable(state) &&
            add!(:atom, mods_of(sem, scope, state, :exch, _CACHE_HINT),
                 :exch, B128, (:address, :b128, :cache_policy), true,
                 _B128_ATOM_SECTION)
    end

    add!(:clusterlaunchcontrol,
         Symbol[:query_cancel, :is_canceled, :pred, :b128],
         :query_pred, Bool, (:b128,), false, _B128_QUERY_SECTION)
    add!(:clusterlaunchcontrol,
         Symbol[:query_cancel, :get_first_ctaid, :v4, :b32, :b128],
         :query_v4, NTuple{4,UInt32}, (:b128,), false, _B128_QUERY_SECTION)
    for dim in (:x, :y, :z)
        add!(:clusterlaunchcontrol,
             Symbol[:query_cancel, Symbol("get_first_ctaid::", dim), :b32,
                    :b128],
             :query_dim, UInt32, (:b128,), false, _B128_QUERY_SECTION)
    end
    isempty(schemas) && error("scalar-b128 form ledger is empty")
    schemas
end

const B128_FORM_SCHEMAS = _b128_form_schemas()

schema(::B128Ledger, op::Symbol, mods::Tuple{Vararg{Symbol}}) =
    get(B128_FORM_SCHEMAS, (op, mods), nothing)

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
