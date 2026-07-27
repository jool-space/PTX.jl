using PTX: B128, Address, Operation, RawOperation, build_call, b128

function _expected_b128_form_keys()
    keys = Set{Tuple{Symbol,Tuple}}()
    add(op, mods) = push!(keys, (op, Tuple(mods)))
    one(x) = x === nothing ? () : (x,)
    states_ld = (nothing, :const, :global, :local, :param,
                 Symbol("param::entry"), Symbol("param::func"), :shared,
                 Symbol("shared::cta"), Symbol("shared::cluster"))
    states_st = (nothing, :global, :local, :param, Symbol("param::func"),
                 :shared, Symbol("shared::cta"), Symbol("shared::cluster"))
    memory_states = (nothing, :global, :shared, Symbol("shared::cta"),
                     Symbol("shared::cluster"))
    scopes = (:cta, :cluster, :gpu, :sys)
    l1s = (nothing, Symbol("L1::evict_normal"),
           Symbol("L1::evict_unchanged"), Symbol("L1::evict_first"),
           Symbol("L1::evict_last"), Symbol("L1::no_allocate"))
    prefetches = (nothing, Symbol("L2::64B"), Symbol("L2::128B"),
                  Symbol("L2::256B"))
    hints = (false, true)

    add(:mov, (:b128,))
    for weak in (false, true), state in states_ld
        weakpart = weak ? (:weak,) : ()
        lead = (weakpart..., one(state)...)
        for cop in (nothing, :ca, :cg, :cs, :lu, :cv), hint in hints,
            prefetch in prefetches
            (hint || prefetch !== nothing) && !(state in (nothing, :global)) &&
                continue
            hintpart = hint ? (Symbol("L2::cache_hint"),) : ()
            add(:ld, (lead..., one(cop)..., hintpart...,
                      one(prefetch)..., :b128))
        end
        for l1 in l1s, hint in hints, prefetch in prefetches
            (hint || prefetch !== nothing) && !(state in (nothing, :global)) &&
                continue
            hintpart = hint ? (Symbol("L2::cache_hint"),) : ()
            add(:ld, (lead..., one(l1)..., hintpart...,
                      one(prefetch)..., :b128))
        end
    end
    for state in (nothing, :global, :local, :shared,
                  Symbol("shared::cta"), Symbol("shared::cluster")),
        prefetch in prefetches
        prefetch !== nothing && !(state in (nothing, :global)) && continue
        add(:ld, (:volatile, one(state)..., one(prefetch)..., :b128))
    end
    for sem in (:relaxed, :acquire), scope in scopes, state in memory_states,
        l1 in l1s, hint in hints, prefetch in prefetches
        (hint || prefetch !== nothing) && !(state in (nothing, :global)) &&
            continue
        hintpart = hint ? (Symbol("L2::cache_hint"),) : ()
        add(:ld, (sem, scope, one(state)..., one(l1)..., hintpart...,
                  one(prefetch)..., :b128))
    end
    for sem in (:relaxed, :acquire), state in (nothing, :global)
        add(:ld, (:mmio, sem, :sys, one(state)..., :b128))
    end
    for cop in (nothing, :ca, :cg, :cs), hint in hints,
        prefetch in prefetches
        hintpart = hint ? (Symbol("L2::cache_hint"),) : ()
        add(:ld, (:global, one(cop)..., :nc, hintpart...,
                  one(prefetch)..., :b128))
    end
    for l1 in l1s, hint in hints, prefetch in prefetches
        hintpart = hint ? (Symbol("L2::cache_hint"),) : ()
        add(:ld, (:global, :nc, one(l1)..., hintpart...,
                  one(prefetch)..., :b128))
    end
    add(:ldu, (:b128,)); add(:ldu, (:global, :b128))

    for weak in (false, true), state in states_st
        weakpart = weak ? (:weak,) : ()
        lead = (weakpart..., one(state)...)
        for cop in (nothing, :wb, :cg, :cs, :wt), hint in hints
            hint && !(state in (nothing, :global)) && continue
            hintpart = hint ? (Symbol("L2::cache_hint"),) : ()
            add(:st, (lead..., one(cop)..., hintpart..., :b128))
        end
        for l1 in l1s, hint in hints
            hint && !(state in (nothing, :global)) && continue
            hintpart = hint ? (Symbol("L2::cache_hint"),) : ()
            add(:st, (lead..., one(l1)..., hintpart..., :b128))
        end
    end
    for state in (nothing, :global, :local, :shared,
                  Symbol("shared::cta"), Symbol("shared::cluster"))
        add(:st, (:volatile, one(state)..., :b128))
    end
    for sem in (:relaxed, :release), scope in scopes, state in memory_states,
        l1 in l1s, hint in hints
        hint && !(state in (nothing, :global)) && continue
        hintpart = hint ? (Symbol("L2::cache_hint"),) : ()
        add(:st, (sem, scope, one(state)..., one(l1)..., hintpart..., :b128))
    end
    for sem in (:relaxed, :release), state in (nothing, :global)
        add(:st, (:mmio, sem, :sys, one(state)..., :b128))
    end

    for sem in (nothing, :relaxed, :acquire, :release, :acq_rel),
        scope in (nothing, scopes...), state in memory_states,
        atomop in (:exch, :cas)
        add(:atom, (one(sem)..., one(scope)..., one(state)...,
                    atomop, :b128))
        atomop === :exch && state in (nothing, :global) &&
            add(:atom, (one(sem)..., one(scope)..., one(state)...,
                        atomop, Symbol("L2::cache_hint"), :b128))
    end

    add(:clusterlaunchcontrol, (:query_cancel, :is_canceled, :pred, :b128))
    add(:clusterlaunchcontrol,
        (:query_cancel, :get_first_ctaid, :v4, :b32, :b128))
    for dim in (:x, :y, :z)
        add(:clusterlaunchcontrol,
            (:query_cancel, Symbol("get_first_ctaid::", dim), :b32, :b128))
    end
    keys
end

@testset "independent exhaustive b128 form matrix" begin
    expected = _expected_b128_form_keys()
    actual = Set(keys(PTX.B128_FORM_SCHEMAS))
    @test isempty(setdiff(actual, expected))
    @test isempty(setdiff(expected, actual))
    histogram = Dict(op => count(key -> key[1] === op, expected)
                     for op in (:mov, :ld, :ldu, :st, :atom,
                                :clusterlaunchcontrol))
    @test histogram == Dict(:mov => 1, :ld => 1528, :ldu => 2, :st => 546,
                            :atom => 300, :clusterlaunchcontrol => 5)
end

# --- Audited grammar roundtrip -----------------------------------------------
# This parse/validate machinery used to run at every package load, where the
# validator (not a hand-picked list) decided B128_FORM_SCHEMAS membership by
# filtering a candidate superset that deliberately includes nearby invalid
# spellings. The table is now constructed directly in
# src/ledgers/b128_forms.jl; the machinery moved here verbatim so the review
# property survives: the roundtrip testset below re-derives the full table
# through the real grammar rules (modifier ordering plus prose restrictions)
# and asserts exact schema-for-schema equality with the shipped table. It is
# deliberately NOT collapsed into _expected_b128_form_keys above, which stays
# an independent hand-spelled oracle of the same inventory.
using PTX: B128FormSchema, VectorResultCoreForm, _TARGET_SM100,
    _ld_prefix_info, _take_optional, _LD_CACHE_OPS, _LD_L1_EVICTION,
    _LD_L2_EVICTION, _LD_PREFETCH, _CACHE_HINT, _VECTOR_SCOPES,
    _B128_STATES_LD, _B128_STATES_ST, _B128_ATOM_STATES, _B128_ST_CACHE_OPS,
    _B128_MOV_SECTION, _B128_LD_SECTION, _B128_LD_NC_SECTION,
    _B128_LDU_SECTION, _B128_ST_SECTION, _B128_ATOM_SECTION,
    _B128_QUERY_SECTION

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

@testset "audited grammar roundtrip still defines the table" begin
    rebuilt = Dict{Tuple{Symbol,Tuple},B128FormSchema}()
    for (op, mods) in _b128_candidate_keys()
        schema = _parse_b128_form_schema(op, mods)
        schema === nothing || (rebuilt[(op, mods)] = schema)
    end
    @test isempty(setdiff(keys(rebuilt), keys(PTX.B128_FORM_SCHEMAS)))
    @test isempty(setdiff(keys(PTX.B128_FORM_SCHEMAS), keys(rebuilt)))
    @test all(rebuilt[key] == PTX.B128_FORM_SCHEMAS[key]
              for key in keys(rebuilt))
    @test rebuilt == PTX.B128_FORM_SCHEMAS
end

const EXPECTED_B128_ACCEPTED = (
    (:mov, (:b128,), :mov, B128, (:b128,)),
    (:ld, (:b128,), :load, B128, (:address,)),
    (:ld, (:global, :b128), :load, B128, (:address,)),
    (:ld, (:weak, :global, :ca, :b128), :load, B128, (:address,)),
    (:ld, (:volatile, :local, :b128), :load, B128, (:address,)),
    (:ld, (:acquire, :sys, :global, :b128), :load, B128, (:address,)),
    (:ld, (:global, Symbol("L2::cache_hint"), :b128), :load, B128,
     (:address, :cache_policy)),
    (:ld, (:global, Symbol("L2::256B"), :b128), :load, B128, (:address,)),
    (:ld, (:global, :nc, :b128), :load, B128, (:address,)),
    (:ld, (:global, :cg, :nc, Symbol("L2::128B"), :b128), :load, B128,
     (:address,)),
    (:ld, (:mmio, :acquire, :sys, :global, :b128), :load, B128, (:address,)),
    (:ldu, (:b128,), :load, B128, (:address,)),
    (:ldu, (:global, :b128), :load, B128, (:address,)),
    (:st, (:b128,), :store, Nothing, (:address, :b128)),
    (:st, (:global, :wb, :b128), :store, Nothing, (:address, :b128)),
    (:st, (:volatile, :local, :b128), :store, Nothing, (:address, :b128)),
    (:st, (:release, :sys, :global, :b128), :store, Nothing,
     (:address, :b128)),
    (:st, (:global, Symbol("L2::cache_hint"), :b128), :store, Nothing,
     (:address, :b128, :cache_policy)),
    (:st, (:mmio, :release, :sys, :global, :b128), :store, Nothing,
     (:address, :b128)),
    (:atom, (:global, :exch, :b128), :exch, B128, (:address, :b128)),
    (:atom, (:acq_rel, :sys, :global, :cas, :b128), :cas, B128,
     (:address, :b128, :b128)),
    (:atom, (:global, :exch, Symbol("L2::cache_hint"), :b128), :exch,
     B128, (:address, :b128, :cache_policy)),
    (:clusterlaunchcontrol, (:query_cancel, :is_canceled, :pred, :b128),
     :query_pred, Bool, (:b128,)),
    (:clusterlaunchcontrol,
     (:query_cancel, :get_first_ctaid, :v4, :b32, :b128), :query_v4,
     NTuple{4,UInt32}, (:b128,)),
    (:clusterlaunchcontrol,
     (:query_cancel, Symbol("get_first_ctaid::x"), :b32, :b128),
     :query_dim, UInt32, (:b128,)),
    (:clusterlaunchcontrol,
     (:query_cancel, Symbol("get_first_ctaid::y"), :b32, :b128),
     :query_dim, UInt32, (:b128,)),
    (:clusterlaunchcontrol,
     (:query_cancel, Symbol("get_first_ctaid::z"), :b32, :b128),
     :query_dim, UInt32, (:b128,)),
)

@testset "closed PTX 9.3 scalar-b128 grammar and carrier" begin
    @test B128 === NTuple{2,UInt64}
    @test b128(0x0123456789abcdef, 0xfedcba9876543210) ==
          (0x0123456789abcdef, 0xfedcba9876543210)
    @test b128(0x89abcdef, 0x01234567, 0x76543210, 0xfedcba98) ==
          (0x0123456789abcdef, 0xfedcba9876543210)
    for (op, mods, kind, result, operands) in EXPECTED_B128_ACCEPTED
        schema = PTX.schema(PTX.B128Ledger(), op, mods)
        @test schema !== nothing
        @test schema.kind === kind
        @test schema.result === result
        @test schema.operands == operands
        @test startswith(schema.section, "ptx/9-instruction-set/")
    end
end

const REJECTED_B128_FORMS = (
    (:ld, (:v2, :b128)),              # vectors cannot exceed 128 bits
    (:ld, (Symbol("L2::evict_last"), :b128)), # only wide v8.b32/v4.b64
    (:ldu, (:shared, :b128)),
    (:st, (:v2, :b128)),
    (:atom, (:global, :add, :b128)),  # Table 35: exch/cas only
    (:atom, (:shared, :cas, Symbol("L2::cache_hint"), :b128)),
    (:clusterlaunchcontrol, (:query_cancel, :is_canceled, :b128)),
    (:clusterlaunchcontrol,
     (:query_cancel, Symbol("get_first_ctaid::w"), :b32, :b128)),
)

@testset "b128 grammar misses fail loud in direct, raw, and lowering" begin
    A = Address{UInt64}
    for (op, mods) in REJECTED_B128_FORMS
        args = op === :atom ? (A, B128) :
               op === :clusterlaunchcontrol ? (B128,) : (A,)
        ordinary = Operation{op,mods}()
        raw = RawOperation{op,mods}()
        @test PTX.schema(PTX.B128Ledger(), op, mods) === nothing
        @test PTX.lowering(ordinary, args).tier === :forbidden
        @test PTX.lowering(raw, args).tier === :forbidden
        @test_throws ArgumentError build_call(op, mods, args)
        @test_throws ArgumentError build_call(op, mods, args; raw = true)
    end
end

@testset "b128 exact ABI rendering" begin
    A = Address{UInt64}
    ld = build_call(:ld, (:global, :b128), (A,))
    @test ld.rettype === B128
    @test ld.constraints == "=l,=l,l,~{memory}"
    @test occursin(".reg .b128 b128_result", ld.asm)
    @test occursin("ld.global.b128 b128_result, [\$2]", ld.asm)
    @test occursin("mov.b128 {\$0, \$1}, b128_result", ld.asm)

    st = build_call(:st, (:global, :b128), (A, B128))
    @test st.rettype === Nothing
    @test st.constraints == "l,l,l,~{memory}"
    @test occursin("mov.b128 b128_value1, {\$1, \$2}", st.asm)
    @test occursin("st.global.b128 [\$0], b128_value1", st.asm)

    atom = build_call(:atom, (:global, :cas, :b128), (A, B128, B128))
    @test atom.rettype === B128
    @test atom.constraints == "=l,=l,l,l,l,l,l,~{memory}"
    @test occursin("atom.global.cas.b128 b128_result, [\$2], b128_value1, b128_value2", atom.asm)

    query = build_call(:clusterlaunchcontrol,
        (:query_cancel, :get_first_ctaid, :v4, :b32, :b128), (B128,))
    @test query.rettype === NTuple{4,UInt32}
    @test query.constraints == "=r,=r,=r,=r,l,l,~{memory}"
    @test occursin("{\$0, \$1, \$2, \$3}, b128_value1", query.asm)

    for (op, mods, _, _, operands) in EXPECTED_B128_ACCEPTED
        argtypes = Tuple(kind === :address ? A :
                         kind === :b128 ? B128 : UInt64 for kind in operands)
        @test PTX.lowering(Operation{op,mods}(), argtypes).tier === :chain_asm
        @test PTX.lowering(RawOperation{op,mods}(), argtypes).tier === :chain_asm
    end
end

@testset "every b128 ledger cell reaches direct and raw emitters" begin
    A = Address{UInt64}
    ordered = sort!(collect(values(PTX.B128_FORM_SCHEMAS));
                    by = schema -> (string(schema.op), join(schema.mods, ".")))
    @test length(ordered) == 2382
    for schema in ordered
        argtypes = Tuple(kind === :address ? A :
                         kind === :b128 ? B128 : UInt64
                         for kind in schema.operands)
        direct_spec = build_call(schema.op, schema.mods, argtypes)
        raw_spec = build_call(schema.op, schema.mods, argtypes; raw = true)
        @test direct_spec.rettype === schema.result
        @test raw_spec.rettype === schema.result
        @test direct_spec.asm == raw_spec.asm
        @test occursin(PTX.build_head(schema.op, schema.mods), direct_spec.asm)
    end
    # Do not add lowering() reflection to this 2,382-cell loop. Each distinct
    # Operation type triggers Julia specialization; an attempted exhaustive
    # reflection sweep ran for more than six minutes. The 27 cross-family
    # accepted representatives above exercise ordinary/raw lowering, while
    # every negative grammar miss is reflected exhaustively. build_call is the
    # shared emitter used by both lowering routes and remains exhaustive here.
end

@testset "b128 transpiler closes declaration and result roles" begin
    src = """.version 8.6
    .target sm_100
    .address_size 64
    .visible .entry b128_probe()
    {
      .reg .b64 %rd<4>;
      .reg .b32 %r<5>;
      .reg .pred %p;
      mov.b128 %handle, {%rd0, %rd1};
      clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 %p, %handle;
      clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%r0, %r1, %r2, %r3}, %handle;
      ret;
    }
    """
    julia = PTX.ptx_to_julia(src)
    @test occursin("handle = ptx\"mov.b128\"((rd0, rd1))", julia)
    @test occursin("p = ptx\"clusterlaunchcontrol.query_cancel.is_canceled.pred.b128\"(handle)", julia)
    @test occursin("(r0, r1, r2, r3) = ptx\"clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128\"(handle)", julia)
    @test Meta.parseall(julia) isa Expr

    bad = replace(src, "{%rd0, %rd1}" => "{%r0, %r1}")
    @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(bad)
    unpack = replace(src,
        "mov.b128 %handle, {%rd0, %rd1};" =>
        ".reg .b128 %handle;\nmov.b128 {%rd0, %rd1}, %handle;")
    @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(unpack)
end
