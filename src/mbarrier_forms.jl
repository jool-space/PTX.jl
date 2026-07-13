# Closed PTX ISA 9.3 §9.7.14.16 mbarrier grammar and ABI ledger.
#
# The terminal `.b64` describes the opaque mbarrier/state width, not a uniform
# result type.  Depending on the exact form, the destination is absent, a sink
# `_`, a u64 state token, a predicate, a u32 pending-arrival count, or the
# two-field `waitComplete|reportPredicate` group, or that group plus the
# optional 8-bit `reportValue`. NVPTX has no i8 inline-asm constraint, so the
# full report ABI carries that byte in the low half of a `UInt16`. Consequently
# no mbarrier spelling may use the generic terminal-modifier result heuristic.
#
# Entries below are generated only from the finite cross-products printed in
# the instruction subsection.  The map key is the complete *Julia* modifier
# chain; `:report_pred` and `:report` are PTX.jl's synthetic selectors for the
# predicate pair without/with reportValue, and are removed from `ptxmods`
# before rendering. Every other modifier is emitted verbatim and in the
# canonical syntax order.

struct MBarrierOperandVariant
    operands::Tuple{Vararg{Symbol}}
    ptx_version::VersionNumber
    min_sm::VersionNumber
end

struct MBarrierFormSchema
    mods::Tuple{Vararg{Symbol}}
    ptxmods::Tuple{Vararg{Symbol}}
    destination::Symbol
    variants::Tuple{Vararg{MBarrierOperandVariant}}
    space::Symbol
    section::String
end

const _MBARRIER_SECTION =
    "ptx/9-instruction-set/9.7.14.16-parallel-synchronization-and-communication-instructions-mbarrier.md"

_mb_max_version(xs...) = maximum(xs)

function _mbarrier_form!(schemas, mods, ptxmods, destination, variants,
                         space)
    destination in (:none, :state, :remote_sink, :predicate, :count,
                    :report_pred, :report) ||
        error("invalid mbarrier destination shape: ", destination)
    space in (:none, :generic, :cta, :cluster) ||
        error("invalid mbarrier state space: ", space)
    any(s -> s.mods == mods, schemas) &&
        error("duplicate mbarrier form: mbarrier.", join(mods, "."))
    push!(schemas, MBarrierFormSchema(mods, ptxmods, destination,
                                     Tuple(variants), space,
                                     _MBARRIER_SECTION))
end

_mb_variant(operands, ptx_version, min_sm) =
    MBarrierOperandVariant(operands, ptx_version, min_sm)

# (modifier suffix, PTX version, target floor, address-space class).
const _MB_LOCAL_SPACES = (
    ((),                         v"7.0", v"8.0", :generic),
    ((:shared,),                 v"7.0", v"8.0", :cta),
    ((Symbol("shared::cta"),),  v"7.8", v"8.0", :cta),
)
const _MB_ALL_SPACES = (
    _MB_LOCAL_SPACES...,
    ((Symbol("shared::cluster"),), v"8.0", v"9.0", :cluster),
)

# `.sem` and `.scope` are inseparable.  Keeping only complete pairs in these
# tables makes lone, reversed, and unsupported qualifiers closed-world misses.
const _MB_TX_SEM_SCOPES = (
    ((),                    v"8.0", v"9.0"),
    ((:relaxed, :cta),      v"8.0", v"9.0"),
    ((:relaxed, :cluster),  v"8.0", v"9.0"),
)
const _MB_ARRIVE_SEM_SCOPES = (
    ((),                    v"7.0", v"8.0"),
    ((:release, :cta),      v"8.0", v"8.0"),
    ((:release, :cluster),  v"8.0", v"9.0"),
    ((:relaxed, :cta),      v"8.6", v"9.0"),
    ((:relaxed, :cluster),  v"8.6", v"9.0"),
)
const _MB_WAIT_SEM_SCOPES = (
    ((),                    v"7.0", v"8.0"),
    ((:acquire, :cta),      v"8.0", v"8.0"),
    ((:acquire, :cluster),  v"8.0", v"9.0"),
    ((:relaxed, :cta),      v"8.6", v"9.0"),
    ((:relaxed, :cluster),  v"8.6", v"9.0"),
)

const MBARRIER_FORM_SCHEMAS = let schemas = MBarrierFormSchema[]
    # §9.7.14.16.12-.13: lifecycle sinks, local CTA storage only.
    for (layout, lp, ls) in (
            ((), v"7.0", v"8.0"),
            ((Symbol("layout::v0"),), v"9.3", v"9.0"),
            ((Symbol("layout::v1"),), v"9.3", v"9.0")),
        (space, sp, ss, space_kind) in _MB_LOCAL_SPACES
        mods = (:init, layout..., space..., :b64)
        pv, sm = _mb_max_version(lp, sp), _mb_max_version(ls, ss)
        _mbarrier_form!(schemas, mods, mods, :none,
                        (_mb_variant((:address, :u32), pv, sm),), space_kind)
    end
    for (space, sp, ss, space_kind) in _MB_LOCAL_SPACES
        mods = (:inval, space..., :b64)
        _mbarrier_form!(schemas, mods, mods, :none,
                        (_mb_variant((:address,), sp, ss),), space_kind)
    end

    # §9.7.14.16.14-.15: tx-count updates are sinks in both CTA and remote
    # cluster shared memory.  Only `.relaxed` exists, always paired with scope.
    for subop in (:expect_tx, :complete_tx),
        (sem, semp, semsm) in _MB_TX_SEM_SCOPES,
        (space, sp, ss, space_kind) in _MB_ALL_SPACES
        mods = (subop, sem..., space..., :b64)
        pv, sm = _mb_max_version(semp, sp), _mb_max_version(semsm, ss)
        _mbarrier_form!(schemas, mods, mods, :none,
                        (_mb_variant((:address, :u32), pv, sm),), space_kind)
    end

    # §9.7.14.16.16-.17: local arrivals return state; remote-cluster
    # arrivals must spell a sink destination.  A count on the ordinary form
    # has an sm_90 floor, while the noComplete count form is the sm_80 route.
    for subop in (:arrive, :arrive_drop),
        (sem, semp, semsm) in _MB_ARRIVE_SEM_SCOPES,
        (space, sp, ss, space_kind) in _MB_ALL_SPACES
        mods = (subop, sem..., space..., :b64)
        basep, basesm = _mb_max_version(semp, sp), _mb_max_version(semsm, ss)
        variants = (
            _mb_variant((:address,), basep, basesm),
            _mb_variant((:address, :u32),
                        _mb_max_version(basep, v"7.8"),
                        _mb_max_version(basesm, v"9.0")),
        )
        destination = space_kind === :cluster ? :remote_sink : :state
        _mbarrier_form!(schemas, mods, mods, destination, variants, space_kind)
    end
    for subop in (:arrive, :arrive_drop),
        (sem, semp, semsm) in _MB_ARRIVE_SEM_SCOPES,
        (space, sp, ss, space_kind) in _MB_ALL_SPACES
        mods = (subop, :expect_tx, sem..., space..., :b64)
        pv = _mb_max_version(v"8.0", semp, sp)
        sm = _mb_max_version(v"9.0", semsm, ss)
        destination = space_kind === :cluster ? :remote_sink : :state
        _mbarrier_form!(schemas, mods, mods, destination,
                        (_mb_variant((:address, :u32), pv, sm),), space_kind)
    end
    for subop in (:arrive, :arrive_drop),
        (sem, semp, semsm) in (((), v"7.0", v"8.0"),
                               ((:release, :cta), v"8.0", v"8.0")),
        (space, sp, ss, space_kind) in _MB_LOCAL_SPACES
        mods = (subop, :noComplete, sem..., space..., :b64)
        pv, sm = _mb_max_version(semp, sp), _mb_max_version(semsm, ss)
        _mbarrier_form!(schemas, mods, mods, :state,
                        (_mb_variant((:address, :u32), pv, sm),), space_kind)
    end

    # §9.7.14.16.19: single-predicate waits plus PTX.jl's explicit
    # `:report_pred` / `:report` selections for the otherwise
    # operand-overloaded primary-phase head. try_wait alone admits the optional
    # u32 timeHint.
    for wait in (:test_wait, :try_wait),
        (sem, semp, semsm) in _MB_WAIT_SEM_SCOPES,
        (space, sp, ss, space_kind) in _MB_LOCAL_SPACES
        waitp = wait === :try_wait ? v"7.8" : v"7.0"
        waitsm = wait === :try_wait ? v"9.0" : v"8.0"
        for phase in ((), (Symbol("phase_type::primary"),))
            mods = (wait, phase..., sem..., space..., :b64)
            pv = _mb_max_version(waitp, semp, sp,
                                 isempty(phase) ? v"7.0" : v"9.3")
            sm = _mb_max_version(waitsm, semsm, ss,
                                 isempty(phase) ? v"8.0" : v"9.0")
            variants = wait === :try_wait ?
                (_mb_variant((:address, :u64), pv, sm),
                 _mb_variant((:address, :u64, :u32), pv, sm)) :
                (_mb_variant((:address, :u64), pv, sm),)
            _mbarrier_form!(schemas, mods, mods, :predicate, variants, space_kind)
        end
        for phase in ((), (Symbol("phase_type::primary"),),
                      (Symbol("phase_type::conditional"),))
            mods = (wait, :parity, phase..., sem..., space..., :b64)
            phasep = isempty(phase) ? v"7.1" : v"9.3"
            phasesm = isempty(phase) ? v"8.0" : v"9.0"
            pv = _mb_max_version(waitp, v"7.1", phasep, semp, sp)
            sm = _mb_max_version(waitsm, phasesm, semsm, ss)
            variants = wait === :try_wait ?
                (_mb_variant((:address, :u32), pv, sm),
                 _mb_variant((:address, :u32, :u32), pv, sm)) :
                (_mb_variant((:address, :u32), pv, sm),)
            _mbarrier_form!(schemas, mods, mods, :predicate, variants, space_kind)
        end
        for parity in (false, true)
            ptxmods = parity ?
                (wait, :parity, Symbol("phase_type::primary"),
                 sem..., space..., :b64) :
                (wait, Symbol("phase_type::primary"),
                 sem..., space..., :b64)
            source = parity ? :u32 : :u64
            pv = _mb_max_version(v"9.3", semp, sp)
            sm = _mb_max_version(v"9.0", semsm, ss)
            variants = wait === :try_wait ?
                (_mb_variant((:address, source), pv, sm),
                 _mb_variant((:address, source, :u32), pv, sm)) :
                (_mb_variant((:address, source), pv, sm),)
            for (selector, destination) in
                    ((:report_pred, :report_pred), (:report, :report))
                julia_mods = parity ?
                    (wait, selector, :parity, Symbol("phase_type::primary"),
                     sem..., space..., :b64) :
                    (wait, selector, Symbol("phase_type::primary"),
                     sem..., space..., :b64)
                _mbarrier_form!(schemas, julia_mods, ptxmods, destination,
                                variants, space_kind)
            end
        end
    end

    # §9.7.14.16.20: state is a bare u64 register and the result is u32.
    for (layout, pv, sm) in (
            ((), v"7.0", v"8.0"),
            ((Symbol("layout::v0"),), v"9.3", v"9.0"))
        mods = (:pending_count, layout..., :b64)
        _mbarrier_form!(schemas, mods, mods, :count,
                        (_mb_variant((:u64,), pv, sm),), :none)
    end

    # §9.7.14.16.21: layout is mandatory; only generic or shared::cta.
    for layout in (Symbol("layout::v0"), Symbol("layout::v1")),
        (space, sp, ss, space_kind) in
            (((), v"9.3", v"9.0", :generic),
             ((Symbol("shared::cta"),), v"9.3", v"9.0", :cta))
        mods = (:check_layout, layout, space..., :b64)
        _mbarrier_form!(schemas, mods, mods, :predicate,
                        (_mb_variant((:address,), sp, ss),), space_kind)
    end

    Tuple(schemas)
end

const _MBARRIER_SCHEMA_BY_FORM =
    Dict(schema.mods => schema for schema in MBARRIER_FORM_SCHEMAS)
length(_MBARRIER_SCHEMA_BY_FORM) == length(MBARRIER_FORM_SCHEMAS) ||
    error("duplicate keys in MBARRIER_FORM_SCHEMAS")

mbarrier_form_schema(op::Symbol, mods::Tuple{Vararg{Symbol}}) =
    op === :mbarrier ? get(_MBARRIER_SCHEMA_BY_FORM, mods, nothing) : nothing

requires_mbarrier_schema(op::Symbol) = op === :mbarrier

function mbarrier_schema_miss(mods::Tuple{Vararg{Symbol}})
    spelling = isempty(mods) ? "mbarrier" : "mbarrier." * join(mods, ".")
    ArgumentError(
        "ptx\"$spelling\" does not match the reviewed PTX 9.3 mbarrier " *
        "grammar. Refusing terminal-.b64 result inference; check the exact " *
        "modifier order, sem/scope pairing, phase/parity combination, and " *
        "state space. The raw tier also rejects grammar misses because only " *
        "an exact ledger hit supplies mbarrier's sink or grouped-result ABI; " *
        "see $_MBARRIER_SECTION")
end

_mbarrier_rettype(schema::MBarrierFormSchema) =
    schema.destination in (:none, :remote_sink) ? Nothing :
    schema.destination === :state ? UInt64 :
    schema.destination === :predicate ? Bool :
    schema.destination === :count ? UInt32 :
    schema.destination === :report_pred ? Tuple{Bool, Bool} :
    schema.destination === :report ? Tuple{Bool, Bool, UInt16} :
    error("invalid mbarrier destination shape: ", schema.destination)

function _mbarrier_operand_accepts(schema::MBarrierFormSchema,
                                   kind::Symbol, ::Type{T}) where {T}
    if kind === :address
        if T <: Core.LLVMPtr
            as = T.parameters[2]
            return schema.space === :generic ? as == 0 : as == 3
        end
        # Transpiled PTX register addresses remain integer carriers until a
        # separately translated symbol/alias supplies LLVMPtr evidence.
        return T === UInt32 || T === Int32 || T === UInt64 || T === Int64
    elseif kind === :u32
        return T === UInt32 || T === Int32 || _integer_immediate(T)
    elseif kind === :u64
        return T === UInt64 || T === Int64 || _integer_immediate(T)
    end
    error("unknown mbarrier operand kind: ", kind)
end

function _mbarrier_operand_description(schema::MBarrierFormSchema, kind::Symbol)
    kind === :address && return schema.space === :generic ?
        "an address-space-0 LLVMPtr or a 32/64-bit integer address carrier" :
        "an address-space-3 LLVMPtr or a 32/64-bit integer address carrier"
    kind === :u32 && return "a 32-bit integer or integer Val immediate"
    kind === :u64 && return "a 64-bit integer or integer Val immediate"
    string(kind)
end

function validate_mbarrier_args(schema::MBarrierFormSchema, argtypes)
    variant = findfirst(v -> length(v.operands) == length(argtypes),
                        schema.variants)
    variant === nothing && throw(ArgumentError(
        "ptx\"mbarrier.$(join(schema.mods, "."))\" accepts source arities " *
        join(sort!(unique(length(v.operands) for v in schema.variants)), " or ") *
        ", got $(length(argtypes)); see $(schema.section)"))
    selected = schema.variants[variant]
    for (i, (kind, T)) in enumerate(zip(selected.operands, argtypes))
        _mbarrier_operand_accepts(schema, kind, T) && continue
        throw(ArgumentError(
            "ptx\"mbarrier.$(join(schema.mods, "."))\" operand $i must use " *
            _mbarrier_operand_description(schema, kind) *
            ", got $T; see $(schema.section)"))
    end
    selected
end
