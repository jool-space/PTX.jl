# The wrapper registry — the single bookkeeping channel for every generated
# wrapper family. Each `_*_register` generator records what it defined here
# instead of maintaining its own module-level accumulator; the test-side
# oracles (which deliberately re-derive the same inventories from the ISA —
# see CLAUDE.md, "double-entry") read it back through the query helpers below.
#
# One record per distinct (family, op, mods, tier, intrinsic) key. A family
# may record several intrinsics under one mods tuple (the tcgen05 mma grids
# fan one modifier spelling out over base/mask/scale intrinsic variants), and
# re-invoking a register helper (the coverage testset in test/host/wrappers.jl
# does) must not grow the registry — the shared `_WRAPPER_KEYS` set is the one
# idempotency mechanism, replacing the fifteen per-accumulator push guards.
#
# Tiers:
#   :intrinsic — the method emits an `NVVM.IntrinsicCall`.
#   :asm       — the method emits inline assembly (family-local spec builder).
#   :core_ir   — the method emits plain LLVM IR (no instruction-specific call).
#   :missing   — no method was defined: the generator expected an intrinsic
#                the pinned registry lacks and has no asm fallback. Recorded
#                so conformance can fail loudly naming the intrinsic instead
#                of a form silently vanishing (see `_mma_sp_register`).

struct WrapperRecord
    family::Symbol
    op::Symbol
    mods::Tuple{Vararg{Symbol}}
    tier::Symbol
    intrinsic::Union{String, Nothing}
end

const WRAPPER_REGISTRY = WrapperRecord[]
const _WRAPPER_KEYS =
    Set{Tuple{Symbol, Symbol, Tuple{Vararg{Symbol}}, Symbol,
              Union{String, Nothing}}}()

"""
    register_wrapper!(family, op, mods, tier[, intrinsic])

Record one wrapper registration in `WRAPPER_REGISTRY`. Idempotent: a key
already recorded is skipped, so registration helpers may be re-invoked
(coverage testsets do) without inflating the inventories the count pins
protect. `tier` must be `:intrinsic`, `:asm`, `:core_ir`, or `:missing`;
`intrinsic` is required for `:intrinsic`/`:missing` records and must be
absent otherwise.
"""
function register_wrapper!(family::Symbol, op::Symbol,
                           mods::Tuple{Vararg{Symbol}}, tier::Symbol,
                           intrinsic::Union{String, Nothing} = nothing)
    tier in (:intrinsic, :asm, :core_ir, :missing) ||
        error("register_wrapper!: unknown tier $tier")
    (tier in (:intrinsic, :missing)) == (intrinsic isa String) ||
        error("register_wrapper!: tier $tier " *
              (intrinsic === nothing ? "requires" : "does not take") *
              " an intrinsic name")
    key = (family, op, mods, tier, intrinsic)
    key in _WRAPPER_KEYS && return nothing
    push!(_WRAPPER_KEYS, key)
    push!(WRAPPER_REGISTRY, WrapperRecord(family, op, mods, tier, intrinsic))
    nothing
end

"""
    wrapper_intrinsic_call(family, op, mods, name; missing_ok = false)

The shared tier-2 shell of every generated wrapper family: verify `name`
against the pinned NVVM registry, record the registration, and return the
`NVVM.IntrinsicCall` singleton for the method body. Generated names bypass
the `nvvm""` macro's expansion-time existence check; this keeps its
can't-survive-load property. With `missing_ok = true` an unregistered name
is recorded as a `:missing` record and `nothing` is returned — the caller
skips the method and the conformance counts fail loudly (used by families
whose forms have no asm fallback).
"""
function wrapper_intrinsic_call(family::Symbol, op::Symbol,
                                mods::Tuple{Vararg{Symbol}}, name::String;
                                missing_ok::Bool = false)
    if !NVVM.isintrinsic(name)
        missing_ok ||
            error("$family generator built an unregistered intrinsic name: $name")
        register_wrapper!(family, op, mods, :missing, name)
        return nothing
    end
    register_wrapper!(family, op, mods, :intrinsic, name)
    contract = form_contract(op, mods)
    contract === nothing &&
        error("$family generator built $name for $op.$(join(mods, '.')) " *
              "but the form registry has no contract to ceil it with")
    NVVM.IntrinsicCall{Symbol(name), _ceiling(contract)}()
end

"""
    ceiled(call::NVVM.IntrinsicCall, op_singleton::Operation) -> NVVM.IntrinsicCall

Thread the reviewed form contract for the PTX spelling onto a bare
`nvvm"..."` singleton, so the effect ceiling is checked at every use. The
spelling is passed as a `ptx"..."` singleton — the (op, mods) the wrapper
is lexically implementing — and resolved to its contract at generation
time. Every `nvvm""` literal in `src/` must flow through this or
`wrapper_intrinsic_call` (enforced by test/host/effect_ceiling.jl).
"""
@generated function ceiled(::NVVM.IntrinsicCall{name, nothing},
                           ::Operation{op, mods}) where {name, op, mods}
    contract = form_contract(op, mods)
    contract === nothing &&
        error("ceiled(nvvm\"$name\", ptx\"$op.$(join(mods, '.'))\"): the " *
              "form registry has no contract for this spelling")
    NVVM.IntrinsicCall{name, _ceiling(contract)}()
end

"""
    wrapper_records(families...) -> Vector{WrapperRecord}

All registry records for the given families, in registration order.
"""
wrapper_records(families::Symbol...) =
    WrapperRecord[r for r in WRAPPER_REGISTRY if r.family in families]

"""
    wrapper_intrinsic_names(families...) -> Vector{String}

The distinct tier-2 intrinsic names the given families stand on, in
registration order (the conformance replays pin these sets and counts).
"""
wrapper_intrinsic_names(families::Symbol...) =
    unique!(String[r.intrinsic for r in WRAPPER_REGISTRY
                   if r.family in families && r.tier === :intrinsic])

"""
    wrapper_asm_forms(families...) -> Vector{Tuple{Vararg{Symbol}}}

The mods tuples of the given families' asm-tier registrations, in
registration order (the forms with no intrinsic at the pinned backend).
"""
wrapper_asm_forms(families::Symbol...) =
    Tuple{Vararg{Symbol}}[r.mods for r in WRAPPER_REGISTRY
                          if r.family in families && r.tier === :asm]

"""
    wrapper_missing_intrinsics(families...) -> Vector{String}

Intrinsic names a generator expected but the pinned NVVM registry lacks
(no method was defined). Non-empty means a form silently lost its lowering
— the conformance suite asserts emptiness per family.
"""
wrapper_missing_intrinsics(families::Symbol...) =
    unique!(String[r.intrinsic for r in WRAPPER_REGISTRY
                   if r.family in families && r.tier === :missing])
