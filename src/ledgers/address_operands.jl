const PTXAddressInteger = Union{Int32, UInt32, Int64, UInt64}

is_ptx_address_integer_type(@nospecialize(T::Type)) =
    T <: PTXAddressInteger && T !== Union{}

"""
    Address{T}

An isbits role marker for a 32- or 64-bit integer that represents a bracketed
PTX address operand. `Address` is not a pointer conversion and is removed
before inline-assembly argument passing. Construct it through [`address`](@ref).

`Core.LLVMPtr` values are deliberately not wrapped: their pointer type already
preserves the address role, address space, and exact typed-wrapper dispatch.
"""
struct Address{T}
    value::T
    # Carrier validation is by dispatch, not a runtime type test: under the
    # GPU compiler's overlay method table on Julia 1.10, `T in (...)` (a
    # `jl_types_equal` ccall) is not concretely evaluated, so the check and
    # its error-string interpolation would survive into kernel IR and fail
    # GPUCompiler's IR validation.
    Address(value::PTXAddressInteger) = new{typeof(value)}(value)
    Address(@nospecialize(value)) = throw(ArgumentError(
        "PTX.address: $(typeof(value)) is not an integer address carrier; " *
        "expected Int32, UInt32, Int64, or UInt64 (Core.LLVMPtr is already " *
        "address-typed and is returned unchanged)"))
end

"""
    address(value)

Preserve a bracketed PTX address role through Julia lowering. A 32- or 64-bit
integer becomes an [`Address`](@ref) marker; an `Address` and a
`Core.LLVMPtr` are returned unchanged. Other carriers are rejected.
"""
@inline address(value::Core.LLVMPtr) = value

@inline address(value::Address) = value
@inline address(value) = Address(value)

has_explicit_address(@nospecialize(argtypes)) =
    any(T -> T <: Address, argtypes)

has_integer_address(@nospecialize(argtypes)) =
    any(T -> T <: Address && is_ptx_address_integer_type(T.parameters[1]),
        argtypes)

has_invalid_address_marker(@nospecialize(argtypes)) =
    any(T -> T <: Address && !is_ptx_address_integer_type(T.parameters[1]),
        argtypes)

# An integer Address must not make a structured typed form fall through to a
# scalar chain whose result/operand convention cannot describe it. These
# reviewed families have grouped results/sources, unsupported b128 register
# carriers, or composite address operands. Exact methods remain authoritative
# (tcgen05 below); only dispatch misses carrying Address, including raw, fail.
struct StructuredAddressFallbackRule
    op::Symbol
    prefix::Tuple{Vararg{Symbol}}
    marker::Union{Symbol, Nothing}
    detail::String
end

const STRUCTURED_ADDRESS_FALLBACK_RULES = (
    StructuredAddressFallbackRule(:ld, (), :v2,
        "vector loads return a register tuple, not the terminal scalar dtype"),
    StructuredAddressFallbackRule(:ld, (), :v4,
        "vector loads return a register tuple, not the terminal scalar dtype"),
    StructuredAddressFallbackRule(:st, (), :v2,
        "vector stores consume a register tuple and return nothing"),
    StructuredAddressFallbackRule(:st, (), :v4,
        "vector stores consume a register tuple and return nothing"),
    StructuredAddressFallbackRule(:atom, (), :v2,
        "vector atomics have grouped operands/results"),
    StructuredAddressFallbackRule(:atom, (), :v4,
        "vector atomics have grouped operands/results"),
    StructuredAddressFallbackRule(:atom, (), :v8,
        "vector atomics have grouped operands/results"),
    StructuredAddressFallbackRule(:red, (), :v2,
        "vector reductions consume a grouped source and return nothing"),
    StructuredAddressFallbackRule(:red, (), :v4,
        "vector reductions consume a grouped source and return nothing"),
    StructuredAddressFallbackRule(:red, (), :v8,
        "vector reductions consume a grouped source and return nothing"),
    StructuredAddressFallbackRule(:multimem, (), :v2,
        "multimem vector forms have grouped operands/results"),
    StructuredAddressFallbackRule(:multimem, (), :v4,
        "multimem vector forms have grouped operands/results"),
    StructuredAddressFallbackRule(:multimem, (), :v8,
        "multimem vector forms have grouped operands/results"),
    StructuredAddressFallbackRule(:ld, (), :b128,
        "a PTX b128 register has no generic scalar Julia result carrier"),
    StructuredAddressFallbackRule(:st, (), :b128,
        "a PTX b128 register has no generic scalar Julia source carrier"),
    StructuredAddressFallbackRule(:atom, (), :b128,
        "a PTX b128 register has no generic scalar Julia carrier"),
    StructuredAddressFallbackRule(:red, (), :b128,
        "a PTX b128 register has no generic scalar Julia carrier"),
    StructuredAddressFallbackRule(:multimem, (), :b128,
        "a PTX b128 register has no generic scalar Julia carrier"),
    StructuredAddressFallbackRule(:ldmatrix, (), nothing,
        "ldmatrix has shape-dependent grouped results"),
    StructuredAddressFallbackRule(:stmatrix, (), nothing,
        "stmatrix has shape-dependent grouped sources and no result"),
    StructuredAddressFallbackRule(:cp, (:async, :bulk, :tensor), nothing,
        "tensor bulk copies have composite tensor-map addresses and coordinates"),
    StructuredAddressFallbackRule(:cp, (:async, :bulk, :prefetch, :tensor), nothing,
        "tensor prefetch has a composite tensor-map address and coordinates"),
    StructuredAddressFallbackRule(:cp, (:reduce, :async, :bulk, :tensor), nothing,
        "tensor bulk reductions have composite addresses and coordinates"),
    StructuredAddressFallbackRule(:tcgen05, (), nothing,
        "tcgen05 has instruction-specific address, vector, sink, and descriptor schemas"),
    StructuredAddressFallbackRule(:fabric, (), nothing,
        "fabric operations use multi-register CFT handles inside one address operand"),
)

function _address_mods_start_with(mods::Tuple{Vararg{Symbol}}, prefix::Tuple)
    length(prefix) <= length(mods) || return false
    all(i -> mods[i] === prefix[i], eachindex(prefix))
end

function structured_address_fallback_rule(op::Symbol,
                                          mods::Tuple{Vararg{Symbol}})
    for rule in STRUCTURED_ADDRESS_FALLBACK_RULES
        rule.op === op || continue
        _address_mods_start_with(mods, rule.prefix) || continue
        rule.marker === nothing || rule.marker in mods || continue
        return rule
    end
    nothing
end

function structured_address_fallback_error(op::Symbol,
                                           mods::Tuple{Vararg{Symbol}}, rule)
    spelling = isempty(mods) ? string(op) : string(op, ".", join(mods, "."))
    ArgumentError(
        "ptx\"$spelling\" requires an exact typed wrapper because " *
        "$(rule.detail). No exact method matched, and the generic/raw scalar chain is " *
        "forbidden rather than guessing a result ABI or operand grouping.")
end

# `clusterlaunchcontrol.try_cancel` has two mandatory address operands and no
# result.  Keep its complete PTX 9.3 grammar island closed: a modifier typo or
# ordering error must not fall through to terminal-result inference or the raw
# tier.  The target fields are evidence metadata; selection remains the
# backend/ptxas responsibility.
struct CLCTryCancelSchema
    shared_cta::Bool
    multicast::Bool
    introduced::VersionNumber
    target_floor::VersionNumber
    multicast_arch_targets::Tuple{Vararg{String}}
    multicast_family_targets::Tuple{Vararg{String}}
    multicast_family_since::Union{VersionNumber, Nothing}
    section::String
end

const CLC_TRY_CANCEL_SECTION =
    "ptx/9-instruction-set/9.7.14.18-parallel-synchronization-and-" *
    "communication-instructions-clusterlaunchcontrol.try_cancel.md, " *
    "heading 9.7.14.18 clusterlaunchcontrol.try_cancel"
const _CLC_COMPLETE_TX = Symbol("mbarrier::complete_tx::bytes")
const _CLC_MULTICAST_ALL = Symbol("multicast::cluster::all")

const CLC_TRY_CANCEL_FORMS = let
    forms = Dict{Tuple{Vararg{Symbol}}, CLCTryCancelSchema}()
    for shared_cta in (false, true), multicast in (false, true)
        mods = Symbol[:try_cancel, :async]
        shared_cta && push!(mods, Symbol("shared::cta"))
        push!(mods, _CLC_COMPLETE_TX)
        multicast && push!(mods, _CLC_MULTICAST_ALL)
        push!(mods, :b128)
        forms[Tuple(mods)] = CLCTryCancelSchema(
            shared_cta, multicast, v"8.6", v"10.0",
            multicast ? ("sm_100a", "sm_110a", "sm_120a") : (),
            multicast ? ("sm_100f", "sm_110f", "sm_120f") : (),
            multicast ? v"8.8" : nothing,
            CLC_TRY_CANCEL_SECTION)
    end
    forms
end

schema(::CLCLedger, op::Symbol, mods::Tuple{Vararg{Symbol}}) =
    op === :clusterlaunchcontrol ? get(CLC_TRY_CANCEL_FORMS, mods, nothing) :
    nothing

claims(::CLCLedger, op::Symbol, mods::Tuple{Vararg{Symbol}}) =
    op === :clusterlaunchcontrol && (:try_cancel in mods)

# `op` is always :clusterlaunchcontrol for a claimed miss; spelling is fixed.
function miss(::CLCLedger, op::Symbol, mods::Tuple{Vararg{Symbol}})
    spelling = isempty(mods) ? "clusterlaunchcontrol" :
               "clusterlaunchcontrol." * join(mods, ".")
    ArgumentError(
        "ptx\"$spelling\" is inside the audited " *
        "clusterlaunchcontrol.try_cancel grammar island but is not one of " *
        "the four PTX 9.3 forms (generic/shared::cta × base/multicast); " *
        "see $CLC_TRY_CANCEL_SECTION. The raw tier cannot bypass an " *
        "instruction-specific operand-role schema.")
end

_address_payload_type(@nospecialize(T::Type)) =
    T <: Address ? T.parameters[1] : T

function _address_arg_error(@nospecialize(T::Type), index::Int,
                            schema::CLCTryCancelSchema)
    (T <: Address || T <: Core.LLVMPtr) || return ArgumentError(
        "clusterlaunchcontrol.try_cancel operand $index is a mandatory " *
        "bracketed PTX address, got $T; wrap an integer carrier with " *
        "PTX.address(...) or pass a pointer in the required address space")

    payload = _address_payload_type(T)
    (is_ptx_address_integer_type(payload) || payload <: Core.LLVMPtr) ||
        return ArgumentError(
        "clusterlaunchcontrol.try_cancel operand $index has unsupported " *
        "address carrier $payload")

    if payload <: Core.LLVMPtr
        as = payload.parameters[2]
        expected = schema.shared_cta ? AS.Shared : AS.Generic
        as == expected || return ArgumentError(
            "clusterlaunchcontrol.try_cancel operand $index uses LLVM " *
            "address space $as, but the " *
            (schema.shared_cta ? ".shared::cta" : "generic-address") *
            " form requires address space $expected")
    end
    nothing
end

function validate_clc_try_cancel_args(schema::CLCTryCancelSchema,
                                      @nospecialize(argtypes))
    length(argtypes) == 2 || throw(ArgumentError(
        "clusterlaunchcontrol.try_cancel has exactly two mandatory address " *
        "operands ([addr], [mbar]); got $(length(argtypes)) operands; see " *
        schema.section))
    for (i, T) in enumerate(argtypes)
        err = _address_arg_error(T, i, schema)
        err === nothing || throw(err)
    end
    nothing
end

validate_ledger_args(::CLCLedger, s::CLCTryCancelSchema,
                     @nospecialize(argtypes)) =
    validate_clc_try_cancel_args(s, argtypes)
