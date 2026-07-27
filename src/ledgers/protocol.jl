# The result-ABI ledger protocol: one protocol, one consultation order.
#
# Every closed grammar island ("ledger") that owns a chain form's result or
# operand ABI implements the same three-method surface on a singleton handle:
#
#   claims(l, op, mods) -> Bool     — does this spelling belong to the ledger's
#                                     closed island? A claimed spelling must
#                                     either resolve to a schema or fail loud;
#                                     it never falls through to the generic
#                                     terminal-modifier rule or the raw tier.
#   schema(l, op, mods)             — the ledger's schema object for an exact
#                                     reviewed form, or `nothing`.
#   miss(l, op, mods)               — the ArgumentError for a claimed-but-
#                                     invalid spelling.
#   validate_ledger_args(l, s, argtypes) — the ledger's argument/carrier
#                                     validator for a resolved schema `s`
#                                     (throws ArgumentError on a violation).
#
# All methods are normalized to `(op, mods)` even where a ledger keys on only
# one of the two (mbarrier ignores `op === :mbarrier`'s mods-independent claim,
# the CLC ledger keys its schema table on mods, ...).
#
# Consumers never hand-write the cascade: `CALL_LEDGERS` below is the single
# ordering site, shared by `build_call` (src/dsl/render.jl), result-ABI
# inference (src/ledgers/types.jl), `lowering` reflection
# (src/dsl/reflection.jl), and the transpiler's instruction emitter
# (src/codegen/instruction.jl + src/codegen/adapters/*.jl).

abstract type FormLedger end

struct ImmediateLedger  <: FormLedger end   # src/ledgers/immediate_forms.jl
struct MBarrierLedger   <: FormLedger end   # src/ledgers/mbarrier_forms.jl
struct StructuredLedger <: FormLedger end   # src/ledgers/structured_results.jl
struct CLCLedger        <: FormLedger end   # src/ledgers/address_operands.jl
struct VectorLedger     <: FormLedger end   # src/ledgers/vector_results.jl
struct ScalarLedger     <: FormLedger end   # src/ledgers/scalar_results.jl
struct B128Ledger       <: FormLedger end   # src/ledgers/b128_forms.jl
struct CvtLedger        <: FormLedger end   # src/ledgers/cvt_forms.jl — NOT in
                                            # CALL_LEDGERS; see below.

function claims end
function schema end
function miss end
function validate_ledger_args end

# THE consultation order. It matches the historical `build_call` cascade
# exactly: immediate → mbarrier → structured → CLC → vector → scalar → b128.
# (The invalid-Address-marker guard historically sat between the structured
# and CLC consults; `build_call` splits its walk there to preserve it.)
#
# `CvtLedger` is deliberately NOT part of this order: it gates ordinary cvt
# *source carriers* for the transpiler (cvt_forms.jl) and the ordinary-cvt
# *result* fallback in types.jl — not the call cascade. It is consulted
# explicitly where `op === :cvt` reaches those two consumers, after every
# CALL_LEDGERS island (cvt.pack belongs to the scalar ledger) and after the
# form registry's `returns` gate. Unlike the other ledgers, its `schema`
# throws its own `miss` for a claimed-but-invalid spelling: every non-pack
# cvt spelling is claimed, so a lookup miss is never a fall-through.
const CALL_LEDGERS = (ImmediateLedger(), MBarrierLedger(), StructuredLedger(),
                      CLCLedger(), VectorLedger(), ScalarLedger(), B128Ledger())

# First ledger in `CALL_LEDGERS` claiming `(op, mods)`, or `nothing`.
function first_claiming(op::Symbol, mods::Tuple{Vararg{Symbol}})
    for l in CALL_LEDGERS
        claims(l, op, mods) && return l
    end
    nothing
end

# Ledgers whose argument validation runs at consult time in `build_call`
# (immediately after their miss check, as the historical cascade did) rather
# than inside a dedicated builder or the generic tail.
prevalidates_call(::FormLedger)      = false
prevalidates_call(::ImmediateLedger) = true
prevalidates_call(::CLCLedger)       = true
prevalidates_call(::B128Ledger)      = true

# --- Shared operand predicates -----------------------------------------------

# A `Val{N}` integer immediate (PTX constants are never Bool predicates).
is_integer_immediate(::Type{Val{V}}) where {V} = V isa Integer && !(V isa Bool)
is_integer_immediate(::Type) = false

# Shared scalar/structured operand-kind acceptance. The modifier still
# determines instruction semantics and each schema still determines Julia
# result signedness; this only answers whether a Julia carrier may occupy a
# reviewed operand slot (PTX §6.1 size compatibility, §5.2.3 alternate-float
# bit carriers). Unknown kinds fail loud: a ledger entry naming a kind this
# table does not know is a ledger bug, not an argument mismatch.
function operand_accepts(kind::Symbol, ::Type{T}) where {T}
    if kind === :pred
        return T === Bool
    elseif kind === :imm8
        return T <: Val && is_integer_immediate(T) &&
               0 <= T.parameters[1] <= 255
    elseif kind === :f16
        # A .b16 register is compatible with .f16 (§6.1); UInt16 is PTX.jl's
        # established bit-size carrier.
        return T === Float16 || T === UInt16
    elseif kind === :bf16
        # PTX §5.2.3 requires a bf16 value to live in a .b16 register; UInt16
        # is PTX.jl's established bit-pattern carrier.
        return T === UInt16
    elseif kind === :f32
        # A .b32 register is compatible with .f32 (§6.1).
        return T === Float32 || T === UInt32
    elseif kind === :f64
        return T === Float64 || T === UInt64
    elseif kind === :u16 || kind === :s16
        return T === UInt16 || T === Int16 || is_integer_immediate(T)
    elseif kind === :u32 || kind === :s32
        # Signed and unsigned integer types of common size are mutually
        # compatible (§6.1).
        return T === UInt32 || T === Int32 || is_integer_immediate(T)
    elseif kind === :u64 || kind === :s64
        return T === UInt64 || T === Int64 || is_integer_immediate(T)
    elseif kind === :b16
        return T === UInt16 || T === Int16 || T === Float16 ||
               is_integer_immediate(T)
    elseif kind === :b32
        # PTX §6.1 makes a bit-size type compatible with every same-size
        # scalar type.  Keep pointer values out of this value-only surface.
        return T === UInt32 || T === Int32 || T === Float32 ||
               is_integer_immediate(T)
    elseif kind === :b64
        return T === UInt64 || T === Int64 || T === Float64 ||
               is_integer_immediate(T)
    elseif kind === :genaddr
        # isspacep's generic-address value: "the source address operand must
        # be of type .u32 or .u64" (§9.7.9.20). Both register widths and an
        # integer immediate assemble under .address_size 64 (CUDA 13 ptxas),
        # so this kind admits either integer width, unlike the fixed-width
        # kinds above. Pointer carriers stay out of this value-only surface;
        # reinterpret an LLVMPtr to UInt64 at the call site.
        return T === UInt32 || T === Int32 || T === UInt64 || T === Int64 ||
               is_integer_immediate(T)
    end
    error("unknown audited operand kind: ", kind)
end

function operand_description(kind::Symbol)
    kind === :pred && return "Bool predicate"
    kind === :imm8 && return "Val{N} with integer N in 0:255"
    kind === :f16  && return "Float16 or UInt16 bit carrier (.f16-compatible)"
    kind === :bf16 && return "UInt16 (.b16 carrier for bf16)"
    kind === :f32  && return "Float32 or UInt32 bit carrier (.f32-compatible)"
    kind === :f64  && return "Float64 or UInt64 bit carrier (.f64-compatible)"
    kind === :u16  && return "a 16-bit integer (.u16-compatible)"
    kind === :s16  && return "a 16-bit integer (.s16-compatible)"
    kind === :u32  && return "a 32-bit integer (.u32-compatible)"
    kind === :s32  && return "a 32-bit integer (.s32-compatible)"
    kind === :u64  && return "a 64-bit integer (.u64-compatible)"
    kind === :s64  && return "a 64-bit integer (.s64-compatible)"
    kind === :b16  && return "a 16-bit scalar (.b16-compatible)"
    kind === :b32  && return "a 32-bit scalar (.b32-compatible)"
    kind === :b64  && return "a 64-bit scalar (.b64-compatible)"
    kind === :genaddr &&
        return "a 32- or 64-bit integer generic-address value (.u32/.u64)"
    string(kind)
end
