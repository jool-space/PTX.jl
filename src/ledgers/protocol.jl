# The result-ABI island partition: one partition function, one lookup.
#
# Every closed grammar island ("ledger") that owns a chain form's result or
# operand ABI implements the same surface on a singleton handle:
#
#   schema(l, op, mods)             — the island's schema object for an exact
#                                     reviewed form, or `nothing`.
#   miss(l, op, mods)               — the ArgumentError for a spelling routed
#                                     to the island with no reviewed schema.
#   validate_ledger_args(l, s, argtypes) — the island's argument/carrier
#                                     validator for a resolved schema `s`
#                                     (throws ArgumentError on a violation).
#   result_type(s)                  — the Julia result type of a resolved
#                                     schema, for the result-bearing islands
#                                     (structured/vector/scalar/mbarrier);
#                                     the sink/128-bit islands stay out of
#                                     scalar inference (see types.jl).
#
# `island_of(op, mods)` is the single routing site: it assigns every spelling
# to at most one island. A routed spelling either resolves to a schema or
# fails loud with that island's `miss`; it never falls through to the generic
# terminal-modifier rule or the raw tier. Consumers never probe islands they
# were not routed to — `build_call` (src/dsl/render.jl), result-ABI inference
# (src/ledgers/types.jl), `lowering` reflection (src/dsl/reflection.jl), and
# the transpiler (src/codegen/instruction.jl + src/codegen/adapters/*.jl)
# all route through `island_of`.

abstract type FormLedger end

struct ImmediateLedger  <: FormLedger end   # src/ledgers/immediate_forms.jl
struct MBarrierLedger   <: FormLedger end   # src/ledgers/mbarrier_forms.jl
struct StructuredLedger <: FormLedger end   # src/ledgers/structured_results.jl
struct CLCLedger        <: FormLedger end   # src/ledgers/address_operands.jl
struct VectorLedger     <: FormLedger end   # src/ledgers/vector_results.jl
struct ScalarLedger     <: FormLedger end   # src/ledgers/scalar_results.jl
struct B128Ledger       <: FormLedger end   # src/ledgers/b128_forms.jl
struct CvtLedger        <: FormLedger end   # src/ledgers/cvt_forms.jl — NOT
                                            # partitioned; see below.

function schema end
function miss end
function validate_ledger_args end
function result_type end

# Vector-result markers live here because the partition keys on them; the
# vector ledger's schema resolver shares this set.
const _VECTOR_MARKERS = Set((:v2, :v4, :v8))

# THE partition. It replaces the historical consultation cascade (immediate →
# mbarrier → structured → CLC → vector → scalar → b128): the islands' claim
# sets were opcode-disjoint, so at most one island ever claimed a spelling and
# the cascade order decided nothing — except at two spots, which are now
# explicit rules instead of consult-order accidents:
#
#   * a vector-marked `ld`/`atom` spelling belongs to the vector island even
#     when a `.b128` token is present (`ld.v2.b128`-class spellings get the
#     vector island's miss, exactly as the old vector-before-b128 consult
#     order chose);
#   * a `clusterlaunchcontrol` spelling with `.try_cancel` belongs to the CLC
#     island even if `.query_cancel`/`.b128` also appear (the old CLC-before-
#     b128 order).
#
# The scalar arm reproduces the scalar ledger's historical claim predicate
# verbatim; see src/ledgers/scalar_results.jl for the per-pattern rationale.
#
# `CvtLedger` is deliberately NOT part of the partition: it gates ordinary cvt
# *source carriers* for the transpiler (cvt_forms.jl) and the ordinary-cvt
# *result* fallback in types.jl — not call routing. It is consulted explicitly
# where `op === :cvt` reaches those two consumers, after the partition
# (cvt.pack belongs to the scalar island) and after the form registry's
# `returns` gate. Unlike the islands, its `schema` throws its own `miss` for
# an invalid spelling: every non-pack cvt spelling is inside its domain, so a
# lookup miss is never a fall-through.
function island_of(op::Symbol, mods::Tuple{Vararg{Symbol}})
    if op === :setmaxnreg || op === :pmevent
        ImmediateLedger()
    elseif op === :mbarrier
        MBarrierLedger()
    elseif op === :setp || op === :lop3 || op === :match || op === :elect ||
           op === :testp || op === :isspacep
        StructuredLedger()
    elseif op === :clusterlaunchcontrol
        :try_cancel in mods                  ? CLCLedger()  :
        :query_cancel in mods && :b128 in mods ? B128Ledger() : nothing
    elseif op === :ld || op === :atom
        any(m -> m in _VECTOR_MARKERS, mods) ? VectorLedger() :
        :b128 in mods                        ? B128Ledger()   : nothing
    elseif op === :multimem
        :ld_reduce in mods && any(m -> m in _VECTOR_MARKERS, mods) ?
            VectorLedger() : nothing
    elseif (op === :mov || op === :ldu || op === :st) && :b128 in mods
        B128Ledger()
    elseif op === :popc || op === :clz || op === :dp2a || op === :dp4a
        ScalarLedger()
    elseif (op === :mul || op === :mad) && :wide in mods
        ScalarLedger()
    elseif op === :cvt && :pack in mods
        ScalarLedger()
    elseif op === :prmt && mods != (:b32,)
        # Base `prmt.b32` is terminal-inference-safe; every mode suffix moves
        # the terminal token off the b32 result, so only that exact base
        # spelling bypasses the closed six-mode island.
        ScalarLedger()
    elseif (op === :add || op === :sub || op === :neg ||
            op === :min || op === :max) &&
           any(t -> t === :u16x2 || t === :s16x2 ||
                    t === :u8x4  || t === :s8x4, mods)
        ScalarLedger()
    elseif (op === :min || op === :max) && :relu in mods
        ScalarLedger()
    elseif (op === :add || op === :sub || op === :fma) && :f32 in mods &&
           any(t -> t === :f16 || t === :bf16, mods)
        # Any permutation containing both the fixed f32 result token and a
        # narrow multiplicand token, so malformed modifier orders cannot fall
        # back to terminal inference.
        ScalarLedger()
    else
        nothing
    end
end

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
