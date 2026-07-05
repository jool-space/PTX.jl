# Pointers always use `l` regardless of address space — NVPTX represents
# non-zero AS pointers as 64-bit at the LLVM IR level even when the underlying
# PTX address is 32-bit.
constraint_letter(::Type{Float64}) = "d"
constraint_letter(::Type{Float32}) = "f"
constraint_letter(::Type{Float16}) = "h"
constraint_letter(::Type{Int8})    = "h"   # NVPTX has no native i8 reg; use i16
constraint_letter(::Type{UInt8})   = "h"
constraint_letter(::Type{Int16})   = "h"
constraint_letter(::Type{UInt16})  = "h"
constraint_letter(::Type{Int32})   = "r"
constraint_letter(::Type{UInt32})  = "r"
constraint_letter(::Type{Int64})   = "l"
constraint_letter(::Type{UInt64})  = "l"
constraint_letter(::Type{Bool})    = "b"
constraint_letter(::Type{<:Core.LLVMPtr}) = "l"

const DTYPE_RETTYPE = Dict{Symbol, Type}(
    :f64 => Float64, :f32 => Float32, :f16 => Float16,
    :bf16 => UInt16, :tf32 => UInt32,
    :u64 => UInt64,  :u32 => UInt32,  :u16 => UInt16, :u8 => UInt8,
    :s64 => Int64,   :s32 => Int32,   :s16 => Int16,  :s8 => Int8,
    :b64 => UInt64,  :b32 => UInt32,  :b16 => UInt16, :b8 => UInt8,
    :pred => Bool,
    :f32x2  => UInt64,
    :f16x2  => UInt32, :bf16x2 => UInt32,
    :e4m3x2 => UInt16, :e5m2x2 => UInt16,
    :e4m3x4 => UInt32, :e5m2x4 => UInt32,
    :e2m1x2 => UInt16, :e2m1x4 => UInt16,
    :e2m3x2 => UInt16, :e3m2x2 => UInt16,
    :e2m3x4 => UInt32, :e3m2x4 => UInt32,
    :ue8m0x2 => UInt16,
    :s2f6x2  => UInt16,
)

# `setp.eq.s32` returns Bool, not Int32 — trailing dtype is the input compare type.
const PRED_RESULT_OPCODES = Set{Symbol}((:setp,))

# Chains whose terminal modifier looks like a dtype but describes an *operand*,
# not a return value. Without this gate `infer_rettype` would emit a leading
# output reg and ptxas would reject with "Arguments mismatch".
const NO_RETURN_PREFIXES = Set{Tuple{Vararg{Symbol}}}((
    (:setmaxnreg,),
    (:tensormap,),
    (:tcgen05, :alloc),
    (:tcgen05, :commit),
    (:tcgen05, :relinquish_alloc_permit),
    # `st.<space>.<dtype>` and `red.<space>.<op>.<dtype>` end in a dtype-suffix
    # but describe the *value being written*, not a return — without this gate
    # the chain reserves $0 for a phantom output and ptxas rejects with
    # "Arguments mismatch for instruction 'st'".
    (:st,),
    (:red,),
    # `cp.async.mbarrier.arrive{.noinc}.shared.b64` — `.b64` is the width
    # of the mbarrier address, not a return type. The chain would (wrongly)
    # reserve $0 as a UInt64 output.
    (:cp, :async, :mbarrier, :arrive),
    # `nanosleep.u32 t;` — `.u32` is the width of the duration operand.
    (:nanosleep,),
    # `multimem.st` / `multimem.red` write memory; the trailing dtype is the
    # value written. `multimem.ld_reduce` DOES return and stays on the
    # trailing-dtype rule.
    (:multimem, :st),
    (:multimem, :red),
))

# Prefixes are stored as `(opcode, mod1, mod2, ...)` for compactness.
function _has_no_return_prefix(op::Symbol, mods::Tuple{Vararg{Symbol}})
    for prefix in NO_RETURN_PREFIXES
        op === prefix[1] || continue
        nrest = length(prefix) - 1
        length(mods) < nrest && continue
        match = true
        for i in 1:nrest
            mods[i] === prefix[i + 1] || (match = false; break)
        end
        match && return true
    end
    return false
end

# `cvt` grammar is `cvt.<modifiers...>.<dst>.<src>` — destination is mods[end-1].
function infer_rettype(op::Symbol, mods::Tuple{Vararg{Symbol}})
    op in PRED_RESULT_OPCODES && return Bool
    _has_no_return_prefix(op, mods) && return Nothing
    if op === :cvt && length(mods) >= 2
        rettype = get(DTYPE_RETTYPE, mods[end - 1], nothing)
        rettype === nothing || return rettype
    end
    isempty(mods) ? Nothing : get(DTYPE_RETTYPE, last(mods), Nothing)
end
