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

# `cvt` grammar is `cvt.<modifiers...>.<dst>.<src>` — destination is mods[end-1].
# Sink forms whose dtype tail names an *operand* (st, red, nanosleep, ...) are
# gated by the form registry's `returns` flag (src/forms.jl) — without that
# gate the chain would reserve $0 for a phantom output and ptxas would reject
# with "Arguments mismatch".
function infer_rettype(op::Symbol, mods::Tuple{Vararg{Symbol}})
    op in PRED_RESULT_OPCODES && return Bool
    c = form_contract(op, mods)
    c !== nothing && !c.returns && return Nothing
    if op === :cvt && length(mods) >= 2
        rettype = get(DTYPE_RETTYPE, mods[end - 1], nothing)
        rettype === nothing || return rettype
    end
    isempty(mods) ? Nothing : get(DTYPE_RETTYPE, last(mods), Nothing)
end
