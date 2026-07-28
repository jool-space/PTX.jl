# Pointers always use `l` regardless of address space — NVPTX represents
# non-zero AS pointers as 64-bit at the LLVM IR level even when the underlying
# PTX address is 32-bit.
constraint_letter(::Type{Float64}) = "d"
constraint_letter(::Type{Float32}) = "f"
constraint_letter(::Type{Float16}) = "h"
# LLVM keeps these operands as i8, but NVPTX's `h` constraint selects a
# 16-bit PTX register and the backend legalizes the narrow SSA value into it.
# The optimized-LLVM + ptxas tripwire lives in cvt_immediate_carriers.jl.
constraint_letter(::Type{Int8})    = "h"
constraint_letter(::Type{UInt8})   = "h"
constraint_letter(::Type{Int16})   = "h"
constraint_letter(::Type{UInt16})  = "h"
constraint_letter(::Type{Int32})   = "r"
constraint_letter(::Type{UInt32})  = "r"
constraint_letter(::Type{Int64})   = "l"
constraint_letter(::Type{UInt64})  = "l"
# LLVM NVPTX's `b` constraint selects the predicate register class for an
# LLVM `i1` operand/result; this is target-specific LLVM syntax, not CUDA C++
# inline-asm syntax.
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

# Ordinary `cvt` grammar is `cvt.<modifiers...>.<dst>.<src>` — destination is
# mods[end-1]. `cvt.pack` is instead covered by the fixed-u32 scalar-result
# ledger before this fallback.
# Sink forms whose dtype tail names an *operand* (st, red, nanosleep, ...) are
# gated by the form registry's `returns` flag (src/ledgers/forms.jl) — without that
# gate the chain would reserve $0 for a phantom output and ptxas would reject
# with "Arguments mismatch".
function _ordinary_cvt_result_abi_error(mods::Tuple{Vararg{Symbol}})
    length(mods) >= 2 && haskey(DTYPE_RETTYPE, mods[end - 1]) &&
        haskey(DTYPE_RETTYPE, mods[end]) && return nothing
    spelling = isempty(mods) ? "cvt" : "cvt." * join(mods, ".")
    ArgumentError(
        "ptx\"$spelling\" does not use the supported canonical ordinary " *
        "cvt.<modifiers...>.<dst>.<src> order with two known terminal " *
        "dtype tokens. Reversed/postfix spellings from contradictory ISA " *
        "examples remain unsupported because terminal inference would " *
        "assign the wrong result ABI; raw cannot supply an explicit " *
        "result ABI.")
end

function ordinary_cvt_result_type(mods::Tuple{Vararg{Symbol}})
    err = _ordinary_cvt_result_abi_error(mods)
    err === nothing || throw(err)
    DTYPE_RETTYPE[mods[end - 1]]
end

# --- Per-ledger result-ABI dispatch families ---------------------------------
#
# `ledger_result_abi_error(l, op, mods)` returns `missing` when the ledger does
# not gate the generic scalar result ABI here, `nothing` when the ledger owns
# the spelling and its ABI is sound, or the ArgumentError to raise.
# `ledger_rettype(l, op, mods)` returns `missing` or the Julia result type.
#
# The immediate, CLC, and b128 islands are deliberately transparent: their
# calls are dispatched to dedicated builders/adapters before the generic tail
# ever asks for a scalar rettype, and their sink/128-bit results never came
# from DTYPE_RETTYPE. Consulting them here would change what
# `infer_rettype(:setmaxnreg, ...)`-class queries observe.

function ledger_result_abi_error(l::FormLedger, op::Symbol,
                                 mods::Tuple{Vararg{Symbol}})
    schema(l, op, mods) === nothing ? miss(l, op, mods) : nothing
end
ledger_result_abi_error(::ImmediateLedger, op::Symbol,
                        mods::Tuple{Vararg{Symbol}}) = missing
ledger_result_abi_error(::CLCLedger, op::Symbol,
                        mods::Tuple{Vararg{Symbol}}) = missing
ledger_result_abi_error(::B128Ledger, op::Symbol,
                        mods::Tuple{Vararg{Symbol}}) = missing
# The ordinary-cvt fallback is consulted explicitly by _result_abi_error at
# its historical position (after the registry's `returns` gate), not through
# the island partition.
ledger_result_abi_error(::CvtLedger, op::Symbol,
                        mods::Tuple{Vararg{Symbol}}) =
    _ordinary_cvt_result_abi_error(mods)

# The result-bearing islands all answer through the uniform `result_type`
# schema accessor (protocol.jl); only the transparency stubs above and the
# explicitly-consulted cvt fallback need their own methods.
function ledger_rettype(l::FormLedger, op::Symbol,
                        mods::Tuple{Vararg{Symbol}})
    s = schema(l, op, mods)
    s === nothing ? missing : result_type(s)
end
ledger_rettype(::ImmediateLedger, op::Symbol,
               mods::Tuple{Vararg{Symbol}}) = missing
ledger_rettype(::CLCLedger, op::Symbol,
               mods::Tuple{Vararg{Symbol}}) = missing
ledger_rettype(::B128Ledger, op::Symbol,
               mods::Tuple{Vararg{Symbol}}) = missing
ledger_rettype(::CvtLedger, op::Symbol, mods::Tuple{Vararg{Symbol}}) =
    ordinary_cvt_result_type(mods)

function _result_abi_error(op::Symbol, mods::Tuple{Vararg{Symbol}})
    l = island_of(op, mods)
    if l !== nothing
        r = ledger_result_abi_error(l, op, mods)
        r === missing || return r
    end
    c = form_contract(op, mods)
    c !== nothing && !c.returns && return nothing
    op === :cvt && return ledger_result_abi_error(CvtLedger(), op, mods)
    rettype = isempty(mods) ? Nothing : get(DTYPE_RETTYPE, last(mods), Nothing)
    if rettype === Nothing && c !== nothing && c.pure && c.returns
        spelling = isempty(mods) ? string(op) : string(op, ".", join(mods, "."))
        return ArgumentError(
            "ptx\"$spelling\" is in the reviewed pure-form registry but " *
            "does not expose a known scalar result ABI. A pure PTX value " *
            "instruction cannot be emitted as void; check modifier order and " *
            "spelling or add an audited result schema. The raw tier cannot " *
            "supply an explicit result ABI.")
    end
    nothing
end

function infer_rettype(op::Symbol, mods::Tuple{Vararg{Symbol}})
    err = _result_abi_error(op, mods)
    err === nothing || throw(err)
    l = island_of(op, mods)
    if l !== nothing
        t = ledger_rettype(l, op, mods)
        t === missing || return t
    end
    c = form_contract(op, mods)
    c !== nothing && !c.returns && return Nothing
    op === :cvt && return ledger_rettype(CvtLedger(), op, mods)
    isempty(mods) ? Nothing : get(DTYPE_RETTYPE, last(mods), Nothing)
end
