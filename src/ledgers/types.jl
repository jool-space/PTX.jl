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

function _result_abi_error(op::Symbol, mods::Tuple{Vararg{Symbol}})
    if requires_mbarrier_schema(op)
        schema = mbarrier_form_schema(op, mods)
        return schema === nothing ? mbarrier_schema_miss(mods) : nothing
    end
    structured = structured_result_schema(op, mods)
    if requires_structured_result_schema(op)
        return structured === nothing ? structured_result_schema_miss(op, mods) : nothing
    end
    vector = vector_result_schema(op, mods)
    vector === nothing || return nothing
    requires_vector_result_schema(op, mods) &&
        return vector_result_schema_miss(op, mods)
    schema = scalar_result_schema(op, mods)
    schema === nothing || return nothing
    requires_scalar_result_schema(op, mods) &&
        return scalar_result_schema_miss(op, mods)
    c = form_contract(op, mods)
    c !== nothing && !c.returns && return nothing
    op === :cvt && return _ordinary_cvt_result_abi_error(mods)
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
    mbarrier = mbarrier_form_schema(op, mods)
    mbarrier === nothing || return _mbarrier_rettype(mbarrier)
    structured = structured_result_schema(op, mods)
    structured === nothing || return structured_result_type(structured)
    vector = vector_result_schema(op, mods)
    vector === nothing || return vector_result_type(vector)
    schema = scalar_result_schema(op, mods)
    schema === nothing || return schema.rettype
    c = form_contract(op, mods)
    c !== nothing && !c.returns && return Nothing
    op === :cvt && return ordinary_cvt_result_type(mods)
    rettype = isempty(mods) ? Nothing : get(DTYPE_RETTYPE, last(mods), Nothing)
    rettype
end
