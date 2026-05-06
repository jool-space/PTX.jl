# Warp-cooperative register→shared matrix store (PTX 9.2 §9.7.14.5.16).
# sm_90+ only — ptxas rejects on lower targets. Mirror of `wrappers/ldmatrix.jl`
# with the brace vector on the input side and address as destination.

const _STMATRIX_IN_COUNT = Dict(:x1 => 1, :x2 => 2, :x4 => 4)

function stmatrix_spec(shape::Symbol, count::Symbol, trans::Bool,
                       ss::Symbol, dtype::Symbol)
    n_in = _STMATRIX_IN_COUNT[count]
    ss_text    = ss === :shared_cta ? "shared::cta" : "shared"
    trans_text = trans ? ".trans" : ""
    head = "stmatrix.sync.aligned.$shape.$count$trans_text.$ss_text.$dtype"
    addr_slot = "[\$0]"
    in_slots  = "{" * join(("\$$i" for i in 1:n_in), ", ") * "}"
    asm = "$head $addr_slot, $in_slots;"
    # Shared-AS pointer needs `r` (i32). NVPTX lowers shared-AS ptrs to u32
    # registers; using `l` (i64) emits `%rd` and ptxas rejects.
    constraints = n_in == 1 ?
        "r,r,~{memory}" :
        join(vcat(["r"], fill("r", n_in), ["~{memory}"]), ",")
    (; n_in, asm, constraints)
end

function _stmatrix_register(shape::Symbol, count::Symbol, trans::Bool,
                            ss::Symbol, dtype::Symbol)
    spec = stmatrix_spec(shape, count, trans, ss, dtype)
    n_in, asm, constraints = spec.n_in, spec.asm, spec.constraints

    ss_part = ss === :shared_cta ? Symbol("shared::cta") : ss
    mods = trans ?
        (:sync, :aligned, shape, count, :trans, ss_part, dtype) :
        (:sync, :aligned, shape, count, ss_part, dtype)

    if n_in == 1
        @eval function (::Operation{:stmatrix, $mods})(
                addr::Core.LLVMPtr{T, AS.Shared},
                reg::UInt32) where T
            Base.@inline
            @asmcall($asm, $constraints, true, Nothing,
                     Tuple{Core.LLVMPtr{T, AS.Shared}, UInt32},
                     addr, reg)
            nothing
        end
    else
        flat_argtypes = vcat([:(Core.LLVMPtr{T, AS.Shared})], fill(:UInt32, n_in))
        reg_args = [:(regs[$i]) for i in 1:n_in]
        @eval function (::Operation{:stmatrix, $mods})(
                addr::Core.LLVMPtr{T, AS.Shared},
                regs::NTuple{$n_in, UInt32}) where T
            Base.@inline
            @asmcall($asm, $constraints, true, Nothing,
                     Tuple{$(flat_argtypes...)},
                     addr, $(reg_args...))
            nothing
        end
    end
    nothing
end

for count in (:x1, :x2, :x4), trans in (false, true), ss in (:shared, :shared_cta)
    _stmatrix_register(:m8n8, count, trans, ss, :b16)
end

# Hopper m16n8.b8 — emitted but unreachable on Ada.
for count in (:x1, :x2, :x4), trans in (false, true), ss in (:shared, :shared_cta)
    _stmatrix_register(:m16n8, count, trans, ss, :b8)
end
