# Warp-cooperative shared→register matrix load (PTX 9.2 §9.7.14.5.15).
# Family generator over shape × count × trans × state-space × element-type.

const _LDMATRIX_OUT_COUNT = Dict(:x1 => 1, :x2 => 2, :x4 => 4)

function _ldmatrix_register(shape::Symbol, count::Symbol, trans::Bool,
                            ss::Symbol, dtype::Symbol)
    n_out = _LDMATRIX_OUT_COUNT[count]

    # `shared::cta` becomes `Symbol("shared::cta")` on the chain;
    # `shared` stays bare.
    ss_part = ss === :shared_cta ? Symbol("shared::cta") : ss
    mods = trans ?
        (:sync, :aligned, shape, count, :trans, ss_part, dtype) :
        (:sync, :aligned, shape, count, ss_part, dtype)

    ss_text = ss === :shared_cta ? "shared::cta" : "shared"
    trans_text = trans ? ".trans" : ""
    head = "ldmatrix.sync.aligned.$shape.$count$trans_text.$ss_text.$dtype"

    out_slots = "{" * join(("\$$i" for i in 0:n_out-1), ", ") * "}"
    addr_slot = "[\$$n_out]"
    asm = "$head $out_slots, $addr_slot;"

    if n_out == 1
        constraints = "=r,r,~{memory}"
        @eval @generated function (::Operation{:ldmatrix, $mods})(
                addr::Core.LLVMPtr{T, AS.Shared}) where T
            quote
                Base.@inline
                @asmcall($($asm), $($constraints), true,
                         UInt32, Tuple{Core.LLVMPtr{$T, AS.Shared}},
                         addr)
            end
        end
    else
        constraints = join(vcat(fill("=r", n_out), ["r", "~{memory}"]), ",")
        @eval @generated function (::Operation{:ldmatrix, $mods})(
                addr::Core.LLVMPtr{T, AS.Shared}) where T
            quote
                Base.@inline
                @asmcall($($asm), $($constraints), true,
                         NTuple{$($n_out), UInt32},
                         Tuple{Core.LLVMPtr{$T, AS.Shared}},
                         addr)
            end
        end
    end
    nothing
end

for count in (:x1, :x2, :x4), trans in (false, true), ss in (:shared, :shared_cta)
    _ldmatrix_register(:m8n8, count, trans, ss, :b16)
end

# Hopper m16n16.b8 — emitted but unreachable on Ada.
for count in (:x1, :x2, :x4), trans in (false, true), ss in (:shared, :shared_cta)
    _ldmatrix_register(:m16n16, count, trans, ss, :b8)
end
