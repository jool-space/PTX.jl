# `tensormap.query.alloc_class_large{.ss}.b1024 p, [addr];` — PTX 9.5
# §9.7.10.30. Sets `p` when the tensor-map object at `addr` has its
# allocation class set to large (by `tensormap.replace` with
# `.alloc_class`). Supported on sm_90a and the sm_100f/sm_110f/sm_120f
# families.
#
# Typed-wrapper-only (TYPED_WRAPPER_ONLY_RULES): the head shares `:tensormap`
# with `tensormap.replace`, whose FORMS contract is a no-result sink, so the
# chain would drop the `.pred` destination.
#
# Wrapped state spaces are the three with an unambiguous pointer carrier:
# generic, `.global`, and `.shared::cta`, all passed as 64-bit `l` like the
# `tensormap.replace` chain. Unlike most shared-window instructions, ptxas
# 13.5 rejects a 32-bit `.shared::cta` address here ("Arguments mismatch for
# instruction 'mov'"). `.const` and `.param{::entry}` are not wrapped: a
# `TMADescriptorPtr` is typed `AS.Const` by convention but holds a generic
# address of global storage (see `reinterpret_addrspace`), so `.const` would
# misread it; reinterpret it to `AS.Generic` and use the generic form.
#
# The query reads the descriptor, so it keeps the memory clobber: it must not
# move above a preceding `tensormap.replace` to the same object.

function _tensormap_query_body(asm::String, constraints::String, T, as::Int)
    quote
        Base.@inline
        @asmcall($asm, $constraints, true, Bool,
                 Tuple{Core.LLVMPtr{$T, $as}}, addr)
    end
end

for (ss, as, constraint) in (((),                       AS.Generic, "l"),
                             ((:global,),               AS.Global,  "l"),
                             ((Symbol("shared::cta"),), AS.Shared,  "l"))
    mods = (:query, :alloc_class_large, ss..., :b1024)
    asm = "tensormap.$(join(mods, '.')) \$0, [\$1];"
    constraints = "=b,$(constraint),~{memory}"
    @eval @generated function (::Operation{:tensormap, $mods})(
            addr::Core.LLVMPtr{T, $as}) where {T}
        _tensormap_query_body($asm, $constraints, T, $as)
    end
end
