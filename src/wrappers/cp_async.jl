# Hand-written: shared destination needs the `r` (i32) constraint to trigger
# NVPTX address-space lowering to a 32-bit register; the chain default `l`
# would emit a u64 address that ptxas rejects. Size N is baked into the asm.

function cp_async_spec(qualifier::Symbol, n::Integer)
    if qualifier === :ca
        n in (4, 8, 16) ||
            error("cp.async.ca.shared.global size must be 4, 8, or 16; got: $n")
    elseif qualifier === :cg
        n == 16 ||
            error("cp.async.cg.shared.global requires size 16, got: $n")
    else
        error("cp_async_spec: unknown qualifier $qualifier (expected :ca or :cg)")
    end
    (; asm = "cp.async.$qualifier.shared.global [\$0], [\$1], $n;",
       constraints = "r,l,~{memory}",
       rettype = Nothing)
end

@generated function (::Operation{:cp, (:async, :ca, :shared, :global)})(
        dst::Core.LLVMPtr{T, AS.Shared},
        src::Core.LLVMPtr{S, AS.Global},
        ::Val{N}) where {T, S, N}
    spec = try
        cp_async_spec(:ca, N)
    catch e
        return :(error($(sprint(showerror, e))))
    end
    asm, constraints = spec.asm, spec.constraints
    quote
        Base.@inline
        @asmcall($asm, $constraints, true, Nothing,
                 Tuple{Core.LLVMPtr{$T, AS.Shared}, Core.LLVMPtr{$S, AS.Global}},
                 dst, src)
        nothing
    end
end

# `.cg` caches at L2 only (not L1). 16-byte size only.
@generated function (::Operation{:cp, (:async, :cg, :shared, :global)})(
        dst::Core.LLVMPtr{T, AS.Shared},
        src::Core.LLVMPtr{S, AS.Global},
        ::Val{N}) where {T, S, N}
    spec = try
        cp_async_spec(:cg, N)
    catch e
        return :(error($(sprint(showerror, e))))
    end
    asm, constraints = spec.asm, spec.constraints
    quote
        Base.@inline
        @asmcall($asm, $constraints, true, Nothing,
                 Tuple{Core.LLVMPtr{$T, AS.Shared}, Core.LLVMPtr{$S, AS.Global}},
                 dst, src)
        nothing
    end
end
