# --- convergent inline asm ---------------------------------------------------
#
# `@asmcall` cannot attach call-site attributes, and `sideeffect` alone does
# NOT forbid duplicating a call site across a divergent branch (jump
# threading, tail duplication) — only `convergent` does. For warp-/warpgroup-
# collective instructions (wgmma.mma_async and mma.sync fallbacks), a split
# call site means different lanes execute different copies: the `active_mask`
# class of miscompile reproduced on hardware. Mbarrier asm uses this path too,
# matching the convergence contract on the complete llvm.nvvm.mbarrier.*
# surface regardless of dispatch tier. The attribute binds in the in-process
# middle end only — llc neither checks nor needs it — so tests assert emitted
# llvmcall IR rather than ptxas acceptance.
#
# Mechanism validated by spikes/raw_asm_attrs.jl (removed in ccdfb8a; view
# with `git show ccdfb8a~1:spikes/raw_asm_attrs.jl`): a `convergent` attribute
# group on an inline-asm call site parses through Base.llvmcall and survives
# the optimized module. This helper builds the same shape `@asmcall` would —
# asm callee returns a scalar or literal struct, entry returns Julia's
# homogeneous-tuple `[N x T]` via extract/insertvalue, Bool passes as i8 —
# plus `#0 = { convergent nomerge nounwind }` on the call.
#
# Keep the complete implementation above `_chain_call_expr`: Julia 1.12
# requires every global called by a generated-function generator to exist in
# the generator's definition world, not merely by the time it is invoked.

_asm_lltype(T::Type) =
    T === Float32 ? "float" :
    T === UInt32  ? "i32"   :
    T === Int32   ? "i32"   :
    T === UInt64  ? "i64"   :
    T === Int64   ? "i64"   :
    T === Float64 ? "double" :
    T === Float16 ? "half"  :
    T === UInt16  ? "i16"   :
    T === Int16   ? "i16"   :
    T === UInt8   ? "i8"    :
    T === Int8    ? "i8"    :
    T === Bool    ? "i8"    :
    # Pointers pass straight into the asm operand (what @asmcall does via
    # the builder API); `i8` pointee to match Julia's LLVMPtr lowering,
    # typed spelling for the Julia ≤ 1.11 device context (NVVM.llvmtype).
    T <: Core.LLVMPtr ? (T.parameters[2] == 0 ? "i8*" :
                         "i8 addrspace($(T.parameters[2]))*") :
    error("convergent_asm_ir: no LLVM mapping for $T")

function convergent_asm_ir(asm::String, constraints::String,
                           rettype::Type, argtypes)::String
    params = ["$(_asm_lltype(T)) %a$(k - 1)" for (k, T) in enumerate(argtypes)]
    callargs = join(("$(_asm_lltype(T)) %a$(k - 1)"
                     for (k, T) in enumerate(argtypes)), ", ")
    asmcall(ret) = "call $ret asm sideeffect \"$asm\", \"$constraints\"($callargs) #0"
    # `nomerge` alongside `convergent`: LLVM ≤ 16 (Julia ≤ 1.11) hoists
    # identical convergent calls from both arms of a divergent branch into
    # one site — the collective-op miscompile. See NVVM.fnattrs.

    body = String[]
    if rettype === Nothing
        entryret = "void"
        push!(body, "  " * asmcall("void"))
        push!(body, "  ret void")
    elseif rettype <: Tuple
        comps = _asm_lltype.(collect(rettype.parameters))
        # asm returns a scalar (1 output) or a literal struct (N>=2 outputs);
        # Julia represents a homogeneous NTuple as [N x T] and a heterogeneous
        # Tuple as a literal struct. The latter matters for exact-raw mbarrier
        # report forms: (Bool, Bool, UInt16) under RAW_CONTRACT remains
        # convergent without corrupting its return ABI.
        callret = length(comps) == 1 ? comps[1] : "{ " * join(comps, ", ") * " }"
        homogeneous = all(==(first(comps)), comps)
        entryret = homogeneous ? "[$(length(comps)) x $(comps[1])]" :
                   "{ " * join(comps, ", ") * " }"
        push!(body, "  %r = " * asmcall(callret))
        prev = "undef"
        for k in 1:length(comps)
            v = length(comps) == 1 ? "%r" : "%e$k"
            length(comps) == 1 ||
                push!(body, "  %e$k = extractvalue $callret %r, $(k - 1)")
            push!(body, "  %t$k = insertvalue $entryret $prev, $(comps[k]) $v, $(k - 1)")
            prev = "%t$k"
        end
        push!(body, "  ret $entryret $prev")
    else
        entryret = _asm_lltype(rettype)
        push!(body, "  %r = " * asmcall(entryret))
        push!(body, "  ret $entryret %r")
    end

    """
    define $entryret @entry($(join(params, ", "))) #1 {
    $(join(body, "\n"))
    }
    attributes #0 = { convergent nomerge nounwind }
    attributes #1 = { alwaysinline }
    """
end
