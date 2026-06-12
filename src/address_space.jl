module AS

# Match CUDACore.AS.* so `LLVMPtr{T, PTX.AS.Global}` and
# `LLVMPtr{T, CUDACore.AS.Global}` are the same type.
const Generic = 0
const Global  = 1
const Shared  = 3
const Const   = 4
const Local   = 5
const Param   = 101

# LLVM NVPTX addrspace(7) = `shared::cluster` (distributed shared memory).
# Not a CUDACore space; tier-2 intrinsics with cluster-window operands
# (cp.async.bulk.tensor g2s, cluster mbarrier sinks) take pointers here.
const SharedCluster = 7

end # module AS

"""
    reinterpret_addrspace(Val(AS′), p::Core.LLVMPtr{T,AS}) -> Core.LLVMPtr{T,AS′}

Reinterpret a pointer's raw 64-bit value in another address space — a
`ptrtoint`/`inttoptr` pair, deliberately NOT an `addrspacecast`. NVPTX
lowers addrspacecast to `cvta` window translation, which is wrong for the
two retypes the tier-2 wrappers need:

- `Shared → SharedCluster`: the ISA defines `shared::cta` addresses as
  valid `shared::cluster` addresses (own-CTA window), so the raw value is
  already correct; the cast's `cvta.shared` + `cvta.to.shared::cluster`
  round-trip is two wasted instructions.
- `Const → Generic` for TMA descriptors: the descriptor blob lives in
  *global* memory and is typed `AS.Const` by convention (TMADescriptorPtr),
  so its raw value is already a generic address — `cvta.const` would
  translate it as if it were a const-window offset and corrupt it.
"""
@generated function reinterpret_addrspace(::Val{To},
        p::Core.LLVMPtr{T, From}) where {To, T, From}
    spell(n) = n == 0 ? "ptr" : "ptr addrspace($n)"
    ir = """
        %i = ptrtoint $(spell(From)) %0 to i64
        %q = inttoptr i64 %i to $(spell(To))
        ret $(spell(To)) %q"""
    quote
        Base.@inline
        Base.llvmcall($ir, Core.LLVMPtr{$T, $To},
                      Tuple{Core.LLVMPtr{$T, $From}}, p)
    end
end
