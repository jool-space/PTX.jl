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

# LLVM NVPTX addrspace(6) = tensor memory (Blackwell TMEM; 32-bit in the
# backend's datalayout). tcgen05 intrinsics take taddr operands here.
const Tmem = 6

end # module AS

"""
    reinterpret_addrspace(Val(AS′), p::Core.LLVMPtr{T,AS}) -> Core.LLVMPtr{T,AS′}

Reinterpret a pointer's raw 64-bit value in another address space — a
`ptrtoint`/`inttoptr` pair, deliberately NOT an `addrspacecast`. NVPTX
lowers addrspacecast to `cvta` window translation, which is wrong for several
raw retypes that tier-2 wrappers need:

- `Shared → SharedCluster`: the ISA defines `shared::cta` addresses as
  valid `shared::cluster` addresses (own-CTA window), so the raw value is
  already correct; the cast's `cvta.shared` + `cvta.to.shared::cluster`
  round-trip is two wasted instructions.
- `Const → Generic` for TMA descriptors: the descriptor blob lives in
  *global* memory and is typed `AS.Const` by convention (TMADescriptorPtr),
  so its raw value is already a generic address — `cvta.const` would
  translate it as if it were a const-window offset and corrupt it.
- `Global → Generic` for a live tensor-map descriptor: global and generic
  use the same full virtual address for global storage. The tensor-map acquire
  intrinsic requires the latter carrier even when the preceding descriptor
  update correctly uses an explicitly global operand.
"""
@generated function reinterpret_addrspace(::Val{To},
        p::Core.LLVMPtr{T, From}) where {To, T, From}
    # Typed spelling — parses on Julia ≤ 1.11's typed-pointer device
    # context and auto-upgrades to opaque on ≥ 1.12 (see NVVM.llvmtype).
    spell(n) = n == 0 ? "i8*" : "i8 addrspace($n)*"
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

# 32-bit raw-address form, for surfaces that carry addresses as UInt32
# rather than pointers (TMEM taddr from tcgen05.alloc, 32-bit SMEM offsets
# from smem_addr_u32). Same bit-preservation contract as above.
@generated function reinterpret_addrspace(::Val{To}, addr::UInt32) where {To}
    # Typed spelling — parses on Julia ≤ 1.11's typed-pointer device
    # context and auto-upgrades to opaque on ≥ 1.12 (see NVVM.llvmtype).
    spell(n) = n == 0 ? "i8*" : "i8 addrspace($n)*"
    ir = """
        %p = inttoptr i32 %0 to $(spell(To))
        ret $(spell(To)) %p"""
    quote
        Base.@inline
        Base.llvmcall($ir, Core.LLVMPtr{UInt8, $To}, Tuple{UInt32}, addr)
    end
end
