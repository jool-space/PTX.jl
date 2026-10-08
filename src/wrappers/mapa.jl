# `mapa.shared::cluster.{u32,u64}` — translate a local-CTA shared-memory
# address into the cluster-mapped address that addresses the equivalent
# SMEM slot in CTA `rank`. PTX 9.2 §9.7.12.16. sm_90+.
#
# Hand-written: the chain default would infer a `UInt32`/`UInt64` return
# from the trailing dtype suffix, but the semantically correct surface is
# `LLVMPtr → LLVMPtr`: the source is a local `shared::cta` pointer (AS 3)
# and the result is a `shared::cluster` pointer (AS 7, `AS.SharedCluster`).
# `.shared::cluster` consumers accept either space, since the ISA defines
# every `shared::cta` address as a valid `shared::cluster` address;
# `.shared::cta` consumers reject AS 7, so a remote address cannot reach
# them without an explicit `reinterpret_addrspace`.

# 32-bit shared-memory pointer form. LLVMPtr-in / LLVMPtr-out keeps the
# pointer through the @asmcall as one register; ptxas accepts it under the
# `r` (32-bit) constraint since shared addresses fit in 32 bits.
@generated function ptx"mapa.shared::cluster.u32"(
        src::Core.LLVMPtr{T, AS.Shared}, rank::UInt32) where T
    quote
        Base.@inline
        @asmcall("mapa.shared::cluster.u32 \$0, \$1, \$2;",
                 "=r,r,r", true,
                 Core.LLVMPtr{$T, AS.SharedCluster},
                 Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt32},
                 src, rank)
    end
end

# 64-bit form (rarely useful for SMEM since SMEM addresses are 32 bits, but
# the instruction is documented and ptxas accepts it for the generic-pointer
# variant). Same shape as the u32 form with `l` constraint.
@generated function ptx"mapa.shared::cluster.u64"(
        src::Core.LLVMPtr{T, AS.Shared}, rank::UInt32) where T
    quote
        Base.@inline
        @asmcall("mapa.shared::cluster.u64 \$0, \$1, \$2;",
                 "=l,l,r", true,
                 Core.LLVMPtr{$T, AS.SharedCluster},
                 Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt32},
                 src, rank)
    end
end
