# `wgmma.mma_async` SMEM operand encoding (PTX 9.2 §9.7.14.5;
# PTX_ISA_DIGEST.md §10.4):
#   bits [13:0]   start_addr      = (smem_addr & 0x3FFFF) >> 4
#   bits [29:16]  leading_offset  = (leading_byte_offset & 0x3FFFF) >> 4
#   bits [45:32]  stride_offset   = (stride_byte_offset  & 0x3FFFF) >> 4
#   bits [51:49]  base_offset     (3 bits — 0 unless swizzle pattern doesn't
#                                  start at a swizzle boundary)
#   bits [63:62]  swizzle mode    (0 = none, 1 = 128B, 2 = 64B, 3 = 32B)
# Each 14-bit field encodes the high 14 bits of an 18-bit window — 16-byte
# aligned, fits in 256KB. SMEM is 228KB max per CTA on H100, so any in-CTA
# offset fits.
# Swizzle alignment: 128B → 1024B-aligned; 64B → 512B; 32B → 256B.

# Plain UInt8 module-level consts — Julia enums don't constant-fold as nicely
# inside @inline GPU code.
module WgmmaSwizzle
    const NONE = UInt8(0)
    const B128 = UInt8(1)
    const B64  = UInt8(2)
    const B32  = UInt8(3)
end

# Bounce a shared-AS pointer through inline asm with the `r` constraint to
# surface the 32-bit SMEM offset NVPTX represents internally.
@inline function smem_addr_u32(p::Core.LLVMPtr{T, AS.Shared}) where T
    @asmcall("mov.u32 \$0, \$1;", "=r,r", false, UInt32,
             Tuple{Core.LLVMPtr{T, AS.Shared}}, p)
end

@inline _wgmma_field14(x::UInt64) = (x & 0x3FFFF) >> 4

@inline function wgmma_descriptor(
        smem_addr_u32::UInt32;
        leading_byte_offset::Integer,
        stride_byte_offset::Integer,
        swizzle::Integer = WgmmaSwizzle.NONE,
        base_offset::Integer = 0)
    addr = _wgmma_field14(UInt64(smem_addr_u32))
    ld   = _wgmma_field14(UInt64(leading_byte_offset)) << 16
    sd   = _wgmma_field14(UInt64(stride_byte_offset))  << 32
    bo   = (UInt64(base_offset) & 0x7) << 49
    sw   = (UInt64(swizzle)     & 0x3) << 62
    addr | ld | sd | bo | sw
end
