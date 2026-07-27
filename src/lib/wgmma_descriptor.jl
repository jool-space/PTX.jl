# `wgmma.mma_async` SMEM operand encoding (PTX ISA 9.3 §9.7.16.5.1.2.2,
# "Matrix Descriptor Format"):
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

"""
    smem_addr_u32(p::Core.LLVMPtr{T, AS.Shared}) -> UInt32

The 32-bit in-CTA SMEM offset of a shared-address-space pointer — the
`smem_addr_u32` argument the descriptor builders take. Bounces the pointer
through inline asm with the `r` constraint to surface the 32-bit offset
NVPTX represents internally.
"""
@inline function smem_addr_u32(p::Core.LLVMPtr{T, AS.Shared}) where T
    @asmcall("mov.u32 \$0, \$1;", "=r,r", false, UInt32,
             Tuple{Core.LLVMPtr{T, AS.Shared}}, p)
end

@inline _wgmma_field14(x::UInt64) = (x & 0x3FFFF) >> 4

"""
    wgmma_descriptor(smem_addr_u32; leading_byte_offset, stride_byte_offset,
                     swizzle = WgmmaSwizzle.NONE, base_offset = 0) -> UInt64

Pack a `wgmma.mma_async` shared-memory operand descriptor. Byte offsets
encode as 14-bit fields covering the high bits of an 18-bit, 16-byte-aligned
window; `swizzle` is one of the `WgmmaSwizzle` codes (0 = none, 1 = 128B,
2 = 64B, 3 = 32B). Get `smem_addr_u32` from [`smem_addr_u32`](@ref).
"""
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

"""
    step_desc(desc::UInt64, byte_offset::Integer) -> UInt64

Advance a wgmma SMEM-operand descriptor's start address by `byte_offset`
bytes. The descriptor's low 14 bits encode `(smem_addr & 0x3FFFF) >> 4`,
so adding `byte_offset >> 4` moves the base by the equivalent byte
distance — as long as the carry stays within the 14-bit start_addr field
(true for any in-CTA SMEM offset on H100). Used to walk a SMEM ring
buffer (per-stage offset) or to step within a tile (per-wgmma-K offset
on the inner loop).

`byte_offset` must be a multiple of 16 — the operand encoding has no
representation for sub-16-byte offsets. Caller guarantees this; the
helper truncates silently if violated.
"""
@inline step_desc(desc::UInt64, byte_offset::Integer) =
    desc + UInt64(byte_offset >> 4)
