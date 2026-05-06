# `tcgen05.mma` consumes a 32-bit instruction descriptor (idesc) and a 64-bit
# shared-memory descriptor (sdesc) per operand. Mirrors CUTLASS/CuTe's
# `UMMA::make_instr_desc` and `UMMA::SmemDescriptor`. Bit layouts from
# PTX 9.2 §9.7.16; PTX_ISA_DIGEST.md §11.5; cross-checked against pyptx.

# 32-bit idesc, dense F16/BF16/TF32 → F32 path:
#   bit  2     sparse
#   bit  3     saturate
#   bits  4–6  C format (0=f16, 1=f32, 2=s32)        — fixed at f32 here
#   bits  7–9  A format (0=f16, 1=bf16, 2=tf32)
#   bits 10–12 B format (same encoding; usually = A)
#   bit  13    scale-A (0 = +1, 1 = -1)
#   bit  14    scale-B (0 = +1, 1 = -1)
#   bit  15    A major axis (0 = K, 1 = MN)
#   bit  16    B major axis (0 = K, 1 = MN)
#   bits 17–22 N >> 3                                — N ∈ [8,256], step 8
#   bits 24–27 M >> 4                                — M ∈ {64, 128, 256}
#   bits 30–31 max_shift                             — only with .ashift
#
# 64-bit smem descriptor:
#   bits  0–13 (smem_addr & 0x3FFF0) >> 4            — 14-bit SMEM offset
#   bits 16–29 (leading_bytes >> 4)                  — leading-dim stride
#   bits 32–45 (stride_bytes  >> 4)                  — stride-byte offset
#   bit  46–47 version                               (default 1)
#   bits 49–51 base_offset                           (3-bit, normally 0)
#   bit  52    lbo_mode                              (1 bit, normally 0)
#   bits 61–63 layout_type                           — see BlackwellLayout
#
# Stride / leading byte values must be 16-aligned (low 4 bits dropped).
# Tested-equivalent: BLACKWELL_MASKED_DESC_B128 = 0x4000404000010000 ≡
#   tcgen05_descriptor(UInt32(0); leading_bytes=16, stride_bytes=1024,
#                      swizzle=BlackwellLayout.B128).

# 3-bit swizzle field at bits [63:61]. Distinct from `WgmmaSwizzle`:
# Blackwell uses 3 bits with non-consecutive values and adds `B128_BASE32B`.
module BlackwellLayout
    const NONE          = UInt8(0)
    const B128_BASE32B  = UInt8(1)
    const B128          = UInt8(2)
    const B64           = UInt8(4)
    const B32           = UInt8(6)
end

# Every dense F16/BF16/TF32 → F32 path through this builder fixes C at f32.
const _TCGEN05_C_FORMAT_F32 = UInt32(1)        # bits  4–6
const _TCGEN05_AB_FORMAT = (
    f16  = UInt32(0),
    bf16 = UInt32(1),
    tf32 = UInt32(2),
)                                              # bits  7–9 (A) / 10–12 (B)
const _TCGEN05_MAJOR = (K = UInt32(0), MN = UInt32(1))   # bits 15 / 16
const _TCGEN05_SCALE_BIT = (1 => UInt32(0), -1 => UInt32(1))  # bits 13 / 14

@inline function tcgen05_instr_desc_f16bf16_f32(;
        m::Integer,
        n::Integer,
        ab_dtype::Symbol,
        a_major::Symbol = :K,
        b_major::Symbol = :K,
        scale_a::Integer = 1,
        scale_b::Integer = 1,
        saturate::Bool = false,
        sparse::Bool = false,
        max_shift::Integer = 0)
    m in (64, 128, 256) ||
        throw(ArgumentError("m must be 64, 128, or 256; got $m"))
    (n % 8 == 0 && 8 <= n <= 256) ||
        throw(ArgumentError("n must be a multiple of 8 in [8, 256]; got $n"))
    haskey(_TCGEN05_AB_FORMAT, ab_dtype) ||
        throw(ArgumentError("ab_dtype must be :f16, :bf16, or :tf32; got $(repr(ab_dtype))"))
    haskey(_TCGEN05_MAJOR, a_major) ||
        throw(ArgumentError("a_major must be :K or :MN; got $(repr(a_major))"))
    haskey(_TCGEN05_MAJOR, b_major) ||
        throw(ArgumentError("b_major must be :K or :MN; got $(repr(b_major))"))
    (scale_a == 1 || scale_a == -1) ||
        throw(ArgumentError("scale_a must be 1 or -1; got $scale_a"))
    (scale_b == 1 || scale_b == -1) ||
        throw(ArgumentError("scale_b must be 1 or -1; got $scale_b"))
    0 <= max_shift <= 3 ||
        throw(ArgumentError("max_shift must be in 0:3; got $max_shift"))

    ab_format = _TCGEN05_AB_FORMAT[ab_dtype]
    desc = UInt32(0)
    desc |= UInt32(sparse)   << 2
    desc |= UInt32(saturate) << 3
    desc |= _TCGEN05_C_FORMAT_F32 << 4
    desc |= ab_format << 7
    desc |= ab_format << 10
    desc |= (scale_a == -1 ? UInt32(1) : UInt32(0)) << 13
    desc |= (scale_b == -1 ? UInt32(1) : UInt32(0)) << 14
    desc |= _TCGEN05_MAJOR[a_major] << 15
    desc |= _TCGEN05_MAJOR[b_major] << 16
    desc |= UInt32(n >> 3) << 17
    desc |= UInt32(m >> 4) << 24
    desc |= UInt32(max_shift) << 30
    desc
end

@inline _tcgen05_field14(x::UInt64) = (x & 0x3FFF0) >> 4

@inline function tcgen05_descriptor(
        smem_addr_u32::UInt32;
        leading_bytes::Integer,
        stride_bytes::Integer,
        swizzle::Integer = BlackwellLayout.NONE,
        version::Integer = 1,
        base_offset::Integer = 0,
        lbo_mode::Integer = 0)
    leading_bytes % 16 == 0 ||
        throw(ArgumentError("leading_bytes must be 16-aligned; got $leading_bytes"))
    stride_bytes % 16 == 0 ||
        throw(ArgumentError("stride_bytes must be 16-aligned; got $stride_bytes"))
    0 <= version <= 0x3 ||
        throw(ArgumentError("version must fit in 2 bits; got $version"))
    0 <= base_offset <= 0x7 ||
        throw(ArgumentError("base_offset must fit in 3 bits; got $base_offset"))
    0 <= lbo_mode <= 0x1 ||
        throw(ArgumentError("lbo_mode must fit in 1 bit; got $lbo_mode"))

    addr = _tcgen05_field14(UInt64(smem_addr_u32))
    ld   = (UInt64(leading_bytes) >> 4) << 16
    sd   = (UInt64(stride_bytes)  >> 4) << 32
    ver  = (UInt64(version)     & 0x3) << 46
    bo   = (UInt64(base_offset) & 0x7) << 49
    lbo  = (UInt64(lbo_mode)    & 0x1) << 52
    sw   = UInt64(swizzle) << 61
    addr | ld | sd | ver | bo | lbo | sw
end
