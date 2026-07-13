# `tcgen05.mma` consumes a 32-bit instruction descriptor (idesc) and a 64-bit
# shared-memory descriptor (sdesc) per operand. Mirrors CUTLASS/CuTe's
# `UMMA::make_instr_desc` and `UMMA::SmemDescriptor`. Bit layouts are pinned to
# PTX 9.3 §9.7.17.4 Tables 43 and 45 and cross-checked against pyptx.

# 32-bit idesc, F16/BF16/TF32 → F32 path:
#   bit  2     sparse
#   bit  3     saturate for integer kinds             — reserved at 0 here
#   bits  4–5  D format (0=f16, 1=f32, 2=s32)        — fixed at f32 here
#   bit  6     reserved                               — fixed at 0
#   bits  7–9  A format (0=f16, 1=bf16, 2=tf32)
#   bits 10–12 B format (same encoding; usually = A)
#   bit  13    scale-A (0 = +1, 1 = -1)
#   bit  14    scale-B (0 = +1, 1 = -1)
#   bit  15    A major axis (0 = K, 1 = MN)
#   bit  16    B major axis (0 = K, 1 = MN)
#   bits 17–22 N >> 3                                — N ∈ [8,256], step 8
#   bit  23    reserved                               — fixed at 0
#   bits 24–28 M >> 4                                — M ∈ {32,64,128,256}
#   bit  29    reserved                               — fixed at 0
#   bits 30–31 max_shift                             — only with `.ws`
#
# 64-bit smem descriptor:
#   bits  0–13 (smem_addr & 0x3FFF0) >> 4            — 14-bit SMEM offset
#   bits 16–29 (leading_bytes >> 4)                  — leading-dim stride
#   bits 32–45 (stride_bytes  >> 4)                  — stride-byte offset
#   bits 46–48 fixed constant 0b001
#   bits 49–51 base_offset                           (3-bit, normally 0)
#   bit  52    lbo_mode                              (1 bit, normally 0)
#   bits 53–60 fixed constant 0
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

# Every dense F16/BF16/TF32 → F32 path through this builder fixes D at f32.
const _TCGEN05_D_FORMAT_F32 = UInt32(1)        # bits 4–5; bit 6 is reserved
const _TCGEN05_AB_FORMAT = (
    f16  = UInt32(0),
    bf16 = UInt32(1),
    tf32 = UInt32(2),
)                                              # bits  7–9 (A) / 10–12 (B)
const _TCGEN05_MAJOR = (K = UInt32(0), MN = UInt32(1))   # bits 15 / 16

"""
    tcgen05_instr_desc_f16bf16_f32(; m, n, ab_dtype,
        a_major=:K, b_major=:K, scale_a=1, scale_b=1,
        sparse=false, max_shift=0) -> UInt32

Pack the PTX 9.3 §9.7.17.4.2 Table 45 instruction descriptor for the
`.kind::f16` / `.kind::tf32` paths whose destination type is `.f32` and whose
A/B types are the same. `ab_dtype` is `:f16`, `:bf16`, or `:tf32`.

The integer-only saturation bit and every reserved bit are fixed at zero and
are not caller-controlled. `sparse=true` selects the `.sp` descriptor bit with
sparsity selector zero. A nonzero `max_shift` is meaningful only when the
descriptor is consumed by a `.ws` form; legality of the complete MMA shape and
form remains the responsibility of the consuming instruction wrapper.
"""
@inline function tcgen05_instr_desc_f16bf16_f32(;
        m::Integer,
        n::Integer,
        ab_dtype::Symbol,
        a_major::Symbol = :K,
        b_major::Symbol = :K,
        scale_a::Integer = 1,
        scale_b::Integer = 1,
        sparse::Bool = false,
        max_shift::Integer = 0)
    m in (32, 64, 128, 256) ||
        throw(ArgumentError("m must be 32, 64, 128, or 256"))
    (n % 8 == 0 && 8 <= n <= 256) ||
        throw(ArgumentError("n must be a multiple of 8 in [8, 256]"))
    haskey(_TCGEN05_AB_FORMAT, ab_dtype) ||
        throw(ArgumentError("ab_dtype must be :f16, :bf16, or :tf32"))
    haskey(_TCGEN05_MAJOR, a_major) ||
        throw(ArgumentError("a_major must be :K or :MN"))
    haskey(_TCGEN05_MAJOR, b_major) ||
        throw(ArgumentError("b_major must be :K or :MN"))
    (scale_a == 1 || scale_a == -1) ||
        throw(ArgumentError("scale_a must be 1 or -1"))
    (scale_b == 1 || scale_b == -1) ||
        throw(ArgumentError("scale_b must be 1 or -1"))
    0 <= max_shift <= 3 ||
        throw(ArgumentError("max_shift must be in 0:3"))

    ab_format = _TCGEN05_AB_FORMAT[ab_dtype]
    desc = UInt32(0)
    desc |= UInt32(sparse)   << 2
    desc |= _TCGEN05_D_FORMAT_F32 << 4
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

const _TCGEN05_SDESC_FIXED = UInt64(1) << 46
const _TCGEN05_SWIZZLES = (
    BlackwellLayout.NONE,
    BlackwellLayout.B128_BASE32B,
    BlackwellLayout.B128,
    BlackwellLayout.B64,
    BlackwellLayout.B32,
)

@inline function _tcgen05_descriptor_field(x::Integer)
    0 <= x <= 0x3FFF0 ||
        throw(ArgumentError("tcgen05 descriptor field must fit the aligned 18-bit input window"))
    x % 16 == 0 ||
        throw(ArgumentError("tcgen05 descriptor fields must be 16-byte aligned"))
    _tcgen05_field14(UInt64(x))
end

"""
    tcgen05_descriptor(smem_addr_u32; leading_bytes, stride_bytes,
        swizzle=BlackwellLayout.NONE, base_offset=0, lbo_mode=0) -> UInt64

Pack the `tcgen05` shared-memory descriptor from PTX 9.3 §9.7.17.4.1
Table 43. The matrix address and both byte fields must be 16-byte aligned and
fit the descriptor's 18-bit input window. `swizzle` must be one of the five
encodings in `BlackwellLayout`.

Bits 46–48 are always the ISA-mandated constant `0b001`; reserved/fixed-zero
bits are never caller-controlled. `lbo_mode=1` selects the absolute leading-
dimension byte-address mode. PTX 9.3 §9.7.17.3.1.2 restricts that mode to
`BlackwellLayout.B128` (128-byte swizzle with 16-byte atomicity) and a zero
`base_offset`, which this builder enforces. The caller must additionally pair
it with K-major A and B descriptors (both instruction-descriptor transpose
bits zero), a 48-byte K dimension, and the architecture-specific `sm_103a`
target. The default `lbo_mode=0` is the relative byte-offset mode.
"""
@inline function tcgen05_descriptor(
        smem_addr_u32::UInt32;
        leading_bytes::Integer,
        stride_bytes::Integer,
        swizzle::Integer = BlackwellLayout.NONE,
        base_offset::Integer = 0,
        lbo_mode::Integer = 0)
    swizzle in _TCGEN05_SWIZZLES ||
        throw(ArgumentError(
            "swizzle must be a BlackwellLayout encoding (0, 1, 2, 4, or 6)"))
    0 <= base_offset <= 0x7 ||
        throw(ArgumentError("base_offset must fit in 3 bits"))
    0 <= lbo_mode <= 0x1 ||
        throw(ArgumentError("lbo_mode must fit in 1 bit"))
    if lbo_mode == 1
        swizzle == BlackwellLayout.B128 ||
            throw(ArgumentError(
                "absolute leading-address mode requires BlackwellLayout.B128"))
        base_offset == 0 ||
            throw(ArgumentError(
                "absolute leading-address mode requires base_offset=0"))
    end

    # Keep the dynamic-address error paths GPU-compilable: these helpers use
    # static diagnostics rather than interpolating device values into strings.
    addr = _tcgen05_descriptor_field(smem_addr_u32)
    ld   = _tcgen05_descriptor_field(leading_bytes) << 16
    sd   = _tcgen05_descriptor_field(stride_bytes) << 32
    bo   = (UInt64(base_offset) & 0x7) << 49
    lbo  = (UInt64(lbo_mode)    & 0x1) << 52
    sw   = UInt64(swizzle) << 61
    addr | ld | sd | _TCGEN05_SDESC_FIXED | bo | lbo | sw
end
