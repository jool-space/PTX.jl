# `tcgen05.mma` consumes a 32-bit instruction descriptor (idesc) and a 64-bit
# shared-memory descriptor (sdesc) per operand. Mirrors CUTLASS/CuTe's
# `UMMA::make_instr_desc` and `UMMA::SmemDescriptor`. Bit layouts: sdesc and
# the f16 idesc path are pinned to PTX 9.3 §9.7.17.4 Tables 43/45 (cross-
# checked against pyptx); the i8/f8f6f4 and block-scale idesc paths to
# PTX 9.4 §9.7.18.4.2 Tables 51–53, with the i8 and f8f6f4 encodings
# runtime-validated on a B200 by the idesc hardware probes. Every
# architecture-gated bit (sm_107f-family K/scale-layout encodings, the
# 15-bit shared-descriptor fields, the lut::b segment bit) defaults to
# zero and is set only through an explicit keyword.

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
#   bit  29    K dimension (sm_107f; f8f6f4 only)   — 0 unless `k` widens
#   bits 30–31 max_shift                             — only with `.ws`
#
# 64-bit smem descriptor:
#   bits  0–13 (smem_addr & 0x3FFF0) >> 4            — 14-bit SMEM offset
#   bit  14    upper address bit                     — `wide=true` only
#   bits 16–29 (leading_bytes >> 4)                  — leading-dim stride
#   bit  30    upper leading-dim bit                 — `wide=true` only
#   bits 32–45 (stride_bytes  >> 4)                  — stride-byte offset
#   bits 46–48 fixed constant 0b001
#   bits 49–51 base_offset                           (3-bit, normally 0)
#   bit  52    lbo_mode                              (1 bit, normally 0)
#   bit  53    lut::b K-segment offset               (0 or 24 bytes)
#   bits 54–60 fixed constant 0
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

# Table 51's shared shape tail: N (bits 17–22, N>>3) and M (bits 24–28, M>>4).
# The mx kinds do NOT use this — Table 52/53 place M>>7 at bits 27–28.
@inline function _tcgen05_idesc_shape(m::Integer, n::Integer)
    m in (32, 64, 128, 256) ||
        throw(ArgumentError("m must be 32, 64, 128, or 256"))
    (n % 8 == 0 && 8 <= n <= 256) ||
        throw(ArgumentError("n must be a multiple of 8 in [8, 256]"))
    UInt32(n >> 3) << 17 | UInt32(m >> 4) << 24
end

@inline function _tcgen05_idesc_major(a_major::Symbol, b_major::Symbol)
    haskey(_TCGEN05_MAJOR, a_major) ||
        throw(ArgumentError("a_major must be :K or :MN"))
    haskey(_TCGEN05_MAJOR, b_major) ||
        throw(ArgumentError("b_major must be :K or :MN"))
    _TCGEN05_MAJOR[a_major] << 15 | _TCGEN05_MAJOR[b_major] << 16
end

@inline function _tcgen05_idesc_negate(scale_a::Integer, scale_b::Integer)
    (scale_a == 1 || scale_a == -1) ||
        throw(ArgumentError("scale_a must be 1 or -1"))
    (scale_b == 1 || scale_b == -1) ||
        throw(ArgumentError("scale_b must be 1 or -1"))
    UInt32(scale_a == -1) << 13 | UInt32(scale_b == -1) << 14
end

@inline function _tcgen05_idesc_max_shift(max_shift::Integer)
    0 <= max_shift <= 3 ||
        throw(ArgumentError("max_shift must be in 0:3"))
    UInt32(max_shift) << 30
end

const _TCGEN05_I8_AB_FORMAT = (u8 = UInt32(0), s8 = UInt32(1))

"""
    tcgen05_instr_desc_i8(; m, n, a_dtype, b_dtype,
        a_major=:K, b_major=:K, saturate=false,
        sparse=false, max_shift=0) -> UInt32

Pack the PTX 9.4 §9.7.18.4.2 Table 51 instruction descriptor for the
`.kind::i8` path (destination type fixed at `.s32`). `a_dtype` and `b_dtype`
are `:u8` or `:s8`, independently.

Negation (bits 13–14) is not supported for `.kind::i8` (§9.7.18.10 Table 62)
and is not caller-controlled; transpose (`a_major`/`b_major` = `:MN`) is.
Bit 29 (the wider-K encoding) is architecture-gated and fixed at zero, so the
descriptor always encodes the base K (dense 32 / sparse 64). `.kind::i8`
itself is a-variant-exclusive (§9.7.18.10 target notes); legality of the
complete MMA form remains the consuming wrapper's responsibility.
"""
@inline function tcgen05_instr_desc_i8(;
        m::Integer,
        n::Integer,
        a_dtype::Symbol,
        b_dtype::Symbol,
        a_major::Symbol = :K,
        b_major::Symbol = :K,
        saturate::Bool = false,
        sparse::Bool = false,
        max_shift::Integer = 0)
    haskey(_TCGEN05_I8_AB_FORMAT, a_dtype) ||
        throw(ArgumentError("a_dtype must be :u8 or :s8"))
    haskey(_TCGEN05_I8_AB_FORMAT, b_dtype) ||
        throw(ArgumentError("b_dtype must be :u8 or :s8"))

    UInt32(sparse) << 2 |
        UInt32(saturate) << 3 |
        UInt32(2) << 4 |                       # dtype = s32
        _TCGEN05_I8_AB_FORMAT[a_dtype] << 7 |
        _TCGEN05_I8_AB_FORMAT[b_dtype] << 10 |
        _tcgen05_idesc_major(a_major, b_major) |
        _tcgen05_idesc_shape(m, n) |
        _tcgen05_idesc_max_shift(max_shift)
end

const _TCGEN05_F8F6F4_AB_FORMAT = (
    e4m3 = UInt32(0),
    e5m2 = UInt32(1),
    e2m3 = UInt32(3),
    e3m2 = UInt32(4),
    e2m1 = UInt32(5),
)
const _TCGEN05_F8F6F4_D_FORMAT = (f16 = UInt32(0), f32 = UInt32(1))

# `.decompress::lut::b` (§9.7.18.10): B is E4M3 only and cannot be
# transposed. The descriptor bits are unchanged; the builder refuses the
# combinations the instruction cannot consume.
@inline function _tcgen05_idesc_lut_b(lut_b::Bool, b_dtype::Symbol,
                                      b_major::Symbol)
    lut_b || return nothing
    b_dtype === :e4m3 ||
        throw(ArgumentError("decompress::lut::b requires b_dtype = :e4m3"))
    b_major === :K ||
        throw(ArgumentError("decompress::lut::b does not transpose B"))
    nothing
end

# Table 62: the 6-/4-bit element types are not transposable in the dense
# wide-K (K=64) encoding.
@inline function _tcgen05_idesc_wide_k_major(a_dtype::Symbol, b_dtype::Symbol,
                                             a_major::Symbol, b_major::Symbol)
    for (dtype, major) in ((a_dtype, a_major), (b_dtype, b_major))
        dtype in (:e2m3, :e3m2, :e2m1) && major === :MN &&
            throw(ArgumentError(
                "the dense K=64 encoding does not transpose e2m3/e3m2/e2m1"))
    end
    nothing
end

"""
    tcgen05_instr_desc_f8f6f4(; m, n, a_dtype, b_dtype, d_dtype=:f32,
        a_major=:K, b_major=:K, scale_a=1, scale_b=1,
        sparse=false, max_shift=0, k=nothing, lut_b=false) -> UInt32

Pack the PTX 9.4 §9.7.18.4.2 Table 51 instruction descriptor for the
`.kind::f8f6f4` path. `a_dtype` and `b_dtype` are independently `:e4m3`,
`:e5m2`, `:e2m3`, `:e3m2`, or `:e2m1`; `d_dtype` is `:f16` or `:f32`.

`k` selects the K dimension: `nothing` (the default) or the base K (dense
32 / sparse 64) leaves bit 29 clear; dense `k = 64` sets it. Table 51 lists
no wide sparse K for this kind, so a sparse `k` other than 64 is rejected.
The wide dense encoding does not transpose the 6-/4-bit element types
(§9.7.18.10 Table 62), and `lut_b = true` (the `.decompress::lut::b`
consumer) requires an E4M3, K-major B; both are enforced rather than
encoded. Bit 29 is valid on `sm_107f` or higher in the same family only.
The integer saturation bit is N/A for float kinds and not caller-controlled.
"""
@inline function tcgen05_instr_desc_f8f6f4(;
        m::Integer,
        n::Integer,
        a_dtype::Symbol,
        b_dtype::Symbol,
        d_dtype::Symbol = :f32,
        a_major::Symbol = :K,
        b_major::Symbol = :K,
        scale_a::Integer = 1,
        scale_b::Integer = 1,
        sparse::Bool = false,
        max_shift::Integer = 0,
        k::Union{Nothing, Integer} = nothing,
        lut_b::Bool = false)
    haskey(_TCGEN05_F8F6F4_AB_FORMAT, a_dtype) ||
        throw(ArgumentError(
            "a_dtype must be :e4m3, :e5m2, :e2m3, :e3m2, or :e2m1"))
    haskey(_TCGEN05_F8F6F4_AB_FORMAT, b_dtype) ||
        throw(ArgumentError(
            "b_dtype must be :e4m3, :e5m2, :e2m3, :e3m2, or :e2m1"))
    haskey(_TCGEN05_F8F6F4_D_FORMAT, d_dtype) ||
        throw(ArgumentError("d_dtype must be :f16 or :f32"))
    wide_k = if k === nothing
        false
    elseif sparse
        k == 64 || throw(ArgumentError(
            "sparse .kind::f8f6f4 encodes K=64 only (Table 51)"))
        false
    else
        k in (32, 64) || throw(ArgumentError(
            "dense .kind::f8f6f4 encodes K=32 or K=64"))
        k == 64
    end
    wide_k && _tcgen05_idesc_wide_k_major(a_dtype, b_dtype, a_major, b_major)
    _tcgen05_idesc_lut_b(lut_b, b_dtype, b_major)

    UInt32(sparse) << 2 |
        _TCGEN05_F8F6F4_D_FORMAT[d_dtype] << 4 |
        _TCGEN05_F8F6F4_AB_FORMAT[a_dtype] << 7 |
        _TCGEN05_F8F6F4_AB_FORMAT[b_dtype] << 10 |
        _tcgen05_idesc_negate(scale_a, scale_b) |
        _tcgen05_idesc_major(a_major, b_major) |
        _tcgen05_idesc_shape(m, n) |
        UInt32(wide_k) << 29 |
        _tcgen05_idesc_max_shift(max_shift)
end

const _TCGEN05_TI16_AB_FORMAT = UInt32(3)      # .s1z4m11 (Table 51)

"""
    tcgen05_instr_desc_ti16(; m, n, scale_a=1, scale_b=1,
        sparse=false, max_shift=0) -> UInt32

Pack the PTX 9.4 §9.7.18.4.2 Table 51 instruction descriptor for the
`.kind::ti16` path: A and B are the ISA-fixed `.s1z4m11` element type and
the destination type `.s32`. Negation (`scale_a`/`scale_b` = -1) is
supported (Table 51 and §9.7.18.10 Table 62 agree).

Transpose is not caller-controlled: Table 51 lists only "No Transpose" for
this kind while Table 62 lists transpose as supported, and no assembler or
hardware evidence resolves the contradiction, so bits 15–16 stay zero until
it is. The integer saturation bit is N/A for this kind and bit 29 has no
defined value; both stay zero.
"""
@inline function tcgen05_instr_desc_ti16(;
        m::Integer,
        n::Integer,
        scale_a::Integer = 1,
        scale_b::Integer = 1,
        sparse::Bool = false,
        max_shift::Integer = 0)
    UInt32(sparse) << 2 |
        UInt32(2) << 4 |                       # dtype = s32
        _TCGEN05_TI16_AB_FORMAT << 7 |
        _TCGEN05_TI16_AB_FORMAT << 10 |
        _tcgen05_idesc_negate(scale_a, scale_b) |
        _tcgen05_idesc_shape(m, n) |
        _tcgen05_idesc_max_shift(max_shift)
end

# Table 52/53 shared pieces: block-scale shape (M>>7 at bits 27–28, same
# N>>3 window as Table 51) and the scale-factor data IDs.
@inline function _tcgen05_idesc_mx_shape(m::Integer, n::Integer)
    m in (128, 256) ||
        throw(ArgumentError("m must be 128 or 256 for block-scale kinds"))
    (n % 8 == 0 && 8 <= n <= 256) ||
        throw(ArgumentError("n must be a multiple of 8 in [8, 256]"))
    UInt32(n >> 3) << 17 | UInt32(m >> 7) << 27
end

# Table 52/53 bit 26: scale-factor-A layout, 32-lane (0) or 128-lane (1).
@inline function _tcgen05_idesc_scale_layout(scale_a_layout::Integer)
    scale_a_layout in (32, 128) ||
        throw(ArgumentError("scale_a_layout must be 32 or 128"))
    UInt32(scale_a_layout == 128) << 26
end

"""
    tcgen05_instr_desc_mxf8f6f4(; m, n, a_dtype, b_dtype,
        scale_a_id, scale_b_id, a_major=:K, b_major=:K,
        scale_a=1, scale_b=1, sparse=false,
        k=nothing, scale_a_layout=32, lut_b=false) -> UInt32

Pack the PTX 9.4 §9.7.18.4.2 Table 52 instruction descriptor for the
`.kind::mxf8f6f4` path. `a_dtype`/`b_dtype` take the same five element types
as `.kind::f8f6f4`; `scale_a_id`/`scale_b_id` are the scale-factor data IDs
(0–3). The scale matrix type is the ISA-fixed `UE8M0` (bit 23).

`k` selects the K dimension: `nothing` or the base K (dense 32 / sparse 64)
leaves bit 31 clear; dense 64 / sparse 128 sets it. `scale_a_layout` is 32
(the default) or 128 lanes (bit 26). Both encodings are valid on `sm_107f`
or higher in the same family only. The wide dense encoding does not
transpose the 6-/4-bit element types and `lut_b = true` requires an E4M3,
K-major B (§9.7.18.10 Table 62); both are enforced rather than encoded.
"""
@inline function tcgen05_instr_desc_mxf8f6f4(;
        m::Integer,
        n::Integer,
        a_dtype::Symbol,
        b_dtype::Symbol,
        scale_a_id::Integer,
        scale_b_id::Integer,
        a_major::Symbol = :K,
        b_major::Symbol = :K,
        scale_a::Integer = 1,
        scale_b::Integer = 1,
        sparse::Bool = false,
        k::Union{Nothing, Integer} = nothing,
        scale_a_layout::Integer = 32,
        lut_b::Bool = false)
    haskey(_TCGEN05_F8F6F4_AB_FORMAT, a_dtype) ||
        throw(ArgumentError(
            "a_dtype must be :e4m3, :e5m2, :e2m3, :e3m2, or :e2m1"))
    haskey(_TCGEN05_F8F6F4_AB_FORMAT, b_dtype) ||
        throw(ArgumentError(
            "b_dtype must be :e4m3, :e5m2, :e2m3, :e3m2, or :e2m1"))
    0 <= scale_a_id <= 3 ||
        throw(ArgumentError("scale_a_id must be in 0:3"))
    0 <= scale_b_id <= 3 ||
        throw(ArgumentError("scale_b_id must be in 0:3"))
    wide_k = if k === nothing
        false
    else
        base = sparse ? 64 : 32
        k in (base, 2base) || throw(ArgumentError(
            "k must be the base K or twice it for .kind::mxf8f6f4"))
        k == 2base
    end
    wide_k && !sparse &&
        _tcgen05_idesc_wide_k_major(a_dtype, b_dtype, a_major, b_major)
    _tcgen05_idesc_lut_b(lut_b, b_dtype, b_major)

    UInt32(sparse) << 2 |
        UInt32(scale_b_id) << 4 |
        _TCGEN05_F8F6F4_AB_FORMAT[a_dtype] << 7 |
        _TCGEN05_F8F6F4_AB_FORMAT[b_dtype] << 10 |
        _tcgen05_idesc_negate(scale_a, scale_b) |
        _tcgen05_idesc_major(a_major, b_major) |
        _tcgen05_idesc_mx_shape(m, n) |
        UInt32(1) << 23 |                      # scale matrix type = UE8M0
        _tcgen05_idesc_scale_layout(scale_a_layout) |
        UInt32(scale_a_id) << 29 |
        UInt32(wide_k) << 31
end

const _TCGEN05_NVF4_SCALE_FORMAT = (
    ue4m3 = UInt32(0),
    ue8m0 = UInt32(1),
    ue5m3 = UInt32(2),
)

# Table 53's two-bit K code: bit 3 is the upper bit and bit 31 the lower.
@inline function _tcgen05_idesc_mxf4_k(k, sparse::Bool)
    k === nothing && return UInt32(0)
    code = if sparse
        k == 128 ? 0 : k == 192 ? 1 : throw(ArgumentError(
            "sparse mxf4 kinds encode K=128 or K=192"))
    else
        k == 64 ? 0 : k == 96 ? 1 : k == 128 ? 2 : throw(ArgumentError(
            "dense mxf4 kinds encode K=64, K=96, or K=128"))
    end
    UInt32(code >> 1) << 3 | UInt32(code & 1) << 31
end

@inline function _tcgen05_idesc_mxf4_common(m, n, scale_a_id, scale_b_id,
                                            scale_a, scale_b, sparse,
                                            sparsity_version, k,
                                            scale_a_layout)
    scale_a_id in (0, 2) ||
        throw(ArgumentError("scale_a_id must be 0 or 2 for the mxf4 kinds"))
    scale_b_id in (0, 2) ||
        throw(ArgumentError("scale_b_id must be 0 or 2 for the mxf4 kinds"))
    0 <= sparsity_version <= 1 ||
        throw(ArgumentError("sparsity_version must be 0 or 1"))

    UInt32(sparse) << 2 |
        _tcgen05_idesc_mxf4_k(k, sparse) |
        UInt32(scale_b_id) << 4 |
        UInt32(1) << 7 |                       # atype = E2M1
        UInt32(1) << 10 |                      # btype = E2M1 (2-bit field)
        UInt32(sparsity_version) << 12 |
        _tcgen05_idesc_negate(scale_a, scale_b) |
        _tcgen05_idesc_mx_shape(m, n) |
        _tcgen05_idesc_scale_layout(scale_a_layout) |
        UInt32(scale_a_id) << 29
end

"""
    tcgen05_instr_desc_mxf4(; m, n, scale_a_id, scale_b_id,
        scale_a=1, scale_b=1, sparse=false, sparsity_version=0) -> UInt32

Pack the PTX 9.4 §9.7.18.4.2 Table 53 instruction descriptor for the
`.kind::mxf4` path. Element types are the ISA-fixed `E2M1` and the scale
matrix type the ISA-fixed `UE8M0`; `scale_a_id`/`scale_b_id` must be 0 or 2.
Transpose is not supported for the mxf4 kinds (§9.7.18.10 Table 62) and the
transpose bits are not caller-controlled; negation is.

`sparsity_version` (bit 12) selects both the target and the metadata layout:
0 uses pair-wise 4:8 sparsity on sm_100a/sm_103a/sm_110a; 1 uses element-wise
2:4 sparsity on sm_107a. A mismatch is undefined behavior. The default is 0;
callers targeting sm_107a must explicitly pass 1 and supply matching metadata.
`k` selects the K dimension (bits 3 and 31): `nothing` or the base K (dense
64 / sparse 128) encodes 0; dense 96 / sparse 192 encodes 1; dense 128
encodes 2. `scale_a_layout` is 32 (the default) or 128 lanes (bit 26).
The wider K codes and the 128-lane layout are valid on `sm_107f` or higher
in the same family only.
"""
@inline function tcgen05_instr_desc_mxf4(;
        m::Integer,
        n::Integer,
        scale_a_id::Integer,
        scale_b_id::Integer,
        scale_a::Integer = 1,
        scale_b::Integer = 1,
        sparse::Bool = false,
        sparsity_version::Integer = 0,
        k::Union{Nothing, Integer} = nothing,
        scale_a_layout::Integer = 32)
    _tcgen05_idesc_mxf4_common(m, n, scale_a_id, scale_b_id,
                               scale_a, scale_b, sparse, sparsity_version,
                               k, scale_a_layout) |
        UInt32(1) << 23                        # scale matrix type = UE8M0
end

"""
    tcgen05_instr_desc_mxf4nvf4(; m, n, scale_dtype, scale_a_id, scale_b_id,
        scale_a=1, scale_b=1, sparse=false, sparsity_version=0,
        k=nothing, scale_a_layout=32) -> UInt32

Pack the PTX 9.4 §9.7.18.4.2 Table 53 instruction descriptor for the
`.kind::mxf4nvf4` path — identical to [`tcgen05_instr_desc_mxf4`](@ref)
except the scale matrix type is caller-chosen: `scale_dtype` is `:ue4m3`,
`:ue8m0`, or `:ue5m3` (bits 23–24).
"""
@inline function tcgen05_instr_desc_mxf4nvf4(;
        m::Integer,
        n::Integer,
        scale_dtype::Symbol,
        scale_a_id::Integer,
        scale_b_id::Integer,
        scale_a::Integer = 1,
        scale_b::Integer = 1,
        sparse::Bool = false,
        sparsity_version::Integer = 0,
        k::Union{Nothing, Integer} = nothing,
        scale_a_layout::Integer = 32)
    haskey(_TCGEN05_NVF4_SCALE_FORMAT, scale_dtype) ||
        throw(ArgumentError("scale_dtype must be :ue4m3, :ue8m0, or :ue5m3"))
    _tcgen05_idesc_mxf4_common(m, n, scale_a_id, scale_b_id,
                               scale_a, scale_b, sparse, sparsity_version,
                               k, scale_a_layout) |
        _TCGEN05_NVF4_SCALE_FORMAT[scale_dtype] << 23
end

@inline _tcgen05_field14(x::UInt64) = (x & 0x3FFF0) >> 4
@inline _tcgen05_field15(x::UInt64) = (x & 0x7FFF0) >> 4

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

@inline function _tcgen05_descriptor_wide_field(x::Integer)
    0 <= x <= 0x7FFF0 ||
        throw(ArgumentError("tcgen05 wide descriptor field must fit the aligned 19-bit input window"))
    x % 16 == 0 ||
        throw(ArgumentError("tcgen05 descriptor fields must be 16-byte aligned"))
    _tcgen05_field15(UInt64(x))
end

"""
    tcgen05_descriptor(smem_addr_u32; leading_bytes, stride_bytes,
        swizzle=BlackwellLayout.NONE, base_offset=0, lbo_mode=0,
        lut_k_offset=0, wide=false) -> UInt64

Pack the `tcgen05` shared-memory descriptor from PTX 9.4 §9.7.18.4.1
Table 49. The matrix address and both byte fields must be 16-byte aligned and
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

`lut_k_offset` (bit 53) is 0 or 24: with `.decompress::lut::b` the
instruction reads a 48-byte K=128 block of compressed B and uses either its
first or its second 24-byte segment. `wide=true` widens the matrix-start
and leading-dimension fields to Table 49's 15 bits (a 19-bit aligned input
window); inputs that fit the 18-bit window encode identically either way.
No assembler or hardware evidence covers the widened fields — ptxas cannot
inspect a register operand — so the default keeps the 14-bit encoding.
"""
@inline function tcgen05_descriptor(
        smem_addr_u32::UInt32;
        leading_bytes::Integer,
        stride_bytes::Integer,
        swizzle::Integer = BlackwellLayout.NONE,
        base_offset::Integer = 0,
        lbo_mode::Integer = 0,
        lut_k_offset::Integer = 0,
        wide::Bool = false)
    swizzle in _TCGEN05_SWIZZLES ||
        throw(ArgumentError(
            "swizzle must be a BlackwellLayout encoding (0, 1, 2, 4, or 6)"))
    0 <= base_offset <= 0x7 ||
        throw(ArgumentError("base_offset must fit in 3 bits"))
    0 <= lbo_mode <= 0x1 ||
        throw(ArgumentError("lbo_mode must fit in 1 bit"))
    lut_k_offset in (0, 24) ||
        throw(ArgumentError("lut_k_offset must be 0 or 24"))
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
    addr = wide ? _tcgen05_descriptor_wide_field(smem_addr_u32) :
                  _tcgen05_descriptor_field(smem_addr_u32)
    ld   = (wide ? _tcgen05_descriptor_wide_field(leading_bytes) :
                   _tcgen05_descriptor_field(leading_bytes)) << 16
    sd   = _tcgen05_descriptor_field(stride_bytes) << 32
    bo   = (UInt64(base_offset) & 0x7) << 49
    lbo  = (UInt64(lbo_mode)    & 0x1) << 52
    lut  = UInt64(lut_k_offset == 24) << 53
    sw   = UInt64(swizzle) << 61
    addr | ld | sd | _TCGEN05_SDESC_FIXED | bo | lbo | lut | sw
end
