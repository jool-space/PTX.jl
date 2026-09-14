# Independent PTX ISA 9.4 §9.7.18.4 oracle for the architecture-gated
# descriptor encodings: the 15-bit shared-descriptor fields and the
# lut::b segment bit (Table 49), the wide-K and scale-layout bits (Tables
# 51–53), and the `.kind::ti16` instruction descriptor. Masks and encoders
# stay local so a shifted or reserved-field edit cannot weaken its own
# test; every default keeps the pre-9.4 encoding, which the sweeps in
# tcgen05_descriptors.jl already pin.

using PTX: BlackwellLayout, tcgen05_descriptor, tcgen05_instr_desc_i8,
           tcgen05_instr_desc_f8f6f4, tcgen05_instr_desc_mxf8f6f4,
           tcgen05_instr_desc_mxf4, tcgen05_instr_desc_mxf4nvf4,
           tcgen05_instr_desc_ti16

@inline _t94_field(x, shift, width) =
    (x >> shift) & ((one(x) << width) - one(x))

@testset "tcgen05 shared descriptor: 15-bit fields and lut segment bit" begin
    base(addr, leading) = UInt64(addr >> 4) | (UInt64(leading >> 4) << 16) |
                          (UInt64(1) << 46)
    # Every encodable 15-bit value in the address and leading fields; the
    # low 14 bits encode identically to the narrow builder.
    mismatches = 0
    checked = 0
    for encoded in UInt32(0):UInt32(0x7fff)
        bytes = encoded << 4
        wide_addr = tcgen05_descriptor(bytes; leading_bytes = 0,
                                       stride_bytes = 0, wide = true)
        wide_lead = tcgen05_descriptor(UInt32(0); leading_bytes = bytes,
                                       stride_bytes = 0, wide = true)
        checked += 2
        wide_addr == base(bytes, 0) || (mismatches += 1)
        wide_lead == base(0, bytes) || (mismatches += 1)
        if encoded <= 0x3fff
            checked += 2
            wide_addr == tcgen05_descriptor(bytes; leading_bytes = 0,
                                            stride_bytes = 0) ||
                (mismatches += 1)
            wide_lead == tcgen05_descriptor(UInt32(0); leading_bytes = bytes,
                                            stride_bytes = 0) ||
                (mismatches += 1)
        else
            @test_throws ArgumentError tcgen05_descriptor(
                bytes; leading_bytes = 0, stride_bytes = 0)
            @test_throws ArgumentError tcgen05_descriptor(
                UInt32(0); leading_bytes = bytes, stride_bytes = 0)
        end
    end
    @test mismatches == 0
    @test checked == 2 * 2^15 + 2 * 2^14
    # Bit 15 and bit 31 stay reserved; the stride field stays 14 bits.
    top = tcgen05_descriptor(UInt32(0x7fff0); leading_bytes = 0x7fff0,
                             stride_bytes = 0x3fff0, wide = true)
    @test _t94_field(top, 0, 15) == 0x7fff
    @test _t94_field(top, 15, 1) == 0
    @test _t94_field(top, 16, 15) == 0x7fff
    @test _t94_field(top, 31, 1) == 0
    @test _t94_field(top, 32, 14) == 0x3fff
    for bytes in (0x80000, 0x7ffff, 0x7fff1)
        @test_throws ArgumentError tcgen05_descriptor(
            UInt32(bytes); leading_bytes = 0, stride_bytes = 0, wide = true)
    end
    @test_throws ArgumentError tcgen05_descriptor(
        UInt32(0); leading_bytes = 0, stride_bytes = 0x40000, wide = true)

    # lut_k_offset: 0 leaves the descriptor unchanged, 24 sets bit 53 only.
    plain = tcgen05_descriptor(UInt32(0x100); leading_bytes = 16,
                               stride_bytes = 1024,
                               swizzle = BlackwellLayout.B128)
    @test tcgen05_descriptor(UInt32(0x100); leading_bytes = 16,
                             stride_bytes = 1024,
                             swizzle = BlackwellLayout.B128,
                             lut_k_offset = 0) == plain
    @test tcgen05_descriptor(UInt32(0x100); leading_bytes = 16,
                             stride_bytes = 1024,
                             swizzle = BlackwellLayout.B128,
                             lut_k_offset = 24) == plain | (UInt64(1) << 53)
    for off in (-24, 1, 16, 48, typemax(Int))
        @test_throws ArgumentError tcgen05_descriptor(
            UInt32(0); leading_bytes = 0, stride_bytes = 0,
            lut_k_offset = off)
    end
end

@testset "tcgen05 f8f6f4 idesc: wide K and lut::b rules" begin
    args = (; m = 128, n = 256, a_dtype = :e4m3, b_dtype = :e4m3)
    plain = tcgen05_instr_desc_f8f6f4(; args...)
    @test tcgen05_instr_desc_f8f6f4(; args..., k = 32) == plain
    @test tcgen05_instr_desc_f8f6f4(; args..., k = 64) ==
          plain | (UInt32(1) << 29)
    sparse = tcgen05_instr_desc_f8f6f4(; args..., sparse = true)
    @test tcgen05_instr_desc_f8f6f4(; args..., sparse = true, k = 64) == sparse
    for k in (16, 48, 96, 128, 0, -64)
        @test_throws ArgumentError tcgen05_instr_desc_f8f6f4(; args..., k)
    end
    @test_throws ArgumentError tcgen05_instr_desc_f8f6f4(
        ; args..., sparse = true, k = 128)
    # Table 62: no transpose of the 6-/4-bit types at dense K=64.
    for dt in (:e2m3, :e3m2, :e2m1)
        @test_throws ArgumentError tcgen05_instr_desc_f8f6f4(
            ; args..., a_dtype = dt, a_major = :MN, k = 64)
        @test_throws ArgumentError tcgen05_instr_desc_f8f6f4(
            ; args..., b_dtype = dt, b_major = :MN, k = 64)
        # ... but fine at the base K, and fine K-major at K=64.
        @test tcgen05_instr_desc_f8f6f4(; args..., a_dtype = dt,
                                        a_major = :MN) ==
              tcgen05_instr_desc_f8f6f4(; args..., a_dtype = dt,
                                        a_major = :MN, k = 32)
        @test _t94_field(tcgen05_instr_desc_f8f6f4(; args..., a_dtype = dt,
                                                   k = 64), 29, 1) == 1
    end
    @test _t94_field(tcgen05_instr_desc_f8f6f4(; args..., a_major = :MN,
                                               b_major = :MN, k = 64),
                     29, 1) == 1
    # lut::b: E4M3, K-major B; no bit changes.
    @test tcgen05_instr_desc_f8f6f4(; args..., lut_b = true) == plain
    @test_throws ArgumentError tcgen05_instr_desc_f8f6f4(
        ; args..., b_dtype = :e5m2, lut_b = true)
    @test_throws ArgumentError tcgen05_instr_desc_f8f6f4(
        ; args..., b_major = :MN, lut_b = true)
    @test tcgen05_instr_desc_f8f6f4(; args..., a_dtype = :e2m1,
                                    a_major = :MN, lut_b = true) ==
          tcgen05_instr_desc_f8f6f4(; args..., a_dtype = :e2m1, a_major = :MN)
end

@testset "tcgen05 mxf8f6f4 idesc: wide K, scale layout, lut::b rules" begin
    args = (; m = 128, n = 256, a_dtype = :e4m3, b_dtype = :e4m3,
            scale_a_id = 0, scale_b_id = 0)
    plain = tcgen05_instr_desc_mxf8f6f4(; args...)
    @test tcgen05_instr_desc_mxf8f6f4(; args..., k = 32) == plain
    @test tcgen05_instr_desc_mxf8f6f4(; args..., k = 64) ==
          plain | (UInt32(1) << 31)
    sparse = tcgen05_instr_desc_mxf8f6f4(; args..., sparse = true)
    @test tcgen05_instr_desc_mxf8f6f4(; args..., sparse = true, k = 64) == sparse
    @test tcgen05_instr_desc_mxf8f6f4(; args..., sparse = true, k = 128) ==
          sparse | (UInt32(1) << 31)
    for k in (16, 48, 96, 128)
        @test_throws ArgumentError tcgen05_instr_desc_mxf8f6f4(; args..., k)
    end
    for k in (32, 96, 256)
        @test_throws ArgumentError tcgen05_instr_desc_mxf8f6f4(
            ; args..., sparse = true, k)
    end
    @test tcgen05_instr_desc_mxf8f6f4(; args..., scale_a_layout = 32) == plain
    @test tcgen05_instr_desc_mxf8f6f4(; args..., scale_a_layout = 128) ==
          plain | (UInt32(1) << 26)
    for layout in (0, 16, 64, 256)
        @test_throws ArgumentError tcgen05_instr_desc_mxf8f6f4(
            ; args..., scale_a_layout = layout)
    end
    for dt in (:e2m3, :e3m2, :e2m1)
        @test_throws ArgumentError tcgen05_instr_desc_mxf8f6f4(
            ; args..., a_dtype = dt, a_major = :MN, k = 64)
        # Sparse wide K carries no transpose rule in Table 62.
        @test _t94_field(tcgen05_instr_desc_mxf8f6f4(
            ; args..., a_dtype = dt, a_major = :MN, sparse = true, k = 128),
            31, 1) == 1
    end
    @test tcgen05_instr_desc_mxf8f6f4(; args..., lut_b = true) == plain
    @test_throws ArgumentError tcgen05_instr_desc_mxf8f6f4(
        ; args..., b_dtype = :e2m1, lut_b = true)
    @test_throws ArgumentError tcgen05_instr_desc_mxf8f6f4(
        ; args..., b_major = :MN, lut_b = true)
end

@testset "tcgen05 mxf4/mxf4nvf4 idesc: K code and scale layout" begin
    mxf4 = (; m = 128, n = 256, scale_a_id = 0, scale_b_id = 0)
    nvf4 = (; mxf4..., scale_dtype = :ue5m3)
    plain = tcgen05_instr_desc_mxf4(; mxf4...)
    nplain = tcgen05_instr_desc_mxf4nvf4(; nvf4...)
    bit3 = UInt32(1) << 3
    bit31 = UInt32(1) << 31
    for (k, code) in ((64, UInt32(0)), (96, bit31), (128, bit3))
        @test tcgen05_instr_desc_mxf4(; mxf4..., k) == plain | code
        @test tcgen05_instr_desc_mxf4nvf4(; nvf4..., k) == nplain | code
    end
    sparse = tcgen05_instr_desc_mxf4(; mxf4..., sparse = true)
    for (k, code) in ((128, UInt32(0)), (192, bit31))
        @test tcgen05_instr_desc_mxf4(; mxf4..., sparse = true, k) ==
              sparse | code
    end
    for k in (32, 48, 192, 256)
        @test_throws ArgumentError tcgen05_instr_desc_mxf4(; mxf4..., k)
        @test_throws ArgumentError tcgen05_instr_desc_mxf4nvf4(; nvf4..., k)
    end
    for k in (64, 96, 256)
        @test_throws ArgumentError tcgen05_instr_desc_mxf4(
            ; mxf4..., sparse = true, k)
    end
    @test tcgen05_instr_desc_mxf4(; mxf4..., scale_a_layout = 128) ==
          plain | (UInt32(1) << 26)
    @test tcgen05_instr_desc_mxf4nvf4(; nvf4..., scale_a_layout = 128) ==
          nplain | (UInt32(1) << 26)
    @test_throws ArgumentError tcgen05_instr_desc_mxf4(
        ; mxf4..., scale_a_layout = 64)
    # The combined encoding keeps every other field.
    both = tcgen05_instr_desc_mxf4nvf4(; nvf4..., k = 128, scale_a_layout = 128,
                                       sparsity_version = 1)
    @test _t94_field(both, 3, 1) == 1
    @test _t94_field(both, 12, 1) == 1
    @test _t94_field(both, 23, 2) == 2
    @test _t94_field(both, 26, 1) == 1
    @test _t94_field(both, 31, 1) == 0
end

# Table 51 zero mask for .kind::ti16: sparsity selector, saturate (NA),
# reserved bits 6 and 23, transpose (pinned), bit 29 (no defined value).
const _T94_TI16_ZERO_MASK = UInt32(0x3) | (UInt32(0x1) << 3) |
    (UInt32(0x1) << 6) | (UInt32(0x3) << 15) | (UInt32(0x1) << 23) |
    (UInt32(0x1) << 29)

_t94_expected_ti16(m, n, sa, sb, sparse, shift) =
    UInt32(sparse) << 2 | UInt32(2) << 4 | UInt32(3) << 7 | UInt32(3) << 10 |
    UInt32(sa == -1) << 13 | UInt32(sb == -1) << 14 |
    UInt32(n >> 3) << 17 | UInt32(m >> 4) << 24 | UInt32(shift) << 30

@testset "tcgen05 ti16 idesc: exhaustive public fields" begin
    # Derived anchor (Table 51 by hand, not hardware): m=128, n=256.
    @test tcgen05_instr_desc_ti16(; m = 128, n = 256) == 0x08400DA0
    mismatches = 0
    checked = 0
    for m in (32, 64, 128, 256), n in 8:8:256, sa in (1, -1), sb in (1, -1),
            sparse in (false, true), shift in 0:3
        desc = tcgen05_instr_desc_ti16(; m, n, scale_a = sa, scale_b = sb,
                                       sparse, max_shift = shift)
        checked += 1
        desc == _t94_expected_ti16(m, n, sa, sb, sparse, shift) &&
            iszero(desc & _T94_TI16_ZERO_MASK) &&
            _t94_field(desc, 4, 2) == 2 &&
            _t94_field(desc, 7, 3) == 3 && _t94_field(desc, 10, 3) == 3 ||
            (mismatches += 1)
    end
    @test mismatches == 0
    @test checked == 4 * 32 * 2 * 2 * 2 * 4
    # The ISA's descriptor table and transpose table disagree for ti16, so
    # transpose is unrepresentable; saturation is NA for this kind.
    for kw in ((; a_major = :MN), (; b_major = :MN), (; saturate = true),
               (; a_dtype = :s1z4m11))
        @test_throws MethodError tcgen05_instr_desc_ti16(; m = 128, n = 256,
                                                         kw...)
    end
    for m in (-1, 0, 31, 96, 257), n in (0, 7, 257)
        @test_throws ArgumentError tcgen05_instr_desc_ti16(; m, n)
    end
    for scale in (-2, 0, 2)
        @test_throws ArgumentError tcgen05_instr_desc_ti16(
            ; m = 128, n = 256, scale_a = scale)
    end
    @test_throws ArgumentError tcgen05_instr_desc_ti16(
        ; m = 128, n = 256, max_shift = 4)
end
