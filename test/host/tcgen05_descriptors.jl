# Independent PTX 9.3 §9.7.17.4 Tables 43/45 oracle. Keep these masks and
# encoders local to the test: importing production constants would let a
# shifted/reserved-field edit weaken its own test.

using PTX: BlackwellLayout, tcgen05_descriptor,
           tcgen05_instr_desc_f16bf16_f32

const _TD_SDESC_FIXED_MASK = UInt64(0x7) << 46
const _TD_SDESC_FIXED_VALUE = UInt64(0x1) << 46
const _TD_SDESC_ZERO_MASK =
    (UInt64(0x3) << 14) |       # reserved
    (UInt64(0x3) << 30) |       # reserved
    (UInt64(0x3) << 47) |       # zero bits of fixed 0b001
    (UInt64(0xff) << 53)        # fixed zero
const _TD_SDESC_VARIABLE_MASK =
    UInt64(0x3fff) |
    (UInt64(0x3fff) << 16) |
    (UInt64(0x3fff) << 32) |
    (UInt64(0x7) << 49) |
    (UInt64(0x1) << 52) |
    (UInt64(0x7) << 61)
const _TD_SDESC_ALLOWED_MASK = _TD_SDESC_FIXED_VALUE | _TD_SDESC_VARIABLE_MASK

const _TD_IDESC_ZERO_MASK =
    UInt32(0x3) |               # sparsity selector fixed at zero by this API
    (UInt32(0x1) << 3) |        # integer-only saturation / NA for float
    (UInt32(0x1) << 6) |        # reserved
    (UInt32(0x1) << 23) |       # reserved
    (UInt32(0x1) << 29)         # reserved

@inline _td_field(x, shift, width) = (x >> shift) & ((one(x) << width) - one(x))

function _td_expected_sdesc(addr, leading, stride, swizzle, base_offset, lbo_mode)
    UInt64(addr >> 4) |
    (UInt64(leading >> 4) << 16) |
    (UInt64(stride >> 4) << 32) |
    _TD_SDESC_FIXED_VALUE |
    (UInt64(base_offset) << 49) |
    (UInt64(lbo_mode) << 52) |
    (UInt64(swizzle) << 61)
end

function _td_expected_idesc(m, n, ab_dtype, a_major, b_major,
                            scale_a, scale_b, sparse, max_shift)
    ab_format = Dict(:f16 => UInt32(0), :bf16 => UInt32(1), :tf32 => UInt32(2))[ab_dtype]
    UInt32(sparse) << 2 |
    UInt32(1) << 4 |           # dtype = f32
    ab_format << 7 |
    ab_format << 10 |
    UInt32(scale_a == -1) << 13 |
    UInt32(scale_b == -1) << 14 |
    UInt32(a_major == :MN) << 15 |
    UInt32(b_major == :MN) << 16 |
    UInt32(n >> 3) << 17 |
    UInt32(m >> 4) << 24 |
    UInt32(max_shift) << 30
end

function _td_sdesc_invariants_hold(desc)
    desc & _TD_SDESC_FIXED_MASK == _TD_SDESC_FIXED_VALUE &&
        iszero(desc & _TD_SDESC_ZERO_MASK) &&
        iszero(desc & ~_TD_SDESC_ALLOWED_MASK)
end

function _td_assert_sdesc_invariants(desc)
    @test desc & _TD_SDESC_FIXED_MASK == _TD_SDESC_FIXED_VALUE
    @test iszero(desc & _TD_SDESC_ZERO_MASK)
    @test iszero(desc & ~_TD_SDESC_ALLOWED_MASK)
end

@testset "tcgen05 shared descriptor: exhaustive field windows" begin
    # Exercise every one of the 2^14 encodable values independently in each
    # address/offset field. This catches both truncation and a one-bit shift
    # into each neighboring reserved field. The sweep stays exhaustive but
    # reports in aggregate: one recorded mismatch per bad encoding, and a
    # final count proving the sweep actually swept, instead of half a million
    # per-value @test records for a 1.7s loop.
    mismatches = String[]
    checked = 0
    for encoded in UInt32(0):UInt32(0x3fff)
        bytes = encoded << 4
        cases = (
            ("addr", tcgen05_descriptor(bytes; leading_bytes = 0,
                                        stride_bytes = 0),
             _td_expected_sdesc(bytes, 0, 0, 0, 0, 0)),
            ("leading", tcgen05_descriptor(UInt32(0); leading_bytes = bytes,
                                           stride_bytes = 0),
             _td_expected_sdesc(0, bytes, 0, 0, 0, 0)),
            ("stride", tcgen05_descriptor(UInt32(0); leading_bytes = 0,
                                          stride_bytes = bytes),
             _td_expected_sdesc(0, 0, bytes, 0, 0, 0)),
        )
        for (field, desc, expected) in cases
            checked += 1
            desc == expected && _td_sdesc_invariants_hold(desc) && continue
            length(mismatches) < 16 && push!(mismatches,
                "$field=0x$(string(bytes, base = 16)): " *
                "got 0x$(string(desc, base = 16)), " *
                "expected 0x$(string(expected, base = 16))")
        end
    end
    isempty(mismatches) ||
        foreach(m -> println("SDESC MISMATCH: ", m), mismatches)
    @test isempty(mismatches)
    @test checked == 3 * 2^14
end

@testset "tcgen05 shared descriptor: legal control cross-product" begin
    swizzles = (BlackwellLayout.NONE, BlackwellLayout.B128_BASE32B,
                BlackwellLayout.B128, BlackwellLayout.B64,
                BlackwellLayout.B32)
    # Relative-offset mode admits every swizzle and base-offset encoding.
    for swizzle in swizzles, base_offset in 0:7
        desc = tcgen05_descriptor(UInt32(0x3fff0);
                                  leading_bytes = 0x12340,
                                  stride_bytes = 0x2abc0,
                                  swizzle, base_offset, lbo_mode = 0)
        @test desc == _td_expected_sdesc(0x3fff0, 0x12340, 0x2abc0,
                                         swizzle, base_offset, 0)
        _td_assert_sdesc_invariants(desc)
        @test _td_field(desc, 49, 3) == base_offset
        @test _td_field(desc, 52, 1) == 0
        @test _td_field(desc, 61, 3) == swizzle
    end

    # PTX §9.7.17.3.1.2 permits absolute-address mode only with the 128B,
    # 16B-atomic swizzle (code 2) and a zero base offset.
    absolute = tcgen05_descriptor(UInt32(0x3fff0);
                                  leading_bytes = 0x12340,
                                  stride_bytes = 0x2abc0,
                                  swizzle = BlackwellLayout.B128,
                                  base_offset = 0, lbo_mode = 1)
    @test absolute == _td_expected_sdesc(0x3fff0, 0x12340, 0x2abc0,
                                         BlackwellLayout.B128, 0, 1)
    _td_assert_sdesc_invariants(absolute)

    # Maintained CUTLASS/pyptx-compatible B128 descriptor.
    @test tcgen05_descriptor(UInt32(0); leading_bytes = 16,
                             stride_bytes = 1024,
                             swizzle = BlackwellLayout.B128) ==
          UInt64(0x4000404000010000)
end

@testset "tcgen05 shared descriptor: unsafe encodings rejected" begin
    # All three matrix-descriptor-encode inputs are explicitly 16B-aligned and
    # limited to the 18-bit input window instead of silently aliasing.
    for addr in (UInt32(1), UInt32(15), UInt32(17), UInt32(0x3ffff),
                 UInt32(0x40000), typemax(UInt32))
        @test_throws ArgumentError tcgen05_descriptor(
            addr; leading_bytes = 0, stride_bytes = 0)
    end
    for field in (-16, -1, 1, 15, 17, 0x3ffff, 0x40000, typemax(Int))
        @test_throws ArgumentError tcgen05_descriptor(
            UInt32(0); leading_bytes = field, stride_bytes = 0)
        @test_throws ArgumentError tcgen05_descriptor(
            UInt32(0); leading_bytes = 0, stride_bytes = field)
    end
    for swizzle in (-1, 3, 5, 7, 8, typemax(Int))
        @test_throws ArgumentError tcgen05_descriptor(
            UInt32(0); leading_bytes = 0, stride_bytes = 0, swizzle)
    end
    for base_offset in (-1, 8, typemax(Int))
        @test_throws ArgumentError tcgen05_descriptor(
            UInt32(0); leading_bytes = 0, stride_bytes = 0, base_offset)
    end
    for lbo_mode in (-1, 2, typemax(Int))
        @test_throws ArgumentError tcgen05_descriptor(
            UInt32(0); leading_bytes = 0, stride_bytes = 0, lbo_mode)
    end
    for swizzle in (BlackwellLayout.NONE, BlackwellLayout.B128_BASE32B,
                    BlackwellLayout.B64, BlackwellLayout.B32)
        @test_throws ArgumentError tcgen05_descriptor(
            UInt32(0); leading_bytes = 0, stride_bytes = 0,
            swizzle, base_offset = 0, lbo_mode = 1)
    end
    for base_offset in 1:7
        @test_throws ArgumentError tcgen05_descriptor(
            UInt32(0); leading_bytes = 0, stride_bytes = 0,
            swizzle = BlackwellLayout.B128, base_offset, lbo_mode = 1)
    end

    # Table 43's 0b001 is fixed, not a public version selector.
    for version in (0, 1, 2, 3)
        @test_throws MethodError tcgen05_descriptor(
            UInt32(0); leading_bytes = 0, stride_bytes = 0, version)
    end
end

@testset "tcgen05 float instruction descriptor: exhaustive public fields" begin
    # Full cross-product of every public keyword; aggregated reporting for
    # the same reason as the shared-descriptor sweep above.
    mismatches = String[]
    checked = 0
    for m in (32, 64, 128, 256), n in 8:8:256,
        ab_dtype in (:f16, :bf16, :tf32), a_major in (:K, :MN),
        b_major in (:K, :MN), scale_a in (1, -1), scale_b in (1, -1),
        sparse in (false, true), max_shift in 0:3

        desc = tcgen05_instr_desc_f16bf16_f32(
            ; m, n, ab_dtype, a_major, b_major, scale_a, scale_b, sparse,
            max_shift)
        expected = _td_expected_idesc(m, n, ab_dtype, a_major, b_major,
                                      scale_a, scale_b, sparse, max_shift)
        checked += 1
        ok = desc == expected &&
             iszero(desc & _TD_IDESC_ZERO_MASK) &&
             _td_field(desc, 4, 2) == 1 &&      # dtype = f32
             _td_field(desc, 7, 3) == _td_field(desc, 10, 3) &&
             _td_field(desc, 17, 6) == n >> 3 &&
             _td_field(desc, 24, 5) == m >> 4
        ok && continue
        length(mismatches) < 16 && push!(mismatches,
            "(m=$m n=$n $ab_dtype a=$a_major b=$b_major " *
            "sa=$scale_a sb=$scale_b sparse=$sparse shift=$max_shift): " *
            "got 0x$(string(desc, base = 16)), " *
            "expected 0x$(string(expected, base = 16))")
    end
    isempty(mismatches) ||
        foreach(m -> println("IDESC MISMATCH: ", m), mismatches)
    @test isempty(mismatches)
    @test checked == 4 * 32 * 3 * 2 * 2 * 2 * 2 * 2 * 4
end

@testset "tcgen05 float instruction descriptor: unsafe encodings rejected" begin
    valid = (; m = 128, n = 256, ab_dtype = :bf16)
    for m in (-1, 0, 31, 33, 63, 96, 257, typemax(Int))
        @test_throws ArgumentError tcgen05_instr_desc_f16bf16_f32(
            ; valid..., m)
    end
    for n in (-8, 0, 7, 12, 257, 264, typemax(Int))
        @test_throws ArgumentError tcgen05_instr_desc_f16bf16_f32(
            ; valid..., n)
    end
    for ab_dtype in (:f8, :f32, :s8)
        @test_throws ArgumentError tcgen05_instr_desc_f16bf16_f32(
            ; valid..., ab_dtype)
    end
    for major in (:M, :N, :T)
        @test_throws ArgumentError tcgen05_instr_desc_f16bf16_f32(
            ; valid..., a_major = major)
        @test_throws ArgumentError tcgen05_instr_desc_f16bf16_f32(
            ; valid..., b_major = major)
    end
    for scale in (-2, 0, 2, typemax(Int))
        @test_throws ArgumentError tcgen05_instr_desc_f16bf16_f32(
            ; valid..., scale_a = scale)
        @test_throws ArgumentError tcgen05_instr_desc_f16bf16_f32(
            ; valid..., scale_b = scale)
    end
    for max_shift in (-1, 4, typemax(Int))
        @test_throws ArgumentError tcgen05_instr_desc_f16bf16_f32(
            ; valid..., max_shift)
    end

    # Table 45 marks bit 3 as integer saturation / NA for float kinds. The
    # former keyword could produce an invalid floating-point descriptor.
    for saturate in (false, true)
        @test_throws MethodError tcgen05_instr_desc_f16bf16_f32(
            ; valid..., saturate)
    end
end

# ── PTX 9.4 §9.7.18.4.2 Tables 51–53: the non-f16 idesc kinds ────────────────
# Independent oracles, same double-entry rule as above. Hardware anchors: the
# two encodings below were runtime-validated on a B200 by the idesc probes
# (s8×s8→s32 all-ones K=32 → 32; e4m3×e4m3→f32 1.0×1.0 → 32.0).

using PTX: tcgen05_instr_desc_i8, tcgen05_instr_desc_f8f6f4,
           tcgen05_instr_desc_mxf8f6f4, tcgen05_instr_desc_mxf4,
           tcgen05_instr_desc_mxf4nvf4

const _TD_I8_FMT = Dict(:u8 => UInt32(0), :s8 => UInt32(1))
const _TD_F864_FMT = Dict(:e4m3 => UInt32(0), :e5m2 => UInt32(1),
                          :e2m3 => UInt32(3), :e3m2 => UInt32(4),
                          :e2m1 => UInt32(5))
const _TD_F864_DFMT = Dict(:f16 => UInt32(0), :f32 => UInt32(1))
const _TD_NVF4_SFMT = Dict(:ue4m3 => UInt32(0), :ue8m0 => UInt32(1),
                           :ue5m3 => UInt32(2))

# Zero masks per table: sparsity selector (Table 51 kinds), every reserved
# bit, and every architecture-gated encoding the builders pin to zero.
const _TD_I8_ZERO_MASK = UInt32(0x3) |                  # sparsity selector
    (UInt32(0x3) << 13) |                               # negate: unsupported
    (UInt32(0x1) << 6) | (UInt32(0x1) << 23) |          # reserved
    (UInt32(0x1) << 29)                                 # gated K encoding
const _TD_F864_ZERO_MASK = UInt32(0x3) |
    (UInt32(0x1) << 3) |                                # saturate: NA
    (UInt32(0x1) << 6) | (UInt32(0x1) << 23) |
    (UInt32(0x1) << 29)
const _TD_MXF864_ZERO_MASK = UInt32(0x3) |
    (UInt32(0x1) << 3) | (UInt32(0x1) << 6) |           # reserved
    (UInt32(0x3) << 24) |                               # reserved
    (UInt32(0x1) << 26) |                               # gated scale layout
    (UInt32(0x1) << 31)                                 # gated K encoding
const _TD_MXF4_ZERO_MASK = UInt32(0x3) |
    (UInt32(0x1) << 3) |                                # gated K upper bit
    (UInt32(0x1) << 6) | (UInt32(0x1) << 25) |          # reserved
    (UInt32(0x3) << 15) |                               # transpose: unsupported
    (UInt32(0x1) << 26) |                               # gated scale layout
    (UInt32(0x1) << 31)                                 # gated K lower bit

function _td_expected_i8(m, n, a, b, amaj, bmaj, sat, sparse, shift)
    UInt32(sparse) << 2 | UInt32(sat) << 3 | UInt32(2) << 4 |
    _TD_I8_FMT[a] << 7 | _TD_I8_FMT[b] << 10 |
    UInt32(amaj == :MN) << 15 | UInt32(bmaj == :MN) << 16 |
    UInt32(n >> 3) << 17 | UInt32(m >> 4) << 24 | UInt32(shift) << 30
end

function _td_expected_f864(m, n, a, b, d, amaj, bmaj, sa, sb, sparse, shift)
    UInt32(sparse) << 2 | _TD_F864_DFMT[d] << 4 |
    _TD_F864_FMT[a] << 7 | _TD_F864_FMT[b] << 10 |
    UInt32(sa == -1) << 13 | UInt32(sb == -1) << 14 |
    UInt32(amaj == :MN) << 15 | UInt32(bmaj == :MN) << 16 |
    UInt32(n >> 3) << 17 | UInt32(m >> 4) << 24 | UInt32(shift) << 30
end

function _td_expected_mxf864(m, n, a, b, said, sbid, amaj, bmaj, sa, sb, sparse)
    UInt32(sparse) << 2 | UInt32(sbid) << 4 |
    _TD_F864_FMT[a] << 7 | _TD_F864_FMT[b] << 10 |
    UInt32(sa == -1) << 13 | UInt32(sb == -1) << 14 |
    UInt32(amaj == :MN) << 15 | UInt32(bmaj == :MN) << 16 |
    UInt32(n >> 3) << 17 | UInt32(1) << 23 | UInt32(m >> 7) << 27 |
    UInt32(said) << 29
end

function _td_expected_mxf4(m, n, sfmt, said, sbid, sa, sb, sparse, spv)
    UInt32(sparse) << 2 | UInt32(sbid) << 4 |
    UInt32(1) << 7 | UInt32(1) << 10 | UInt32(spv) << 12 |
    UInt32(sa == -1) << 13 | UInt32(sb == -1) << 14 |
    UInt32(n >> 3) << 17 | sfmt << 23 | UInt32(m >> 7) << 27 |
    UInt32(said) << 29
end

@testset "tcgen05 idesc: hardware-anchored encodings" begin
    # Exact values proven on silicon by the idesc probes (B200, sm_100a).
    @test tcgen05_instr_desc_i8(; m = 128, n = 256,
                                a_dtype = :s8, b_dtype = :s8) == 0x084004A0
    @test tcgen05_instr_desc_f8f6f4(; m = 128, n = 256,
                                    a_dtype = :e4m3, b_dtype = :e4m3,
                                    d_dtype = :f32) == 0x08400010
end

@testset "tcgen05 i8 idesc: exhaustive public fields" begin
    mismatches = String[]
    checked = 0
    for m in (32, 64, 128, 256), n in 8:8:256,
        a in (:u8, :s8), b in (:u8, :s8), amaj in (:K, :MN), bmaj in (:K, :MN),
        sat in (false, true), sparse in (false, true), shift in 0:3

        desc = tcgen05_instr_desc_i8(; m, n, a_dtype = a, b_dtype = b,
                                     a_major = amaj, b_major = bmaj,
                                     saturate = sat, sparse, max_shift = shift)
        expected = _td_expected_i8(m, n, a, b, amaj, bmaj, sat, sparse, shift)
        checked += 1
        ok = desc == expected && iszero(desc & _TD_I8_ZERO_MASK) &&
             _td_field(desc, 4, 2) == 2        # dtype = s32, always
        ok && continue
        length(mismatches) < 16 && push!(mismatches,
            "(m=$m n=$n $a×$b amaj=$amaj bmaj=$bmaj sat=$sat " *
            "sparse=$sparse shift=$shift): got 0x$(string(desc, base = 16))")
    end
    isempty(mismatches) ||
        foreach(x -> println("I8 IDESC MISMATCH: ", x), mismatches)
    @test isempty(mismatches)
    @test checked == 4 * 32 * 2 * 2 * 2 * 2 * 2 * 2 * 4
end

@testset "tcgen05 f8f6f4 idesc: exhaustive public fields" begin
    mismatches = String[]
    checked = 0
    for m in (32, 64, 128, 256), n in 8:8:256,
        a in (:e4m3, :e5m2, :e2m3, :e3m2, :e2m1), b in (:e4m3, :e2m1),
        d in (:f16, :f32), amaj in (:K, :MN), bmaj in (:K, :MN),
        sa in (1, -1), sb in (1, -1), sparse in (false, true), shift in 0:3

        desc = tcgen05_instr_desc_f8f6f4(; m, n, a_dtype = a, b_dtype = b,
                                         d_dtype = d, a_major = amaj,
                                         b_major = bmaj, scale_a = sa,
                                         scale_b = sb, sparse,
                                         max_shift = shift)
        expected = _td_expected_f864(m, n, a, b, d, amaj, bmaj, sa, sb,
                                     sparse, shift)
        checked += 1
        ok = desc == expected && iszero(desc & _TD_F864_ZERO_MASK)
        ok && continue
        length(mismatches) < 16 && push!(mismatches,
            "(m=$m n=$n $a×$b→$d amaj=$amaj bmaj=$bmaj sa=$sa sb=$sb " *
            "sparse=$sparse shift=$shift): got 0x$(string(desc, base = 16))")
    end
    isempty(mismatches) ||
        foreach(x -> println("F8F6F4 IDESC MISMATCH: ", x), mismatches)
    @test isempty(mismatches)
    @test checked == 4 * 32 * 5 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 4
end

@testset "tcgen05 mxf8f6f4 idesc: exhaustive public fields" begin
    mismatches = String[]
    checked = 0
    for m in (128, 256), n in 8:8:256,
        a in (:e4m3, :e5m2, :e2m3, :e3m2, :e2m1), b in (:e4m3, :e2m1),
        said in 0:3, sbid in 0:3, amaj in (:K, :MN), bmaj in (:K, :MN),
        sa in (1, -1), sb in (1, -1), sparse in (false, true)

        desc = tcgen05_instr_desc_mxf8f6f4(; m, n, a_dtype = a, b_dtype = b,
                                           scale_a_id = said,
                                           scale_b_id = sbid,
                                           a_major = amaj, b_major = bmaj,
                                           scale_a = sa, scale_b = sb, sparse)
        expected = _td_expected_mxf864(m, n, a, b, said, sbid, amaj, bmaj,
                                       sa, sb, sparse)
        checked += 1
        ok = desc == expected && iszero(desc & _TD_MXF864_ZERO_MASK) &&
             _td_field(desc, 23, 1) == 1       # scale type = UE8M0, always
        ok && continue
        length(mismatches) < 16 && push!(mismatches,
            "(m=$m n=$n $a×$b ids=$said/$sbid amaj=$amaj bmaj=$bmaj " *
            "sa=$sa sb=$sb sparse=$sparse): got 0x$(string(desc, base = 16))")
    end
    isempty(mismatches) ||
        foreach(x -> println("MXF8F6F4 IDESC MISMATCH: ", x), mismatches)
    @test isempty(mismatches)
    @test checked == 2 * 32 * 5 * 2 * 4 * 4 * 2 * 2 * 2 * 2 * 2
end

@testset "tcgen05 mxf4/mxf4nvf4 idesc: exhaustive public fields" begin
    mismatches = String[]
    checked = 0
    for m in (128, 256), n in 8:8:256, said in (0, 2), sbid in (0, 2),
        sa in (1, -1), sb in (1, -1), sparse in (false, true), spv in (0, 1)

        mx = tcgen05_instr_desc_mxf4(; m, n, scale_a_id = said,
                                     scale_b_id = sbid, scale_a = sa,
                                     scale_b = sb, sparse,
                                     sparsity_version = spv)
        ok = mx == _td_expected_mxf4(m, n, UInt32(1), said, sbid, sa, sb,
                                     sparse, spv) &&
             iszero(mx & _TD_MXF4_ZERO_MASK)
        checked += 1
        ok || length(mismatches) >= 16 || push!(mismatches,
            "mxf4(m=$m n=$n ids=$said/$sbid sa=$sa sb=$sb sparse=$sparse " *
            "spv=$spv): got 0x$(string(mx, base = 16))")

        for sfmt in (:ue4m3, :ue8m0, :ue5m3)
            nv = tcgen05_instr_desc_mxf4nvf4(; m, n, scale_dtype = sfmt,
                                             scale_a_id = said,
                                             scale_b_id = sbid, scale_a = sa,
                                             scale_b = sb, sparse,
                                             sparsity_version = spv)
            nvok = nv == _td_expected_mxf4(m, n, _TD_NVF4_SFMT[sfmt], said,
                                           sbid, sa, sb, sparse, spv) &&
                   iszero(nv & (_TD_MXF4_ZERO_MASK & ~(UInt32(0x3) << 23)))
            checked += 1
            nvok || length(mismatches) >= 16 || push!(mismatches,
                "mxf4nvf4(m=$m n=$n $sfmt ids=$said/$sbid sa=$sa sb=$sb " *
                "sparse=$sparse spv=$spv): got 0x$(string(nv, base = 16))")
        end
    end
    isempty(mismatches) ||
        foreach(x -> println("MXF4 IDESC MISMATCH: ", x), mismatches)
    @test isempty(mismatches)
    @test checked == 2 * 32 * 2 * 2 * 2 * 2 * 2 * 2 * 4
end

@testset "tcgen05 non-f16 idesc: unsafe encodings rejected" begin
    i8 = (; m = 128, n = 256, a_dtype = :s8, b_dtype = :s8)
    f864 = (; m = 128, n = 256, a_dtype = :e4m3, b_dtype = :e4m3)
    mx864 = (; m = 128, n = 256, a_dtype = :e4m3, b_dtype = :e4m3,
             scale_a_id = 0, scale_b_id = 0)
    mxf4 = (; m = 128, n = 256, scale_a_id = 0, scale_b_id = 0)
    nvf4 = (; mxf4..., scale_dtype = :ue4m3)

    for m in (-1, 0, 31, 96, 257, typemax(Int))
        @test_throws ArgumentError tcgen05_instr_desc_i8(; i8..., m)
        @test_throws ArgumentError tcgen05_instr_desc_f8f6f4(; f864..., m)
    end
    # Block-scale M is the narrower M>>7 window: 32/64 must NOT be accepted.
    for m in (-1, 0, 32, 64, 96, 384, 512, typemax(Int))
        @test_throws ArgumentError tcgen05_instr_desc_mxf8f6f4(; mx864..., m)
        @test_throws ArgumentError tcgen05_instr_desc_mxf4(; mxf4..., m)
        @test_throws ArgumentError tcgen05_instr_desc_mxf4nvf4(; nvf4..., m)
    end
    for n in (-8, 0, 7, 12, 257, typemax(Int))
        @test_throws ArgumentError tcgen05_instr_desc_i8(; i8..., n)
        @test_throws ArgumentError tcgen05_instr_desc_f8f6f4(; f864..., n)
        @test_throws ArgumentError tcgen05_instr_desc_mxf8f6f4(; mx864..., n)
        @test_throws ArgumentError tcgen05_instr_desc_mxf4(; mxf4..., n)
    end
    for dt in (:s4, :f16, :e4m3, :i8)
        @test_throws ArgumentError tcgen05_instr_desc_i8(; i8..., a_dtype = dt)
        @test_throws ArgumentError tcgen05_instr_desc_i8(; i8..., b_dtype = dt)
    end
    for dt in (:u8, :f16, :ue8m0, :e8m0)
        @test_throws ArgumentError tcgen05_instr_desc_f8f6f4(
            ; f864..., a_dtype = dt)
        @test_throws ArgumentError tcgen05_instr_desc_mxf8f6f4(
            ; mx864..., b_dtype = dt)
    end
    for d in (:s32, :bf16, :f64)
        @test_throws ArgumentError tcgen05_instr_desc_f8f6f4(
            ; f864..., d_dtype = d)
    end
    for sfmt in (:e4m3, :ue2m1, :f32)
        @test_throws ArgumentError tcgen05_instr_desc_mxf4nvf4(
            ; mxf4..., scale_dtype = sfmt)
    end
    for id in (-1, 4, typemax(Int))
        @test_throws ArgumentError tcgen05_instr_desc_mxf8f6f4(
            ; mx864..., scale_a_id = id)
        @test_throws ArgumentError tcgen05_instr_desc_mxf8f6f4(
            ; mx864..., scale_b_id = id)
    end
    # Table 53 restricts the mxf4-kind scale-factor IDs to {0, 2}.
    for id in (-1, 1, 3, 4, typemax(Int))
        @test_throws ArgumentError tcgen05_instr_desc_mxf4(
            ; mxf4..., scale_a_id = id)
        @test_throws ArgumentError tcgen05_instr_desc_mxf4nvf4(
            ; nvf4..., scale_b_id = id)
    end
    for spv in (-1, 2, typemax(Int))
        @test_throws ArgumentError tcgen05_instr_desc_mxf4(
            ; mxf4..., sparsity_version = spv)
    end

    # §9.7.18.10 Table 62: negate is unsupported for .kind::i8 — the keyword
    # must not exist, so an invalid descriptor is unrepresentable.
    for kw in ((; scale_a = -1), (; scale_b = -1))
        @test_throws MethodError tcgen05_instr_desc_i8(; i8..., kw...)
    end
    # Saturation is integer-only: absent from every float-kind builder.
    @test_throws MethodError tcgen05_instr_desc_f8f6f4(
        ; f864..., saturate = true)
    @test_throws MethodError tcgen05_instr_desc_mxf8f6f4(
        ; mx864..., saturate = true)
    # Table 62: transpose is unsupported for the mxf4 kinds.
    for kw in ((; a_major = :MN), (; b_major = :MN))
        @test_throws MethodError tcgen05_instr_desc_mxf4(; mxf4..., kw...)
        @test_throws MethodError tcgen05_instr_desc_mxf4nvf4(; nvf4..., kw...)
    end
    # max_shift is a Table 51 field only (.ws); absent from block-scale kinds.
    @test_throws MethodError tcgen05_instr_desc_mxf8f6f4(
        ; mx864..., max_shift = 1)
    @test_throws MethodError tcgen05_instr_desc_mxf4(; mxf4..., max_shift = 1)
end
