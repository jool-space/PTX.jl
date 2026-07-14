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
