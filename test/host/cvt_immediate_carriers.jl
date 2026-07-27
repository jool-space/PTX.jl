# Independent test-side reconstruction of the ordinary-cvt source ABI from
# PTX ISA 9.3 §9.7.9.22. The expected keys and operand roles are not derived
# from ORDINARY_CVT_SOURCE_SCHEMAS, so a production-ledger edit cannot narrow
# its own oracle.

const _EXPECTED_CVT_SOURCE_CARRIERS = Dict(
    :u8 => :u8, :u16 => :u16, :u32 => :u32, :u64 => :u64,
    :s8 => :s8, :s16 => :s16, :s32 => :s32, :s64 => :s64,
    :f16 => :f16, :f32 => :f32, :f64 => :f64, :bf16 => :b16,
    :f16x2 => :b32, :bf16x2 => :b32,
    :e4m3x2 => :b16, :e5m2x2 => :b16,
    # PTX's e2m1x2 source is b8. PTX.jl bridges it through b16 because
    # LLVM NVPTX has no i8 inline-assembly constraint.
    :e2m1x2 => :b16,
    :e2m3x2 => :b16, :e3m2x2 => :b16,
    :ue8m0x2 => :b16, :s2f6x2 => :b16,
)

@testset "PTX integer constant evaluator uses fixed s64/u64 semantics" begin
    cases = (
        "0377" => Int64(255),
        "(WARP_SZ >> 1)" => Int64(16),
        "(0xffffffff << 32)" => Int64(-4294967296),
        "(0xffffffffffffffff + 2)" => UInt64(1),
        "(0x7fffffffffffffff + 1)" => typemin(Int64),
        "(((.u64) -8 >> 60) & 0xff)" => UInt64(15),
        # §4.5.5 says usual conversions; Table 5 conflicts, but ptxas 12.8
        # produces a signed bitwise result and therefore an arithmetic shift.
        "(((-1) & (-1)) >> 63)" => Int64(-1),
        "(-1 % 16)" => Int64(15),
    )
    for (text, expected) in cases
        @test PTX.Codegen._ptx_integer_constant(text) === expected
    end
    for bad in ("18446744073709551616", "0x10000000000000000",
                "42U", "0b1010", "1 ^ 2", "1 ? 2 : 3",
                "0 && (1 / 0)", "1 || (1 / 0)")
        @test_throws ArgumentError PTX.Codegen._ptx_integer_constant(bad)
    end
end

function _expected_ordinary_cvt_source_schemas()
    expected = Dict{Tuple{Symbol,Symbol,Bool,Bool},NamedTuple}()
    add!(destination, source, operands;
         stochastic = false, scaled = false, vector_source = false) = begin
        key = (destination, source, stochastic, scaled)
        @assert !haskey(expected, key)
        expected[key] = (; operands, vector_source)
    end

    fundamental =
        (:u8, :u16, :u32, :u64, :s8, :s16, :s32, :s64,
         :bf16, :f16, :f32, :f64)
    for destination in fundamental, source in fundamental
        add!(destination, source, (_EXPECTED_CVT_SOURCE_CARRIERS[source],))
    end

    # Two f32 inputs pack into the two 16-bit formats. Stochastic rounding
    # adds one b32 register carrying random bits.
    for destination in (:f16x2, :bf16x2)
        add!(destination, :f32, (:f32, :f32))
        add!(destination, :f32, (:f32, :f32, :b32); stochastic = true)
    end
    add!(:tf32, :f32, (:f32,))

    # FP8, FP4, and FP6 x2 down/up-conversion cross-products.
    narrow_x2 = (:e4m3x2, :e5m2x2, :e2m1x2, :e2m3x2, :e3m2x2)
    for destination in narrow_x2
        add!(destination, :f32, (:f32, :f32))
        for source in (:f16x2, :bf16x2)
            add!(destination, source,
                 (_EXPECTED_CVT_SOURCE_CARRIERS[source],))
        end
        for destination16 in (:f16x2, :bf16x2)
            add!(destination16, destination,
                 (_EXPECTED_CVT_SOURCE_CARRIERS[destination],))
        end
    end

    add!(:ue8m0x2, :f32, (:f32, :f32))
    add!(:ue8m0x2, :bf16x2, (:b32,))
    add!(:bf16x2, :ue8m0x2, (:b16,))

    add!(:s2f6x2, :f32, (:f32, :f32))
    add!(:s2f6x2, :bf16x2, (:b32,))
    add!(:bf16x2, :s2f6x2, (:b16,))

    # Optional packed ue8m0 scaling is a trailing b16 source role.
    for source in (:e4m3x2, :e5m2x2, :e2m1x2,
                   :e2m3x2, :e3m2x2, :s2f6x2)
        add!(:bf16x2, source,
             (_EXPECTED_CVT_SOURCE_CARRIERS[source], :b16); scaled = true)
    end
    add!(:s2f6x2, :f32, (:f32, :f32, :b16); scaled = true)
    add!(:s2f6x2, :bf16x2, (:b32, :b16); scaled = true)

    # Stochastic packed-x4 forms take one four-register f32 vector plus a
    # separately declared b32 random-bits register.
    for destination in (:e4m3x4, :e5m2x4, :e2m1x4, :e2m3x4, :e3m2x4)
        add!(destination, :f32, (:f32, :b32);
             stochastic = true, vector_source = true)
    end
    expected
end

@testset "ordinary cvt source-carrier ledger" begin
    @test length(_EXPECTED_CVT_SOURCE_CARRIERS) == 21
    @test Dict(PTX.ORDINARY_CVT_SOURCE_CARRIERS) ==
          _EXPECTED_CVT_SOURCE_CARRIERS

    expected = _expected_ordinary_cvt_source_schemas()
    actual = Dict(
        (schema.destination, schema.source, schema.stochastic, schema.scaled) =>
            schema
        for schema in PTX.ORDINARY_CVT_SOURCE_SCHEMAS)
    @test length(expected) == 193
    @test length(actual) == 193
    @test Set(keys(actual)) == Set(keys(expected))
    @test count(key -> key[1] in
                (:u8, :u16, :u32, :u64, :s8, :s16, :s32, :s64,
                 :bf16, :f16, :f32, :f64) &&
                key[2] in
                (:u8, :u16, :u32, :u64, :s8, :s16, :s32, :s64,
                 :bf16, :f16, :f32, :f64), keys(actual)) == 144
    @test count(key -> key[3], keys(actual)) == 7
    @test count(key -> key[4], keys(actual)) == 8

    for (key, want) in expected
        schema = actual[key]
        @test schema.operands == want.operands
        @test schema.vector_source == want.vector_source
        @test schema.section ==
              "ptx/9-instruction-set/9.7.9.22-data-movement-and-conversion-instructions-cvt.md"
    end
end

@testset "ordinary cvt schema selection is structural, not prefix grammar" begin
    ordinary = PTX.schema(PTX.CvtLedger(), :cvt, (:rn, :f32, :s32))
    @test ordinary.destination === :f32
    @test ordinary.source === :s32
    @test ordinary.operands == (:s32,)

    parsed_scaled = PTX.schema(PTX.CvtLedger(), :cvt,
        (:rn, Symbol("scaled::n2::ue8m0"), :bf16x2, :e4m3x2))
    direct_scaled = PTX.schema(PTX.CvtLedger(), :cvt,
        (:rn, :scaled__n2__ue8m0, :bf16x2, :e4m3x2))
    @test parsed_scaled === direct_scaled
    @test parsed_scaled.operands == (:b16, :b16)

    # The ledger closes terminal destination/source pairs and structural
    # rs/scaled operand roles. It intentionally leaves the large legal prefix
    # modifier cross-product to ptxas, matching the generic chain policy.
    vendor_prefix = PTX.schema(PTX.CvtLedger(), :cvt,
        (:future_modifier, :rn, :f32, :s32))
    @test vendor_prefix === ordinary

    for mods in (
        (), (:f32,), (:rn, :unknown, :s32),
        (:f64, :bf16, :rp),          # contradictory postfix ISA example
        (:bf16, :f16, :rz),          # contradictory postfix ISA example
        (:rs, :rs, :f16x2, :f32),
        (:rn, :scaled__n2__ue8m0, :scaled__n2__ue8m0,
         :bf16x2, :e4m3x2),
        (:rs, :f32, :s32),
        (:rn, :scaled__n2__ue8m0, :f32, :s32),
    )
        @test_throws ArgumentError PTX.schema(PTX.CvtLedger(), :cvt, mods)
    end
    @test PTX.schema(PTX.CvtLedger(), :cvt, (:pack, :sat, :u8, :s32, :b32)) ===
          nothing
end

function _cvt_immediate_module(instructions;
                               declarations = """
                               .reg .f16 %h<8>;
                               .reg .f32 %f<16>;
                               .reg .f64 %fd<4>;
                               .reg .b16 %b16<8>;
                               .reg .b32 %r<16>;
                               .reg .b64 %rd<8>;
                               .reg .u16 %u16<8>;
                               .reg .s16 %s16<8>;
                               .reg .u32 %u32<8>;
                               """)
    """
    .version 9.3
    .target sm_121a
    .address_size 64
    .visible .entry cvt_immediate_transpile()
    {
        $declarations
        $instructions
        ret;
    }
    """
end

function _assert_parseable_julia(source)
    parsed = Meta.parseall(source)
    @test parsed isa Expr && parsed.head == :toplevel
    @test !any(arg -> arg isa Expr && arg.head == :error, parsed.args)
end

@testset "ordinary cvt immediates use source-position carriers" begin
    source = _cvt_immediate_module("""
        cvt.rn.f32.s32 %f0, -7;
        cvt.rn.f32.u8 %f1, 255;
        cvt.u64.u32 %rd0, 17;
        cvt.u16.u8 %u160, 256;
        cvt.s16.s8 %s160, 255;
        cvt.u32.u32 %u320, -1;
        cvt.u32.u32 %u321, 0x100000000;
        cvt.u16.u8 %u162, 0377;
        cvt.u64.u64 %rd1, (0xffffffff << 32);
        cvt.u64.u64 %rd2, ((.u64) -1 >> 63);
        cvt.s64.s64 %rd3, (0x4000000000000000 << 1);
        cvt.rn.f16.f32 %h0, 1.25;
        cvt.rn.f16.f32 %h1, 0F3fc00000;
        cvt.rn.f16.f32 %h2, 0D3ff8000000000000;
        cvt.rn.f32.f64 %f2, 0F3f800000;
        cvt.f64.f32 %fd0, 0D3ff0000000000000;
        cvt.rn.satfinite.e4m3x2.f32 %b160, 1.0, 2.0;
        cvt.rn.f32.u32 %f3, WARP_SZ;
        cvt.rn.f32.u32 %f4, (WARP_SZ >> 1);
        cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 %r0, %b161, 0x7f7f;
        cvt.rs.f16x2.f32 %r2, 3.0, 4.0, %r1;
        cvt.rs.satfinite.e4m3x4.f32 %r3, {%f5, %f6, %f7, %f8}, %r4;
        cvt.pack.sat.u8.s32.b32 %r5, 1, 2, 3;
    """)
    out = PTX.ptx_to_julia(source)
    _assert_parseable_julia(out)
    for line in (
        "f0 = ptx\"cvt.rn.f32.s32\"(Int32(-7))",
        "f1 = ptx\"cvt.rn.f32.u8\"(UInt8(0xff))",
        "rd0 = ptx\"cvt.u64.u32\"(UInt32(0x00000011))",
        "u160 = ptx\"cvt.u16.u8\"(UInt8(0x00))",
        "s160 = ptx\"cvt.s16.s8\"(Int8(-1))",
        "u320 = ptx\"cvt.u32.u32\"(UInt32(0xffffffff))",
        "u321 = ptx\"cvt.u32.u32\"(UInt32(0x00000000))",
        "u162 = ptx\"cvt.u16.u8\"(UInt8(0xff))",
        "rd1 = ptx\"cvt.u64.u64\"(UInt64(0xffffffff00000000))",
        "rd2 = ptx\"cvt.u64.u64\"(UInt64(0x0000000000000001))",
        "rd3 = ptx\"cvt.s64.s64\"(Int64(-9223372036854775808))",
        "h0 = ptx\"cvt.rn.f16.f32\"(Float32(1.25))",
        "h1 = ptx\"cvt.rn.f16.f32\"(Float32(reinterpret(Float32, 0x3fc00000)))",
        "h2 = ptx\"cvt.rn.f16.f32\"(Float32(reinterpret(Float64, 0x3ff8000000000000)))",
        "f2 = ptx\"cvt.rn.f32.f64\"(Float64(reinterpret(Float32, 0x3f800000)))",
        "fd0 = ptx\"cvt.f64.f32\"(Float32(reinterpret(Float64, 0x3ff0000000000000)))",
        "b160 = ptx\"cvt.rn.satfinite.e4m3x2.f32\"(Float32(1.0), Float32(2.0))",
        "f3 = ptx\"cvt.rn.f32.u32\"(Val(32))",
        "f4 = ptx\"cvt.rn.f32.u32\"(UInt32(0x00000010))",
        "r0 = ptx\"cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2\"(b161, UInt16(0x7f7f))",
        "r2 = ptx\"cvt.rs.f16x2.f32\"(Float32(3.0), Float32(4.0), r1)",
        "r3 = ptx\"cvt.rs.satfinite.e4m3x4.f32\"((f5, f6, f7, f8), r4)",
        "r5 = ptx\"cvt.pack.sat.u8.s32.b32\"(Int32(1), Int32(2), UInt32(3))",
    )
        @test occursin(line, out)
    end
end

@testset "ordinary cvt rejects unrepresentable constant and role shapes" begin
    bad_instructions = (
        "cvt.f32.f16 %f0, 1.0;",
        "cvt.f32.bf16 %f0, 0x3f80;",
        "cvt.rn.f16x2.e4m3x2 %r0, 0x1234;",
        "cvt.rn.f16.f32 %h0, 1;",
        "cvt.rn.f32.s32 %f0, 1.0;",
        "cvt.rn.f32.s32 %f0, 0F3f800000;",
        "cvt.u64.u64 %rd0, 18446744073709551616;",
        "cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 %r0, %b160, 1.0;",
        "cvt.rs.f16x2.f32 %r0, 1.0, 2.0, 3;",
        "cvt.rs.f16x2.f32 %r0, 1.0, 2.0, %f0;",
        "cvt.rs.f16x2.f32 %r0, 1.0, %r1;",
        "cvt.rs.satfinite.e4m3x4.f32 %r0, {%f0, %f1, %f2}, %r1;",
        "cvt.rs.satfinite.e4m3x4.f32 %r0, {%f0, 1.0, %f2, %f3}, %r1;",
        "cvt.f64.bf16.rp %fd0, %b160;",
    )
    for instruction in bad_instructions
        @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(
            _cvt_immediate_module(instruction))
    end

    # Named registers without `%` are labels lexically; declaration lookup
    # still proves the b32 rbits role and must render the variable, not a label.
    named = _cvt_immediate_module(
        "cvt.rs.f16x2.f32 %r0, 1.0, 2.0, random_bits;";
        declarations = """
        .reg .b32 random_bits;
        .reg .b32 %r<2>;
        """)
    out = PTX.ptx_to_julia(named)
    @test occursin(
        "r0 = ptx\"cvt.rs.f16x2.f32\"(Float32(1.0), Float32(2.0), random_bits)",
        out)

    declared_data = _cvt_immediate_module(
        """
        cvt.rn.f32.s32 %f0, value;
        cvt.rn.f32.s32 %f1, end;
        """;
        declarations = """
        .reg .s32 value;
        .reg .s32 end;
        .reg .f32 %f<2>;
        """)
    declared_out = PTX.ptx_to_julia(declared_data)
    @test occursin("f0 = ptx\"cvt.rn.f32.s32\"(value)", declared_out)
    @test occursin("f1 = ptx\"cvt.rn.f32.s32\"(end_)", declared_out)

    wrong_named = _cvt_immediate_module(
        "cvt.rs.f16x2.f32 %r0, 1.0, 2.0, random_bits;";
        declarations = """
        .reg .f32 random_bits;
        .reg .b32 %r<2>;
        """)
    @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(wrong_named)
end

@testset "cvt register inventory is token-safe and scope-local" begin
    # The parser's RegDecl node retains only the first declarator. Recover the
    # rest from lexer tokens so a legal all-b32 stochastic source vector is
    # validated without weakening the role to arbitrary 32-bit integer regs.
    all_b32 = _cvt_immediate_module(
        "cvt.rs.satfinite.e4m3x4.f32 %dst, {%a, %b, %e, %f}, %rbits;";
        declarations =
            ".reg .b32 %dst, %a, %b, %e, %f, %rbits;")
    all_b32_out = PTX.ptx_to_julia(all_b32)
    @test occursin(
        "dst = ptx\"cvt.rs.satfinite.e4m3x4.f32\"((a, b, e, f), rbits)",
        all_b32_out)

    for dtype in (".u32", ".s32")
        wrong_integer = _cvt_immediate_module(
            "cvt.rs.satfinite.e4m3x4.f32 %dst, {%a, %b, %e, %f}, %rbits;";
            declarations = """
            .reg .b32 %dst, %rbits;
            .reg $dtype %a, %b, %e, %f;
            """)
        @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(wrong_integer)
    end

    # Commas and semicolons inside comments are one COMMENT token. They must
    # neither truncate the real declaration nor manufacture `%fake`.
    commented_decls =
        ".reg .b32 %dst, /* %fake, ; */ %a, %b, %e, %f, %rbits;"
    commented = _cvt_immediate_module(
        "cvt.rs.satfinite.e4m3x4.f32 %dst, {%a, %b, %e, %f}, %rbits;";
        declarations = commented_decls)
    @test occursin(
        "dst = ptx\"cvt.rs.satfinite.e4m3x4.f32\"((a, b, e, f), rbits)",
        PTX.ptx_to_julia(commented))
    fake = _cvt_immediate_module(
        "cvt.rs.satfinite.e4m3x4.f32 %dst, {%a, %b, %e, %fake}, %rbits;";
        declarations = commented_decls)
    @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(fake)

    scoped = _cvt_immediate_module("""
        {
            .reg .b32 %inner;
            cvt.rs.f16x2.f32 %dst, 1.0, 2.0, %inner;
        }
        """; declarations = ".reg .b32 %dst;")
    @test occursin(
        "dst = ptx\"cvt.rs.f16x2.f32\"(Float32(1.0), Float32(2.0), inner)",
        PTX.ptx_to_julia(scoped))

    leaked = _cvt_immediate_module("""
        {
            .reg .b32 %inner;
            cvt.rs.f16x2.f32 %dst, 1.0, 2.0, %inner;
        }
        cvt.rs.f16x2.f32 %dst, 1.0, 2.0, %inner;
        """; declarations = ".reg .b32 %dst;")
    @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(leaked)
end

@testset "ordinary cvt prefix policy preserves the canonical ABI boundary" begin
    # Prefix legality is intentionally deferred to ptxas, but canonical
    # terminal placement and the reviewed source carrier remain mandatory.
    accepted = PTX.ptx_to_julia(_cvt_immediate_module(
        "cvt.future_modifier.rn.f32.s32 %f0, 7;"))
    @test occursin(
        "f0 = ptx\"cvt.future_modifier.rn.f32.s32\"(Int32(7))",
        accepted)
    @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(_cvt_immediate_module(
        "cvt.f32.s32.future_modifier %f0, 7;"))
end
