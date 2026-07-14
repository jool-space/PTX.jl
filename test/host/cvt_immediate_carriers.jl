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
    ordinary = PTX.ordinary_cvt_source_schema((:rn, :f32, :s32))
    @test ordinary.destination === :f32
    @test ordinary.source === :s32
    @test ordinary.operands == (:s32,)

    parsed_scaled = PTX.ordinary_cvt_source_schema(
        (:rn, Symbol("scaled::n2::ue8m0"), :bf16x2, :e4m3x2))
    direct_scaled = PTX.ordinary_cvt_source_schema(
        (:rn, :scaled__n2__ue8m0, :bf16x2, :e4m3x2))
    @test parsed_scaled === direct_scaled
    @test parsed_scaled.operands == (:b16, :b16)

    # The ledger closes terminal destination/source pairs and structural
    # rs/scaled operand roles. It intentionally leaves the large legal prefix
    # modifier cross-product to ptxas, matching the generic chain policy.
    vendor_prefix = PTX.ordinary_cvt_source_schema(
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
        @test_throws ArgumentError PTX.ordinary_cvt_source_schema(mods)
    end
    @test PTX.ordinary_cvt_source_schema((:pack, :sat, :u8, :s32, :b32)) ===
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
        "f0 = ptx\"cvt.rn.f32.s32\"((-7) % Int32)",
        "f1 = ptx\"cvt.rn.f32.u8\"((255) % UInt8)",
        "rd0 = ptx\"cvt.u64.u32\"((17) % UInt32)",
        "u160 = ptx\"cvt.u16.u8\"((256) % UInt8)",
        "s160 = ptx\"cvt.s16.s8\"((255) % Int8)",
        "u320 = ptx\"cvt.u32.u32\"((-1) % UInt32)",
        "u321 = ptx\"cvt.u32.u32\"((0x100000000) % UInt32)",
        "h0 = ptx\"cvt.rn.f16.f32\"(Float32(1.25))",
        "h1 = ptx\"cvt.rn.f16.f32\"(Float32(reinterpret(Float32, 0x3fc00000)))",
        "h2 = ptx\"cvt.rn.f16.f32\"(Float32(reinterpret(Float64, 0x3ff8000000000000)))",
        "f2 = ptx\"cvt.rn.f32.f64\"(Float64(reinterpret(Float32, 0x3f800000)))",
        "fd0 = ptx\"cvt.f64.f32\"(Float32(reinterpret(Float64, 0x3ff0000000000000)))",
        "b160 = ptx\"cvt.rn.satfinite.e4m3x2.f32\"(Float32(1.0), Float32(2.0))",
        "f3 = ptx\"cvt.rn.f32.u32\"(Val(32))",
        "f4 = ptx\"cvt.rn.f32.u32\"(((32 >> 1)) % UInt32)",
        "r0 = ptx\"cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2\"(b161, (0x7f7f) % UInt16)",
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
        "cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 %r0, %b160, 1.0;",
        "cvt.rs.f16x2.f32 %r0, 1.0, 2.0, 3;",
        "cvt.rs.f16x2.f32 %r0, 1.0, 2.0, %f0;",
        "cvt.rs.f16x2.f32 %r0, 1.0, %r1;",
        "cvt.rs.satfinite.e4m3x4.f32 %r0, {%f0, %f1, %f2}, %r1;",
        "cvt.rs.satfinite.e4m3x4.f32 %r0, {%f0, 1.0, %f2, %f3}, %r1;",
        "cvt.f64.bf16.rp %fd0, %b160;",
    )
    for instruction in bad_instructions
        @test_throws ArgumentError PTX.ptx_to_julia(
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
    @test_throws ArgumentError PTX.ptx_to_julia(wrong_named)
end

@testset "ordinary cvt prefix policy preserves the canonical ABI boundary" begin
    # Prefix legality is intentionally deferred to ptxas, but canonical
    # terminal placement and the reviewed source carrier remain mandatory.
    accepted = PTX.ptx_to_julia(_cvt_immediate_module(
        "cvt.future_modifier.rn.f32.s32 %f0, 7;"))
    @test occursin(
        "f0 = ptx\"cvt.future_modifier.rn.f32.s32\"((7) % Int32)",
        accepted)
    @test_throws ArgumentError PTX.ptx_to_julia(_cvt_immediate_module(
        "cvt.f32.s32.future_modifier %f0, 7;"))
end
