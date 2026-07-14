using .PTX: Operation, RawOperation, build_call, format_call

const _STRUCTURED_GENERAL_SECTION =
    "ptx/9-instruction-set/9.7.6.2-comparison-and-selection-instructions-setp.md"
const _STRUCTURED_HALF_SECTION =
    "ptx/9-instruction-set/9.7.7.2-half-precision-comparison-instructions-setp.md"
const _STRUCTURED_LOP3_SECTION =
    "ptx/9-instruction-set/9.7.8.6-logic-and-shift-instructions-lop3.md"
const _STRUCTURED_MATCH_SECTION =
    "ptx/9-instruction-set/9.7.14.11-parallel-synchronization-and-communication-instructions-match.sync.md"
const _STRUCTURED_ELECT_SECTION =
    "ptx/9-instruction-set/9.7.14.15-parallel-synchronization-and-communication-instructions-elect.sync.md"

# Independent PTX 9.3 grammar oracle. This deliberately does not consume the
# production schema constructors: a modifier-product edit on either side must
# disagree visibly rather than narrowing its own test oracle.
function _expected_structured_results()
    expected = Dict{Tuple{Symbol,Tuple},NamedTuple}()
    function add!(op, mods, ptxmods, outputs, sinkable, operands,
                  ptx_version, min_sm, section)
        key = (op, mods)
        haskey(expected, key) && error("duplicate test oracle key $key")
        expected[key] = (; ptxmods, outputs, sinkable,
                          max_sinks = any(sinkable) ? 1 : 0, operands,
                          ptx_version, min_sm, section)
    end

    basic = (:eq, :ne, :lt, :le, :gt, :ge)
    floating = (basic..., :equ, :neu, :ltu, :leu, :gtu, :geu, :num, :nan)
    boolops = (nothing, :and, :or, :xor)
    general = (
        ((:b16, :b32, :b64), (:eq, :ne), false),
        ((:s16, :s32, :s64), basic, false),
        ((:u16, :u32, :u64),
         (basic..., :lo, :ls, :hi, :hs), false),
        ((:f64,), floating, false),
        ((:f32,), floating, true),
    )
    for (types, cmps, admits_ftz) in general, dtype in types, cmp in cmps,
        boolop in boolops, ftz in (admits_ftz ? (false, true) : (false,))
        mods = (cmp,
                (boolop === nothing ? () : (boolop,))...,
                (ftz ? (:ftz,) : ())..., dtype)
        operands = boolop === nothing ? (dtype, dtype) :
                                         (dtype, dtype, :pred)
        min_sm = dtype === :f64 ? v"1.3" : nothing
        add!(:setp, mods, mods, (:pred,), (true,), operands,
             v"1.0", min_sm, _STRUCTURED_GENERAL_SECTION)
        add!(:setp, (:dual, mods...), mods, (:pred, :pred), (true, true),
             operands, v"1.0", min_sm, _STRUCTURED_GENERAL_SECTION)
    end

    for (dtype, input, packed, ptx_version, min_sm, admits_ftz) in (
        (:f16,    :f16,  false, v"4.2", v"5.3", true),
        (:f16x2,  :b32,  true,  v"4.2", v"5.3", true),
        (:bf16,   :bf16, false, v"7.8", v"9.0", false),
        (:bf16x2, :b32,  true,  v"7.8", v"9.0", false),
    ), cmp in floating, boolop in boolops,
        ftz in (admits_ftz ? (false, true) : (false,))
        mods = (cmp,
                (boolop === nothing ? () : (boolop,))...,
                (ftz ? (:ftz,) : ())..., dtype)
        operands = boolop === nothing ? (input, input) :
                                         (input, input, :pred)
        outputs = packed ? (:pred, :pred) : (:pred,)
        add!(:setp, mods, mods, outputs, ntuple(_ -> true, length(outputs)),
             operands, ptx_version, min_sm, _STRUCTURED_HALF_SECTION)
    end

    add!(:lop3, (:b32,), (:b32,), (:b32,), (false,),
         (:b32, :b32, :b32, :imm8), v"4.3", v"5.0",
         _STRUCTURED_LOP3_SECTION)
    for boolop in (:or, :and)
        mods = (boolop, :b32)
        add!(:lop3, mods, mods, (:b32, :pred), (true, false),
             (:b32, :b32, :b32, :imm8, :pred), v"8.2", v"7.0",
             _STRUCTURED_LOP3_SECTION)
    end
    add!(:elect, (:sync,), (:sync,), (:b32, :pred), (true, false),
         (:b32,), v"8.0", v"9.0", _STRUCTURED_ELECT_SECTION)
    for dtype in (:b32, :b64), mode in (:any, :all)
        mods = (mode, :sync, dtype)
        add!(:match, mods, mods, (:b32,), (mode === :all,),
             (dtype, :b32), v"6.0", v"7.0", _STRUCTURED_MATCH_SECTION)
        mode === :all || continue
        add!(:match, (mods..., :pred), mods, (:b32, :pred), (true, true),
             (dtype, :b32), v"6.0", v"7.0", _STRUCTURED_MATCH_SECTION)
    end
    expected
end

const EXPECTED_STRUCTURED_RESULTS = _expected_structured_results()

@testset "structured-result PTX 9.3 closed-world inventory" begin
    actual = Dict((schema.op, schema.mods) => schema
                  for schema in PTX.STRUCTURED_RESULT_SCHEMAS)
    @test length(EXPECTED_STRUCTURED_RESULTS) == 1114
    @test length(actual) == 1114
    @test Set(keys(actual)) == Set(keys(EXPECTED_STRUCTURED_RESULTS))

    for (key, expected) in EXPECTED_STRUCTURED_RESULTS
        schema = actual[key]
        @test schema.ptxmods == expected.ptxmods
        @test schema.outputs == expected.outputs
        @test schema.sinkable == expected.sinkable
        @test schema.max_sinks == expected.max_sinks
        @test schema.operands == expected.operands
        @test schema.ptx_version == expected.ptx_version
        @test schema.min_sm == expected.min_sm
        @test schema.feature_set === :baseline
        @test schema.section == expected.section
    end

    by_op = Dict(op => count(s -> s.op === op, values(actual))
                 for op in (:setp, :lop3, :match, :elect))
    @test by_op == Dict(:setp => 1104, :lop3 => 3,
                        :match => 6, :elect => 1)
    @test count(s -> length(s.outputs) == 2, values(actual)) == 557
    @test count(s -> length(s.outputs) == 1, values(actual)) == 557
    # Only base lop3 and the two match.any forms prohibit every sink.
    @test count(s -> !any(s.sinkable), values(actual)) == 3

    @test count(s -> s.op === :setp && first(s.mods) === :dual,
                values(actual)) == 384
    @test count(s -> s.op === :setp &&
                      last(s.mods) in (:f16x2, :bf16x2),
                values(actual)) == 168
    @test count(s -> s.op === :lop3 && length(s.outputs) == 2,
                values(actual)) == 2
end

@testset "structured-result exact ABI rendering" begin
    cases = (
        ((:setp, (:dual, :lt, :and, :ftz, :f32),
          (Float32, Float32, Bool)),
         "setp.lt.and.ftz.f32 \$0|\$1, \$2, \$3, \$4;",
         "=b,=b,f,f,b", Tuple{Bool, Bool}, false, false),
        ((:setp, (:nan, :bf16x2), (UInt32, UInt32)),
         "setp.nan.bf16x2 \$0|\$1, \$2, \$3;",
         "=b,=b,r,r", Tuple{Bool, Bool}, false, false),
        ((:lop3, (:or, :b32),
          (UInt32, UInt32, UInt32, Val{0x96}, Bool)),
         "lop3.or.b32 \$0|\$1, \$2, \$3, \$4, 150, \$5;",
         "=r,=b,r,r,r,b", Tuple{UInt32, Bool}, false, false),
        ((:elect, (:sync,), (UInt32,)),
         "elect.sync \$0|\$1, \$2;",
         "=r,=b,r,~{memory}", Tuple{UInt32, Bool}, true, true),
        ((:match, (:all, :sync, :b64, :pred), (UInt64, UInt32)),
         "match.all.sync.b64 \$0|\$1, \$2, \$3;",
         "=r,=b,l,r,~{memory}", Tuple{UInt32, Bool}, true, true),
        ((:match, (:any, :sync, :b64),
          (UInt64, Val{0xffffffff})),
         "match.any.sync.b64 \$0, \$1, 4294967295;",
         "=r,l,~{memory}", UInt32, true, true),
    )
    for ((op, mods, argtypes), asm, constraints, rettype,
         side_effects, convergent) in cases
        spec = build_call(op, mods, argtypes)
        @test spec.asm == asm
        @test spec.constraints == constraints
        @test spec.rettype === rettype
        @test spec.side_effects === side_effects
        @test spec.convergent === convergent
        @test format_call(Operation{op, mods}(), Tuple{argtypes...}) == asm
        info = PTX.lowering(Operation{op, mods}(), argtypes)
        @test info.tier === :chain_asm
        @test info.rettype === rettype
        @test info.asm == asm
    end

    # Exact raw calls keep the schema's register/immediate syntax while using
    # RAW_CONTRACT's conservative optimizer attributes. They must not inherit
    # raw's generic pointer-bracketing guess.
    raw = build_call(:setp, (:dual, :eq, :u32),
                     (UInt32, UInt32); raw = true)
    @test raw.asm == "setp.eq.u32 \$0|\$1, \$2, \$3;"
    @test !occursin('[', raw.asm)
    @test raw.side_effects && raw.convergent
    @test occursin("~{memory}", raw.constraints)
    raw_info = PTX.lowering(
        RawOperation{:setp, (:dual, :eq, :u32)}(), (UInt32, UInt32))
    @test raw_info.tier === :chain_asm
    @test raw_info.asm == raw.asm
end

@testset "structured-result misses fail loud on normal and raw routes" begin
    misses = (
        (:setp, (:dual, :eq, :f16), (Float16, Float16)),
        (:setp, (:lt, :b32), (UInt32, UInt32)),
        (:setp, (:eq, :ftz, :f64), (Float64, Float64)),
        (:setp, (:eq, :xor, :b32), (UInt32, UInt32)),
        (:lop3, (:xor, :b32),
         (UInt32, UInt32, UInt32, Val{0}, Bool)),
        (:match, (:any, :sync, :b32, :pred), (UInt32, UInt32)),
        (:elect, (), (UInt32,)),
    )
    for (op, mods, argtypes) in misses
        for raw in (false, true)
            operation = raw ? RawOperation{op, mods}() : Operation{op, mods}()
            info = PTX.lowering(operation, argtypes)
            @test info.tier === :forbidden
            @test info.asm === nothing
            @test_throws ArgumentError build_call(op, mods, argtypes; raw)
        end
    end

    bad_operands = (
        (:setp, (:eq, :u32), (UInt16, UInt16)),
        (:setp, (:eq, :u32), (UInt32,)),
        (:lop3, (:b32,), (UInt32, UInt32, UInt32, UInt32)),
        (:lop3, (:b32,), (UInt32, UInt32, UInt32, Val{256})),
        (:elect, (:sync,), (UInt64,)),
        (:match, (:all, :sync, :b64, :pred), (UInt32, UInt32)),
    )
    for (op, mods, argtypes) in bad_operands
        @test PTX.lowering(Operation{op, mods}(), argtypes).tier === :forbidden
        @test PTX.lowering(RawOperation{op, mods}(), argtypes).tier === :forbidden
        @test_throws ArgumentError build_call(op, mods, argtypes)
        @test_throws ArgumentError build_call(op, mods, argtypes; raw = true)
    end
end

function _structured_transpile(instructions::AbstractString;
                               declarations::AbstractString = """
        .reg .pred %p<8>;
        .reg .b16 %h<4>;
        .reg .b32 %r<12>;
        .reg .b64 %rd<4>;
        .reg .f32 %f<4>;
        """)
    PTX.ptx_to_julia(""".version 9.3
    .target sm_90
    .address_size 64
    .visible .entry structured()
    {
    $declarations
    $instructions
    ret;
    }
    """)
end

@testset "structured-result transpiler grouping, sinks, and carriers" begin
    julia = _structured_transpile("""
        setp.lt.and.ftz.f32 %p0|%p1, %f0, %f1, %p2;
        setp.eq.f16 _, %h0, %h1;
        setp.nan.bf16x2 %p3|_, %r0, %r1;
        lop3.or.b32 _|%p4, %r0, %r1, %r2, ((0x1 << 4) | 3), %p2;
        match.any.sync.b64 %r3, %rd0, 0xffffffff;
        match.all.sync.b64 %r4|_, %rd0, %r0;
        elect.sync _|%p5, 0xffffffff;
        setp.eq.u32 %p6, %r0, WARP_SZ;
        """)
    @test occursin(
        "(p0, p1) = ptx\"setp.dual.lt.and.ftz.f32\"(f0, f1, p2)", julia)
    @test occursin("_ = ptx\"setp.eq.f16\"(h0, h1)", julia)
    @test occursin("(p3, _) = ptx\"setp.nan.bf16x2\"(r0, r1)", julia)
    @test occursin(
        "(_, p4) = ptx\"lop3.or.b32\"(r0, r1, r2, Val(19), p2)", julia)
    @test occursin(
        "r3 = ptx\"match.any.sync.b64\"(rd0, UInt32(0xffffffff))", julia)
    @test occursin(
        "(r4, _) = ptx\"match.all.sync.b64.pred\"(rd0, r0)", julia)
    @test occursin(
        "(_, p5) = ptx\"elect.sync\"(UInt32(0xffffffff))", julia)
    @test occursin(
        "p6 = ptx\"setp.eq.u32\"(r0, UInt32(0x00000020))", julia)
    @test !occursin("local _", julia)
    @test Meta.parseall(julia).head === :toplevel

    # Registers without `%` are LabelOperand nodes in the parser. The generic
    # declared-register lookup must still prove their types, including counted
    # declarations, and retain them as assigned Julia locals.
    bare = _structured_transpile(
        "lop3.or.b32 r0|p0, r1, r2, r3, 0x96, p1;";
        declarations = ".reg .pred p<2>;\n.reg .b32 r<4>;")
    @test occursin(
        "(r0, p0) = ptx\"lop3.or.b32\"(r1, r2, r3, Val(150), p1)", bare)

    # RegDecl structurally retains only the first name in a comma-packed PTX
    # declaration. Type validation must recover the remaining declarators from
    # its lossless raw snapshot, without accepting an actually undeclared name.
    comma_packed = _structured_transpile(
        "lop3.or.b32 r0|p0, r1, r2, r3, 0x96, p1;";
        declarations = ".reg .pred p0, p1;\n.reg .b32 r0, r1, r2, r3;")
    @test occursin(
        "(r0, p0) = ptx\"lop3.or.b32\"(r1, r2, r3, Val(150), p1)",
        comma_packed)

    # Lexer-token recovery must ignore punctuation and identifiers inside
    # comments. In particular, a comment semicolon cannot truncate the real
    # p1 declaration and the fake name cannot satisfy schema validation.
    commented = _structured_transpile(
        "setp.eq.u32 p1, r0, r1;";
        declarations =
            ".reg .pred p0, /* fake, ; */ p1;\n.reg .u32 r0, r1;")
    @test occursin("p1 = ptx\"setp.eq.u32\"(r0, r1)", commented)
    @test_throws ArgumentError _structured_transpile(
        "setp.eq.u32 fake, r0, r1;";
        declarations =
            ".reg .pred p0, /* fake, ; */ p1;\n.reg .u32 r0, r1;")

    # A declaration inside a PTX brace scope must leave the active register
    # table with that scope. Otherwise schema validation could accept an
    # out-of-scope register and emit plausible Julia for invalid PTX.
    @test_throws ArgumentError _structured_transpile("""
        {
            .reg .pred inner;
            setp.eq.u32 inner, r0, r1;
        }
        setp.eq.u32 inner, r0, r1;
        """; declarations = ".reg .u32 r0, r1;")
end

@testset "structured integer and predicate immediates use PTX semantics" begin
    julia = _structured_transpile("""
        setp.eq.u32 %p0, %r0, -1;
        setp.eq.s16 %p1, %h0, 0xffff;
        setp.eq.b64 %p2, %rd0, (0xffffffff << 32);
        setp.eq.and.u32 %p3, %r0, 1, 2;
        setp.eq.and.u32 %p4, %r0, 1, !2;
        """)
    @test occursin(
        "p0 = ptx\"setp.eq.u32\"(r0, UInt32(0xffffffff))", julia)
    @test occursin(
        "p1 = ptx\"setp.eq.s16\"(h0, Int16(-1))", julia)
    @test occursin(
        "p2 = ptx\"setp.eq.b64\"(rd0, UInt64(0xffffffff00000000))", julia)
    @test occursin(
        "p3 = ptx\"setp.eq.and.u32\"(r0, UInt32(0x00000001), true)",
        julia)
    @test occursin(
        "p4 = ptx\"setp.eq.and.u32\"(r0, UInt32(0x00000001), false)",
        julia)
    @test Meta.parseall(julia).head === :toplevel

    floating = _structured_transpile("""
        setp.eq.f32 %p0, %f0, 0.0;
        setp.eq.f64 %p1, %fd0, 0.0;
        """; declarations = """
        .reg .pred %p<2>;
        .reg .f32 %f0;
        .reg .f64 %fd0;
        """)
    @test occursin("ptx\"setp.eq.f32\"(f0, Float32(0.0))", floating)
    @test occursin("ptx\"setp.eq.f64\"(fd0, Float64(0.0))", floating)
end

@testset "lop3 transpiler evaluates every parser-supported constant shape" begin
    cases = (
        "42" => 42,
        "0x2A" => 42,
        "052" => 42,
        "WARP_SZ" => 32,
        "(WARP_SZ >> 1)" => 16,
        "(!0)" => 1,
        "(((~0) & 0xff) << 0)" => 255,
        "((0x1 << 4) | 3)" => 19,
        "(((.u64) 1 << 7) | 3)" => 131,
        "(((3 * 7) - 1) << 0)" => 20,
        "((8 / 2) << 1)" => 8,
        "((9 % 4) << 1)" => 2,
        "(((-1 % 16) + 1) << 0)" => 16,
        "(((.s64) -8 >> 1) & 0xff)" => 252,
        "(((.u64) -8 >> 60) & 0xff)" => 15,
        "((3 < 4) << 7)" => 128,
        "(((4 >= 4) + (3 == 3) + (3 != 4)) << 0)" => 3,
        "(((0 || 7) && 9) << 0)" => 1,
        "(((.u64) -1 > 0) << 0)" => 1,
    )
    for (literal, expected) in cases
        julia = _structured_transpile(
            "lop3.b32 %r0, %r1, %r2, %r3, $literal;")
        @test occursin("ptx\"lop3.b32\"(r1, r2, r3, Val($expected))", julia)
    end

    for bad in ("-1", "256", "(1 << 8)", "WARP_SZ_limit", "%r4",
                "((1 / 0) << 0)", "09", "18446744073709551616")
        @test_throws ArgumentError _structured_transpile(
            "lop3.b32 %r0, %r1, %r2, %r3, $bad;")
    end
end

@testset "PTX 64-bit integer constant semantics" begin
    evaluate = PTX.Codegen._ptx_integer_constant
    carrier = PTX.Codegen._ptx_integer_carrier_expr

    # PTX literals are always s64/u64, independent of the host language's
    # preferred width for a hexadecimal literal. This is the regression that
    # a Julia-source expression such as `0xffffffff << 32` gets wrong.
    @test evaluate("0xffffffff << 32") === Int64(-4294967296)
    @test evaluate(
        "(0xffffffff << 32) == 0xffffffff00000000") === Int64(1)

    # Arithmetic is defined in the 64-bit PTX domain and wraps in both the
    # signed and usual-conversion-to-unsigned cases.
    @test evaluate("0x7fffffffffffffff + 1") === typemin(Int64)
    @test evaluate("0xffffffffffffffff + 1") === UInt64(0)
    @test evaluate("((.u64) -1 >> 63)") === UInt64(1)
    @test evaluate("((-1 & -1) >> 63)") === Int64(-1)

    @test evaluate("052") === Int64(42)
    @test evaluate("WARP_SZ") === Int64(32)
    @test_throws ArgumentError evaluate("09")
    @test_throws ArgumentError evaluate("18446744073709551616")
    @test_throws ArgumentError evaluate("0 && (1 / 0)")
    @test_throws ArgumentError evaluate("1 || (1 / 0)")

    # The shared carrier bridge performs PTX's use-site truncation without
    # Julia checked-conversion failures, including the 8-bit widths used by
    # fixed scalar/vector result schemas outside this family.
    @test carrier("-1", UInt8) == "UInt8(0xff)"
    @test carrier("0xff", Int8) == "Int8(-1)"
    @test carrier("0x100000001", UInt32) == "UInt32(0x00000001)"
end

@testset "structured-result transpiler rejects illegal sinks and declarations" begin
    invalid = (
        "lop3.b32 _, %r0, %r1, %r2, 0x96;",
        "lop3.or.b32 %r0|_, %r1, %r2, %r3, 0x96, %p0;",
        "match.any.sync.b32 _, %r0, %r1;",
        "elect.sync %r0|_, %r1;",
        "setp.eq.u32 _|_, %r0, %r1;",
    )
    for instruction in invalid
        @test_throws ArgumentError _structured_transpile(instruction)
    end

    @test_throws ArgumentError _structured_transpile(
        "setp.eq.u32 %r0, %r1, %r2;")
    @test_throws ArgumentError _structured_transpile(
        "match.all.sync.b64 %r0|%r1, %rd0, %r2;")
    @test_throws ArgumentError _structured_transpile(
        "setp.eq.u32 %p0, %r0, %r1;";
        declarations = ".reg .pred %p0;\n.reg .b16 %r<2>;")
    @test_throws ArgumentError _structured_transpile(
        "setp.eq.u32 %p0, %f0, %f1;";
        declarations = ".reg .pred %p0;\n.reg .f32 %f<2>;")
    @test_throws ArgumentError _structured_transpile(
        "setp.eq.f32 %p0, %f0, 0;")
    @test_throws ArgumentError _structured_transpile(
        "setp.eq.f16 %p0, %h0, 0.0;")
    @test_throws ArgumentError _structured_transpile(
        "setp.eq.bf16 %p0, %h0, 0.0;")
end
