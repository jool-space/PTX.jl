using PTX.IR: Instruction, RegisterOperand, ImmediateOperand, LabelOperand,
              VectorOperand, AddressOperand, ParenthesizedOperand,
              NegatedOperand, PipeOperand, Predicate, FormattingInfo,
              Version, Target, AddressSize, RegDecl, VarDecl, Comment,
              BlankLine, RawLine, Block, Param, FunctionDirective,
              ScalarType, StateSpace, LinkingDirective, format
using PTX.IR: ptx, scalar_type_from_ptx, state_space_from_ptx, linking_from_ptx

# IR uses String (not Symbol) for opcodes/modifiers — preserves raw lexemes
# for round-trip fidelity. Operand text is also stored as raw strings.

@testset "format(Instruction): basic forms" begin
    # add.f32  $0, $1, $2;
    inst = Instruction("add", (".f32",),
                       (RegisterOperand("\$0"), RegisterOperand("\$1"), RegisterOperand("\$2")))
    @test format(inst) == "add.f32 \$0, \$1, \$2;"

    # mov.u32  $0, %tid.x;
    inst = Instruction("mov", (".u32",),
                       (RegisterOperand("\$0"), RegisterOperand("%tid.x")))
    @test format(inst) == "mov.u32 \$0, %tid.x;"

    # bar.sync 0;
    inst = Instruction("bar", (".sync",), (ImmediateOperand("0"),))
    @test format(inst) == "bar.sync 0;"

    # Multi-modifier with `::` qualifier preserved as-is.
    inst = Instruction("cp",
                       (".async", ".bulk", ".tensor", ".2d", ".shared::cta", ".global"),
                       (RegisterOperand("\$0"), RegisterOperand("\$1")))
    @test format(inst) == "cp.async.bulk.tensor.2d.shared::cta.global \$0, \$1;"

    # Operand-less.
    inst = Instruction("trap", (), ())
    @test format(inst) == "trap;"
end

@testset "format(Instruction): predication" begin
    inst = Instruction("bra", (), (LabelOperand("L_END"),),
                       Predicate("%p0", false))
    @test format(inst) == "@%p0 bra L_END;"

    inst = Instruction("bra", (), (LabelOperand("L_END"),),
                       Predicate("%p0", true))
    @test format(inst) == "@!%p0 bra L_END;"
end

@testset "format(Instruction): operand kinds" begin
    # Vector operand (ldmatrix-style braced register list)
    vec = VectorOperand((RegisterOperand("%r0"), RegisterOperand("%r1"),
                         RegisterOperand("%r2"), RegisterOperand("%r3")))
    inst = Instruction("ldmatrix", (".sync", ".aligned", ".x4", ".m8n8", ".shared", ".b16"),
                       (vec, AddressOperand("%r10")))
    @test format(inst) ==
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%r0, %r1, %r2, %r3}, [%r10];"

    # Address with offset
    inst = Instruction("ld", (".global", ".u32"),
                       (RegisterOperand("%r0"), AddressOperand("%rd1", "16")))
    @test format(inst) == "ld.global.u32 %r0, [%rd1+16];"

    # Pipe (setp dual-pred)
    pipe = PipeOperand(RegisterOperand("%p0"), RegisterOperand("%p1"))
    inst = Instruction("setp", (".lt", ".f32"),
                       (pipe, RegisterOperand("%f0"), RegisterOperand("%f1")))
    @test format(inst) == "setp.lt.f32 %p0|%p1, %f0, %f1;"

    # Negated
    inst = Instruction("not", (".pred",),
                       (RegisterOperand("%p0"), NegatedOperand(RegisterOperand("%p1"))))
    @test format(inst) == "not.pred %p0, !%p1;"

    # Parenthesized (call return list)
    paren = ParenthesizedOperand((RegisterOperand("%r0"),))
    inst = Instruction("call", (".uni",), (paren, LabelOperand("foo")))
    @test format(inst) == "call.uni (%r0), foo;"

    # Immediate stored as raw text — preserves lexeme exactly.
    inst = Instruction("mov", (".b32",),
                       (RegisterOperand("%r0"), ImmediateOperand("0xDEADBEEF")))
    @test format(inst) == "mov.b32 %r0, 0xDEADBEEF;"
end

@testset "ScalarType / StateSpace / LinkingDirective enums (EnumX)" begin
    @test ptx(ScalarType.B32) == ".b32"
    @test ptx(ScalarType.F32) == ".f32"
    @test ptx(ScalarType.E4M3) == ".e4m3"
    @test scalar_type_from_ptx(".f32") == ScalarType.F32
    @test scalar_type_from_ptx("u32") == ScalarType.U32

    @test ptx(StateSpace.SHARED) == ".shared"
    @test ptx(StateSpace.SHARED_CTA) == ".shared::cta"
    @test state_space_from_ptx(".shared::cluster") == StateSpace.SHARED_CLUSTER
    @test state_space_from_ptx("global") == StateSpace.GLOBAL
    @test state_space_from_ptx("const") == StateSpace.CONST   # no longer needs CONST_SS workaround

    @test ptx(LinkingDirective.VISIBLE) == ".visible"
    @test linking_from_ptx(".extern") == LinkingDirective.EXTERN

    @test_throws ArgumentError scalar_type_from_ptx(".bogus")
end

@testset "FormattingInfo defaults + raw_line escape" begin
    fi = FormattingInfo()
    @test fi.indent == ""
    @test fi.raw_line === nothing

    fi2 = FormattingInfo(indent = "    ", raw_line = "  weird.line  // pre-formatted")
    @test fi2.indent == "    "
    @test fi2.raw_line == "  weird.line  // pre-formatted"
end

# ---------------------------------------------------------------------------
# IR.unraw / IR.normalize / IR.diff
# ---------------------------------------------------------------------------

using PTX.Parser: parse as parse_ptx

@testset "IR.unraw: clears raw_source only" begin
    src = """.version 8.0
.target sm_89
.address_size 64

.visible .entry foo()
{
\tret;
}
"""
    m  = parse_ptx(src)
    @test m.raw_source !== nothing
    @test m.raw_header !== nothing

    u = PTX.IR.unraw(m)
    @test u.raw_source === nothing
    @test u.raw_header === m.raw_header           # raw_header survives
    @test u.directives === m.directives           # directives unchanged
    @test u.leading === m.leading

    # The whole point of unraw: format() now walks structural emit instead
    # of returning raw_source verbatim.
    @test format(u) != ""
    @test format(m) == src                         # original still uses fast path
end

@testset "IR.normalize: full canonical form" begin
    # Source with comments + blank lines + indented instructions — the
    # cosmetic content normalize() should strip.
    src = """// preamble comment
.version 8.0
.target sm_89
.address_size 64

// between header and body
.visible .entry foo()
{
\t// inside body
\t.reg .b32 %r<2>;

\tmov.u32 %r0, 1;
\tret;
}
"""
    m = parse_ptx(src)
    n = PTX.IR.normalize(m)

    @test n.raw_source === nothing
    @test n.raw_header === nothing
    @test n.leading == ()

    # Top-level directives: only Function survives; Comments dropped.
    types = unique(typeof(d) for d in n.directives)
    @test types == [PTX.IR.Function]

    # Body: Comment / BlankLine dropped; Instructions kept; FormattingInfo
    # cleared on every statement.
    f = first(d for d in n.directives if d isa PTX.IR.Function)
    body_types = unique(typeof(s) for s in f.body)
    @test PTX.IR.Comment   ∉ body_types
    @test PTX.IR.BlankLine ∉ body_types
    @test PTX.IR.Instruction ∈ body_types
    @test PTX.IR.RegDecl     ∈ body_types
    for s in f.body
        if hasfield(typeof(s), :formatting)
            @test getfield(s, :formatting) === nothing
        end
    end

    # Idempotent — normalize ∘ normalize = normalize.
    @test PTX.IR.normalize(n) == n
end

@testset "IR.normalize: equality across cosmetic variation" begin
    # Two textually different sources that mean the same thing.
    a = """.version 8.0
.target sm_89
.address_size 64
.visible .entry foo()
{
\tret;
}
"""
    b = """// a comment
.version 8.0
.target sm_89
.address_size 64

.visible .entry foo()
{
    // body comment
    ret;

}
"""
    @test PTX.IR.normalize(parse_ptx(a)) == PTX.IR.normalize(parse_ptx(b))
    @test isempty(PTX.IR.diff(parse_ptx(a), parse_ptx(b)))
end

@testset "IR.diff: empty for equal modules" begin
    src = """.version 8.0
.target sm_89
.address_size 64
.visible .entry foo()
{
\tret;
}
"""
    m = parse_ptx(src)
    @test isempty(PTX.IR.diff(m, m))
end

@testset "IR.diff: surfaces header / structural differences" begin
    base = """.version 8.0
.target sm_89
.address_size 64
.visible .entry foo()
{
\tmov.u32 %r0, 1;
\tret;
}
"""
    # version mismatch
    d = PTX.IR.diff(parse_ptx(base),
                    parse_ptx(replace(base, ".version 8.0" => ".version 8.5")))
    @test any(line -> occursin("version:", line), d)

    # target mismatch
    d = PTX.IR.diff(parse_ptx(base),
                    parse_ptx(replace(base, "sm_89" => "sm_90a")))
    @test any(line -> occursin("target:", line), d)

    # address_size mismatch
    d = PTX.IR.diff(parse_ptx(base),
                    parse_ptx(replace(base, "address_size 64" => "address_size 32")))
    @test any(line -> occursin("address_size:", line), d)

    # function rename
    d = PTX.IR.diff(parse_ptx(base),
                    parse_ptx(replace(base, "foo" => "bar")))
    @test any(line -> occursin("name:", line), d)

    # opcode change
    d = PTX.IR.diff(parse_ptx(base),
                    parse_ptx(replace(base, "mov.u32" => "mov.b32")))
    @test any(line -> occursin("mods:", line), d)

    # body length differs
    d = PTX.IR.diff(parse_ptx(base),
                    parse_ptx(replace(base, "\tret;\n" => "\tadd.u32 %r1, %r0, 1;\n\tret;\n")))
    @test any(line -> occursin("body length:", line), d)
end

@testset "IR.diff: entry_only filters helper functions" begin
    # Two modules differ only in a `.func` helper. With entry_only=true the
    # diff should be empty; without, it should surface.
    base = """.version 8.0
.target sm_89
.address_size 64
.visible .entry main()
{
\tret;
}
"""
    with_helper = base * """
.func helper()
{
\tret;
}
"""
    da = PTX.IR.diff(parse_ptx(base), parse_ptx(with_helper))
    @test !isempty(da)
    db = PTX.IR.diff(parse_ptx(base), parse_ptx(with_helper); entry_only = true)
    @test isempty(db)
end

@testset "format(::Function): preceding_comments survive structural emit" begin
    # Comments stashed in Function.formatting.preceding_comments by the
    # parser (collected via `_take_comments!` between linking-prefix and
    # `.entry`/`.func`) must survive the structural emit path. Regression
    # for a v0 bug where the Function emitter ignored the field.
    src = """.version 8.0
.target sm_90a
.address_size 64

.visible
// banner between visible and entry
.entry foo()
{
\tret;
}
"""
    m = parse_ptx(src)
    out = format(PTX.IR.unraw(m))
    @test occursin("// banner between visible and entry", out)
    @test isempty(PTX.IR.diff(PTX.IR.normalize(m),
                              PTX.IR.normalize(parse_ptx(out))))

    # Programmatic construction: comments only on formatting field.
    f = PTX.IR.Function(
        is_entry = true, name = "bar",
        body = (PTX.IR.Instruction(opcode = "ret",
                                   formatting = FormattingInfo(indent = "\t")),),
        formatting = FormattingInfo(preceding_comments = ("// hand-written",)),
    )
    @test occursin("// hand-written", format(f))
end
