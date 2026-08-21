using PTX.Parser
using PTX.Parser: tokenize, Token, TokenKind, LexError, ParseError
using PTX.Parser: parse as parse_ptx
using PTX.IR: Version, Target, AddressSize, Instruction, Label, Block,
              RegDecl, VarDecl, PragmaDirective, Comment, BlankLine, RawLine,
              RegisterOperand, ImmediateOperand, LabelOperand, VectorOperand,
              AddressOperand, ParenthesizedOperand, NegatedOperand,
              PipeOperand, Predicate, format,
              ScalarType, StateSpace, LinkingDirective

# Helper: only the "interesting" tokens (drop newlines, comments, TokenKind.EOF) so
# tests don't have to spell out trivial structure.
function _significant(toks)
    [t for t in toks if t.kind ∉ (TokenKind.NEWLINE, TokenKind.COMMENT, TokenKind.EOF)]
end

@testset "lexer: empty + TokenKind.EOF" begin
    @test tokenize("")[end].kind == TokenKind.EOF
    @test length(tokenize("")) == 1
end

@testset "lexer: simple instruction" begin
    toks = _significant(tokenize("add.f32 %r0, %r1, %r2;"))
    @test [t.kind for t in toks] ==
        [TokenKind.IDENTIFIER, TokenKind.DIRECTIVE, TokenKind.REGISTER, TokenKind.COMMA, TokenKind.REGISTER, TokenKind.COMMA, TokenKind.REGISTER, TokenKind.SEMICOLON]
    @test toks[1].text == "add"
    @test toks[2].text == ".f32"
    @test toks[3].text == "%r0"
end

@testset "lexer: chained ::-qualified directives" begin
    @test _significant(tokenize(".shared::cta"))[1].text == ".shared::cta"
    @test _significant(tokenize(".mbarrier::complete_tx::bytes"))[1].text ==
        ".mbarrier::complete_tx::bytes"
    # Must not greedily eat past a non-`::` colon.
    @test _significant(tokenize(".shared:foo"))[1].text == ".shared"
end

@testset "lexer: integer literals" begin
    @test _significant(tokenize("42"))[1].text == "42"
    @test _significant(tokenize("0xFF"))[1].text == "0xFF"
    @test _significant(tokenize("0Xdeadbeef"))[1].text == "0Xdeadbeef"
    @test _significant(tokenize("42"))[1].kind == TokenKind.INTEGER
    @test _significant(tokenize("0x10"))[1].kind == TokenKind.INTEGER
end

@testset "lexer: float literals" begin
    # decimal
    @test _significant(tokenize("3.14"))[1].kind == TokenKind.FLOAT
    @test _significant(tokenize("1.5e10"))[1].text == "1.5e10"
    @test _significant(tokenize("1.5e-3"))[1].text == "1.5e-3"
    # hex-encoded
    @test _significant(tokenize("0f3F800000"))[1].kind == TokenKind.FLOAT
    @test _significant(tokenize("0d3FF0000000000000"))[1].kind == TokenKind.FLOAT
end

@testset "lexer: register names" begin
    @test _significant(tokenize("%tid.x"))[1].text == "%tid.x"
    @test _significant(tokenize("%ntid.y"))[1].text == "%ntid.y"
    @test _significant(tokenize("%r0"))[1].text == "%r0"
    @test _significant(tokenize("%dynamic_smem_size"))[1].text == "%dynamic_smem_size"
end

@testset "lexer: comments preserved" begin
    toks = tokenize("// hello\nadd.f32 %r0;")
    @test toks[1].kind == TokenKind.COMMENT
    @test toks[1].text == "// hello"
    @test toks[2].kind == TokenKind.NEWLINE
    @test toks[3].kind == TokenKind.IDENTIFIER && toks[3].text == "add"
end

@testset "lexer: block comments" begin
    toks = tokenize("/* multi\nline */ add")
    @test toks[1].kind == TokenKind.COMMENT
    @test occursin("multi", toks[1].text)
    @test occursin("line", toks[1].text)
end

@testset "lexer: leading whitespace recorded" begin
    toks = _significant(tokenize("    add.f32"))
    @test toks[1].text == "add"
    @test toks[1].leading_whitespace == "    "
end

@testset "lexer: line/col tracking" begin
    toks = tokenize("add\n  sub")
    # `add` at line 1, col 1
    @test toks[1].line == 1 && toks[1].col == 1
    # newline -> line 2 starts; significant `sub` at line 2 col 3 (after 2 spaces)
    sub_tok = filter(t -> t.text == "sub", toks)[1]
    @test sub_tok.line == 2
    @test sub_tok.col == 3
end

@testset "lexer: round-trip token text reconstructs source" begin
    src = "add.f32 %r0, %r1, %r2;\nbar.sync 0;"
    toks = tokenize(src)
    rebuilt = join((t.leading_whitespace * t.text for t in toks if t.kind != TokenKind.EOF))
    @test rebuilt == src
end

@testset "lexer: punctuation map" begin
    @test _significant(tokenize("{ }"))[1].kind == TokenKind.LBRACE
    @test _significant(tokenize("{ }"))[2].kind == TokenKind.RBRACE
    @test _significant(tokenize("[ ] @ ! | ~ &"))[1].kind == TokenKind.LBRACKET
    @test _significant(tokenize("@%p0"))[1].kind == TokenKind.AT
end

@testset "lexer: rejects unknown characters" begin
    @test_throws LexError tokenize("\$\$\$#")  # `#` not in the punctuation table
end

@testset "lexer: multi-line PTX snippet" begin
    src = """
    .version 8.0
    .target sm_89
    .address_size 64

    .visible .entry foo()
    {
        .reg .b32 %r<3>;
        mov.u32 %r1, %tid.x;
        ret;
    }
    """
    toks = tokenize(src)
    sig = _significant(toks)
    # Smoke checks on a real-ish PTX module.
    @test any(t -> t.kind == TokenKind.DIRECTIVE && t.text == ".version", sig)
    @test any(t -> t.kind == TokenKind.DIRECTIVE && t.text == ".target", sig)
    @test any(t -> t.kind == TokenKind.IDENTIFIER && t.text == "sm_89", sig)
    @test any(t -> t.kind == TokenKind.IDENTIFIER && t.text == "foo", sig)
    @test any(t -> t.kind == TokenKind.REGISTER && t.text == "%tid.x", sig)
    # Round-trip preserves source byte-identical.
    rebuilt = join((t.leading_whitespace * t.text for t in toks if t.kind != TokenKind.EOF))
    @test rebuilt == src
end

# ===========================================================================
# Parser tests
# ===========================================================================

@testset "parser: module header" begin
    m = parse_ptx(""".version 8.5
.target sm_89
.address_size 64
""")
    @test m.version == Version(8, 5)
    @test m.target == Target(("sm_89",))
    @test m.address_size == AddressSize(64)
end

@testset "parser: target with multiple features" begin
    m = parse_ptx(""".version 8.0
.target sm_90a, debug
.address_size 64
""")
    @test m.target.targets == ("sm_90a", "debug")
end

@testset "parser: simple instruction" begin
    m = parse_ptx(""".version 8.0
.target sm_89
.address_size 64

add.f32 %r0, %r1, %r2;
""")
    insts = filter(d -> d isa Instruction, collect(m.directives))
    @test length(insts) == 1
    inst = insts[1]
    @test inst.opcode == "add"
    @test inst.modifiers == (".f32",)
    @test length(inst.operands) == 3
    @test inst.operands[1] == RegisterOperand("%r0")
    @test inst.operands[2] == RegisterOperand("%r1")
    @test inst.operands[3] == RegisterOperand("%r2")
    @test inst.predicate === nothing
end

@testset "parser: predicated instruction" begin
    m = parse_ptx(""".version 8.0
.target sm_89
.address_size 64

@%p0 bra L_END;
""")
    insts = filter(d -> d isa Instruction, collect(m.directives))
    @test length(insts) == 1
    @test insts[1].predicate == Predicate("%p0", false)

    m2 = parse_ptx(""".version 8.0
.target sm_89
.address_size 64

@!%p0 bra L_END;
""")
    @test filter(d -> d isa Instruction, collect(m2.directives))[1].predicate ==
        Predicate("%p0", true)
end

@testset "parser: vector + address operands (ldmatrix-style)" begin
    m = parse_ptx(""".version 8.0
.target sm_89
.address_size 64

ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%r0, %r1, %r2, %r3}, [%rd0];
""")
    insts = filter(d -> d isa Instruction, collect(m.directives))
    @test length(insts) == 1
    inst = insts[1]
    @test inst.opcode == "ldmatrix"
    @test inst.modifiers == (".sync", ".aligned", ".x4", ".m8n8", ".shared", ".b16")
    @test inst.operands[1] isa VectorOperand
    @test length((inst.operands[1]::VectorOperand).elements) == 4
    @test inst.operands[2] == AddressOperand("%rd0", nothing)
end

@testset "parser: address with offset" begin
    m = parse_ptx(""".version 8.0
.target sm_89
.address_size 64

ld.global.u32 %r0, [%rd1+16];
""")
    inst = filter(d -> d isa Instruction, collect(m.directives))[1]
    @test inst.operands[2] == AddressOperand("%rd1", "16")
end

@testset "parser: pipe operand (setp dual-pred)" begin
    m = parse_ptx(""".version 8.0
.target sm_89
.address_size 64

setp.lt.f32 %p0|%p1, %f0, %f1;
""")
    inst = filter(d -> d isa Instruction, collect(m.directives))[1]
    @test inst.operands[1] isa PipeOperand
    pop = inst.operands[1]::PipeOperand
    @test pop.left == RegisterOperand("%p0")
    @test pop.right == RegisterOperand("%p1")
end

@testset "parser: immediate operands preserve raw text" begin
    m = parse_ptx(""".version 8.0
.target sm_89
.address_size 64

mov.b32 %r0, 0xDEADBEEF;
mov.b32 %r1, 42;
mov.f32 %f0, 1.5;
mov.b64 %rd0, 0d3FF0000000000000;
""")
    insts = filter(d -> d isa Instruction, collect(m.directives))
    @test (insts[1].operands[2]::ImmediateOperand).text == "0xDEADBEEF"
    @test (insts[2].operands[2]::ImmediateOperand).text == "42"
    @test (insts[3].operands[2]::ImmediateOperand).text == "1.5"
    @test (insts[4].operands[2]::ImmediateOperand).text == "0d3FF0000000000000"
end

@testset "parser: register declarations" begin
    m = parse_ptx(""".version 8.0
.target sm_89
.address_size 64

.reg .b32 %r<10>;
.reg .pred %p0;
""")
    decls = filter(d -> d isa RegDecl, collect(m.directives))
    @test length(decls) == 2
    @test decls[1].type == ScalarType.B32
    @test decls[1].name == "%r"
    @test decls[1].count == 10
    @test decls[2].type == ScalarType.PRED
    @test decls[2].name == "%p0"
    @test decls[2].count === nothing
end

@testset "parser: var decls" begin
    m = parse_ptx(""".version 8.0
.target sm_89
.address_size 64

.shared .align 16 .b8 smem[4096];
.global .b32 counter = 0;
""")
    decls = filter(d -> d isa VarDecl, collect(m.directives))
    @test length(decls) == 2
    @test decls[1].state_space == StateSpace.SHARED
    @test decls[1].alignment == 16
    @test decls[1].array_size == 4096
    @test decls[1].name == "smem"
    @test decls[2].state_space == StateSpace.GLOBAL
    @test decls[2].initializer == ("0",)
end

@testset "parser: pragma" begin
    m = parse_ptx(""".version 8.0
.target sm_89
.address_size 64

.pragma "nounroll";
""")
    p = filter(d -> d isa PragmaDirective, collect(m.directives))[1]
    @test p.value == "nounroll"
end

@testset "parser: labels" begin
    m = parse_ptx(""".version 8.0
.target sm_89
.address_size 64

L_LOOP:
add.u32 %r0, %r0, 1;
""")
    labels = filter(d -> d isa Label, collect(m.directives))
    @test length(labels) == 1
    @test labels[1].name == "L_LOOP"
end

@testset "parser: label-as-operand" begin
    m = parse_ptx(""".version 8.0
.target sm_89
.address_size 64

bra L_END;
""")
    inst = filter(d -> d isa Instruction, collect(m.directives))[1]
    @test inst.operands[1] == LabelOperand("L_END")
end

@testset "parser: raw_line captures the source line" begin
    m = parse_ptx(""".version 8.0
.target sm_89
.address_size 64

    add.f32   %r0,    %r1,    %r2;  // weird spacing
""")
    inst = filter(d -> d isa Instruction, collect(m.directives))[1]
    @test occursin("add.f32", inst.formatting.raw_line)
    @test occursin("// weird spacing", inst.formatting.raw_line)
end

@testset "parser: header rejections" begin
    # Empty / no header at all.
    @test_throws ParseError parse_ptx("")
    @test_throws ParseError parse_ptx("\n\n   \n")

    # Missing one of the two required directives. address_size is optional.
    @test_throws ParseError parse_ptx(".target sm_89\n.address_size 64\n")
    @test_throws ParseError parse_ptx(".version 8.0\n.address_size 64\n")
    omitted = parse_ptx(".version 8.0\n.target sm_89\n")
    @test omitted.address_size == AddressSize(32)
    @test !omitted.address_size_explicit

    # Out-of-order header.
    @test_throws ParseError parse_ptx(".address_size 64\n.version 8.0\n.target sm_89\n")
    @test_throws ParseError parse_ptx(".target sm_89\n.version 8.0\n.address_size 64\n")

    # Malformed .version literal.
    @test_throws ParseError parse_ptx(".version oops\n.target sm_89\n.address_size 64\n")
    # Numeric-token adjacency is rejected at the lexer boundary rather than
    # being split into a plausible float and directive.
    @test_throws LexError parse_ptx(".version 8.x\n.target sm_89\n.address_size 64\n")
    @test_throws ParseError parse_ptx(".version 8 .y\n.target sm_89\n.address_size 64\n")
    @test_throws ParseError parse_ptx(".version -8.0\n.target sm_89\n.address_size 64\n")
    @test_throws ParseError parse_ptx(".version\n.target sm_89\n.address_size 64\n")

    # .target without identifier or with trailing comma.
    @test_throws ParseError parse_ptx(".version 8.0\n.target\n.address_size 64\n")
    @test_throws ParseError parse_ptx(".version 8.0\n.target sm_89,\n.address_size 64\n")
    @test_throws ParseError parse_ptx(".version 8.0\n.target .bogus\n.address_size 64\n")

    # .address_size with non-integer.
    @test_throws ParseError parse_ptx(".version 8.0\n.target sm_89\n.address_size oops\n")
    @test_throws ParseError parse_ptx(".version 8.0\n.target sm_89\n.address_size .b32\n")
    @test_throws ParseError parse_ptx(".version 8.0\n.target sm_89\n.address_size\n")
end

@testset "parser: garbage post-header → RawLine fallback" begin
    # The parser's contract: header is structural (throws on malformed); past
    # the header, anything that fails to parse structurally is captured as a
    # RawLine so the round-trip is lossless. These cases must NOT throw.
    head = """.version 8.0
              .target sm_89
              .address_size 64
              """
    fallback_cases = [
        "instr without semi"    => "add.f32 %r1, %r2, %r3",
        "regdecl without semi"  => ".reg .b32 %r0",
        "dangling comma"        => "add.f32 %r0, %r1, ;",
        "stray closing brace"   => "}",
        "unclosed brace"        => ".entry foo() { ret;",
        "bare directive"        => ".bogus_directive",
    ]
    for (label, body) in fallback_cases
        m = parse_ptx(head * body * "\n")
        @test any(d -> d isa RawLine, m.directives)
        # Round-trip preserves the malformed bytes verbatim — that's the
        # whole point of the RawLine escape.
        @test format(m) == head * body * "\n"
    end
end

@testset "parser: permissive opcode/operand acceptance" begin
    # By design, the parser doesn't whitelist opcodes (PTX has hundreds + new
    # arch-specific ones land every release). A bogus opcode that's still
    # syntactically shaped like an instruction parses as a structured
    # Instruction. ptxas — not us — is the gate on opcode validity.
    head = """.version 8.0
              .target sm_89
              .address_size 64
              """
    m = parse_ptx(head * "xyzzy.f32 %r1, %r2;\n")
    insts = filter(d -> d isa Instruction, collect(m.directives))
    @test length(insts) == 1
    @test insts[1].opcode == "xyzzy"

    # Same for operand types — the parser accepts any structurally-valid
    # operand list. Type/mode mismatches are ptxas's problem.
    m = parse_ptx(head * "add.f32 %r1, 0xDEADBEEF, 3.14;\n")
    insts = filter(d -> d isa Instruction, collect(m.directives))
    @test length(insts) == 1
    @test insts[1].opcode == "add"
end

@testset "lexer: lexical boundaries" begin
    # `#` owns a complete cpp line and `?` is ternary punctuation.
    @test tokenize("#")[1].kind == TokenKind.PREPROCESSOR
    @test tokenize("?")[1].kind == TokenKind.QUESTION
    @test_throws LexError tokenize("`")
    @test_throws LexError tokenize("\"unterminated")
    @test_throws LexError tokenize("αβγ")
    # Literal range is a semantic/use-site concern, not a lexical one.
    @test (tokenize("0xDEADBEEFCAFEBABE"); true)
end

# --- function-level parsing -----------------------------------------------

using PTX.IR: Function, Param, FunctionDirective

@testset "parser: .entry with linking and params" begin
    m = parse_ptx(""".version 8.0
.target sm_89
.address_size 64

.visible .entry add_one(
    .param .u64 in_ptr,
    .param .u64 out_ptr
)
{
    .reg .b32 %r<3>;
    ld.global.u32 %r1, [in_ptr];
    add.u32 %r2, %r1, 1;
    st.global.u32 [out_ptr], %r2;
    ret;
}
""")
    funs = filter(d -> d isa Function, collect(m.directives))
    @test length(funs) == 1
    f = funs[1]
    @test f.is_entry == true
    @test f.linking == LinkingDirective.VISIBLE
    @test f.name == "add_one"
    @test length(f.params) == 2
    @test f.params[1].state_space == StateSpace.PARAM
    @test f.params[1].type == ScalarType.U64
    @test f.params[1].name == "in_ptr"
    @test f.params[2].name == "out_ptr"
    @test length(f.body) >= 5    # 1 RegDecl + 4 Instructions at minimum
    @test any(st -> st isa RegDecl, f.body)
    @test any(st -> st isa Instruction && st.opcode == "ret", f.body)
end

@testset "parser: function directive (.maxnreg)" begin
    m = parse_ptx(""".version 8.0
.target sm_89
.address_size 64

.visible .entry foo()
.maxnreg 32
{ ret; }
""")
    f = first(filter(d -> d isa Function, collect(m.directives)))
    @test length(f.directives) == 1
    @test f.directives[1].name == "maxnreg"
    @test f.directives[1].values == (32,)
end

@testset "parser: function directive (.minperctamemory, PTX 9.4 §11.4.10)" begin
    source = """.version 9.4
.target sm_107
.address_size 64

.visible .entry foo()
.minperctamemory 2048
{
    .reg .u64 %a;
    .reg .u32 %b;
    mov.u64 %a, %perctamemoryoffset;
    mov.u32 %b, %perctamemorysize;
    ret;
}
"""
    m = parse_ptx(source)
    f = first(filter(d -> d isa Function, collect(m.directives)))
    @test length(f.directives) == 1
    @test f.directives[1].name == "minperctamemory"
    @test f.directives[1].values == (2048,)
    # Structural, not recovery: the entry header must not degrade to a raw
    # line, and the per-CTA-memory sregs are ordinary scalar sources. The
    # structural spelling joins function directives onto the header line.
    @test format(_deep_unraw(m)) ==
          replace(source, ".visible .entry foo()\n.minperctamemory 2048" =>
                          ".visible .entry foo() .minperctamemory 2048")
    @test format(m) == source
end

@testset "parser: PTX 9.4 intermediate-IR debug surface holds .loc parity" begin
    # .loc_intermediate (§11.5.5) and .nv_intermediate_source_section
    # (§11.5.6) are deliberately raw-tier, exactly like .loc/.file and the
    # .section debug constructs they extend: lossless round-trip, preserved
    # by canonicalize, no structural model. Promoting them is a separate
    # decision from 9.4 support; this pins the parity so a parser change
    # that regresses either path is loud.
    source = """.version 9.4
.target sm_107
.address_size 64
.file 1 "kernel.py"
.file 2 "kernel.tile"
.nv_intermediate_source_section {
.code_block {
.ir_name: "TileIR"
.sourceFileName: 2
.source_begin
tile.load %t0
.source_end
}
}
.entry k()
{
    .loc 1 10 5
    .loc_intermediate 2 4 1
    ret;
}
"""
    m = parse_ptx(source)
    @test format(m) == source
    canon = PTX.IR.canonicalize(m)
    @test occursin(".loc_intermediate 2 4 1", format(canon))
    @test occursin(".nv_intermediate_source_section", format(canon))
    @test occursin(".source_begin", format(canon))
end

@testset "parser: .func with return params" begin
    m = parse_ptx(""".version 8.0
.target sm_89
.address_size 64

.func (.param .b32 ret) helper(
    .param .b32 a
)
{
    ret;
}
""")
    f = first(filter(d -> d isa Function, collect(m.directives)))
    @test f.is_entry == false
    @test f.return_params !== nothing
    @test length(f.return_params) == 1
    @test f.return_params[1].name == "ret"
    @test length(f.params) == 1
    @test f.params[1].name == "a"
end

@testset "parser: .param .ptr .global .align" begin
    m = parse_ptx(""".version 8.0
.target sm_89
.address_size 64

.visible .entry kernel(
    .param .u64 .ptr .global .align 16 input
)
{ ret; }
""")
    f = first(filter(d -> d isa Function, collect(m.directives)))
    p = f.params[1]
    @test p.ptr_state_space == StateSpace.GLOBAL
    @test p.ptr_alignment == 16
end

@testset "parser: nested {} block in function body" begin
    m = parse_ptx(""".version 8.0
.target sm_89
.address_size 64

.visible .entry foo()
{
    .reg .b32 %r0;
    {
        .reg .b32 %r1;
        mov.b32 %r1, %r0;
    }
    ret;
}
""")
    f = first(filter(d -> d isa Function, collect(m.directives)))
    blocks = filter(st -> st isa Block, collect(f.body))
    @test length(blocks) == 1
    inner_body = blocks[1].body
    @test any(st -> st isa RegDecl && st.name == "%r1", inner_body)
    @test any(st -> st isa Instruction && st.opcode == "mov", inner_body)
end

@testset "parser: .extern .global with linking" begin
    m = parse_ptx(""".version 8.0
.target sm_89
.address_size 64

.extern .global .b32 counter;
""")
    vd = first(filter(d -> d isa VarDecl, collect(m.directives)))
    @test vd.linking == LinkingDirective.EXTERN
    @test vd.state_space == StateSpace.GLOBAL
    @test vd.name == "counter"
end

@testset "parser: empty function body" begin
    m = parse_ptx(""".version 8.0
.target sm_89
.address_size 64

.visible .entry empty()
{
}
""")
    f = first(filter(d -> d isa Function, collect(m.directives)))
    @test f.name == "empty"
    @test length(f.body) == 0 || all(st -> st isa BlankLine, f.body)
end

# ---------------------------------------------------------------------------
# v0.5-F: constant-expression operand promotion
#
# `((128 >> 4) << 16)` and friends appear in descriptor bit-packing (e.g.
# `corpus/less_slow_sm90a.ptx` wgmma descriptors). Without promotion they
# either fall to RawLine or wedge the operand grammar; with promotion they
# round-trip as a single ImmediateOperand.
# ---------------------------------------------------------------------------

# Helper: parse a single instruction body and return its operands tuple.
function _operands_of(src::AbstractString)
    m = parse_ptx(""".version 8.0
.target sm_89
.address_size 64
.visible .entry foo()
{
\t.reg .b64 %rd<2>;
\t$src
\tret;
}
""")
    f = first(filter(d -> d isa Function, collect(m.directives)))
    inst = first(s for s in f.body if s isa Instruction && s.opcode != "ret")
    inst.operands
end

@testset "parser: const-expr operand → ImmediateOperand" begin
    ops = _operands_of("or.b64 %rd0, %rd0, ((128 >> 4) << 16);")
    @test length(ops) == 3
    @test ops[3] isa ImmediateOperand
    @test ops[3].text == "((128 >> 4) << 16)"

    # right shift only
    ops = _operands_of("or.b64 %rd0, %rd0, (4096 >> 4);")
    @test ops[3] isa ImmediateOperand
    @test ops[3].text == "(4096 >> 4)"

    # left shift only
    ops = _operands_of("or.b64 %rd0, %rd0, (1 << 32);")
    @test ops[3] isa ImmediateOperand
    @test ops[3].text == "(1 << 32)"

    # Deeper nesting still captured as one immediate.
    ops = _operands_of("or.b64 %rd0, %rd0, (((1 << 4) << 8) << 16);")
    @test ops[3] isa ImmediateOperand
    @test ops[3].text == "(((1 << 4) << 8) << 16)"
end

@testset "parser: parenthesized non-shift stays ParenthesizedOperand" begin
    # call return list — must not be promoted.
    m = parse_ptx(""".version 8.0
.target sm_89
.address_size 64
.visible .entry foo()
{
\t.reg .b32 %r<2>;
\tcall.uni (%r0), bar;
\tret;
}
""")
    f = first(filter(d -> d isa Function, collect(m.directives)))
    inst = first(s for s in f.body if s isa Instruction && s.opcode == "call")
    @test inst.operands[1] isa ParenthesizedOperand
    @test inst.operands[1].elements[1] == RegisterOperand("%r0")
end

@testset "parser: const-expr round-trip is byte-identical" begin
    src = """.version 8.0
.target sm_90a
.address_size 64
.visible .entry foo()
{
\t.reg .b64 %rd<2>;
\tor.b64 %rd0, %rd0, ((128 >> 4) << 16);
\tret;
}
"""
    m = parse_ptx(src)
    @test format(m) == src                              # raw_source path
    @test format(PTX.IR.unraw(m)) == src                # structural path
end

@testset "parser: .language function directive (PTX ISA 9.3)" begin
    src = """.version 9.3
.target sm_121a
.address_size 64
.visible .entry foo()
.reqntid 128
.language 7
{
\tret;
}
"""
    m = parse_ptx(src)
    f = first(filter(d -> d isa Function, collect(m.directives)))
    @test PTX.IR.FunctionDirective("language", (7,)) in f.directives
    # The body must parse — before `.language` support, directive consumption
    # stopped short of `{` and the function silently kept an empty body.
    @test any(s -> s isa Instruction && s.opcode == "ret", f.body)
    @test format(m) == src

    # String and mixed forms per the spec.
    mixed = replace(src, ".language 7" => ".language \"tile ir\", 3")
    fm = first(filter(d -> d isa Function, collect(parse_ptx(mixed).directives)))
    @test PTX.IR.FunctionDirective("language", ("tile ir", 3)) in fm.directives
    @test format(PTX.IR.unraw(parse_ptx(mixed))) ==
          format(parse_ptx(format(PTX.IR.unraw(parse_ptx(mixed)))))
end

@testset "parser: module-level .section debug block" begin
    src = """.version 8.5
.target sm_90a
.address_size 64
.visible .entry foo()
{
\tret;
}
.section\t.debug_str
{
info_label:
.b8 105, 110, 102, 111
}
"""
    m = parse_ptx(src)
    sections = [d for d in m.directives if d isa PTX.IR.Section]
    @test length(sections) == 1
    @test sections[1].name == ".debug_str"
    @test occursin("info_label:", sections[1].raw)
    @test format(m) == src                              # raw_source path
    @test format(PTX.IR.unraw(m)) == src                # structural path
end
