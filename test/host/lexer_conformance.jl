using PTX.Parser: tokenize, TokenKind, LexError
using PTX.Parser: parse as parse_ptx
using PTX.IR: RawLine, Function, format

_lexer_sig(source) = filter(tokenize(source)) do token
    token.kind ∉ (TokenKind.NEWLINE, TokenKind.COMMENT, TokenKind.EOF)
end

function _lex_error(source)
    try
        tokenize(source)
        nothing
    catch error
        error
    end
end

# PTX ISA 9.3 §4.5.1: C-like nonnegative integer literals, with an
# immediate uppercase U as the only type suffix. Signs remain unary tokens.
@testset "lexer conformance: integer grammar" begin
    legal = (
        "0", "0U", "42", "42U", "00", "077", "077U",
        "0x0", "0XdeadBEEF", "0xDEADBEEFU",
        "0b0", "0B101010", "0b101U",
    )
    for source in legal
        tokens = _lexer_sig(source)
        @test length(tokens) == 1
        @test only(tokens).kind == TokenKind.INTEGER
        @test only(tokens).text == source
    end

    signed = _lexer_sig("-42U + +0b1")
    @test [t.kind for t in signed] ==
        [TokenKind.MINUS, TokenKind.INTEGER, TokenKind.PLUS,
         TokenKind.PLUS, TokenKind.INTEGER]

    malformed = (
        "0x", "0xU", "0x1G", "0x1.tail", "0b", "0b2", "0b102",
        "08", "09U", "42u", "42UU", "123abc",
    )
    for source in malformed
        error = _lex_error(source)
        @test error isa LexError
        @test error.line == 1
        @test error.col == 1
    end
end

@testset "lexer conformance: PTX source is ASCII" begin
    for (source, line, col) in (("café", 1, 4), ("first\nα", 2, 1),
                                ("mov.u32 %r0, ١;", 1, 14))
        error = _lex_error(source)
        @test error isa LexError
        @test error.line == line
        @test error.col == col
        @test occursin("ASCII", error.msg)
    end

    tokens = tokenize(" \t\v\fadd.u32")
    @test tokens[1].text == "add"
    @test tokens[1].leading_whitespace == " \t\v\f"
end

# §4.5.2 allows a decimal point and a signed decimal exponent
# independently. Exact machine literals are fixed-width bit encodings.
@testset "lexer conformance: floating-point grammar" begin
    decimal = (
        "1.0", "1.", ".5", "0.", "1e2", "1E+2", "1e-2",
        "1.e2", ".5e+2", "123.456E-78",
    )
    exact = (
        "0f00000000", "0F3f800000", "0fFFFFFFFF",
        "0d0000000000000000", "0D3ff0000000000000",
        "0dFFFFFFFFFFFFFFFF",
    )
    for source in (decimal..., exact...)
        tokens = _lexer_sig(source)
        @test length(tokens) == 1
        @test only(tokens).kind == TokenKind.FLOAT
        @test only(tokens).text == source
    end

    malformed_decimal = (
        "1e", "1e+", "1e-", "1.0foo", "1..0",
    )
    malformed_exact = (
        "0f", "0f0000000", "0f000000000", "0f0000000g",
        "0d000000000000000", "0d00000000000000000",
        "0d000000000000000g", "0f3f800000U", "0f3f800000.tail",
    )
    for source in (malformed_decimal..., malformed_exact...)
        error = _lex_error(source)
        @test error isa LexError
        @test error.line == 1
        @test error.col == 1
    end
end

@testset "lexer conformance: numeric boundaries and constant punctuation" begin
    tokens = _lexer_sig("0x10+1e2,0f3f800000; .5")
    @test [t.kind for t in tokens] ==
        [TokenKind.INTEGER, TokenKind.PLUS, TokenKind.FLOAT, TokenKind.COMMA,
         TokenKind.FLOAT, TokenKind.SEMICOLON, TokenKind.FLOAT]

    modifiers = _lexer_sig(".3d .16x128b")
    @test all(t -> t.kind == TokenKind.DIRECTIVE, modifiers)

    # `%3` is itself a legal percent-prefixed identifier, so whitespace is
    # required to disambiguate the remainder operator before a digit.
    expression = _lexer_sig("(7 % 3)^1 ? 2 : 4")
    @test [t.kind for t in expression] ==
        [TokenKind.LPAREN, TokenKind.INTEGER, TokenKind.PERCENT,
         TokenKind.INTEGER, TokenKind.RPAREN, TokenKind.CARET,
         TokenKind.INTEGER, TokenKind.QUESTION, TokenKind.INTEGER,
         TokenKind.COLON, TokenKind.INTEGER]

    # A percent-prefixed identifier still follows the exact §4.4 boundary.
    @test only(_lexer_sig("%value\$1")).kind == TokenKind.REGISTER
    @test only(_lexer_sig("%value\$1")).text == "%value\$1"
end

@testset "lexer conformance: cpp lines are opaque and lossless" begin
    simple = "#include \"common.ptx\"\n"
    simple_tokens = tokenize(simple)
    @test simple_tokens[1].kind == TokenKind.PREPROCESSOR
    @test simple_tokens[1].text == "#include \"common.ptx\""
    @test simple_tokens[2].kind == TokenKind.NEWLINE

    commented = "#define VALUE 1 // an unmatched \" is comment text\n"
    @test tokenize(commented)[1].kind == TokenKind.PREPROCESSOR

    continued = "#define ADD(dst, a, b) " * "\\" *
                "\n    add.u32 dst, a, b;\n"
    continued_tokens = tokenize(continued)
    @test continued_tokens[1].kind == TokenKind.PREPROCESSOR
    @test continued_tokens[1].text == chomp(continued)
    @test continued_tokens[2].kind == TokenKind.NEWLINE
    @test continued_tokens[2].line == 2
    @test join((t.leading_whitespace * t.text for t in continued_tokens
                if t.kind != TokenKind.EOF)) == continued

    source = """#line 40 \"generated.ptx\"
.version 9.3
.target sm_89
.address_size 64
#define VALUE 42U
.visible .entry lexical_probe()
{
#if 1
    mov.u32 %r0, VALUE;
#endif
    ret;
}
"""
    parsed = parse_ptx(source)
    @test parsed.leading[1] isa RawLine
    @test parsed.leading[1].text == "#line 40 \"generated.ptx\""
    fn = only(filter(x -> x isa Function, parsed.directives))
    @test count(x -> x isa RawLine, fn.body) == 2
    @test format(parsed) == source

    split_header = """.version 9.3
#if __CUDA_ARCH__
.target sm_89
#endif
.address_size 64
"""
    @test format(parse_ptx(split_header)) == split_header

    for directive in ("#include", "#define", "#if", "#ifdef", "#else",
                      "#endif", "#line", "#file")
        @test tokenize(directive * " payload\n")[1].kind ==
            TokenKind.PREPROCESSOR
    end
end

@testset "lexer conformance: unterminated constructs fail at their start" begin
    cases = (
        ("prefix \"unterminated", 1, 8, "string"),
        ("first\n  /* unterminated", 2, 3, "comment"),
        ("#define X " * "\\", 1, 11, "continuation"),
        ("#include \"unterminated\n", 1, 10, "quoted"),
        ("#define X /* unterminated", 1, 11, "comment"),
    )
    for (source, line, col, fragment) in cases
        error = _lex_error(source)
        @test error isa LexError
        @test error.line == line
        @test error.col == col
        @test occursin(fragment, lowercase(error.msg))
        @test occursin("$line:$col", sprint(showerror, error))
    end
end

# This is intentionally independent of the parser corpus tests: every external
# source must remain byte-reconstructible from lexer tokens alone. It catches
# maximal-munch regressions even when parser raw-source snapshots mask them.
@testset "lexer conformance: external corpus token losslessness" begin
    root = joinpath(@__DIR__, "..", "corpus", "external")
    files = String[]
    for (dir, _, names) in walkdir(root), name in names
        endswith(name, ".ptx") && push!(files, joinpath(dir, name))
    end
    @test length(files) > 100
    for path in files
        source = read(path, String)
        tokens = tokenize(source)
        rebuilt = join((t.leading_whitespace * t.text for t in tokens
                        if t.kind != TokenKind.EOF))
        @test rebuilt == source
    end
end
