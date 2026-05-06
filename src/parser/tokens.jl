using EnumX: @enumx

"""
    TokenKind

Token kinds produced by [`tokenize`](@ref). `EnumX`-nested so values are
referenced as `TokenKind.IDENTIFIER`, `TokenKind.NEWLINE`, etc.
"""
@enumx TokenKind begin
    INTEGER
    FLOAT
    STRING

    IDENTIFIER
    REGISTER

    # `.`-prefixed: type suffixes, state spaces, keywords (.version, .entry,
    # .func), including `::`-qualified forms (.shared::cta).
    DIRECTIVE

    LBRACE
    RBRACE
    LPAREN
    RPAREN
    LBRACKET
    RBRACKET
    COMMA
    SEMICOLON
    COLON
    AT
    BANG
    PLUS
    MINUS
    PIPE
    LESS
    GREATER
    EQUALS
    STAR
    SLASH
    TILDE
    AMPERSAND

    # Preserved for round-trip fidelity.
    NEWLINE
    COMMENT

    EOF
end

"""
    Token

A single token from the PTX source. `leading_whitespace` carries the
spaces/tabs that preceded this token on the same line, used by the
parser to reconstruct `FormattingInfo` for byte-identical round-trip.
"""
struct Token
    kind::TokenKind.T
    text::String
    line::Int
    col::Int
    leading_whitespace::String
end

Token(kind::TokenKind.T, text::AbstractString, line::Int, col::Int) =
    Token(kind, String(text), line, col, "")

function Base.show(io::IO, t::Token)
    print(io, "Token(", t.kind, ", ", repr(t.text), ", ", t.line, ":", t.col, ")")
end
