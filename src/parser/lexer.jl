# Ported from pyptx/parser/lexer.py (https://github.com/patrick-toulme/pyptx).
# Copyright 2026 Patrick Toulmé. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Translated to Julia and adapted.

"""
    LexError

Raised by [`tokenize`](@ref) on unrecognizable input.
"""
struct LexError <: Exception
    msg::String
    line::Int
    col::Int
end

Base.showerror(io::IO, e::LexError) =
    print(io, "LexError at ", e.line, ":", e.col, ": ", e.msg)

mutable struct Lexer
    src::String
    pos::Int          # 1-based byte index into src
    line::Int
    col::Int
end

Lexer(src::AbstractString) = Lexer(String(src), 1, 1, 1)

@inline _at_end(L::Lexer) = L.pos > lastindex(L.src)

@inline function _peek(L::Lexer, offset::Int = 0)
    i = L.pos + offset
    i > lastindex(L.src) && return '\0'
    L.src[i]
end

@inline function _advance!(L::Lexer)
    ch = L.src[L.pos]
    L.pos = nextind(L.src, L.pos)
    if ch == '\n'
        L.line += 1
        L.col = 1
    else
        L.col += 1
    end
    ch
end

@inline _is_hex_digit(ch::Char) =
    ('0' <= ch <= '9') || ('a' <= ch <= 'f') || ('A' <= ch <= 'F')

"""
    tokenize(source::AbstractString) -> Vector{Token}

Tokenize PTX source text. The token stream includes NEWLINE / COMMENT
tokens (needed for formatting preservation) and ends with an EOF token.
Raises [`LexError`](@ref) on unrecognizable characters.
"""
function tokenize(source::AbstractString)
    L = Lexer(source)
    tokens = Token[]

    while !_at_end(L)
        ws = _consume_horizontal_whitespace!(L)
        _at_end(L) && break

        ch = _peek(L)

        if ch == '\n'
            line, col = L.line, L.col
            _advance!(L)
            push!(tokens, Token(TokenKind.NEWLINE, "\n", line, col, ws))
            continue
        end

        if ch == '\r'
            _advance!(L)
            _peek(L) == '\n' && _advance!(L)
            continue
        end

        if ch == '/' && _peek(L, 1) == '*'
            push!(tokens, _lex_block_comment!(L, ws))
            continue
        end

        if ch == '/' && _peek(L, 1) == '/'
            push!(tokens, _lex_line_comment!(L, ws))
            continue
        end

        if ch == '"'
            push!(tokens, _lex_string!(L, ws))
            continue
        end

        if ch == '%'
            push!(tokens, _lex_register!(L, ws))
            continue
        end

        if ch == '.' && (isletter(_peek(L, 1)) || isdigit(_peek(L, 1)))
            push!(tokens, _lex_directive!(L, ws))
            continue
        end

        if isdigit(ch)
            push!(tokens, _lex_number!(L, ws))
            continue
        end

        if isletter(ch) || ch == '_' || ch == '$'
            push!(tokens, _lex_identifier!(L, ws))
            continue
        end

        push!(tokens, _lex_punctuation!(L, ws))
    end

    push!(tokens, Token(TokenKind.EOF, "", L.line, L.col, ""))
    tokens
end

function _consume_horizontal_whitespace!(L::Lexer)
    start = L.pos
    while !_at_end(L) && (_peek(L) == ' ' || _peek(L) == '\t')
        _advance!(L)
    end
    L.src[start:prevind(L.src, L.pos)]
end

function _lex_line_comment!(L::Lexer, ws::String)
    line, col = L.line, L.col
    start = L.pos
    while !_at_end(L) && _peek(L) != '\n'
        _advance!(L)
    end
    Token(TokenKind.COMMENT, L.src[start:prevind(L.src, L.pos)], line, col, ws)
end

function _lex_block_comment!(L::Lexer, ws::String)
    line, col = L.line, L.col
    start = L.pos
    _advance!(L)  # /
    _advance!(L)  # *
    while !_at_end(L)
        if _peek(L) == '*' && _peek(L, 1) == '/'
            _advance!(L); _advance!(L)
            break
        end
        _advance!(L)
    end
    Token(TokenKind.COMMENT, L.src[start:prevind(L.src, L.pos)], line, col, ws)
end

function _lex_string!(L::Lexer, ws::String)
    line, col = L.line, L.col
    start = L.pos
    _advance!(L)  # opening "
    while !_at_end(L)
        ch = _peek(L)
        if ch == '\\'
            _advance!(L)
            _at_end(L) || _advance!(L)
        elseif ch == '"'
            _advance!(L)
            break
        else
            _advance!(L)
        end
    end
    Token(TokenKind.STRING, L.src[start:prevind(L.src, L.pos)], line, col, ws)
end

function _lex_register!(L::Lexer, ws::String)
    line, col = L.line, L.col
    start = L.pos
    _advance!(L)  # %
    while !_at_end(L)
        ch = _peek(L)
        if isletter(ch) || isdigit(ch) || ch == '_'
            _advance!(L)
        elseif ch == '.' && isletter(_peek(L, 1))
            _advance!(L)
        else
            break
        end
    end
    Token(TokenKind.REGISTER, L.src[start:prevind(L.src, L.pos)], line, col, ws)
end

function _lex_directive!(L::Lexer, ws::String)
    line, col = L.line, L.col
    start = L.pos
    _advance!(L)  # .
    while !_at_end(L) && (isletter(_peek(L)) || isdigit(_peek(L)) || _peek(L) == '_')
        _advance!(L)
    end
    while _peek(L) == ':' && _peek(L, 1) == ':'
        _advance!(L); _advance!(L)
        while !_at_end(L) && (isletter(_peek(L)) || isdigit(_peek(L)) || _peek(L) == '_')
            _advance!(L)
        end
    end
    Token(TokenKind.DIRECTIVE, L.src[start:prevind(L.src, L.pos)], line, col, ws)
end

function _lex_number!(L::Lexer, ws::String)
    line, col = L.line, L.col
    start = L.pos

    # 0x... — integer hex literal
    if _peek(L) == '0' && (_peek(L, 1) == 'x' || _peek(L, 1) == 'X')
        _advance!(L); _advance!(L)
        while !_at_end(L) && _is_hex_digit(_peek(L))
            _advance!(L)
        end
        return Token(TokenKind.INTEGER, L.src[start:prevind(L.src, L.pos)], line, col, ws)
    end

    # 0d... — double-as-hex literal
    if _peek(L) == '0' && (_peek(L, 1) == 'd' || _peek(L, 1) == 'D')
        _advance!(L); _advance!(L)
        while !_at_end(L) && _is_hex_digit(_peek(L))
            _advance!(L)
        end
        return Token(TokenKind.FLOAT, L.src[start:prevind(L.src, L.pos)], line, col, ws)
    end

    # 0f... — float-as-hex literal (only when followed by hex digit)
    if _peek(L) == '0' && (_peek(L, 1) == 'f' || _peek(L, 1) == 'F') &&
            _is_hex_digit(_peek(L, 2))
        _advance!(L); _advance!(L)
        while !_at_end(L) && _is_hex_digit(_peek(L))
            _advance!(L)
        end
        return Token(TokenKind.FLOAT, L.src[start:prevind(L.src, L.pos)], line, col, ws)
    end

    while !_at_end(L) && isdigit(_peek(L))
        _advance!(L)
    end

    if _peek(L) == '.' && isdigit(_peek(L, 1))
        _advance!(L)
        while !_at_end(L) && isdigit(_peek(L))
            _advance!(L)
        end
        if _peek(L) == 'e' || _peek(L) == 'E'
            _advance!(L)
            (_peek(L) == '+' || _peek(L) == '-') && _advance!(L)
            while !_at_end(L) && isdigit(_peek(L))
                _advance!(L)
            end
        end
        return Token(TokenKind.FLOAT, L.src[start:prevind(L.src, L.pos)], line, col, ws)
    end

    Token(TokenKind.INTEGER, L.src[start:prevind(L.src, L.pos)], line, col, ws)
end

function _lex_identifier!(L::Lexer, ws::String)
    line, col = L.line, L.col
    start = L.pos
    while !_at_end(L)
        ch = _peek(L)
        (isletter(ch) || isdigit(ch) || ch == '_' || ch == '$') || break
        _advance!(L)
    end
    Token(TokenKind.IDENTIFIER, L.src[start:prevind(L.src, L.pos)], line, col, ws)
end

const _PUNCT = Dict{Char, TokenKind.T}(
    '{' => TokenKind.LBRACE,    '}' => TokenKind.RBRACE,
    '(' => TokenKind.LPAREN,    ')' => TokenKind.RPAREN,
    '[' => TokenKind.LBRACKET,  ']' => TokenKind.RBRACKET,
    ',' => TokenKind.COMMA,     ';' => TokenKind.SEMICOLON,
    ':' => TokenKind.COLON,     '@' => TokenKind.AT,
    '!' => TokenKind.BANG,      '+' => TokenKind.PLUS,
    '-' => TokenKind.MINUS,     '|' => TokenKind.PIPE,
    '*' => TokenKind.STAR,      '/' => TokenKind.SLASH,
    '~' => TokenKind.TILDE,     '&' => TokenKind.AMPERSAND,
    '<' => TokenKind.LESS,      '>' => TokenKind.GREATER,
    '=' => TokenKind.EQUALS,
)

function _lex_punctuation!(L::Lexer, ws::String)
    line, col = L.line, L.col
    ch = _advance!(L)
    kind = get(_PUNCT, ch, nothing)
    kind === nothing && throw(LexError("Unexpected character: $(repr(ch))", line, col))
    Token(kind, string(ch), line, col, ws)
end

@public tokenize, Token, TokenKind, LexError
