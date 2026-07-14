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

_at_end(L::Lexer) = L.pos > lastindex(L.src)

function _peek(L::Lexer, offset::Int = 0)
    i = L.pos + offset
    i > lastindex(L.src) && return '\0'
    L.src[i]
end

function _advance!(L::Lexer)
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

_is_hex_digit(ch::Char) =
    ('0' <= ch <= '9') || ('a' <= ch <= 'f') || ('A' <= ch <= 'F')
_is_octal_digit(ch::Char) = '0' <= ch <= '7'
_is_binary_digit(ch::Char) = ch == '0' || ch == '1'
_is_followsym(ch::Char) = isletter(ch) || isdigit(ch) || ch == '_' || ch == '$'

function _dot_starts_float(L::Lexer)
    isdigit(_peek(L, 1)) || return false
    offset = 1
    while isdigit(_peek(L, offset))
        offset += 1
    end
    ch = _peek(L, offset)
    # Digit-leading PTX modifiers such as `.3d` and `.16x128b` share the
    # prefix. Only e/E can introduce a decimal-float suffix after the digits.
    !(isletter(ch) || ch == '_') || ch == 'e' || ch == 'E'
end

_lexerror_at(line::Int, col::Int, message::AbstractString) =
    throw(LexError(String(message), line, col))

function _validate_ascii(source::AbstractString)
    line, col = 1, 1
    for ch in source
        isascii(ch) || _lexerror_at(line, col,
            "PTX source must be ASCII; found $(repr(ch))")
        if ch == '\n'
            line, col = line + 1, 1
        else
            col += 1
        end
    end
end

"""
    tokenize(source::AbstractString) -> Vector{Token}

Tokenize PTX source text. The token stream includes NEWLINE / COMMENT
tokens (needed for formatting preservation) and ends with an EOF token.
Raises [`LexError`](@ref) on unrecognizable characters.
"""
function tokenize(source::AbstractString)
    _validate_ascii(source)
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

        # PTX 9.3 §4.1 gives cpp ownership of complete lines beginning in
        # column one with `#`. Preserve the construct opaquely: the PTX parser
        # must not reinterpret tokens inside a macro replacement.
        if ch == '#' && L.col == 1
            push!(tokens, _lex_preprocessor!(L, ws))
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

        # A leading `%` requires a followsym for an identifier; a standalone
        # `%` is the remainder operator from §4.5.4.
        if ch == '%' && _is_followsym(_peek(L, 1))
            push!(tokens, _lex_register!(L, ws))
            continue
        end

        if ch == '.' && _dot_starts_float(L)
            push!(tokens, _lex_number!(L, ws))
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
    while !_at_end(L) && _peek(L) in (' ', '\t', '\v', '\f')
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
    closed = false
    while !_at_end(L)
        if _peek(L) == '*' && _peek(L, 1) == '/'
            _advance!(L); _advance!(L)
            closed = true
            break
        end
        _advance!(L)
    end
    closed || _lexerror_at(line, col, "Unterminated block comment")
    Token(TokenKind.COMMENT, L.src[start:prevind(L.src, L.pos)], line, col, ws)
end

function _lex_string!(L::Lexer, ws::String)
    line, col = L.line, L.col
    start = L.pos
    _advance!(L)  # opening "
    closed = false
    while !_at_end(L)
        ch = _peek(L)
        if ch == '\\'
            _advance!(L)
            _at_end(L) && _lexerror_at(line, col, "Unterminated string literal")
            (_peek(L) == '\n' || _peek(L) == '\r') &&
                _lexerror_at(line, col, "Unterminated string literal before newline")
            _advance!(L)
        elseif ch == '"'
            _advance!(L)
            closed = true
            break
        elseif ch == '\n' || ch == '\r'
            _lexerror_at(line, col, "Unterminated string literal before newline")
        else
            _advance!(L)
        end
    end
    closed || _lexerror_at(line, col, "Unterminated string literal")
    Token(TokenKind.STRING, L.src[start:prevind(L.src, L.pos)], line, col, ws)
end

function _lex_preprocessor!(L::Lexer, ws::String)
    line, col = L.line, L.col
    start = L.pos
    quote_char = '\0'
    quote_line, quote_col = line, col
    in_block_comment = false
    comment_line, comment_col = line, col

    while !_at_end(L)
        ch = _peek(L)

        if in_block_comment
            if ch == '*' && _peek(L, 1) == '/'
                _advance!(L); _advance!(L)
                in_block_comment = false
            else
                _advance!(L)
            end
            continue
        end

        if quote_char != '\0'
            if ch == '\\'
                _advance!(L)
                _at_end(L) && _lexerror_at(quote_line, quote_col,
                    "Unterminated quoted string in preprocessor directive")
                _advance!(L) # escaped byte, including a spliced newline
            elseif ch == quote_char
                _advance!(L)
                quote_char = '\0'
            elseif ch == '\n' || ch == '\r'
                _lexerror_at(quote_line, quote_col,
                    "Unterminated quoted string in preprocessor directive")
            else
                _advance!(L)
            end
            continue
        end

        if ch == '/' && _peek(L, 1) == '*'
            comment_line, comment_col = L.line, L.col
            _advance!(L); _advance!(L)
            in_block_comment = true
        elseif ch == '/' && _peek(L, 1) == '/'
            while !_at_end(L) && _peek(L) != '\n' && _peek(L) != '\r'
                _advance!(L)
            end
        elseif ch == '"' || ch == '\''
            quote_char = ch
            quote_line, quote_col = L.line, L.col
            _advance!(L)
        elseif ch == '\\' && _peek(L, 1) == '\n'
            _advance!(L); _advance!(L)
        elseif ch == '\\' && _peek(L, 1) == '\r' && _peek(L, 2) == '\n'
            _advance!(L); _advance!(L); _advance!(L)
        elseif ch == '\\' && _peek(L, 1) == '\0'
            _lexerror_at(L.line, L.col, "Unterminated preprocessor line continuation")
        elseif ch == '\n' || ch == '\r'
            break
        else
            _advance!(L)
        end
    end

    in_block_comment && _lexerror_at(comment_line, comment_col,
        "Unterminated block comment in preprocessor directive")
    quote_char != '\0' && _lexerror_at(quote_line, quote_col,
        "Unterminated quoted string in preprocessor directive")
    Token(TokenKind.PREPROCESSOR, L.src[start:prevind(L.src, L.pos)], line, col, ws)
end

function _lex_register!(L::Lexer, ws::String)
    line, col = L.line, L.col
    start = L.pos
    _advance!(L)  # %
    while !_at_end(L)
        ch = _peek(L)
        if isletter(ch) || isdigit(ch) || ch == '_' || ch == '$'
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

    # PTX exact machine floats have a fixed payload width. Validate the whole
    # spelling here so a short/long encoding cannot split into plausible
    # tokens and survive through RawLine fallback.
    if _peek(L) == '0' && (_peek(L, 1) == 'f' || _peek(L, 1) == 'F' ||
                          _peek(L, 1) == 'd' || _peek(L, 1) == 'D')
        prefix = _peek(L, 1)
        digits = prefix == 'f' || prefix == 'F' ? 8 : 16
        _advance!(L); _advance!(L)
        for _ in 1:digits
            _is_hex_digit(_peek(L)) || _lexerror_at(line, col,
                "Exact $(digits == 8 ? "single" : "double")-precision literal " *
                "requires exactly $digits hexadecimal digits")
            _advance!(L)
        end
        (_is_followsym(_peek(L)) || _peek(L) == '.') && _lexerror_at(line, col,
            "Exact floating-point literal has trailing characters")
        return Token(TokenKind.FLOAT, L.src[start:prevind(L.src, L.pos)], line, col, ws)
    end

    # C-like base-prefixed integer literals from §4.5.1.
    if _peek(L) == '0' && (_peek(L, 1) == 'x' || _peek(L, 1) == 'X' ||
                          _peek(L, 1) == 'b' || _peek(L, 1) == 'B')
        prefix = _peek(L, 1)
        is_hex = prefix == 'x' || prefix == 'X'
        valid_digit = is_hex ? _is_hex_digit : _is_binary_digit
        base_name = is_hex ? "hexadecimal" : "binary"
        _advance!(L); _advance!(L)
        valid_digit(_peek(L)) || _lexerror_at(line, col,
            "$base_name integer literal requires at least one digit")
        while valid_digit(_peek(L))
            _advance!(L)
        end
        _peek(L) == 'U' && _advance!(L)
        (_is_followsym(_peek(L)) || _peek(L) == '.') && _lexerror_at(line, col,
            "Invalid digit or suffix in $base_name integer literal")
        return Token(TokenKind.INTEGER, L.src[start:prevind(L.src, L.pos)], line, col, ws)
    end

    starts_with_dot = _peek(L) == '.'
    leading_zero = !starts_with_dot && _peek(L) == '0'
    digit_count = 0
    invalid_octal_digit = false
    while !_at_end(L) && isdigit(_peek(L))
        invalid_octal_digit |= leading_zero && !_is_octal_digit(_peek(L))
        _advance!(L)
        digit_count += 1
    end

    saw_point = false
    if _peek(L) == '.'
        saw_point = true
        _advance!(L)
        while !_at_end(L) && isdigit(_peek(L))
            _advance!(L)
        end
    end

    saw_exponent = false
    if _peek(L) == 'e' || _peek(L) == 'E'
        saw_exponent = true
        _advance!(L)
        (_peek(L) == '+' || _peek(L) == '-') && _advance!(L)
        isdigit(_peek(L)) || _lexerror_at(line, col,
            "Floating-point exponent requires at least one digit")
        while !_at_end(L) && isdigit(_peek(L))
            _advance!(L)
        end
    end

    if saw_point || saw_exponent
        (_is_followsym(_peek(L)) || _peek(L) == '.') && _lexerror_at(line, col,
            "Floating-point literal has trailing characters")
        return Token(TokenKind.FLOAT, L.src[start:prevind(L.src, L.pos)], line, col, ws)
    end

    # A bare decimal zero is valid by the C syntax incorporated by §4.5.1;
    # a multi-digit spelling beginning with zero is octal.
    invalid_octal_digit && digit_count > 1 && _lexerror_at(line, col,
        "Invalid digit in octal integer literal")
    _peek(L) == 'U' && _advance!(L)
    (_is_followsym(_peek(L)) || _peek(L) == '.') && _lexerror_at(line, col,
        "Invalid suffix in integer literal; PTX permits only uppercase U")
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
    '^' => TokenKind.CARET,     '?' => TokenKind.QUESTION,
    '%' => TokenKind.PERCENT,
)

function _lex_punctuation!(L::Lexer, ws::String)
    line, col = L.line, L.col
    ch = _advance!(L)
    kind = get(_PUNCT, ch, nothing)
    kind === nothing && throw(LexError("Unexpected character: $(repr(ch))", line, col))
    Token(kind, string(ch), line, col, ws)
end

@public tokenize, Token, TokenKind, LexError
