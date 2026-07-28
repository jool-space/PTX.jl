# Ported from pyptx/parser/parser.py (https://github.com/patrick-toulme/pyptx).
# Copyright 2026 Patrick Toulmé. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Translated to Julia and adapted.

using ..IR: Module, Version, Target, AddressSize, TargetDirective, FormattingInfo,
            Instruction, Label, RegDecl, VarDecl, PragmaDirective,
            Comment, BlankLine, RawLine, Block, Statement,
            Operand, RegisterOperand, ImmediateOperand, LabelOperand,
            VectorOperand, AddressOperand, ParenthesizedOperand,
            NegatedOperand, PipeOperand, Predicate,
            Param, FunctionDirective, Function,
            StateSpace, VectorShape, LinkingDirective,
            scalar_type_from_ptx, vector_shape_from_ptx,
            validate_vector_declaration,
            state_space_from_ptx, linking_from_ptx

"""
    ParseError

Raised by [`parse`](@ref) on input the recursive-descent parser cannot
consume. Carries the source `line` and `col` of the failed token.
"""
struct ParseError <: Exception
    msg::String
    line::Int
    col::Int
end

Base.showerror(io::IO, e::ParseError) =
    print(io, "ParseError at ", e.line, ":", e.col, ": ", e.msg)

mutable struct ParserState
    tokens::Vector{Token}
    pos::Int                              # 1-based index into tokens
    source::String
    source_lines::Vector{String}          # split on '\n', 0-padded boundary
    pending_stmts::Vector{Statement}      # stash for multi-stmt productions
    collected_comments::Vector{String}    # comments before the next statement
end

function ParserState(tokens::Vector{Token}, source::AbstractString = "")
    src = String(source)
    lines = isempty(src) ? String[] : split(src, '\n', keepempty = true)
    ParserState(tokens, 1, src, String.(lines), Statement[], String[])
end

_peek(s::ParserState) = s.tokens[s.pos]
_peek_kind(s::ParserState) = s.tokens[s.pos].kind
_peek_offset(s::ParserState, off::Int) = s.tokens[s.pos + off]

function _advance!(s::ParserState)
    t = s.tokens[s.pos]
    s.pos += 1
    t
end

function _at_end(s::ParserState)
    s.pos > length(s.tokens) || _peek_kind(s) == TokenKind.EOF
end

function _err(s::ParserState, msg::AbstractString)
    t = _peek(s)
    ParseError(String(msg), t.line, t.col)
end

function _expect!(s::ParserState, kind::TokenKind.T, text::Union{String, Nothing} = nothing)
    t = _peek(s)
    t.kind == kind || throw(_err(s, "Expected $kind, got $(t.kind) ($(repr(t.text)))"))
    text === nothing || t.text == text ||
        throw(_err(s, "Expected $(repr(text)), got $(repr(t.text))"))
    _advance!(s)
end

function _match!(s::ParserState, kind::TokenKind.T, text::Union{String, Nothing} = nothing)
    t = _peek(s)
    t.kind == kind || return nothing
    text === nothing || t.text == text || return nothing
    _advance!(s)
end

# Comments and blank lines are stashed onto `collected_comments` for the next
# statement to pick up.
function _skip_newlines_and_comments!(s::ParserState)
    blank_lines = 0
    prev_was_newline = false
    while _peek_kind(s) in (TokenKind.NEWLINE, TokenKind.COMMENT,
                            TokenKind.PREPROCESSOR)
        t = _peek(s)
        if t.kind == TokenKind.NEWLINE
            if prev_was_newline
                blank_lines += 1
                push!(s.collected_comments, "")    # blank-line marker
            end
            prev_was_newline = true
        else
            push!(s.collected_comments, t.leading_whitespace * t.text)
            prev_was_newline = false
        end
        _advance!(s)
    end
    blank_lines
end

function _take_comments!(s::ParserState)
    cs = Tuple(s.collected_comments)
    empty!(s.collected_comments)
    cs
end

# After `;`, consume a same-line trailing comment so it isn't re-collected as
# the next statement's preceding comment (which would emit it twice — raw_line
# on this stmt + preceding-comment on next stmt).
function _take_same_line_trailing_comment!(s::ParserState, end_line::Int)
    if _peek_kind(s) == TokenKind.COMMENT && _peek(s).line == end_line
        text = _peek(s).leading_whitespace * _peek(s).text
        _advance!(s)
        return text
    end
    ""
end

"""
    parse(source::AbstractString) -> IR.Module

Parse PTX source text into a `Module`. Module-header (.version / .target /
.address_size) and supported module/function/body statements are parsed into
structured IR; blank lines and comments are preserved. Instructions do not
require a known-opcode inventory. A post-header statement that fails the
fault-tolerant statement parser is generally retained as an opaque `RawLine`,
but lexical errors, invalid initial headers, malformed or misplaced
`.version`/`.target`/`.address_size` directives, and target invariants still
throw. `RawLine` preserves text only and is not a semantic parse.

The original nonempty input is retained in `Module.raw_source`, so formatting
an unmodified parsed module is byte-identical without exercising field-driven
structural reconstruction.
"""
function parse(source::AbstractString)
    tokens = tokenize(source)
    s = ParserState(tokens, source)

    leading = Statement[]
    prev_nl = false
    while _peek_kind(s) in (TokenKind.NEWLINE, TokenKind.COMMENT, TokenKind.PREPROCESSOR)
        t = _peek(s)
        if t.kind == TokenKind.NEWLINE
            prev_nl && push!(leading, BlankLine())
            prev_nl = true
        elseif t.kind == TokenKind.COMMENT
            push!(leading, Comment(t.leading_whitespace * t.text))
            prev_nl = false
        else
            # cpp owns the line; retain it without pretending it is a PTX
            # comment or interpreting its replacement-list tokens.
            push!(leading, RawLine(t.leading_whitespace * t.text))
            prev_nl = false
        end
        _advance!(s)
    end

    version      = _parse_version!(s)
    _skip_inline_comment!(s); _skip_newlines_and_comments!(s)
    target       = _parse_target!(s, version)
    _skip_inline_comment!(s); _skip_newlines_and_comments!(s)
    address_size_explicit = _peek_kind(s) == TokenKind.DIRECTIVE &&
                            _peek(s).text == ".address_size"
    address_size = if address_size_explicit
        parsed = _parse_address_size!(s, version)
        _skip_inline_comment!(s)
        parsed
    else
        AddressSize(32)
    end

    raw_header = _find_raw_header(s, address_size_explicit)

    # Comments skipped while probing for an optional `.address_size` belong to
    # the header when it is present.  When it is omitted they are ordinary
    # ordered module content and must survive structural (raw-free) emission.
    directives = Statement[]
    if address_size_explicit
        empty!(s.collected_comments)
    else
        for text in _take_comments!(s)
            push!(directives, isempty(text) ? BlankLine() : Comment(text))
        end
    end
    prev_nl = false
    while _peek_kind(s) != TokenKind.EOF
        t = _peek(s)
        if t.kind == TokenKind.NEWLINE
            prev_nl && push!(directives, BlankLine())
            prev_nl = true
            _advance!(s)
            continue
        end
        if t.kind == TokenKind.COMMENT
            push!(directives, Comment(t.leading_whitespace * t.text))
            _advance!(s)
            prev_nl = false
            continue
        end
        if t.kind == TokenKind.PREPROCESSOR
            push!(directives, RawLine(t.leading_whitespace * t.text))
            _advance!(s)
            prev_nl = false
            continue
        end
        prev_nl = false
        if t.kind == TokenKind.DIRECTIVE && t.text == ".target"
            push!(directives, _parse_later_target!(s, version))
            continue
        elseif _is_module_header_directive(t)
            throw(_err(s, "$(t.text) is not permitted after the module header"))
        end
        saved_pos = s.pos
        try
            push!(directives, _parse_top_level_statement!(s))
        catch e
            e isa ParseError || rethrow()
            _parse_error_points_to_module_directive(s, e) && rethrow()
            # Rewind so raw-line capture sees every token of the failed line.
            s.pos = saved_pos
            push!(directives, _capture_raw_line!(s, t.leading_whitespace))
        end
    end

    _validate_module_target_options!(s, target, Tuple(directives))

    Module(
        version = version,
        target = target,
        address_size = address_size,
        address_size_explicit = address_size_explicit,
        leading = Tuple(leading),
        directives = Tuple(directives),
        raw_header = raw_header,
        raw_source = isempty(s.source) ? nothing : s.source,
    )
end

function _skip_inline_comment!(s::ParserState)
    if _peek_kind(s) == TokenKind.COMMENT
        _advance!(s)
    end
end

function _parse_version!(s::ParserState)
    _expect!(s, TokenKind.DIRECTIVE, ".version")
    _skip_newlines_and_comments!(s)
    t = _peek(s)
    if t.kind == TokenKind.FLOAT
        _advance!(s)
        parts = split(t.text, '.')
        major = tryparse(Int, parts[1])
        minor = length(parts) >= 2 ? tryparse(Int, parts[2]) : nothing
        (major === nothing || minor === nothing) &&
            throw(_err(s, "Expected version number, got malformed float $(repr(t.text))"))
        return Version(major, minor)
    elseif t.kind == TokenKind.INTEGER
        major = tryparse(Int, _advance!(s).text)
        major === nothing &&
            throw(_err(s, "Expected version number, got malformed integer"))
        minor_tok = _expect!(s, TokenKind.DIRECTIVE)        # ".5" form
        minor = tryparse(Int, lstrip(minor_tok.text, '.'))
        minor === nothing &&
            throw(_err(s, "Expected version minor, got $(repr(minor_tok.text))"))
        return Version(major, minor)
    end
    throw(_err(s, "Expected version number, got $(t.kind)"))
end

function _parse_target!(s::ParserState, version::Version)
    directive = _expect!(s, TokenKind.DIRECTIVE, ".target")
    _skip_newlines_and_comments!(s)
    targets = String[_expect!(s, TokenKind.IDENTIFIER).text]
    while _match!(s, TokenKind.COMMA) !== nothing
        _skip_newlines_and_comments!(s)
        push!(targets, _expect!(s, TokenKind.IDENTIFIER).text)
    end
    next = _peek(s)
    if next.line == directive.line &&
       next.kind ∉ (TokenKind.NEWLINE, TokenKind.COMMENT, TokenKind.EOF,
                    TokenKind.DIRECTIVE)
        throw(_err(s, ".target specifiers must be comma-separated"))
    end
    _validate_target!(s, Target(Tuple(targets)), version)
end

function _parse_address_size!(s::ParserState, version::Version)
    _expect!(s, TokenKind.DIRECTIVE, ".address_size")
    _version_at_least(version, Version(2, 3)) ||
        throw(_err(s, ".address_size requires PTX ISA 2.3 or later"))
    _skip_newlines_and_comments!(s)
    tok = _expect!(s, TokenKind.INTEGER)
    tok.text in ("32", "64") ||
        throw(_err(s, ".address_size must be the decimal token 32 or 64, got $(repr(tok.text))"))
    size = Base.parse(Int, tok.text)
    next = _peek(s)
    if next.line == tok.line &&
       next.kind ∉ (TokenKind.NEWLINE, TokenKind.COMMENT, TokenKind.EOF,
                    TokenKind.DIRECTIVE)
        throw(_err(s, "unexpected token after .address_size"))
    end
    AddressSize(size)
end

_is_module_header_directive(t::Token) =
    t.kind == TokenKind.DIRECTIVE && t.text in (".version", ".address_size")
_is_module_directive(t::Token) =
    t.kind == TokenKind.DIRECTIVE && t.text in (".version", ".target", ".address_size")

function _parse_later_target!(s::ParserState, version::Version)
    start_line = _peek(s).line
    prev_pos = s.pos
    indent = _peek(s).leading_whitespace
    target = _parse_target!(s, version)
    trailing = _take_same_line_trailing_comment!(s, start_line)
    raw_line = (isempty(s.source_lines) || !_owns_line(s, prev_pos, start_line, start_line)) ?
               nothing : _extract_source(s, start_line, start_line)
    TargetDirective(target = target,
                    formatting = FormattingInfo(indent = indent,
                                                trailing = trailing,
                                                raw_line = raw_line))
end

const _VAR_DECL_DIRECTIVES = Set{String}((
    ".global", ".shared", ".local", ".const", ".param",
    ".shared::cta", ".shared::cluster",
))
const _LINKING_DIRECTIVES = Set{String}((".visible", ".extern", ".weak", ".common"))

function _parse_top_level_statement!(s::ParserState)
    t = _peek(s)
    indent = t.leading_whitespace

    if t.kind == TokenKind.DIRECTIVE && t.text in _LINKING_DIRECTIVES
        return _parse_function_or_global_with_linking!(s)
    end

    if t.kind == TokenKind.DIRECTIVE && (t.text == ".entry" || t.text == ".func")
        return _parse_function!(s, nothing)
    end

    return _parse_statement!(s, indent)
end

function _parse_statement!(s::ParserState, indent::String)
    t = _peek(s)
    if t.kind == TokenKind.DIRECTIVE &&
       t.text in (".version", ".target", ".address_size")
        throw(_err(s, "module directive $(t.text) is not permitted in this scope"))
    end
    if t.kind == TokenKind.AT
        return _parse_instruction!(s, indent)
    end
    if t.kind == TokenKind.DIRECTIVE
        if t.text == ".reg"
            return _parse_reg_decl!(s, indent)
        end
        if t.text in _VAR_DECL_DIRECTIVES
            return _parse_var_decl!(s, indent)
        end
        if t.text == ".pragma"
            return _parse_pragma!(s, indent)
        end
        return _parse_instruction!(s, indent)
    end
    if t.kind == TokenKind.IDENTIFIER
        return _parse_label_or_instruction!(s, indent)
    end
    throw(_err(s, "Expected statement, got $(t.kind) ($(repr(t.text)))"))
end

function _parse_label_or_instruction!(s::ParserState, indent::String)
    if _peek_kind(s) == TokenKind.IDENTIFIER
        saved = s.pos
        ident = _advance!(s)
        _skip_newlines_and_comments!(s)
        if _peek_kind(s) == TokenKind.COLON
            _advance!(s)
            preceding = _take_comments!(s)
            return Label(
                name = ident.text,
                formatting = FormattingInfo(indent = indent,
                                             preceding_comments = preceding),
            )
        end
        s.pos = saved   # backtrack
    end
    _parse_instruction!(s, indent)
end

function _parse_predicate!(s::ParserState)
    _expect!(s, TokenKind.AT)
    negated = _match!(s, TokenKind.BANG) !== nothing
    t = _peek(s)
    (t.kind == TokenKind.REGISTER || t.kind == TokenKind.IDENTIFIER) ||
        throw(_err(s, "Expected predicate register, got $(t.kind) ($(repr(t.text)))"))
    Predicate(_advance!(s).text, negated)
end

function _parse_instruction!(s::ParserState, indent::String)
    start_line = _peek(s).line
    prev_pos = s.pos

    pred = _peek_kind(s) == TokenKind.AT ? _parse_predicate!(s) : nothing

    t = _peek(s)
    if t.kind == TokenKind.IDENTIFIER || t.kind == TokenKind.DIRECTIVE
        opcode = _advance!(s).text
    else
        throw(_err(s, "Expected opcode, got $(t.kind) ($(repr(t.text)))"))
    end

    modifiers = String[]
    while _peek_kind(s) == TokenKind.DIRECTIVE
        push!(modifiers, _advance!(s).text)
    end

    _skip_newlines_and_comments!(s)

    operands = Operand[]
    if _peek_kind(s) != TokenKind.SEMICOLON
        operands = _parse_operand_list!(s)
    end

    _skip_newlines_and_comments!(s)
    semi = _expect!(s, TokenKind.SEMICOLON)
    end_line = semi.line

    trailing = _take_same_line_trailing_comment!(s, end_line)

    raw_line = (isempty(s.source_lines) || !_owns_line(s, prev_pos, start_line, end_line)) ?
                nothing : _extract_source(s, start_line, end_line)
    preceding = _take_comments!(s)

    Instruction(
        opcode = opcode,
        modifiers = Tuple(modifiers),
        operands = Tuple(operands),
        predicate = pred,
        formatting = FormattingInfo(
            indent = indent,
            trailing = trailing,
            preceding_comments = preceding,
            raw_line = raw_line,
        ),
    )
end

function _parse_operand_list!(s::ParserState)
    operands = Operand[]
    _skip_newlines_and_comments!(s)
    push!(operands, _parse_operand_with_pipe!(s))
    while _match!(s, TokenKind.COMMA) !== nothing
        _skip_newlines_and_comments!(s)
        push!(operands, _parse_operand_with_pipe!(s))
    end
    operands
end

function _parse_operand_with_pipe!(s::ParserState)
    left = _parse_single_operand!(s)
    if _peek_kind(s) == TokenKind.PIPE
        _advance!(s)
        right = _parse_single_operand!(s)
        return PipeOperand(left, right)
    end
    left
end

function _parse_single_operand!(s::ParserState)
    t = _peek(s)

    if t.kind == TokenKind.BANG
        _advance!(s)
        return NegatedOperand(_parse_single_operand!(s))
    end

    if t.kind == TokenKind.LBRACE
        return _parse_vector_operand!(s)
    end

    if t.kind == TokenKind.LPAREN
        return _parse_parenthesized_operand!(s)
    end

    if t.kind == TokenKind.LBRACKET
        return _parse_address_operand!(s)
    end

    if t.kind == TokenKind.REGISTER
        _advance!(s)
        return RegisterOperand(t.text)
    end

    if t.kind in (TokenKind.INTEGER, TokenKind.FLOAT)
        _advance!(s)
        return ImmediateOperand(t.text)
    end

    if t.kind == TokenKind.MINUS
        _advance!(s)
        nt = _peek(s)
        if nt.kind in (TokenKind.INTEGER, TokenKind.FLOAT)
            _advance!(s)
            return ImmediateOperand("-" * nt.text)
        end
        throw(_err(s, "Expected number after '-', got $(repr(nt.text))"))
    end

    if t.kind == TokenKind.IDENTIFIER
        _advance!(s)
        return LabelOperand(t.text)
    end

    throw(_err(s, "Expected operand, got $(t.kind) ($(repr(t.text)))"))
end

function _parse_vector_operand!(s::ParserState)
    _expect!(s, TokenKind.LBRACE)
    _skip_newlines_and_comments!(s)
    elements = Operand[]
    if _peek_kind(s) != TokenKind.RBRACE
        push!(elements, _parse_single_operand!(s))
        while _match!(s, TokenKind.COMMA) !== nothing
            _skip_newlines_and_comments!(s)
            push!(elements, _parse_single_operand!(s))
        end
    end
    _skip_newlines_and_comments!(s)
    _expect!(s, TokenKind.RBRACE)
    VectorOperand(Tuple(elements))
end

function _parse_parenthesized_operand!(s::ParserState)
    # `<<` or `>>` before the closing paren ⇒ constant expression
    # (`((128 >> 4) << 16)` for descriptor bit-packing). Promote to
    # ImmediateOperand so the whole expression formats verbatim.
    if _is_const_expr_ahead(s)
        return _parse_const_expr!(s)
    end
    _expect!(s, TokenKind.LPAREN)
    _skip_newlines_and_comments!(s)
    elements = Operand[]
    if _peek_kind(s) != TokenKind.RPAREN
        push!(elements, _parse_operand_with_pipe!(s))
        _skip_newlines_and_comments!(s)
        while _match!(s, TokenKind.COMMA) !== nothing
            _skip_newlines_and_comments!(s)
            push!(elements, _parse_operand_with_pipe!(s))
            _skip_newlines_and_comments!(s)
        end
    end
    _expect!(s, TokenKind.RPAREN)
    ParenthesizedOperand(Tuple(elements))
end

function _is_const_expr_ahead(s::ParserState)
    depth = 0
    p = s.pos
    n = length(s.tokens)
    while p <= n
        t = s.tokens[p]
        if t.kind == TokenKind.LPAREN
            depth += 1
        elseif t.kind == TokenKind.RPAREN
            depth -= 1
            depth == 0 && return false
        elseif (t.kind == TokenKind.GREATER || t.kind == TokenKind.LESS) &&
               p + 1 <= n && s.tokens[p + 1].kind == t.kind
            return true
        elseif t.kind == TokenKind.EOF
            return false
        end
        p += 1
    end
    false
end

function _parse_const_expr!(s::ParserState)
    parts = String[]
    depth = 0
    while !_at_end(s)
        t = _peek(s)
        if t.kind == TokenKind.LPAREN
            depth += 1
        elseif t.kind == TokenKind.RPAREN
            depth -= 1
        end
        push!(parts, t.leading_whitespace)
        push!(parts, t.text)
        _advance!(s)
        depth == 0 && break
    end
    ImmediateOperand(String(strip(join(parts))))
end

function _parse_address_operand!(s::ParserState)
    _expect!(s, TokenKind.LBRACKET)
    t = _peek(s)
    base = if t.kind == TokenKind.REGISTER || t.kind == TokenKind.IDENTIFIER
        _advance!(s).text
    else
        throw(_err(s, "Expected register or symbol in address, got $(repr(t.text))"))
    end

    # TMA tensor-coordinate form `[base, {c1, c2, ...}]` — structured, so
    # canonical renaming reaches the coordinate registers.
    if _peek_kind(s) == TokenKind.COMMA &&
       s.pos + 1 <= length(s.tokens) &&
       s.tokens[s.pos + 1].kind == TokenKind.LBRACE
        _advance!(s)   # the comma
        coords = _parse_vector_operand!(s).elements
        _expect!(s, TokenKind.RBRACKET)
        return AddressOperand(base, nothing, coords)
    end

    offset::Union{String, Nothing} = nothing
    if _peek_kind(s) != TokenKind.RBRACKET
        parts = String[]
        depth = 0
        while true
            tt = _peek(s)
            if tt.kind == TokenKind.RBRACKET && depth == 0
                break
            end
            tt.kind == TokenKind.EOF && throw(_err(s, "Unexpected TokenKind.EOF in address expression"))
            if tt.kind == TokenKind.LBRACE
                depth += 1
            elseif tt.kind == TokenKind.RBRACE
                depth -= 1
            end
            push!(parts, tt.leading_whitespace, tt.text)
            _advance!(s)
        end
        raw = join(parts)
        stripped = lstrip(raw)
        if startswith(stripped, "+")
            stripped = lstrip(SubString(stripped, 2))
        end
        offset = isempty(stripped) ? nothing : String(stripped)
    end

    _expect!(s, TokenKind.RBRACKET)
    AddressOperand(base, offset)
end

function _parse_reg_decl!(s::ParserState, indent::String)
    start_line = _peek(s).line
    prev_pos = s.pos
    _advance!(s)   # .reg
    _skip_newlines_and_comments!(s)

    scalar_type, vector_shape = _parse_declaration_type!(s, StateSpace.REG)
    _skip_newlines_and_comments!(s)

    t = _peek(s)
    (t.kind == TokenKind.REGISTER || t.kind == TokenKind.IDENTIFIER) ||
        throw(_err(s, "Expected register name, got $(t.kind) ($(repr(t.text)))"))
    name = _advance!(s).text

    count::Union{Int, Nothing} = nothing
    if _peek_kind(s) == TokenKind.LESS
        _advance!(s)
        count = Base.parse(Int,_expect!(s, TokenKind.INTEGER).text)
        _expect!(s, TokenKind.GREATER)
    end

    # Comma-separated extras: raw_line preserves them; the structural
    # representation only carries the first decl.
    while _peek_kind(s) == TokenKind.COMMA
        _advance!(s)
        _skip_newlines_and_comments!(s)
        nt = _peek(s)
        (nt.kind == TokenKind.REGISTER || nt.kind == TokenKind.IDENTIFIER) ||
            throw(_err(s, "Expected register name, got $(nt.kind)"))
        _advance!(s)
        if _peek_kind(s) == TokenKind.LESS
            _advance!(s)
            _expect!(s, TokenKind.INTEGER)
            _expect!(s, TokenKind.GREATER)
        end
    end

    semi = _expect!(s, TokenKind.SEMICOLON)
    end_line = semi.line
    _take_same_line_trailing_comment!(s, end_line)

    raw_line = (isempty(s.source_lines) || !_owns_line(s, prev_pos, start_line, end_line)) ?
                nothing : _extract_source(s, start_line, end_line)
    preceding = _take_comments!(s)

    RegDecl(
        type = scalar_type,
        name = name,
        count = count,
        vector_shape = vector_shape,
        formatting = FormattingInfo(indent = indent,
                                     preceding_comments = preceding,
                                     raw_line = raw_line),
    )
end

function _parse_var_decl!(s::ParserState, indent::String)
    start_line = _peek(s).line
    prev_pos = s.pos
    ss_tok = _advance!(s)
    state_space = state_space_from_ptx(ss_tok.text)
    _skip_newlines_and_comments!(s)

    alignment::Union{Int, Nothing} = nothing
    if _peek_kind(s) == TokenKind.DIRECTIVE && _peek(s).text == ".align"
        _advance!(s)
        _skip_newlines_and_comments!(s)
        alignment = Base.parse(Int,_expect!(s, TokenKind.INTEGER).text)
        _skip_newlines_and_comments!(s)
    end

    scalar_type, vector_shape = _parse_declaration_type!(s, state_space)
    _skip_newlines_and_comments!(s)

    name_tok = _expect!(s, TokenKind.IDENTIFIER)
    name = name_tok.text

    array_dims = Union{Int, String}[]
    while _peek_kind(s) == TokenKind.LBRACKET
        _advance!(s)
        if _peek_kind(s) == TokenKind.RBRACKET
            push!(array_dims, "")
        else
            push!(array_dims, Base.parse(Int,_expect!(s, TokenKind.INTEGER).text))
        end
        _expect!(s, TokenKind.RBRACKET)
    end

    array_size::Union{Int, Nothing} = nothing
    if length(array_dims) == 1
        d = array_dims[1]
        if d isa Int
            array_size = d
        else
            name = name * "[]"
        end
    elseif length(array_dims) > 1
        name = name * join((d isa Int ? "[$d]" : "[]" for d in array_dims))
    end

    initializer::Union{Tuple{Vararg{String}}, Nothing} = nothing
    if _peek_kind(s) == TokenKind.EQUALS
        _advance!(s)
        _skip_newlines_and_comments!(s)
        parts = String[]
        if _peek_kind(s) == TokenKind.LBRACE
            _advance!(s)
            while _peek_kind(s) != TokenKind.RBRACE
                k = _peek_kind(s)
                if k in (TokenKind.NEWLINE, TokenKind.COMMENT) || k == TokenKind.COMMA
                    _advance!(s)
                else
                    push!(parts, _advance!(s).text)
                end
            end
            _expect!(s, TokenKind.RBRACE)
        else
            push!(parts, _advance!(s).text)
        end
        initializer = Tuple(parts)
    end

    semi = _expect!(s, TokenKind.SEMICOLON)
    end_line = semi.line
    _take_same_line_trailing_comment!(s, end_line)
    raw_line = (isempty(s.source_lines) || !_owns_line(s, prev_pos, start_line, end_line)) ?
                nothing : _extract_source(s, start_line, end_line)
    preceding = _take_comments!(s)

    VarDecl(
        state_space = state_space,
        type = scalar_type,
        name = name,
        array_size = array_size,
        alignment = alignment,
        initializer = initializer,
        vector_shape = vector_shape,
        formatting = FormattingInfo(indent = indent,
                                     preceding_comments = preceding,
                                     raw_line = raw_line),
    )
end

function _parse_scalar_type!(s::ParserState)
    type_tok = _expect!(s, TokenKind.DIRECTIVE)
    try
        scalar_type_from_ptx(type_tok.text)
    catch err
        err isa ArgumentError || rethrow()
        throw(ParseError(string(err.msg), type_tok.line, type_tok.col))
    end
end

function _parse_declaration_type!(s::ParserState, state_space::StateSpace.T)
    vector_shape::Union{VectorShape.T, Nothing} = nothing
    if _peek_kind(s) == TokenKind.DIRECTIVE &&
       _peek(s).text in (".v2", ".v4")
        shape_tok = _advance!(s)
        vector_shape = vector_shape_from_ptx(shape_tok.text)
        _skip_newlines_and_comments!(s)
    end
    scalar_type = _parse_scalar_type!(s)
    if vector_shape !== nothing
        try
            validate_vector_declaration(vector_shape, scalar_type, state_space)
        catch err
            err isa ArgumentError || rethrow()
            type_tok = s.tokens[s.pos - 1]
            throw(ParseError(string(err.msg), type_tok.line, type_tok.col))
        end
    end
    scalar_type, vector_shape
end

function _parse_pragma!(s::ParserState, indent::String)
    _advance!(s)   # .pragma
    _skip_newlines_and_comments!(s)
    str_tok = _expect!(s, TokenKind.STRING)
    value = String(SubString(str_tok.text, 2, lastindex(str_tok.text) - 1))
    semi = _expect!(s, TokenKind.SEMICOLON)
    _take_same_line_trailing_comment!(s, semi.line)
    PragmaDirective(value = value, formatting = FormattingInfo(indent = indent))
end

# --- functions -------------------------------------------------------------

const _FUNCTION_DIRECTIVE_NAMES = Set{String}((
    ".maxnreg", ".maxntid", ".reqntid",
    ".minnctapersm", ".maxnctapersm",
    ".minperctamemory",
    ".noreturn", ".pragma",
    ".explicitcluster", ".reqnctapercluster", ".cluster_dim",
))

function _parse_function_or_global_with_linking!(s::ParserState)
    linking_tok = _advance!(s)
    linking = linking_from_ptx(linking_tok.text)
    _skip_newlines_and_comments!(s)

    t = _peek(s)
    if t.kind == TokenKind.DIRECTIVE && (t.text == ".entry" || t.text == ".func")
        return _parse_function!(s, linking)
    end
    if t.kind == TokenKind.DIRECTIVE && t.text in _VAR_DECL_DIRECTIVES
        vd = _parse_var_decl!(s, "")
        return VarDecl(
            state_space = vd.state_space,
            type = vd.type,
            name = vd.name,
            array_size = vd.array_size,
            alignment = vd.alignment,
            initializer = vd.initializer,
            linking = linking,
            vector_shape = vd.vector_shape,
            formatting = vd.formatting,
        )
    end
    throw(_err(s, "Expected .entry/.func or state-space directive after $(linking_tok.text)"))
end

function _parse_function!(s::ParserState, linking::Union{LinkingDirective.T, Nothing})
    preceding = _take_comments!(s)

    func_tok = _advance!(s)
    is_entry = func_tok.text == ".entry"
    _skip_newlines_and_comments!(s)

    # `.func (.param .b32 ret) name(...)` form.
    return_params::Union{Tuple{Vararg{Param}}, Nothing} = nothing
    if !is_entry && _peek_kind(s) == TokenKind.LPAREN
        return_params = _parse_param_list!(s)
        _skip_newlines_and_comments!(s)
    end

    name_tok = _expect!(s, TokenKind.IDENTIFIER)
    name = name_tok.text
    _skip_newlines_and_comments!(s)

    params = if _peek_kind(s) == TokenKind.LPAREN
        _parse_param_list!(s)
    else
        ()
    end
    _skip_newlines_and_comments!(s)

    func_directives = FunctionDirective[]
    while _peek_kind(s) == TokenKind.DIRECTIVE && _peek(s).text in _FUNCTION_DIRECTIVE_NAMES
        push!(func_directives, _parse_function_directive!(s))
        _skip_newlines_and_comments!(s)
    end

    body = if _peek_kind(s) == TokenKind.LBRACE
        _parse_function_body!(s)
    else
        ()
    end

    Function(
        is_entry = is_entry,
        name = name,
        params = params,
        return_params = return_params,
        body = body,
        linking = linking,
        directives = Tuple(func_directives),
        formatting = isempty(preceding) ? nothing :
                     FormattingInfo(preceding_comments = preceding),
    )
end

function _parse_param_list!(s::ParserState)
    _expect!(s, TokenKind.LPAREN)
    _skip_newlines_and_comments!(s)
    params = Param[]
    if _peek_kind(s) != TokenKind.RPAREN
        push!(params, _parse_param!(s))
        _skip_newlines_and_comments!(s)
        while _match!(s, TokenKind.COMMA) !== nothing
            _skip_newlines_and_comments!(s)
            push!(params, _parse_param!(s))
            _skip_newlines_and_comments!(s)
        end
    end
    _expect!(s, TokenKind.RPAREN)
    Tuple(params)
end

function _parse_param!(s::ParserState)
    t = _peek(s)
    (t.kind == TokenKind.DIRECTIVE && (t.text == ".param" || t.text == ".reg")) ||
        throw(_err(s, "Expected '.param' or '.reg', got $(repr(t.text))"))
    state_space = state_space_from_ptx(_advance!(s).text)
    _skip_newlines_and_comments!(s)

    alignment::Union{Int, Nothing} = nothing
    if _peek_kind(s) == TokenKind.DIRECTIVE && _peek(s).text == ".align"
        _advance!(s)
        _skip_newlines_and_comments!(s)
        alignment = Base.parse(Int, _expect!(s, TokenKind.INTEGER).text)
        _skip_newlines_and_comments!(s)
    end

    scalar_type = _parse_scalar_type!(s)
    _skip_newlines_and_comments!(s)

    ptr_state_space::Union{StateSpace.T, Nothing} = nothing
    ptr_alignment::Union{Int, Nothing} = nothing
    if _peek_kind(s) == TokenKind.DIRECTIVE && _peek(s).text == ".ptr"
        _advance!(s)
        _skip_newlines_and_comments!(s)
        ss_tok = _expect!(s, TokenKind.DIRECTIVE)
        ptr_state_space = state_space_from_ptx(ss_tok.text)
        _skip_newlines_and_comments!(s)
        if _peek_kind(s) == TokenKind.DIRECTIVE && _peek(s).text == ".align"
            _advance!(s)
            _skip_newlines_and_comments!(s)
            ptr_alignment = Base.parse(Int, _expect!(s, TokenKind.INTEGER).text)
            _skip_newlines_and_comments!(s)
        end
    end

    name_tok = _expect!(s, TokenKind.IDENTIFIER)
    name = name_tok.text

    array_size::Union{Int, Nothing} = nothing
    if _peek_kind(s) == TokenKind.LBRACKET
        _advance!(s)
        array_size = Base.parse(Int, _expect!(s, TokenKind.INTEGER).text)
        _expect!(s, TokenKind.RBRACKET)
    end
    Param(
        state_space = state_space,
        type = scalar_type,
        name = name,
        array_size = array_size,
        alignment = alignment,
        ptr_state_space = ptr_state_space,
        ptr_alignment = ptr_alignment,
    )
end

function _parse_function_directive!(s::ParserState)
    tok = _advance!(s)
    name = lstrip(tok.text, '.')
    _skip_newlines_and_comments!(s)
    values = Union{Int, String}[]
    if _peek_kind(s) == TokenKind.INTEGER
        push!(values, Base.parse(Int, _advance!(s).text))
        while _match!(s, TokenKind.COMMA) !== nothing
            _skip_newlines_and_comments!(s)
            push!(values, Base.parse(Int, _expect!(s, TokenKind.INTEGER).text))
        end
    end
    FunctionDirective(String(name), Tuple(values))
end

function _parse_function_body!(s::ParserState)
    _expect!(s, TokenKind.LBRACE)
    _parse_body_until!(s, TokenKind.RBRACE)
end

function _parse_body_until!(s::ParserState, end_kind::TokenKind.T)
    statements = Statement[]
    prev_nl = false

    while _peek_kind(s) != end_kind && _peek_kind(s) != TokenKind.EOF
        t = _peek(s)

        if t.kind == TokenKind.NEWLINE
            prev_nl && push!(statements, BlankLine())
            prev_nl = true
            _advance!(s)
            continue
        end

        if t.kind == TokenKind.COMMENT
            push!(statements, Comment(t.leading_whitespace * t.text))
            _advance!(s)
            prev_nl = false
            continue
        end

        if t.kind == TokenKind.PREPROCESSOR
            push!(statements, RawLine(t.leading_whitespace * t.text))
            _advance!(s)
            prev_nl = false
            continue
        end

        prev_nl = false

        if t.kind == TokenKind.LBRACE
            block_indent = t.leading_whitespace
            _advance!(s)
            inner = _parse_body_until!(s, TokenKind.RBRACE)
            push!(statements, Block(
                body = inner,
                formatting = FormattingInfo(indent = block_indent),
            ))
            continue
        end

        # Fault-tolerant: capture as RawLine on parse failure.
        saved_pos = s.pos
        try
            stmt = _parse_statement!(s, t.leading_whitespace)
            push!(statements, stmt)
            if !isempty(s.pending_stmts)
                append!(statements, s.pending_stmts)
                empty!(s.pending_stmts)
            end
        catch e
            e isa ParseError || rethrow()
            _is_module_directive(t) && rethrow()
            s.pos = saved_pos
            push!(statements, _capture_raw_line!(s, t.leading_whitespace))
        end
    end

    _expect!(s, end_kind)
    Tuple(statements)
end

@public parse, ParseError
