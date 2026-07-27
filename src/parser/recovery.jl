# Ported from pyptx/parser/parser.py (https://github.com/patrick-toulme/pyptx).
# Copyright 2026 Patrick Toulmé. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Translated to Julia and adapted.

function _extract_source(s::ParserState, start_line::Int, end_line::Int)
    isempty(s.source_lines) && return ""
    lo = clamp(start_line, 1, length(s.source_lines))
    hi = clamp(end_line,   1, length(s.source_lines))
    join(view(s.source_lines, lo:hi), "\n")
end

# Capturing `raw_line` would duplicate text when a statement shares its source
# line with neighbours — only capture when no token before/after also lives on
# that line.
function _owns_line(s::ParserState, prev_pos::Int, start_line::Int, end_line::Int)
    if prev_pos > 1
        prev = s.tokens[prev_pos - 1]
        prev.line >= start_line && return false
    end
    p = s.pos
    while p <= length(s.tokens)
        t = s.tokens[p]
        t.kind == TokenKind.EOF && return true
        t.kind == TokenKind.NEWLINE && return true
        t.line > end_line && return true
        t.kind == TokenKind.COMMENT || return false
        p += 1
    end
    true
end

function _capture_raw_line!(s::ParserState, leading::String)
    start_line = _peek(s).line
    end_line = start_line

    # Walk every token through the trailing newline — including a comment past
    # the semicolon — so it isn't re-emitted as a standalone Comment.
    while !_at_end(s) && _peek_kind(s) ∉ (TokenKind.NEWLINE, TokenKind.EOF)
        t = _advance!(s)
        end_line = t.line
    end

    text = isempty(s.source_lines) ?
           leading :
           _extract_source(s, start_line, end_line)
    RawLine(text)
end

function _find_raw_header(s::ParserState, address_size_explicit::Bool)
    isempty(s.source_lines) && return nothing
    header_start = 1
    for (i, line) in enumerate(s.source_lines)
        startswith(strip(line), ".version") || continue
        header_start = i
        break
    end
    header_end = header_start
    # The first matching target is the required header target; later targets
    # remain ordered statements in `directives`.
    terminal = address_size_explicit ? ".address_size" : ".target"
    for i in header_start:length(s.source_lines)
        startswith(strip(s.source_lines[i]), terminal) || continue
        header_end = i
        break
    end
    join(view(s.source_lines, header_start:header_end), "\n")
end

_parse_error_points_to_module_directive(s::ParserState, e::ParseError) =
    any(t -> t.line == e.line && t.col == e.col && _is_module_directive(t),
        s.tokens)
