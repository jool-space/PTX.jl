# A deliberately small, non-eval PTX integer-constant evaluator shared by
# every transpiler path that must materialize a PTX constant in Julia. Copying
# raw PTX text into Julia would misread widths and octal notation and would
# turn untrusted identifiers into Julia code.
#
# This covers every integer-literal/operator shape the current PTX parser can
# represent structurally: decimal/hex/octal literals, WARP_SZ, unary operators,
# signed/unsigned casts, arithmetic, shifts, comparisons, bitwise AND/OR, and
# logical AND/OR. The lexer accepts binary literals, U suffixes, XOR, and
# ternary punctuation, but this deliberately smaller evaluator still rejects
# shapes it does not implement rather than copying them into Julia code.

struct _PTXIntValue
    bits::UInt64
    unsigned::Bool
end

mutable struct _PTXIntParser
    text::String
    pos::Int
end

_ptx_const_error(p::_PTXIntParser, message) = ArgumentError(
    "invalid PTX integer constant $(repr(p.text)): $message at byte $(p.pos)")

function _ptx_skipspace!(p::_PTXIntParser)
    while p.pos <= ncodeunits(p.text) && isspace(Char(codeunit(p.text, p.pos)))
        p.pos += 1
    end
end

function _ptx_startswith(p::_PTXIntParser, token::AbstractString)
    _ptx_skipspace!(p)
    stop = p.pos + ncodeunits(token) - 1
    stop <= ncodeunits(p.text) || return false
    SubString(p.text, p.pos, stop) == token
end

function _ptx_consume!(p::_PTXIntParser, token::AbstractString)
    _ptx_startswith(p, token) || return false
    p.pos += ncodeunits(token)
    true
end

function _ptx_consume_single!(p::_PTXIntParser, token::AbstractString,
                              longer::Tuple{Vararg{String}})
    _ptx_startswith(p, token) || return false
    any(candidate -> _ptx_startswith(p, candidate), longer) && return false
    p.pos += ncodeunits(token)
    true
end

_ptx_signed(v::_PTXIntValue) = reinterpret(Int64, v.bits)
_ptx_truth(v::_PTXIntValue) = v.bits != 0

function _ptx_from_int128(value::Int128, unsigned::Bool)
    modulus = Int128(1) << 64
    _PTXIntValue(UInt64(mod(value, modulus)), unsigned)
end

function _ptx_parse_literal!(p::_PTXIntParser)
    _ptx_skipspace!(p)
    start = p.pos
    start <= ncodeunits(p.text) || throw(_ptx_const_error(p, "expected literal"))
    isdigit(Char(codeunit(p.text, start))) ||
        throw(_ptx_const_error(p, "expected integer literal"))

    base = 10
    if start + 1 <= ncodeunits(p.text) && codeunit(p.text, start) == UInt8('0') &&
            codeunit(p.text, start + 1) in (UInt8('x'), UInt8('X'))
        base = 16
        p.pos += 2
        digits = p.pos
        while p.pos <= ncodeunits(p.text) &&
                isxdigit(Char(codeunit(p.text, p.pos)))
            p.pos += 1
        end
        p.pos > digits || throw(_ptx_const_error(p, "empty hexadecimal literal"))
    else
        while p.pos <= ncodeunits(p.text) &&
                isdigit(Char(codeunit(p.text, p.pos)))
            p.pos += 1
        end
        p.pos - start > 1 && codeunit(p.text, start) == UInt8('0') &&
            (base = 8)
    end

    literal = SubString(p.text, start, p.pos - 1)
    digits = base == 16 ? SubString(literal, 3) : literal
    value = tryparse(BigInt, digits; base)
    value === nothing && throw(_ptx_const_error(
        p, base == 8 ? "malformed octal literal" : "malformed literal"))
    value <= typemax(UInt64) || throw(_ptx_const_error(p, "literal exceeds 64 bits"))
    _PTXIntValue(UInt64(value), value > typemax(Int64))
end

function _ptx_consume_cast!(p::_PTXIntParser)
    save = p.pos
    _ptx_consume!(p, "(") || return nothing
    if _ptx_consume!(p, ".s64")
        _ptx_consume!(p, ")") || throw(_ptx_const_error(p, "unterminated .s64 cast"))
        return false
    elseif _ptx_consume!(p, ".u64")
        _ptx_consume!(p, ")") || throw(_ptx_const_error(p, "unterminated .u64 cast"))
        return true
    end
    p.pos = save
    nothing
end

function _ptx_parse_primary!(p::_PTXIntParser)
    if _ptx_consume!(p, "(")
        value = _ptx_parse_logical_or!(p)
        _ptx_consume!(p, ")") || throw(_ptx_const_error(p, "missing `)`"))
        return value
    end
    _ptx_parse_literal!(p)
end

function _ptx_parse_unary!(p::_PTXIntParser)
    cast = _ptx_consume_cast!(p)
    cast === nothing || begin
        value = _ptx_parse_unary!(p)
        return _PTXIntValue(value.bits, cast)
    end
    _ptx_consume!(p, "+") && return _ptx_parse_unary!(p)
    if _ptx_consume!(p, "-")
        value = _ptx_parse_unary!(p)
        return _PTXIntValue(-value.bits, value.unsigned)
    elseif _ptx_consume!(p, "!")
        return _PTXIntValue(_ptx_truth(_ptx_parse_unary!(p)) ? 0 : 1, false)
    elseif _ptx_consume!(p, "~")
        return _PTXIntValue(~_ptx_parse_unary!(p).bits, true)
    end
    _ptx_parse_primary!(p)
end

function _ptx_arithmetic(op::Symbol, a::_PTXIntValue, b::_PTXIntValue)
    unsigned = a.unsigned || b.unsigned
    if op === :+
        return _PTXIntValue(a.bits + b.bits, unsigned)
    elseif op === :-
        return _PTXIntValue(a.bits - b.bits, unsigned)
    elseif op === :*
        return _PTXIntValue(a.bits * b.bits, unsigned)
    elseif op === :/
        b.bits == 0 && throw(DivideError())
        if unsigned
            return _PTXIntValue(div(a.bits, b.bits), true)
        end
        value = div(Int128(_ptx_signed(a)), Int128(_ptx_signed(b)))
        return _ptx_from_int128(value, false)
    elseif op === :%
        b.bits == 0 && throw(DivideError())
        # PTX defines remainder over unsigned interpretations and a signed
        # result (PTX 9.3 Table 5).
        return _PTXIntValue(rem(a.bits, b.bits), false)
    end
    error("unknown PTX arithmetic operator $op")
end

function _ptx_parse_mul!(p::_PTXIntParser)
    value = _ptx_parse_unary!(p)
    while true
        op = _ptx_consume!(p, "*") ? :* :
             _ptx_consume!(p, "/") ? :/ :
             _ptx_consume!(p, "%") ? :% : nothing
        op === nothing && return value
        value = try
            _ptx_arithmetic(op, value, _ptx_parse_unary!(p))
        catch err
            err isa DivideError || rethrow()
            throw(_ptx_const_error(p, "division or remainder by zero"))
        end
    end
end

function _ptx_parse_add!(p::_PTXIntParser)
    value = _ptx_parse_mul!(p)
    while true
        op = _ptx_consume!(p, "+") ? :+ :
             _ptx_consume!(p, "-") ? :- : nothing
        op === nothing && return value
        value = _ptx_arithmetic(op, value, _ptx_parse_mul!(p))
    end
end

function _ptx_parse_shift!(p::_PTXIntParser)
    value = _ptx_parse_add!(p)
    while true
        left = _ptx_consume!(p, "<<")
        right = left ? false : _ptx_consume!(p, ">>")
        (left || right) || return value
        amount = _ptx_parse_add!(p).bits
        bits = if amount >= 64
            left ? UInt64(0) :
            (value.unsigned || _ptx_signed(value) >= 0 ? UInt64(0) : typemax(UInt64))
        elseif left
            value.bits << Int(amount)
        elseif value.unsigned
            value.bits >> Int(amount)
        else
            reinterpret(UInt64, _ptx_signed(value) >> Int(amount))
        end
        value = _PTXIntValue(bits, value.unsigned)
    end
end

function _ptx_compare(op::Symbol, a::_PTXIntValue, b::_PTXIntValue)
    result = if op === :(==)
        a.bits == b.bits
    elseif op === :(!=)
        a.bits != b.bits
    elseif a.unsigned || b.unsigned
        op === :<  ? a.bits < b.bits :
        op === :<= ? a.bits <= b.bits :
        op === :>  ? a.bits > b.bits : a.bits >= b.bits
    else
        av, bv = _ptx_signed(a), _ptx_signed(b)
        op === :<  ? av < bv : op === :<= ? av <= bv :
        op === :>  ? av > bv : av >= bv
    end
    _PTXIntValue(result ? 1 : 0, false)
end

function _ptx_parse_relational!(p::_PTXIntParser)
    value = _ptx_parse_shift!(p)
    while true
        op = _ptx_consume!(p, "<=") ? :<= :
             _ptx_consume!(p, ">=") ? :>= :
             _ptx_consume_single!(p, "<", ("<<", "<=")) ? :< :
             _ptx_consume_single!(p, ">", (">>", ">=")) ? :> : nothing
        op === nothing && return value
        value = _ptx_compare(op, value, _ptx_parse_shift!(p))
    end
end

function _ptx_parse_equality!(p::_PTXIntParser)
    value = _ptx_parse_relational!(p)
    while true
        op = _ptx_consume!(p, "==") ? :(==) :
             _ptx_consume!(p, "!=") ? :(!=) : nothing
        op === nothing && return value
        value = _ptx_compare(op, value, _ptx_parse_relational!(p))
    end
end

function _ptx_parse_bitand!(p::_PTXIntParser)
    value = _ptx_parse_equality!(p)
    while _ptx_consume_single!(p, "&", ("&&",))
        other = _ptx_parse_equality!(p)
        # PTX 9.3 §4.5.5 says bitwise operands use the usual conversions,
        # while §4.5.6 Table 5 instead prints u64 operands/results. ptxas 12.8
        # resolves that editorial contradiction in the prose's favor: a
        # global initialized with `((-1 & -1) >> 63)` contains all-one bytes,
        # proving a signed bitwise result and arithmetic right shift.
        value = _PTXIntValue(value.bits & other.bits,
                             value.unsigned || other.unsigned)
    end
    value
end

function _ptx_parse_bitor!(p::_PTXIntParser)
    value = _ptx_parse_bitand!(p)
    while _ptx_consume_single!(p, "|", ("||",))
        other = _ptx_parse_bitand!(p)
        value = _PTXIntValue(value.bits | other.bits,
                             value.unsigned || other.unsigned)
    end
    value
end

function _ptx_parse_logical_and!(p::_PTXIntParser)
    value = _ptx_parse_bitor!(p)
    while _ptx_consume!(p, "&&")
        # Deliberately evaluate the RHS even when `value` is false. ptxas
        # diagnoses division by zero in `0 && (1 / 0)`, so PTX constant
        # expressions do not use C's runtime short-circuit suppression.
        value = _PTXIntValue(_ptx_truth(value) &
                             _ptx_truth(_ptx_parse_bitor!(p)) ? 1 : 0, false)
    end
    value
end

function _ptx_parse_logical_or!(p::_PTXIntParser)
    value = _ptx_parse_logical_and!(p)
    while _ptx_consume!(p, "||")
        # As above, `1 || (1 / 0)` is rejected by ptxas rather than hiding the
        # invalid RHS.
        value = _PTXIntValue(_ptx_truth(value) |
                             _ptx_truth(_ptx_parse_logical_and!(p)) ? 1 : 0,
                             false)
    end
    value
end

function _ptx_integer_constant(text::AbstractString)
    normalized = _replace_predefined_immediate_tokens(text)
    p = _PTXIntParser(normalized, 1)
    value = _ptx_parse_logical_or!(p)
    _ptx_skipspace!(p)
    p.pos > ncodeunits(p.text) || throw(_ptx_const_error(p, "unexpected token"))
    value.unsigned ? value.bits : _ptx_signed(value)
end

function _ptx_integer_carrier_expr(text::AbstractString, ::Type{T}) where
        {T <: Integer}
    T in (UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32, Int64) ||
        error("unsupported PTX integer carrier $T")
    value = _ptx_integer_constant(text)
    bits = value isa UInt64 ? value : reinterpret(UInt64, value)
    U = unsigned(T)
    narrowed = bits % U
    digits = 2 * sizeof(T)
    literal = "0x" * string(narrowed; base = 16, pad = digits)
    if T <: Unsigned
        return "$T($literal)"
    end
    # The modular narrowing already happened above; spelling the resulting
    # signed value directly keeps generated Julia readable and is always a
    # checked-conversion-safe constructor argument.
    "$T($(reinterpret(T, narrowed)))"
end

_ptx_predicate_constant(text::AbstractString) =
    _ptx_integer_constant(text) != 0

# --- Structured-IR adapters -----------------------------------------------
# The evaluator and carrier helpers above have no instruction-family or IR
# assumptions. The remaining helpers reconstruct parser operands and enforce
# lop3's instruction-specific 8-bit LUT domain.

function _ptx_integer_constant_text(op::Operand)
    op isa ImmediateOperand && return op.text
    if op isa LabelOperand && haskey(IR.PREDEFINED_IMMEDIATES, op.name)
        return op.name
    elseif op isa NegatedOperand
        return "!" * _ptx_integer_constant_text(op.operand)
    elseif op isa ParenthesizedOperand && length(op.elements) == 1
        return "(" * _ptx_integer_constant_text(only(op.elements)) * ")"
    end
    throw(ArgumentError(
        "PTX transpiler: operand must be a PTX integer constant"))
end

function _lop3_lut_value(op::Operand)
    text = _ptx_integer_constant_text(op)
    value = _ptx_integer_constant(text)
    0 <= value <= 255 || throw(ArgumentError(
        "PTX transpiler: lop3 immLut must evaluate to 0:255, got " *
        "$(repr(text)) => $value"))
    Int(value)
end
