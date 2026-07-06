module IR

using Republic: @public

# Ported from pyptx/pyptx/ir/nodes.py (https://github.com/patrick-toulme/pyptx).
# Copyright 2026 Patrick Toulmé. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Translated to Julia and adapted.

include("types.jl")

# Whitespace metadata for round-trip fidelity. `raw_line`, when set, is emitted
# verbatim instead of structurally — the lossless escape valve.
Base.@kwdef struct FormattingInfo
    indent::String = ""
    trailing::String = ""
    blank_lines_before::Int = 0
    preceding_comments::Tuple{Vararg{String}} = ()
    raw_line::Union{String, Nothing} = nothing
end

abstract type Operand end

struct RegisterOperand <: Operand
    name::String
end

# Numeric literal stored as raw text for lossless round-trip
# (`'42'`, `'0xFF'`, `'0d3FF0000000000000'`, `'1.0'`, `'-1'`).
struct ImmediateOperand <: Operand
    text::String
end

struct LabelOperand <: Operand
    name::String
end

# `{%r0, %r1, %r2, %r3}` — operand vector.
struct VectorOperand <: Operand
    elements::Tuple{Vararg{Operand}}
end

# `[base+offset]` or `[base, {c1, c2, ...}]` — `base` is a register or
# symbol name; `offset` is the raw text of an offset register/literal, or
# `nothing`; `coords` is the parsed tensor-coordinate vector of the TMA
# bracket form (cp.async.bulk.tensor, tensormap ops), or `nothing`. Coords
# are structured operands so canonical renaming descends into them.
struct AddressOperand <: Operand
    base::String
    offset::Union{String, Nothing}
    coords::Union{Tuple{Vararg{Operand}}, Nothing}
end
AddressOperand(base) = AddressOperand(base, nothing, nothing)
AddressOperand(base, offset) = AddressOperand(base, offset, nothing)

# `(op1, op2, ...)` — used in call instruction return / argument lists.
struct ParenthesizedOperand <: Operand
    elements::Tuple{Vararg{Operand}}
end

# `!operand` — logical negation (setp, logical ops).
struct NegatedOperand <: Operand
    operand::Operand
end

# `%p0|%p1` — dual predicate output in setp.
struct PipeOperand <: Operand
    left::Operand
    right::Operand
end

# `@%p0` or `@!%p0` — instruction predication.
struct Predicate
    register::String
    negated::Bool
end
Predicate(register::AbstractString) = Predicate(String(register), false)

struct Version
    major::Int
    minor::Int
end

struct Target
    targets::Tuple{Vararg{String}}
end

struct AddressSize
    size::Int
end

abstract type Statement end

# `opcode`: base opcode (`"mov"`, `"ld"`, `"wgmma"`, `"bra"`)
# `modifiers`: dot-prefixed strings in declaration order (`(".global", ".nc", ".b32")`)
# `operands`: destination(s) first, then source(s)
Base.@kwdef struct Instruction <: Statement
    opcode::String
    modifiers::Tuple{Vararg{String}} = ()
    operands::Tuple{Vararg{Operand}} = ()
    predicate::Union{Predicate, Nothing} = nothing
    formatting::Union{FormattingInfo, Nothing} = nothing
end
Instruction(opcode::AbstractString, modifiers::Tuple, operands::Tuple) =
    Instruction(; opcode = String(opcode),
                  modifiers = Tuple(String(m) for m in modifiers),
                  operands = operands)
Instruction(opcode::AbstractString, modifiers::Tuple, operands::Tuple,
            predicate::Predicate) =
    Instruction(; opcode = String(opcode),
                  modifiers = Tuple(String(m) for m in modifiers),
                  operands = operands, predicate = predicate)

Base.@kwdef struct Label <: Statement
    name::String
    formatting::Union{FormattingInfo, Nothing} = nothing
end
Label(name::AbstractString) = Label(; name = String(name))

# `.reg .type name<count>;`. `count = nothing` for a single-register decl.
Base.@kwdef struct RegDecl <: Statement
    type::ScalarType.T
    name::String
    count::Union{Int, Nothing} = nothing
    formatting::Union{FormattingInfo, Nothing} = nothing
end

# Variable declaration; covers both function-body and module-level
# (`.shared`, `.global`, `.local`, `.const`).
Base.@kwdef struct VarDecl <: Statement
    state_space::StateSpace.T
    type::ScalarType.T
    name::String
    array_size::Union{Int, Nothing} = nothing
    alignment::Union{Int, Nothing} = nothing
    initializer::Union{Tuple{Vararg{String}}, Nothing} = nothing
    linking::Union{LinkingDirective.T, Nothing} = nothing
    formatting::Union{FormattingInfo, Nothing} = nothing
end

Base.@kwdef struct PragmaDirective <: Statement
    value::String
    formatting::Union{FormattingInfo, Nothing} = nothing
end

struct Comment <: Statement
    text::String
end

struct BlankLine <: Statement end

# A line the parser couldn't structurally parse — emitted verbatim.
struct RawLine <: Statement
    text::String
end

# Nested `{ }` scope (PTX register-lifetime scoping).
Base.@kwdef struct Block <: Statement
    body::Tuple{Vararg{Statement}}
    formatting::Union{FormattingInfo, Nothing} = nothing
end

# Construction-time only — parser never produces this.
Base.@kwdef struct IntrinsicScope <: Statement
    name::String
    args_repr::String
    body::Tuple{Vararg{Statement}}
    formatting::Union{FormattingInfo, Nothing} = nothing
end

# `ptr_state_space` only set when this Param is a `.func .ptr` arg.
Base.@kwdef struct Param
    state_space::StateSpace.T
    type::ScalarType.T
    name::String
    array_size::Union{Int, Nothing} = nothing
    alignment::Union{Int, Nothing} = nothing
    ptr_state_space::Union{StateSpace.T, Nothing} = nothing
    ptr_alignment::Union{Int, Nothing} = nothing
end

# Performance-hint directive (`.maxnreg`, `.maxntid`, ...).
struct FunctionDirective
    name::String
    values::Tuple{Vararg{Union{Int, String}}}
end

# `.func` or `.entry` definition.
Base.@kwdef struct Function <: Statement
    is_entry::Bool
    name::String
    params::Tuple{Vararg{Param}} = ()
    return_params::Union{Tuple{Vararg{Param}}, Nothing} = nothing
    body::Tuple{Vararg{Statement}} = ()
    linking::Union{LinkingDirective.T, Nothing} = nothing
    directives::Tuple{Vararg{FunctionDirective}} = ()
    formatting::Union{FormattingInfo, Nothing} = nothing
end

# `leading` holds content before `.version` (license blocks, file-level
# doc comments). Kept separate from `directives` so structural emission
# can place the header in its canonical position when `raw_header` /
# `raw_source` are absent.
Base.@kwdef struct Module
    version::Version
    target::Target
    address_size::AddressSize
    leading::Tuple{Vararg{Statement}} = ()
    directives::Tuple{Vararg{Statement}} = ()
    raw_header::Union{String, Nothing} = nothing
    raw_source::Union{String, Nothing} = nothing
end

include("format.jl")
include("normalize.jl")
include("canonical.jl")

end # module IR
