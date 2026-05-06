# `format(::Module)` returns `raw_source` verbatim when set; otherwise each
# statement consults `formatting.raw_line` first and falls back to building
# the line from fields. Function/Block bodies always walk children — emitting
# raw text for a container would silently discard modifications to its body.

@public format

format_operand(op::RegisterOperand)  = op.name
format_operand(op::ImmediateOperand) = op.text
format_operand(op::LabelOperand)     = op.name
format_operand(op::NegatedOperand)   = "!" * format_operand(op.operand)
format_operand(op::VectorOperand) =
    "{" * join(format_operand.(op.elements), ", ") * "}"
format_operand(op::ParenthesizedOperand) =
    "(" * join(format_operand.(op.elements), ", ") * ")"
format_operand(op::PipeOperand) =
    format_operand(op.left) * "|" * format_operand(op.right)
function format_operand(op::AddressOperand)
    op.offset === nothing && return "[" * op.base * "]"
    "[" * op.base * "+" * op.offset * "]"
end

_raw_line(::Nothing) = nothing
_raw_line(fi::FormattingInfo) = fi.raw_line
_indent(::Nothing) = ""
_indent(fi::FormattingInfo) = fi.indent
_trailing(::Nothing) = ""
_trailing(fi::FormattingInfo) = fi.trailing

format_modifiers(mods::Tuple{Vararg{String}}) = isempty(mods) ? "" : join(mods)

format_predicate(::Nothing) = ""
format_predicate(p::Predicate) = "@" * (p.negated ? "!" : "") * p.register * " "

function format(inst::Instruction)
    raw = _raw_line(inst.formatting)
    raw !== nothing && return raw
    indent = _indent(inst.formatting)
    head = format_predicate(inst.predicate) *
           inst.opcode *
           format_modifiers(inst.modifiers)
    body = isempty(inst.operands) ? head * ";" :
           head * " " * join(format_operand.(inst.operands), ", ") * ";"
    trailing = _trailing(inst.formatting)
    indent * body * trailing
end

function format(lbl::Label)
    raw = _raw_line(lbl.formatting)
    raw !== nothing && return raw
    _indent(lbl.formatting) * lbl.name * ":"
end

function format(rd::RegDecl)
    raw = _raw_line(rd.formatting)
    raw !== nothing && return raw
    suffix = rd.count === nothing ? "" : "<$(rd.count)>"
    _indent(rd.formatting) * ".reg " * ptx(rd.type) * " " * rd.name * suffix * ";"
end

function format(vd::VarDecl)
    raw = _raw_line(vd.formatting)
    raw !== nothing && return raw
    parts = String[]
    vd.linking === nothing || push!(parts, ptx(vd.linking))
    push!(parts, ptx(vd.state_space))
    vd.alignment === nothing || push!(parts, ".align $(vd.alignment)")
    push!(parts, ptx(vd.type))
    name = vd.array_size === nothing ? vd.name : "$(vd.name)[$(vd.array_size)]"
    push!(parts, name)
    line = _indent(vd.formatting) * join(parts, " ")
    if vd.initializer !== nothing
        line *= " = " * (length(vd.initializer) == 1 ? vd.initializer[1] :
                         "{" * join(vd.initializer, ", ") * "}")
    end
    line * ";"
end

function format(p::PragmaDirective)
    _indent(p.formatting) * ".pragma \"" * p.value * "\";"
end

format(c::Comment)   = c.text
format(::BlankLine)  = ""
format(r::RawLine)   = r.text

function format(b::Block)
    indent = _indent(b.formatting)
    body = isempty(b.body) ? "" : "\n" * join(String[format(s) for s in b.body], "\n") * "\n" * indent
    indent * "{" * body * "}"
end

function format(p::Param)
    parts = String[]
    push!(parts, ptx(p.state_space))
    p.alignment === nothing || push!(parts, ".align $(p.alignment)")
    push!(parts, ptx(p.type))
    if p.ptr_state_space !== nothing
        push!(parts, ".ptr")
        push!(parts, ptx(p.ptr_state_space))
        p.ptr_alignment === nothing || push!(parts, ".align $(p.ptr_alignment)")
    end
    name = p.array_size === nothing ? p.name : "$(p.name)[$(p.array_size)]"
    push!(parts, name)
    join(parts, " ")
end

function format(d::FunctionDirective)
    isempty(d.values) && return "." * d.name
    "." * d.name * " " * join(d.values, ", ")
end

function format(f::Function)
    parts = String[]
    f.linking === nothing || push!(parts, ptx(f.linking))
    push!(parts, f.is_entry ? ".entry" : ".func")
    if f.return_params !== nothing
        push!(parts, "(" * join(String[format(p) for p in f.return_params], ", ") * ")")
    end
    sig_args = isempty(f.params) ? "()" :
               "(" * join(String[format(p) for p in f.params], ", ") * ")"
    push!(parts, f.name * sig_args)
    for d in f.directives
        push!(parts, format(d))
    end
    head = join(parts, " ")
    if f.formatting !== nothing && !isempty(f.formatting.preceding_comments)
        head = join(f.formatting.preceding_comments, "\n") * "\n" * head
    end
    body = isempty(f.body) ? "" : "\n" * join(String[format(s) for s in f.body], "\n") * "\n"
    head * "\n{" * body * "}"
end

"""
    format(mod::IR.Module) -> String

Reconstruct PTX text from a parsed `IR.Module`. Returns `mod.raw_source`
verbatim when set (the lossless fast path used by parser-produced IR);
otherwise emits the module structurally — header, leading prelude, then
each directive — falling back per statement to `formatting.raw_line`
when present and to field-driven reconstruction when not.

Per-statement `format(stmt)` methods (one per `IR.Statement` kind)
implement the structural fallback and can be called individually.
"""
function format(m::Module)
    m.raw_source !== nothing && return m.raw_source

    parts = String[]
    for stmt in m.leading
        push!(parts, format(stmt))
    end
    if m.raw_header !== nothing
        push!(parts, m.raw_header)
    else
        push!(parts, ".version $(m.version.major).$(m.version.minor)")
        push!(parts, ".target " * join(m.target.targets, ", "))
        push!(parts, ".address_size $(m.address_size.size)")
    end
    for stmt in m.directives
        push!(parts, format(stmt))
    end
    # Trailing newline so format()→parse()→format() is a fixed point regardless
    # of whether the source had trailing whitespace.
    join(parts, "\n") * "\n"
end
