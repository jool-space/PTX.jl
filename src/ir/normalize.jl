# Mirrors pyptx/pyptx/ir/normalize.py's `normalize_module`
# (https://github.com/patrick-toulme/pyptx).
# Copyright 2026 Patrick Toulmé. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Translated to Julia and adapted.

# Disable the module-level raw_source short-circuit so `format(m)` walks the IR
# structurally. Per-statement raw_line is kept.
unraw(m::Module) = Module(
    version      = m.version,
    target       = m.target,
    address_size = m.address_size,
    address_size_explicit = m.address_size_explicit,
    leading      = m.leading,
    directives   = m.directives,
    raw_header   = m.raw_header,
    raw_source   = nothing,
)

# Canonical form for structural comparison: drops cosmetic content
# (raw_source/raw_header, leading prelude, Comment/BlankLine, FormattingInfo).
# A RawLine is an opaque parser fallback, not formatting, so it remains in
# every directive/body position. Likewise, Block and IntrinsicScope preserve
# lexical scope: flattening braces can erase declaration lifetime and name
# reuse.
normalize(m::Module) = Module(
    version      = m.version,
    target       = m.target,
    address_size = m.address_size,
    address_size_explicit = m.address_size_explicit,
    leading      = (),
    directives   = _normalize_directives(m.directives),
    raw_header   = nothing,
    raw_source   = nothing,
)

function _normalize_directives(directives::Tuple{Vararg{Statement}})
    _normalize_statements(directives)
end

_normalize_function(f::Function) = Function(
    is_entry      = f.is_entry,
    name          = f.name,
    params        = f.params,
    return_params = f.return_params,
    body          = _normalize_body(f.body),
    linking       = f.linking,
    directives    = f.directives,
    formatting    = nothing,
)

_normalize_body(body::Tuple{Vararg{Statement}}) = _normalize_statements(body)

# Module directives and function bodies share the same statement model. Keep
# their normalization in one exhaustive dispatcher so a newly modeled
# statement cannot silently retain FormattingInfo in only one position.
function _normalize_statements(statements::Tuple{Vararg{Statement}})
    out = Statement[]
    for s in statements
        if s isa RawLine
            # An opaque parser fallback is structure, not whitespace. It must
            # survive in both module and function positions.
            push!(out, s)
        elseif s isa Comment || s isa BlankLine
            continue
        elseif s isa Function
            push!(out, _normalize_function(s))
        elseif s isa Instruction
            push!(out, _normalize_instruction(s))
        elseif s isa RegDecl
            # Legacy module-scoped registers are still semantic PTX. They are
            # uncommon under the ABI, but formatting must not make their
            # comparison noisy.
            push!(out, RegDecl(type = s.type, name = s.name, count = s.count))
        elseif s isa VarDecl
            push!(out, _normalize_var_decl(s))
        elseif s isa Label
            push!(out, Label(name = s.name))
        elseif s isa Block
            push!(out, Block(body = _normalize_body(s.body), formatting = nothing))
        elseif s isa IntrinsicScope
            push!(out, IntrinsicScope(name = s.name, args_repr = s.args_repr,
                                      body = _normalize_body(s.body),
                                      formatting = nothing))
        elseif s isa PragmaDirective
            push!(out, PragmaDirective(value = s.value))
        elseif s isa TargetDirective
            push!(out, TargetDirective(target = s.target, formatting = nothing))
        else
            throw(ArgumentError("normalize does not handle statement type $(typeof(s)); add an explicit structural normalization case"))
        end
    end
    Tuple(out)
end

_normalize_instruction(i::Instruction) = Instruction(
    opcode     = i.opcode,
    modifiers  = i.modifiers,
    operands   = i.operands,
    predicate  = i.predicate,
    formatting = nothing,
)

_normalize_var_decl(v::VarDecl) = VarDecl(
    state_space = v.state_space,
    type        = v.type,
    name        = v.name,
    array_size  = v.array_size,
    alignment   = v.alignment,
    initializer = v.initializer,
    linking     = v.linking,
    formatting  = nothing,
)

# Compares every semantic module node after normalizing cosmetic information.
# `entry_only` excludes every non-entry `.func` Function directive in full—its
# declaration, signature, linkage, function directives, and body—but keeps
# other module directives: globals, opaque fallbacks, and pragmas can affect an
# entry kernel.
# Returns human-readable difference lines; empty ⇔ identical normalized IR
# structure (with opaque RawLine text compared verbatim). This is not a
# substitute for PTX ISA or ptxas semantic validation.
function diff(a::Module, b::Module; entry_only::Bool = false)
    a = normalize(a)
    b = normalize(b)
    diffs = String[]

    a.version       == b.version       || push!(diffs, "version: $(a.version) vs $(b.version)")
    a.target        == b.target        || push!(diffs, "target: $(a.target) vs $(b.target)")
    a.address_size  == b.address_size  || push!(diffs, "address_size: $(a.address_size) vs $(b.address_size)")
    a.address_size_explicit == b.address_size_explicit ||
        push!(diffs, "address_size_explicit: $(a.address_size_explicit) vs $(b.address_size_explicit)")

    a_directives = _comparison_directives(a.directives, entry_only)
    b_directives = _comparison_directives(b.directives, entry_only)
    _diff_statements!(diffs, a_directives, b_directives, "directive")

    diffs
end

function _comparison_directives(directives::Tuple{Vararg{Statement}},
                                entry_only::Bool)
    entry_only || return directives
    Tuple(d for d in directives if !(d isa Function && !d.is_entry))
end

function _diff_statements!(diffs::Vector{String},
                           a_stmts::Tuple{Vararg{Statement}},
                           b_stmts::Tuple{Vararg{Statement}},
                           prefix::String)
    if length(a_stmts) != length(b_stmts)
        push!(diffs, "$prefix length: $(length(a_stmts)) vs $(length(b_stmts))")
        for (j, s) in enumerate(a_stmts[1:min(end, 5)])
            push!(diffs, "  a[$j]: $(_stmt_summary(s))")
        end
        for (j, s) in enumerate(b_stmts[1:min(end, 5)])
            push!(diffs, "  b[$j]: $(_stmt_summary(s))")
        end
        return nothing
    end

    for (i, (sa, sb)) in enumerate(zip(a_stmts, b_stmts))
        _diff_statement!(diffs, sa, sb, "$prefix[$i]")
    end
    nothing
end

function _diff_statement!(diffs::Vector{String}, sa::Statement, sb::Statement,
                          prefix::String)
    if typeof(sa) !== typeof(sb)
        push!(diffs, "$prefix type: $(nameof(typeof(sa))) vs $(nameof(typeof(sb)))")
        return nothing
    end

    if sa isa Function
        _diff_function!(diffs, sa, sb::Function, prefix)
    elseif sa isa Block
        _diff_statements!(diffs, sa.body, (sb::Block).body, prefix * ".body")
    elseif sa isa IntrinsicScope
        other = sb::IntrinsicScope
        sa.name == other.name ||
            push!(diffs, "$prefix name: $(repr(sa.name)) vs $(repr(other.name))")
        sa.args_repr == other.args_repr ||
            push!(diffs, "$prefix args: $(repr(sa.args_repr)) vs $(repr(other.args_repr))")
        _diff_statements!(diffs, sa.body, other.body, prefix * ".body")
    elseif sa isa Instruction
        sa.opcode    == sb.opcode    || push!(diffs, "$prefix opcode: $(repr(sa.opcode)) vs $(repr(sb.opcode))")
        sa.modifiers == sb.modifiers || push!(diffs, "$prefix mods: $(sa.modifiers) vs $(sb.modifiers)")
        sa.operands  == sb.operands  || begin
            push!(diffs, "$prefix operands differ")
            push!(diffs, "  a: $(sa.operands)")
            push!(diffs, "  b: $(sb.operands)")
        end
        sa.predicate == sb.predicate || push!(diffs, "$prefix pred: $(sa.predicate) vs $(sb.predicate)")
    elseif sa isa RawLine
        sa.text == (sb::RawLine).text ||
            push!(diffs, "$prefix raw: $(repr(sa.text)) vs $(repr((sb::RawLine).text))")
    elseif !_eq_ignoring_formatting(sa, sb)
        push!(diffs, "$prefix: $(_stmt_summary(sa)) vs $(_stmt_summary(sb))")
    end
    nothing
end

function _diff_function!(diffs::Vector{String}, af::Function, bf::Function,
                         prefix::String)
    af.name == bf.name || push!(diffs, "$prefix name: $(repr(af.name)) vs $(repr(bf.name))")
    af.is_entry == bf.is_entry ||
        push!(diffs, "$prefix is_entry: $(af.is_entry) vs $(bf.is_entry)")
    af.params == bf.params || push!(diffs, "$prefix params differ")
    af.return_params == bf.return_params || push!(diffs, "$prefix return params differ")
    af.linking == bf.linking || push!(diffs, "$prefix linking: $(af.linking) vs $(bf.linking)")
    af.directives == bf.directives || push!(diffs, "$prefix directives differ")
    _diff_statements!(diffs, af.body, bf.body, prefix * ".body")
    nothing
end

_stmt_summary(s::Instruction) = s.opcode * join(s.modifiers)
_stmt_summary(s::RegDecl)     = ".reg $(s.type) $(s.name)$(s.count === nothing ? "" : "<$(s.count)>")"
_stmt_summary(s::VarDecl)     = "VarDecl($(s.name))"
_stmt_summary(s::Label)       = "$(s.name):"
_stmt_summary(s::RawLine)     = "RawLine($(repr(s.text)))"
_stmt_summary(s::Block)       = "Block($(length(s.body)) statements)"
_stmt_summary(s::IntrinsicScope) = "IntrinsicScope($(repr(s.name)))"
_stmt_summary(s)              = repr(s)

function _eq_ignoring_formatting(a::T, b::T) where {T <: Statement}
    for f in fieldnames(T)
        f === :formatting && continue
        getfield(a, f) == getfield(b, f) || return false
    end
    true
end
_eq_ignoring_formatting(a, b) = a == b
