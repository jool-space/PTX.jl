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
    leading      = m.leading,
    directives   = m.directives,
    raw_header   = m.raw_header,
    raw_source   = nothing,
)

# Canonical form for semantic comparison: drops cosmetic content
# (raw_source/raw_header, leading prelude, Comment/BlankLine, FormattingInfo).
# RawLine is kept inside function bodies (real, parser-modeled-as-text
# instructions) but stripped at the top level (typically parser fallback).
normalize(m::Module) = Module(
    version      = m.version,
    target       = m.target,
    address_size = m.address_size,
    leading      = (),
    directives   = _normalize_directives(m.directives),
    raw_header   = nothing,
    raw_source   = nothing,
)

function _normalize_directives(directives::Tuple{Vararg{Statement}})
    out = Statement[]
    for d in directives
        d isa Comment   && continue
        d isa BlankLine && continue
        d isa RawLine   && continue
        if d isa Function
            push!(out, _normalize_function(d))
        elseif d isa VarDecl
            push!(out, _normalize_var_decl(d))
        elseif d isa PragmaDirective
            push!(out, PragmaDirective(value = d.value))
        else
            push!(out, d)
        end
    end
    Tuple(out)
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

function _normalize_body(body::Tuple{Vararg{Statement}})
    out = Statement[]
    for s in body
        s isa Comment   && continue
        s isa BlankLine && continue
        if s isa RawLine
            push!(out, s)               # body RawLines carry real instructions
        elseif s isa Instruction
            push!(out, _normalize_instruction(s))
        elseif s isa RegDecl
            push!(out, RegDecl(type = s.type, name = s.name, count = s.count))
        elseif s isa VarDecl
            push!(out, _normalize_var_decl(s))
        elseif s isa Label
            push!(out, Label(name = s.name))
        elseif s isa Block
            # Flatten the {} scope — register lifetime is lost but the
            # instruction sequence survives for semantic comparison.
            append!(out, _normalize_body(s.body))
        elseif s isa IntrinsicScope
            append!(out, _normalize_body(s.body))
        elseif s isa PragmaDirective
            push!(out, PragmaDirective(value = s.value))
        else
            push!(out, s)
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

# Filters cosmetic content on the fly — works on either raw or normalized
# Modules. Returns a list of human-readable difference lines; empty ⇔
# semantically identical (after `normalize`).
function diff(a::Module, b::Module; entry_only::Bool = false)
    diffs = String[]

    a.version       == b.version       || push!(diffs, "version: $(a.version) vs $(b.version)")
    a.target        == b.target        || push!(diffs, "target: $(a.target) vs $(b.target)")
    a.address_size  == b.address_size  || push!(diffs, "address_size: $(a.address_size) vs $(b.address_size)")

    a_funcs = collect(d for d in a.directives if d isa Function)
    b_funcs = collect(d for d in b.directives if d isa Function)
    if entry_only
        a_funcs = filter(f -> f.is_entry, a_funcs)
        b_funcs = filter(f -> f.is_entry, b_funcs)
    end

    if length(a_funcs) != length(b_funcs)
        push!(diffs, "function count: $(length(a_funcs)) vs $(length(b_funcs))")
        return diffs
    end

    for (i, (af, bf)) in enumerate(zip(a_funcs, b_funcs))
        prefix = "func[$i] ($(af.name))"
        af.name     == bf.name     || push!(diffs, "$prefix name: $(repr(af.name)) vs $(repr(bf.name))")
        af.is_entry == bf.is_entry || push!(diffs, "$prefix is_entry: $(af.is_entry) vs $(bf.is_entry)")

        a_stmts = collect(s for s in af.body if !(s isa Comment || s isa BlankLine))
        b_stmts = collect(s for s in bf.body if !(s isa Comment || s isa BlankLine))

        if length(a_stmts) != length(b_stmts)
            push!(diffs, "$prefix body length: $(length(a_stmts)) vs $(length(b_stmts))")
            for (j, s) in enumerate(a_stmts[1:min(end,5)])
                push!(diffs, "  a[$j]: $(_stmt_summary(s))")
            end
            for (j, s) in enumerate(b_stmts[1:min(end,5)])
                push!(diffs, "  b[$j]: $(_stmt_summary(s))")
            end
            continue
        end

        for (j, (sa, sb)) in enumerate(zip(a_stmts, b_stmts))
            if typeof(sa) !== typeof(sb)
                push!(diffs, "$prefix body[$j] type: $(nameof(typeof(sa))) vs $(nameof(typeof(sb)))")
                continue
            end
            if sa isa Instruction
                sa.opcode    == sb.opcode    || push!(diffs, "$prefix body[$j] opcode: $(repr(sa.opcode)) vs $(repr(sb.opcode))")
                sa.modifiers == sb.modifiers || push!(diffs, "$prefix body[$j] mods: $(sa.modifiers) vs $(sb.modifiers)")
                sa.operands  == sb.operands  || begin
                    push!(diffs, "$prefix body[$j] operands differ")
                    push!(diffs, "  a: $(sa.operands)")
                    push!(diffs, "  b: $(sb.operands)")
                end
                sa.predicate == sb.predicate || push!(diffs, "$prefix body[$j] pred: $(sa.predicate) vs $(sb.predicate)")
            elseif !_eq_ignoring_formatting(sa, sb)
                push!(diffs, "$prefix body[$j]: $(_stmt_summary(sa)) vs $(_stmt_summary(sb))")
            end
        end
    end

    diffs
end

_stmt_summary(s::Instruction) = s.opcode * join(s.modifiers)
_stmt_summary(s::RegDecl)     = ".reg $(s.type) $(s.name)$(s.count === nothing ? "" : "<$(s.count)>")"
_stmt_summary(s::VarDecl)     = "VarDecl($(s.name))"
_stmt_summary(s::Label)       = "$(s.name):"
_stmt_summary(s)              = repr(s)

function _eq_ignoring_formatting(a::T, b::T) where {T <: Statement}
    for f in fieldnames(T)
        f === :formatting && continue
        getfield(a, f) == getfield(b, f) || return false
    end
    true
end
_eq_ignoring_formatting(a, b) = a == b
