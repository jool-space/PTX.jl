emit_stmt!(cg::CodeGenState, s::Instruction) = emit_instruction!(cg, s)

emit_stmt!(cg::CodeGenState, s::Label) =
    emit!(cg, "@label " * julia_label(s.name))

# RegDecl structurally carries only the first name in a comma-packed
# declaration. Recover the complete declaration once, when it enters the
# active register table, from lexer tokens rather than regexing raw source.
# COMMENT tokens keep commas/semicolons inside comments from manufacturing or
# truncating declarations. If a manually constructed FormattingInfo snapshot
# is inconsistent, retain only the trusted structural declaration.
function _reg_declarators(s::RegDecl)
    formatting = s.formatting
    formatting === nothing && return (s,)
    raw = formatting.raw_line
    raw === nothing && return (s,)

    tokens = try
        tokenize(raw)
    catch err
        err isa LexError || rethrow()
        return (s,)
    end
    i = 1
    function next_token()
        while i <= length(tokens) &&
                tokens[i].kind in (TokenKind.COMMENT, TokenKind.NEWLINE)
            i += 1
        end
        i <= length(tokens) ? tokens[i] : nothing
    end
    function consume(kind)
        token = next_token()
        token !== nothing && token.kind == kind || return nothing
        i += 1
        token
    end

    directive = consume(TokenKind.DIRECTIVE)
    directive !== nothing && directive.text == ".reg" || return (s,)
    dtype = consume(TokenKind.DIRECTIVE)
    dtype !== nothing && dtype.text == ptx(s.type) || return (s,)

    declarations = RegDecl[]
    while true
        token = next_token()
        token !== nothing &&
            token.kind in (TokenKind.REGISTER, TokenKind.IDENTIFIER) ||
            return (s,)
        i += 1
        name = token.text
        count = nothing
        token = next_token()
        if token !== nothing && token.kind == TokenKind.LESS
            i += 1
            count_token = consume(TokenKind.INTEGER)
            count_token === nothing && return (s,)
            count = tryparse(Int, count_token.text)
            count === nothing && return (s,)
            consume(TokenKind.GREATER) === nothing && return (s,)
        end
        push!(declarations,
              RegDecl(type = s.type, name = name, count = count))

        token = next_token()
        token === nothing && return (s,)
        if token.kind == TokenKind.COMMA
            i += 1
            continue
        elseif token.kind == TokenKind.SEMICOLON
            first_decl = first(declarations)
            first_decl.name == s.name && first_decl.count == s.count ||
                return (s,)
            return Tuple(declarations)
        end
        return (s,)
    end
end

# Julia introduces vars on assignment — track every declarator (used by the
# predicated-assignment hoist and reviewed operand-schema validation) and emit
# nothing.
function emit_stmt!(cg::CodeGenState, s::RegDecl)
    for declaration in _reg_declarators(s)
        cg.reg_decls[declaration.name] = declaration
    end
    nothing
end

# `.shared` / `.shared::cta` VarDecls translate to `CuStaticSharedArray`. The
# Julia binding shadows the PTX symbol, and `pointer_aliases[name]` is seeded
# so any later reference (`mov.u64 %rd, name` etc.) substitutes
# `pointer(name)` and the defining instruction is dropped — see
# `_try_alias_def!` in instruction.jl. Other state spaces (`.global`,
# `.const`, `.local`, `.shared::cluster`) still emit a placeholder comment.
function emit_stmt!(cg::CodeGenState, s::VarDecl)
    if s.state_space === StateSpace.SHARED || s.state_space === StateSpace.SHARED_CTA
        jname = julia_var(s.name)
        jt    = scalar_to_julia(s.type)
        n     = _shared_array_count(s)
        emit!(cg, jname * " = CuStaticSharedArray(" * string(jt) * ", " * string(n) * ")")
        push!(cg.shared_vars, jname)
        cg.pointer_aliases[jname] = "pointer(" * jname * ")"
        union!(cg.declared, [jname])
        return
    end
    suffix = s.array_size === nothing ? "" : "[$(s.array_size)]"
    emit!(cg, "# " * ptx(s.state_space) * " " * ptx(s.type) * " " *
              s.name * suffix * ";")
end

# `.shared .b8 buf[64]` → 64 elements of UInt8.
# `.shared .b32 buf[16]` → 16 elements of UInt32.
# Scalar (`.shared .b32 x`) → length-1 array; callers can `[1]`-index.
_shared_array_count(s::VarDecl) = s.array_size === nothing ? 1 : s.array_size

emit_stmt!(cg::CodeGenState, s::PragmaDirective) =
    emit!(cg, "# .pragma " * repr(s.value))

# Drop LLVM/NVPTX debug-info comments the transpiler can't usefully carry
# into Julia. Keep everything else: PTX source comments may carry algorithm notes.
function _drop_comment(text::AbstractString)
    s = strip(lstrip(strip(text), '/'))
    s == "begin inline asm" && return true
    s == "end inline asm"   && return true
    s == "demoted variable" && return true
    s == "-- End function"  && return true
    startswith(s, "%bb.")   && return true
    startswith(s, "%L") && all(c -> isdigit(c) || c == '_' || isletter(c),
                                SubString(s, 2)) && return true
    false
end

emit_stmt!(cg::CodeGenState, s::Comment) =
    _drop_comment(s.text) ? nothing : emit!(cg, "# " * lstrip(s.text))

emit_stmt!(::CodeGenState, ::BlankLine) = nothing  # collapse blanks

emit_stmt!(cg::CodeGenState, s::RawLine) =
    emit!(cg, "# raw: " * lstrip(s.text))

# PTX `{ ... }` register-lifetime scope → Julia `let ... end`. Save/restore
# declared-var set so registers defined inside the block don't leak out.
function _declares_julia_register(decl::RegDecl, name::AbstractString)
    base = julia_var(decl.name)
    decl.count === nothing && return base == name
    startswith(name, base) || return false
    first_suffix = nextind(name, lastindex(base))
    first_suffix <= lastindex(name) || return false
    index = tryparse(Int, SubString(name, first_suffix))
    index !== nothing && 0 <= index < decl.count
end

function emit_stmt!(cg::CodeGenState, s::Block)
    saved_declared = copy(cg.declared)
    saved_pred     = copy(cg.predicated_assigns)
    saved_reg_decls = copy(cg.reg_decls)
    saved_shared_vars = copy(cg.shared_vars)
    saved_aliases = copy(cg.pointer_aliases)
    saved_b128 = copy(cg.inferred_b128_regs)
    emit!(cg, "let")
    indent!(cg)
    for sub in s.body
        emit_stmt!(cg, sub)
    end
    dedent!(cg)
    emit!(cg, "end")
    outer_aliases = copy(saved_aliases)
    for (name, expression) in cg.pointer_aliases
        any(decl -> _declares_julia_register(decl, name),
            values(saved_reg_decls)) || continue
        outer_aliases[name] = expression
    end
    cg.declared = saved_declared
    cg.predicated_assigns = saved_pred
    cg.reg_decls = saved_reg_decls
    cg.shared_vars = saved_shared_vars
    cg.pointer_aliases = outer_aliases
    cg.inferred_b128_regs = saved_b128
end
