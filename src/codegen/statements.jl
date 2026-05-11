emit_stmt!(cg::CodeGenState, s::Instruction) = emit_instruction!(cg, s)

emit_stmt!(cg::CodeGenState, s::Label) =
    emit!(cg, "@label " * julia_label(s.name))

# Julia introduces vars on assignment — track the decl (used by the
# predicated-assignment hoist) and emit nothing.
function emit_stmt!(cg::CodeGenState, s::RegDecl)
    cg.reg_decls[s.name] = s
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
function emit_stmt!(cg::CodeGenState, s::Block)
    saved_declared = copy(cg.declared)
    saved_pred     = copy(cg.predicated_assigns)
    emit!(cg, "let")
    indent!(cg)
    for sub in s.body
        emit_stmt!(cg, sub)
    end
    dedent!(cg)
    emit!(cg, "end")
    cg.declared = saved_declared
    cg.predicated_assigns = saved_pred
end
