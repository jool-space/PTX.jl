emit_stmt!(cg::CodeGenState, s::Instruction) = emit_instruction!(cg, s)

emit_stmt!(cg::CodeGenState, s::Label) =
    emit!(cg, "@label " * julia_label(s.name))

# Julia introduces vars on assignment — track the decl (used by the
# predicated-assignment hoist) and emit nothing.
function emit_stmt!(cg::CodeGenState, s::RegDecl)
    cg.reg_decls[s.name] = s
    nothing
end

function emit_stmt!(cg::CodeGenState, s::VarDecl)
    suffix = s.array_size === nothing ? "" : "[$(s.array_size)]"
    emit!(cg, "# " * ptx(s.state_space) * " " * ptx(s.type) * " " *
              s.name * suffix * ";")
end

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
