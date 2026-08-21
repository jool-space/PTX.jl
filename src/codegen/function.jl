# Encode a Param's type spec in dot-separated format — lossless capture of the
# original `.param` decl.
#   .param .u64 param0                       → "u64"
#   .param .u64 .ptr .global param0          → "u64.ptr.global"
#   .param .u64 .ptr .global .align 8 param0 → "u64.ptr.global.palign8"
#   .param .align 64 .b8 param4[128]         → "b8.align64.array128"
function param_type_string(p::Param)
    s = string(IR.SCALAR_TYPE_NAMES[p.type])
    if p.ptr_state_space !== nothing
        s *= ".ptr." * string(IR.STATE_SPACE_NAMES[p.ptr_state_space])
        s = replace(s, "::" => "__")
    end
    p.alignment      === nothing || (s *= ".align"  * string(p.alignment))
    if p.ptr_alignment !== nothing && p.ptr_state_space !== nothing
        s *= ".palign" * string(p.ptr_alignment)
    end
    p.array_size     === nothing || (s *= ".array"  * string(p.array_size))
    s
end

function emit_metadata!(cg::CodeGenState, func::Function,
                        arch::AbstractString, version::Tuple{Int, Int})
    emit!(cg, "# @ptx_kernel arch=$(arch) version=$(version[1]).$(version[2])")
    if !isempty(func.params)
        ps = ["(\"$(param_type_string(p))\", \"$(p.name)\")" for p in func.params]
        emit!(cg, "#   raw_params  = [" * join(ps, ", ") * "]")
    end
    if !isempty(func.directives)
        ds = String[]
        for d in func.directives
            vals = isempty(d.values) ? "()" :
                   length(d.values) == 1 ? "(" * string(d.values[1]) * ",)" :
                                            "(" * join(d.values, ", ") * ")"
            push!(ds, "(\"" * d.name * "\", " * vals * ")")
        end
        emit!(cg, "#   directives  = [" * join(ds, ", ") * "]")
    end
    if func.linking !== nothing
        emit!(cg, "#   linking     = \"" * lstrip(IR.ptx(func.linking), '.') * "\"")
    end
end

# The hoist pass holds Julia variable names (render_dst output); recover the
# `.reg` declaration by re-prefixing `%`, falling back to the bare spelling
# for declarations that omitted it.
function _register_decl_for_julia_var(cg::CodeGenState, var::String)
    decl = _declared_register(cg, "%" * var)
    decl === nothing || return decl
    _declared_register(cg, var)
end

function emit_function!(cg::CodeGenState, func::Function,
                        arch::AbstractString = "",
                        version::Tuple{Int, Int} = (0, 0))
    empty!(cg.reg_decls)
    empty!(cg.declared)
    empty!(cg.predicated_assigns)
    empty!(cg.inferred_b128_regs)
    cg.param_names = [julia_var(p.name) for p in func.params]

    isempty(arch) || emit_metadata!(cg, func, arch, version)

    name = julia_var(func.name)
    args = join(cg.param_names, ", ")
    emit!(cg, "function " * name * "(" * args * ")")
    indent!(cg)

    # Emit body to a side buffer first, then prepend `local` hoists for any
    # register assigned under predication (Julia's soft scope makes vars
    # assigned inside `if`-blocks local to the block; readers outside would
    # see undefined-var errors).
    body_buf = IOBuffer()
    body_cg = CodeGenState()
    body_cg.indent = cg.indent
    body_cg.io = body_buf
    body_cg.param_names = cg.param_names

    for s in func.body
        emit_stmt!(body_cg, s)
    end

    # PTX registers read before any write yield an undefined value; Julia
    # variables throw. Give every register whose assignment can be bypassed —
    # under predication, or jumped over by a forward branch — a typed zero
    # default, so out-of-bounds threads compute-and-discard garbage exactly
    # like the hardware instead of trapping on UndefVarError. Zero is a
    # legitimate instance of "undefined", and a program whose result depends
    # on it was already unspecified in PTX.
    hoist = sort(collect(union(setdiff(body_cg.predicated_assigns, body_cg.declared),
                               body_cg.skippable_assigns)))
    for var in hoist
        decl = _register_decl_for_julia_var(body_cg, var)
        if decl === nothing
            emit!(cg, "local " * var)
        else
            emit!(cg, "local " * var * " = zero(" *
                      string(scalar_to_julia(decl.type)) * ")")
        end
    end

    write(cg.io, take!(body_buf))

    dedent!(cg)
    emit!(cg, "end")
end
