# Returns (dst_expr_string, defined_var_names::Vector{String}). The var-name
# list lets the function pass hoist `local` decls when any was assigned under
# predication.
function render_dst(op::RegisterOperand, cg::CodeGenState)
    name = julia_var(op.name)
    (name, [name])
end

function render_dst(op::LabelOperand, cg::CodeGenState)
    _declared_register(cg, op) === nothing &&
        return (render_operand(op, cg), String[])
    render_dst(RegisterOperand(op.name), cg)
end

function render_dst(op::PipeOperand, cg::CodeGenState)
    l, lnames = render_dst(op.left, cg)
    r, rnames = render_dst(op.right, cg)
    ("(" * l * ", " * r * ")", [lnames; rnames])
end

function render_dst(op::VectorOperand, cg::CodeGenState)
    parts = String[]
    names = String[]
    for e in op.elements
        s = render_operand(e, cg)
        push!(parts, s)
        e isa RegisterOperand && push!(names, s)
    end
    ("(" * join(parts, ", ") * ")", names)
end

render_dst(op::Operand, cg::CodeGenState) = (render_operand(op, cg), String[])

render_pred_cond(p::Predicate) = (p.negated ? "!" : "") * julia_var(p.register)

function chain_expr(::CodeGenState, opcode::AbstractString,
                    @nospecialize(modifiers::Tuple{Vararg{String}}))
    "ptx\"" * opcode * join(modifiers) * "\""
end

_schema_modifiers(modifiers::Tuple{Vararg{String}}) =
    Tuple(Symbol(lstrip(modifier, '.')) for modifier in modifiers)

_is_sink_operand(op::LabelOperand) = op.name == "_"
_is_sink_operand(op::RegisterOperand) = op.name == "_"
_is_sink_operand(::Operand) = false

function _declared_register(cg::CodeGenState, name::AbstractString)
    decl = get(cg.reg_decls, String(name), nothing)
    decl === nothing || return decl
    for candidate in values(cg.reg_decls)
        candidate.count === nothing && continue
        startswith(name, candidate.name) || continue
        suffix = SubString(name, nextind(name, lastindex(candidate.name)))
        index = tryparse(Int, suffix)
        index === nothing && continue
        0 <= index < candidate.count && return candidate
    end
    nothing
end

_declared_register(cg::CodeGenState, op::RegisterOperand) =
    _declared_register(cg, op.name)

# PTX permits `.reg` declarations whose names omit `%`; the lexer necessarily
# represents those instruction operands as LabelOperand. They are registers
# only when the active declaration table proves it.
_declared_register(cg::CodeGenState, op::LabelOperand) =
    _declared_register(cg, op.name)

const _REGISTER_TYPES_8 =
    (ScalarType.B8, ScalarType.U8, ScalarType.S8)
const _REGISTER_TYPES_32 =
    (ScalarType.B32, ScalarType.U32, ScalarType.S32)
const _REGISTER_TYPES_64 =
    (ScalarType.B64, ScalarType.U64, ScalarType.S64)

_schema_operand_hint(kind::Symbol) = kind === :bf16 ? :b16 : kind

function _render_schema_source(op::Operand, cg::CodeGenState, kind::Symbol)
    render_operand(op, cg; type_hint = _schema_operand_hint(kind))
end

function emit_instruction!(cg::CodeGenState, inst::Instruction)
    # Drop debug directives.
    (inst.opcode == ".loc" || inst.opcode == ".file") && return

    # CC.CF is implicit architectural state and is not preserved across calls.
    # Instruction-at-a-time Julia emission would split a straight-line carry
    # chain into independent inline-asm calls with no LLVM-visible dependency;
    # it can also let pointer-alias propagation erase an add.cc/sub.cc producer.
    # Reject until the transpiler can fuse the complete chain into one block.
    uses_implicit_cc(inst.opcode, inst.modifiers) && throw(ArgumentError(
        "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) accesses the " *
        "implicit CC.CF flag; instruction-at-a-time lowering cannot preserve " *
        "that dependency. Use PTX.add_with_carry, PTX.sub_with_borrow, or " *
        "PTX.mul_wide in Julia source."))

    # Validate constant-only, destinationless instructions before generic
    # destination inference or pointer-alias absorption can reinterpret their
    # sole source as a definition. Constant expressions are reduced to Val so
    # reconstructed Julia retains the compile-time ISA contract.
    immediate_checked = _instruction_immediate_form_contract(inst)
    if immediate_checked !== nothing
        chain = chain_expr(cg, inst.opcode, inst.modifiers)
        call = chain * "(Val($(immediate_checked.value)))"
        emit_with_predicate!(cg, call, inst.predicate, String[])
        return
    end

    mbarrier_schema = _instruction_mbarrier_schema(cg, inst)
    mbarrier_schema === nothing ||
        return _emit_mbarrier!(cg, inst, mbarrier_schema)

    b128_checked = _instruction_b128_schema(cg, inst)
    b128_checked === nothing || return _emit_b128!(cg, inst, b128_checked)

    structured_checked = _instruction_structured_result_schema(cg, inst)

    # Close fixed-result grammar islands before any instruction can be erased
    # by pointer-alias absorption. The schema also provides a distinct type
    # hint for every source operand; one terminal hint is wrong for mixed
    # precision, mixed-sign dot products, widened arithmetic, and cvt.pack.
    vector_schema = _instruction_vector_result_schema(cg, inst)
    vector_schema === nothing ||
        return _emit_vector_result!(cg, inst, vector_schema)
    scalar_schema = structured_checked === nothing ?
                    _instruction_scalar_result_schema(inst) : nothing
    cvt_schema = _instruction_cvt_source_schema(cg, inst, scalar_schema)
    clc_schema = _instruction_clc_try_cancel_schema(inst)

    if inst.opcode == "ret" && isempty(inst.operands)
        if inst.predicate === nothing
            emit!(cg, "return nothing")
        else
            emit!(cg, "if " * render_pred_cond(inst.predicate) * "; return nothing; end")
        end
        return
    end

    if inst.opcode == "bra" && length(inst.operands) == 1 &&
            inst.operands[1] isa LabelOperand
        target = julia_label((inst.operands[1]::LabelOperand).name)
        if inst.predicate === nothing
            emit!(cg, "@goto " * target)
        else
            emit!(cg, "if " * render_pred_cond(inst.predicate) *
                      "; @goto " * target * "; end")
        end
        return
    end

    # Shared-pointer alias propagation: if this `mov`/`add`/`sub` defines a
    # register from a translated shared symbol or an existing alias, record
    # the alias and emit nothing. Use sites of the dst register are
    # substituted via `pointer_aliases` in `render_operand`.
    structured_checked === nothing && scalar_schema === nothing &&
        inst.predicate === nothing &&
        _try_alias_def!(cg, inst) && return

    if clc_schema !== nothing
        chain = chain_expr(cg, inst.opcode, inst.modifiers)
        args = join((render_operand(op, cg) for op in inst.operands), ", ")
        emit_with_predicate!(cg, chain * "(" * args * ")",
                             inst.predicate, String[])
        return
    end

    # Exact-schema islands above retain their specialized emission. Every
    # remaining instruction is emitted from the same finite form/role ledger
    # used by the module preflight; do not reintroduce address/result guesses.
    if structured_checked === nothing && scalar_schema === nothing &&
            cvt_schema === nothing
        return _emit_transpile_form!(cg, inst)
    end

    # Structured-result schemas choose their reviewed API modifiers. Shfl's
    # existing intrinsic wrapper retains its trailing `.pred` selector.
    modifiers = structured_checked === nothing ? inst.modifiers :
                Tuple("." * string(mod) for mod in structured_checked.mods)
    if structured_checked === nothing && !isempty(inst.operands) &&
            inst.operands[1] isa PipeOperand
        if inst.opcode == "shfl"
            modifiers = (modifiers..., ".pred")
        end
    end
    chain = chain_expr(cg, inst.opcode, modifiers)

    dst_expr, dst_names = structured_checked === nothing ?
                          render_dst(inst.operands[1], cg) :
                          _render_structured_dst(inst.operands[1], cg)
    src_strs = if structured_checked !== nothing
        [_render_structured_source(op, cg, kind)
         for (op, kind) in zip(inst.operands[2:end],
                               structured_checked.schema.operands)]
    elseif cvt_schema !== nothing
        [_render_cvt_source(op, cg, kind, cvt_schema, index)
         for (index, (op, kind)) in
             enumerate(zip(inst.operands[2:end], cvt_schema.operands))]
    else
        [_render_schema_source(op, cg, kind)
         for (op, kind) in zip(inst.operands[2:end], scalar_schema.operands)]
    end
    args = join(src_strs, ", ")
    line = dst_expr * " = " * chain * "(" * args * ")"
    emit_with_predicate!(cg, line, inst.predicate, dst_names)
end

# Returns true if the instruction was absorbed as a pointer-alias definition
# (and therefore should not be emitted). Handles:
#   mov.<sz>  %dst, X           where X resolves through pointer_aliases / shared_vars
#   add.<sz>  %dst, X, Y        where exactly one of X, Y is a known pointer
#   sub.<sz>  %dst, X, Y        where X is a known pointer (Y is the offset)
# Wider patterns (mad.lo, shifts on a pointer-derived offset, ...) fall back
# to plain emission — the substitution at use-site stays correct, the line
# just isn't elided.
function _try_alias_def!(cg::CodeGenState, inst::Instruction)
    isempty(inst.operands) && return false
    inst.operands[1] isa RegisterOperand || return false
    dst = julia_var((inst.operands[1]::RegisterOperand).name)

    if inst.opcode == "mov" && length(inst.operands) == 2
        expr = _alias_expr(cg, inst.operands[2])
        expr === nothing && return false
        cg.pointer_aliases[dst] = expr
        return true
    end

    type_hint = operand_type_hint(inst.opcode, inst.modifiers)

    if inst.opcode == "add" && length(inst.operands) == 3
        a = _alias_expr(cg, inst.operands[2])
        b = _alias_expr(cg, inst.operands[3])
        # Exactly one operand must be the pointer; the other must resolve to
        # a plain (non-pointer) value that can be added as a byte offset.
        if a !== nothing && b === nothing
            off = render_operand(inst.operands[3], cg; type_hint)
            cg.pointer_aliases[dst] = "(" * a * ") + " * off
            return true
        elseif b !== nothing && a === nothing
            off = render_operand(inst.operands[2], cg; type_hint)
            cg.pointer_aliases[dst] = "(" * b * ") + " * off
            return true
        end
        return false
    end

    if inst.opcode == "sub" && length(inst.operands) == 3
        a = _alias_expr(cg, inst.operands[2])
        a === nothing && return false
        _alias_expr(cg, inst.operands[3]) === nothing || return false
        off = render_operand(inst.operands[3], cg; type_hint)
        cg.pointer_aliases[dst] = "(" * a * ") - " * off
        return true
    end

    false
end

# Returns the alias expression if this operand is (or already aliases) a
# translated shared symbol; nothing otherwise. Bare symbol references — `smem`
# in `mov.u64 %rd, smem` — are parsed as `LabelOperand` (the lexer can't
# distinguish a label target from a state-space symbol), so both forms map.
function _alias_expr(cg::CodeGenState, op::Operand)
    name = if op isa RegisterOperand
        op.name
    elseif op isa LabelOperand
        op.name
    else
        return nothing
    end
    jname = julia_var(name)
    haskey(cg.pointer_aliases, jname) && return cg.pointer_aliases[jname]
    nothing
end

# Tracks dst names assigned under predication so the function pass can emit
# a `local` hoist (Julia's soft scope makes vars assigned inside `if`-blocks
# local to the block).
function emit_with_predicate!(cg::CodeGenState, line::String,
                              pred::Union{Predicate, Nothing},
                              dst_names::Vector{String})
    if pred === nothing
        emit!(cg, line)
        union!(cg.declared, dst_names)
    else
        emit!(cg, "if " * render_pred_cond(pred) * "; " * line * "; end")
        union!(cg.predicated_assigns, dst_names)
    end
end
