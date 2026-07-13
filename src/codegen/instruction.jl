# Returns (dst_expr_string, defined_var_names::Vector{String}). The var-name
# list lets the function pass hoist `local` decls when any was assigned under
# predication.
function render_dst(op::RegisterOperand, cg::CodeGenState)
    name = julia_var(op.name)
    (name, [name])
end

function render_dst(op::PipeOperand, cg::CodeGenState)
    l = render_operand(op.left, cg)
    r = render_operand(op.right, cg)
    ("(" * l * ", " * r * ")", [l, r])
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

# Opcodes that have no destination operand even when they take inputs (the
# first operand is a barrier id, group count, etc.). Memory-sink ops with
# `AddressOperand` first arg are caught separately.
const NO_DEST_OPCODES = Set{String}((
    "bar", "barrier",
    "ret", "exit", "trap", "brkpt",
))

# Same idea, but the no-dest variant lives behind a leading modifier — the
# rest of the family (e.g. `wgmma.mma_async`) keeps a destination.
const NO_DEST_OPCODE_MODIFIERS = Set{Tuple{String, String}}((
    ("wgmma", ".wait_group"),
))

_is_no_dest(inst::Instruction) =
    inst.opcode in NO_DEST_OPCODES ||
    (!isempty(inst.modifiers) &&
     (inst.opcode, inst.modifiers[1]) in NO_DEST_OPCODE_MODIFIERS)

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

    # PTX `ld.param.<type> %rd, [paramN]` → `rd = paramN`. Julia kernel args
    # ARE the values; rebinding suffices.
    if inst.opcode == "ld" && any(==(".param"), inst.modifiers) &&
            length(inst.operands) == 2 &&
            inst.operands[2] isa AddressOperand
        addr = inst.operands[2]::AddressOperand
        if !startswith(addr.base, "%") && addr.offset === nothing
            dst_expr, dst_names = render_dst(inst.operands[1], cg)
            line = dst_expr * " = " * julia_var(addr.base)
            emit_with_predicate!(cg, line, inst.predicate, dst_names)
            return
        end
    end

    # Shared-pointer alias propagation: if this `mov`/`add`/`sub` defines a
    # register from a translated shared symbol or an existing alias, record
    # the alias and emit nothing. Use sites of the dst register are
    # substituted via `pointer_aliases` in `render_operand`.
    inst.predicate === nothing && _try_alias_def!(cg, inst) && return

    # PipeOperand destination needs an extra modifier — `.dual` for setp,
    # `.pred` for shfl — so the wrapped multi-output method dispatches.
    modifiers = inst.modifiers
    if !isempty(inst.operands) && inst.operands[1] isa PipeOperand
        if inst.opcode == "setp"
            modifiers = (".dual", modifiers...)
        elseif inst.opcode == "shfl"
            modifiers = (modifiers..., ".pred")
        end
    end
    chain = chain_expr(cg, inst.opcode, modifiers)

    if isempty(inst.operands)
        emit_with_predicate!(cg, chain * "()", inst.predicate, String[])
        return
    end

    type_hint = operand_type_hint(inst.opcode, inst.modifiers)

    if _is_no_dest(inst)
        all_args = join((render_operand(op, cg; type_hint) for op in inst.operands), ", ")
        emit_with_predicate!(cg, chain * "(" * all_args * ")",
                             inst.predicate, String[])
        return
    end

    # Memory-sink ops (st.*, red.*, atom.* with address-as-dst). The address
    # is u64 — don't wrap with the trailing-modifier dtype hint — but value
    # operands that follow do get it.
    if inst.operands[1] isa AddressOperand
        first_arg = render_operand(inst.operands[1], cg)
        rest = (render_operand(op, cg; type_hint) for op in inst.operands[2:end])
        all_args = join(Iterators.flatten(((first_arg,), rest)), ", ")
        emit_with_predicate!(cg, chain * "(" * all_args * ")",
                             inst.predicate, String[])
        return
    end

    dst_expr, dst_names = render_dst(inst.operands[1], cg)
    src_strs = [render_operand(op, cg; type_hint) for op in inst.operands[2:end]]
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
