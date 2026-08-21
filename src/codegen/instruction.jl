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

# Transpiler adapter-selection dispatch family. Each method consults its
# ledger's `_instruction_*` adapter (which throws the ledger's miss for a
# claimed-but-invalid spelling) and, on a hit, performs the adapter's
# specialized emission. Returns whether the instruction was handled.
# Methods live next to their adapters in src/codegen/adapters/*.jl.
function transpile_ledger! end

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

    # The closed grammar islands, routed by `island_of` (protocol.jl). This
    # selection is behavior-shielded by the module preflight
    # (_validate_exact_schema! in contract.jl): emission only ever sees
    # instructions whose unique exact schema already validated.
    island = island_of(Symbol(inst.opcode), _schema_modifiers(inst.modifiers))
    island !== nothing && transpile_ledger!(island, cg, inst) && return

    # Keep the parser/transpiler on the same result-ABI boundary as direct
    # calls. This catches noncanonical cvt and any reviewed pure form that
    # would otherwise infer void before pointer-alias absorption can erase
    # a malformed mov/add/sub definition.
    infer_rettype(Symbol(inst.opcode), _schema_modifiers(inst.modifiers))

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
        # A forward branch (target label not yet emitted) can jump over any
        # assignment between here and the label; a backward branch cannot
        # make an already-linear assignment skippable.
        target in cg.emitted_labels || push!(cg.open_branch_targets, target)
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
    # substituted via `pointer_aliases` in `render_operand`. Audited scalar/
    # structured forms never reach this point — their ledger emitted above.
    inst.predicate === nothing && _try_alias_def!(cg, inst) && return

    # Ordinary cvt's source-carrier ledger sits outside the island partition
    # (see protocol.jl); cvt.pack was already handled by the scalar island
    # above.
    transpile_ledger!(CvtLedger(), cg, inst) && return

    # Exact-schema islands above retain their specialized emission. Every
    # remaining instruction is emitted from the same finite form/role ledger
    # used by the module preflight; do not reintroduce address/result guesses.
    _emit_transpile_form!(cg, inst)
end

# The generic destination/source emission shared by the scalar, structured,
# and cvt islands: `dst = ptx"op.mods"(srcs...)` under the predicate.
function _emit_schema_call!(cg::CodeGenState, inst::Instruction,
                            modifiers::Tuple{Vararg{String}},
                            dst_expr::String, dst_names::Vector{String},
                            src_strs::Vector{String})
    chain = chain_expr(cg, inst.opcode, modifiers)
    line = dst_expr * " = " * chain * "(" * join(src_strs, ", ") * ")"
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
    isempty(cg.open_branch_targets) || union!(cg.skippable_assigns, dst_names)
end
