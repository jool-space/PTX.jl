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

_schema_modifiers(modifiers::Tuple{Vararg{String}}) =
    Tuple(Symbol(lstrip(modifier, '.')) for modifier in modifiers)

function _instruction_scalar_result_schema(inst::Instruction)
    op = Symbol(inst.opcode)
    mods = _schema_modifiers(inst.modifiers)
    schema = scalar_result_schema(op, mods)
    schema === nothing && requires_scalar_result_schema(op, mods) &&
        throw(scalar_result_schema_miss(op, mods))
    if schema === nothing
        # Keep the parser/transpiler on the same result-ABI boundary as direct
        # calls. This catches noncanonical cvt and any reviewed pure form that
        # would otherwise infer void before pointer-alias absorption can erase
        # a malformed mov/add/sub definition.
        infer_rettype(op, mods)
        return nothing
    end
    expected = length(schema.operands) + 1 # explicit destination + sources
    length(inst.operands) == expected || throw(ArgumentError(
        "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) is an audited " *
        "scalar form with $(length(schema.operands)) source operands, got " *
        "$(max(length(inst.operands) - 1, 0)); see $(schema.section)"))
    schema
end

_is_sink_operand(op::LabelOperand) = op.name == "_"
_is_sink_operand(op::RegisterOperand) = op.name == "_"
_is_sink_operand(::Operand) = false

function _mbarrier_source_operand(kind::Symbol, op::Operand)
    # Coordinate lists are a distinct TMA address grammar. Ordinary mbarrier
    # addresses admit a base plus constant offset, but never `[base, {coords}]`.
    # `render_operand` intentionally renders only base/offset, so accepting a
    # coordinate-bearing node here would silently discard source operands.
    kind === :address && return op isa AddressOperand && op.coords === nothing
    kind === :u64 && return op isa RegisterOperand || op isa ImmediateOperand ||
                            op isa LabelOperand
    kind === :u32 && return op isa RegisterOperand || op isa ImmediateOperand ||
                            op isa LabelOperand
    false
end

function _mbarrier_sink_selector_mods(mods::Tuple{Vararg{Symbol}})
    isempty(mods) && return mods
    first(mods) in (:arrive, :arrive_drop) || return mods
    Symbol("shared::cluster") in mods && return mods
    (first(mods), :sink, Base.tail(mods)...)
end

# Resolve the PTX operand-overloaded primary-report head into an explicit Julia
# selector, then validate the exact destination/source structure before shared
# pointer aliasing or generic destination heuristics can erase it.
function _mbarrier_register_decl(cg::CodeGenState, name::AbstractString)
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

_mbarrier_register_decl(cg::CodeGenState, op::RegisterOperand) =
    _mbarrier_register_decl(cg, op.name)

function _mbarrier_require_register_type(cg::CodeGenState,
                                         op::RegisterOperand,
                                         accepted,
                                         role::AbstractString)
    decl = _mbarrier_register_decl(cg, op)
    decl === nothing && throw(ArgumentError(
        "PTX transpiler: mbarrier $role register $(op.name) has no preceding " *
        ".reg declaration"))
    decl.type in accepted || throw(ArgumentError(
        "PTX transpiler: mbarrier $role register $(op.name) must be " *
        join(IR.ptx.(accepted), "/") * ", got $(IR.ptx(decl.type))"))
end

const _MBARRIER_REG8 =
    (ScalarType.B8, ScalarType.U8, ScalarType.S8)
const _MBARRIER_REG32 =
    (ScalarType.B32, ScalarType.U32, ScalarType.S32)
const _MBARRIER_REG64 =
    (ScalarType.B64, ScalarType.U64, ScalarType.S64)

function _validate_mbarrier_destination_types!(cg::CodeGenState, schema,
                                               destinations)
    for (i, op) in enumerate(destinations)
        accepted, role = if schema.destination === :state
            (_MBARRIER_REG64, "state destination")
        elseif schema.destination === :predicate
            ((ScalarType.PRED,), "predicate destination")
        elseif schema.destination === :count
            (_MBARRIER_REG32, "pending-count destination")
        elseif schema.destination === :report_pred ||
               (schema.destination === :report && i <= 2)
            ((ScalarType.PRED,), i == 1 ? "waitComplete destination" :
                                         "reportPredicate destination")
        elseif schema.destination === :report && i == 3
            (_MBARRIER_REG8, "reportValue destination")
        else
            error("invalid mbarrier destination type shape $(schema.destination)")
        end
        _mbarrier_require_register_type(cg, op, accepted, role)
    end
end

function _validate_mbarrier_source_type!(cg::CodeGenState, kind::Symbol,
                                         op::Operand, index::Int)
    if op isa RegisterOperand && kind in (:u32, :u64)
        accepted = kind === :u32 ? _MBARRIER_REG32 : _MBARRIER_REG64
        _mbarrier_require_register_type(cg, op, accepted, "source $index")
    elseif op isa AddressOperand && startswith(op.base, "%")
        decl = _mbarrier_register_decl(cg, op.base)
        decl === nothing && throw(ArgumentError(
            "PTX transpiler: mbarrier address register $(op.base) has no " *
            "preceding .reg declaration"))
        decl.type in (_MBARRIER_REG32..., _MBARRIER_REG64...) ||
            throw(ArgumentError(
                "PTX transpiler: mbarrier address register $(op.base) must " *
                "be 32 or 64 bits, got $(IR.ptx(decl.type))"))
    end
end

function _instruction_mbarrier_schema(cg::CodeGenState, inst::Instruction)
    inst.opcode == "mbarrier" || return nothing
    mods = _schema_modifiers(inst.modifiers)
    operands = inst.operands
    destination_operands = Operand[]
    source_start = 1

    if !isempty(operands) && operands[1] isa PipeOperand
        isempty(mods) && throw(mbarrier_schema_miss(mods))
        pipe = operands[1]::PipeOperand
        pipe.left isa RegisterOperand && pipe.right isa RegisterOperand ||
            throw(ArgumentError(
                "PTX transpiler: mbarrier primary report destination must be " *
                "a predicate register pair waitComplete|reportPredicate"))
        if length(operands) >= 2 && operands[2] isa AddressOperand
            selector = :report_pred
            append!(destination_operands, (pipe.left, pipe.right))
            source_start = 2
        elseif length(operands) >= 3 && operands[3] isa AddressOperand
            operands[2] isa RegisterOperand || throw(ArgumentError(
                "PTX transpiler: mbarrier reportValue destination must be a register"))
            selector = :report
            append!(destination_operands, (pipe.left, pipe.right, operands[2]))
            source_start = 3
        else
            throw(ArgumentError(
                "PTX transpiler: mbarrier primary report must place an address " *
                "after waitComplete|reportPredicate and optional reportValue"))
        end
        mods = (first(mods), selector, Base.tail(mods)...)
    elseif !isempty(operands) && _is_sink_operand(operands[1])
        # PTX spells the discarded state as destination `_`; the Julia chain
        # needs an explicit synthetic selector because argument/result arity
        # alone cannot distinguish it from the state-returning form. Explicit
        # shared::cluster already has a mandatory sink ABI in its base schema.
        mods = _mbarrier_sink_selector_mods(mods)
    end

    schema = mbarrier_form_schema(:mbarrier, mods)
    schema === nothing && throw(mbarrier_schema_miss(mods))

    if isempty(destination_operands)
        if schema.destination === :none
            source_start = 1
        elseif schema.destination in (:sink, :remote_sink)
            !isempty(operands) && _is_sink_operand(operands[1]) ||
                throw(ArgumentError(
                    "PTX transpiler: sink-result mbarrier arrival requires " *
                    "the `_` destination"))
            source_start = 2
        elseif schema.destination === :state
            isempty(operands) && throw(ArgumentError(
                "PTX transpiler: mbarrier state form is missing its destination"))
            operands[1] isa RegisterOperand && !_is_sink_operand(operands[1]) ||
                throw(ArgumentError(
                    "PTX transpiler: mbarrier state destination must be a register"))
            push!(destination_operands, operands[1])
            source_start = 2
        elseif schema.destination in (:predicate, :count)
            !isempty(operands) && operands[1] isa RegisterOperand ||
                throw(ArgumentError(
                    "PTX transpiler: mbarrier destination must be a register"))
            push!(destination_operands, operands[1])
            source_start = 2
        else
            throw(ArgumentError(
                "PTX transpiler: grouped mbarrier report destination must use `|`"))
        end
    end

    sources = source_start > length(operands) ? Operand[] :
              collect(operands[source_start:end])
    variant_index = findfirst(v -> length(v.operands) == length(sources),
                              schema.variants)
    variant_index === nothing && throw(ArgumentError(
        "PTX transpiler: mbarrier.$(join(schema.mods, ".")) has invalid source " *
        "arity $(length(sources)); see $(schema.section)"))
    variant = schema.variants[variant_index]
    for (i, (kind, op)) in enumerate(zip(variant.operands, sources))
        _mbarrier_source_operand(kind, op) && continue
        throw(ArgumentError(
            "PTX transpiler: mbarrier.$(join(schema.mods, ".")) source $i " *
            "does not match the audited $kind operand role; see $(schema.section)"))
    end
    _validate_mbarrier_destination_types!(cg, schema, destination_operands)
    for (i, (kind, op)) in enumerate(zip(variant.operands, sources))
        _validate_mbarrier_source_type!(cg, kind, op, i)
    end
    (; schema, variant, sources, destination_operands)
end

function _emit_mbarrier!(cg::CodeGenState, inst::Instruction, checked)
    schema, variant = checked.schema, checked.variant
    chain = chain_expr(cg, "mbarrier",
                       Tuple("." * string(mod) for mod in schema.mods))
    args = String[]
    for (kind, operand) in zip(variant.operands, checked.sources)
        push!(args, kind === :address ? render_operand(operand, cg) :
              render_operand(operand, cg; type_hint = kind))
    end
    call = chain * "(" * join(args, ", ") * ")"

    if isempty(checked.destination_operands)
        emit_with_predicate!(cg, call, inst.predicate, String[])
        return
    end
    names = String[]
    rendered = String[]
    for operand in checked.destination_operands
        dst, dst_names = render_dst(operand, cg)
        push!(rendered, dst)
        append!(names, dst_names)
    end
    dst = length(rendered) == 1 ? only(rendered) :
          "(" * join(rendered, ", ") * ")"
    emit_with_predicate!(cg, dst * " = " * call, inst.predicate, names)
end

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

    mbarrier_schema = _instruction_mbarrier_schema(cg, inst)
    mbarrier_schema === nothing ||
        return _emit_mbarrier!(cg, inst, mbarrier_schema)

    # Close fixed-result grammar islands before any instruction can be erased
    # by pointer-alias absorption. The schema also provides a distinct type
    # hint for every source operand; one terminal hint is wrong for mixed
    # precision, mixed-sign dot products, widened arithmetic, and cvt.pack.
    scalar_schema = _instruction_scalar_result_schema(inst)

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
    scalar_schema === nothing && inst.predicate === nothing &&
        _try_alias_def!(cg, inst) && return

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
    src_strs = if scalar_schema === nothing
        [render_operand(op, cg; type_hint) for op in inst.operands[2:end]]
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
