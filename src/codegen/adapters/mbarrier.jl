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

function _mbarrier_is_declared_register(cg::CodeGenState, op::Operand)
    (op isa RegisterOperand || op isa LabelOperand) || return false
    _is_sink_operand(op) && return false
    _declared_register(cg, op) !== nothing
end

function _mbarrier_require_register_type(cg::CodeGenState,
                                         op::Union{RegisterOperand, LabelOperand},
                                         accepted,
                                         role::AbstractString)
    decl = _declared_register(cg, op)
    decl === nothing && throw(ArgumentError(
        "PTX transpiler: mbarrier $role register $(op.name) has no preceding " *
        ".reg declaration"))
    decl.type in accepted || throw(ArgumentError(
        "PTX transpiler: mbarrier $role register $(op.name) must be " *
        join(IR.ptx.(accepted), "/") * ", got $(IR.ptx(decl.type))"))
end

function _validate_mbarrier_destination_types!(cg::CodeGenState, schema,
                                               destinations)
    for (i, op) in enumerate(destinations)
        accepted, role = if schema.destination === :state
            (_REGISTER_TYPES_64, "state destination")
        elseif schema.destination === :predicate
            ((ScalarType.PRED,), "predicate destination")
        elseif schema.destination === :count
            (_REGISTER_TYPES_32, "pending-count destination")
        elseif schema.destination === :report_pred ||
               (schema.destination === :report && i <= 2)
            ((ScalarType.PRED,), i == 1 ? "waitComplete destination" :
                                         "reportPredicate destination")
        elseif schema.destination === :report && i == 3
            (_REGISTER_TYPES_8, "reportValue destination")
        else
            error("invalid mbarrier destination type shape $(schema.destination)")
        end
        _mbarrier_require_register_type(cg, op, accepted, role)
    end
end

function _validate_mbarrier_source_type!(cg::CodeGenState, kind::Symbol,
                                         op::Operand, index::Int)
    if (op isa RegisterOperand || op isa LabelOperand) &&
            kind in (:u32, :u64) &&
            (op isa RegisterOperand || _declared_register(cg, op) !== nothing)
        accepted = kind === :u32 ? _REGISTER_TYPES_32 : _REGISTER_TYPES_64
        _mbarrier_require_register_type(cg, op, accepted, "source $index")
    elseif op isa AddressOperand
        decl = _declared_register(cg, op.base)
        startswith(op.base, "%") && decl === nothing && throw(ArgumentError(
            "PTX transpiler: mbarrier address register $(op.base) has no " *
            "preceding .reg declaration"))
        decl === nothing && return
        decl.type in (_REGISTER_TYPES_32..., _REGISTER_TYPES_64...) ||
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
        isempty(mods) && throw(miss(MBarrierLedger(), :mbarrier, mods))
        pipe = operands[1]::PipeOperand
        _mbarrier_is_declared_register(cg, pipe.left) &&
            _mbarrier_is_declared_register(cg, pipe.right) ||
            throw(ArgumentError(
                "PTX transpiler: mbarrier primary report destination must be " *
                "a declared predicate register pair " *
                "waitComplete|reportPredicate"))
        if length(operands) >= 2 && operands[2] isa AddressOperand
            selector = :report_pred
            append!(destination_operands, (pipe.left, pipe.right))
            source_start = 2
        elseif length(operands) >= 3 && operands[3] isa AddressOperand
            _mbarrier_is_declared_register(cg, operands[2]) ||
                throw(ArgumentError(
                    "PTX transpiler: mbarrier reportValue destination must " *
                    "be a declared register"))
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

    s = schema(MBarrierLedger(), :mbarrier, mods)
    s === nothing && throw(miss(MBarrierLedger(), :mbarrier, mods))

    if isempty(destination_operands)
        if s.destination === :none
            source_start = 1
        elseif s.destination in (:sink, :remote_sink)
            !isempty(operands) && _is_sink_operand(operands[1]) ||
                throw(ArgumentError(
                    "PTX transpiler: sink-result mbarrier arrival requires " *
                    "the `_` destination"))
            source_start = 2
        elseif s.destination === :state
            isempty(operands) && throw(ArgumentError(
                "PTX transpiler: mbarrier state form is missing its destination"))
            _mbarrier_is_declared_register(cg, operands[1]) ||
                throw(ArgumentError(
                    "PTX transpiler: mbarrier state destination must be a " *
                    "declared register"))
            push!(destination_operands, operands[1])
            source_start = 2
        elseif s.destination in (:predicate, :count)
            !isempty(operands) &&
                _mbarrier_is_declared_register(cg, operands[1]) ||
                throw(ArgumentError(
                    "PTX transpiler: mbarrier destination must be a declared " *
                    "register"))
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
                              s.variants)
    variant_index === nothing && throw(ArgumentError(
        "PTX transpiler: mbarrier.$(join(s.mods, ".")) has invalid source " *
        "arity $(length(sources)); see $(s.section)"))
    variant = s.variants[variant_index]
    for (i, (kind, op)) in enumerate(zip(variant.operands, sources))
        _mbarrier_source_operand(kind, op) && continue
        throw(ArgumentError(
            "PTX transpiler: mbarrier.$(join(s.mods, ".")) source $i " *
            "does not match the audited $kind operand role; see $(s.section)"))
    end
    _validate_mbarrier_destination_types!(cg, s, destination_operands)
    for (i, (kind, op)) in enumerate(zip(variant.operands, sources))
        _validate_mbarrier_source_type!(cg, kind, op, i)
    end
    (; schema = s, variant, sources, destination_operands)
end

function transpile_ledger!(::MBarrierLedger, cg::CodeGenState,
                           inst::Instruction)
    checked = _instruction_mbarrier_schema(cg, inst)
    checked === nothing && return false
    _emit_mbarrier!(cg, inst, checked)
    true
end

function _emit_mbarrier!(cg::CodeGenState, inst::Instruction, checked)
    schema, variant = checked.schema, checked.variant
    chain = chain_expr(cg, "mbarrier",
                       Tuple("." * string(mod) for mod in schema.mods))
    args = String[]
    for (kind, operand) in zip(variant.operands, checked.sources)
        rendered_operand = if operand isa LabelOperand &&
                _declared_register(cg, operand) !== nothing
            RegisterOperand(operand.name)
        else
            operand
        end
        push!(args, kind === :address ? render_operand(rendered_operand, cg) :
              render_operand(rendered_operand, cg; type_hint = kind))
    end
    call = chain * "(" * join(args, ", ") * ")"

    if isempty(checked.destination_operands)
        emit_with_predicate!(cg, call, inst.predicate, String[])
        return
    end
    names = String[]
    rendered = String[]
    for operand in checked.destination_operands
        dst, dst_names = operand isa LabelOperand ?
            render_dst(RegisterOperand(operand.name), cg) : render_dst(operand, cg)
        push!(rendered, dst)
        append!(names, dst_names)
    end
    dst = length(rendered) == 1 ? only(rendered) :
          "(" * join(rendered, ", ") * ")"
    emit_with_predicate!(cg, dst * " = " * call, inst.predicate, names)
end
