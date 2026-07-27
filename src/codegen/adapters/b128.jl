const _B128_REGISTER_TYPES = (ScalarType.B128,)

function _b128_require_register_type(cg::CodeGenState,
                                     op::Union{RegisterOperand, LabelOperand},
                                     accepted, role::AbstractString)
    decl = _declared_register(cg, op)
    decl === nothing && accepted === _B128_REGISTER_TYPES &&
        op.name in cg.inferred_b128_regs && return
    decl === nothing && throw(ArgumentError(
        "PTX transpiler: b128 $role register $(op.name) has no preceding " *
        ".reg declaration"))
    decl.type in accepted || throw(ArgumentError(
        "PTX transpiler: b128 $role register $(op.name) must be " *
        join(IR.ptx.(accepted), "/") * ", got $(IR.ptx(decl.type))"))
end

function _instruction_b128_schema(cg::CodeGenState, inst::Instruction)
    op = Symbol(inst.opcode)
    mods = _schema_modifiers(inst.modifiers)
    s = schema(B128Ledger(), op, mods)
    s === nothing && claims(B128Ledger(), op, mods) &&
        throw(miss(B128Ledger(), op, mods))
    s === nothing && return nothing

    operands = inst.operands
    source_start = 2
    destination = nothing
    sources = Operand[]
    if s.kind === :store
        length(operands) == length(s.operands) || throw(ArgumentError(
            "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) requires " *
            "$(length(s.operands)) operands, got $(length(operands)); " *
            "see $(s.section)"))
        append!(sources, operands)
    else
        length(operands) == length(s.operands) + 1 || throw(ArgumentError(
            "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) requires " *
            "one destination and $(length(s.operands)) source " *
            "operand(s), got $(length(operands)); see $(s.section)"))
        destination = operands[1]
        append!(sources, operands[2:end])
    end

    if s.kind === :mov
        (destination isa RegisterOperand || destination isa LabelOperand) ||
            throw(ArgumentError("PTX transpiler: mov.b128 destination must be a register"))
        decl = _declared_register(cg, destination)
        decl === nothing || _b128_require_register_type(
            cg, destination, _B128_REGISTER_TYPES, "mov destination")
        source = only(sources)
        source isa VectorOperand || throw(ArgumentError(
            "PTX transpiler: mov.b128 pack source must be a brace-enclosed " *
            "v2.b64 or v4.b32 register group"))
        lane_types = length(source.elements) == 2 ? _REGISTER_TYPES_64 :
                     length(source.elements) == 4 ? _REGISTER_TYPES_32 : nothing
        lane_types === nothing && throw(ArgumentError(
            "PTX transpiler: mov.b128 pack source must have two b64 or four b32 lanes"))
        for lane in source.elements
            (lane isa RegisterOperand || lane isa LabelOperand) ||
                throw(ArgumentError("PTX transpiler: mov.b128 pack lanes must be registers"))
            _b128_require_register_type(cg, lane, lane_types, "mov source lane")
        end
        return (; schema = s, destination, sources, mov_source = source,
                  discard_result = false)
    end

    if s.kind in (:load, :exch, :cas)
        if !_is_sink_operand(destination)
            (destination isa RegisterOperand || destination isa LabelOperand) ||
                throw(ArgumentError("PTX transpiler: b128 destination must be a register or `_`"))
            _b128_require_register_type(cg, destination, _B128_REGISTER_TYPES,
                                        "destination")
        else
            throw(ArgumentError(
                "PTX transpiler: scalar b128 load/atom destination `_` is not " *
                "accepted: PTX 9.3 grants sinks only to specific vector/simple-" *
                "reduction cases, not atom.{exch,cas}.b128"))
        end
    elseif s.kind === :query_pred
        (destination isa RegisterOperand || destination isa LabelOperand) ||
            throw(ArgumentError("PTX transpiler: query_cancel predicate destination must be a register"))
        _b128_require_register_type(cg, destination, (ScalarType.PRED,),
                                    "query predicate destination")
    elseif s.kind === :query_dim
        (destination isa RegisterOperand || destination isa LabelOperand) ||
            throw(ArgumentError("PTX transpiler: query_cancel dimension destination must be a register"))
        _b128_require_register_type(cg, destination, _REGISTER_TYPES_32,
                                    "query dimension destination")
    elseif s.kind === :query_v4
        destination isa VectorOperand && length(destination.elements) == 4 ||
            throw(ArgumentError(
                "PTX transpiler: query_cancel v4 destination must contain four registers"))
        for lane in destination.elements
            (lane isa RegisterOperand || lane isa LabelOperand) ||
                throw(ArgumentError("PTX transpiler: query_cancel v4 lanes must be registers"))
            _b128_require_register_type(cg, lane, _REGISTER_TYPES_32,
                                        "query v4 destination lane")
        end
    end

    for (i, (kind, source)) in enumerate(zip(s.operands, sources))
        if kind === :address
            source isa AddressOperand || throw(ArgumentError(
                "PTX transpiler: b128 operand $i must be bracketed in input PTX"))
            _validate_vector_address!(cg, source, "b128 address operand $i")
        elseif kind === :b128
            (source isa RegisterOperand || source isa LabelOperand) ||
                throw(ArgumentError("PTX transpiler: b128 operand $i must be a register"))
            _b128_require_register_type(cg, source, _B128_REGISTER_TYPES,
                                        "source $i")
        elseif kind === :cache_policy
            (source isa RegisterOperand || source isa LabelOperand) ||
                throw(ArgumentError("PTX transpiler: b128 cache policy must be a register"))
            _b128_require_register_type(cg, source, _REGISTER_TYPES_64,
                                        "cache policy")
        end
    end
    (; schema = s, destination, sources, mov_source = nothing,
       discard_result = destination !== nothing && _is_sink_operand(destination))
end

function _render_b128_register(cg::CodeGenState, op::Operand)
    op isa LabelOperand && _declared_register(cg, op) !== nothing &&
        return render_operand(RegisterOperand(op.name), cg)
    render_operand(op, cg)
end

function _emit_b128!(cg::CodeGenState, inst::Instruction, checked)
    schema = checked.schema
    chain = chain_expr(cg, inst.opcode,
                       Tuple("." * string(mod) for mod in schema.mods))
    args = String[]
    if schema.kind === :mov
        lanes = [_render_b128_register(cg, lane)
                 for lane in checked.mov_source.elements]
        carrier = length(lanes) == 2 ? "(" * join(lanes, ", ") * ")" :
                  "b128(" * join(lanes, ", ") * ")"
        push!(args, carrier)
    else
        for (kind, source) in zip(schema.operands, checked.sources)
            rendered = kind === :address ? render_operand(source, cg) :
                       _render_b128_register(cg, source)
            push!(args, rendered)
        end
    end
    call = chain * "(" * join(args, ", ") * ")"
    if schema.kind === :store || checked.discard_result
        emit_with_predicate!(cg, call, inst.predicate, String[])
        return
    end
    destination = checked.destination
    if destination isa VectorOperand
        rendered = String[]
        names = String[]
        for lane in destination.elements
            dst, dst_names = lane isa LabelOperand ?
                render_dst(RegisterOperand(lane.name), cg) : render_dst(lane, cg)
            push!(rendered, dst)
            append!(names, dst_names)
        end
        emit_with_predicate!(cg, "(" * join(rendered, ", ") * ") = " * call,
                             inst.predicate, names)
    else
        dst, names = destination isa LabelOperand ?
            render_dst(RegisterOperand(destination.name), cg) :
            render_dst(destination, cg)
        emit_with_predicate!(cg, dst * " = " * call, inst.predicate, names)
        schema.kind === :mov && push!(cg.inferred_b128_regs,
                                      checked.destination.name)
    end
end

function transpile_ledger!(::B128Ledger, cg::CodeGenState,
                           inst::Instruction)
    checked = _instruction_b128_schema(cg, inst)
    checked === nothing && return false
    _emit_b128!(cg, inst, checked)
    true
end
