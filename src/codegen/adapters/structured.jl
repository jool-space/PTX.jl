function _structured_api_modifiers(inst::Instruction)
    mods = _schema_modifiers(inst.modifiers)
    isempty(inst.operands) && return mods
    inst.operands[1] isa PipeOperand || return mods
    if inst.opcode == "setp"
        # Packed half types intrinsically require p|q. General scalar setp
        # needs the PTX.jl-only :dual selector because the emitted spelling
        # also admits a single destination. Scalar f16/bf16 + pipe becomes an
        # intentional schema miss.
        any(t -> t in (:f16x2, :bf16x2), mods) && return mods
        return (:dual, mods...)
    elseif inst.opcode == "match"
        return (mods..., :pred)
    end
    mods
end

function _structured_decl_types(kind::Symbol)
    kind === :pred && return (ScalarType.PRED,)
    kind === :f16 && return (ScalarType.F16, ScalarType.B16)
    kind === :bf16 && return (ScalarType.B16, ScalarType.BF16)
    kind === :f32 && return (ScalarType.F32, ScalarType.B32)
    kind === :f64 && return (ScalarType.F64, ScalarType.B64)
    kind === :b8 && return (ScalarType.B8, ScalarType.U8, ScalarType.S8)
    kind in (:u8, :s8) &&
        return (ScalarType.B8, ScalarType.U8, ScalarType.S8)
    kind === :b16 && return (ScalarType.B16, ScalarType.U16,
                             ScalarType.S16, ScalarType.F16,
                             ScalarType.BF16)
    kind === :b32 &&
        return (ScalarType.B32, ScalarType.U32, ScalarType.S32,
                ScalarType.F32, ScalarType.F16X2, ScalarType.BF16X2,
                ScalarType.TF32)
    kind in (:u32, :s32) &&
        return (ScalarType.B32, ScalarType.U32, ScalarType.S32)
    kind === :b64 &&
        return (ScalarType.B64, ScalarType.U64, ScalarType.S64,
                ScalarType.F64)
    kind in (:u64, :s64) &&
        return (ScalarType.B64, ScalarType.U64, ScalarType.S64)
    kind in (:u16, :s16) &&
        return (ScalarType.B16, ScalarType.U16, ScalarType.S16)
    ()
end

function _structured_destinations(op::Operand)
    op isa PipeOperand ? (op.left, op.right) : (op,)
end

function _validate_structured_decl!(cg::CodeGenState, op::Operand,
                                    kind::Symbol, role::AbstractString)
    _is_sink_operand(op) && return
    (op isa RegisterOperand || op isa LabelOperand) || throw(ArgumentError(
        "PTX transpiler: structured $role must be a register or `_`"))
    decl = _declared_register(cg, op)
    decl === nothing && throw(ArgumentError(
        "PTX transpiler: structured $role register $(op.name) has no " *
        "preceding .reg declaration"))
    accepted = _structured_decl_types(kind)
    decl.type in accepted || throw(ArgumentError(
        "PTX transpiler: structured $role register $(op.name) must be " *
        join(IR.ptx.(accepted), "/") * ", got $(IR.ptx(decl.type))"))
end

function _structured_source_base(op::Operand)
    op isa NegatedOperand ? op.operand : op
end

const _STRUCTURED_INTEGER_SOURCE_KINDS =
    (:b16, :b32, :b64, :u16, :u32, :u64, :s16, :s32, :s64)

function _structured_integer_carrier_type(kind::Symbol)
    kind in (:b16, :u16) && return UInt16
    kind in (:b32, :u32) && return UInt32
    kind in (:b64, :u64) && return UInt64
    kind === :s16 && return Int16
    kind === :s32 && return Int32
    kind === :s64 && return Int64
    error("unknown structured integer source role $kind")
end

function _structured_integer_text(op::Operand)
    try
        _ptx_integer_constant_text(op)
    catch err
        err isa ArgumentError || rethrow()
        nothing
    end
end

function _validate_structured_integer_constant(text::AbstractString,
                                               kind::Symbol, index::Int)
    try
        _ptx_integer_constant(text)
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError(
            "PTX transpiler: structured source $index has invalid $kind " *
            "integer constant: $(sprint(showerror, err))"))
    end
    nothing
end

function _validate_structured_source!(cg::CodeGenState, op::Operand,
                                      kind::Symbol, index::Int)
    if kind === :imm8
        try
            _lop3_lut_value(op)
        catch err
            err isa ArgumentError || rethrow()
            throw(ArgumentError(
                "PTX transpiler: structured source $index has invalid " *
                "lop3 immLut: $(sprint(showerror, err))"))
        end
        return
    end
    integer_text = _structured_integer_text(op)
    if kind in _STRUCTURED_INTEGER_SOURCE_KINDS && integer_text !== nothing
        _validate_structured_integer_constant(integer_text, kind, index)
        return
    elseif kind === :pred && integer_text !== nothing
        _validate_structured_integer_constant(integer_text, kind, index)
        return
    elseif kind in (:f16, :bf16) && integer_text !== nothing
        # CUDA 13.3 ptxas rejects both integer and floating immediates in the
        # half/bfloat setp source positions. Their schema carriers are
        # therefore registers only.
        throw(ArgumentError(
            "PTX transpiler: structured source $index for $kind must be a " *
            "register; ptxas rejects immediate operands for half setp"))
    elseif kind in (:f32, :f64) && integer_text !== nothing
        # Integer constants are not compatible with floating instruction
        # types (PTX §6.1), and ptxas rejects them. A floating literal or
        # expression does not parse through the integer evaluator and remains
        # accepted by the ordinary floating renderer below.
        integer_value = try
            _ptx_integer_constant(integer_text)
        catch err
            err isa ArgumentError || rethrow()
            nothing
        end
        integer_value === nothing || throw(ArgumentError(
            "PTX transpiler: structured source $index for $kind cannot use " *
            "the integer constant $(repr(integer_text)); use a floating " *
            "constant or a compatible register"))
        return
    end
    base = _structured_source_base(op)
    base isa ImmediateOperand && throw(ArgumentError(
        "PTX transpiler: structured source $index does not match the audited " *
        "$kind operand role"))
    if kind === :pred
        (op isa NegatedOperand || op === base) || throw(ArgumentError(
            "PTX transpiler: predicate source $index has invalid negation shape"))
    end
    (base isa RegisterOperand || base isa LabelOperand) || throw(ArgumentError(
        "PTX transpiler: structured source $index does not match the audited " *
        "$kind operand role"))
    _validate_structured_decl!(cg, base, kind, "source $index")
end

function _instruction_structured_result_schema(cg::CodeGenState,
                                               inst::Instruction)
    op = Symbol(inst.opcode)
    mods = _structured_api_modifiers(inst)
    claims(StructuredLedger(), op, mods) || return nothing
    s = schema(StructuredLedger(), op, mods)
    s === nothing && throw(miss(StructuredLedger(), op, mods))
    isempty(inst.operands) && throw(ArgumentError(
        "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) is missing " *
        "its destination"))
    destinations = _structured_destinations(inst.operands[1])
    length(destinations) == length(s.outputs) || throw(ArgumentError(
        "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) requires " *
        "$(length(s.outputs)) destination(s), got $(length(destinations)); " *
        "see $(s.section)"))
    length(inst.operands) == length(s.operands) + 1 || throw(ArgumentError(
        "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) requires " *
        "$(length(s.operands)) source operands, got " *
        "$(max(length(inst.operands) - 1, 0)); see $(s.section)"))
    sink_count = count(_is_sink_operand, destinations)
    sink_count <= s.max_sinks || throw(ArgumentError(
        "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) may discard " *
        "at most $(s.max_sinks) destination(s) with `_`; see " *
        s.section))
    for (i, (dest, kind)) in enumerate(zip(destinations, s.outputs))
        if _is_sink_operand(dest)
            s.sinkable[i] || throw(ArgumentError(
                "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) " *
                "destination $i may not use `_`; see $(s.section)"))
            continue
        end
        _validate_structured_decl!(cg, dest, kind, "destination $i")
    end
    for (i, (source, kind)) in enumerate(zip(inst.operands[2:end],
                                              s.operands))
        _validate_structured_source!(cg, source, kind, i)
    end
    (; schema = s, mods)
end

# Structured-result schemas choose their reviewed API modifiers (e.g. the
# synthetic :dual/:pred selectors for grouped destinations).
function transpile_ledger!(::StructuredLedger, cg::CodeGenState,
                           inst::Instruction)
    checked = _instruction_structured_result_schema(cg, inst)
    checked === nothing && return false
    modifiers = Tuple("." * string(mod) for mod in checked.mods)
    dst_expr, dst_names = _render_structured_dst(inst.operands[1], cg)
    src_strs = [_render_structured_source(op, cg, kind)
                for (op, kind) in zip(inst.operands[2:end],
                                      checked.schema.operands)]
    _emit_schema_call!(cg, inst, modifiers, dst_expr, dst_names, src_strs)
    true
end

function _render_structured_dst(op::LabelOperand, cg::CodeGenState)
    _declared_register(cg, op) === nothing ? render_dst(op, cg) :
        render_dst(RegisterOperand(op.name), cg)
end

function _render_structured_dst(op::PipeOperand, cg::CodeGenState)
    left, left_names = _render_structured_dst(op.left, cg)
    right, right_names = _render_structured_dst(op.right, cg)
    ("(" * left * ", " * right * ")", [left_names; right_names])
end

_render_structured_dst(op::Operand, cg::CodeGenState) = render_dst(op, cg)

function _render_structured_source(op::Operand, cg::CodeGenState, kind::Symbol)
    if kind === :imm8
        return "Val($(_lop3_lut_value(op)))"
    end
    integer_text = _structured_integer_text(op)
    if kind in _STRUCTURED_INTEGER_SOURCE_KINDS && integer_text !== nothing
        carrier = _structured_integer_carrier_type(kind)
        return _ptx_integer_carrier_expr(integer_text, carrier)
    elseif kind === :pred && integer_text !== nothing
        return _ptx_predicate_constant(integer_text) ? "true" : "false"
    end
    _render_schema_source(op, cg, kind)
end
