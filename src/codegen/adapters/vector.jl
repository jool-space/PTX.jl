function _instruction_vector_result_schema(cg::CodeGenState, inst::Instruction)
    op = Symbol(inst.opcode)
    mods = _schema_modifiers(inst.modifiers)
    schema = vector_result_schema(op, mods)
    schema === nothing && requires_vector_result_schema(op, mods) &&
        throw(vector_result_schema_miss(op, mods))
    schema === nothing && return nothing

    source_count = max(length(inst.operands) - 1, 0)
    roles = vector_result_operand_roles(schema, source_count)
    roles === nothing && throw(ArgumentError(
        "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) is an audited " *
        "vector-result form with invalid source arity $source_count; " *
        "see $(schema.form.section)"))
    destination = inst.operands[1]
    # PTX's atom bit bucket replaces the complete vector destination with one
    # `_`; it is not a per-lane vector sink. Preserve the operation by emitting
    # the ordinary tuple-returning chain call as an unused Julia statement.
    discard_result = schema.form.op === :atom && _is_sink_operand(destination)
    mask = ntuple(_ -> true, schema.form.lanes)
    if !discard_result
        destination isa VectorOperand || throw(ArgumentError(
            "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) requires a " *
            "brace-enclosed v$(schema.form.lanes) destination" *
            (schema.form.op === :atom ? " or the complete bit bucket `_`" : "")))
        length(destination.elements) == schema.form.lanes || throw(ArgumentError(
            "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) destination " *
            "has $(length(destination.elements)) lanes, expected $(schema.form.lanes)"))
        mask = Tuple(!_is_sink_operand(element) for element in destination.elements)
        validate_vector_result_mask(schema, mask)
        for (i, element) in enumerate(destination.elements)
            _is_sink_operand(element) && continue
            element isa RegisterOperand || element isa LabelOperand ||
                throw(ArgumentError(
                    "PTX transpiler: vector destination lane $i must be a " *
                    "register or `_`"))
        end
    end

    sources = inst.operands[2:end]
    for (i, (kind, operand)) in enumerate(zip(roles, sources))
        if kind === :address
            operand isa AddressOperand || throw(ArgumentError(
                "PTX transpiler: vector-result source $i must be an address operand"))
        elseif kind === :vector
            operand isa VectorOperand || throw(ArgumentError(
                "PTX transpiler: vector atom source $i must be brace-enclosed"))
            length(operand.elements) == schema.form.lanes || throw(ArgumentError(
                "PTX transpiler: vector atom source has $(length(operand.elements)) " *
                "lanes, expected $(schema.form.lanes)"))
            any(_is_sink_operand, operand.elements) && throw(ArgumentError(
                "PTX transpiler: `_` is not legal in a vector atom input"))
        elseif kind === :cache_policy
            operand isa RegisterOperand || operand isa LabelOperand ||
                operand isa ImmediateOperand || throw(ArgumentError(
                    "PTX transpiler: cache policy must be a 64-bit register or immediate"))
        end
    end
    _validate_vector_result_operands!(cg, schema, destination, sources, roles,
                                      discard_result)
    (; schema, destination, sources, roles, mask, discard_result)
end

# The vector-result API exposes one exact Julia carrier per PTX lane. PTX `ld`
# also permits a destination register wider than its instruction type (§9.4.1),
# but that changes the observable result width and extension semantics. Until
# the structured API can represent those widened lanes, accept only exact-size
# declarations and fail before emitting a narrower Julia tuple.
const _VECTOR_RESULT_REGISTER_TYPES = Dict(
    :b8 => (ScalarType.B8, ScalarType.U8, ScalarType.S8),
    :u8 => (ScalarType.B8, ScalarType.U8, ScalarType.S8),
    :s8 => (ScalarType.B8, ScalarType.U8, ScalarType.S8),
    :b16 => (ScalarType.B16, ScalarType.U16, ScalarType.S16, ScalarType.F16),
    :u16 => (ScalarType.B16, ScalarType.U16, ScalarType.S16),
    :s16 => (ScalarType.B16, ScalarType.U16, ScalarType.S16),
    :f16 => (ScalarType.B16, ScalarType.F16),
    # Alternate floating-point formats are not fundamental register types;
    # PTX 9.3 §5.2.3 and §5.2.5.1 require bit-size carriers for them.
    :bf16 => (ScalarType.B16,),
    :b32 => (ScalarType.B32, ScalarType.U32, ScalarType.S32,
             ScalarType.F16X2, ScalarType.F32),
    :u32 => (ScalarType.B32, ScalarType.U32, ScalarType.S32),
    :s32 => (ScalarType.B32, ScalarType.U32, ScalarType.S32),
    :f32 => (ScalarType.B32, ScalarType.F32),
    :f16x2 => (ScalarType.B32, ScalarType.F16X2),
    :bf16x2 => (ScalarType.B32,),
    :b64 => (ScalarType.B64, ScalarType.U64, ScalarType.S64, ScalarType.F64),
    :u64 => (ScalarType.B64, ScalarType.U64, ScalarType.S64),
    :s64 => (ScalarType.B64, ScalarType.U64, ScalarType.S64),
    :f64 => (ScalarType.B64, ScalarType.F64),
    :e4m3 => (ScalarType.B8,),
    :e5m2 => (ScalarType.B8,),
    :e4m3x2 => (ScalarType.B16,),
    :e5m2x2 => (ScalarType.B16,),
    :e4m3x4 => (ScalarType.B32,),
    :e5m2x4 => (ScalarType.B32,),
)

function _vector_require_register_type(
        cg::CodeGenState, op::Union{RegisterOperand, LabelOperand}, accepted,
        role::AbstractString)
    decl = _declared_register(cg, op)
    decl === nothing && throw(ArgumentError(
        "PTX transpiler: vector-result $role register $(op.name) has no " *
        "preceding .reg declaration"))
    decl.type in accepted || throw(ArgumentError(
        "PTX transpiler: vector-result $role register $(op.name) must use " *
        "an exact-size " * join(IR.ptx.(accepted), "/") *
        " carrier, got $(IR.ptx(decl.type)); this exact homogeneous tuple " *
        "ABI does not lower wider or type-incompatible vector registers"))
end

function _validate_vector_address!(cg::CodeGenState, op::AddressOperand,
                                   role::AbstractString)
    op.coords === nothing || throw(ArgumentError(
        "PTX transpiler: vector-result $role uses an ordinary memory address, " *
        "not a tensor coordinate list"))
    startswith(op.base, "%") || return
    decl = _declared_register(cg, op.base)
    decl === nothing && throw(ArgumentError(
        "PTX transpiler: vector-result $role register $(op.base) has no " *
        "preceding .reg declaration"))
    decl.type in (_REGISTER_TYPES_32..., _REGISTER_TYPES_64...) ||
        throw(ArgumentError(
            "PTX transpiler: vector-result $role register $(op.base) must be " *
            "a 32- or 64-bit integer/bit register, got $(IR.ptx(decl.type))"))
end

function _validate_vector_result_operands!(cg::CodeGenState, schema,
                                           destination::Operand,
                                           sources, roles,
                                           discard_result::Bool)
    accepted = get(_VECTOR_RESULT_REGISTER_TYPES, schema.form.lane_kind, nothing)
    accepted === nothing && error(
        "missing vector-result declaration types for $(schema.form.lane_kind)")
    if !discard_result
        for (i, operand) in enumerate(destination.elements)
            _is_sink_operand(operand) && continue
            _vector_require_register_type(cg, operand, accepted,
                                          "destination lane $i")
        end
    end

    for (i, (kind, operand)) in enumerate(zip(roles, sources))
        if kind === :address
            _validate_vector_address!(cg, operand, "address operand $i")
        elseif kind === :vector
            for (lane, element) in enumerate(operand.elements)
                if element isa ImmediateOperand
                    schema.form.lane_kind === :f32 &&
                        _is_vector_f32_float_literal(element.text) && continue
                    throw(ArgumentError(
                        "PTX transpiler: vector atom source lane $lane for " *
                        ".$(schema.form.lane_kind) must be a compatible " *
                        "register" *
                        (schema.form.lane_kind === :f32 ?
                         " or floating-point constant" :
                         "; CUDA 12.9/13.3 ptxas reject immediate lanes " *
                         "for this format")))
                end
                (element isa RegisterOperand || element isa LabelOperand) ||
                    throw(ArgumentError(
                        "PTX transpiler: vector source lane $lane must be a " *
                        "compatible register"))
                _vector_require_register_type(cg, element, accepted,
                                              "source lane $lane")
            end
        elseif kind === :cache_policy
            operand isa ImmediateOperand && continue
            if operand isa LabelOperand &&
                    haskey(IR.PREDEFINED_IMMEDIATES, operand.name)
                continue
            end
            (operand isa RegisterOperand || operand isa LabelOperand) ||
                error("validated cache-policy operand changed shape")
            _vector_require_register_type(cg, operand, _REGISTER_TYPES_64,
                                          "cache-policy")
        end
    end
end

function _vector_lane_hint(kind::Symbol)
    kind === :bf16 && return :b16
    kind in (:f16x2, :bf16x2) && return :b32
    kind in (:e4m3, :e5m2) && return :b8
    kind in (:e4m3x2, :e5m2x2) && return :b16
    kind in (:e4m3x4, :e5m2x4) && return :b32
    kind
end

const _VECTOR_F32_EXACT_FLOAT_LITERAL =
    r"^0[fF][0-9a-fA-F]{8}$|^0[dD][0-9a-fA-F]{16}$"
const _VECTOR_F32_DECIMAL_FLOAT_LITERAL =
    r"^-?(?:(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+)$"

_is_vector_f32_float_literal(text::AbstractString) =
    occursin(_VECTOR_F32_EXACT_FLOAT_LITERAL, text) ||
    occursin(_VECTOR_F32_DECIMAL_FLOAT_LITERAL, text)

function _render_vector_f32_immediate(op::ImmediateOperand,
                                      cg::CodeGenState)
    # Exact 0f/0d literals retain their encoded width; decimal literals are
    # f64 constants. PTX converts all three accepted spellings to the f32 atom
    # source type at use. Make that conversion explicit in generated Julia.
    text = op.text
    rendered = if occursin(_VECTOR_F32_EXACT_FLOAT_LITERAL, text)
        if lowercase(text[2]) == 'f'
            bits = parse(UInt32, SubString(text, 3); base = 16)
            "reinterpret(Float32, 0x" *
                string(bits, base = 16, pad = 8) * ")"
        else
            bits = parse(UInt64, SubString(text, 3); base = 16)
            "reinterpret(Float64, 0x" *
                string(bits, base = 16, pad = 16) * ")"
        end
    else
        String(text)
    end
    "Float32(" * rendered * ")"
end

function _render_vector_value(op::Operand, cg::CodeGenState, schema)
    hint = _vector_lane_hint(schema.form.lane_kind)
    if op isa VectorOperand
        inner = join((_render_vector_value(element, cg, schema)
                      for element in op.elements), ", ")
        return "(" * inner * ")"
    elseif op isa ImmediateOperand
        schema.form.lane_kind === :f32 &&
            _is_vector_f32_float_literal(op.text) ||
            error("validated vector immediate changed shape")
        return _render_vector_f32_immediate(op, cg)
    elseif op isa LabelOperand && _declared_register(cg, op) !== nothing
        # A `.reg` name without `%` is lexed as a label; declaration validation
        # above proves that this occurrence is a value register.
        return render_operand(RegisterOperand(op.name), cg; type_hint = hint)
    end
    render_operand(op, cg; type_hint = hint)
end

function _render_vector_cache_policy(op::Operand, cg::CodeGenState)
    op isa ImmediateOperand &&
        return _ptx_integer_carrier_expr(op.text, UInt64)
    op isa LabelOperand && _declared_register(cg, op) !== nothing &&
        return render_operand(RegisterOperand(op.name), cg; type_hint = :u64)
    render_operand(op, cg; type_hint = :u64)
end

function _emit_vector_result!(cg::CodeGenState, inst::Instruction, checked)
    schema = checked.schema
    chain = chain_expr(cg, inst.opcode, inst.modifiers)
    args = String[]
    for (kind, operand) in zip(checked.roles, checked.sources)
        if kind === :address
            push!(args, render_operand(operand, cg))
        elseif kind === :cache_policy
            push!(args, _render_vector_cache_policy(operand, cg))
        else
            push!(args, _render_vector_value(operand, cg, schema))
        end
    end

    masked = !all(checked.mask)
    call = if masked
        mask_expr = "(" * join(string.(checked.mask), ", ") * ")"
        "vector_load(" * chain * ", " * join(args, ", ") *
            ", Val(" * mask_expr * "))"
    else
        chain * "(" * join(args, ", ") * ")"
    end

    if checked.discard_result
        # The chain call keeps its exact tuple ABI; ignoring that Julia value
        # models PTX atom's whole-result `_` without adding a synthetic direct/
        # raw selector. The inline asm remains side-effecting and memory-clobbering.
        emit_with_predicate!(cg, call, inst.predicate, String[])
        return
    end

    live = Operand[element for (element, keep) in
                   zip(checked.destination.elements, checked.mask) if keep]
    rendered = String[]
    names = String[]
    for operand in live
        rendered_operand = operand isa LabelOperand ?
            RegisterOperand(operand.name) : operand
        dst, dst_names = render_dst(rendered_operand, cg)
        push!(rendered, dst)
        append!(names, dst_names)
    end
    dst = length(rendered) == 1 ? "(" * only(rendered) * ",)" :
          "(" * join(rendered, ", ") * ")"
    emit_with_predicate!(cg, dst * " = " * call, inst.predicate, names)
end
