# Returns (dst_expr_string, defined_var_names::Vector{String}). The var-name
# list lets the function pass hoist `local` decls when any was assigned under
# predication.
function render_dst(op::RegisterOperand, cg::CodeGenState)
    name = julia_var(op.name)
    (name, [name])
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
    requires_structured_result_schema(op) || return nothing
    mods = _structured_api_modifiers(inst)
    schema = structured_result_schema(op, mods)
    schema === nothing && throw(structured_result_schema_miss(op, mods))
    isempty(inst.operands) && throw(ArgumentError(
        "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) is missing " *
        "its destination"))
    destinations = _structured_destinations(inst.operands[1])
    length(destinations) == length(schema.outputs) || throw(ArgumentError(
        "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) requires " *
        "$(length(schema.outputs)) destination(s), got $(length(destinations)); " *
        "see $(schema.section)"))
    length(inst.operands) == length(schema.operands) + 1 || throw(ArgumentError(
        "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) requires " *
        "$(length(schema.operands)) source operands, got " *
        "$(max(length(inst.operands) - 1, 0)); see $(schema.section)"))
    sink_count = count(_is_sink_operand, destinations)
    sink_count <= schema.max_sinks || throw(ArgumentError(
        "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) may discard " *
        "at most $(schema.max_sinks) destination(s) with `_`; see " *
        schema.section))
    for (i, (dest, kind)) in enumerate(zip(destinations, schema.outputs))
        if _is_sink_operand(dest)
            schema.sinkable[i] || throw(ArgumentError(
                "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) " *
                "destination $i may not use `_`; see $(schema.section)"))
            continue
        end
        _validate_structured_decl!(cg, dest, kind, "destination $i")
    end
    for (i, (source, kind)) in enumerate(zip(inst.operands[2:end],
                                              schema.operands))
        _validate_structured_source!(cg, source, kind, i)
    end
    (; schema, mods)
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

const _REGISTER_TYPES_8 =
    (ScalarType.B8, ScalarType.U8, ScalarType.S8)
const _REGISTER_TYPES_32 =
    (ScalarType.B32, ScalarType.U32, ScalarType.S32)
const _REGISTER_TYPES_64 =
    (ScalarType.B64, ScalarType.U64, ScalarType.S64)

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
        isempty(mods) && throw(mbarrier_schema_miss(mods))
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
            _mbarrier_is_declared_register(cg, operands[1]) ||
                throw(ArgumentError(
                    "PTX transpiler: mbarrier state destination must be a " *
                    "declared register"))
            push!(destination_operands, operands[1])
            source_start = 2
        elseif schema.destination in (:predicate, :count)
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

_schema_operand_hint(kind::Symbol) = kind === :bf16 ? :b16 : kind

function _render_schema_source(op::Operand, cg::CodeGenState, kind::Symbol)
    render_operand(op, cg; type_hint = _schema_operand_hint(kind))
end

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

const _CVT_IMMEDIATE_DATA_SOURCES =
    Set((:u8, :u16, :u32, :u64, :s8, :s16, :s32, :s64, :f32, :f64))
const _CVT_INTEGER_OPERAND_KINDS =
    Set((:u8, :u16, :u32, :u64, :s8, :s16, :s32, :s64,
         :b8, :b16, :b32, :b64))
const _PTX_EXACT_FLOAT_LITERAL = r"^0[fF][0-9a-fA-F]{8}$|^0[dD][0-9a-fA-F]{16}$"
const _PTX_DECIMAL_FLOAT_LITERAL =
    r"^-?(?:(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+)$"

_is_ptx_float_literal(text::AbstractString) =
    occursin(_PTX_EXACT_FLOAT_LITERAL, text) ||
    occursin(_PTX_DECIMAL_FLOAT_LITERAL, text)

function _render_cvt_source(op::Operand, cg::CodeGenState, kind::Symbol,
                            schema, index::Int)
    # Bare-name PTX registers are LabelOperand lexically. Once a preceding
    # declaration proves the role, render them through the variable path so
    # Julia keyword escaping and name demangling match percent-prefixed regs.
    if op isa LabelOperand &&
            _predefined_immediate_expr(op.name) === nothing &&
            _declared_register(cg, op) !== nothing
        op = RegisterOperand(op.name)
    end

    data_operand = !(schema.scaled && index == length(schema.operands)) &&
                   !(schema.stochastic && index == length(schema.operands))
    predefined = op isa LabelOperand &&
                 _predefined_immediate_expr(op.name) !== nothing
    legacy_warp_size = op isa RegisterOperand &&
                       op.name == IR.LEGACY_WARP_SIZE_SREG
    is_immediate = op isa ImmediateOperand || predefined || legacy_warp_size

    if data_operand && is_immediate &&
            !(schema.source in _CVT_IMMEDIATE_DATA_SOURCES)
        throw(ArgumentError(
            "PTX transpiler: cvt source format .$(schema.source) requires a " *
            "register operand; PTX constants cannot carry that instruction " *
            "type. See $(schema.section) and PTX ISA 9.3 §4.5."))
    end

    if data_operand && schema.source in (:f32, :f64) &&
            is_immediate &&
            !(op isa ImmediateOperand && _is_ptx_float_literal(op.text))
        throw(ArgumentError(
            "PTX transpiler: cvt source format .$(schema.source) requires a " *
            "floating-point constant or register; integer constants do not " *
            "match that PTX instruction type. See $(schema.section) and " *
            "PTX ISA 9.3 §4.5.2."))
    end

    if op isa ImmediateOperand && _is_ptx_float_literal(op.text)
        kind in (:f32, :f64) || throw(ArgumentError(
            "PTX transpiler: floating constant $(op.text) cannot carry the " *
            "reviewed .$kind cvt source role; see $(schema.section) and " *
            "PTX ISA 9.3 §4.5.2."))

        # Exact 0f/0d literals retain their own width in the parser, but PTX
        # converts them to the instruction source type at use. Preserve that
        # contextual conversion instead of letting render_operand's early
        # exact-literal path select the wrong LLVM register class.
        rendered = render_operand(op, cg)
        exact = occursin(_PTX_EXACT_FLOAT_LITERAL, op.text)
        return exact ? string(_schema_operand_hint(kind) === :f32 ? "Float32" :
                              "Float64", "(", rendered, ")") :
                       render_operand(op, cg; type_hint = kind)
    end

    if op isa ImmediateOperand && kind in _CVT_INTEGER_OPERAND_KINDS
        # Evaluate with PTX's fixed s64/u64 expression rules (§4.5.5), then
        # reduce at the operand use site (§4.5.1). Never copy the raw expression
        # into Julia: its literal widths, shifts, and overflow rules differ.
        return _ptx_integer_carrier_expr(op.text, DTYPE_RETTYPE[kind])
    end

    _render_schema_source(op, cg, kind)
end

function _instruction_cvt_source_schema(cg::CodeGenState, inst::Instruction,
                                        scalar_schema)
    inst.opcode == "cvt" || return nothing
    # cvt.pack is already closed by the scalar-result ledger, including exact
    # per-position source carriers. It must never enter the ordinary-cvt path.
    scalar_schema === nothing || return nothing

    mods = _schema_modifiers(inst.modifiers)
    schema = ordinary_cvt_source_schema(mods)
    schema === nothing && throw(ordinary_cvt_source_schema_miss(mods))
    sources = inst.operands[2:end]
    length(sources) == length(schema.operands) || throw(ArgumentError(
        "PTX transpiler: cvt.$(join(mods, ".")) has " *
        "$(length(schema.operands)) reviewed source operand(s), got " *
        "$(length(sources)); see $(schema.section)"))

    if schema.vector_source
        vector = first(sources)
        vector isa VectorOperand && length(vector.elements) == 4 ||
            throw(ArgumentError(
                "PTX transpiler: stochastic cvt to $(schema.destination) " *
                "requires one four-register source vector; see $(schema.section)"))
        for element in vector.elements
            decl = (element isa RegisterOperand || element isa LabelOperand) ?
                   _declared_register(cg, element) : nothing
            decl !== nothing && decl.type in (ScalarType.F32, ScalarType.B32) ||
                throw(ArgumentError(
                    "PTX transpiler: stochastic packed-x4 cvt requires four " *
                    "declared .f32/.b32 source registers; see " *
                    "$(schema.section)"))
        end
    end

    # PTX ISA 9.3 §9.7.9.22 calls rbits a .b32 register operand. Constants
    # have different use-site conversion semantics and are not legal here.
    if schema.stochastic
        rbits = last(sources)
        decl = (rbits isa RegisterOperand || rbits isa LabelOperand) ?
               _declared_register(cg, rbits) : nothing
        decl !== nothing && decl.type in _REGISTER_TYPES_32 ||
            throw(ArgumentError(
                "PTX transpiler: stochastic cvt rbits must be a declared " *
                ".b32/.u32/.s32 register operand; see $(schema.section)"))
    end
    schema
end

function _instruction_clc_try_cancel_schema(inst::Instruction)
    inst.opcode == "clusterlaunchcontrol" || return nothing
    op = :clusterlaunchcontrol
    mods = _schema_modifiers(inst.modifiers)
    schema = clc_try_cancel_schema(mods)
    schema === nothing && requires_clc_try_cancel_schema(op, mods) &&
        throw(clc_try_cancel_schema_miss(mods))
    schema === nothing && return nothing
    length(inst.operands) == 2 || throw(ArgumentError(
        "PTX transpiler: clusterlaunchcontrol.try_cancel has exactly two " *
        "mandatory address operands ([addr], [mbar]); got " *
        "$(length(inst.operands))"))
    for (i, operand) in enumerate(inst.operands)
        operand isa AddressOperand || throw(ArgumentError(
            "PTX transpiler: clusterlaunchcontrol.try_cancel operand $i " *
            "must be bracketed in the input PTX, got $(typeof(operand))"))
        (operand::AddressOperand).coords === nothing || throw(ArgumentError(
            "PTX transpiler: clusterlaunchcontrol.try_cancel operand $i " *
            "must be a scalar PTX address, not a tensor-coordinate address"))
    end
    schema
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
    schema = b128_form_schema(op, mods)
    schema === nothing && requires_b128_form_schema(op, mods) &&
        throw(b128_form_schema_miss(op, mods))
    schema === nothing && return nothing

    operands = inst.operands
    source_start = 2
    destination = nothing
    sources = Operand[]
    if schema.kind === :store
        length(operands) == length(schema.operands) || throw(ArgumentError(
            "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) requires " *
            "$(length(schema.operands)) operands, got $(length(operands)); " *
            "see $(schema.section)"))
        append!(sources, operands)
    else
        length(operands) == length(schema.operands) + 1 || throw(ArgumentError(
            "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) requires " *
            "one destination and $(length(schema.operands)) source " *
            "operand(s), got $(length(operands)); see $(schema.section)"))
        destination = operands[1]
        append!(sources, operands[2:end])
    end

    if schema.kind === :mov
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
        return (; schema, destination, sources, mov_source = source,
                  discard_result = false)
    end

    if schema.kind in (:load, :exch, :cas)
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
    elseif schema.kind === :query_pred
        (destination isa RegisterOperand || destination isa LabelOperand) ||
            throw(ArgumentError("PTX transpiler: query_cancel predicate destination must be a register"))
        _b128_require_register_type(cg, destination, (ScalarType.PRED,),
                                    "query predicate destination")
    elseif schema.kind === :query_dim
        (destination isa RegisterOperand || destination isa LabelOperand) ||
            throw(ArgumentError("PTX transpiler: query_cancel dimension destination must be a register"))
        _b128_require_register_type(cg, destination, _REGISTER_TYPES_32,
                                    "query dimension destination")
    elseif schema.kind === :query_v4
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

    for (i, (kind, source)) in enumerate(zip(schema.operands, sources))
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
    (; schema, destination, sources, mov_source = nothing,
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
    _instruction_clc_try_cancel_schema(inst)

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
    structured_checked === nothing && scalar_schema === nothing &&
        inst.predicate === nothing &&
        _try_alias_def!(cg, inst) && return

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
    elseif scalar_schema === nothing
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
