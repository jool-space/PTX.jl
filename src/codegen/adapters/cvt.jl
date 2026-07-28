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

    data_operand = !(schema.scaled !== :none &&
                     index == length(schema.operands)) &&
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
    s = schema(CvtLedger(), :cvt, mods)
    s === nothing && throw(miss(CvtLedger(), :cvt, mods))
    sources = inst.operands[2:end]
    length(sources) == length(s.operands) || throw(ArgumentError(
        "PTX transpiler: cvt.$(join(mods, ".")) has " *
        "$(length(s.operands)) reviewed source operand(s), got " *
        "$(length(sources)); see $(s.section)"))

    if s.vector_source
        vector = first(sources)
        vector isa VectorOperand && length(vector.elements) == 4 ||
            throw(ArgumentError(
                "PTX transpiler: stochastic cvt to $(s.destination) " *
                "requires one four-register source vector; see $(s.section)"))
        for element in vector.elements
            decl = (element isa RegisterOperand || element isa LabelOperand) ?
                   _declared_register(cg, element) : nothing
            decl !== nothing && decl.type in (ScalarType.F32, ScalarType.B32) ||
                throw(ArgumentError(
                    "PTX transpiler: stochastic packed-x4 cvt requires four " *
                    "declared .f32/.b32 source registers; see " *
                    "$(s.section)"))
        end
    end

    # PTX ISA 9.3 §9.7.9.22 calls rbits a .b32 register operand. Constants
    # have different use-site conversion semantics and are not legal here.
    if s.stochastic
        rbits = last(sources)
        decl = (rbits isa RegisterOperand || rbits isa LabelOperand) ?
               _declared_register(cg, rbits) : nothing
        decl !== nothing && decl.type in _REGISTER_TYPES_32 ||
            throw(ArgumentError(
                "PTX transpiler: stochastic cvt rbits must be a declared " *
                ".b32/.u32/.s32 register operand; see $(s.section)"))
    end
    s
end

# Ordinary cvt sits outside the island partition (see protocol.jl):
# emit_instruction!
# consults it explicitly after the shared walk, ret/bra, and alias absorption.
# cvt.pack never reaches this method — the scalar ledger claims and handles it
# inside the shared walk.
function transpile_ledger!(::CvtLedger, cg::CodeGenState, inst::Instruction)
    s = _instruction_cvt_source_schema(cg, inst, nothing)
    s === nothing && return false
    dst_expr, dst_names = render_dst(inst.operands[1], cg)
    src_strs = [_render_cvt_source(op, cg, kind, s, index)
                for (index, (op, kind)) in
                    enumerate(zip(inst.operands[2:end], s.operands))]
    _emit_schema_call!(cg, inst, inst.modifiers, dst_expr, dst_names, src_strs)
    true
end
