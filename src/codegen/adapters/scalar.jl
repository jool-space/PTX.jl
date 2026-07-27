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
