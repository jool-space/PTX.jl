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
