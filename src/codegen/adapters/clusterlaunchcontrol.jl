function _instruction_clc_try_cancel_schema(inst::Instruction)
    inst.opcode == "clusterlaunchcontrol" || return nothing
    op = :clusterlaunchcontrol
    mods = _schema_modifiers(inst.modifiers)
    s = schema(CLCLedger(), op, mods)
    s === nothing && island_of(op, mods) isa CLCLedger &&
        throw(miss(CLCLedger(), op, mods))
    s === nothing && return nothing
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
    s
end

function transpile_ledger!(::CLCLedger, cg::CodeGenState, inst::Instruction)
    _instruction_clc_try_cancel_schema(inst) === nothing && return false
    chain = chain_expr(cg, inst.opcode, inst.modifiers)
    args = join((render_operand(op, cg) for op in inst.operands), ", ")
    emit_with_predicate!(cg, chain * "(" * args * ")",
                         inst.predicate, String[])
    true
end
