function _instruction_immediate_form_contract(inst::Instruction)
    op = Symbol(inst.opcode)
    requires_immediate_form_contract(op) || return nothing
    mods = _schema_modifiers(inst.modifiers)
    contract = immediate_form_contract(op, mods)
    contract === nothing && throw(immediate_form_contract_miss(op, mods))
    length(inst.operands) == 1 || throw(ArgumentError(
        "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) requires " *
        "exactly one immediate source operand, got $(length(inst.operands)); " *
        "see $(contract.section)"))
    source = only(inst.operands)
    text = try
        _ptx_integer_constant_text(source)
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError(
            "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) source " *
            "must be an integer constant: $(sprint(showerror, err)); see " *
            contract.section))
    end
    value = try
        _ptx_integer_constant(text)
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError(
            "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) has an " *
            "invalid integer constant $(repr(text)): $(sprint(showerror, err))"))
    end
    validate_immediate_value(
        contract, value;
        context = "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) source")
    (; contract, value = Int(value))
end
