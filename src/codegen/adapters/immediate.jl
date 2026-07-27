function _instruction_immediate_form_contract(inst::Instruction)
    op = Symbol(inst.opcode)
    mods = _schema_modifiers(inst.modifiers)
    claims(ImmediateLedger(), op, mods) || return nothing
    contract = schema(ImmediateLedger(), op, mods)
    contract === nothing && throw(miss(ImmediateLedger(), op, mods))
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

# Validate constant-only, destinationless instructions before generic
# destination inference or pointer-alias absorption can reinterpret their
# sole source as a definition. Constant expressions are reduced to Val so
# reconstructed Julia retains the compile-time ISA contract.
function transpile_ledger!(::ImmediateLedger, cg::CodeGenState,
                           inst::Instruction)
    checked = _instruction_immediate_form_contract(inst)
    checked === nothing && return false
    chain = chain_expr(cg, inst.opcode, inst.modifiers)
    call = chain * "(Val($(checked.value)))"
    emit_with_predicate!(cg, call, inst.predicate, String[])
    true
end
