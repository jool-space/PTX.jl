# `result_boundary` controls the miss-side result-ABI check: the module
# preflight (contract.jl) keeps it here so noncanonical cvt and void-inferring
# pure forms fail at their historical position, while emit_instruction! runs
# the same check itself after island routing (a valid `.b128` spelling has no
# DTYPE_RETTYPE entry and must not trip it).
function _instruction_scalar_result_schema(inst::Instruction;
                                           result_boundary::Bool = true)
    op = Symbol(inst.opcode)
    mods = _schema_modifiers(inst.modifiers)
    s = schema(ScalarLedger(), op, mods)
    s === nothing && island_of(op, mods) isa ScalarLedger &&
        throw(miss(ScalarLedger(), op, mods))
    if s === nothing
        # Keep the parser/transpiler on the same result-ABI boundary as direct
        # calls. This catches noncanonical cvt and any reviewed pure form that
        # would otherwise infer void before pointer-alias absorption can erase
        # a malformed mov/add/sub definition.
        result_boundary && infer_rettype(op, mods)
        return nothing
    end
    expected = length(s.operands) + 1 # explicit destination + sources
    length(inst.operands) == expected || throw(ArgumentError(
        "PTX transpiler: $(inst.opcode)$(join(inst.modifiers)) is an audited " *
        "scalar form with $(length(s.operands)) source operands, got " *
        "$(max(length(inst.operands) - 1, 0)); see $(s.section)"))
    s
end

function transpile_ledger!(::ScalarLedger, cg::CodeGenState,
                           inst::Instruction)
    s = _instruction_scalar_result_schema(inst; result_boundary = false)
    s === nothing && return false
    dst_expr, dst_names = render_dst(inst.operands[1], cg)
    src_strs = [_render_schema_source(op, cg, kind)
                for (op, kind) in zip(inst.operands[2:end], s.operands)]
    _emit_schema_call!(cg, inst, inst.modifiers, dst_expr, dst_names, src_strs)
    true
end
