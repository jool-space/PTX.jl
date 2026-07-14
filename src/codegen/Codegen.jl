module Codegen

using Republic: @public

using ..IR
using ..IR: Module, Function, Instruction, Label, RegDecl, VarDecl, Param,
            FunctionDirective, PragmaDirective, Comment, BlankLine, RawLine,
            Block, Statement, Operand, Predicate,
            RegisterOperand, ImmediateOperand, LabelOperand, VectorOperand,
            AddressOperand, ParenthesizedOperand, NegatedOperand, PipeOperand,
            ScalarType, StateSpace, LinkingDirective, ptx
using ..Parser: parse as parse_ptx, tokenize, TokenKind, LexError
using ..PTX: uses_implicit_cc, structured_result_schema,
             requires_structured_result_schema, structured_result_schema_miss,
             immediate_form_contract, requires_immediate_form_contract,
             immediate_form_contract_miss, validate_immediate_value,
             scalar_result_schema,
             requires_scalar_result_schema, scalar_result_schema_miss,
             vector_result_schema, requires_vector_result_schema,
             vector_result_schema_miss, validate_vector_result_mask,
             vector_result_operand_roles,
             B128, b128_form_schema, requires_b128_form_schema,
             b128_form_schema_miss, validate_b128_form_args,
             infer_rettype, ordinary_cvt_source_schema,
             ordinary_cvt_source_schema_miss,
             mbarrier_form_schema, mbarrier_schema_miss, DTYPE_RETTYPE,
             clc_try_cancel_schema,
             requires_clc_try_cancel_schema, clc_try_cancel_schema_miss

include("state.jl")
include("operands.jl")
include("constants.jl")
include("instruction.jl")
include("statements.jl")
include("function.jl")

@public ptx_to_julia, ir_to_julia

"""
    ptx_to_julia(source) -> String

Parse PTX source text and emit Julia source code. Returns a string of one
or more `function ... end` definitions calling the `ptx"..."` macro.
"""
ptx_to_julia(source::AbstractString) = ir_to_julia(parse_ptx(source))

"""
    ir_to_julia(mod::IR.Module) -> String

Convert a parsed `IR.Module` into Julia source code. Each `Function` in
the module becomes one Julia function definition.
"""
function ir_to_julia(mod::Module)
    cg = CodeGenState()
    arch    = isempty(mod.target.targets) ? "" : mod.target.targets[1]
    version = (mod.version.major, mod.version.minor)
    funcs = [d for d in mod.directives if d isa Function]
    for (i, f) in enumerate(funcs)
        i == 1 || blank!(cg)
        emit_function!(cg, f, arch, version)
    end
    String(take!(cg.io))
end

end # module Codegen
