module Codegen

using Republic: @public

using ..IR
using ..IR: Module, Function, Instruction, Label, RegDecl, VarDecl, Param,
            PragmaDirective, Comment, BlankLine, RawLine,
            Block, IntrinsicScope, TargetDirective, Statement, Operand, Predicate,
            RegisterOperand, ImmediateOperand, LabelOperand, VectorOperand,
            AddressOperand, ParenthesizedOperand, NegatedOperand, PipeOperand,
            ScalarType, StateSpace, LinkingDirective, ptx
using ..Parser: parse as parse_ptx, tokenize, TokenKind, LexError
using ..PTX: uses_implicit_cc, claims, schema, miss, CALL_LEDGERS,
             ImmediateLedger, MBarrierLedger, StructuredLedger,
             CLCLedger, VectorLedger, ScalarLedger, B128Ledger, CvtLedger,
             validate_immediate_value, validate_vector_result_mask,
             vector_result_operand_roles,
             form_contract, requires_typed_wrapper,
             infer_rettype, DTYPE_RETTYPE

include("state.jl")
include("operands.jl")
include("constants.jl")
include("instruction.jl")
include("adapters/scalar.jl")
include("adapters/immediate.jl")
include("adapters/structured.jl")
include("adapters/vector.jl")
include("adapters/mbarrier.jl")
include("adapters/cvt.jl")
include("adapters/clusterlaunchcontrol.jl")
include("adapters/b128.jl")
include("statements.jl")
include("function.jl")
include("contract.jl")

@public ptx_to_julia, ir_to_julia, TranspilerError

"""
    ptx_to_julia(source) -> String

Parse PTX source text and emit Julia source code. Returns a string of one
or more `function ... end` definitions calling the `ptx"..."` macro after
validating the complete module against the transpiler's deliberately closed
subset. Lexical, parse, and transpiler-contract errors throw instead of
returning a partial program. Acceptance is a lowering-contract guarantee, not
a proof of semantic equivalence for every accepted PTX program.
"""
ptx_to_julia(source::AbstractString) = ir_to_julia(parse_ptx(source))

"""
    ir_to_julia(mod::IR.Module) -> String

Validate a parsed `IR.Module` and convert it into Julia source code. Each
accepted `Function` becomes one Julia function definition. Validation covers
the complete module and finishes before emission, so an unsupported node,
declaration, control-flow edge, or instruction form raises
`TranspilerError` without returning partial Julia.
"""
function ir_to_julia(mod::Module)
    validate_transpilable(mod)
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
