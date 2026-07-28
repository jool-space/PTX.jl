# Closed PTX 9.3 contracts for instructions whose source operand is required
# to be an integer constant in an instruction-specific domain — the immediate
# island is exactly setmaxnreg/pmevent. lop3's immLut (also an ISA-required
# integer constant, 0:255) is NOT recorded here: it lives as the `:imm8`
# operand kind inside the structured island's three lop3 schemas, which own
# lop3's grouped ABI and validate the constant through `operand_accepts`.

struct ImmediateFormContract
    op::Symbol
    forms::Tuple{Vararg{Tuple{Vararg{Symbol}}}}
    operand_index::Int
    minimum::Int
    maximum::Int
    multiple::Int
    returns::Bool
    side_effect::Bool
    ptx_version::VersionNumber
    min_sm::Union{Nothing,VersionNumber}
    feature_set::Symbol
    target_note::String
    section::String
end

const _SETMAXNREG_SECTION =
    "ptx/9-instruction-set/9.7.21.5-miscellaneous-instructions-setmaxnreg.md"
const _PMEVENT_SECTION =
    "ptx/9-instruction-set/9.7.21.3-miscellaneous-instructions-pmevent.md"

const IMMEDIATE_FORM_CONTRACTS = ImmediateFormContract[
    ImmediateFormContract(
        :setmaxnreg, ((:inc, :sync, :aligned, :u32),), 1,
        24, 256, 8, false, true, v"8.0", v"9.0", :arch,
        "sm_90a/sm_100a/sm_110a/sm_120a; PTX 8.8 also admits " *
        "sm_100f/sm_110f/sm_120f and later targets in the same family",
        _SETMAXNREG_SECTION),
    ImmediateFormContract(
        :setmaxnreg, ((:dec, :sync, :aligned, :u32),), 1,
        24, 256, 8, false, true, v"8.0", v"9.0", :arch,
        "sm_90a/sm_100a/sm_110a/sm_120a; PTX 8.8 also admits " *
        "sm_100f/sm_110f/sm_120f and later targets in the same family",
        _SETMAXNREG_SECTION),
    ImmediateFormContract(
        :pmevent, ((),), 1, 0, 15, 1, false, true,
        v"1.4", nothing, :baseline, "all targets", _PMEVENT_SECTION),
    ImmediateFormContract(
        :pmevent, ((:mask,),), 1, 0, 0xffff, 1, false, true,
        v"3.0", v"2.0", :baseline, "sm_20 or higher", _PMEVENT_SECTION),
]

length(IMMEDIATE_FORM_CONTRACTS) == 4 ||
    error("immediate-form contract count changed; re-audit PTX 9.3 grammar")

const _IMMEDIATE_FORM_CONTRACT_BY_FORM = let by_form = Dict()
    for contract in IMMEDIATE_FORM_CONTRACTS, mods in contract.forms
        key = (contract.op, mods)
        haskey(by_form, key) && error("duplicate immediate-form contract: $key")
        by_form[key] = contract
    end
    by_form
end

schema(::ImmediateLedger, op::Symbol, mods::Tuple{Vararg{Symbol}}) =
    get(_IMMEDIATE_FORM_CONTRACT_BY_FORM, (op, mods), nothing)

function miss(::ImmediateLedger, op::Symbol,
              mods::Tuple{Vararg{Symbol}})
    spelling = isempty(mods) ? string(op) : string(op, ".", join(mods, "."))
    ArgumentError(
        "ptx\"$spelling\" is inside the closed PTX 9.3 immediate grammar " *
        "for setmaxnreg/pmevent but does not match a reviewed form. " *
        "The raw tier cannot make an invalid immediate spelling safe.")
end

function validate_immediate_value(contract::ImmediateFormContract, value;
                                  context::AbstractString = "immediate operand")
    value isa Integer && !(value isa Bool) || throw(ArgumentError(
        "$context must be an integer constant, got $(repr(value))"))
    contract.minimum <= value <= contract.maximum || throw(ArgumentError(
        "$context must be in $(contract.minimum):$(contract.maximum), " *
        "got $value; see $(contract.section)"))
    value % contract.multiple == 0 || throw(ArgumentError(
        "$context must be a multiple of $(contract.multiple), got $value; " *
        "see $(contract.section)"))
    value
end

function validate_immediate_form_args(contract::ImmediateFormContract,
                                      argtypes)
    length(argtypes) == 1 || throw(ArgumentError(
        "ptx immediate form requires exactly one source operand, got " *
        "$(length(argtypes)); see $(contract.section)"))
    T = argtypes[contract.operand_index]
    T <: Val || throw(ArgumentError(
        "ptx immediate source must be Val{N}, got $T; see $(contract.section)"))
    is_integer_immediate(T) || throw(ArgumentError(
        "ptx immediate source must be Val{N} with non-Bool integer N, got $T; " *
        "see $(contract.section)"))
    validate_immediate_value(contract, T.parameters[1])
    nothing
end

validate_ledger_args(::ImmediateLedger, c::ImmediateFormContract, argtypes) =
    validate_immediate_form_args(c, argtypes)
