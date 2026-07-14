# Closed semantic boundary for IR -> Julia transpilation.
#
# Parsing and formatting intentionally retain more PTX than the Julia emitter
# can represent.  This visitor is the single place where that broader IR is
# narrowed: every accepted node and generic operand role is explicit, while a
# future node reaches the fallback error instead of disappearing from output.

# PTX ISA 9.3 §5.2.1/Table 8 permits fundamental types in declarations.
# Alternate formats use their bit-size carrier instead: bf16 -> b16,
# bf16x2/tf32 -> b32 (§§5.2.3, 5.2.5.1).
const _TRANSPILE_PARAM_TYPES = Set((
    ScalarType.B8, ScalarType.B16, ScalarType.B32, ScalarType.B64,
    ScalarType.U8, ScalarType.U16, ScalarType.U32, ScalarType.U64,
    ScalarType.S8, ScalarType.S16, ScalarType.S32, ScalarType.S64,
    ScalarType.F16, ScalarType.F32, ScalarType.F64,
))

const _TRANSPILE_REG_TYPES = Set((
    _TRANSPILE_PARAM_TYPES...,
    ScalarType.B128, ScalarType.F16X2, ScalarType.PRED,
))

# CuStaticSharedArray preserves element representation and extent for these
# scalar carriers.  Explicit alignment, initialization, linking, cluster
# scope, and packed/alternate element formats remain declaration coverage
# gaps and are rejected below.
const _TRANSPILE_SHARED_TYPES = Set((
    ScalarType.B8, ScalarType.B16, ScalarType.B32, ScalarType.B64,
    ScalarType.U8, ScalarType.U16, ScalarType.U32, ScalarType.U64,
    ScalarType.S8, ScalarType.S16, ScalarType.S32, ScalarType.S64,
    ScalarType.F16, ScalarType.F32, ScalarType.F64,
))

# FormContract is an optimizer contract, not an operand grammar: it does not
# prove arity, carriers, or which operand is an address.  Generic transpilation
# therefore uses a separate finite ledger keyed by the complete PTX spelling.
# Each rule pins a destination carrier (`nothing` for a sink) and every legal
# source-role sequence.  Exact ABI islands remain independently validated.
struct _TranspileFormRule
    destination::Union{Symbol, Nothing}
    sources::Tuple{Vararg{Tuple{Vararg{Symbol}}}}
    section::String
end

_transpile_result(destination::Symbol, sources::Tuple{Vararg{Symbol}},
                  section::String) =
    _TranspileFormRule(destination, (sources,), section)
_transpile_sink(sources::Tuple{Vararg{Symbol}}, section::String) =
    _TranspileFormRule(nothing, (sources,), section)

const _TRANSPILE_GENERIC_FORMS = Dict{Tuple{String, Tuple{Vararg{String}}},
                                      _TranspileFormRule}(
    # Scalar moves (PTX ISA 9.3 §9.7.9.3).
    (("mov", (".u32",)) => _transpile_result(:u32, (:u32,), "§9.7.9.3")),
    (("mov", (".u64",)) => _TranspileFormRule(
        :u64, ((:u64_or_shared_symbol,),), "§9.7.9.3")),
    (("mov", (".b32",)) => _transpile_result(:b32, (:b32,), "§9.7.9.3")),
    (("mov", (".b64",)) => _transpile_result(:b64, (:b64,), "§9.7.9.3")),
    (("mov", (".s32",)) => _transpile_result(:s32, (:s32,), "§9.7.9.3")),
    (("mov", (".f32",)) => _transpile_result(:f32, (:f32,), "§9.7.9.3")),
    (("mov", (".pred",)) => _transpile_result(:pred, (:pred,), "§9.7.9.3")),

    # Integer add/sub (§§9.7.1.1–2) and floating add/sub (§§9.7.3.3–4).
    (("add", (".u32",)) => _transpile_result(:u32, (:u32, :u32), "§9.7.1.1")),
    (("add", (".u64",)) => _transpile_result(:u64, (:u64, :u64), "§9.7.1.1")),
    (("add", (".s32",)) => _transpile_result(:s32, (:s32, :s32), "§9.7.1.1")),
    (("add", (".f32",)) => _transpile_result(:f32, (:f32, :f32), "§9.7.3.3")),
    (("sub", (".u32",)) => _transpile_result(:u32, (:u32, :u32), "§9.7.1.2")),
    (("sub", (".u64",)) => _transpile_result(:u64, (:u64, :u64), "§9.7.1.2")),
    (("sub", (".s32",)) => _transpile_result(:s32, (:s32, :s32), "§9.7.1.2")),
    (("sub", (".f32",)) => _transpile_result(:f32, (:f32, :f32), "§9.7.3.4")),

    # Integer multiply-add (§9.7.1.4) and shifts (§9.7.8.8).  The shl
    # shift-count operand is explicitly an unsigned 32-bit quantity.
    (("mad", (".lo", ".s32")) =>
        _transpile_result(:s32, (:s32, :s32, :s32), "§9.7.1.4")),
    (("mad", (".lo", ".u32")) =>
        _transpile_result(:u32, (:u32, :u32, :u32), "§9.7.1.4")),
    (("shl", (".b32",)) => _transpile_result(:b32, (:b32, :u32), "§9.7.8.8")),
    (("shl", (".b64",)) => _transpile_result(:b64, (:b64, :u32), "§9.7.8.8")),

    # Scalar parameter/global/shared memory subset (load §9.7.9.8,
    # store §9.7.9.11).
    (("ld", (".param", ".u32")) =>
        _transpile_result(:u32, (:param_u32_address,), "§9.7.9.8")),
    (("ld", (".param", ".u64")) =>
        _transpile_result(:u64, (:param_u64_address,), "§9.7.9.8")),
    (("ld", (".global", ".f32")) =>
        _transpile_result(:f32, (:global_address,), "§9.7.9.8")),
    (("ld", (".global", ".u32")) =>
        _transpile_result(:u32, (:global_address,), "§9.7.9.8")),
    (("ld", (".global", ".b32")) =>
        _transpile_result(:b32, (:global_address,), "§9.7.9.8")),
    (("ld", (".global", ".b64")) =>
        _transpile_result(:b64, (:global_address,), "§9.7.9.8")),
    (("ld", (".shared", ".b32")) =>
        _transpile_result(:b32, (:shared_address,), "§9.7.9.8")),
    (("st", (".global", ".f32")) =>
        _transpile_sink((:global_address, :f32), "§9.7.9.11")),
    (("st", (".global", ".b32")) =>
        _transpile_sink((:global_address, :b32), "§9.7.9.11")),
    (("st", (".global", ".b64")) =>
        _transpile_sink((:global_address, :b64), "§9.7.9.11")),
    (("st", (".shared", ".b32")) =>
        _transpile_sink((:shared_address, :b32), "§9.7.9.11")),

    # Exact no-result synchronization/control spellings (§§9.7.14.1–2).
    (("bar", (".sync",)) => _TranspileFormRule(
        nothing, ((:u32,), (:u32, :u32)), "§9.7.14.1")),
    (("bar", (".warp", ".sync")) =>
        _transpile_sink((:u32,), "§9.7.14.2")),
    (("membar", (".cta",)) => _transpile_sink((), "§9.7.14.4")),
    (("membar", (".gl",)) => _transpile_sink((), "§9.7.14.4")),
    (("membar", (".sys",)) => _transpile_sink((), "§9.7.14.4")),
    (("exit", ()) => _transpile_sink((), "§9.7.13.7")),
    (("trap", ()) => _transpile_sink((), "§9.7.20.4")),
    (("brkpt", ()) => _transpile_sink((), "§9.7.20.1")),
)

const _TRANSPILE_INTEGER_LITERAL =
    r"^-?(?:0[xX][0-9a-fA-F]+|0[bB][01]+|0[0-7]+|[0-9]+)[uU]?$"
const _TRANSPILE_EXACT_FLOAT_LITERAL =
    r"^0[fF][0-9a-fA-F]{8}$|^0[dD][0-9a-fA-F]{16}$"
const _TRANSPILE_DECIMAL_FLOAT_LITERAL =
    r"^-?(?:(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+)$"

mutable struct _TranspileContractState
    cg::CodeGenState
    params::Dict{String, Param}
    shared_symbols::Set{String}
    julia_bindings::Dict{String, Tuple{Symbol, String}}
    pointer_alias_regs::Set{String}
    labels::Dict{String, String}
    julia_labels::Dict{String, String}
    branches::Vector{Tuple{String, String, String}}
end

_TranspileContractState() = _TranspileContractState(
    CodeGenState(), Dict{String, Param}(), Set{String}(),
    Dict{String, Tuple{Symbol, String}}(),
    Set{String}(),
    Dict{String, String}(), Dict{String, String}(),
    Tuple{String, String, String}[],
)

function _counted_name_contains(base::AbstractString, count::Int,
                                name::AbstractString)
    startswith(name, base) || return false
    first_suffix = nextind(name, lastindex(base))
    first_suffix <= lastindex(name) || return false
    index = tryparse(BigInt, SubString(name, first_suffix))
    index !== nothing && 0 <= index < count
end

function _counted_names_overlap(base_a::AbstractString, count_a::Int,
                                base_b::AbstractString, count_b::Int)
    base_a == base_b && return true
    function prefixed_overlap(short, short_count, long, long_count)
        startswith(long, short) || return false
        first_suffix = nextind(long, lastindex(short))
        first_suffix <= lastindex(long) || return false
        suffix = SubString(long, first_suffix)
        all(isdigit, suffix) || return false
        first(suffix) == '0' && return false
        prefix = tryparse(BigInt, suffix)
        prefix !== nothing && long_count > 0 && prefix * 10 < short_count
    end
    prefixed_overlap(base_a, count_a, base_b, count_b) ||
        prefixed_overlap(base_b, count_b, base_a, count_a)
end

function _reg_declarations_collide(a::RegDecl, b::RegDecl,
                                   transform = identity)
    name_a, name_b = transform(a.name), transform(b.name)
    if a.count === nothing
        return b.count === nothing ? name_a == name_b :
               _counted_name_contains(name_b, b.count, name_a)
    elseif b.count === nothing
        return _counted_name_contains(name_a, a.count, name_b)
    end
    _counted_names_overlap(name_a, a.count, name_b, b.count)
end

function _reg_decl_contains_binding(decl::RegDecl, binding::AbstractString)
    base = julia_var(decl.name)
    decl.count === nothing ? base == binding :
        _counted_name_contains(base, decl.count, binding)
end

function _check_reg_mangling_shape(decl::RegDecl, path::String)
    decl.count === nothing && return
    base = julia_var(decl.name)
    julia_var(decl.name * "0") == base * "0" || _transpile_reject(path,
        "counted register name $(repr(decl.name)) has non-affine Julia mangling")
end

function _claim_binding!(state::_TranspileContractState, raw::String,
                         kind::Symbol, path::String)
    jname = julia_var(raw)
    existing = get(state.julia_bindings, jname, nothing)
    existing === nothing || _transpile_reject(path,
        "$kind name $(repr(raw)) collides after Julia mangling with " *
        "$(existing[1]) $(repr(existing[2]))")
    for declaration in values(state.cg.reg_decls)
        _reg_decl_contains_binding(declaration, jname) && _transpile_reject(path,
            "$kind name $(repr(raw)) collides after Julia mangling with " *
            "register declaration $(repr(declaration.name))")
    end
    state.julia_bindings[jname] = (kind, raw)
end

"""
    TranspilerError(path, category, detail)

Raised before PTX-to-Julia emission when parsed IR falls outside the closed
semantic subset. `path` identifies the rejected IR node, `category` separates
unsupported structure/operands from reviewed schema misses, and `detail`
explains the missing contract. This is not evidence that the input PTX is
invalid; parsing and lossless formatting intentionally support a broader IR.
"""
struct TranspilerError <: Exception
    path::String
    category::Symbol
    detail::String
end

function Base.showerror(io::IO, err::TranspilerError)
    print(io, "PTX transpiler contract [", err.category, "] at ",
          err.path, ": ", err.detail)
end

function _transpile_reject(path::AbstractString, detail::AbstractString;
                           category::Symbol = :unsupported)
    throw(TranspilerError(String(path), category, String(detail)))
end

_stmt_path(parent::AbstractString, i::Integer, s::Statement) =
    "$parent[$i]::$(nameof(typeof(s)))"

function _validate_nonsemantic_statement(s::Statement, path::String)
    (s isa Comment || s isa BlankLine) && return
    _transpile_reject(path,
        "module-level $(nameof(typeof(s))) is not in the supported subset; " *
        "it would otherwise be silently omitted")
end

function _validate_param!(state::_TranspileContractState, p::Param,
                          path::String, is_entry::Bool)
    p.state_space in (StateSpace.PARAM, StateSpace.REG) ||
        _transpile_reject(path, "parameter state space $(ptx(p.state_space)) is unsupported")
    is_entry && p.state_space !== StateSpace.PARAM &&
        _transpile_reject(path, ".entry parameters must use .param state space")
    p.type in _TRANSPILE_PARAM_TYPES ||
        _transpile_reject(path, "parameter type $(ptx(p.type)) has no reviewed Julia carrier")
    p.array_size === nothing ||
        _transpile_reject(path, "array parameters are not represented by one Julia argument")
    p.alignment === nothing ||
        _transpile_reject(path, "parameter .align is not preserved by the emitted Julia signature")
    p.ptr_state_space === nothing || _transpile_reject(path,
        "parameter .ptr state-space metadata is emitted only as a comment")
    p.ptr_alignment === nothing || _transpile_reject(path,
        "parameter .ptr .align metadata is not preserved by the emitted Julia signature")
    haskey(state.params, p.name) &&
        _transpile_reject(path, "duplicate parameter name $(repr(p.name))")
    jname = julia_var(p.name)
    any(q -> julia_var(q) == jname, keys(state.params)) &&
        _transpile_reject(path, "parameter name collides after Julia mangling: $jname")
    _claim_binding!(state, p.name, :parameter, path)
    state.params[p.name] = p
end

function _collect_labels!(state::_TranspileContractState,
                          body::Tuple{Vararg{Statement}}, scope::String,
                          path::String)
    for (i, stmt) in enumerate(body)
        spath = _stmt_path(path, i, stmt)
        if stmt isa Label
            haskey(state.labels, stmt.name) &&
                _transpile_reject(spath, "duplicate PTX label $(repr(stmt.name))")
            jlabel = julia_label(stmt.name)
            haskey(state.julia_labels, jlabel) &&
                _transpile_reject(spath,
                    "label collides after Julia mangling with $(repr(state.julia_labels[jlabel]))")
            state.labels[stmt.name] = scope
            state.julia_labels[jlabel] = stmt.name
        elseif stmt isa Block
            _collect_labels!(state, stmt.body, spath, spath * ".body")
        end
    end
end

function _validate_predicate!(state::_TranspileContractState,
                              pred::Union{Predicate, Nothing}, path::String)
    pred === nothing && return
    decl = _declared_register(state.cg, pred.register)
    decl === nothing &&
        _transpile_reject(path, "predicate register $(pred.register) has no preceding .reg declaration")
    decl.type === ScalarType.PRED ||
        _transpile_reject(path, "predicate register $(pred.register) is $(ptx(decl.type)), not .pred")
end

function _validate_register_source!(state::_TranspileContractState,
                                    op::RegisterOperand, path::String)
    op.name == IR.LEGACY_WARP_SIZE_SREG && return
    op.name in SPECIAL_REGS && return
    op.name in IR.V4_SPECIAL_REG_ROOTS &&
        _transpile_reject(path, "vector-valued special-register roots require component selection")
    _declared_register(state.cg, op) === nothing &&
        _transpile_reject(path, "register $(op.name) has no preceding in-scope .reg declaration")
end

_is_transpile_literal(text::AbstractString) =
    occursin(_TRANSPILE_INTEGER_LITERAL, text) ||
    occursin(_TRANSPILE_EXACT_FLOAT_LITERAL, text) ||
    occursin(_TRANSPILE_DECIMAL_FLOAT_LITERAL, text)

const _TRANSPILE_INTEGER_SOURCE_ROLES = Set((
    :b8, :b16, :b32, :b64,
    :u8, :u16, :u32, :u64,
    :s8, :s16, :s32, :s64,
))
const _TRANSPILE_FLOAT_SOURCE_ROLES = Set((:f16, :f32, :f64))

function _validate_immediate_source!(op::ImmediateOperand, path::String,
                                     role::Symbol)
    _is_transpile_literal(op.text) || _transpile_reject(path,
        "opaque constant expression $(repr(op.text)) is outside the reviewed " *
        "literal-only generic operand subset")
    role in _TRANSPILE_INTEGER_SOURCE_ROLES &&
        !occursin(_TRANSPILE_INTEGER_LITERAL, op.text) && _transpile_reject(path,
            "floating constant $(repr(op.text)) cannot carry reviewed .$role " *
            "integer/bit operand semantics; see PTX ISA 9.3 §§4.5 and 6.1")
    role in _TRANSPILE_FLOAT_SOURCE_ROLES &&
        !occursin(_TRANSPILE_EXACT_FLOAT_LITERAL, op.text) &&
        !occursin(_TRANSPILE_DECIMAL_FLOAT_LITERAL, op.text) &&
        _transpile_reject(path,
            "integer constant $(repr(op.text)) cannot carry reviewed .$role " *
            "floating operand semantics; see PTX ISA 9.3 §§4.5 and 6.1")
    if role === :pred
        occursin(_TRANSPILE_INTEGER_LITERAL, op.text) || _transpile_reject(path,
            "predicate constants must use the PTX integer constant domain")
        _ptx_integer_constant(op.text)
    end
end

function _validate_address!(state::_TranspileContractState,
                            op::AddressOperand, path::String,
                            role::Symbol = :generic_address)
    op.coords === nothing || _transpile_reject(path,
        "tensor-coordinate addresses remain outside the generic operand subset")
    decl = _declared_register(state.cg, op.base)
    if decl !== nothing
        decl.type in (_REGISTER_TYPES_32..., _REGISTER_TYPES_64...) ||
            _transpile_reject(path,
                "address base $(op.base) must use a 32-/64-bit integer or bit register")
    elseif startswith(op.base, "%")
        _transpile_reject(path, "address base $(op.base) has no preceding .reg declaration")
    elseif role in (:param_u32_address, :param_u64_address)
        param = get(state.params, op.base, nothing)
        param === nothing && _transpile_reject(path,
            "parameter address base $(repr(op.base)) has no matching parameter")
        expected = role === :param_u32_address ? ScalarType.U32 : ScalarType.U64
        param.type === expected || _transpile_reject(path,
            "parameter $(repr(op.base)) is $(ptx(param.type)), expected $(ptx(expected))")
        op.offset === nothing || _transpile_reject(path,
            "only a simple parameter address is supported")
    elseif role === :shared_address
        op.base in state.shared_symbols || _transpile_reject(path,
            "shared address base $(repr(op.base)) has no in-scope .shared declaration")
    elseif role in (:global_address, :generic_address)
        _transpile_reject(path,
            "symbolic $(role === :global_address ? "global" : "generic") address " *
            "$(repr(op.base)) has no represented storage declaration")
    else
        error("unknown transpiler address role $role")
    end
    op.offset === nothing || occursin(_TRANSPILE_INTEGER_LITERAL, op.offset) ||
        _transpile_reject(path,
            "non-literal address offset $(repr(op.offset)) remains FRONT-OPERAND scope")
end

function _validate_generic_destination!(state::_TranspileContractState,
                                        op::Operand, kind::Symbol, path::String)
    (op isa RegisterOperand || op isa LabelOperand) || _transpile_reject(path,
        "generic result destinations must be declared scalar registers, not $(nameof(typeof(op)))")
    decl = _declared_register(state.cg, op)
    decl === nothing &&
        _transpile_reject(path, "destination register $(op.name) has no preceding .reg declaration")
    accepted = _structured_decl_types(kind)
    decl.type in accepted || _transpile_reject(path,
        "destination $(op.name) is $(ptx(decl.type)), expected " *
        join(ptx.(accepted), "/"))
end

function _validate_typed_source!(state::_TranspileContractState, op::Operand,
                                 kind::Symbol, path::String)
    if kind === :u64_or_shared_symbol
        if op isa LabelOperand && op.name in state.shared_symbols
            return
        end
        return _validate_typed_source!(state, op, :u64, path)
    elseif kind in (:generic_address, :global_address, :shared_address,
                    :param_u32_address, :param_u64_address)
        op isa AddressOperand ||
            _transpile_reject(path, "role $kind requires an explicit bracketed address")
        _validate_address!(state, op, path, kind)
        return
    elseif kind === :shared_symbol
        op isa LabelOperand && op.name in state.shared_symbols ||
            _transpile_reject(path, "role .shared_symbol requires a preceding .shared declaration")
        return
    elseif op isa ImmediateOperand
        return _validate_immediate_source!(op, path, kind)
    elseif op isa RegisterOperand
        if op.name == IR.LEGACY_WARP_SIZE_SREG
            (kind in _TRANSPILE_INTEGER_SOURCE_ROLES || kind === :pred) ||
                _transpile_reject(path,
                "%warpsize/WARP_SZ is an integer predefined immediate and " *
                "cannot carry .$kind semantics")
            return
        end
        if op.name in SPECIAL_REGS
            # Only the ubiquitous thread-index components are admitted in the
            # finite generic ledger. Other special-register carrier types are
            # not encoded by the inventory and must remain explicit follow-up.
            op.name in ("%tid.x", "%tid.y", "%tid.z", "%ntid.x", "%ntid.y",
                        "%ntid.z", "%ctaid.x", "%ctaid.y", "%ctaid.z",
                        "%nctaid.x", "%nctaid.y", "%nctaid.z", "%laneid") &&
                kind === :u32 || _transpile_reject(path,
                    "special register $(op.name) has no reviewed .$kind carrier mapping")
            return
        end
        _validate_register_source!(state, op, path)
        decl = _declared_register(state.cg, op)
        accepted = _structured_decl_types(kind)
        decl.type in accepted || _transpile_reject(path,
            "source $(op.name) is $(ptx(decl.type)), expected " * join(ptx.(accepted), "/"))
        return
    elseif op isa LabelOperand && _declared_register(state.cg, op) !== nothing
        decl = _declared_register(state.cg, op)
        accepted = _structured_decl_types(kind)
        decl.type in accepted || _transpile_reject(path,
            "source $(op.name) is $(ptx(decl.type)), expected " * join(ptx.(accepted), "/"))
        return
    elseif op isa LabelOperand && _predefined_immediate_expr(op.name) !== nothing
        (kind in _TRANSPILE_INTEGER_SOURCE_ROLES || kind === :pred) ||
            _transpile_reject(path,
            "$(op.name) is an integer predefined immediate and cannot carry " *
            ".$kind semantics")
        return
    end
    _transpile_reject(path,
        "$(nameof(typeof(op))) is not a reviewed .$kind source carrier")
end

function _scalar_destination_role(schema)
    schema.rettype === Float32 && return :f32
    schema.rettype === Float64 && return :f64
    schema.rettype === Int32 && return :s32
    schema.rettype === Int64 && return :s64
    schema.rettype === UInt64 && return :u64
    if schema.rettype === UInt32
        # prmt and packed lane arithmetic use a typeless 32-bit carrier. The
        # remaining UInt32-result islands (popc/clz, unsigned dp/wide, and
        # cvt.pack) have an exact .u32 destination.
        packed = any(mod -> occursin(r"(?:8x4|16x2)$", String(mod)), schema.mods)
        return schema.op === :prmt || packed ? :b32 : :u32
    end
    error("unmapped scalar-result destination carrier $(schema.rettype) for " *
          "$(schema.op).$(join(schema.mods, '.'))")
end

function _validate_scalar_schema!(state::_TranspileContractState,
                                  inst::Instruction, schema, path::String)
    _validate_generic_destination!(state, inst.operands[1],
                                   _scalar_destination_role(schema),
                                   path * ".destination")
    for (i, (source, role)) in enumerate(zip(inst.operands[2:end], schema.operands))
        _validate_typed_source!(state, source, role, "$path.source[$i]")
    end
end

# PTX declaration carriers are not the same thing as PTX.jl's LLVM inline-asm
# bridge. In particular, e2m1x2 is physically .b8 even though the wrapper uses
# UInt16 because NVPTX has no i8 asm constraint. Keep destination storage
# explicit instead of reversing DTYPE_RETTYPE.
const _TRANSPILE_CVT_DEST_DECL_TYPES = Dict{Symbol, Tuple}(
    :bf16 => (ScalarType.B16,),
    :tf32 => (ScalarType.B32,),
    :f16x2 => (ScalarType.F16X2, ScalarType.B32),
    :bf16x2 => (ScalarType.B32,),
    :e4m3x2 => (ScalarType.B16,), :e5m2x2 => (ScalarType.B16,),
    :e2m1x2 => (ScalarType.B8,),
    :e2m3x2 => (ScalarType.B16,), :e3m2x2 => (ScalarType.B16,),
    :ue8m0x2 => (ScalarType.B16,), :s2f6x2 => (ScalarType.B16,),
    :e4m3x4 => (ScalarType.B32,), :e5m2x4 => (ScalarType.B32,),
    :e2m1x4 => (ScalarType.B16,),
    :e2m3x4 => (ScalarType.B32,), :e3m2x4 => (ScalarType.B32,),
)

function _validate_cvt_destination!(state::_TranspileContractState,
                                    op::Operand, format::Symbol, path::String)
    (op isa RegisterOperand || op isa LabelOperand) || _transpile_reject(path,
        "cvt destination must be a declared scalar register")
    decl = _declared_register(state.cg, op)
    decl === nothing && _transpile_reject(path,
        "cvt destination register $(op.name) has no preceding .reg declaration")
    accepted = get(_TRANSPILE_CVT_DEST_DECL_TYPES, format,
                   _structured_decl_types(format))
    isempty(accepted) && error("unmapped cvt declaration format $format")
    decl.type in accepted || _transpile_reject(path,
        "cvt .$format destination $(op.name) is $(ptx(decl.type)), expected " *
        join(ptx.(accepted), "/"))
end

function _validate_cvt_source!(state::_TranspileContractState, op::Operand,
                               role::Symbol, schema, index::Int, path::String)
    predefined = op isa LabelOperand &&
                 _predefined_immediate_expr(op.name) !== nothing
    legacy_warp = op isa RegisterOperand &&
                  op.name == IR.LEGACY_WARP_SIZE_SREG
    if op isa ImmediateOperand || predefined || legacy_warp
        # Ordinary cvt is an exact constant-expression consumer. Exercise its
        # non-evaluating PTX integer/float conversion now so invalid constants
        # fail during preflight, while valid expressions do not get mistaken
        # for the generic literal-only operand subset.
        _render_cvt_source(op, state.cg, role, schema, index)
        return
    end
    _validate_typed_source!(state, op, role, path)
end

function _validate_cvt_schema!(state::_TranspileContractState,
                               inst::Instruction, schema, path::String)
    _validate_cvt_destination!(state, inst.operands[1], schema.destination,
                               path * ".destination")
    sources = inst.operands[2:end]
    if schema.vector_source
        first(schema.operands) === :f32 || error(
            "reviewed vector-source cvt does not encode f32 lane carriers")
        vector = first(sources)::VectorOperand
        for (lane, source) in enumerate(vector.elements)
            _validate_typed_source!(state, source, first(schema.operands),
                                    "$path.source[1].lane[$lane]")
        end
        for (i, (source, role)) in enumerate(zip(sources[2:end],
                                                schema.operands[2:end]))
            _validate_cvt_source!(state, source, role, schema, i + 1,
                                  "$path.source[$(i + 1)]")
        end
        return
    end
    for (i, (source, role)) in enumerate(zip(sources, schema.operands))
        _validate_cvt_source!(state, source, role, schema, i,
                              "$path.source[$i]")
    end
end

function _validate_clc_schema!(state::_TranspileContractState,
                               inst::Instruction, path::String)
    role = any(mod -> occursin("shared::cta", mod), inst.modifiers) ?
           :shared_address : :generic_address
    for (i, operand) in enumerate(inst.operands)
        _validate_address!(state, operand::AddressOperand, "$path.source[$i]",
                           role)
    end
end


function _exact_address_role(modifiers)
    names = string.(modifiers)
    any(name -> occursin("shared", name), names) && return :shared_address
    any(name -> occursin("global", name), names) && return :global_address
    :generic_address
end

function _validate_exact_addresses!(state::_TranspileContractState, sources,
                                    roles, path::String, address_role::Symbol)
    for (i, (source, role)) in enumerate(zip(sources, roles))
        role === :address || continue
        _validate_address!(state, source::AddressOperand,
                           "$path.source[$i]", address_role)
    end
end

function _validate_exact_schema!(state::_TranspileContractState,
                                 inst::Instruction, path::String)
    immediate = _instruction_immediate_form_contract(inst)
    immediate === nothing || return true
    mbarrier = _instruction_mbarrier_schema(state.cg, inst)
    if mbarrier !== nothing
        address_role = mbarrier.schema.space in (:cta, :cluster) ?
                       :shared_address : :generic_address
        _validate_exact_addresses!(state, mbarrier.sources,
                                   mbarrier.variant.operands, path, address_role)
        return true
    end
    b128 = _instruction_b128_schema(state.cg, inst)
    if b128 !== nothing
        _validate_exact_addresses!(state, b128.sources, b128.schema.operands,
                                   path, _exact_address_role(b128.schema.mods))
        if b128.schema.kind === :mov &&
                _declared_register(state.cg, b128.destination) === nothing &&
                !(b128.destination.name in state.cg.inferred_b128_regs)
            _claim_binding!(state, b128.destination.name, :inferred_b128,
                            path * ".destination")
            push!(state.cg.inferred_b128_regs, b128.destination.name)
        end
        return true
    end
    structured = _instruction_structured_result_schema(state.cg, inst)
    vector = _instruction_vector_result_schema(state.cg, inst)
    if vector !== nothing
        _validate_exact_addresses!(state, vector.sources, vector.roles, path,
                                   _exact_address_role(vector.schema.mods))
        for (i, (source, role)) in enumerate(zip(vector.sources, vector.roles))
            role === :cache_policy && source isa ImmediateOperand || continue
            _ptx_integer_carrier_expr(source.text, UInt64)
        end
        return true
    end
    scalar = structured === nothing ? _instruction_scalar_result_schema(inst) : nothing
    cvt = _instruction_cvt_source_schema(state.cg, inst, scalar)
    clc = _instruction_clc_try_cancel_schema(inst)
    scalar === nothing || _validate_scalar_schema!(state, inst, scalar, path)
    cvt === nothing || _validate_cvt_schema!(state, inst, cvt, path)
    clc === nothing || _validate_clc_schema!(state, inst, path)
    structured !== nothing || scalar !== nothing || cvt !== nothing || clc !== nothing
end

function _transpile_form_roles(rule::_TranspileFormRule, source_count::Int)
    variants = Tuple[variant for variant in rule.sources
                     if length(variant) == source_count]
    length(variants) == 1 || return nothing
    only(variants)
end

function _render_transpile_role(cg::CodeGenState, op::Operand, role::Symbol)
    if role === :pred
        if op isa ImmediateOperand
            return _ptx_predicate_constant(op.text) ? "true" : "false"
        elseif op isa LabelOperand && _predefined_immediate_expr(op.name) !== nothing
            return _ptx_predicate_constant(op.name) ? "true" : "false"
        elseif op isa RegisterOperand && op.name == IR.LEGACY_WARP_SIZE_SREG
            return _ptx_predicate_constant("WARP_SZ") ? "true" : "false"
        end
    end
    if role === :u64_or_shared_symbol
        if op isa LabelOperand && julia_var(op.name) in cg.shared_vars
            return "pointer(" * julia_var(op.name) * ")"
        end
        return render_operand(op, cg; type_hint = :u64)
    end
    role in (:generic_address, :global_address, :shared_address,
             :param_u32_address, :param_u64_address, :shared_symbol) &&
        return render_operand(op, cg)
    render_operand(op, cg; type_hint = _schema_operand_hint(role))
end

function _is_pointer_alias_source(state::_TranspileContractState, op::Operand)
    if op isa LabelOperand
        op.name in state.shared_symbols && return true
        return julia_var(op.name) in state.pointer_alias_regs
    elseif op isa RegisterOperand
        return julia_var(op.name) in state.pointer_alias_regs
    end
    false
end

function _record_pointer_alias!(state::_TranspileContractState,
                                inst::Instruction, path::String)
    inst.predicate === nothing || return
    isempty(inst.operands) && return
    destination = inst.operands[1]
    destination isa RegisterOperand || return # matches _try_alias_def!
    produces = if inst.opcode == "mov" && length(inst.operands) == 2
        _is_pointer_alias_source(state, inst.operands[2])
    elseif inst.opcode == "add" && length(inst.operands) == 3
        _is_pointer_alias_source(state, inst.operands[2]) ⊻
            _is_pointer_alias_source(state, inst.operands[3])
    elseif inst.opcode == "sub" && length(inst.operands) == 3
        _is_pointer_alias_source(state, inst.operands[2]) &&
            !_is_pointer_alias_source(state, inst.operands[3])
    else
        false
    end
    produces || return
    isempty(state.labels) || _transpile_reject(path,
        "shared-pointer alias absorption is not control-flow-aware; functions " *
        "with labels/branches require explicit dataflow lowering")
    push!(state.pointer_alias_regs, julia_var(destination.name))
end

# Emission consumes the same finite ledger as validation. There is no second
# result/sink guess and no terminal-modifier broadcast across heterogeneous
# roles (e.g. shl.b64's count remains u32).
function _emit_transpile_form!(cg::CodeGenState, inst::Instruction)
    rule = get(_TRANSPILE_GENERIC_FORMS, (inst.opcode, inst.modifiers), nothing)
    rule === nothing && error("validated transpiler form lost its ledger entry")
    source_start = rule.destination === nothing ? 1 : 2
    sources = inst.operands[source_start:end]
    roles = _transpile_form_roles(rule, length(sources))
    roles === nothing && error("validated transpiler form lost its source-role variant")

    if roles in ((:param_u32_address,), (:param_u64_address,))
        destination, names = render_dst(inst.operands[1], cg)
        address_operand = only(sources)::AddressOperand
        emit_with_predicate!(cg, destination * " = " * julia_var(address_operand.base),
                             inst.predicate, names)
        return
    end

    chain = chain_expr(cg, inst.opcode, inst.modifiers)
    args = join((_render_transpile_role(cg, source, role)
                 for (source, role) in zip(sources, roles)), ", ")
    call = chain * "(" * args * ")"
    if rule.destination === nothing
        emit_with_predicate!(cg, call, inst.predicate, String[])
    else
        destination, names = render_dst(inst.operands[1], cg)
        emit_with_predicate!(cg, destination * " = " * call,
                             inst.predicate, names)
    end
end

function _validate_instruction!(state::_TranspileContractState,
                                inst::Instruction, scope::String, path::String)
    # Programmatically constructed `.file`/`.loc` Instruction nodes affect
    # debug information only. Source parsing currently represents those
    # directives as RawLine, which the body visitor rejects until FRONT-DECL
    # gives them structural nodes. Predicating either would be nonsensical.
    if inst.opcode in (".file", ".loc")
        inst.predicate === nothing ||
            _transpile_reject(path, "debug metadata directives cannot be predicated")
        return
    end

    _validate_predicate!(state, inst.predicate, path * ".predicate")

    if inst.opcode == "ret"
        isempty(inst.modifiers) && isempty(inst.operands) ||
            _transpile_reject(path, "supported ret has no modifiers or operands")
        return
    elseif inst.opcode == "bra"
        isempty(inst.modifiers) ||
            _transpile_reject(path, "supported bra has no modifiers")
        length(inst.operands) == 1 && inst.operands[1] isa LabelOperand ||
            _transpile_reject(path, "bra requires exactly one label target")
        target = (inst.operands[1]::LabelOperand).name
        push!(state.branches, (target, scope, path))
        return
    end

    uses_implicit_cc(inst.opcode, inst.modifiers) && _transpile_reject(path,
        "implicit CC.CF cannot cross instruction-at-a-time Julia calls")
    try
        _validate_exact_schema!(state, inst, path) && return
    catch err
        # Exact-schema helpers deliberately use ArgumentError for direct/raw
        # API validation.  Only this reviewed preflight call boundary converts
        # those policy misses; parser failures and arbitrary emitter errors are
        # never accepted as transpiler-contract rejections.
        err isa ArgumentError || rethrow()
        _transpile_reject(path, sprint(showerror, err); category = :schema)
    end

    mods = _schema_modifiers(inst.modifiers)
    contract = form_contract(Symbol(inst.opcode), mods)
    contract === nothing && _transpile_reject(path,
        "unregistered or forbidden instruction $(inst.opcode)$(join(inst.modifiers))")
    requires_typed_wrapper(Symbol(inst.opcode), mods) && _transpile_reject(path,
        "typed-wrapper-only instruction lacks a reviewed parser-to-wrapper ABI")

    rule = get(_TRANSPILE_GENERIC_FORMS, (inst.opcode, inst.modifiers), nothing)
    rule === nothing && _transpile_reject(path,
        "instruction has no finite transpiler form/role entry; FormContract " *
        "registration alone does not prove its operands")
    source_start = rule.destination === nothing ? 1 : 2
    if rule.destination !== nothing
        isempty(inst.operands) && _transpile_reject(path, "result form is missing its destination")
        _validate_generic_destination!(state, inst.operands[1], rule.destination,
                                       path * ".destination")
    end
    sources = inst.operands[source_start:end]
    variants = [variant for variant in rule.sources if length(variant) == length(sources)]
    isempty(variants) && _transpile_reject(path,
        "expected source arity " * join(sort!(unique(length.(rule.sources))), "/") *
        ", got $(length(sources)); see $(rule.section)")
    accepted = false
    failures = String[]
    for roles in variants
        try
            for (i, (source, role)) in enumerate(zip(sources, roles))
                _validate_typed_source!(state, source, role, "$path.source[$i]")
            end
            accepted = true
            break
        catch err
            err isa TranspilerError || rethrow()
            push!(failures, sprint(showerror, err))
        end
    end
    accepted || _transpile_reject(path,
        "operands do not match the reviewed role variant: " * join(failures, " | ");
        category = :operand)
    _record_pointer_alias!(state, inst, path)
end

function _validate_reg_decl!(state::_TranspileContractState,
                             stmt::RegDecl, path::String)
    stmt.vector_shape === nothing || _transpile_reject(path,
        "vector .reg declarations are structurally parsed but have no reviewed " *
        "Julia local-variable ABI")
    stmt.type in _TRANSPILE_REG_TYPES ||
        _transpile_reject(path, "register type $(ptx(stmt.type)) has no reviewed carrier")
    for declaration in _reg_declarators(stmt)
        declaration.count === nothing || declaration.count > 0 ||
            _transpile_reject(path, "register ranges must have positive extent")
        _check_reg_mangling_shape(declaration, path)
        for existing in values(state.cg.reg_decls)
            (_reg_declarations_collide(existing, declaration) ||
             _reg_declarations_collide(existing, declaration, julia_var)) &&
                _transpile_reject(path,
                    "register declaration $(repr(declaration.name)) overlaps or " *
                    "collides after Julia mangling with $(repr(existing.name))")
        end
        for (jname, (kind, raw)) in state.julia_bindings
            _reg_decl_contains_binding(declaration, jname) &&
                _transpile_reject(path,
                    "register declaration $(repr(declaration.name)) collides " *
                    "after Julia mangling with $kind $(repr(raw))")
        end
        state.cg.reg_decls[declaration.name] = declaration
    end
end

function _validate_var_decl!(state::_TranspileContractState,
                             stmt::VarDecl, path::String)
    stmt.vector_shape === nothing || _transpile_reject(path,
        "vector variable declarations are structurally parsed but have no " *
        "reviewed Julia storage ABI")
    stmt.state_space === StateSpace.SHARED ||
        _transpile_reject(path,
            "$(ptx(stmt.state_space)) storage is not a legal reviewed declaration space")
    stmt.type in _TRANSPILE_SHARED_TYPES ||
        _transpile_reject(path, "shared type $(ptx(stmt.type)) has no exact storage carrier")
    stmt.alignment === nothing ||
        _transpile_reject(path, "explicit shared alignment is not preserved")
    stmt.initializer === nothing ||
        _transpile_reject(path, "shared initializers are not preserved")
    stmt.linking === nothing ||
        _transpile_reject(path, "linked shared declarations are not preserved")
    stmt.name in state.shared_symbols &&
        _transpile_reject(path, "duplicate shared symbol $(repr(stmt.name))")
    _claim_binding!(state, stmt.name, :shared, path)
    push!(state.shared_symbols, stmt.name)
end

function _validate_body!(state::_TranspileContractState,
                         body::Tuple{Vararg{Statement}}, scope::String,
                         path::String)
    for (i, stmt) in enumerate(body)
        spath = _stmt_path(path, i, stmt)
        if stmt isa Comment || stmt isa BlankLine || stmt isa Label
            continue
        elseif stmt isa RegDecl
            _validate_reg_decl!(state, stmt, spath)
        elseif stmt isa VarDecl
            _validate_var_decl!(state, stmt, spath)
        elseif stmt isa Instruction
            _validate_instruction!(state, stmt, scope, spath)
        elseif stmt isa Block
            saved_regs = copy(state.cg.reg_decls)
            saved_shared = copy(state.shared_symbols)
            saved_bindings = copy(state.julia_bindings)
            saved_pointer_regs = copy(state.pointer_alias_regs)
            saved_shared_vars = copy(state.cg.shared_vars)
            saved_aliases = copy(state.cg.pointer_aliases)
            saved_b128 = copy(state.cg.inferred_b128_regs)
            _validate_body!(state, stmt.body, spath, spath * ".body")
            outer_pointer_regs = copy(saved_pointer_regs)
            for name in state.pointer_alias_regs
                any(decl -> _reg_decl_contains_binding(decl, name),
                    values(saved_regs)) && push!(outer_pointer_regs, name)
            end
            state.cg.reg_decls = saved_regs
            state.shared_symbols = saved_shared
            state.julia_bindings = saved_bindings
            state.pointer_alias_regs = outer_pointer_regs
            state.cg.shared_vars = saved_shared_vars
            state.cg.pointer_aliases = saved_aliases
            state.cg.inferred_b128_regs = saved_b128
        elseif stmt isa RawLine
            _transpile_reject(spath,
                "opaque RawLine cannot be commented out without changing execution")
        elseif stmt isa PragmaDirective
            _transpile_reject(spath,
                "function-body .pragma is not represented by emitted Julia")
        elseif stmt isa IntrinsicScope
            _transpile_reject(spath,
                "construction-only IntrinsicScope has no PTX-to-Julia contract")
        else
            _transpile_reject(spath,
                "unsupported statement node $(nameof(typeof(stmt)))")
        end
    end
end

function _validate_function!(func::Function, path::String)
    func.return_params === nothing ||
        _transpile_reject(path, ".func return parameters are not represented")
    func.linking in (nothing, LinkingDirective.VISIBLE) ||
        _transpile_reject(path, "$(ptx(func.linking)) function linkage is not preserved")
    isempty(func.directives) || _transpile_reject(path,
        "function directives are currently emitted as comments, not semantics")
    !func.is_entry && isempty(func.body) &&
        _transpile_reject(path, "body-less .func declarations cannot become Julia definitions")

    state = _TranspileContractState()
    for (i, param) in enumerate(func.params)
        _validate_param!(state, param, "$path.params[$i]", func.is_entry)
    end
    _collect_labels!(state, func.body, path, path * ".body")
    _validate_body!(state, func.body, path, path * ".body")
    for (target, scope, branch_path) in state.branches
        haskey(state.labels, target) ||
            _transpile_reject(branch_path, "branch target $(repr(target)) is undefined")
        state.labels[target] == scope || _transpile_reject(branch_path,
            "branch crosses a PTX brace scope; Julia @goto cannot preserve that edge")
    end
end

"""
    validate_transpilable(mod::IR.Module)

Validate the complete, deliberately narrow IR subset that `ir_to_julia` can
lower without dropping declarations or guessing operand/result roles.  The
visitor runs before emission and rejects every unsupported node with its tree
path.  Parsing/formatting remain broader; rejection here is not a claim that
the input PTX itself is invalid.
"""
function validate_transpilable(mod::Module)
    mod.address_size.size == 64 || _transpile_reject(
        "module.address_size",
        "Julia/NVPTX pointer lowering requires .address_size 64")
    length(mod.target.targets) == 1 || _transpile_reject("module.target",
        "target options are not applied by emitted Julia")
    for (i, stmt) in enumerate(mod.leading)
        _validate_nonsemantic_statement(stmt, _stmt_path("module.leading", i, stmt))
    end

    functions = Function[]
    julia_names = Dict{String, String}()
    for (i, stmt) in enumerate(mod.directives)
        path = _stmt_path("module.directives", i, stmt)
        if stmt isa Function
            jname = julia_var(stmt.name)
            haskey(julia_names, jname) && _transpile_reject(path,
                "function name collides after Julia mangling with $(repr(julia_names[jname]))")
            julia_names[jname] = stmt.name
            push!(functions, stmt)
        elseif stmt isa Comment || stmt isa BlankLine
            continue
        elseif stmt isa TargetDirective
            _transpile_reject(path,
                "a later .target changes the active feature set and cannot be ignored")
        else
            _validate_nonsemantic_statement(stmt, path)
        end
    end
    isempty(functions) &&
        _transpile_reject("module.directives", "module contains no function definition")
    for (i, func) in enumerate(functions)
        _validate_function!(func, "module.functions[$i]($(func.name))")
    end
    nothing
end
