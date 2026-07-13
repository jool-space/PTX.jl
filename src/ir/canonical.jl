# Canonical renaming for structural PTX comparison:
# the golden harness compares instruction sequences *modulo naming*, because
# the backend renumbers virtual registers the moment lowering changes — a
# byte-exact golden dies on every harmless diff. Built on `normalize`, then:
#
#   - virtual registers (`%r5`, `%rd2`, `%p1`, ...) renamed per class in
#     first-appearance order; non-matching names (`%SP`, sregs like `%tid.x`)
#     are stable and pass through. Digit-suffixed special registers
#     (`%envreg0`, `%pm0`, `%clock64`) match the virtual-register shape and
#     are excluded explicitly — renaming them first-appearance would make a
#     kernel reading `%envreg3` canonicalize identically to one reading
#     `%envreg0`, masking a real semantic change from the golden harness
#   - labels (`$L__BB0_3`) renamed in first-appearance order
#   - function names, parameters, and variable declarations renamed
#     positionally (they embed gensym counters and kernel-name mangles)
#   - `.reg` declarations dropped entirely — register counts are allocator
#     output, exactly the noise "modulo naming" must ignore
#
# For parser-produced PTX IR, the result still `format`s to readable
# PTX-shaped text, including lexical brace scopes. Construction-time
# `IntrinsicScope` nodes are retained for structural comparison but have no
# PTX spelling, so formatting such a canonical result intentionally errors.

@public canonicalize

mutable struct _Renamer
    regs::Dict{String,String}
    regcount::Dict{String,Int}
    labels::Dict{String,String}
    names::Dict{String,String}
    namecount::Int
end
_Renamer(names::Dict{String,String}, namecount::Int) =
    _Renamer(Dict{String,String}(), Dict{String,Int}(),
             Dict{String,String}(), names, namecount)

# SPECIAL_REGS is the reviewed full PTX 9.3 inventory from
# special_registers.jl. Most names contain . or _ and cannot match _VREG; the
# digit-suffixed forms (%envregN, %pmN, %clock64) must be excluded explicitly
# so semantic changes are never allocator-renamed away.

const _VREG = r"^%([a-z]+)(\d+)$"

function _reg(rn::_Renamer, name::String)
    r = get(rn.regs, name, nothing)
    r !== nothing && return r
    m = Base.match(_VREG, name)
    m === nothing && return name
    cls = String(m.captures[1])
    k = get(rn.regcount, cls, 0)
    rn.regcount[cls] = k + 1
    rn.regs[name] = "%$cls$k"
end

_label(rn::_Renamer, name::String) =
    get!(() -> "\$L$(length(rn.labels))", rn.labels, name)

"Rename a symbol that may be a known name, a virtual register, or a label."
function _sym(rn::_Renamer, s::String)
    n = get(rn.names, s, nothing)
    n !== nothing && return n
    # Hardware names, stable by definition — checked before the virtual-
    # register shape so `%envreg0`-class sregs never enter the renamer.
    s in SPECIAL_REGS && return s
    Base.match(_VREG, s) !== nothing && return _reg(rn, s)
    startswith(s, "\$L") && return _label(rn, s)
    return s
end

_newname(rn::_Renamer, old::String, stem::String) =
    get!(rn.names, old) do
        rn.namecount += 1
        "$stem$(rn.namecount - 1)"
    end

_canon_op(rn::_Renamer, op::RegisterOperand)  = RegisterOperand(_sym(rn, op.name))
_canon_op(::_Renamer,   op::ImmediateOperand) = op
_canon_op(rn::_Renamer, op::LabelOperand)     = LabelOperand(_sym(rn, op.name))
_canon_op(rn::_Renamer, op::NegatedOperand)   = NegatedOperand(_canon_op(rn, op.operand))
_canon_op(rn::_Renamer, op::PipeOperand) =
    PipeOperand(_canon_op(rn, op.left), _canon_op(rn, op.right))
_canon_op(rn::_Renamer, op::VectorOperand) =
    VectorOperand(Tuple(_canon_op(rn, o) for o in op.elements))
_canon_op(rn::_Renamer, op::ParenthesizedOperand) =
    ParenthesizedOperand(Tuple(_canon_op(rn, o) for o in op.elements))
_canon_op(rn::_Renamer, op::AddressOperand) =
    AddressOperand(_sym(rn, op.base),
                   op.offset === nothing ? nothing : _sym(rn, op.offset),
                   op.coords === nothing ? nothing :
                       Tuple(_canon_op(rn, o) for o in op.coords))

_canon_pred(::_Renamer, ::Nothing) = nothing
_canon_pred(rn::_Renamer, p::Predicate) =
    Predicate(_sym(rn, p.register), p.negated)

function _canon_body(rn::_Renamer, body::Tuple{Vararg{Statement}})
    out = Statement[]
    for s in body
        if s isa RegDecl
            continue
        elseif s isa Instruction
            push!(out, Instruction(
                opcode    = s.opcode,
                modifiers = s.modifiers,
                operands  = Tuple(_canon_op(rn, o) for o in s.operands),
                predicate = _canon_pred(rn, s.predicate)))
        elseif s isa Label
            push!(out, Label(name = _sym(rn, s.name)))
        elseif s isa VarDecl
            push!(out, VarDecl(
                state_space = s.state_space, type = s.type,
                name = _newname(rn, s.name, "var"),
                array_size = s.array_size, alignment = s.alignment,
                initializer = s.initializer, linking = s.linking))
        elseif s isa Block
            # Braces carry lexical declaration/lifetime scope. Keep the node
            # while canonicalizing the names nested inside it.
            push!(out, Block(body = _canon_body(rn, s.body)))
        elseif s isa IntrinsicScope
            # Construction-time scopes have the same nested-name requirement
            # as parsed braces. Retain their metadata for IR comparison.
            push!(out, IntrinsicScope(name = s.name, args_repr = s.args_repr,
                                      body = _canon_body(rn, s.body)))
        else
            push!(out, s)   # RawLine, PragmaDirective, ...
        end
    end
    Tuple(out)
end

_canon_param(p::Param, name::String) = Param(
    state_space = p.state_space, type = p.type, name = name,
    array_size = p.array_size, alignment = p.alignment,
    ptr_state_space = p.ptr_state_space, ptr_alignment = p.ptr_alignment)

function _canon_function(f::Function, names::Dict{String,String},
                         namecount::Int, idx::Int)
    rn = _Renamer(copy(names), namecount)
    params = Param[]
    for (i, p) in enumerate(f.params)
        nm = "param$(i-1)"
        rn.names[p.name] = nm
        push!(params, _canon_param(p, nm))
    end
    rets = f.return_params
    if rets !== nothing
        out = Param[]
        for (i, p) in enumerate(rets)
            nm = "ret$(i-1)"
            rn.names[p.name] = nm
            push!(out, _canon_param(p, nm))
        end
        rets = Tuple(out)
    end
    Function(is_entry = f.is_entry, name = "entry$idx",
             params = Tuple(params), return_params = rets,
             body = _canon_body(rn, f.body),
             linking = f.linking, directives = f.directives)
end

"""
    canonicalize(m::Module) -> Module

Canonical form for comparing PTX *structure*: `normalize`d, with virtual
registers, labels, parameters, declared variables, and function names
renamed to position-stable canonical names, and `.reg` declarations
dropped. Lexical scope nodes remain, with their nested operands renamed.
For parser-produced PTX IR, format-identical canonical results have the same
structure modulo allocator-style naming — the golden-harness equivalence.
Construction-time `IntrinsicScope` nodes retain metadata but intentionally
cannot be formatted; compare their canonical IR directly instead.
"""
function canonicalize(m::Module)
    m = normalize(m)
    names = Dict{String,String}()
    count = 0
    for d in m.directives           # module-level symbols, declaration order
        if d isa VarDecl
            names[d.name] = "var$count"
            count += 1
        end
    end
    dirs = Statement[]
    nfun = 0
    # Non-function module statements use one renamer in directive order, just
    # like sibling statements in a function body. Function-local renamers
    # remain isolated because a function is its own declaration namespace.
    module_rn = _Renamer(copy(names), count)
    for d in m.directives
        if d isa Function
            push!(dirs, _canon_function(d, names, count, nfun))
            nfun += 1
        elseif d isa VarDecl
            push!(dirs, VarDecl(
                state_space = d.state_space, type = d.type,
                name = names[d.name],
                array_size = d.array_size, alignment = d.alignment,
                initializer = d.initializer, linking = d.linking))
        else
            # Top-level parsing permits instructions, labels, and `.reg`;
            # retained programmatic scopes recurse too. RawLine/pragma remain
            # unchanged.
            append!(dirs, _canon_body(module_rn, (d,)))
        end
    end
    Module(version = m.version, target = m.target,
           address_size = m.address_size, directives = Tuple(dirs))
end
