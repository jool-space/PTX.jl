# PTX 9.3 §9.7.2: these are the only instruction families that access the
# implicit condition-code flag CC.CF.  Keep this predicate independent of the
# lowering registry: the parser/code generator needs the same semantic boundary
# before `ledgers/forms.jl` and the `dsl/` chain machinery are loaded.
_cc_modifier(m::Symbol) = m === :cc
_cc_modifier(m::AbstractString) = m == "cc" || m == ".cc"

function uses_implicit_cc(op::Union{Symbol, AbstractString}, modifiers)
    name = op isa Symbol ? op : Symbol(op)
    name in (:addc, :subc, :madc) ||
        (name in (:add, :sub, :mad) && any(_cc_modifier, modifiers))
end
