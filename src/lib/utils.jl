"""
Parse-time metaprogramming helpers for register-resident device code.

Two utilities, both existing because GPU kernels routinely need loop
bodies where the induction value is a *compile-time* constant — `Val(i)`
arguments, distinct per-iteration variable names (an indexed collection
of loop-carried state demotes to local memory), tuple `getindex` with a
constant — and where large `ntuple(...) do` closures escape the inliner
and become real device CALLs:

- [`@unrolled`](@ref Utils.@unrolled): statement-level loop unrolling at
  macro-expansion time. The LLVM-hint kind of unrolling (loopinfo
  metadata) cannot provide any of the above; this is the guaranteed,
  constant-substituting kind.
- [`strided_reduce`](@ref Utils.strided_reduce): `@generated` strided
  tuple reduction with a pinned pairwise tree — the closure-free
  replacement for hand-written reduction trees.

Pure leaf module: no PTX dependencies, host-testable.
"""
module Utils

using Republic: @public

@public @unrolled, strided_reduce

# ── @unrolled ───────────────────────────────────────────────────────────

# Literal iteration values, or a loud refusal. Parse-time expansion can
# only see literals: a bound like `N - 1` or a `Val`-derived constant is
# invisible here (route those through a @generated function instead).
function _unrolled_values(iter)
    if Meta.isexpr(iter, :call) && iter.args[1] === :(:) &&
       all(a -> a isa Integer, iter.args[2:end])
        length(iter.args) == 3 && return collect(iter.args[2]:iter.args[3])
        length(iter.args) == 4 && return collect(iter.args[2]:iter.args[3]:iter.args[4])
    end
    Meta.isexpr(iter, :tuple) && return iter.args
    error("@unrolled: the iterator must be a literal range (`0:3`, `0:2:6`) " *
          "or a literal tuple; got `$iter`. Non-literal trip counts need a " *
          "@generated function, not a parse-time unroll.")
end

function _unrolled_bindings(lhs, val)
    if lhs isa Symbol
        return Pair{Symbol, Any}[lhs => val]
    elseif Meta.isexpr(lhs, :tuple) && all(a -> a isa Symbol, lhs.args)
        Meta.isexpr(val, :tuple) && length(val.args) == length(lhs.args) ||
            error("@unrolled: destructuring `$lhs` needs every iterator " *
                  "element to be a literal tuple of $(length(lhs.args)); " *
                  "got `$val`")
        return Pair{Symbol, Any}[k => v for (k, v) in zip(lhs.args, val.args)]
    end
    error("@unrolled: the loop binding must be a symbol or a tuple of " *
          "symbols; got `$lhs`")
end

_substitute(x, _) = x
function _substitute(s::Symbol, binds::Vector{Pair{Symbol, Any}})
    for (k, v) in binds
        s === k && return v
    end
    str = String(s)
    for (k, v) in binds
        v isa Union{Integer, Symbol} || continue
        suffix = '_' * String(k)
        if endswith(str, suffix) && length(str) > length(suffix)
            return Symbol(str[1:end - length(suffix)], '_', v)
        end
    end
    return s
end
function _substitute(e::Expr, binds::Vector{Pair{Symbol, Any}})
    e.head in (:quote, :meta) && return e
    # keyword-argument names are not references to the loop variable
    e.head === :kw &&
        return Expr(:kw, e.args[1],
                    map(a -> _substitute(a, binds), e.args[2:end])...)
    Expr(e.head, map(a -> _substitute(a, binds), e.args)...)
end

"""
    @unrolled for x in <literal range or tuple>
        body
    end

Expand the loop at macro-expansion time: the body is repeated once per
iteration with `x` replaced by that iteration's value, so `x` is a
compile-time constant inside each copy — `Val(x)`, constant tuple
indexing, and constant shifts all work.

Two substitution rules, following `Base.Cartesian`'s conventions:

- `x` as a standalone symbol becomes the iteration value (for tuple
  iterators, the element expression verbatim).
- An identifier with a literal `_x` suffix gets the suffix replaced by
  the value: `ph_x` → `ph_0`, `ph_1`, ... This is how per-iteration
  *names* are minted — the mechanism that keeps loop-carried state in
  distinct registers instead of an indexable (hence local-memory)
  collection.

The iterator must be a parse-time literal: a range of integer literals
(`0:3`, `0:2:6`) or a tuple, optionally destructured —
`for (s, tma) in ((0, tma_K), (1, tma_V))` binds `s` to a constant and
`tma` to the spliced expression. Anything else is refused at expansion
time.

The expanded bodies are spliced into the *enclosing* scope (like
`Base.Cartesian.@nexprs`, unlike a real `for`): assignments made in the
body persist after the loop, which is what per-iteration phase variables
need.
"""
macro unrolled(loop)
    Meta.isexpr(loop, :for) ||
        error("@unrolled needs a `for` loop, got `$(loop isa Expr ? loop.head : loop)`")
    head, body = loop.args
    Meta.isexpr(head, :(=), 2) ||
        error("@unrolled supports exactly one induction binding " *
              "(`for x in iter`), not comma-separated loops")
    lhs, iter = head.args
    vals = _unrolled_values(iter)
    esc(Expr(:block,
             (_substitute(body, _unrolled_bindings(lhs, val)) for val in vals)...))
end

"""
    @unrolled CAP for x in start:stop        body end
    @unrolled CAP for x in start:step:stop   body end

The capped form: for trip counts that are compile-time constants but
not parse-time literals (`Val`-derived tile parameters — the norm
kernels' `@nexprs 16` + folded-guard idiom, generated).

`CAP` (a literal positive integer) is the maximum number of
iterations; `start` and `step` must be literal integers, `stop` may be
any expression. The macro expands `CAP` guarded copies — copy `i`
substitutes the literal value `start + i·step` and executes only when
that value is `≤ stop` (`≥ stop` for a negative step). The guard
compares against the *spliced stop expression*, so `0:(D8 - 1)` and
`1:D8` both do exactly what they say; there is no privileged start
value. `stop` is evaluated once, before the first copy.

Semantics: identical to `for x in start:step:stop` (with the enclosing-
scope splice of the uncapped form) **provided the range has at most
`CAP` iterations** — iterations beyond the cap are silently absent,
so the cap is a caller-asserted ceiling; check it where the values are
chosen (host side), the way the norm kernels assert `v4_iters ≤ 16`.
When `stop` folds to a specialization-time constant the guards vanish;
a genuinely runtime `stop` leaves a predicated unroll, which is still
correct.
"""
macro unrolled(cap, loop)
    cap isa Integer && cap >= 1 ||
        error("@unrolled: the cap must be a literal positive integer, got `$cap`")
    Meta.isexpr(loop, :for) ||
        error("@unrolled needs a `for` loop, got `$(loop isa Expr ? loop.head : loop)`")
    head, body = loop.args
    Meta.isexpr(head, :(=), 2) ||
        error("@unrolled supports exactly one induction binding " *
              "(`for x in iter`), not comma-separated loops")
    lhs, iter = head.args
    lhs isa Symbol ||
        error("@unrolled with a cap needs a plain symbol binding; " *
              "destructuring only makes sense for literal tuple iterators")
    Meta.isexpr(iter, :call) && iter.args[1] === :(:) &&
        length(iter.args) in (3, 4) ||
        error("@unrolled with a cap needs a range iterator " *
              "(`start:stop` or `start:step:stop`); got `$iter`")
    start = iter.args[2]
    step  = length(iter.args) == 4 ? iter.args[3] : 1
    stop  = iter.args[end]
    start isa Integer ||
        error("@unrolled: the range start must be a literal integer, got `$start`")
    step isa Integer && step != 0 ||
        error("@unrolled: the range step must be a nonzero literal integer, " *
              "got `$step`")
    cmp = step > 0 ? :(<=) : :(>=)
    stopvar = gensym(:unrolled_stop)
    copies = map(0:(cap - 1)) do i
        v = start + i * step
        Expr(:if, Expr(:call, cmp, v, stopvar),
             _substitute(body, _unrolled_bindings(lhs, v)))
    end
    esc(Expr(:block, :(local $stopvar = $stop), copies...))
end

# ── strided_reduce ──────────────────────────────────────────────────────

# Balanced tree over `leaves` with up-to-F-ary nodes; a short leftover
# group joins as one node (or passes through unpaired when singleton).
# F=2, 8 leaves → ((l1⊕l2)⊕(l3⊕l4))⊕((l5⊕l6)⊕(l7⊕l8)).
# F=3, 8 leaves → op(op(l1,l2,l3), op(l4,l5,l6), op(l7,l8)).
function _reduce_tree(leaves::Vector{Any}, fanout::Int)
    while length(leaves) > 1
        next = Any[]
        for i in 1:fanout:length(leaves)
            group = leaves[i:min(i + fanout - 1, end)]
            push!(next, length(group) == 1 ? group[1] : :(op($(group...))))
        end
        leaves = next
    end
    leaves[1]
end

# `NTuple{N, Any}` (not `NTuple{N, T}`) in both methods below: a
# length-0 tuple would leave T unbound, and the generator never needs
# the element type. (This comment sits ABOVE the docstring because a
# comment line between a docstring and its definition silently detaches
# the doc.)
"""
    strided_reduce(op, w::NTuple{N}, ::Val{S}[, ::Val{F}]) -> NTuple{S}

Fold an `N`-tuple into `S` strided accumulators:
`out[a] = op(w[a], w[a+S], w[a+2S], ...)` for `a in 1:S`, with `N`
divisible by `S`.

Fully unrolled by `@generated` expansion — no closure, nothing for the
inliner to drop — so the operands stay in registers. Each accumulator
reduces through a balanced tree of up-to-`F`-ary `op` calls (default
`F = 2`, pairwise), and the shape is part of the contract: for
non-associative `op` (floating-point `+`) the pairwise result is
bit-reproducible against any other pairwise implementation, at tree
depth `⌈log₂(N/S)⌉` rather than the `N/S - 1` of a left fold.

`F = 3` groups leaves in threes — `op(op(l₁,l₂,l₃), op(l₄,l₅,l₆), ...)`
— which NVPTX fuses into the 3-input `max`/`min` instructions on
`sm_100`+ for `max`/`min` reductions. Only reach for it when `op` is
exactly associative (`max`, `min`, integer `+`): for floating-point
addition a fanout change is an association change, i.e. different
numerics.
"""
@inline strided_reduce(op::F, w::NTuple{N, Any}, s::Val) where {F, N} =
    strided_reduce(op, w, s, Val(2))
@generated function strided_reduce(op::F, w::NTuple{N, Any}, ::Val{S},
                                   ::Val{FAN}) where {F, N, S, FAN}
    S isa Int && 1 <= S <= N ||
        error("strided_reduce: S must be an Int in 1:$N, got $S")
    N % S == 0 ||
        error("strided_reduce: tuple length $N is not divisible by S = $S")
    FAN isa Int && FAN >= 2 ||
        error("strided_reduce: fanout must be an Int >= 2, got $FAN")
    accs = [_reduce_tree(Any[:(w[$i]) for i in a:S:N], FAN) for a in 1:S]
    quote
        $(Expr(:meta, :inline))
        tuple($(accs...))
    end
end

end # module Utils
