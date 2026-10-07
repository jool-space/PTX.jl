struct Operation{op, mods} end

struct Chain{mods} end

# Type-level chain composition. See `@mod_str` for semantics and patterns.
Base.:*(::Operation{op, M1}, ::Chain{M2}) where {op, M1, M2} =
    Operation{op, (M1..., M2...)}()
Base.:*(::Operation{op, M},  s::Symbol) where {op, M} =
    Operation{op, (M..., s)}()
Base.:*(::Chain{M1}, ::Chain{M2}) where {M1, M2} =
    Chain{(M1..., M2...)}()
Base.:*(::Chain{M},  s::Symbol) where {M} =
    Chain{(M..., s)}()

struct SpecialReg{name} end

# --- The raw tier -------------------------------------------------------------
#
# `ptx"..."raw` — the explicit opt-in for eligible chains the form registry
# doesn't know. Same rendering machinery as the registered chain, but under
# RAW_CONTRACT: sideeffect + memory clobber + convergent, pointer operands
# bracketed, generic scalar return inference. Audited fixed-result forms retain
# their exact ABI and carrier validation, and an unmatched spelling inside one
# of those grammar islands is rejected because raw has no explicit-result
# syntax. A semantic guard still rejects hidden state, such as CC.CF, that one
# call cannot model.
# Composition mirrors Operation so a raw chain can be extended with `*`.

struct RawOperation{op, mods} end

Base.:*(::RawOperation{op, M1}, ::Chain{M2}) where {op, M1, M2} =
    RawOperation{op, (M1..., M2...)}()
Base.:*(::RawOperation{op, M},  s::Symbol) where {op, M} =
    RawOperation{op, (M..., s)}()

"""
    ptx"opcode.mod1.mod2..."
    ptx"%name"

Construct an `Operation{op, mods}` singleton — `op::Symbol` is the opcode
(first segment), `mods::Tuple{Vararg{Symbol}}` is the modifier chain. Splits
on `.`; each segment becomes one Symbol verbatim, so `::` (PTX sub-namespace
separator), digit-leading tokens (`3d`, `m16n8k32`), and underscores in
modifier names all flow through cleanly.

Supports `\$x` / `\$(expr)` interpolation. The macro expands to a chain of
type-domain `*` compositions:

  * Static segments fold into the initial `Operation{op, mods}()` and
    subsequent `Chain{(...)}()` constants.
  * A *glued* interpolation (literal chars adjacent on either side, e.g.
    `x\$(N)` or `\$(N)d`) emits `Symbol(pre, expr, post)` and composes via
    `Operation * Symbol`. With a compile-time-constant interpolated value
    (e.g. `N` from a `Val{N}` unwrap) the whole site folds to the same
    singleton the literal form would produce — making it safe to use inside
    device kernels.
  * A *bare* interpolation (between two `.`s) emits `_ptx_dyn_seg(expr)`,
    which yields a `Chain` — supports `String` values containing `.` (split
    into multiple modifier segments) for host-side use.
  * An interpolated *opcode* (first segment) falls back to
    `_ptx_op_from_string(...)`. Device kernels should keep the opcode
    literal.

Examples:

    ptx"add.f32"(a, b)
    ptx"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"(a, b, c)
    ptx"cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(...)
    ptx"bar.sync"(Val(0))
    ptx"mov.u32"(ptx"%tid.x")

    dt = "u32"
    ptx"mov.\$dt"(x)              # ≡ ptx"mov.u32"(x)
    ptx"st.\$(space).b32"(p, v)   # \$(...) for non-identifier exprs

    # Glued — folds to a literal singleton when N is Val-known:
    @inline f(p, ::Val{N}) where {N} =
        ptx"ldmatrix.sync.aligned.m8n8.x\$(N).shared.b16"(p)

Empty literal parts (consecutive `.`, leading/trailing `.`, or empty string)
error at expansion for the static path, and at runtime for the interp path.

A spelling that starts with `%` is a special register rather than an
instruction: `ptx"%tid.x"` constructs the `SpecialReg{Symbol("%tid.x")}`
operand, which renders as the verbatim PTX token. The name must be a scalar
special register of the PTX inventory (`%tid` alone is a vector; read
`%tid.x`). Interpolation folds the same way as a glued modifier:

    @inline block_index(axis::Symbol) = ptx"mov.u32"(ptx"%ctaid.\$axis")

A static spelling is also a method-definition head. The same `ptx""` that
calls a wrapper defines it, so the method is dispatchable by its spelling by
construction:

    @inline ptx"add.f32"(a::Float32, b::Float32) = ...

Interpolated spellings are call-only.

See also: [`@mod_str`](@ref) for modifier-only chains usable on the right
side of `*`.
"""
macro ptx_str(s::String)
    startswith(s, '%') && return _special_reg_expr(s)
    if !occursin('$', s)
        op, mods = _static_head("ptx", s)
        return esc(_operation_binding!(__module__, s, op, mods))
    end
    isempty(s) && error("ptx\"\": empty modifier chain")
    return _ptx_build_interp(s)
end

# A static `ptx"..."` expands to a name, not to `Operation{op, mods}()`,
# because a name is the one form Julia accepts both as a value and as the
# head of `f(args) = body`. The name is a `const` in the expanding module
# bound to the singleton; a definition whose head is a const singleton
# instance adds the method to the singleton's type, so `ptx"X"(a) = ...`
# defines the method that `ptx"X"(a)` calls. The binding is created on first
# expansion of a spelling. It has to live in the expanding module: during
# precompilation that is the only module open for evaluation, so a fixed
# home such as a `PTX.Ops` submodule would need every spelling bound ahead
# of time. Accessing the const folds to the singleton exactly like the
# literal, so a call site compiles to the same code either way.
#
# A `ptx"..."` quoted inside a `@generated` body is expanded when the
# generator runs, under inference, where `eval` is forbidden and a new
# binding would land in a newer world than the one the body compiles in.
# That context can only call, never define, so it gets the literal.
function _operation_binding!(mod::Module, spelling::String, op::Symbol, mods::Tuple)
    literal = :( $Operation{$(QuoteNode(op)), $mods}() )
    ccall(:jl_is_in_pure_context, Bool, ()) && return literal
    name = Symbol("#ptx#", spelling)
    isdefined(mod, name) || Core.eval(mod, :(const $name = $literal))
    name
end

# Shared static-spelling parser for `ptx""` and `ptx""raw`: one splitter so a
# spelling means the same `(op, mods)` everywhere.
function _static_head(macroname::String, s::String)
    isempty(s) && error(macroname * "\"\": empty modifier chain")
    raw = split(s, '.')
    any(isempty, raw) &&
        error(macroname * "\"" * s * "\": empty modifier between '.'")
    Symbol(raw[1]), Tuple(Symbol(p) for p in @view raw[2:end])
end

# `ptx"..."raw` — RawOperation escape hatch for eligible unregistered chains
# (static chains only: the raw tier is a deliberate, spelled-out act;
# interpolation belongs to the registered surface). Semantic guards such as
# implicit CC.CF remain fail-loud because conservative attributes cannot model
# their hidden input/output dependency.
macro ptx_str(s::String, flag::String)
    flag == "raw" ||
        error("ptx\"...\"$flag: unknown flag (only `raw` is supported)")
    occursin('$', s) &&
        error("ptx\"...\"raw: interpolation is not supported on the raw tier")
    startswith(s, '%') &&
        error("ptx\"" * s * "\"raw: special registers have no raw tier")
    op, mods = _static_head("ptx", s)
    :( $RawOperation{$(QuoteNode(op)), $mods}() )
end

"""
    mod"mod1.mod2..."

Construct a `Chain{mods}` singleton — a sequence of PTX modifiers with no
opcode. Splits on `.` like [`@ptx_str`](@ref) but does not accept `\$`
interpolation: there is no opcode context to interpolate against, and the
type-level `*` (below) already handles compile-time composition.

`mod""` is the empty chain `Chain{()}`. Composing it via `*` is a no-op —
useful as an "absent modifier" sentinel for conditional helpers.
"""
macro mod_str(s::String)
    occursin('\$', s) && error("mod\"" * s * "\": interpolation not supported (use `*` to compose)")
    isempty(s) && return :( $Chain{()}() )
    raw = split(s, '.')
    any(isempty, raw) && error("mod\"" * s * "\": empty modifier between '.'")
    mods = Tuple(Symbol(p) for p in raw)
    :( $Chain{$mods}() )
end

# Parse `$x` / `$(expr)` from a ptx-string macro body into an
# `Expr(:string, ...)` whose pieces are literal substrings and Julia exprs.
# `Meta.parse(..., greedy=false)` stops at `.`, so `$x.f32` interpolates only
# `x` — exactly what we want for dot-separated PTX modifier chains.
function _ptx_parse_interp(s::String)
    pieces = Any[]
    buf = IOBuffer()
    i = firstindex(s)
    n = lastindex(s)
    while i <= n
        c = s[i]
        if c == '$'
            data = String(take!(buf))
            isempty(data) || push!(pieces, data)
            j = nextind(s, i)
            j > n && error("ptx\"" * s * "\": dangling '\$' at end of string")
            expr, k = Meta.parse(s, j; greedy=false)
            expr === nothing && error("ptx\"" * s * "\": empty interpolation after '\$'")
            push!(pieces, expr)
            i = k
        else
            print(buf, c)
            i = nextind(s, i)
        end
    end
    data = String(take!(buf))
    isempty(data) || push!(pieces, data)
    Expr(:string, pieces...)
end

# Walk the macro body and emit `Operation{op,init}() * seg * Chain{run}() * ...`,
# where each `seg` for a dynamic atom is either `Symbol(pre, expr, post, ...)`
# (glued — adjacent literal chars) or `_ptx_dyn_seg(expr)` (bare — sits alone
# between two `.`s). When all interpolated values are compile-time constants
# (or `Val`-unwrapped type params), the whole site folds to the same singleton
# the literal form would produce.
function _ptx_build_interp(s::String)
    atoms = _ptx_interp_atoms(s)
    # Validation deferred to runtime so error type matches pre-existing tests.
    for atom in atoms
        if isempty(atom)
            return :( error($("ptx\"" * s * "\": empty modifier between '.'")) )
        end
    end
    first_atom = atoms[1]
    # Dyn opcode → fall back to host-only string assembly.
    if !(length(first_atom) == 1 && first_atom[1] isa String)
        str_expr = _ptx_parse_interp(s)
        return :( $_ptx_op_from_string($(esc(str_expr))) )
    end
    op_sym = Symbol(first_atom[1])
    idx = 2
    init_mods = Symbol[]
    while idx <= length(atoms) && length(atoms[idx]) == 1 && atoms[idx][1] isa String
        push!(init_mods, Symbol(atoms[idx][1]))
        idx += 1
    end
    init_tup = Tuple(init_mods)
    acc = :( $Operation{$(QuoteNode(op_sym)), $init_tup}() )
    while idx <= length(atoms)
        acc = :( $acc * $(_ptx_atom_expr(atoms[idx])) )
        idx += 1
        run = Symbol[]
        while idx <= length(atoms) && length(atoms[idx]) == 1 && atoms[idx][1] isa String
            push!(run, Symbol(atoms[idx][1]))
            idx += 1
        end
        if !isempty(run)
            run_tup = Tuple(run)
            acc = :( $acc * $Chain{$run_tup}() )
        end
    end
    return acc
end

# Group the macro body into modifier-level atoms. Each atom is a `Vector{Any}`
# whose elements are literal `String` chunks and user `Expr`s (in source order).
# `.` outside an interpolation closes the current atom.
function _ptx_interp_atoms(s::String)
    atoms = Vector{Any}[]
    cur   = Any[]
    buf   = IOBuffer()
    flush_buf! = () -> begin
        data = String(take!(buf))
        isempty(data) || push!(cur, data)
    end
    i = firstindex(s)
    n = lastindex(s)
    while i <= n
        c = s[i]
        if c == '$'
            flush_buf!()
            j = nextind(s, i)
            j > n && error("ptx\"" * s * "\": dangling '\$' at end of string")
            expr, k = Meta.parse(s, j; greedy=false)
            expr === nothing && error("ptx\"" * s * "\": empty interpolation after '\$'")
            push!(cur, expr)
            i = k
        elseif c == '.'
            flush_buf!()
            push!(atoms, cur)
            cur = Any[]
            i = nextind(s, i)
        else
            print(buf, c)
            i = nextind(s, i)
        end
    end
    flush_buf!()
    push!(atoms, cur)
    return atoms
end

# One modifier atom containing ≥1 interpolation → Expr building a Symbol
# (when glued) or a Chain (when the atom is exactly one bare `$expr`).
function _ptx_atom_expr(parts::Vector{Any})
    if length(parts) == 1 && !(parts[1] isa String)
        return :( $_ptx_dyn_seg($(esc(parts[1]))) )
    end
    args = Any[]
    for p in parts
        push!(args, p isa String ? p : esc(p))
    end
    return :( $_ptx_sym($(args...)) )
end

# Runtime path for bare `$expr` modifier atoms — value-dependent number of
# resulting segments. Preserves the legacy "string with `.`s splits into
# multiple parts" behavior. For `Symbol` / `Integer` / `Val{V}` inputs the
# result is a 1-segment Chain that folds at compile time.
@inline _ptx_dyn_seg(x::Symbol) = Chain{(x,)}()
@inline _ptx_dyn_seg(x::Integer) = Chain{(_ptx_sym(x),)}()
@inline _ptx_dyn_seg(::Val{V}) where {V} = Chain{(_ptx_sym(V),)}()
@inline function _ptx_dyn_seg(x::AbstractString)
    if occursin('.', x)
        parts = split(x, '.')
        any(isempty, parts) && error("ptx interp: empty modifier between '.' in ", repr(String(x)))
        Chain{Tuple(Symbol(p) for p in parts)}()
    else
        Chain{(Symbol(x),)}()
    end
end

# Glued-Symbol builder for the macro's interp path. Marked `:total` so that
# `Symbol(literal_str, N, ...)` folds at compile time when `N` is a constant
# (e.g. a `Val{N}` unwrap). Without this assertion, the compiler refuses to
# const-fold through `Base.print_to_string`'s heap allocation, even for
# concrete inputs — and the call site stops being device-safe.
Base.@assume_effects :total _ptx_sym(args...) = Symbol(args...)

function _ptx_op_from_string(s::AbstractString)
    raw = split(s, '.')
    any(isempty, raw) && error("ptx string \"" * String(s) * "\": empty modifier between '.'")
    op = Symbol(raw[1])
    mods = Tuple(Symbol(p) for p in @view raw[2:end])
    Operation{op, mods}()
end

# `ptx"%name"` is a special-register operand: no PTX opcode begins with `%`.
# A static name is checked here; an interpolated one folds to its
# `SpecialReg` when the pieces are constants and is checked where the operand
# renders.
function _special_reg_expr(s::String)
    if occursin('$', s)
        pieces = _ptx_parse_interp(s).args
        return :( $_special_reg($((p isa String ? p : esc(p) for p in pieces)...)) )
    end
    _check_special_reg(s)
    :( $SpecialReg{$(QuoteNode(Symbol(s)))}() )
end

@inline _special_reg(args...) = SpecialReg{_ptx_sym(args...)}()

function _check_special_reg(name::AbstractString)
    name in IR.SCALAR_SPECIAL_REGS && return
    hint = name in IR.V4_SPECIAL_REG_ROOTS ?
        " (read one component, e.g. ptx\"$name.x\")" : ""
    error("ptx\"" * name * "\": not a scalar PTX special register" * hint)
end
