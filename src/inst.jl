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

"""
    ptx"opcode.mod1.mod2..."

Construct an `Operation{op, mods}` singleton — `op::Symbol` is the opcode
(first segment), `mods::Tuple{Vararg{Symbol}}` is the modifier chain. Splits
on `.`; each segment becomes one Symbol verbatim, so `::` (PTX sub-namespace
separator), digit-leading tokens (`3d`, `m16n8k32`), and underscores in
modifier names all flow through cleanly.

Supports `\$x` / `\$(expr)` interpolation. Interpolated values are
`string`-ified, spliced into the modifier string, and split on `.` — so an
interpolated value containing `.` produces multiple parts. With constant
inputs the call still folds to the singleton; with runtime values the
`Operation{op, mods}()` construction happens at the call site.

Examples:

    ptx"add.f32"(a, b)
    ptx"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"(a, b, c)
    ptx"cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(...)
    ptx"bar.sync"(Val(0))
    ptx"mov.u32"(sreg"%tid.x")

    dt = "u32"
    ptx"mov.\$dt"(x)              # ≡ ptx"mov.u32"(x)
    ptx"st.\$(space).b32"(p, v)   # \$(...) for non-identifier exprs

Empty parts (consecutive `.`, leading/trailing `.`, or empty string) error at expansion.

See also: [`@mod_str`](@ref) for modifier-only chains usable on the right
side of `*`.
"""
macro ptx_str(s::String)
    isempty(s) && error("ptx\"\": empty modifier chain")
    if !occursin('$', s)
        raw = split(s, '.')
        any(isempty, raw) && error("ptx\"" * s * "\": empty modifier between '.'")
        op = Symbol(raw[1])
        mods = Tuple(Symbol(p) for p in @view raw[2:end])
        return :( $Operation{$(QuoteNode(op)), $mods}() )
    end
    str_expr = _ptx_parse_interp(s)
    return :( $_ptx_op_from_string($(esc(str_expr))) )
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

function _ptx_op_from_string(s::AbstractString)
    raw = split(s, '.')
    any(isempty, raw) && error("ptx string \"" * String(s) * "\": empty modifier between '.'")
    op = Symbol(raw[1])
    mods = Tuple(Symbol(p) for p in @view raw[2:end])
    Operation{op, mods}()
end

"""
    sreg"name"

Construct a `SpecialReg{Symbol("%name")}` singleton — a compile-time
literal for a PTX special register. Bakes the verbatim asm token, so
underscore-bearing names (`%cluster_ctarank`, `%lanemask_eq`,
`%total_smem_size`) round-trip losslessly. Accepts either form:

    sreg"tid.x"            ≡ sreg"%tid.x"            → "%tid.x"
    sreg"cluster_ctarank"  ≡ sreg"%cluster_ctarank"  → "%cluster_ctarank"
"""
macro sreg_str(s::String)
    isempty(s) && error("sreg\"\": empty register name")
    name = startswith(s, '%') ? s : '%' * s
    sym = Symbol(name)
    :( $SpecialReg{$(QuoteNode(sym))}() )
end

const NONPURE_OPCODES = Set{Symbol}((
    :bar, :mbarrier, :fence, :wgmma, :tcgen05, :cluster, :cp,
    :setmaxnreg, :elect, :prefetch, :tensormap,
    :ld, :st, :atom, :red, :ldmatrix, :stmatrix,
    # Warp-collective ops: each lane's result depends on every other lane's
    # input. Without `~{memory}` LLVM hoists/constant-folds them as if pure
    # and silently loses the cross-lane semantics.
    :vote, :shfl, :match, :redux, :activemask, :membar,
    # sm_90 cluster intrinsics: observable cross-CTA visibility.
    :mapa, :getctarank,
    # Inter-launch / kernel-control.
    :griddepcontrol, :clusterlaunchcontrol, :exit,
))

# Memory-op opcodes whose pointer args render as `[%addr]`; cvta/mov etc. don't.
const BRACKET_PTR_OPCODES = Set{Symbol}((
    :ld, :st, :atom, :red, :cp, :mbarrier, :ldmatrix, :stmatrix,
    :prefetch, :tcgen05, :tensormap, :fence,
))

is_nonpure_opcode(op::Symbol) = op in NONPURE_OPCODES

bracket_pointers(op::Symbol) = op in BRACKET_PTR_OPCODES

# Reading a special register is observable.
function has_special_reg(argtypes)
    for T in argtypes
        T <: SpecialReg && return true
    end
    false
end

# (constraint_letter, argtype, lane) — `lane` is set for tuple args that emit
# a braced register-vector group (one slot per lane).
const InputSlot = Tuple{String, Type, Union{Nothing, Int}}

function render_arg(::Type{P}, slot::Int, bracket::Bool) where {P <: Core.LLVMPtr}
    op_str = bracket ? "[\$" * string(slot) * "]" : "\$" * string(slot)
    op_str, InputSlot[("l", P, nothing)], slot + 1
end

function render_arg(::Type{SpecialReg{S}}, slot::Int, ::Bool) where {S}
    return String(S), InputSlot[], slot
end

function render_arg(::Type{Val{V}}, slot::Int, ::Bool) where {V}
    V isa Integer || error("PTX: Val{} arg must be an Integer, got ", V)
    return string(V), InputSlot[], slot
end

function render_arg(::Type{T}, slot::Int, ::Bool) where {T <: Tuple}
    n = fieldcount(T)
    n == 0 && error("PTX: empty tuple arg has no operand mapping")
    types = fieldtypes(T)
    et = first(types)
    all(t -> t === et, types) ||
        error("PTX: heterogeneous tuple args not supported (got $T)")
    letter = constraint_letter(et)
    op_str = "{" * join(("\$" * string(slot + i - 1) for i in 1:n), ", ") * "}"
    slots = InputSlot[(letter, et, i) for i in 1:n]
    op_str, slots, slot + n
end

function render_arg(::Type{T}, slot::Int, ::Bool) where {T}
    op_str = "\$" * string(slot)
    op_str, InputSlot[(constraint_letter(T), T, nothing)], slot + 1
end

build_head(op::Symbol, mods::Tuple{Vararg{Symbol}}) =
    isempty(mods) ? string(op) : string(op) * "." * join(string.(mods), ".")

# Pure: no LLVM, no GPU, no @asmcall. Used by both the runtime call site and
# host-side golden tests.
function build_call(op::Symbol, mods::Tuple{Vararg{Symbol}}, @nospecialize(argtypes))
    rettype = infer_rettype(op, mods)
    nonpure = is_nonpure_opcode(op) || has_special_reg(argtypes)
    bracket = bracket_pointers(op)
    head = build_head(op, mods)

    operand_strs   = String[]
    input_letters  = String[]
    passthrough    = Type[]
    passthrough_ix = Tuple{Int, Union{Nothing, Int}}[]
    slot = (rettype === Nothing) ? 0 : 1     # `$0` reserved for output

    for (i, T) in enumerate(argtypes)
        op_str, slots, slot = render_arg(T, slot, bracket)
        push!(operand_strs, op_str)
        for (letter, atype, lane) in slots
            push!(input_letters, letter)
            push!(passthrough, atype)
            push!(passthrough_ix, (i, lane))
        end
    end

    full_operands = rettype === Nothing ? operand_strs : ["\$0"; operand_strs]
    asm = isempty(full_operands) ? head * ";" : head * " " * join(full_operands, ", ") * ";"

    cparts = String[]
    rettype === Nothing || push!(cparts, "=" * constraint_letter(rettype))
    append!(cparts, input_letters)
    nonpure && push!(cparts, "~{memory}")
    constraints = join(cparts, ",")

    return (; asm, constraints, side_effects = nonpure, rettype,
              passthrough_argtypes = Tuple(passthrough),
              passthrough_indices  = Tuple(passthrough_ix))
end

@generated function (::Operation{op, mods})(args::Vararg{Any,N}) where {op, mods, N}
    spec = build_call(op, mods, args)
    arg_exprs = (
        lane === nothing ? :(args[$i]) : :(args[$i][$lane])
        for (i, lane) in spec.passthrough_indices
    )
    quote
        Base.@inline $(LLVM.Interop).@asmcall(
            $(spec.asm), $(spec.constraints), $(spec.side_effects),
            $(spec.rettype),
            Tuple{$(spec.passthrough_argtypes...)},
            $(arg_exprs...))
    end
end

# Drives byte-exact golden tests.
function format_call(::Operation{op, mods}, @nospecialize(argtypes::Type{<:Tuple})) where {op, mods}
    build_call(op, mods, Tuple(argtypes.parameters)).asm
end
