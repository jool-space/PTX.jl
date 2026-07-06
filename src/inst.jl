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
    ptx"mov.u32"(sreg"%tid.x")

    dt = "u32"
    ptx"mov.\$dt"(x)              # ≡ ptx"mov.u32"(x)
    ptx"st.\$(space).b32"(p, v)   # \$(...) for non-identifier exprs

    # Glued — folds to a literal singleton when N is Val-known:
    @inline f(p, ::Val{N}) where {N} =
        ptx"ldmatrix.sync.aligned.m8n8.x\$(N).shared.b16"(p)

Empty literal parts (consecutive `.`, leading/trailing `.`, or empty string)
error at expansion for the static path, and at runtime for the interp path.

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
    return _ptx_build_interp(s)
end

# `ptx"..."raw` — RawOperation escape hatch for unregistered chains (static
# chains only: the raw tier is a deliberate, spelled-out act; interpolation
# belongs to the registered surface).
macro ptx_str(s::String, flag::String)
    flag == "raw" ||
        error("ptx\"...\"$flag: unknown flag (only `raw` is supported)")
    isempty(s) && error("ptx\"\"raw: empty modifier chain")
    occursin('$', s) &&
        error("ptx\"...\"raw: interpolation is not supported on the raw tier")
    raw = split(s, '.')
    any(isempty, raw) && error("ptx\"" * s * "\"raw: empty modifier between '.'")
    op = Symbol(raw[1])
    mods = Tuple(Symbol(p) for p in @view raw[2:end])
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

# Purity / clobber / convergence / bracket classification lives in the form
# registry (src/forms.jl) — one central table, opcode defaults with
# mods-prefix overrides. `build_call` consumes it as a FormContract.

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
# host-side golden tests. `contract` defaults to the registry lookup; pass
# one explicitly to build a spec for an unregistered form (the raw tier,
# host-side rendering tests).
function build_call(op::Symbol, mods::Tuple{Vararg{Symbol}}, @nospecialize(argtypes);
                    contract::Union{FormContract, Nothing} = form_contract(op, mods))
    contract === nothing && error(
        "ptx\"$(build_head(op, mods))\": opcode :$op is not in the form registry " *
        "(src/forms.jl). The chain default makes optimizer promises (purity, " *
        "memory, convergence) that must be reviewed per form — add a registry " *
        "entry, or use ptx\"...\"raw for the maximally-conservative contract " *
        "(sideeffect + memory clobber + convergent; pointer operands bracketed).")
    rettype = contract.returns ? infer_rettype(op, mods) : Nothing
    nonpure = !contract.pure || has_special_reg(argtypes)
    bracket = contract.brackets
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

    return (; asm, constraints, side_effects = nonpure,
              convergent = contract.convergent, rettype,
              passthrough_argtypes = Tuple(passthrough),
              passthrough_indices  = Tuple(passthrough_ix))
end

# Shared emission body for Operation (registry contract) and RawOperation
# (RAW_CONTRACT). Convergent forms route through convergent_asm_ir so the
# call site carries `convergent nomerge` — @asmcall cannot attach call-site
# attributes, and `sideeffect` alone permits duplicating a collective op
# across a divergent branch (the activemask miscompile class).
function _chain_call_expr(spec)
    arg_exprs = (
        lane === nothing ? :(args[$i]) : :(args[$i][$lane])
        for (i, lane) in spec.passthrough_indices
    )
    if spec.convergent
        ir = convergent_asm_ir(spec.asm, spec.constraints, spec.rettype,
                               spec.passthrough_argtypes)
        quote
            Base.@inline
            Base.llvmcall(($ir, "entry"), $(spec.rettype),
                          Tuple{$(spec.passthrough_argtypes...)},
                          $(arg_exprs...))
        end
    else
        quote
            Base.@inline $(LLVM.Interop).@asmcall(
                $(spec.asm), $(spec.constraints), $(spec.side_effects),
                $(spec.rettype),
                Tuple{$(spec.passthrough_argtypes...)},
                $(arg_exprs...))
        end
    end
end

@generated function (::Operation{op, mods})(args::Vararg{Any,N}) where {op, mods, N}
    _chain_call_expr(build_call(op, mods, args))
end

# --- The raw tier -------------------------------------------------------------
#
# `ptx"..."raw` — the explicit opt-in for chains the form registry doesn't
# know (DESIGN.md, "A blessing boundary"). Same rendering machinery as the
# registered chain, but under RAW_CONTRACT: sideeffect + memory clobber +
# convergent, pointer operands bracketed, trailing-dtype return inference
# (wrong guesses die loudly in ptxas, never silently in the optimizer).
# Composition mirrors Operation so a raw chain can be extended with `*`.

struct RawOperation{op, mods} end

Base.:*(::RawOperation{op, M1}, ::Chain{M2}) where {op, M1, M2} =
    RawOperation{op, (M1..., M2...)}()
Base.:*(::RawOperation{op, M},  s::Symbol) where {op, M} =
    RawOperation{op, (M..., s)}()

@generated function (::RawOperation{op, mods})(args::Vararg{Any,N}) where {op, mods, N}
    _chain_call_expr(build_call(op, mods, args; contract = RAW_CONTRACT))
end

# --- Property notation: composition + completion ------------------------------
#
# `ptx"cvt".rn.f32.f16` ≡ ptx"cvt.rn.f32.f16" — property access composes in
# the type domain exactly like `*`, so a literal dot chain folds to the same
# singleton (device-safe). Segments that aren't valid identifiers spell as
# `var"..."`: `ptx"st".var"shared::cta".b32`, `ptx"cp".async.bulk.tensor.var"3d"`.
#
# `propertynames` makes the notation explorable: REPL tab completion on a
# chain value lists known continuations, drawn from the form registry's
# override prefixes plus the NVVM intrinsic name table (best-effort — the
# intrinsic spelling tracks the PTX chain for most families; absence of a
# suggestion never blocks composition).
@inline Base.getproperty(o::Operation, s::Symbol)    = o * s
@inline Base.getproperty(o::RawOperation, s::Symbol) = o * s
@inline Base.getproperty(c::Chain, s::Symbol)        = c * s

# The wrapped surface is enumerable from the method table: every wrapper
# form is a `(::Operation{op, mods})(...)` method whose mods tuple is
# spelled in ISA vocabulary by construction (that's what dispatches). This
# is the ONLY sound completion source besides the registry — NVVM intrinsic
# names are NOT one: their grammar diverges from the ISA chain exactly
# where a family is irregular (llvm.nvvm.mma.* drops `.sync.aligned` and
# leads with the shape, so name-derived suggestions offered segments that
# are invalid at that chain position while omitting `sync`, the only
# valid one).
function _visit_operation_mods(f)
    opname = Base.unwrap_unionall(Operation).name
    mt = @static if isdefined(Core, :methodtable)
        Core.methodtable   # 1.12+: unified global method table
    else
        opname.mt          # ≤ 1.11: per-type method table
    end
    Base.visit(mt) do m
        sig = try Base.unwrap_unionall(m.sig) catch; return end
        sig isa DataType || return
        isempty(sig.parameters) && return
        p1 = sig.parameters[1]
        p1 isa DataType || return
        p1.name === opname || return
        length(p1.parameters) == 2 || return
        op, mods = p1.parameters
        (op isa Symbol && mods isa Tuple &&
         all(s -> s isa Symbol, mods)) || return
        f(op, mods)
    end
end

function _next_segments(op::Symbol, mods::Tuple{Vararg{Symbol}})
    segs = Set{Symbol}()
    fam = get(FORMS, op, nothing)
    if fam !== nothing
        for (prefix, _) in fam.overrides
            length(prefix) > length(mods) || continue
            all(i -> prefix[i] === mods[i], eachindex(mods)) || continue
            push!(segs, prefix[length(mods) + 1])
        end
    end
    _visit_operation_mods() do mop, mmods
        mop === op || return
        length(mmods) > length(mods) || return
        all(i -> mmods[i] === mods[i], eachindex(mods)) || return
        push!(segs, mmods[length(mods) + 1])
    end
    sort!(collect(segs))
end

Base.propertynames(::Operation{op, mods}, ::Bool = false) where {op, mods} =
    Tuple(_next_segments(op, mods))
Base.propertynames(::RawOperation{op, mods}, ::Bool = false) where {op, mods} =
    Tuple(_next_segments(op, mods))

# Drives byte-exact golden tests.
function format_call(::Operation{op, mods}, @nospecialize(argtypes::Type{<:Tuple})) where {op, mods}
    build_call(op, mods, Tuple(argtypes.parameters)).asm
end

# --- convergent inline asm (warp-collective asm-tier forms) ------------------
#
# `@asmcall` cannot attach call-site attributes, and `sideeffect` alone does
# NOT forbid duplicating a call site across a divergent branch (jump
# threading, tail duplication) — only `convergent` does. For warp-/warpgroup-
# collective instructions (wgmma.mma_async, the mma.sync asm fallbacks) a
# split call site means different lanes execute different copies of a
# collective op: the `active_mask` class of miscompile the convergence spike
# reproduced on hardware. The attribute binds in the in-process middle end
# only — llc neither checks nor needs it — so it is asserted by tests on the
# emitted llvmcall IR, not by ptxas acceptance (see CONCERNS.md, "Convergence
# on the asm tier").
#
# Mechanism validated by spikes/raw_asm_attrs.jl: a `convergent` attribute
# group on an inline-asm call site parses through Base.llvmcall and survives
# the optimized module. This helper builds the same shape `@asmcall` would —
# asm callee returns a scalar or literal struct, entry returns Julia's
# homogeneous-tuple `[N x T]` via extract/insertvalue, Bool passes as i8 —
# plus `#0 = { convergent nounwind }` on the call.

_asm_lltype(T::Type) =
    T === Float32 ? "float" :
    T === UInt32  ? "i32"   :
    T === Int32   ? "i32"   :
    T === UInt64  ? "i64"   :
    T === Int64   ? "i64"   :
    T === Float64 ? "double" :
    T === Float16 ? "half"  :
    T === UInt16  ? "i16"   :
    T === Int16   ? "i16"   :
    T === UInt8   ? "i8"    :
    T === Int8    ? "i8"    :
    T === Bool    ? "i8"    :
    # Pointers pass straight into the asm operand (what @asmcall does via
    # the builder API); `i8` pointee to match Julia's LLVMPtr lowering,
    # typed spelling for the Julia ≤ 1.11 device context (NVVM.llvmtype).
    T <: Core.LLVMPtr ? (T.parameters[2] == 0 ? "i8*" :
                         "i8 addrspace($(T.parameters[2]))*") :
    error("convergent_asm_ir: no LLVM mapping for $T")

function convergent_asm_ir(asm::String, constraints::String,
                           rettype::Type, argtypes)::String
    params = ["$(_asm_lltype(T)) %a$(k - 1)" for (k, T) in enumerate(argtypes)]
    callargs = join(("$(_asm_lltype(T)) %a$(k - 1)"
                     for (k, T) in enumerate(argtypes)), ", ")
    asmcall(ret) = "call $ret asm sideeffect \"$asm\", \"$constraints\"($callargs) #0"
    # `nomerge` alongside `convergent`: LLVM ≤ 16 (Julia ≤ 1.11) hoists
    # identical convergent calls from both arms of a divergent branch into
    # one site — the collective-op miscompile. See NVVM.fnattrs.

    body = String[]
    if rettype === Nothing
        entryret = "void"
        push!(body, "  " * asmcall("void"))
        push!(body, "  ret void")
    elseif rettype <: Tuple
        comps = _asm_lltype.(collect(rettype.parameters))
        # asm returns a scalar (1 output) or a literal struct (N>=2 outputs);
        # Julia represents the homogeneous NTuple as [N x T].
        callret = length(comps) == 1 ? comps[1] : "{ " * join(comps, ", ") * " }"
        entryret = "[$(length(comps)) x $(comps[1])]"
        push!(body, "  %r = " * asmcall(callret))
        prev = "undef"
        for k in 1:length(comps)
            v = length(comps) == 1 ? "%r" : "%e$k"
            length(comps) == 1 ||
                push!(body, "  %e$k = extractvalue $callret %r, $(k - 1)")
            push!(body, "  %t$k = insertvalue $entryret $prev, $(comps[k]) $v, $(k - 1)")
            prev = "%t$k"
        end
        push!(body, "  ret $entryret $prev")
    else
        entryret = _asm_lltype(rettype)
        push!(body, "  %r = " * asmcall(entryret))
        push!(body, "  ret $entryret %r")
    end

    """
    define $entryret @entry($(join(params, ", "))) #1 {
    $(join(body, "\n"))
    }
    attributes #0 = { convergent nomerge nounwind }
    attributes #1 = { alwaysinline }
    """
end

# NVVM-backed shortcut for `mov.u32 %reg, %sreg`. Routing through
# `llvm.nvvm.read.ptx.sreg.*` lets LLVM CSE redundant reads and propagate
# range metadata — neither expressible via `@asmcall` + `~{memory}`. Only
# invariant-per-thread sregs are listed; volatile ones (clock, clock64,
# globaltimer, activemask, pm0..7, smid, warpid, nsmid, gridid) deliberately
# fall through to the asm path so LLVM doesn't collapse repeated reads.
# Names must exist in the backend registry (asserted in test/host/inst.jl);
# the IN-PROCESS LLVM need not know them — emission goes through the tier-2
# IntrinsicCall (declare+call), not `ccall(name, llvmcall, ...)`, precisely
# because ccall resolves against the in-process intrinsic table and demotes
# unknown names (the cluster sregs on LLVM ≤ 16 / Julia ≤ 1.11) to a
# runtime trap.
const NVVM_SREG_U32 = Dict{Symbol, String}(
    Symbol("%tid.x")    => "tid.x",    Symbol("%tid.y")    => "tid.y",    Symbol("%tid.z")    => "tid.z",
    Symbol("%ntid.x")   => "ntid.x",   Symbol("%ntid.y")   => "ntid.y",   Symbol("%ntid.z")   => "ntid.z",
    Symbol("%ctaid.x")  => "ctaid.x",  Symbol("%ctaid.y")  => "ctaid.y",  Symbol("%ctaid.z")  => "ctaid.z",
    Symbol("%nctaid.x") => "nctaid.x", Symbol("%nctaid.y") => "nctaid.y", Symbol("%nctaid.z") => "nctaid.z",
    Symbol("%laneid")   => "laneid",   Symbol("%warpsize") => "warpsize",
    Symbol("%lanemask_eq") => "lanemask.eq", Symbol("%lanemask_lt") => "lanemask.lt",
    Symbol("%lanemask_le") => "lanemask.le", Symbol("%lanemask_ge") => "lanemask.ge",
    Symbol("%lanemask_gt") => "lanemask.gt",
    Symbol("%cluster_ctaid.x")  => "cluster.ctaid.x",
    Symbol("%cluster_ctaid.y")  => "cluster.ctaid.y",
    Symbol("%cluster_ctaid.z")  => "cluster.ctaid.z",
    Symbol("%cluster_nctaid.x") => "cluster.nctaid.x",
    Symbol("%cluster_nctaid.y") => "cluster.nctaid.y",
    Symbol("%cluster_nctaid.z") => "cluster.nctaid.z",
    Symbol("%cluster_ctarank")  => "cluster.ctarank",
    Symbol("%cluster_nctarank") => "cluster.nctarank",
    Symbol("%clusterid.x")  => "clusterid.x",
    Symbol("%clusterid.y")  => "clusterid.y",
    Symbol("%clusterid.z")  => "clusterid.z",
    Symbol("%nclusterid.x") => "nclusterid.x",
    Symbol("%nclusterid.y") => "nclusterid.y",
    Symbol("%nclusterid.z") => "nclusterid.z",
)

@generated function (::Operation{:mov, (:u32,)})(::SpecialReg{S}) where {S}
    suffix = get(NVVM_SREG_U32, S, nothing)
    if suffix !== nothing
        intr = "llvm.nvvm.read.ptx.sreg." * suffix
        return :( $(NVVM.IntrinsicCall{Symbol(intr)}())() )
    end
    spec = build_call(:mov, (:u32,), (SpecialReg{S},))
    quote
        Base.@inline $(LLVM.Interop).@asmcall(
            $(spec.asm), $(spec.constraints), $(spec.side_effects),
            $(spec.rettype),
            Tuple{$(spec.passthrough_argtypes...)})
    end
end
