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

# `ptx"..."raw` — RawOperation escape hatch for eligible unregistered chains
# (static chains only: the raw tier is a deliberate, spelled-out act;
# interpolation belongs to the registered surface). Semantic guards such as
# implicit CC.CF remain fail-loud because conservative attributes cannot model
# their hidden input/output dependency.
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
`%total_smem_size`) round-trip losslessly. The legacy spelling
`%warpsize` is the exception: PTX 9.3 defines `WARP_SZ` as an immediate,
so it returns `Val(32)`. Other names accept either form:

    sreg"tid.x"            ≡ sreg"%tid.x"            → "%tid.x"
    sreg"cluster_ctarank"  ≡ sreg"%cluster_ctarank"  → "%cluster_ctarank"
    sreg"warpsize"         ≡ sreg"%warpsize"         → Val(32)
"""
macro sreg_str(s::String)
    isempty(s) && error("sreg\"\": empty register name")
    name = startswith(s, '%') ? s : '%' * s
    # PTX 9.3 has WARP_SZ (an immediate constant), not %warpsize. Preserve
    # legacy spelling at the public macro boundary while making it valid in
    # every immediate-accepting instruction position.
    name == IR.LEGACY_WARP_SIZE_SREG &&
        return :( Base.Val($(IR.PREDEFINED_IMMEDIATES["WARP_SZ"])) )
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
const InputSlot = Tuple{String, Type, Union{Nothing, Int}, Bool}

function render_arg(::Type{P}, slot::Int, bracket::Bool) where {P <: Core.LLVMPtr}
    op_str = bracket ? "[\$" * string(slot) * "]" : "\$" * string(slot)
    op_str, InputSlot[("l", P, nothing, false)], slot + 1
end

function render_arg(::Type{Address{T}}, slot::Int, ::Bool) where {T}
    is_ptx_address_integer_type(T) || throw(ArgumentError(
        "PTX.Address{$T} is invalid: Address is reserved for 32-/64-bit " *
        "integer carriers; Core.LLVMPtr values remain unwrapped"))
    # An explicit role marker always brackets, independently of the opcode's
    # legacy pointer-wide contract.  Pass the payload type/constraint to LLVM;
    # `_chain_call_expr` unwraps `.value` before the asm call.
    letter = constraint_letter(T)
    "[\$$(slot)]", InputSlot[(letter, T, nothing, true)], slot + 1
end

function render_arg(::Type{SpecialReg{S}}, slot::Int, ::Bool) where {S}
    # `SpecialReg` is intentionally not exported, but direct construction
    # should not resurrect the obsolete `%warpsize` pseudo-register spelling.
    String(S) == IR.LEGACY_WARP_SIZE_SREG &&
        return string(IR.PREDEFINED_IMMEDIATES["WARP_SZ"]), InputSlot[], slot
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
    slots = InputSlot[(letter, et, i, false) for i in 1:n]
    op_str, slots, slot + n
end

function render_arg(::Type{T}, slot::Int, ::Bool) where {T}
    op_str = "\$" * string(slot)
    op_str, InputSlot[(constraint_letter(T), T, nothing, false)], slot + 1
end

build_head(op::Symbol, mods::Tuple{Vararg{Symbol}}) =
    isempty(mods) ? string(op) : string(op) * "." * join(string.(mods), ".")

function _build_structured_result_call(schema::StructuredResultSchema,
                                       @nospecialize(argtypes),
                                       contract::FormContract)
    validate_structured_result_args(schema, argtypes)
    output_types = map(_structured_output_type, schema.outputs)
    rettype = structured_result_type(schema)
    output_operand = length(output_types) == 1 ? "\$0" :
                     join(("\$" * string(i - 1) for i in eachindex(output_types)), "|")
    output_letters = ["=" * constraint_letter(T) for T in output_types]

    operand_strs = String[]
    input_letters = String[]
    passthrough = Type[]
    passthrough_ix = Tuple{Int, Union{Nothing, Int}}[]
    passthrough_unwrap = Bool[]
    slot = length(output_types)
    for (i, T) in enumerate(argtypes)
        # The closed schema knows these are register/immediate operands.
        # RAW_CONTRACT's generic pointer-bracketing guess must not turn an
        # otherwise exact raw setp/lop3/match/elect call into invalid PTX.
        op_str, slots, slot = render_arg(T, slot, false)
        push!(operand_strs, op_str)
        for (letter, atype, lane, unwrap_address) in slots
            push!(input_letters, letter)
            push!(passthrough, atype)
            push!(passthrough_ix, (i, lane))
            push!(passthrough_unwrap, unwrap_address)
        end
    end

    head = build_head(schema.op, schema.ptxmods)
    asm = head * " " * join([output_operand; operand_strs], ", ") * ";"
    nonpure = !contract.pure || has_special_reg(argtypes)
    cparts = [output_letters; input_letters]
    nonpure && push!(cparts, "~{memory}")
    return (; asm, constraints = join(cparts, ","), side_effects = nonpure,
              convergent = contract.convergent, rettype,
              passthrough_argtypes = Tuple(passthrough),
              passthrough_indices = Tuple(passthrough_ix),
              passthrough_unwrap_address = Tuple(passthrough_unwrap))
end

function _build_mbarrier_call(schema::MBarrierFormSchema,
                              @nospecialize(argtypes),
                              contract::FormContract)
    variant = validate_mbarrier_args(schema, argtypes)
    rettype = _mbarrier_rettype(schema)
    output_operands, output_letters, slot =
        schema.destination === :none ? (String[], String[], 0) :
        schema.destination in (:sink, :remote_sink) ? (["_"], String[], 0) :
        schema.destination === :state ? (["\$0"], ["=l"], 1) :
        schema.destination === :predicate ? (["\$0"], ["=b"], 1) :
        schema.destination === :count ? (["\$0"], ["=r"], 1) :
        schema.destination === :report_pred ?
            (["\$0|\$1"], ["=b", "=b"], 2) :
        schema.destination === :report ?
            (["\$0|\$1", "report_value"], ["=b", "=b", "=h"], 3) :
        error("invalid mbarrier destination shape: ", schema.destination)

    operand_strs = String[]
    input_letters = String[]
    passthrough = Type[]
    passthrough_ix = Tuple{Int, Union{Nothing, Int}}[]
    passthrough_unwrap = Bool[]
    for (i, (kind, T)) in enumerate(zip(variant.operands, argtypes))
        op_str, slots, slot = render_arg(T, slot, false)
        # Address already renders its payload in brackets. LLVMPtr and the
        # legacy bare integer carriers need the schema to supply brackets.
        kind === :address && !(T <: Address) &&
            (op_str = "[" * op_str * "]")
        push!(operand_strs, op_str)
        for (letter, atype, lane, unwrap_address) in slots
            # LLVM retains the addrspace(3) pointer value, while NVPTX's `r`
            # inline-asm constraint selects the 32-bit PTX register required
            # for an explicitly shared address. This is the same intentional
            # exception used by the exact mbarrier wrappers; generic addresses
            # retain the ordinary 64-bit `l` pointer constraint.
            kind === :address && T <: Core.LLVMPtr &&
                schema.space !== :generic && (letter = "r")
            push!(input_letters, letter)
            push!(passthrough, atype)
            push!(passthrough_ix, (i, lane))
            push!(passthrough_unwrap, unwrap_address)
        end
    end

    head = build_head(:mbarrier, schema.ptxmods)
    operands = [output_operands; operand_strs]
    asm = isempty(operands) ? head * ";" :
          head * " " * join(operands, ", ") * ";"
    # PTX 9.3's opaque reportValue is a `.b8` destination (the CUDA API
    # describes mbarrier.layout::v1 status as 1-byte wide). NVPTX has no i8
    # inline-asm constraint, so bridge it through the low byte of a UInt16.
    schema.destination === :report &&
        (asm = "{ .reg .b8 report_value; " * asm *
               " mov.b16 \$2, {report_value, 0}; }")
    constraints = join([output_letters; input_letters; "~{memory}"], ",")
    return (; asm, constraints, side_effects = true,
              convergent = contract.convergent, rettype,
              passthrough_argtypes = Tuple(passthrough),
              passthrough_indices = Tuple(passthrough_ix),
              passthrough_unwrap_address = Tuple(passthrough_unwrap))
end

function _build_vector_result_call(schema::VectorResultSchema,
                                   @nospecialize(argtypes),
                                   contract::FormContract;
                                   sink_mask = nothing)
    validate_vector_result_args(schema, argtypes)
    n = schema.form.lanes
    mask = sink_mask === nothing ? ntuple(_ -> true, n) :
           validate_vector_result_mask(schema, sink_mask)
    live = count(identity, mask)
    lane_type = schema.form.lane_type
    rettype = NTuple{live, lane_type}

    output_operands = String[]
    output_letters = String[]
    output_slot = 0
    b8_bridge = schema.form.op === :multimem &&
                schema.form.lane_kind in (:e4m3, :e5m2)
    bridge_index = 0
    for keep in mask
        if keep
            push!(output_operands,
                  b8_bridge ? "vector_result_lane" * string(bridge_index) :
                              "\$" * string(output_slot))
            push!(output_letters, "=" * constraint_letter(lane_type))
            output_slot += 1
        else
            push!(output_operands, "_")
        end
        bridge_index += 1
    end
    destination = "{" * join(output_operands, ", ") * "}"

    operand_strs = String[]
    input_letters = String[]
    passthrough = Type[]
    passthrough_ix = Tuple{Int, Union{Nothing, Int}}[]
    passthrough_unwrap = Bool[]
    slot = live
    roles = something(vector_result_operand_roles(schema, length(argtypes)))
    for (i, (kind, T)) in enumerate(zip(roles, argtypes))
        op_str, slots, slot = render_arg(T, slot, false)
        # Address already renders its payload in brackets. LLVMPtr and the
        # legacy bare integer carriers need the schema to supply brackets.
        kind === :address && !(T <: Address) && (op_str = "[" * op_str * "]")
        push!(operand_strs, op_str)
        for (letter, atype, lane, unwrap_address) in slots
            push!(input_letters, letter)
            push!(passthrough, atype)
            push!(passthrough_ix, (i, lane))
            push!(passthrough_unwrap, unwrap_address)
        end
    end

    head = build_head(schema.form.op, schema.mods)
    asm = head * " " * join([destination; operand_strs], ", ") * ";"
    if b8_bridge
        # PTX FP8 scalar lanes require actual .b8 destination registers;
        # ptxas rejects the .b16 registers selected by LLVM's smallest (`h`)
        # inline-asm class. Keep the public Julia carrier UInt8, materialize
        # each architectural lane in a block-local .b8 register, then bridge
        # its low byte through a legal .b16 output exactly like report-value
        # mbarrier lowering. LLVM truncates the low byte back to i8.
        moves = ["mov.b16 \$$(i - 1), {vector_result_lane$(i - 1), 0};"
                 for i in 1:n]
        asm = "{ .reg .b8 vector_result_lane<$(n)>; " * asm * " " *
              join(moves, " ") * " }"
    end
    # These three families are observable memory operations even when a caller
    # passes a synthetic contract in a host rendering test. The clobber also
    # prevents raw from pretending that the vector result is pure.
    constraints = join([output_letters; input_letters; "~{memory}"], ",")
    return (; asm, constraints, side_effects = true,
              convergent = contract.convergent, rettype,
              passthrough_argtypes = Tuple(passthrough),
              passthrough_indices = Tuple(passthrough_ix),
              passthrough_unwrap_address = Tuple(passthrough_unwrap))
end

function _build_b128_call(schema::B128FormSchema, @nospecialize(argtypes),
                          contract::FormContract)
    validate_b128_form_args(schema, argtypes)
    output_types = schema.result === Nothing ? Type[] :
                   schema.result === B128 ? Type[UInt64, UInt64] :
                   schema.result <: Tuple ? Type[schema.result.parameters...] :
                   Type[schema.result]
    output_letters = ["=" * constraint_letter(T) for T in output_types]
    slot = length(output_types)
    operand_strs = String[]
    input_letters = String[]
    passthrough = Type[]
    passthrough_ix = Tuple{Int, Union{Nothing, Int}}[]
    passthrough_unwrap = Bool[]
    for (i, (kind, T)) in enumerate(zip(schema.operands, argtypes))
        op_str, slots, slot = render_arg(T, slot, false)
        kind === :address && !(T <: Address) && (op_str = "[" * op_str * "]")
        push!(operand_strs, op_str)
        for (letter, atype, lane, unwrap_address) in slots
            push!(input_letters, letter)
            push!(passthrough, atype)
            push!(passthrough_ix, (i, lane))
            push!(passthrough_unwrap, unwrap_address)
        end
    end

    head = build_head(schema.op, schema.mods)
    b128_inputs = findall(==(:b128), schema.operands)
    local_name(i) = "b128_value" * string(i)
    declarations = String[]
    setup = String[]
    for (n, index) in enumerate(b128_inputs)
        push!(declarations, ".reg .b128 " * local_name(n) * ";")
        push!(setup, "mov.b128 " * local_name(n) * ", " * operand_strs[index] * ";")
    end

    instruction = if schema.kind === :mov
        "mov.b128 b128_result, " * local_name(1) * ";"
    elseif schema.kind === :load
        head * " b128_result, " * join(operand_strs, ", ") * ";"
    elseif schema.kind === :store
        args = copy(operand_strs)
        args[findfirst(==(:b128), schema.operands)] = local_name(1)
        head * " " * join(args, ", ") * ";"
    elseif schema.kind in (:exch, :cas)
        args = copy(operand_strs)
        for (n, index) in enumerate(b128_inputs)
            args[index] = local_name(n)
        end
        head * " b128_result, " * join(args, ", ") * ";"
    elseif schema.kind === :query_pred || schema.kind === :query_dim
        head * " \$0, " * local_name(1) * ";"
    elseif schema.kind === :query_v4
        head * " {\$0, \$1, \$2, \$3}, " * local_name(1) * ";"
    else
        error("unknown b128 form kind: ", schema.kind)
    end

    teardown = String[]
    if schema.result === B128
        push!(declarations, ".reg .b128 b128_result;")
        push!(teardown, "mov.b128 {\$0, \$1}, b128_result;")
    end
    asm = isempty(declarations) ? instruction :
          "{ " * join([declarations; setup; instruction; teardown], " ") * " }"
    nonpure = !contract.pure || has_special_reg(argtypes)
    constraints = [output_letters; input_letters]
    nonpure && push!(constraints, "~{memory}")
    (; asm, constraints = join(constraints, ","), side_effects = nonpure,
       convergent = contract.convergent, rettype = schema.result,
       passthrough_argtypes = Tuple(passthrough),
       passthrough_indices = Tuple(passthrough_ix),
       passthrough_unwrap_address = Tuple(passthrough_unwrap))
end

# Pure: no LLVM, no GPU, no @asmcall. Used by both the runtime call site and
# host-side golden tests. `contract` defaults to the registry lookup; pass one
# explicitly for host-side rendering tests, or select the raw tier with the
# distinct `raw=true` signal. Semantic guards are checked first.
function build_call(op::Symbol, mods::Tuple{Vararg{Symbol}}, @nospecialize(argtypes);
                    contract::Union{FormContract, Nothing, Missing} = missing,
                    raw::Bool = false)
    raw && contract !== missing && throw(ArgumentError(
        "PTX.build_call: raw=true selects RAW_CONTRACT internally; " *
        "do not also pass contract=" * repr(contract)))
    immediate = immediate_form_contract(op, mods)
    requires_immediate_form_contract(op) && immediate === nothing &&
        throw(immediate_form_contract_miss(op, mods))
    immediate === nothing || immediate.delegated ||
        validate_immediate_form_args(immediate, argtypes)
    mbarrier = mbarrier_form_schema(op, mods)
    requires_mbarrier_schema(op) && mbarrier === nothing &&
        throw(mbarrier_schema_miss(mods))
    structured = structured_result_schema(op, mods)
    requires_structured_result_schema(op) && structured === nothing &&
        throw(structured_result_schema_miss(op, mods))
    has_invalid_address_marker(argtypes) && throw(ArgumentError(
        "PTX.Address is reserved for Int32, UInt32, Int64, and UInt64; " *
        "Core.LLVMPtr already preserves its address role and exact typed " *
        "dispatch, so call address(pointer) instead of constructing " *
        "Address{<:Core.LLVMPtr}"))
    clc_schema = clc_try_cancel_schema(mods)
    if op === :clusterlaunchcontrol
        clc_schema === nothing && requires_clc_try_cancel_schema(op, mods) &&
            throw(clc_try_cancel_schema_miss(mods))
        clc_schema === nothing || validate_clc_try_cancel_args(clc_schema, argtypes)
    end
    vector = vector_result_schema(op, mods)
    vector === nothing && requires_vector_result_schema(op, mods) &&
        throw(vector_result_schema_miss(op, mods))
    schema = scalar_result_schema(op, mods)
    schema === nothing && requires_scalar_result_schema(op, mods) &&
        throw(scalar_result_schema_miss(op, mods))
    b128_schema = b128_form_schema(op, mods)
    b128_schema === nothing && requires_b128_form_schema(op, mods) &&
        throw(b128_form_schema_miss(op, mods))
    b128_schema === nothing || validate_b128_form_args(b128_schema, argtypes)
    selected_contract = raw ? RAW_CONTRACT :
                        contract === missing ? form_contract(op, mods) : contract
    uses_implicit_cc(op, mods) && throw(ArgumentError(
        "ptx\"$(build_head(op, mods))\" accesses the implicit PTX CC.CF " *
        "flag, which cannot safely cross an LLVM inline-asm call boundary. " *
        "The raw tier does not repair that hidden dependency. Use the typed " *
        "wrapper with explicit Bool carry/borrow, or PTX.add_with_carry, " *
        "PTX.sub_with_borrow, or PTX.mul_wide for a fused operation."))
    rule = typed_wrapper_only_rule(op, mods)
    rule !== nothing && !raw && throw(ArgumentError(
        "ptx\"$(build_head(op, mods))\" requires an exact typed wrapper: " *
        rule.detail * ". No typed method matched this call, and the generic " *
        "scalar chain cannot preserve its operand/result structure. Check the " *
        "modifier spelling, arity, tuple widths, and carrier types, or use " *
        "ptx\"$(build_head(op, mods))\"raw for an explicit conservative " *
        "textual escape hatch."))
    # The closed mbarrier schema is authoritative for its result and operand
    # roles. Only non-mbarrier chain fallbacks reach the generic structured-
    # address and contract checks below.
    if mbarrier !== nothing
        selected_contract === nothing && error(
            "mbarrier is missing its reviewed form contract")
        return _build_mbarrier_call(mbarrier, argtypes, selected_contract)
    end
    b128_schema === nothing ||
        return _build_b128_call(b128_schema, argtypes, selected_contract)
    # The audited vector-result schema is likewise authoritative: it validates
    # and brackets its address operand itself, so an integer Address routed to
    # a reviewed vector form must not hit the exact-wrapper-only fallback rule.
    address_rule = has_integer_address(argtypes) && vector === nothing ?
                   structured_address_fallback_rule(op, mods) : nothing
    address_rule === nothing ||
        throw(structured_address_fallback_error(op, mods, address_rule))
    selected_contract === nothing && error(
        "ptx\"$(build_head(op, mods))\": opcode :$op is not in the form registry " *
        "(src/forms.jl). The chain default makes optimizer promises (purity, " *
        "memory, convergence) that must be reviewed per form — add a registry " *
        "entry, or use ptx\"...\"raw for the maximally-conservative contract " *
        "(sideeffect + memory clobber + convergent; pointer operands bracketed).")
    mbarrier === nothing ||
        return _build_mbarrier_call(mbarrier, argtypes, selected_contract)
    structured === nothing ||
        return _build_structured_result_call(structured, argtypes,
                                             selected_contract)
    vector === nothing ||
        return _build_vector_result_call(vector, argtypes, selected_contract)
    has_explicit_address(argtypes) && !selected_contract.brackets &&
        throw(ArgumentError(
            "ptx\"$(build_head(op, mods))\" has no reviewed bracketed-address " *
             "operand role; PTX.address(...) is only accepted by memory/address " *
             "forms. Passing it here would emit invalid square brackets."))
    schema === nothing || validate_scalar_result_args(schema, argtypes)
    # Immediate side-effect forms have no PTX destination even on the raw
    # tier, whose generic contract otherwise assumes a scalar result.
    rettype = immediate !== nothing && !immediate.returns ? Nothing :
              selected_contract.returns ? infer_rettype(op, mods) : Nothing
    nonpure = !selected_contract.pure || has_special_reg(argtypes)
    bracket = selected_contract.brackets
    head = build_head(op, mods)

    operand_strs   = String[]
    input_letters  = String[]
    passthrough    = Type[]
    passthrough_ix = Tuple{Int, Union{Nothing, Int}}[]
    passthrough_unwrap = Bool[]
    slot = (rettype === Nothing) ? 0 : 1     # `$0` reserved for output

    for (i, T) in enumerate(argtypes)
        op_str, slots, slot = render_arg(T, slot, bracket)
        push!(operand_strs, op_str)
        for (letter, atype, lane, unwrap_address) in slots
            push!(input_letters, letter)
            push!(passthrough, atype)
            push!(passthrough_ix, (i, lane))
            push!(passthrough_unwrap, unwrap_address)
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
              convergent = selected_contract.convergent, rettype,
              passthrough_argtypes = Tuple(passthrough),
              passthrough_indices  = Tuple(passthrough_ix),
              passthrough_unwrap_address = Tuple(passthrough_unwrap))
end

# --- convergent inline asm ---------------------------------------------------
#
# `@asmcall` cannot attach call-site attributes, and `sideeffect` alone does
# NOT forbid duplicating a call site across a divergent branch (jump
# threading, tail duplication) — only `convergent` does. For warp-/warpgroup-
# collective instructions (wgmma.mma_async and mma.sync fallbacks), a split
# call site means different lanes execute different copies: the `active_mask`
# class of miscompile reproduced on hardware. Mbarrier asm uses this path too,
# matching the convergence contract on the complete llvm.nvvm.mbarrier.*
# surface regardless of dispatch tier. The attribute binds in the in-process
# middle end only — llc neither checks nor needs it — so tests assert emitted
# llvmcall IR rather than ptxas acceptance.
#
# Mechanism validated by spikes/raw_asm_attrs.jl: a `convergent` attribute
# group on an inline-asm call site parses through Base.llvmcall and survives
# the optimized module. This helper builds the same shape `@asmcall` would —
# asm callee returns a scalar or literal struct, entry returns Julia's
# homogeneous-tuple `[N x T]` via extract/insertvalue, Bool passes as i8 —
# plus `#0 = { convergent nomerge nounwind }` on the call.
#
# Keep the complete implementation above `_chain_call_expr`: Julia 1.12
# requires every global called by a generated-function generator to exist in
# the generator's definition world, not merely by the time it is invoked.

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
        # Julia represents a homogeneous NTuple as [N x T] and a heterogeneous
        # Tuple as a literal struct. The latter matters for exact-raw mbarrier
        # report forms: (Bool, Bool, UInt16) under RAW_CONTRACT remains
        # convergent without corrupting its return ABI.
        callret = length(comps) == 1 ? comps[1] : "{ " * join(comps, ", ") * " }"
        homogeneous = all(==(first(comps)), comps)
        entryret = homogeneous ? "[$(length(comps)) x $(comps[1])]" :
                   "{ " * join(comps, ", ") * " }"
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

# Shared emission body for Operation (registry contract) and RawOperation
# (RAW_CONTRACT). Convergent forms route through convergent_asm_ir so the
# call site carries `convergent nomerge` — @asmcall cannot attach call-site
# attributes, and `sideeffect` alone permits duplicating a collective op
# across a divergent branch (the activemask miscompile class).

function _chain_call_expr(spec)
    arg_exprs = (
        unwrap_address ? :(getfield(args[$i], :value)) :
        lane === nothing ? :(args[$i]) : :(args[$i][$lane])
        for ((i, lane), unwrap_address) in
            zip(spec.passthrough_indices, spec.passthrough_unwrap_address)
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

"""
    vector_load(ptx"ld....vN.type", address, Val(mask))

Emit a wide PTX vector load with an explicit destination-lane mask. `true`
returns that lane and `false` emits PTX's `_` sink, which guarantees the
corresponding memory location is not read. Sink lanes are supported only for
the ISA's wide `ld.v8.{b32,u32,s32,f32}` and
`ld.v4.{b64,u64,s64,f64}` forms. The result is the homogeneous tuple of live
lanes in source order. All-false masks are rejected: current ptxas cannot
assemble an all-`_` destination, and eliding a load would require a separate
memory-model audit.
"""
@generated function vector_load(::Operation{:ld, mods}, addr::A,
                                ::Val{mask}) where {mods, A, mask}
    schema = vector_result_schema(:ld, mods)
    schema === nothing && throw(vector_result_schema_miss(:ld, mods))
    spec = _build_vector_result_call(schema, (A,), form_contract(:ld, mods);
                                     sink_mask = mask)
    body = _chain_call_expr(spec)
    quote
        Base.@inline
        args = (addr,)
        $body
    end
end


@generated function vector_load(::Operation{:ld, mods}, addr::A, policy::P,
                                ::Val{mask}) where {mods, A, P, mask}
    schema = vector_result_schema(:ld, mods)
    schema === nothing && throw(vector_result_schema_miss(:ld, mods))
    spec = _build_vector_result_call(schema, (A, P), form_contract(:ld, mods);
                                     sink_mask = mask)
    body = _chain_call_expr(spec)
    quote
        Base.@inline
        args = (addr, policy)
        $body
    end
end

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

@generated function (::RawOperation{op, mods})(args::Vararg{Any,N}) where {op, mods, N}
    _chain_call_expr(build_call(op, mods, args; raw = true))
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
function _visit_operation_methods(f)
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
        f(m, op, mods)
    end
end

_visit_operation_mods(f) = _visit_operation_methods((m, op, mods) -> f(op, mods))

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

# Objects print as the literal that reconstructs them — the notation is the
# canonical spelling, not the type parameters (`ptx"mma.sync.aligned"`, not
# `Operation{:mma, (:sync, :aligned)}()`).
_chain_str(op::Symbol, mods::Tuple) = join((string(op), string.(mods)...), ".")
Base.show(io::IO, ::Operation{op, mods}) where {op, mods} =
    print(io, "ptx\"", _chain_str(op, mods), '"')
Base.show(io::IO, ::RawOperation{op, mods}) where {op, mods} =
    print(io, "ptx\"", _chain_str(op, mods), "\"raw")
Base.show(io::IO, ::Chain{mods}) where {mods} =
    print(io, "mod\"", join(string.(mods), "."), '"')

# Drives byte-exact golden tests.
function format_call(::Operation{op, mods}, @nospecialize(argtypes::Type{<:Tuple})) where {op, mods}
    build_call(op, mods, Tuple(argtypes.parameters)).asm
end

# NVVM-backed shortcut for `mov.u32 %reg, %sreg`. Routing through
# `llvm.nvvm.read.ptx.sreg.*` lets LLVM CSE redundant reads and propagate
# range metadata — neither expressible via `@asmcall` + `~{memory}`. Only
# invariant-per-thread sregs are listed; volatile ones (clock, clock64,
# globaltimer, pm0..7, smid, warpid, nsmid, gridid) deliberately fall through
# to the asm path so LLVM doesn't collapse repeated reads. `activemask` is a
# PTX instruction, not a special register.
# This deliberately is not a mirror of the generated backend registry: LLVM
# retains `llvm.nvvm.read.ptx.sreg.warpsize`, while the PTX source surface uses
# WARP_SZ and therefore excludes `%warpsize` here. Names must exist in the
# backend registry (asserted in test/host/inst.jl);
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
    Symbol("%laneid")   => "laneid",
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

# --- Lowering reflection ------------------------------------------------------
#
# `lowering(op, argtypes)` answers "what will this call actually become?"
# without compiling for a device. Chain-vs-wrapper is a dispatch question
# (`which`); wrapper tier is a typed-IR question: tier-2 wrappers reference
# an `NVVM.IntrinsicCall{name}` singleton whose name is a *type parameter*,
# recoverable structurally, and the remaining wrappers split on whether the
# body's llvmcall IR contains an inline-`asm` call (asm tier) or plain
# target-independent instructions (tier-1 core IR).

# The two chain-default methods, captured for identity comparison. `add.f32`
# never gets a wrapper, so `which` resolves both at load time.
const _CHAIN_METHOD     = which(Operation{:add, (:f32,)}(),    (Float32, Float32))
const _RAW_CHAIN_METHOD = which(RawOperation{:add, (:f32,)}(), (Float32, Float32))

function _walk_intrinsics!(names::Vector{String}, @nospecialize(x))
    if x isa NVVM.IntrinsicCall
        push!(names, String(typeof(x).parameters[1]))
    elseif x isa QuoteNode
        _walk_intrinsics!(names, x.value)
    elseif x isa GlobalRef
        if isdefined(x.mod, x.name)
            v = getglobal(x.mod, x.name)
            v isa NVVM.IntrinsicCall &&
                push!(names, String(typeof(v).parameters[1]))
        end
    elseif x isa Expr
        for a in x.args
            _walk_intrinsics!(names, a)
        end
    end
    return nothing
end

"""
    lowering(op::Operation, argtypes) -> NamedTuple

Reflect on how calling `op` with arguments of the given types will lower,
without a device. `argtypes` is a tuple of types or a `Tuple{...}` type.
Returns `(; tier, method, rettype, intrinsics, asm)`:

- `tier = :intrinsic` — a wrapper routes to `llvm.nvvm.*` intrinsics (tier 2);
  their names are in `intrinsics`, and `NVVM.intrinsic(name)` has each record.
- `tier = :core` — a wrapper emits target-independent LLVM IR (tier 1):
  a real `fence`, `load`/`store`, etc. No intrinsic, no asm.
- `tier = :asm` — a hand-written wrapper embeds inline PTX asm (no intrinsic
  or core-IR spelling exists at the pinned backend).
- `tier = :chain_asm` — no wrapper: the chain default renders inline asm
  from the mods and argument types under the form registry's contract
  (`RAW_CONTRACT` for a `RawOperation`); the text is in `asm`.
- `tier = :unregistered` — no wrapper and the opcode is not in the form
  registry: the call errors at the blessing boundary.
- `tier = :forbidden` — the selected generic path is unsafe: a spelling
  accesses implicit architectural state that cannot cross the call boundary,
  a typed-wrapper-only form missed its exact method, or a closed structured-,
  vector-, scalar-result, or immediate grammar island missed its audited ABI
  or constant domain. Explicit raw remains available for the typed-wrapper
  structural case, but not hidden state, an unknown result ABI, or an invalid
  ISA-required immediate.

Binding is not selectability: an `:intrinsic` form can still fail ISel below
its capability floor — that gate lives in the backend, not the registry.
"""
function lowering(o::Union{Operation, RawOperation}, @nospecialize(argtypes))
    argts = argtypes isa Type ? Tuple(argtypes.parameters) : Tuple(argtypes)
    op, mods = typeof(o).parameters
    m = which(o, Tuple{argts...})
    if m === _CHAIN_METHOD || m === _RAW_CHAIN_METHOD
        raw = m === _RAW_CHAIN_METHOD
        has_invalid_address_marker(argts) &&
            return (; tier = :forbidden, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        b128 = b128_form_schema(op, mods)
        address_rule = has_integer_address(argts) &&
                       vector_result_schema(op, mods) === nothing &&
                       b128 === nothing ?
                       structured_address_fallback_rule(op, mods) : nothing
        address_rule === nothing ||
            return (; tier = :forbidden, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        uses_implicit_cc(op, mods) &&
            return (; tier = :forbidden, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        immediate = immediate_form_contract(op, mods)
        immediate === nothing && requires_immediate_form_contract(op) &&
            return (; tier = :forbidden, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        if immediate !== nothing && !immediate.delegated
            try
                validate_immediate_form_args(immediate, argts)
            catch err
                err isa ArgumentError || rethrow()
                return (; tier = :forbidden, method = m, rettype = nothing,
                          intrinsics = String[], asm = nothing)
            end
        end
        if requires_mbarrier_schema(op)
            mbarrier = mbarrier_form_schema(op, mods)
            mbarrier === nothing &&
                return (; tier = :forbidden, method = m, rettype = nothing,
                          intrinsics = String[], asm = nothing)
            try
                validate_mbarrier_args(mbarrier, argts)
            catch err
                err isa ArgumentError || rethrow()
                return (; tier = :forbidden, method = m, rettype = nothing,
                          intrinsics = String[], asm = nothing)
            end
        end
        structured = structured_result_schema(op, mods)
        structured === nothing && requires_structured_result_schema(op) &&
            return (; tier = :forbidden, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        if structured !== nothing
            try
                validate_structured_result_args(structured, argts)
            catch err
                err isa ArgumentError || rethrow()
                return (; tier = :forbidden, method = m, rettype = nothing,
                          intrinsics = String[], asm = nothing)
            end
        end
        vector_result_schema(op, mods) === nothing &&
            requires_vector_result_schema(op, mods) &&
            return (; tier = :forbidden, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        vector = vector_result_schema(op, mods)
        if vector !== nothing
            try
                validate_vector_result_args(vector, argts)
            catch err
                err isa ArgumentError || rethrow()
                return (; tier = :forbidden, method = m, rettype = nothing,
                          intrinsics = String[], asm = nothing)
            end
        end
        b128 === nothing && requires_b128_form_schema(op, mods) &&
            return (; tier = :forbidden, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        if b128 !== nothing
            try
                validate_b128_form_args(b128, argts)
            catch err
                err isa ArgumentError || rethrow()
                return (; tier = :forbidden, method = m, rettype = nothing,
                          intrinsics = String[], asm = nothing)
            end
        end
        !raw && requires_typed_wrapper(op, mods) &&
            return (; tier = :forbidden, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        scalar_result_schema(op, mods) === nothing &&
            requires_scalar_result_schema(op, mods) &&
            return (; tier = :forbidden, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        clc_schema = op === :clusterlaunchcontrol ?
                     clc_try_cancel_schema(mods) : nothing
        if op === :clusterlaunchcontrol
            clc_schema === nothing && requires_clc_try_cancel_schema(op, mods) &&
                return (; tier = :forbidden, method = m, rettype = nothing,
                          intrinsics = String[], asm = nothing)
            if clc_schema !== nothing
                try
                    validate_clc_try_cancel_args(clc_schema, argts)
                catch err
                    err isa ArgumentError || rethrow()
                    return (; tier = :forbidden, method = m, rettype = nothing,
                              intrinsics = String[], asm = nothing)
                end
            end
        end
        schema = scalar_result_schema(op, mods)
        if schema !== nothing
            try
                validate_scalar_result_args(schema, argts)
            catch err
                err isa ArgumentError || rethrow()
                return (; tier = :forbidden, method = m, rettype = nothing,
                          intrinsics = String[], asm = nothing)
            end
        end
        contract = raw ? RAW_CONTRACT : form_contract(op, mods)
        contract === nothing &&
            return (; tier = :unregistered, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        has_explicit_address(argts) && !contract.brackets &&
            return (; tier = :forbidden, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        if b128 === nothing && _result_abi_error(op, mods) !== nothing
            return (; tier = :forbidden, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        end
        # All policy failures have been classified above. Do not catch a broad
        # ArgumentError here: rendering/constraint bugs must remain visible to
        # callers of this introspection API rather than masquerading as policy.
        spec = raw ? build_call(op, mods, argts; raw = true) :
                     build_call(op, mods, argts; contract)
        return (; tier = :chain_asm, method = m, rettype = spec.rettype,
                  intrinsics = String[], asm = spec.asm)
    end
    # Structural pass: tier-2 wrappers reference IntrinsicCall singletons in
    # their unoptimized body — the intrinsic name is a type parameter.
    ci, rettype = only(Base.code_typed(o, argts; optimize = false))
    names = String[]
    for st in ci.code
        _walk_intrinsics!(names, st)
    end
    # Optimized pass: inlining exposes the llvmcall IR text — catches
    # intrinsics declared in literal IR (belt and braces) and, for the
    # remaining wrappers, splits asm tier from tier-1 core IR. (`@asmcall`
    # bodies only surface their asm string after inlining.)
    s = string(first(only(Base.code_typed(o, argts))))
    for match in eachmatch(r"llvm\.nvvm\.[A-Za-z0-9_.]+", s)
        push!(names, match.match)
    end
    unique!(names)
    isempty(names) ||
        return (; tier = :intrinsic, method = m, rettype,
                  intrinsics = names, asm = nothing)
    # `call <ty> asm ...` is the LLVM inline-asm spelling inside llvmcall IR.
    tier = occursin(" asm ", s) ? :asm : :core
    return (; tier, method = m, rettype, intrinsics = names, asm = nothing)
end
