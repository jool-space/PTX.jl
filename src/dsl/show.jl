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
