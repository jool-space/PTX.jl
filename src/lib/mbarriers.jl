"""
Verb-style API over the raw `mbarrier.*` wrappers, plus two storage
types: [`BarrierArray`](@ref MBarriers.BarrierArray) (one homogeneous
slot ring) and [`BarrierSet`](@ref MBarriers.BarrierSet) (a typed,
named layout over a whole SMEM mbarrier arena — the shape a
warp-specialized kernel's synchronization plan actually has). Mirrors
`<cuda/barrier>` in scope: init / arrive / arrive_expect_tx / wait /
cluster-arrive, with the arrive variants covering producer (expect_tx)
and consumer (plain arrive) sides.

Sister module `PTX.Pipelines` builds the N-stage producer/consumer
ring on top of these — mirrors `<cuda/pipeline>`.

Contents are plain sm_90 (mbarrier, mapa, fence). Kernels that use
them only become arch-specific via what other ops they pull in
(e.g. wgmma → sm_90a, tcgen05.mma → sm_100a).
"""
module MBarriers

using ..PTX: AS, @ptx_str
using Republic: @public

@public BarrierArray, BarrierSet, BarrierGroup,
        barrier_bytes, barrier_offset, barrierset_init!,
        barrier_init, barrier_arrive, barrier_arrive_expect_tx,
        barrier_wait, barrier_try_wait, barrier_arrive_cluster

"""
    BarrierArray{N}(base::Core.LLVMPtr{UInt64, AS.Shared})

Typed SMEM mbarrier array. `N` is the slot count; `base` points at
the first of N consecutive 8-byte mbarriers. Indexing (0-based stage)
returns the raw shared-AS pointer expected by all mbarrier ops (and by
TMA copies that take a completion mbarrier), so callers can pass
`full[stage]` directly to wrappers without an unwrap step.
"""
struct BarrierArray{N}
    base::Core.LLVMPtr{UInt64, AS.Shared}
end

@inline Base.getindex(ba::BarrierArray, stage::Integer) =
    ba.base + Int(stage) * sizeof(UInt64)

# ── mbarrier verb-style API ────────────────────────────────────────────

"""
    barrier_init(mbar, count)

Initialize an SMEM mbarrier with an arrival `count` —
`mbarrier.init.shared.b64`.
"""
@inline barrier_init(mbar::Core.LLVMPtr{UInt64, AS.Shared}, count::Integer) =
    ptx"mbarrier.init.shared.b64"(mbar, UInt32(count))

"""
    barrier_arrive(mbar)

Consumer-side arrival — `mbarrier.arrive.shared.b64`. Returns the
`UInt64` state token.
"""
@inline barrier_arrive(mbar::Core.LLVMPtr{UInt64, AS.Shared}) =
    ptx"mbarrier.arrive.shared.b64"(mbar)

"""
    barrier_arrive_expect_tx(mbar, tx)

Producer-side arrival declaring `tx` expected transaction bytes —
`mbarrier.arrive.expect_tx.shared.b64`. Pair with TMA copies whose
completion mechanism is `mbarrier::complete_tx::bytes`.
"""
@inline barrier_arrive_expect_tx(mbar::Core.LLVMPtr{UInt64, AS.Shared},
                                  tx::Integer) =
    ptx"mbarrier.arrive.expect_tx.shared.b64"(mbar, UInt32(tx))

"""
    barrier_wait(mbar, phase)

Spin on `mbarrier.test_wait.parity` until the named phase has been
observed. Lowers to a `setp ; @!pred bra` loop — identical to the
hand-rolled `while !test_wait_parity ... end` callers were writing.

`test_wait` vs `try_wait` are NOT interchangeable: test_wait is a
non-suspending poll; try_wait may park the thread until a scheduler
wake. The Blackwell tcgen05 commit-drain idiom is specifically
`try_wait.parity`, and casually swapping the two — or the phase
literal — is a known hang footgun. Hence two distinct verbs
([`barrier_try_wait`](@ref) is the other), not one with a flag: when
porting, pick the one the source kernel uses, verbatim.
"""
@inline function barrier_wait(mbar::Core.LLVMPtr{UInt64, AS.Shared},
                              phase::Integer)
    while !ptx"mbarrier.test_wait.parity.shared.b64"(mbar, UInt32(phase))
    end
    nothing
end

"""
    barrier_try_wait(mbar, phase)

Spin on `mbarrier.try_wait.parity` (the potentially-suspending form)
until `phase` is observed — the tcgen05 `commit → mbarrier::arrive::one`
drain. Same loop shape as [`barrier_wait`](@ref); lowers byte-identically
to the hand-rolled `while !try_wait_parity ... end` the blackwell
kernels were writing.
"""
@inline function barrier_try_wait(mbar::Core.LLVMPtr{UInt64, AS.Shared},
                                  phase::Integer)
    while !ptx"mbarrier.try_wait.parity.shared.b64"(mbar, UInt32(phase))
    end
    nothing
end

"""
    barrier_arrive_cluster(mbar, remote_rank)

Cluster-scope arrive: translate the local mbarrier address through
`mapa.shared::cluster` to the remote CTA's view, then arrive on it.
The mbarrier itself lives in some CTA's SMEM; mapa rebases the SMEM
offset to a cluster-mapped address so the remote CTA's hardware sees
the arrival.
"""
@inline function barrier_arrive_cluster(mbar::Core.LLVMPtr{UInt64, AS.Shared},
                                         remote_rank::Integer)
    remote = ptx"mapa.shared::cluster.u32"(mbar, UInt32(remote_rank))
    ptx"mbarrier.arrive.shared::cluster.b64"(remote)
    nothing
end

# ── BarrierSet: a typed, named SMEM mbarrier arena ─────────────────────

"""
    BarrierSet{SPEC}(base::Core.LLVMPtr{UInt64, AS.Shared})

Typed, named layout over a contiguous SMEM mbarrier arena. `SPEC` is a
NamedTuple **value** type parameter mapping each group name to a
`(shape, count)` pair:

- `shape` — `()` for a single barrier, `(n,)` / `(n, m)` / deeper for
  indexed groups (row-major, 0-based, matching `BarrierArray`);
- `count` — the arrival count [`barrierset_init!`](@ref) initializes
  every barrier of the group with.

Groups are laid out in declaration order, 8 bytes per barrier.
`bars.name` returns the raw shared-AS pointer for a `()`-shaped group
and an indexable [`BarrierGroup`](@ref) otherwise; every offset is a
compile-time constant (the name lookup is a tuple recursion over
`keys(SPEC)` that constant propagation folds to a literal).

```julia
const RING = (full = ((4,), 1), free = ((4,), 1),
              stats = ((2, 2), 128), done = ((), 1))
bars = BarrierSet{RING}(pointer(smem_bars))
barrier_wait(bars.full[slot], phase)
barrier_arrive(bars.stats[stage, parity])
barrier_arrive_expect_tx(bars.done, tx)
```

The arena footprint for the enclosing SMEM map is
`barrier_bytes(typeof(bars))`; per-group offsets are exposed for
host-side layout assertions via [`barrier_offset`](@ref).
"""
struct BarrierSet{SPEC}
    base::Core.LLVMPtr{UInt64, AS.Shared}
end

"""
    BarrierGroup{SHAPE}

Indexed view over one named group of a [`BarrierSet`](@ref). Row-major,
0-based indexing with one index per shape dimension returns the raw
shared-AS pointer, like `BarrierArray`. Obtained via `bars.name`; not
constructed directly.
"""
struct BarrierGroup{SHAPE}
    base::Core.LLVMPtr{UInt64, AS.Shared}
end

@inline _bs_shape_len(shape::Tuple{Vararg{Int}}) =
    isempty(shape) ? 1 : prod(shape)

# Compile-time offset walk: tuple recursion over the SPEC keys, so a
# constant group name folds the whole search to a literal byte offset.
@inline _bs_offset(name::Symbol, ::Tuple{}, spec) =
    error("BarrierSet has no group named ", name)
@inline function _bs_offset(name::Symbol, names::Tuple, spec)
    k = first(names)
    name === k ? 0 :
        8 * _bs_shape_len(getfield(spec, k)[1]) +
            _bs_offset(name, Base.tail(names), spec)
end

@inline function Base.getproperty(bs::BarrierSet{SPEC},
                                  name::Symbol) where {SPEC}
    name === :base && return getfield(bs, :base)
    ptr = getfield(bs, :base) + _bs_offset(name, keys(SPEC), SPEC)
    shape = getfield(SPEC, name)[1]
    shape === () ? ptr : BarrierGroup{shape}(ptr)
end

Base.propertynames(::BarrierSet{SPEC}) where {SPEC} = (keys(SPEC)..., :base)

# Row-major fold; the leading multiply hits the zero accumulator, so each
# index ends up scaled by the product of the dimensions after it.
@inline _bs_rowmajor(acc::Int, ::Tuple{}, ::Tuple{}) = acc
@inline _bs_rowmajor(acc::Int, shape::Tuple, I::Tuple) =
    _bs_rowmajor(acc * first(shape) + Int(first(I)),
                 Base.tail(shape), Base.tail(I))

@inline function Base.getindex(g::BarrierGroup{SHAPE},
                               I::Vararg{Integer}) where {SHAPE}
    length(I) == length(SHAPE) ||
        error("BarrierGroup expects one index per shape dimension")
    getfield(g, :base) + 8 * _bs_rowmajor(0, SHAPE, I)
end

"""
    barrier_bytes(::Type{<:BarrierSet}) -> Int

Total arena footprint in bytes (8 per barrier, groups in declaration
order) — for computing the enclosing SMEM map.
"""
barrier_bytes(::Type{BarrierSet{SPEC}}) where {SPEC} =
    sum(8 * _bs_shape_len(g[1]) for g in values(SPEC); init = 0)

"""
    barrier_offset(::Type{<:BarrierSet}, name::Symbol) -> Int

Byte offset of group `name` from the arena base. Host-side companion to
the device accessors, for layout assertions in tests.
"""
barrier_offset(::Type{BarrierSet{SPEC}}, name::Symbol) where {SPEC} =
    _bs_offset(name, keys(SPEC), SPEC)

"""
    barrierset_init!(bs::BarrierSet)

Initialize every barrier in the arena with its group's declared arrival
count, then publish with `fence.proxy.async.shared::cta`. Same caller
contract as `PTX.Pipelines.pipeline_init!`: invoke from a single
thread and follow with `bar.sync`. Ring self-crediting
pre-arrives remain the caller's job — this initializes counts only.

Body is fully unrolled via `@generated`; the emitted PTX matches the
hand-rolled init sequence byte-for-byte.
"""
@generated function barrierset_init!(bs::BarrierSet{SPEC}) where {SPEC}
    body = Expr(:block)
    push!(body.args, Expr(:meta, :inline))
    off = 0
    for name in keys(SPEC)
        shape, count = getfield(SPEC, name)
        shape isa Tuple{Vararg{Int}} && all(>=(1), shape) ||
            error("BarrierSet group $name: shape must be a tuple of positive Ints")
        count isa Int && count >= 1 ||
            error("BarrierSet group $name: arrival count must be a positive Int")
        for i in 0:(_bs_shape_len(shape) - 1)
            push!(body.args, :(barrier_init(getfield(bs, :base) + $(off + 8i),
                                            UInt32($count))))
        end
        off += 8 * _bs_shape_len(shape)
    end
    push!(body.args, :(ptx"fence.proxy.async.shared::cta"()))
    push!(body.args, :(nothing))
    body
end

end # module MBarriers
