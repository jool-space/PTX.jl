# CTA execution barriers: `bar.{sync,arrive,red}`, `bar.warp.sync`, and the
# `barrier.{sync,arrive,red}{.aligned}` spellings (PTX 9.2 §9.7.12.1) —
# tier-2 migration of the chain forms, golden-locked at the asm baseline
# first (test/golden/barrier@sm75.ptx, test/golden/barrier_red@sm75.ptx).
#
# Mapping: PTX defines `bar.*` ≡ `barrier.*.aligned` at CTA scope (§9.7.12.1
# — every warp arrives at the same instruction), so `bar.sync`/`bar.arrive`
# ride the `.aligned` intrinsics and the `barrier.*` spellings ride the
# unaligned ones. ISel renders the classic spellings back (`bar.sync`,
# `barrier.sync`, ...) and folds immediate operands that the asm tier's
# "r" constraints had to materialize through registers.
#
# Operands: `Val{N}` (the chain surface's immediate spelling) and runtime
# UInt32/Int32, matching the ISA's `.u32` operands. Wider integers (a bare
# `0` literal is Int64) miss these methods and fall through to the generic
# chain, whose 64-bit register ptxas rejects for `bar`; spell immediates
# `Val(0)`. The transpiler emits UInt32 operands.
#
# Reductions take the predicate `c` as a `Bool` (an `i1`, allocated to a
# `.pred` register); the `{!}c` complement is spelled `!c` in Julia. `.popc`
# returns the UInt32 count, `.and`/`.or` return Bool.
#
# Methods are written out literally (no name-building loop) so every
# intrinsic this file stands on is greppable — test/host/conformance.jl
# scans for `nvvm"..."` literals and requires a probe for each.

@inline _barrier_u32(::Val{N}) where {N} = UInt32(N)
@inline _barrier_u32(x::UInt32) = x
@inline _barrier_u32(x::Int32) = reinterpret(UInt32, x)
const _BarrierOperand = Union{Val, UInt32, Int32}

# `bar.sync a{, b};` — aligned CTA execution barrier (+ memory ordering).
@inline optype"bar.sync"(id::_BarrierOperand) =
    ceiled(nvvm"barrier.cta.sync.aligned.all", ptx"bar.sync")(_barrier_u32(id))
@inline optype"bar.sync"(id::_BarrierOperand,
                                      count::_BarrierOperand) =
    ceiled(nvvm"barrier.cta.sync.aligned.count",
           ptx"bar.sync")(_barrier_u32(id), _barrier_u32(count))

# `bar.warp.sync membermask;` — warp-level sync (PTX 6.0).
@inline optype"bar.warp.sync"(mask::_BarrierOperand) =
    ceiled(nvvm"bar.warp.sync", ptx"bar.warp.sync")(_barrier_u32(mask))

# `bar.arrive a, b;` — arrive without waiting; count is mandatory.
@inline optype"bar.arrive"(id::_BarrierOperand,
                                        count::_BarrierOperand) =
    ceiled(nvvm"barrier.cta.arrive.aligned.count",
           ptx"bar.arrive")(_barrier_u32(id), _barrier_u32(count))

# `barrier.sync a{, b};` — unaligned: threads may arrive from different
# program points, as long as all non-exited threads reach some barrier.
@inline optype"barrier.sync"(id::_BarrierOperand) =
    ceiled(nvvm"barrier.cta.sync.all", ptx"barrier.sync")(_barrier_u32(id))
@inline optype"barrier.sync"(id::_BarrierOperand,
                                          count::_BarrierOperand) =
    ceiled(nvvm"barrier.cta.sync.count",
           ptx"barrier.sync")(_barrier_u32(id), _barrier_u32(count))

# `barrier.sync.aligned` — explicit-aligned spelling; same instruction as
# bar.sync, and ISel renders it as such (semantic notation, not WYSIWYG
# text: the emitted spelling is `bar.sync`).
@inline optype"barrier.sync.aligned"(id::_BarrierOperand) =
    ceiled(nvvm"barrier.cta.sync.aligned.all",
           ptx"barrier.sync.aligned")(_barrier_u32(id))
@inline optype"barrier.sync.aligned"(id::_BarrierOperand,
                                                   count::_BarrierOperand) =
    ceiled(nvvm"barrier.cta.sync.aligned.count",
           ptx"barrier.sync.aligned")(_barrier_u32(id), _barrier_u32(count))

# `barrier.arrive{.aligned} a, b;`
@inline optype"barrier.arrive"(id::_BarrierOperand,
                                            count::_BarrierOperand) =
    ceiled(nvvm"barrier.cta.arrive.count",
           ptx"barrier.arrive")(_barrier_u32(id), _barrier_u32(count))
@inline optype"barrier.arrive.aligned"(id::_BarrierOperand,
                                                     count::_BarrierOperand) =
    ceiled(nvvm"barrier.cta.arrive.aligned.count",
           ptx"barrier.arrive.aligned")(_barrier_u32(id), _barrier_u32(count))

# `bar.red.popc.u32 d, a{, b}, {!}c;` — aligned barrier that also counts the
# participating threads whose predicate is true. PTX: `bar.red` ≡
# `barrier.red.aligned`, and mixing `.red` with `.sync`/`.arrive` on the same
# active barrier is unpredictable.
@inline optype"bar.red.popc.u32"(id::_BarrierOperand, c::Bool) =
    ceiled(nvvm"barrier.cta.red.popc.aligned.all",
           ptx"bar.red.popc.u32")(_barrier_u32(id), c)
@inline optype"bar.red.popc.u32"(id::_BarrierOperand,
                                 count::_BarrierOperand, c::Bool) =
    ceiled(nvvm"barrier.cta.red.popc.aligned.count",
           ptx"bar.red.popc.u32")(_barrier_u32(id), _barrier_u32(count), c)

# `bar.red.{and,or}.pred p, a{, b}, {!}c;` — all/any participating thread's
# predicate is true.
@inline optype"bar.red.and.pred"(id::_BarrierOperand, c::Bool) =
    ceiled(nvvm"barrier.cta.red.and.aligned.all",
           ptx"bar.red.and.pred")(_barrier_u32(id), c)
@inline optype"bar.red.and.pred"(id::_BarrierOperand,
                                 count::_BarrierOperand, c::Bool) =
    ceiled(nvvm"barrier.cta.red.and.aligned.count",
           ptx"bar.red.and.pred")(_barrier_u32(id), _barrier_u32(count), c)
@inline optype"bar.red.or.pred"(id::_BarrierOperand, c::Bool) =
    ceiled(nvvm"barrier.cta.red.or.aligned.all",
           ptx"bar.red.or.pred")(_barrier_u32(id), c)
@inline optype"bar.red.or.pred"(id::_BarrierOperand,
                                count::_BarrierOperand, c::Bool) =
    ceiled(nvvm"barrier.cta.red.or.aligned.count",
           ptx"bar.red.or.pred")(_barrier_u32(id), _barrier_u32(count), c)

# `barrier.red.{popc,and,or}` — unaligned reductions.
@inline optype"barrier.red.popc.u32"(id::_BarrierOperand, c::Bool) =
    ceiled(nvvm"barrier.cta.red.popc.all",
           ptx"barrier.red.popc.u32")(_barrier_u32(id), c)
@inline optype"barrier.red.popc.u32"(id::_BarrierOperand,
                                     count::_BarrierOperand, c::Bool) =
    ceiled(nvvm"barrier.cta.red.popc.count",
           ptx"barrier.red.popc.u32")(_barrier_u32(id), _barrier_u32(count), c)
@inline optype"barrier.red.and.pred"(id::_BarrierOperand, c::Bool) =
    ceiled(nvvm"barrier.cta.red.and.all",
           ptx"barrier.red.and.pred")(_barrier_u32(id), c)
@inline optype"barrier.red.and.pred"(id::_BarrierOperand,
                                     count::_BarrierOperand, c::Bool) =
    ceiled(nvvm"barrier.cta.red.and.count",
           ptx"barrier.red.and.pred")(_barrier_u32(id), _barrier_u32(count), c)
@inline optype"barrier.red.or.pred"(id::_BarrierOperand, c::Bool) =
    ceiled(nvvm"barrier.cta.red.or.all",
           ptx"barrier.red.or.pred")(_barrier_u32(id), c)
@inline optype"barrier.red.or.pred"(id::_BarrierOperand,
                                    count::_BarrierOperand, c::Bool) =
    ceiled(nvvm"barrier.cta.red.or.count",
           ptx"barrier.red.or.pred")(_barrier_u32(id), _barrier_u32(count), c)

# `barrier.red.{popc,and,or}.aligned` — same instructions as `bar.red.*`, and
# ISel renders them as such.
@inline optype"barrier.red.popc.aligned.u32"(id::_BarrierOperand, c::Bool) =
    ceiled(nvvm"barrier.cta.red.popc.aligned.all",
           ptx"barrier.red.popc.aligned.u32")(_barrier_u32(id), c)
@inline optype"barrier.red.popc.aligned.u32"(id::_BarrierOperand,
                                             count::_BarrierOperand, c::Bool) =
    ceiled(nvvm"barrier.cta.red.popc.aligned.count",
           ptx"barrier.red.popc.aligned.u32")(_barrier_u32(id),
                                              _barrier_u32(count), c)
@inline optype"barrier.red.and.aligned.pred"(id::_BarrierOperand, c::Bool) =
    ceiled(nvvm"barrier.cta.red.and.aligned.all",
           ptx"barrier.red.and.aligned.pred")(_barrier_u32(id), c)
@inline optype"barrier.red.and.aligned.pred"(id::_BarrierOperand,
                                             count::_BarrierOperand, c::Bool) =
    ceiled(nvvm"barrier.cta.red.and.aligned.count",
           ptx"barrier.red.and.aligned.pred")(_barrier_u32(id),
                                              _barrier_u32(count), c)
@inline optype"barrier.red.or.aligned.pred"(id::_BarrierOperand, c::Bool) =
    ceiled(nvvm"barrier.cta.red.or.aligned.all",
           ptx"barrier.red.or.aligned.pred")(_barrier_u32(id), c)
@inline optype"barrier.red.or.aligned.pred"(id::_BarrierOperand,
                                            count::_BarrierOperand, c::Bool) =
    ceiled(nvvm"barrier.cta.red.or.aligned.count",
           ptx"barrier.red.or.aligned.pred")(_barrier_u32(id),
                                             _barrier_u32(count), c)
