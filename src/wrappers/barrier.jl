# CTA execution barriers: `bar.{sync,arrive}`, `bar.warp.sync`, and the
# unaligned `barrier.{sync,arrive}` spellings (PTX 9.2 §9.7.12.1) — tier-2
# migration of the chain forms, golden-locked at the asm baseline first
# (test/golden/barrier@sm75.ptx).
#
# Mapping: PTX defines `bar.*` ≡ `barrier.*.aligned` at CTA scope (§9.7.12.1
# — every warp arrives at the same instruction), so `bar.sync`/`bar.arrive`
# ride the `.aligned` intrinsics and the `barrier.*` spellings ride the
# unaligned ones. ISel renders the classic spellings back (`bar.sync`,
# `barrier.sync`, ...) and folds immediate operands that the asm tier's
# "r" constraints had to materialize through registers.
#
# Operands: `Val{N}` (the chain surface's immediate spelling) and runtime
# UInt32/Int32. Wider integers stay on the asm-tier chain fallback
# unchanged — the frozen transpiler emits `ptx"bar.sync"(0)` with Int
# literals, and that path must keep byte-identical behavior.
#
# bar.red/barrier.red are absent: predicate operands were never expressible
# on the asm tier (NVPTX inline asm has no `.pred` constraint), so those
# forms are new surface for a future decision, not migration.
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
    nvvm"barrier.cta.sync.aligned.all"(_barrier_u32(id))
@inline optype"bar.sync"(id::_BarrierOperand,
                                      count::_BarrierOperand) =
    nvvm"barrier.cta.sync.aligned.count"(_barrier_u32(id),
                                         _barrier_u32(count))

# `bar.warp.sync membermask;` — warp-level sync (PTX 6.0).
@inline optype"bar.warp.sync"(mask::_BarrierOperand) =
    nvvm"bar.warp.sync"(_barrier_u32(mask))

# `bar.arrive a, b;` — arrive without waiting; count is mandatory.
@inline optype"bar.arrive"(id::_BarrierOperand,
                                        count::_BarrierOperand) =
    nvvm"barrier.cta.arrive.aligned.count"(_barrier_u32(id),
                                           _barrier_u32(count))

# `barrier.sync a{, b};` — unaligned: threads may arrive from different
# program points, as long as all non-exited threads reach some barrier.
@inline optype"barrier.sync"(id::_BarrierOperand) =
    nvvm"barrier.cta.sync.all"(_barrier_u32(id))
@inline optype"barrier.sync"(id::_BarrierOperand,
                                          count::_BarrierOperand) =
    nvvm"barrier.cta.sync.count"(_barrier_u32(id), _barrier_u32(count))

# `barrier.sync.aligned` — explicit-aligned spelling; same instruction as
# bar.sync, and ISel renders it as such (semantic notation, not WYSIWYG
# text: the emitted spelling is `bar.sync`).
@inline optype"barrier.sync.aligned"(id::_BarrierOperand) =
    nvvm"barrier.cta.sync.aligned.all"(_barrier_u32(id))
@inline optype"barrier.sync.aligned"(id::_BarrierOperand,
                                                   count::_BarrierOperand) =
    nvvm"barrier.cta.sync.aligned.count"(_barrier_u32(id),
                                         _barrier_u32(count))

# `barrier.arrive{.aligned} a, b;`
@inline optype"barrier.arrive"(id::_BarrierOperand,
                                            count::_BarrierOperand) =
    nvvm"barrier.cta.arrive.count"(_barrier_u32(id), _barrier_u32(count))
@inline optype"barrier.arrive.aligned"(id::_BarrierOperand,
                                                     count::_BarrierOperand) =
    nvvm"barrier.cta.arrive.aligned.count"(_barrier_u32(id),
                                           _barrier_u32(count))
