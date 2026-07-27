# `cp.async.bulk.tensor.<N>d.*` — TMA tile copies (Hopper sm_90+), third
# family migrated to tier-2 intrinsic lowering.
# The notation surface is unchanged; the `[base, {coords}]` operand encoding,
# qualifier rendering, and per-arity constraint strings the asm tier built by
# hand are now ISel's job.
#
# Intrinsic mapping:
#   - `shared::cluster` loads (plain / multicast / cta_group::2) →
#     `g2s.tile.<N>d`. The intrinsic's destination is `ptr addrspace(7)`;
#     the wrapper keeps the package's AS.Shared surface and retypes the raw
#     value (`reinterpret_addrspace`) — the ISA defines `shared::cta`
#     addresses as valid `shared::cluster` addresses, which is exactly what
#     the asm tier's `r` constraint relied on.
#   - `shared::cta` loads → `g2s.cta.tile.<N>d`. ISel requires PTX 8.6 —
#     the same floor ptxas already imposed on the asm spelling.
#   - stores → `s2g.tile.<N>d` (selects the identical
#     `...global.shared::cta.tile.bulk_group` instruction at sm_90/ptx80).
#   - The tensormap operand is `ptr` (generic) in all of them; the package's
#     TMADescriptorPtr is a *global* address typed AS.Const by convention,
#     so it is retyped raw, never addrspacecast (see reinterpret_addrspace).
#
# Optional-operand scheme: the intrinsics carry every optional qualifier as
# an (operand, i1 immarg flag) pair — multicast mask + flag, cache hint +
# flag — plus an i32 `cta_group` immarg (0 = none, 1/2 = `.cta_group::<n>`,
# sm_100a). A false flag drops the qualifier and ignores the operand. Tile
# prefetch exposes cache hints explicitly; load/store forms still pass
# (UInt64(0), Val(false)).
#
# Notation is non-WYSIWYG in one spot: ISel renders `.cta_group::2` after
# `.mbarrier::complete_tx::bytes`, where the asm tier (pyptx order) placed
# it after `.<N>d`. ptxas accepts both; the goldens pin the new spelling.
#
# Residue: `shared::cta` × `cta_group::2` stays on the asm tier — no NVVM
# intrinsic carries both (`g2s.cta` has no cta_group operand, `g2s` renders
# `shared::cluster`).
#
# Methods are written out literally (no name-building loop) so every
# intrinsic this file stands on is greppable — test/host/conformance.jl
# scans for `nvvm"..."` literals and requires a probe for each.

# The two raw retypes at the wrapper boundary (see address_space.jl).
@inline _tma_tmap(p::Core.LLVMPtr) = reinterpret_addrspace(Val(AS.Generic), p)
@inline _tma_cluster(p::Core.LLVMPtr) =
    reinterpret_addrspace(Val(AS.SharedCluster), p)

# --- Loads: shared::cluster destination --------------------------------------

@inline function optype"cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, mbar::Core.LLVMPtr{U, AS.Shared}) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.tile.1d"(
        _tma_cluster(dst), mbar, _tma_tmap(tmap), Int32(c1),
        UInt16(0), UInt64(0), Val(false), Val(false), Val(0))
end

@inline function optype"cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.tile.2d"(
        _tma_cluster(dst), mbar, _tma_tmap(tmap), Int32(c1), Int32(c2),
        UInt16(0), UInt64(0), Val(false), Val(false), Val(0))
end

@inline function optype"cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer, c3::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.tile.3d"(
        _tma_cluster(dst), mbar, _tma_tmap(tmap),
        Int32(c1), Int32(c2), Int32(c3),
        UInt16(0), UInt64(0), Val(false), Val(false), Val(0))
end

@inline function optype"cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer, c3::Integer, c4::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.tile.4d"(
        _tma_cluster(dst), mbar, _tma_tmap(tmap),
        Int32(c1), Int32(c2), Int32(c3), Int32(c4),
        UInt16(0), UInt64(0), Val(false), Val(false), Val(0))
end

@inline function optype"cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer, c3::Integer, c4::Integer, c5::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.tile.5d"(
        _tma_cluster(dst), mbar, _tma_tmap(tmap),
        Int32(c1), Int32(c2), Int32(c3), Int32(c4), Int32(c5),
        UInt16(0), UInt64(0), Val(false), Val(false), Val(0))
end

# --- Loads: shared::cluster + multicast::cluster ------------------------------
# One global read lands in every CTA whose bit is set in the u16 mask.

@inline function optype"cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, mbar::Core.LLVMPtr{U, AS.Shared},
        mask::Integer) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.tile.1d"(
        _tma_cluster(dst), mbar, _tma_tmap(tmap), Int32(c1),
        UInt16(mask), UInt64(0), Val(true), Val(false), Val(0))
end

@inline function optype"cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer, mbar::Core.LLVMPtr{U, AS.Shared},
        mask::Integer) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.tile.2d"(
        _tma_cluster(dst), mbar, _tma_tmap(tmap), Int32(c1), Int32(c2),
        UInt16(mask), UInt64(0), Val(true), Val(false), Val(0))
end

@inline function optype"cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer, c3::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}, mask::Integer) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.tile.3d"(
        _tma_cluster(dst), mbar, _tma_tmap(tmap),
        Int32(c1), Int32(c2), Int32(c3),
        UInt16(mask), UInt64(0), Val(true), Val(false), Val(0))
end

@inline function optype"cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer, c3::Integer, c4::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}, mask::Integer) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.tile.4d"(
        _tma_cluster(dst), mbar, _tma_tmap(tmap),
        Int32(c1), Int32(c2), Int32(c3), Int32(c4),
        UInt16(mask), UInt64(0), Val(true), Val(false), Val(0))
end

@inline function optype"cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer, c3::Integer, c4::Integer, c5::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}, mask::Integer) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.tile.5d"(
        _tma_cluster(dst), mbar, _tma_tmap(tmap),
        Int32(c1), Int32(c2), Int32(c3), Int32(c4), Int32(c5),
        UInt16(mask), UInt64(0), Val(true), Val(false), Val(0))
end

# --- Loads: shared::cluster + cta_group::2 (Blackwell 2-SM, sm_100a) ----------
# Both cluster CTAs issue the same instruction; hardware splits the read.

@inline function optype"cp.async.bulk.tensor.1d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, mbar::Core.LLVMPtr{U, AS.Shared}) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.tile.1d"(
        _tma_cluster(dst), mbar, _tma_tmap(tmap), Int32(c1),
        UInt16(0), UInt64(0), Val(false), Val(false), Val(2))
end

@inline function optype"cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.tile.2d"(
        _tma_cluster(dst), mbar, _tma_tmap(tmap), Int32(c1), Int32(c2),
        UInt16(0), UInt64(0), Val(false), Val(false), Val(2))
end

@inline function optype"cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer, c3::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.tile.3d"(
        _tma_cluster(dst), mbar, _tma_tmap(tmap),
        Int32(c1), Int32(c2), Int32(c3),
        UInt16(0), UInt64(0), Val(false), Val(false), Val(2))
end

@inline function optype"cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer, c3::Integer, c4::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.tile.4d"(
        _tma_cluster(dst), mbar, _tma_tmap(tmap),
        Int32(c1), Int32(c2), Int32(c3), Int32(c4),
        UInt16(0), UInt64(0), Val(false), Val(false), Val(2))
end

@inline function optype"cp.async.bulk.tensor.5d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer, c3::Integer, c4::Integer, c5::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.tile.5d"(
        _tma_cluster(dst), mbar, _tma_tmap(tmap),
        Int32(c1), Int32(c2), Int32(c3), Int32(c4), Int32(c5),
        UInt16(0), UInt64(0), Val(false), Val(false), Val(2))
end

# --- Loads: shared::cluster + cta_group::2 + multicast::cluster ---------------

@inline function optype"cp.async.bulk.tensor.1d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, mbar::Core.LLVMPtr{U, AS.Shared},
        mask::Integer) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.tile.1d"(
        _tma_cluster(dst), mbar, _tma_tmap(tmap), Int32(c1),
        UInt16(mask), UInt64(0), Val(true), Val(false), Val(2))
end

@inline function optype"cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer, mbar::Core.LLVMPtr{U, AS.Shared},
        mask::Integer) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.tile.2d"(
        _tma_cluster(dst), mbar, _tma_tmap(tmap), Int32(c1), Int32(c2),
        UInt16(mask), UInt64(0), Val(true), Val(false), Val(2))
end

@inline function optype"cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer, c3::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}, mask::Integer) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.tile.3d"(
        _tma_cluster(dst), mbar, _tma_tmap(tmap),
        Int32(c1), Int32(c2), Int32(c3),
        UInt16(mask), UInt64(0), Val(true), Val(false), Val(2))
end

@inline function optype"cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer, c3::Integer, c4::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}, mask::Integer) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.tile.4d"(
        _tma_cluster(dst), mbar, _tma_tmap(tmap),
        Int32(c1), Int32(c2), Int32(c3), Int32(c4),
        UInt16(mask), UInt64(0), Val(true), Val(false), Val(2))
end

@inline function optype"cp.async.bulk.tensor.5d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer, c3::Integer, c4::Integer, c5::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}, mask::Integer) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.tile.5d"(
        _tma_cluster(dst), mbar, _tma_tmap(tmap),
        Int32(c1), Int32(c2), Int32(c3), Int32(c4), Int32(c5),
        UInt16(mask), UInt64(0), Val(true), Val(false), Val(2))
end

# --- Loads: shared::cta destination (PTX 8.6) ---------------------------------

@inline function optype"cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, mbar::Core.LLVMPtr{U, AS.Shared}) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.cta.tile.1d"(
        dst, mbar, _tma_tmap(tmap), Int32(c1), UInt64(0), Val(false))
end

@inline function optype"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.cta.tile.2d"(
        dst, mbar, _tma_tmap(tmap), Int32(c1), Int32(c2),
        UInt64(0), Val(false))
end

@inline function optype"cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer, c3::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.cta.tile.3d"(
        dst, mbar, _tma_tmap(tmap), Int32(c1), Int32(c2), Int32(c3),
        UInt64(0), Val(false))
end

@inline function optype"cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer, c3::Integer, c4::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.cta.tile.4d"(
        dst, mbar, _tma_tmap(tmap),
        Int32(c1), Int32(c2), Int32(c3), Int32(c4),
        UInt64(0), Val(false))
end

@inline function optype"cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer, c3::Integer, c4::Integer, c5::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}) where {T, S, U}
    nvvm"cp.async.bulk.tensor.g2s.cta.tile.5d"(
        dst, mbar, _tma_tmap(tmap),
        Int32(c1), Int32(c2), Int32(c3), Int32(c4), Int32(c5),
        UInt64(0), Val(false))
end

# --- Stores: shared::cta → global, bulk_group completion ----------------------

@inline function optype"cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group"(
        tmap::Core.LLVMPtr{S, AS.Const}, c1::Integer,
        src::Core.LLVMPtr{T, AS.Shared}) where {S, T}
    nvvm"cp.async.bulk.tensor.s2g.tile.1d"(
        src, _tma_tmap(tmap), Int32(c1), UInt64(0), Val(false))
end

@inline function optype"cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"(
        tmap::Core.LLVMPtr{S, AS.Const}, c1::Integer, c2::Integer,
        src::Core.LLVMPtr{T, AS.Shared}) where {S, T}
    nvvm"cp.async.bulk.tensor.s2g.tile.2d"(
        src, _tma_tmap(tmap), Int32(c1), Int32(c2), UInt64(0), Val(false))
end

@inline function optype"cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"(
        tmap::Core.LLVMPtr{S, AS.Const}, c1::Integer, c2::Integer,
        c3::Integer, src::Core.LLVMPtr{T, AS.Shared}) where {S, T}
    nvvm"cp.async.bulk.tensor.s2g.tile.3d"(
        src, _tma_tmap(tmap), Int32(c1), Int32(c2), Int32(c3),
        UInt64(0), Val(false))
end

@inline function optype"cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"(
        tmap::Core.LLVMPtr{S, AS.Const}, c1::Integer, c2::Integer,
        c3::Integer, c4::Integer,
        src::Core.LLVMPtr{T, AS.Shared}) where {S, T}
    nvvm"cp.async.bulk.tensor.s2g.tile.4d"(
        src, _tma_tmap(tmap), Int32(c1), Int32(c2), Int32(c3), Int32(c4),
        UInt64(0), Val(false))
end

@inline function optype"cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group"(
        tmap::Core.LLVMPtr{S, AS.Const}, c1::Integer, c2::Integer,
        c3::Integer, c4::Integer, c5::Integer,
        src::Core.LLVMPtr{T, AS.Shared}) where {S, T}
    nvvm"cp.async.bulk.tensor.s2g.tile.5d"(
        src, _tma_tmap(tmap), Int32(c1), Int32(c2), Int32(c3), Int32(c4),
        Int32(c5), UInt64(0), Val(false))
end

# --- Prefetch: global → L2 through a tensor map -------------------------------
# No destination operand and no completion mechanism — fire-and-forget L2
# warming. This is the instruction CUTLASS's weight-prefetch mainloop
# (examples/63) stands on: a dedicated warp walks the weight tensor's
# K-tiles ahead of the TMA loads that will actually consume them.
# PTX imposes no collective participation rule. The NVVM intrinsic records
# are nevertheless `convergent`, so tier-2 emission preserves that conservative
# optimizer boundary without claiming warp-cooperative semantics.

@inline function optype"cp.async.bulk.prefetch.tensor.1d.L2.global.tile"(
        tmap::Core.LLVMPtr{S, AS.Const}, c1::Integer) where {S}
    nvvm"cp.async.bulk.tensor.prefetch.tile.1d"(
        _tma_tmap(tmap), Int32(c1), UInt64(0), Val(false))
end

@inline function optype"cp.async.bulk.prefetch.tensor.2d.L2.global.tile"(
        tmap::Core.LLVMPtr{S, AS.Const}, c1::Integer, c2::Integer) where {S}
    nvvm"cp.async.bulk.tensor.prefetch.tile.2d"(
        _tma_tmap(tmap), Int32(c1), Int32(c2), UInt64(0), Val(false))
end

@inline function optype"cp.async.bulk.prefetch.tensor.3d.L2.global.tile"(
        tmap::Core.LLVMPtr{S, AS.Const}, c1::Integer, c2::Integer,
        c3::Integer) where {S}
    nvvm"cp.async.bulk.tensor.prefetch.tile.3d"(
        _tma_tmap(tmap), Int32(c1), Int32(c2), Int32(c3),
        UInt64(0), Val(false))
end

@inline function optype"cp.async.bulk.prefetch.tensor.4d.L2.global.tile"(
        tmap::Core.LLVMPtr{S, AS.Const}, c1::Integer, c2::Integer,
        c3::Integer, c4::Integer) where {S}
    nvvm"cp.async.bulk.tensor.prefetch.tile.4d"(
        _tma_tmap(tmap), Int32(c1), Int32(c2), Int32(c3), Int32(c4),
        UInt64(0), Val(false))
end

@inline function optype"cp.async.bulk.prefetch.tensor.5d.L2.global.tile"(
        tmap::Core.LLVMPtr{S, AS.Const}, c1::Integer, c2::Integer,
        c3::Integer, c4::Integer, c5::Integer) where {S}
    nvvm"cp.async.bulk.tensor.prefetch.tile.5d"(
        _tma_tmap(tmap), Int32(c1), Int32(c2), Int32(c3), Int32(c4),
        Int32(c5), UInt64(0), Val(false))
end

# PTX requires the qualifier and u64 cache-policy operand as a pair. The policy
# is only a performance hint; it does not change weak-memory semantics.
@inline function optype"cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::cache_hint"(
        tmap::Core.LLVMPtr{S, AS.Const}, c1::Integer,
        cache_policy::UInt64) where {S}
    nvvm"cp.async.bulk.tensor.prefetch.tile.1d"(
        _tma_tmap(tmap), Int32(c1), UInt64(cache_policy), Val(true))
end

@inline function optype"cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint"(
        tmap::Core.LLVMPtr{S, AS.Const}, c1::Integer, c2::Integer,
        cache_policy::UInt64) where {S}
    nvvm"cp.async.bulk.tensor.prefetch.tile.2d"(
        _tma_tmap(tmap), Int32(c1), Int32(c2),
        UInt64(cache_policy), Val(true))
end

@inline function optype"cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint"(
        tmap::Core.LLVMPtr{S, AS.Const}, c1::Integer, c2::Integer,
        c3::Integer, cache_policy::UInt64) where {S}
    nvvm"cp.async.bulk.tensor.prefetch.tile.3d"(
        _tma_tmap(tmap), Int32(c1), Int32(c2), Int32(c3),
        UInt64(cache_policy), Val(true))
end

@inline function optype"cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::cache_hint"(
        tmap::Core.LLVMPtr{S, AS.Const}, c1::Integer, c2::Integer,
        c3::Integer, c4::Integer, cache_policy::UInt64) where {S}
    nvvm"cp.async.bulk.tensor.prefetch.tile.4d"(
        _tma_tmap(tmap), Int32(c1), Int32(c2), Int32(c3), Int32(c4),
        UInt64(cache_policy), Val(true))
end

@inline function optype"cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::cache_hint"(
        tmap::Core.LLVMPtr{S, AS.Const}, c1::Integer, c2::Integer,
        c3::Integer, c4::Integer, c5::Integer,
        cache_policy::UInt64) where {S}
    nvvm"cp.async.bulk.tensor.prefetch.tile.5d"(
        _tma_tmap(tmap), Int32(c1), Int32(c2), Int32(c3), Int32(c4),
        Int32(c5), UInt64(cache_policy), Val(true))
end

# Base im2col prefetch is a separate grammar island from tile prefetch. PTX
# 9.3 §9.7.9.26.5.4 admits ranks 3d..5d and requires exactly N signed tensor
# coordinates followed by N-2 signed 16-bit im2col offsets (Figures 11–15).
# Keep this surface exact: the later `.im2col::w[::128]` modes and
# `.tile::gather4` have different operands and target restrictions.

@inline function optype"cp.async.bulk.prefetch.tensor.3d.L2.global.im2col"(
        tmap::Core.LLVMPtr{UInt8, AS.Const}, c1::Int32, c2::Int32, c3::Int32,
        o1::Int16)
    nvvm"cp.async.bulk.tensor.prefetch.im2col.3d"(
        _tma_tmap(tmap), c1, c2, c3, o1, UInt64(0), Val(false))
end

@inline function optype"cp.async.bulk.prefetch.tensor.4d.L2.global.im2col"(
        tmap::Core.LLVMPtr{UInt8, AS.Const}, c1::Int32, c2::Int32,
        c3::Int32, c4::Int32,
        o1::Int16, o2::Int16)
    nvvm"cp.async.bulk.tensor.prefetch.im2col.4d"(
        _tma_tmap(tmap), c1, c2, c3, c4, o1, o2,
        UInt64(0), Val(false))
end

@inline function optype"cp.async.bulk.prefetch.tensor.5d.L2.global.im2col"(
        tmap::Core.LLVMPtr{UInt8, AS.Const}, c1::Int32, c2::Int32,
        c3::Int32, c4::Int32,
        c5::Int32, o1::Int16, o2::Int16, o3::Int16)
    nvvm"cp.async.bulk.tensor.prefetch.im2col.5d"(
        _tma_tmap(tmap), c1, c2, c3, c4, c5, o1, o2, o3,
        UInt64(0), Val(false))
end

# As for tile prefetch, PTX couples the optional cache-policy operand to the
# `.L2::cache_hint` qualifier. It remains a weak, non-observable hint.
@inline function optype"cp.async.bulk.prefetch.tensor.3d.L2.global.im2col.L2::cache_hint"(
        tmap::Core.LLVMPtr{UInt8, AS.Const}, c1::Int32, c2::Int32, c3::Int32,
        o1::Int16, cache_policy::UInt64)
    nvvm"cp.async.bulk.tensor.prefetch.im2col.3d"(
        _tma_tmap(tmap), c1, c2, c3, o1, cache_policy, Val(true))
end

@inline function optype"cp.async.bulk.prefetch.tensor.4d.L2.global.im2col.L2::cache_hint"(
        tmap::Core.LLVMPtr{UInt8, AS.Const}, c1::Int32, c2::Int32,
        c3::Int32, c4::Int32,
        o1::Int16, o2::Int16, cache_policy::UInt64)
    nvvm"cp.async.bulk.tensor.prefetch.im2col.4d"(
        _tma_tmap(tmap), c1, c2, c3, c4, o1, o2, cache_policy, Val(true))
end

@inline function optype"cp.async.bulk.prefetch.tensor.5d.L2.global.im2col.L2::cache_hint"(
        tmap::Core.LLVMPtr{UInt8, AS.Const}, c1::Int32, c2::Int32,
        c3::Int32, c4::Int32,
        c5::Int32, o1::Int16, o2::Int16, o3::Int16,
        cache_policy::UInt64)
    nvvm"cp.async.bulk.tensor.prefetch.im2col.5d"(
        _tma_tmap(tmap), c1, c2, c3, c4, c5, o1, o2, o3,
        cache_policy, Val(true))
end

# --- Residue: shared::cta × cta_group::2 (asm tier) ---------------------------
# No NVVM intrinsic carries both qualifiers at 22.1.7: `g2s.cta` has no
# cta_group operand and `g2s` renders `shared::cluster`. Asm strings keep
# the pyptx modifier order (cta_group after `.<N>d`).

@generated function optype"cp.async.bulk.tensor.1d.cta_group::2.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, mbar::Core.LLVMPtr{U, AS.Shared}) where {T, S, U}
    quote
        Base.@inline
        @asmcall("cp.async.bulk.tensor.1d.cta_group::2.shared::cta.global.tile.mbarrier::complete_tx::bytes [\$0], [\$1, {\$2}], [\$3];",
                 "r,l,r,r,~{memory}", true, Nothing,
                 Tuple{Core.LLVMPtr{$T, AS.Shared}, Core.LLVMPtr{$S, AS.Const},
                       Int32, Core.LLVMPtr{$U, AS.Shared}},
                 dst, tmap, Int32(c1), mbar)
        nothing
    end
end

@generated function optype"cp.async.bulk.tensor.2d.cta_group::2.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}) where {T, S, U}
    quote
        Base.@inline
        @asmcall("cp.async.bulk.tensor.2d.cta_group::2.shared::cta.global.tile.mbarrier::complete_tx::bytes [\$0], [\$1, {\$2, \$3}], [\$4];",
                 "r,l,r,r,r,~{memory}", true, Nothing,
                 Tuple{Core.LLVMPtr{$T, AS.Shared}, Core.LLVMPtr{$S, AS.Const},
                       Int32, Int32, Core.LLVMPtr{$U, AS.Shared}},
                 dst, tmap, Int32(c1), Int32(c2), mbar)
        nothing
    end
end

@generated function optype"cp.async.bulk.tensor.3d.cta_group::2.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer, c3::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}) where {T, S, U}
    quote
        Base.@inline
        @asmcall("cp.async.bulk.tensor.3d.cta_group::2.shared::cta.global.tile.mbarrier::complete_tx::bytes [\$0], [\$1, {\$2, \$3, \$4}], [\$5];",
                 "r,l,r,r,r,r,~{memory}", true, Nothing,
                 Tuple{Core.LLVMPtr{$T, AS.Shared}, Core.LLVMPtr{$S, AS.Const},
                       Int32, Int32, Int32, Core.LLVMPtr{$U, AS.Shared}},
                 dst, tmap, Int32(c1), Int32(c2), Int32(c3), mbar)
        nothing
    end
end

@generated function optype"cp.async.bulk.tensor.4d.cta_group::2.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer, c3::Integer, c4::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}) where {T, S, U}
    quote
        Base.@inline
        @asmcall("cp.async.bulk.tensor.4d.cta_group::2.shared::cta.global.tile.mbarrier::complete_tx::bytes [\$0], [\$1, {\$2, \$3, \$4, \$5}], [\$6];",
                 "r,l,r,r,r,r,r,~{memory}", true, Nothing,
                 Tuple{Core.LLVMPtr{$T, AS.Shared}, Core.LLVMPtr{$S, AS.Const},
                       Int32, Int32, Int32, Int32, Core.LLVMPtr{$U, AS.Shared}},
                 dst, tmap, Int32(c1), Int32(c2), Int32(c3), Int32(c4), mbar)
        nothing
    end
end

@generated function optype"cp.async.bulk.tensor.5d.cta_group::2.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
        dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
        c1::Integer, c2::Integer, c3::Integer, c4::Integer, c5::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}) where {T, S, U}
    quote
        Base.@inline
        @asmcall("cp.async.bulk.tensor.5d.cta_group::2.shared::cta.global.tile.mbarrier::complete_tx::bytes [\$0], [\$1, {\$2, \$3, \$4, \$5, \$6}], [\$7];",
                 "r,l,r,r,r,r,r,r,~{memory}", true, Nothing,
                 Tuple{Core.LLVMPtr{$T, AS.Shared}, Core.LLVMPtr{$S, AS.Const},
                       Int32, Int32, Int32, Int32, Int32,
                       Core.LLVMPtr{$U, AS.Shared}},
                 dst, tmap, Int32(c1), Int32(c2), Int32(c3), Int32(c4),
                 Int32(c5), mbar)
        nothing
    end
end
