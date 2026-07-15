# Design ported from pyptx/pyptx/ptx.py `_Tcgen05`
# (https://github.com/patrick-toulme/pyptx).
# Copyright 2026 Patrick Toulmé. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Translated to Julia and adapted.

# `tcgen05.*` — fourth family migrated to tier-2 intrinsic lowering.
# The notation surface is unchanged: taddr operands stay raw
# `UInt32` TMEM addresses (returned by `tcgen05.alloc`) and SMEM operands
# stay 32-bit offsets from `smem_addr_u32`; the wrappers retype them into
# the address spaces the intrinsics want (`reinterpret_addrspace` — TMEM
# is addrspace(6), 32-bit in the backend datalayout, so taddr operands
# stay 32-bit registers in the emitted PTX).
#
# Intrinsic mapping (probed against llc 22.1.7, sm_100a):
#   - alloc → alloc.shared.cgN (p3 dst): ISel emits `.shared::cta` explicitly,
#     including for the ISA's equivalent generic-address spelling.
#   - dealloc/shift/wait/relinquish → 1:1, identical spellings; cp too,
#     except intrinsic names flatten the multicast qualifier into the
#     shape stem (64x128b.warpx2::02_13 → 64x128b_warpx2_02_13).
#   - ld/st → ld.<shape>.<count> / st.<shape>.<count>: identical
#     spellings. The data moves as an LLVM vector (v<N>i32), so the
#     wrappers repack to/from the notation surface's plain NTuple{N,
#     UInt32}; the i1 immarg is the .pack::16b/.unpack::16b flag,
#     surfaced as an explicit modifier (plain spellings pass Val(false)).
#     The 16x32bx2 shape carries its extra `immHalfSplitoff` i64 immarg
#     as a positional `Val(off)` operand after taddr.
#   - commit → commit.shared.cgN / commit.mc.shared.cgN. ISel emits the
#     `.shared::cluster` spelling for BOTH state-space notations (a
#     shared::cta address is valid in the cluster window — same
#     explicit-widening as mbarrier's expect_tx default), and renders
#     `.shared::cluster.multicast::cluster` where pyptx put multicast
#     first. ptxas accepts both spellings.
#   - dense mma (kinds f16/tf32/f8f6f4/i8) → mma.{shared,tensor} (A from
#     an SMEM descriptor vs a TMEM address) with immargs kind (0=f16,
#     1=tf32, 2=f8f6f4, 3=i8), cta_group, collector_usage (0 = discard =
#     the ISA default; emitted explicitly; 1=lastuse, 2=fill, 3=use). The
#     maskless forms omit the disable-output-lane operand entirely; the
#     masked forms are separate .disable_output_lane.cgN intrinsics with
#     an explicit 4/8-word vector operand. scale-input-d rides separate
#     .scale_d records (f16/tf32 only). Requires PTX 8.8.
#   - mx kinds (mxf8f6f4/mxf4/mxf4nvf4) stay on the asm tier with the ISA's
#     complete block-scale operand schema.  Scale-A and scale-B are explicit
#     TMEM addresses, and the modifier inventory distinguishes architecture-
#     specific `.scale_vec::*` forms from family-compatible `.block*` aliases.
#
# ld/st per-lane register count = `base * count` where
#   base = 1 for shape ∈ {16x64b, 32x32b, 16x32bx2}; 2 for 16x128b;
#   4 for 16x256b
# (PTX 9.3 §9.7.17.8 Table 52). count ∈ {x1..x128} with per_lane ≤ 128.
# The `.pack::16b`/`.unpack::16b` repack variants cover the same grid.
# `tcgen05.ld.red` (load-with-reduction, PTX 9.1+) has no NVVM intrinsic
# records at the pinned backend and stays outside the wrapper surface.
#
# Methods are written out literally (no name-building loop) so every
# intrinsic this file stands on is greppable — test/host/conformance.jl
# scans for `nvvm"..."` literals and requires a probe for each. The ld/st
# blocks are mechanical expansions of Table 49; edit with care.

# taddr/SMEM-offset retypes (see address_space.jl).
@inline _tmem(taddr::UInt32) = reinterpret_addrspace(Val(AS.Tmem), taddr)
@inline _tc_smem(addr::UInt32) = reinterpret_addrspace(Val(AS.Shared), addr)

# tcgen05.ld returns / tcgen05.st takes LLVM vectors; the notation surface
# uses plain tuples (scalar for the single-register forms).
@inline _tc_unvec(x::UInt32) = x
@inline _tc_unvec(v::NTuple{N, VecElement{UInt32}}) where {N} =
    ntuple(i -> v[i].value, Val(N))
@inline _tc_vec(t::NTuple{N, UInt32}) where {N} =
    ntuple(i -> VecElement(t[i]), Val(N))

# --- shift / dealloc / cp -----------------------------------------------------

@inline (::Operation{:tcgen05, (:shift, Symbol("cta_group::1"), :down)})(taddr::UInt32) =
    nvvm"tcgen05.shift.down.cg1"(_tmem(taddr))
@inline (::Operation{:tcgen05, (:shift, Symbol("cta_group::2"), :down)})(taddr::UInt32) =
    nvvm"tcgen05.shift.down.cg2"(_tmem(taddr))

@inline (::Operation{:tcgen05, (:dealloc, Symbol("cta_group::1"), :sync, :aligned, :b32)})(
        taddr::UInt32, ncols::UInt32) =
    nvvm"tcgen05.dealloc.cg1"(_tmem(taddr), ncols)
@inline (::Operation{:tcgen05, (:dealloc, Symbol("cta_group::2"), :sync, :aligned, :b32)})(
        taddr::UInt32, ncols::UInt32) =
    nvvm"tcgen05.dealloc.cg2"(_tmem(taddr), ncols)

@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::1"), Symbol("128x256b"))})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.128x256b.cg1"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::2"), Symbol("128x256b"))})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.128x256b.cg2"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::1"), Symbol("4x256b"))})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.4x256b.cg1"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::2"), Symbol("4x256b"))})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.4x256b.cg2"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::1"), Symbol("128x128b"))})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.128x128b.cg1"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::2"), Symbol("128x128b"))})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.128x128b.cg2"(_tmem(taddr), s_desc)

# Multicast shapes (PTX 9.3 §9.7.17.9: `.64x128b` requires one of the
# `.warpx2::*` pairings, `.32x128b` requires `.warpx4`) and optional
# `.b8x16.{b6x16_p32,b4x16_p64}` decompression during the copy. The
# intrinsic names flatten the multicast qualifier into the shape stem
# (`64x128b_warpx2_02_13`); the notation keeps the ISA's modifier order.
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::1"), Symbol("128x256b"), :b8x16, :b6x16_p32)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.128x256b.b6x16_p32.cg1"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::2"), Symbol("128x256b"), :b8x16, :b6x16_p32)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.128x256b.b6x16_p32.cg2"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::1"), Symbol("128x256b"), :b8x16, :b4x16_p64)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.128x256b.b4x16_p64.cg1"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::2"), Symbol("128x256b"), :b8x16, :b4x16_p64)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.128x256b.b4x16_p64.cg2"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::1"), Symbol("4x256b"), :b8x16, :b6x16_p32)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.4x256b.b6x16_p32.cg1"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::2"), Symbol("4x256b"), :b8x16, :b6x16_p32)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.4x256b.b6x16_p32.cg2"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::1"), Symbol("4x256b"), :b8x16, :b4x16_p64)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.4x256b.b4x16_p64.cg1"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::2"), Symbol("4x256b"), :b8x16, :b4x16_p64)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.4x256b.b4x16_p64.cg2"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::1"), Symbol("128x128b"), :b8x16, :b6x16_p32)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.128x128b.b6x16_p32.cg1"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::2"), Symbol("128x128b"), :b8x16, :b6x16_p32)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.128x128b.b6x16_p32.cg2"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::1"), Symbol("128x128b"), :b8x16, :b4x16_p64)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.128x128b.b4x16_p64.cg1"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::2"), Symbol("128x128b"), :b8x16, :b4x16_p64)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.128x128b.b4x16_p64.cg2"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::1"), Symbol("64x128b"), Symbol("warpx2::02_13"))})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.64x128b_warpx2_02_13.cg1"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::2"), Symbol("64x128b"), Symbol("warpx2::02_13"))})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.64x128b_warpx2_02_13.cg2"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::1"), Symbol("64x128b"), Symbol("warpx2::02_13"), :b8x16, :b6x16_p32)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.64x128b_warpx2_02_13.b6x16_p32.cg1"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::2"), Symbol("64x128b"), Symbol("warpx2::02_13"), :b8x16, :b6x16_p32)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.64x128b_warpx2_02_13.b6x16_p32.cg2"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::1"), Symbol("64x128b"), Symbol("warpx2::02_13"), :b8x16, :b4x16_p64)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.64x128b_warpx2_02_13.b4x16_p64.cg1"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::2"), Symbol("64x128b"), Symbol("warpx2::02_13"), :b8x16, :b4x16_p64)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.64x128b_warpx2_02_13.b4x16_p64.cg2"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::1"), Symbol("64x128b"), Symbol("warpx2::01_23"))})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.64x128b_warpx2_01_23.cg1"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::2"), Symbol("64x128b"), Symbol("warpx2::01_23"))})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.64x128b_warpx2_01_23.cg2"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::1"), Symbol("64x128b"), Symbol("warpx2::01_23"), :b8x16, :b6x16_p32)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.64x128b_warpx2_01_23.b6x16_p32.cg1"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::2"), Symbol("64x128b"), Symbol("warpx2::01_23"), :b8x16, :b6x16_p32)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.64x128b_warpx2_01_23.b6x16_p32.cg2"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::1"), Symbol("64x128b"), Symbol("warpx2::01_23"), :b8x16, :b4x16_p64)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.64x128b_warpx2_01_23.b4x16_p64.cg1"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::2"), Symbol("64x128b"), Symbol("warpx2::01_23"), :b8x16, :b4x16_p64)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.64x128b_warpx2_01_23.b4x16_p64.cg2"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::1"), Symbol("32x128b"), :warpx4)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.32x128b_warpx4.cg1"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::2"), Symbol("32x128b"), :warpx4)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.32x128b_warpx4.cg2"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::1"), Symbol("32x128b"), :warpx4, :b8x16, :b6x16_p32)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.32x128b_warpx4.b6x16_p32.cg1"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::2"), Symbol("32x128b"), :warpx4, :b8x16, :b6x16_p32)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.32x128b_warpx4.b6x16_p32.cg2"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::1"), Symbol("32x128b"), :warpx4, :b8x16, :b4x16_p64)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.32x128b_warpx4.b4x16_p64.cg1"(_tmem(taddr), s_desc)
@inline (::Operation{:tcgen05, (:cp, Symbol("cta_group::2"), Symbol("32x128b"), :warpx4, :b8x16, :b4x16_p64)})(
        taddr::UInt32, s_desc::UInt64) =
    nvvm"tcgen05.cp.32x128b_warpx4.b4x16_p64.cg2"(_tmem(taddr), s_desc)

# --- ld / st (Table 49 expansion) ---------------------------------------------

# 16x64b (base 1)
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x64b"), :x1, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x64b.x1"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x64b"), :x1, :b32)})(taddr::UInt32, src::NTuple{1, UInt32}) =
    nvvm"tcgen05.st.16x64b.x1"(_tmem(taddr), src[1], Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x64b"), :x2, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x64b.x2"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x64b"), :x2, :b32)})(taddr::UInt32, src::NTuple{2, UInt32}) =
    nvvm"tcgen05.st.16x64b.x2"(_tmem(taddr), _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x64b"), :x4, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x64b.x4"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x64b"), :x4, :b32)})(taddr::UInt32, src::NTuple{4, UInt32}) =
    nvvm"tcgen05.st.16x64b.x4"(_tmem(taddr), _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x64b"), :x8, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x64b.x8"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x64b"), :x8, :b32)})(taddr::UInt32, src::NTuple{8, UInt32}) =
    nvvm"tcgen05.st.16x64b.x8"(_tmem(taddr), _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x64b"), :x16, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x64b.x16"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x64b"), :x16, :b32)})(taddr::UInt32, src::NTuple{16, UInt32}) =
    nvvm"tcgen05.st.16x64b.x16"(_tmem(taddr), _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x64b"), :x32, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x64b.x32"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x64b"), :x32, :b32)})(taddr::UInt32, src::NTuple{32, UInt32}) =
    nvvm"tcgen05.st.16x64b.x32"(_tmem(taddr), _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x64b"), :x64, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x64b.x64"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x64b"), :x64, :b32)})(taddr::UInt32, src::NTuple{64, UInt32}) =
    nvvm"tcgen05.st.16x64b.x64"(_tmem(taddr), _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x64b"), :x128, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x64b.x128"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x64b"), :x128, :b32)})(taddr::UInt32, src::NTuple{128, UInt32}) =
    nvvm"tcgen05.st.16x64b.x128"(_tmem(taddr), _tc_vec(src), Val(false))

# 32x32b (base 1)
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("32x32b"), :x1, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.32x32b.x1"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("32x32b"), :x1, :b32)})(taddr::UInt32, src::NTuple{1, UInt32}) =
    nvvm"tcgen05.st.32x32b.x1"(_tmem(taddr), src[1], Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("32x32b"), :x2, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.32x32b.x2"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("32x32b"), :x2, :b32)})(taddr::UInt32, src::NTuple{2, UInt32}) =
    nvvm"tcgen05.st.32x32b.x2"(_tmem(taddr), _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("32x32b"), :x4, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.32x32b.x4"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("32x32b"), :x4, :b32)})(taddr::UInt32, src::NTuple{4, UInt32}) =
    nvvm"tcgen05.st.32x32b.x4"(_tmem(taddr), _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("32x32b"), :x8, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.32x32b.x8"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("32x32b"), :x8, :b32)})(taddr::UInt32, src::NTuple{8, UInt32}) =
    nvvm"tcgen05.st.32x32b.x8"(_tmem(taddr), _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("32x32b"), :x16, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.32x32b.x16"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("32x32b"), :x16, :b32)})(taddr::UInt32, src::NTuple{16, UInt32}) =
    nvvm"tcgen05.st.32x32b.x16"(_tmem(taddr), _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("32x32b"), :x32, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.32x32b.x32"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("32x32b"), :x32, :b32)})(taddr::UInt32, src::NTuple{32, UInt32}) =
    nvvm"tcgen05.st.32x32b.x32"(_tmem(taddr), _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("32x32b"), :x64, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.32x32b.x64"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("32x32b"), :x64, :b32)})(taddr::UInt32, src::NTuple{64, UInt32}) =
    nvvm"tcgen05.st.32x32b.x64"(_tmem(taddr), _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("32x32b"), :x128, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.32x32b.x128"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("32x32b"), :x128, :b32)})(taddr::UInt32, src::NTuple{128, UInt32}) =
    nvvm"tcgen05.st.32x32b.x128"(_tmem(taddr), _tc_vec(src), Val(false))

# 16x128b (base 2)
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x128b"), :x1, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x128b.x1"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x128b"), :x1, :b32)})(taddr::UInt32, src::NTuple{2, UInt32}) =
    nvvm"tcgen05.st.16x128b.x1"(_tmem(taddr), _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x128b"), :x2, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x128b.x2"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x128b"), :x2, :b32)})(taddr::UInt32, src::NTuple{4, UInt32}) =
    nvvm"tcgen05.st.16x128b.x2"(_tmem(taddr), _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x128b"), :x4, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x128b.x4"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x128b"), :x4, :b32)})(taddr::UInt32, src::NTuple{8, UInt32}) =
    nvvm"tcgen05.st.16x128b.x4"(_tmem(taddr), _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x128b"), :x8, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x128b.x8"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x128b"), :x8, :b32)})(taddr::UInt32, src::NTuple{16, UInt32}) =
    nvvm"tcgen05.st.16x128b.x8"(_tmem(taddr), _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x128b"), :x16, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x128b.x16"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x128b"), :x16, :b32)})(taddr::UInt32, src::NTuple{32, UInt32}) =
    nvvm"tcgen05.st.16x128b.x16"(_tmem(taddr), _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x128b"), :x32, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x128b.x32"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x128b"), :x32, :b32)})(taddr::UInt32, src::NTuple{64, UInt32}) =
    nvvm"tcgen05.st.16x128b.x32"(_tmem(taddr), _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x128b"), :x64, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x128b.x64"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x128b"), :x64, :b32)})(taddr::UInt32, src::NTuple{128, UInt32}) =
    nvvm"tcgen05.st.16x128b.x64"(_tmem(taddr), _tc_vec(src), Val(false))

# 16x256b (base 4)
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x256b"), :x1, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x256b.x1"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x256b"), :x1, :b32)})(taddr::UInt32, src::NTuple{4, UInt32}) =
    nvvm"tcgen05.st.16x256b.x1"(_tmem(taddr), _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x256b"), :x2, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x256b.x2"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x256b"), :x2, :b32)})(taddr::UInt32, src::NTuple{8, UInt32}) =
    nvvm"tcgen05.st.16x256b.x2"(_tmem(taddr), _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x256b"), :x4, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x256b.x4"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x256b"), :x4, :b32)})(taddr::UInt32, src::NTuple{16, UInt32}) =
    nvvm"tcgen05.st.16x256b.x4"(_tmem(taddr), _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x256b"), :x8, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x256b.x8"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x256b"), :x8, :b32)})(taddr::UInt32, src::NTuple{32, UInt32}) =
    nvvm"tcgen05.st.16x256b.x8"(_tmem(taddr), _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x256b"), :x16, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x256b.x16"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x256b"), :x16, :b32)})(taddr::UInt32, src::NTuple{64, UInt32}) =
    nvvm"tcgen05.st.16x256b.x16"(_tmem(taddr), _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x256b"), :x32, :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x256b.x32"(_tmem(taddr), Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x256b"), :x32, :b32)})(taddr::UInt32, src::NTuple{128, UInt32}) =
    nvvm"tcgen05.st.16x256b.x32"(_tmem(taddr), _tc_vec(src), Val(false))

# 16x32bx2 (base 1; two 16x32b accesses — the second starts at taddr +
# immHalfSplitoff, an immediate operand passed as `Val(off)`)
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x32bx2"), :x1, :b32)})(taddr::UInt32, halfsplitoff::Val) =
    _tc_unvec(nvvm"tcgen05.ld.16x32bx2.x1"(_tmem(taddr), halfsplitoff, Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x32bx2"), :x1, :b32)})(taddr::UInt32, halfsplitoff::Val, src::NTuple{1, UInt32}) =
    nvvm"tcgen05.st.16x32bx2.x1"(_tmem(taddr), halfsplitoff, src[1], Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x32bx2"), :x2, :b32)})(taddr::UInt32, halfsplitoff::Val) =
    _tc_unvec(nvvm"tcgen05.ld.16x32bx2.x2"(_tmem(taddr), halfsplitoff, Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x32bx2"), :x2, :b32)})(taddr::UInt32, halfsplitoff::Val, src::NTuple{2, UInt32}) =
    nvvm"tcgen05.st.16x32bx2.x2"(_tmem(taddr), halfsplitoff, _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x32bx2"), :x4, :b32)})(taddr::UInt32, halfsplitoff::Val) =
    _tc_unvec(nvvm"tcgen05.ld.16x32bx2.x4"(_tmem(taddr), halfsplitoff, Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x32bx2"), :x4, :b32)})(taddr::UInt32, halfsplitoff::Val, src::NTuple{4, UInt32}) =
    nvvm"tcgen05.st.16x32bx2.x4"(_tmem(taddr), halfsplitoff, _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x32bx2"), :x8, :b32)})(taddr::UInt32, halfsplitoff::Val) =
    _tc_unvec(nvvm"tcgen05.ld.16x32bx2.x8"(_tmem(taddr), halfsplitoff, Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x32bx2"), :x8, :b32)})(taddr::UInt32, halfsplitoff::Val, src::NTuple{8, UInt32}) =
    nvvm"tcgen05.st.16x32bx2.x8"(_tmem(taddr), halfsplitoff, _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x32bx2"), :x16, :b32)})(taddr::UInt32, halfsplitoff::Val) =
    _tc_unvec(nvvm"tcgen05.ld.16x32bx2.x16"(_tmem(taddr), halfsplitoff, Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x32bx2"), :x16, :b32)})(taddr::UInt32, halfsplitoff::Val, src::NTuple{16, UInt32}) =
    nvvm"tcgen05.st.16x32bx2.x16"(_tmem(taddr), halfsplitoff, _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x32bx2"), :x32, :b32)})(taddr::UInt32, halfsplitoff::Val) =
    _tc_unvec(nvvm"tcgen05.ld.16x32bx2.x32"(_tmem(taddr), halfsplitoff, Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x32bx2"), :x32, :b32)})(taddr::UInt32, halfsplitoff::Val, src::NTuple{32, UInt32}) =
    nvvm"tcgen05.st.16x32bx2.x32"(_tmem(taddr), halfsplitoff, _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x32bx2"), :x64, :b32)})(taddr::UInt32, halfsplitoff::Val) =
    _tc_unvec(nvvm"tcgen05.ld.16x32bx2.x64"(_tmem(taddr), halfsplitoff, Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x32bx2"), :x64, :b32)})(taddr::UInt32, halfsplitoff::Val, src::NTuple{64, UInt32}) =
    nvvm"tcgen05.st.16x32bx2.x64"(_tmem(taddr), halfsplitoff, _tc_vec(src), Val(false))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x32bx2"), :x128, :b32)})(taddr::UInt32, halfsplitoff::Val) =
    _tc_unvec(nvvm"tcgen05.ld.16x32bx2.x128"(_tmem(taddr), halfsplitoff, Val(false)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x32bx2"), :x128, :b32)})(taddr::UInt32, halfsplitoff::Val, src::NTuple{128, UInt32}) =
    nvvm"tcgen05.st.16x32bx2.x128"(_tmem(taddr), halfsplitoff, _tc_vec(src), Val(false))

# pack::16b / unpack::16b — optional 16-bit repack during the transfer
# (ld packs two 16-bit TMEM chunks per 32-bit register; st unpacks two
# 16-bit chunks per source register). Register counts are unchanged:
# the flag flips the intrinsics' trailing i1 immarg.
# 16x64b (base 1)
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x64b"), :x1, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x64b.x1"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x64b"), :x1, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{1, UInt32}) =
    nvvm"tcgen05.st.16x64b.x1"(_tmem(taddr), src[1], Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x64b"), :x2, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x64b.x2"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x64b"), :x2, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{2, UInt32}) =
    nvvm"tcgen05.st.16x64b.x2"(_tmem(taddr), _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x64b"), :x4, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x64b.x4"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x64b"), :x4, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{4, UInt32}) =
    nvvm"tcgen05.st.16x64b.x4"(_tmem(taddr), _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x64b"), :x8, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x64b.x8"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x64b"), :x8, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{8, UInt32}) =
    nvvm"tcgen05.st.16x64b.x8"(_tmem(taddr), _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x64b"), :x16, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x64b.x16"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x64b"), :x16, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{16, UInt32}) =
    nvvm"tcgen05.st.16x64b.x16"(_tmem(taddr), _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x64b"), :x32, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x64b.x32"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x64b"), :x32, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{32, UInt32}) =
    nvvm"tcgen05.st.16x64b.x32"(_tmem(taddr), _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x64b"), :x64, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x64b.x64"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x64b"), :x64, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{64, UInt32}) =
    nvvm"tcgen05.st.16x64b.x64"(_tmem(taddr), _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x64b"), :x128, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x64b.x128"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x64b"), :x128, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{128, UInt32}) =
    nvvm"tcgen05.st.16x64b.x128"(_tmem(taddr), _tc_vec(src), Val(true))
# 32x32b (base 1)
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("32x32b"), :x1, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.32x32b.x1"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("32x32b"), :x1, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{1, UInt32}) =
    nvvm"tcgen05.st.32x32b.x1"(_tmem(taddr), src[1], Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("32x32b"), :x2, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.32x32b.x2"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("32x32b"), :x2, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{2, UInt32}) =
    nvvm"tcgen05.st.32x32b.x2"(_tmem(taddr), _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("32x32b"), :x4, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.32x32b.x4"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("32x32b"), :x4, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{4, UInt32}) =
    nvvm"tcgen05.st.32x32b.x4"(_tmem(taddr), _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("32x32b"), :x8, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.32x32b.x8"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("32x32b"), :x8, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{8, UInt32}) =
    nvvm"tcgen05.st.32x32b.x8"(_tmem(taddr), _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("32x32b"), :x16, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.32x32b.x16"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("32x32b"), :x16, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{16, UInt32}) =
    nvvm"tcgen05.st.32x32b.x16"(_tmem(taddr), _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("32x32b"), :x32, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.32x32b.x32"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("32x32b"), :x32, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{32, UInt32}) =
    nvvm"tcgen05.st.32x32b.x32"(_tmem(taddr), _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("32x32b"), :x64, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.32x32b.x64"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("32x32b"), :x64, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{64, UInt32}) =
    nvvm"tcgen05.st.32x32b.x64"(_tmem(taddr), _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("32x32b"), :x128, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.32x32b.x128"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("32x32b"), :x128, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{128, UInt32}) =
    nvvm"tcgen05.st.32x32b.x128"(_tmem(taddr), _tc_vec(src), Val(true))
# 16x128b (base 2)
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x128b"), :x1, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x128b.x1"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x128b"), :x1, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{2, UInt32}) =
    nvvm"tcgen05.st.16x128b.x1"(_tmem(taddr), _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x128b"), :x2, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x128b.x2"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x128b"), :x2, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{4, UInt32}) =
    nvvm"tcgen05.st.16x128b.x2"(_tmem(taddr), _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x128b"), :x4, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x128b.x4"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x128b"), :x4, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{8, UInt32}) =
    nvvm"tcgen05.st.16x128b.x4"(_tmem(taddr), _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x128b"), :x8, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x128b.x8"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x128b"), :x8, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{16, UInt32}) =
    nvvm"tcgen05.st.16x128b.x8"(_tmem(taddr), _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x128b"), :x16, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x128b.x16"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x128b"), :x16, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{32, UInt32}) =
    nvvm"tcgen05.st.16x128b.x16"(_tmem(taddr), _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x128b"), :x32, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x128b.x32"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x128b"), :x32, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{64, UInt32}) =
    nvvm"tcgen05.st.16x128b.x32"(_tmem(taddr), _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x128b"), :x64, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x128b.x64"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x128b"), :x64, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{128, UInt32}) =
    nvvm"tcgen05.st.16x128b.x64"(_tmem(taddr), _tc_vec(src), Val(true))
# 16x256b (base 4)
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x256b"), :x1, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x256b.x1"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x256b"), :x1, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{4, UInt32}) =
    nvvm"tcgen05.st.16x256b.x1"(_tmem(taddr), _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x256b"), :x2, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x256b.x2"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x256b"), :x2, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{8, UInt32}) =
    nvvm"tcgen05.st.16x256b.x2"(_tmem(taddr), _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x256b"), :x4, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x256b.x4"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x256b"), :x4, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{16, UInt32}) =
    nvvm"tcgen05.st.16x256b.x4"(_tmem(taddr), _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x256b"), :x8, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x256b.x8"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x256b"), :x8, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{32, UInt32}) =
    nvvm"tcgen05.st.16x256b.x8"(_tmem(taddr), _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x256b"), :x16, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x256b.x16"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x256b"), :x16, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{64, UInt32}) =
    nvvm"tcgen05.st.16x256b.x16"(_tmem(taddr), _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x256b"), :x32, Symbol("pack::16b"), :b32)})(taddr::UInt32) =
    _tc_unvec(nvvm"tcgen05.ld.16x256b.x32"(_tmem(taddr), Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x256b"), :x32, Symbol("unpack::16b"), :b32)})(taddr::UInt32, src::NTuple{128, UInt32}) =
    nvvm"tcgen05.st.16x256b.x32"(_tmem(taddr), _tc_vec(src), Val(true))
# 16x32bx2 (base 1)
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x32bx2"), :x1, Symbol("pack::16b"), :b32)})(taddr::UInt32, halfsplitoff::Val) =
    _tc_unvec(nvvm"tcgen05.ld.16x32bx2.x1"(_tmem(taddr), halfsplitoff, Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x32bx2"), :x1, Symbol("unpack::16b"), :b32)})(taddr::UInt32, halfsplitoff::Val, src::NTuple{1, UInt32}) =
    nvvm"tcgen05.st.16x32bx2.x1"(_tmem(taddr), halfsplitoff, src[1], Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x32bx2"), :x2, Symbol("pack::16b"), :b32)})(taddr::UInt32, halfsplitoff::Val) =
    _tc_unvec(nvvm"tcgen05.ld.16x32bx2.x2"(_tmem(taddr), halfsplitoff, Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x32bx2"), :x2, Symbol("unpack::16b"), :b32)})(taddr::UInt32, halfsplitoff::Val, src::NTuple{2, UInt32}) =
    nvvm"tcgen05.st.16x32bx2.x2"(_tmem(taddr), halfsplitoff, _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x32bx2"), :x4, Symbol("pack::16b"), :b32)})(taddr::UInt32, halfsplitoff::Val) =
    _tc_unvec(nvvm"tcgen05.ld.16x32bx2.x4"(_tmem(taddr), halfsplitoff, Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x32bx2"), :x4, Symbol("unpack::16b"), :b32)})(taddr::UInt32, halfsplitoff::Val, src::NTuple{4, UInt32}) =
    nvvm"tcgen05.st.16x32bx2.x4"(_tmem(taddr), halfsplitoff, _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x32bx2"), :x8, Symbol("pack::16b"), :b32)})(taddr::UInt32, halfsplitoff::Val) =
    _tc_unvec(nvvm"tcgen05.ld.16x32bx2.x8"(_tmem(taddr), halfsplitoff, Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x32bx2"), :x8, Symbol("unpack::16b"), :b32)})(taddr::UInt32, halfsplitoff::Val, src::NTuple{8, UInt32}) =
    nvvm"tcgen05.st.16x32bx2.x8"(_tmem(taddr), halfsplitoff, _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x32bx2"), :x16, Symbol("pack::16b"), :b32)})(taddr::UInt32, halfsplitoff::Val) =
    _tc_unvec(nvvm"tcgen05.ld.16x32bx2.x16"(_tmem(taddr), halfsplitoff, Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x32bx2"), :x16, Symbol("unpack::16b"), :b32)})(taddr::UInt32, halfsplitoff::Val, src::NTuple{16, UInt32}) =
    nvvm"tcgen05.st.16x32bx2.x16"(_tmem(taddr), halfsplitoff, _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x32bx2"), :x32, Symbol("pack::16b"), :b32)})(taddr::UInt32, halfsplitoff::Val) =
    _tc_unvec(nvvm"tcgen05.ld.16x32bx2.x32"(_tmem(taddr), halfsplitoff, Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x32bx2"), :x32, Symbol("unpack::16b"), :b32)})(taddr::UInt32, halfsplitoff::Val, src::NTuple{32, UInt32}) =
    nvvm"tcgen05.st.16x32bx2.x32"(_tmem(taddr), halfsplitoff, _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x32bx2"), :x64, Symbol("pack::16b"), :b32)})(taddr::UInt32, halfsplitoff::Val) =
    _tc_unvec(nvvm"tcgen05.ld.16x32bx2.x64"(_tmem(taddr), halfsplitoff, Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x32bx2"), :x64, Symbol("unpack::16b"), :b32)})(taddr::UInt32, halfsplitoff::Val, src::NTuple{64, UInt32}) =
    nvvm"tcgen05.st.16x32bx2.x64"(_tmem(taddr), halfsplitoff, _tc_vec(src), Val(true))
@inline (::Operation{:tcgen05, (:ld, :sync, :aligned, Symbol("16x32bx2"), :x128, Symbol("pack::16b"), :b32)})(taddr::UInt32, halfsplitoff::Val) =
    _tc_unvec(nvvm"tcgen05.ld.16x32bx2.x128"(_tmem(taddr), halfsplitoff, Val(true)))
@inline (::Operation{:tcgen05, (:st, :sync, :aligned, Symbol("16x32bx2"), :x128, Symbol("unpack::16b"), :b32)})(taddr::UInt32, halfsplitoff::Val, src::NTuple{128, UInt32}) =
    nvvm"tcgen05.st.16x32bx2.x128"(_tmem(taddr), halfsplitoff, _tc_vec(src), Val(true))

# --- alloc / relinquish / wait / commit ----------------------------------------

# Generic-address alloc form. The ISA permits omitting `.shared::cta` while
# still requiring `dst` to lie in the CTA shared-memory window. Keep the
# pointer schema exact so a wrong address space, pointee, arity, or nCols
# carrier cannot fall through to generic tcgen05 rendering.
@inline (::Operation{:tcgen05, (:alloc, Symbol("cta_group::1"), :sync, :aligned,
        :b32)})(dst::Core.LLVMPtr{UInt32, AS.Shared}, ncols::UInt32) =
    nvvm"tcgen05.alloc.shared.cg1"(dst, ncols)
@inline (::Operation{:tcgen05, (:alloc, Symbol("cta_group::2"), :sync, :aligned,
        :b32)})(dst::Core.LLVMPtr{UInt32, AS.Shared}, ncols::UInt32) =
    nvvm"tcgen05.alloc.shared.cg2"(dst, ncols)

@inline (::Operation{:tcgen05, (:alloc, Symbol("cta_group::1"), :sync, :aligned,
        Symbol("shared::cta"), :b32)})(dst::UInt32, ncols::UInt32) =
    nvvm"tcgen05.alloc.shared.cg1"(_tc_smem(dst), ncols)
@inline (::Operation{:tcgen05, (:alloc, Symbol("cta_group::2"), :sync, :aligned,
        Symbol("shared::cta"), :b32)})(dst::UInt32, ncols::UInt32) =
    nvvm"tcgen05.alloc.shared.cg2"(_tc_smem(dst), ncols)

@inline (::Operation{:tcgen05, (:relinquish_alloc_permit, Symbol("cta_group::1"),
        :sync, :aligned)})() =
    nvvm"tcgen05.relinq.alloc.permit.cg1"()
@inline (::Operation{:tcgen05, (:relinquish_alloc_permit, Symbol("cta_group::2"),
        :sync, :aligned)})() =
    nvvm"tcgen05.relinq.alloc.permit.cg2"()

@inline (::Operation{:tcgen05, (Symbol("wait::ld"), :sync, :aligned)})() =
    nvvm"tcgen05.wait.ld"()
@inline (::Operation{:tcgen05, (Symbol("wait::st"), :sync, :aligned)})() =
    nvvm"tcgen05.wait.st"()

# Specialized thread-synchronization fences are side-effecting, no-argument
# code-motion barriers (PTX 9.3 §9.7.17.11.1). Keep them as inline PTX with
# a memory clobber: LLVM 15 does not recognize the tcgen05 NVVM intrinsics and
# trusts their generated
# `readnone` declaration, which lets it delete the void calls. `sideeffect`
# retains each fence and `~{memory}` prevents tcgen05 and execution-ordering
# operations from moving across it. Exact methods still prevent an accidental
# operand from becoming a literal extra PTX operand through the generic chain.
@inline (::Operation{:tcgen05, (Symbol("fence::before_thread_sync"),)})() =
    @asmcall("tcgen05.fence::before_thread_sync;", "~{memory}", true,
             Nothing, Tuple{})
@inline (::Operation{:tcgen05, (Symbol("fence::after_thread_sync"),)})() =
    @asmcall("tcgen05.fence::after_thread_sync;", "~{memory}", true,
             Nothing, Tuple{})

# Both state-space notations lower to the same intrinsic and emit the
# `.shared::cluster` spelling (see header).
@inline (::Operation{:tcgen05, (:commit, Symbol("cta_group::1"),
        Symbol("mbarrier::arrive::one"), Symbol("shared::cluster"), :b64)})(
        mbar::Core.LLVMPtr{UInt64, AS.Shared}) =
    nvvm"tcgen05.commit.shared.cg1"(mbar)
@inline (::Operation{:tcgen05, (:commit, Symbol("cta_group::2"),
        Symbol("mbarrier::arrive::one"), Symbol("shared::cluster"), :b64)})(
        mbar::Core.LLVMPtr{UInt64, AS.Shared}) =
    nvvm"tcgen05.commit.shared.cg2"(mbar)

@inline (::Operation{:tcgen05, (:commit, Symbol("cta_group::1"),
        Symbol("mbarrier::arrive::one"), Symbol("shared::cta"), :b64)})(mbar::UInt32) =
    nvvm"tcgen05.commit.shared.cg1"(_tc_smem(mbar))
@inline (::Operation{:tcgen05, (:commit, Symbol("cta_group::2"),
        Symbol("mbarrier::arrive::one"), Symbol("shared::cta"), :b64)})(mbar::UInt32) =
    nvvm"tcgen05.commit.shared.cg2"(_tc_smem(mbar))
@inline (::Operation{:tcgen05, (:commit, Symbol("cta_group::1"),
        Symbol("mbarrier::arrive::one"), Symbol("shared::cluster"), :b64)})(mbar::UInt32) =
    nvvm"tcgen05.commit.shared.cg1"(_tc_smem(mbar))
@inline (::Operation{:tcgen05, (:commit, Symbol("cta_group::2"),
        Symbol("mbarrier::arrive::one"), Symbol("shared::cluster"), :b64)})(mbar::UInt32) =
    nvvm"tcgen05.commit.shared.cg2"(_tc_smem(mbar))

@inline (::Operation{:tcgen05, (:commit, Symbol("cta_group::1"),
        Symbol("mbarrier::arrive::one"), Symbol("multicast::cluster"),
        Symbol("shared::cluster"), :b64)})(mbar::UInt32, mask::Integer) =
    nvvm"tcgen05.commit.mc.shared.cg1"(_tc_smem(mbar), UInt16(mask))
@inline (::Operation{:tcgen05, (:commit, Symbol("cta_group::2"),
        Symbol("mbarrier::arrive::one"), Symbol("multicast::cluster"),
        Symbol("shared::cluster"), :b64)})(mbar::UInt32, mask::Integer) =
    nvvm"tcgen05.commit.mc.shared.cg2"(_tc_smem(mbar), UInt16(mask))

# --- dense mma (A from SMEM descriptor) ----------------------------------------
# kind immarg: 0=f16, 1=tf32, 2=f8f6f4, 3=i8; collector_usage 0 = discard
# (the ISA default, now spelled explicitly in the output).
@inline (::Operation{:tcgen05, (:mma, Symbol("cta_group::1"), Symbol("kind::f16"))})(
        d::UInt32, a_desc::UInt64, b_desc::UInt64,
        idesc::UInt32, enable_input_d::Bool) =
    nvvm"tcgen05.mma.shared"(_tmem(d), a_desc, b_desc, idesc, enable_input_d,
                             Val(0), Val(1), Val(0))
@inline (::Operation{:tcgen05, (:mma, Symbol("cta_group::2"), Symbol("kind::f16"))})(
        d::UInt32, a_desc::UInt64, b_desc::UInt64,
        idesc::UInt32, enable_input_d::Bool) =
    nvvm"tcgen05.mma.shared"(_tmem(d), a_desc, b_desc, idesc, enable_input_d,
                             Val(0), Val(2), Val(0))
@inline (::Operation{:tcgen05, (:mma, Symbol("cta_group::1"), Symbol("kind::tf32"))})(
        d::UInt32, a_desc::UInt64, b_desc::UInt64,
        idesc::UInt32, enable_input_d::Bool) =
    nvvm"tcgen05.mma.shared"(_tmem(d), a_desc, b_desc, idesc, enable_input_d,
                             Val(1), Val(1), Val(0))
@inline (::Operation{:tcgen05, (:mma, Symbol("cta_group::2"), Symbol("kind::tf32"))})(
        d::UInt32, a_desc::UInt64, b_desc::UInt64,
        idesc::UInt32, enable_input_d::Bool) =
    nvvm"tcgen05.mma.shared"(_tmem(d), a_desc, b_desc, idesc, enable_input_d,
                             Val(1), Val(2), Val(0))
@inline (::Operation{:tcgen05, (:mma, Symbol("cta_group::1"), Symbol("kind::f8f6f4"))})(
        d::UInt32, a_desc::UInt64, b_desc::UInt64,
        idesc::UInt32, enable_input_d::Bool) =
    nvvm"tcgen05.mma.shared"(_tmem(d), a_desc, b_desc, idesc, enable_input_d,
                             Val(2), Val(1), Val(0))
@inline (::Operation{:tcgen05, (:mma, Symbol("cta_group::2"), Symbol("kind::f8f6f4"))})(
        d::UInt32, a_desc::UInt64, b_desc::UInt64,
        idesc::UInt32, enable_input_d::Bool) =
    nvvm"tcgen05.mma.shared"(_tmem(d), a_desc, b_desc, idesc, enable_input_d,
                             Val(2), Val(2), Val(0))
@inline (::Operation{:tcgen05, (:mma, Symbol("cta_group::1"), Symbol("kind::i8"))})(
        d::UInt32, a_desc::UInt64, b_desc::UInt64,
        idesc::UInt32, enable_input_d::Bool) =
    nvvm"tcgen05.mma.shared"(_tmem(d), a_desc, b_desc, idesc, enable_input_d,
                             Val(3), Val(1), Val(0))
@inline (::Operation{:tcgen05, (:mma, Symbol("cta_group::2"), Symbol("kind::i8"))})(
        d::UInt32, a_desc::UInt64, b_desc::UInt64,
        idesc::UInt32, enable_input_d::Bool) =
    nvvm"tcgen05.mma.shared"(_tmem(d), a_desc, b_desc, idesc, enable_input_d,
                             Val(3), Val(2), Val(0))

# --- dense mma completion (generated) -------------------------------------------
# The remaining PTX 9.3 §9.7.17.10 dense forms are a modifier/operand
# product too large for literal methods: A source (SMEM descriptor vs
# TMEM address, distinguished by UInt64 vs UInt32 dispatch exactly like
# the ISA distinguishes `a-desc` vs `[a-tmem]`), the collector buffer
# (`collector::a::{lastuse,fill,use}` — spelled-nothing defaults to
# discard, the literal methods above), `.ashift` (TMEM A only), an
# optional disable-output-lane mask vector (4 words for cta_group::1,
# 8 for ::2, positional before enable-input-d), and an optional
# scale-input-d immediate (f16/tf32 only, `Val(s)` with s ∈ 0:15,
# positional after enable-input-d). Notation keeps the ISA order
# `kind{.ashift}{.collector::a::op}`; ISel renders the collector before
# `.ashift` — same accepted non-WYSIWYG rendering class as commit's
# multicast. Methods are generated from one spec;
# TCGEN05_MMA_DENSE_INTRINSIC_NAMES records every tier-2 name for the
# conformance replay (the mma.jl generated-family standing guarantee).
#
# Empirical NVVM collector enum (llc 22.1.7): 0=discard, 1=lastuse,
# 2=fill, 3=use. The ashift variants' immarg range [0, 2) is exactly the
# ISA's "no fill/use with ashift" rule.
const _TCGEN05_DENSE_KINDS =
    ((:f16, 0, true), (:tf32, 1, true), (:f8f6f4, 2, false), (:i8, 3, false))
const _TCGEN05_DENSE_COLLECTORS =
    ((nothing, 0), (Symbol("collector::a::lastuse"), 1),
     (Symbol("collector::a::fill"), 2), (Symbol("collector::a::use"), 3))

const TCGEN05_MMA_DENSE_INTRINSIC_NAMES = String[]

function _tcgen05_dense_register(kind::Symbol, kindval::Int, scale_ok::Bool,
                                 cg::Int, tmem_a::Bool, ashift::Bool,
                                 collmod, collval::Int)
    mods = (:mma, Symbol("cta_group::", cg), Symbol("kind::", kind),
            (ashift ? (:ashift,) : ())...,
            (collmod === nothing ? () : (collmod,))...)
    maskN = cg == 1 ? 4 : 8
    stem = "llvm.nvvm.tcgen05.mma." * (tmem_a ? "tensor" : "shared")
    sh = ashift ? ".ashift" : ""
    aT = tmem_a ? UInt32 : UInt64
    aexpr = tmem_a ? :(_tmem(a)) : :a
    function reg(name)
        name in TCGEN05_MMA_DENSE_INTRINSIC_NAMES ||
            push!(TCGEN05_MMA_DENSE_INTRINSIC_NAMES, name)
        NVVM.IntrinsicCall{Symbol(name)}()
    end

    # [d], a, b, idesc, enable — the literal SMEM-A discard methods above
    # already own that cell of the grid.
    if tmem_a || ashift || collmod !== nothing
        call = reg(stem * sh)
        @eval @inline function (::Operation{:tcgen05, $mods})(
                d::UInt32, a::$aT, b_desc::UInt64, idesc::UInt32,
                enable_input_d::Bool)
            $call(_tmem(d), $aexpr, b_desc, idesc, enable_input_d,
                  Val($kindval), Val($cg), Val($collval))
        end
    end

    # [d], a, b, idesc, {disable-output-lane}, enable
    call = reg(stem * ".disable_output_lane.cg$cg" * sh)
    @eval @inline function (::Operation{:tcgen05, $mods})(
            d::UInt32, a::$aT, b_desc::UInt64, idesc::UInt32,
            mask::NTuple{$maskN, UInt32}, enable_input_d::Bool)
        $call(_tmem(d), $aexpr, b_desc, idesc, enable_input_d,
              _tc_vec(mask), Val($kindval), Val($collval))
    end

    scale_ok || return nothing

    # [d], a, b, idesc, enable, scale-input-d
    call = reg(stem * ".scale_d" * sh)
    @eval @inline function (::Operation{:tcgen05, $mods})(
            d::UInt32, a::$aT, b_desc::UInt64, idesc::UInt32,
            enable_input_d::Bool, scale::Val)
        $call(_tmem(d), $aexpr, b_desc, idesc, enable_input_d,
              scale, Val($kindval), Val($cg), Val($collval))
    end

    # [d], a, b, idesc, {disable-output-lane}, enable, scale-input-d
    call = reg(stem * ".scale_d.disable_output_lane.cg$cg" * sh)
    @eval @inline function (::Operation{:tcgen05, $mods})(
            d::UInt32, a::$aT, b_desc::UInt64, idesc::UInt32,
            mask::NTuple{$maskN, UInt32}, enable_input_d::Bool, scale::Val)
        $call(_tmem(d), $aexpr, b_desc, idesc, enable_input_d,
              scale, _tc_vec(mask), Val($kindval), Val($collval))
    end
    nothing
end

for (kind, kindval, scale_ok) in _TCGEN05_DENSE_KINDS, cg in (1, 2),
        (collmod, collval) in _TCGEN05_DENSE_COLLECTORS
    _tcgen05_dense_register(kind, kindval, scale_ok, cg, false, false,
                            collmod, collval)
    _tcgen05_dense_register(kind, kindval, scale_ok, cg, true, false,
                            collmod, collval)
    collval < 2 &&
        _tcgen05_dense_register(kind, kindval, scale_ok, cg, true, true,
                                collmod, collval)
end

# --- sparse mma (generated) ------------------------------------------------------
# tcgen05.mma.sp (PTX 9.3 §9.7.17.10.9.2) mirrors the dense grid with one
# extra operand: the sparsity-metadata TMEM address, spelled between the
# B descriptor and idesc exactly as PTX orders `[sp-meta-tmem]`. The
# intrinsics instead carry sp-meta after enable-input-d; the wrappers
# reorder. Kinds, collector enum, .ashift restrictions, mask widths, and
# scale-input-d legality are identical to dense. The sp block-scale (MX)
# records stay outside this family, like their dense counterparts.
const TCGEN05_MMA_SP_DENSE_INTRINSIC_NAMES = String[]

function _tcgen05_sp_register(kind::Symbol, kindval::Int, scale_ok::Bool,
                              cg::Int, tmem_a::Bool, ashift::Bool,
                              collmod, collval::Int)
    mods = (:mma, :sp, Symbol("cta_group::", cg), Symbol("kind::", kind),
            (ashift ? (:ashift,) : ())...,
            (collmod === nothing ? () : (collmod,))...)
    maskN = cg == 1 ? 4 : 8
    stem = "llvm.nvvm.tcgen05.mma.sp." * (tmem_a ? "tensor" : "shared")
    sh = ashift ? ".ashift" : ""
    aT = tmem_a ? UInt32 : UInt64
    aexpr = tmem_a ? :(_tmem(a)) : :a
    function reg(name)
        name in TCGEN05_MMA_SP_DENSE_INTRINSIC_NAMES ||
            push!(TCGEN05_MMA_SP_DENSE_INTRINSIC_NAMES, name)
        NVVM.IntrinsicCall{Symbol(name)}()
    end

    # [d], a, b, [sp-meta], idesc, enable
    call = reg(stem * sh)
    @eval @inline function (::Operation{:tcgen05, $mods})(
            d::UInt32, a::$aT, b_desc::UInt64, sp_meta::UInt32,
            idesc::UInt32, enable_input_d::Bool)
        $call(_tmem(d), $aexpr, b_desc, idesc, enable_input_d,
              _tmem(sp_meta), Val($kindval), Val($cg), Val($collval))
    end

    # [d], a, b, [sp-meta], idesc, {disable-output-lane}, enable
    call = reg(stem * ".disable_output_lane.cg$cg" * sh)
    @eval @inline function (::Operation{:tcgen05, $mods})(
            d::UInt32, a::$aT, b_desc::UInt64, sp_meta::UInt32,
            idesc::UInt32, mask::NTuple{$maskN, UInt32},
            enable_input_d::Bool)
        $call(_tmem(d), $aexpr, b_desc, idesc, enable_input_d,
              _tmem(sp_meta), _tc_vec(mask), Val($kindval), Val($collval))
    end

    scale_ok || return nothing

    # [d], a, b, [sp-meta], idesc, enable, scale-input-d
    call = reg(stem * ".scale_d" * sh)
    @eval @inline function (::Operation{:tcgen05, $mods})(
            d::UInt32, a::$aT, b_desc::UInt64, sp_meta::UInt32,
            idesc::UInt32, enable_input_d::Bool, scale::Val)
        $call(_tmem(d), $aexpr, b_desc, idesc, enable_input_d,
              _tmem(sp_meta), scale, Val($kindval), Val($cg), Val($collval))
    end

    # [d], a, b, [sp-meta], idesc, {disable-output-lane}, enable, scale
    call = reg(stem * ".scale_d.disable_output_lane.cg$cg" * sh)
    @eval @inline function (::Operation{:tcgen05, $mods})(
            d::UInt32, a::$aT, b_desc::UInt64, sp_meta::UInt32,
            idesc::UInt32, mask::NTuple{$maskN, UInt32},
            enable_input_d::Bool, scale::Val)
        $call(_tmem(d), $aexpr, b_desc, idesc, enable_input_d,
              _tmem(sp_meta), scale, _tc_vec(mask), Val($kindval),
              Val($collval))
    end
    nothing
end

for (kind, kindval, scale_ok) in _TCGEN05_DENSE_KINDS, cg in (1, 2),
        (collmod, collval) in _TCGEN05_DENSE_COLLECTORS
    _tcgen05_sp_register(kind, kindval, scale_ok, cg, false, false,
                         collmod, collval)
    _tcgen05_sp_register(kind, kindval, scale_ok, cg, true, false,
                         collmod, collval)
    collval < 2 &&
        _tcgen05_sp_register(kind, kindval, scale_ok, cg, true, true,
                             collmod, collval)
end

# --- weight-stationary mma (generated) --------------------------------------------
# tcgen05.mma.ws (PTX 9.3 §9.7.17.10.9.3) is cta_group::1-only. The
# collector buffer belongs to matrix B and is addressed:
# `collector::bN::op` with N ∈ 0:3; spelled-nothing defaults to
# b0::discard, so every other buffer×op pair is an explicit modifier.
# The optional zero-column-mask descriptor is a runtime 64-bit operand
# trailing enable-input-d (separate .zero_col_mask records). The sp
# forms insert the sparsity-metadata TMEM address between the B
# descriptor and idesc. Empirical NVVM enums (llc 22.1.7): the buffer
# immarg is identity (0..3 → b0..b3); the op immarg matches dense
# (0=discard, 1=lastuse, 2=fill, 3=use).
const _TCGEN05_WS_COLLECTORS = begin
    out = Tuple{Any, Int, Int}[(nothing, 0, 0)]
    for buf in 0:3, (op, opval) in ((:discard, 0), (:lastuse, 1),
                                    (:fill, 2), (:use, 3))
        buf == 0 && op === :discard && continue
        push!(out, (Symbol("collector::b$buf::$op"), buf, opval))
    end
    Tuple(out)
end

const TCGEN05_MMA_WS_INTRINSIC_NAMES = String[]

function _tcgen05_ws_register(kind::Symbol, kindval::Int, sp::Bool,
                              tmem_a::Bool, collmod, bufval::Int,
                              opval::Int)
    mods = (:mma, :ws, (sp ? (:sp,) : ())..., Symbol("cta_group::1"),
            Symbol("kind::", kind),
            (collmod === nothing ? () : (collmod,))...)
    stem = "llvm.nvvm.tcgen05.mma.ws." * (sp ? "sp." : "") *
           (tmem_a ? "tensor" : "shared")
    aT = tmem_a ? UInt32 : UInt64
    aexpr = tmem_a ? :(_tmem(a)) : :a
    meta_arg = sp ? (:(sp_meta::UInt32),) : ()
    meta_val = sp ? (:(_tmem(sp_meta)),) : ()
    function reg(name)
        name in TCGEN05_MMA_WS_INTRINSIC_NAMES ||
            push!(TCGEN05_MMA_WS_INTRINSIC_NAMES, name)
        NVVM.IntrinsicCall{Symbol(name)}()
    end

    # [d], a, b{, [sp-meta]}, idesc, enable
    call = reg(stem)
    @eval @inline function (::Operation{:tcgen05, $mods})(
            d::UInt32, a::$aT, b_desc::UInt64, $(meta_arg...),
            idesc::UInt32, enable_input_d::Bool)
        $call(_tmem(d), $aexpr, b_desc, idesc, enable_input_d,
              $(meta_val...), Val($kindval), Val($bufval), Val($opval))
    end

    # [d], a, b{, [sp-meta]}, idesc, enable, zero-column-mask-desc
    call = reg(stem * ".zero_col_mask")
    @eval @inline function (::Operation{:tcgen05, $mods})(
            d::UInt32, a::$aT, b_desc::UInt64, $(meta_arg...),
            idesc::UInt32, enable_input_d::Bool, zero_col_mask::UInt64)
        $call(_tmem(d), $aexpr, b_desc, idesc, enable_input_d,
              $(meta_val...), zero_col_mask,
              Val($kindval), Val($bufval), Val($opval))
    end
    nothing
end

for (kind, kindval, _) in _TCGEN05_DENSE_KINDS, sp in (false, true),
        tmem_a in (false, true),
        (collmod, bufval, opval) in _TCGEN05_WS_COLLECTORS
    _tcgen05_ws_register(kind, kindval, sp, tmem_a, collmod, bufval, opval)
end

# --- mx block-scaled mma (asm tier) ------------------------------------------
#
# PTX 9.3 §9.7.17.10.9.1 gives MX forms a different grammar from dense
# tcgen05.mma: there is no disable-output-lane vector.  Instead, both scale
# matrices are mandatory TMEM address operands:
#
#   [d], a-desc/[a], b-desc, idesc, [scale-A], [scale-B], enable-input-d
#
# Require an explicit scale-vector/block qualifier even where the ISA defines
# a default.  Besides making the semantic choice visible at the call site,
# this keeps the target contract visible: `.scale_vec::*` is the sm_100a /
# sm_110a spelling, while `.block16`/`.block32` is the family-target spelling.
# A call using the old five-argument shape or a kind without `.block_scale`
# therefore misses these methods and stops at the typed-wrapper-only boundary.
#
# This table is the exact Table 60 cross-product after removing aliases:
# (kind, explicit scale-vector spelling, equivalent block-size spelling).
const _TCGEN05_MX_SCALE_VARIANTS = (
    (:mxf8f6f4, Symbol("scale_vec::1X"), :block32),
    (:mxf4,     Symbol("scale_vec::2X"), :block32),
    (:mxf4nvf4, Symbol("scale_vec::2X"), :block32),
    (:mxf4nvf4, Symbol("scale_vec::4X"), :block16),
)

function _tcgen05_mx_register(kind::Symbol, scale::Symbol, cta_group::Int)
    cta_group in (1, 2) || throw(ArgumentError("invalid tcgen05 cta_group: $cta_group"))
    cg = Symbol("cta_group::", cta_group)
    kmod = Symbol("kind::", kind)
    mods = (:mma, cg, kmod, :block_scale, scale)
    head = "tcgen05.mma.$cg.$kmod.block_scale.$scale"

    # Shared-memory A descriptor form.
    let asm = "$head [\$0], \$1, \$2, \$3, [\$4], [\$5], \$6;",
            constraints = "r,l,l,r,r,r,b,~{memory}"
        @eval @inline function (::Operation{:tcgen05, $mods})(
                d::UInt32, a_desc::UInt64, b_desc::UInt64, idesc::UInt32,
                scale_a::UInt32, scale_b::UInt32, enable_input_d::Bool)
            @asmcall($asm, $constraints, true, Nothing,
                     Tuple{UInt32, UInt64, UInt64, UInt32,
                           UInt32, UInt32, Bool},
                     d, a_desc, b_desc, idesc,
                     scale_a, scale_b, enable_input_d)
            nothing
        end
    end

    # Tensor-memory A address form.  UInt32 versus UInt64 on operand two makes
    # the two ISA alternatives unambiguous at Julia dispatch.
    let asm = "$head [\$0], [\$1], \$2, \$3, [\$4], [\$5], \$6;",
            constraints = "r,r,l,r,r,r,b,~{memory}"
        @eval @inline function (::Operation{:tcgen05, $mods})(
                d::UInt32, a_tmem::UInt32, b_desc::UInt64, idesc::UInt32,
                scale_a::UInt32, scale_b::UInt32, enable_input_d::Bool)
            @asmcall($asm, $constraints, true, Nothing,
                     Tuple{UInt32, UInt32, UInt64, UInt32,
                           UInt32, UInt32, Bool},
                     d, a_tmem, b_desc, idesc,
                     scale_a, scale_b, enable_input_d)
            nothing
        end
    end
    nothing
end

for (kind, scale_vec, block) in _TCGEN05_MX_SCALE_VARIANTS,
        scale in (scale_vec, block), cta_group in (1, 2)
    _tcgen05_mx_register(kind, scale, cta_group)
end

# Exact integer-address adapters for the reviewed tcgen05 surface. PTX 9.3
# §9.7.17 does not give the family one uniform operand convention: most forms
# bracket operand 1, `dealloc` brackets no operand, and MX block-scale MMA also
# brackets its scale descriptors plus an optional tensor-memory A operand.
# Encode the complete Julia signature, including every Address role, so an
# arity/carrier/role miss reaches the typed-wrapper-only rejection instead of
# silently shedding one marker and entering generic asm. Generic-address alloc
# and pointer commit forms need no entry because `address(::Core.LLVMPtr)` is
# identity.
struct TCGen05IntegerAddressAdapter
    mods::Tuple{Vararg{Symbol}}
    argtypes::Tuple{Vararg{Type}}
end

const TCGEN05_INTEGER_ADDRESS_ADAPTERS = let
    specs = TCGen05IntegerAddressAdapter[]
    A32 = Address{UInt32}
    add(mods, argtypes) =
        push!(specs, TCGen05IntegerAddressAdapter(mods, argtypes))
    for cg in 1:2
        cta = Symbol("cta_group::", cg)
        add((:shift, cta, :down), (A32,))
        for shapemods in ((Symbol("128x256b"),), (Symbol("4x256b"),),
                          (Symbol("128x128b"),),
                          (Symbol("64x128b"), Symbol("warpx2::02_13")),
                          (Symbol("64x128b"), Symbol("warpx2::01_23")),
                          (Symbol("32x128b"), :warpx4)),
                fmt in ((), (:b8x16, :b6x16_p32), (:b8x16, :b4x16_p64))
            add((:cp, cta, shapemods..., fmt...), (A32, UInt64))
        end
    end
    for (shape, base, counts) in (
            (Symbol("16x64b"),   1, (1, 2, 4, 8, 16, 32, 64, 128)),
            (Symbol("32x32b"),   1, (1, 2, 4, 8, 16, 32, 64, 128)),
            (Symbol("16x128b"),  2, (1, 2, 4, 8, 16, 32, 64)),
            (Symbol("16x256b"),  4, (1, 2, 4, 8, 16, 32)),
            (Symbol("16x32bx2"), 1, (1, 2, 4, 8, 16, 32, 64, 128)))
        split = shape === Symbol("16x32bx2") ? (Val,) : ()
        for count in counts, op in (:ld, :st), repack in (false, true)
            flag = repack ?
                (Symbol(op === :ld ? "pack::16b" : "unpack::16b"),) : ()
            mods = (op, :sync, :aligned, shape, Symbol("x", count),
                    flag..., :b32)
            if op === :ld
                add(mods, (A32, split...))
            else
                add(mods, (A32, split..., NTuple{base * count, UInt32}))
            end
        end
    end
    for cg in 1:2
        cta = Symbol("cta_group::", cg)
        add((:alloc, cta, :sync, :aligned,
             Symbol("shared::cta"), :b32), (A32, UInt32))
        for space in (Symbol("shared::cta"), Symbol("shared::cluster"))
            add((:commit, cta, Symbol("mbarrier::arrive::one"), space, :b64),
                (A32,))
        end
        add((:commit, cta, Symbol("mbarrier::arrive::one"),
             Symbol("multicast::cluster"),
             Symbol("shared::cluster"), :b64), (A32, Integer))
        for (kind, _, scale_ok) in _TCGEN05_DENSE_KINDS,
                (collmod, collval) in _TCGEN05_DENSE_COLLECTORS,
                (tmem_a, ashift) in ((false, false), (true, false),
                                     (true, true))
            ashift && collval >= 2 && continue
            mods = (:mma, cta, Symbol("kind::", kind),
                    (ashift ? (:ashift,) : ())...,
                    (collmod === nothing ? () : (collmod,))...)
            aT = tmem_a ? A32 : UInt64
            maskT = NTuple{cg == 1 ? 4 : 8, UInt32}
            add(mods, (A32, aT, UInt64, UInt32, Bool))
            add(mods, (A32, aT, UInt64, UInt32, maskT, Bool))
            if scale_ok
                add(mods, (A32, aT, UInt64, UInt32, Bool, Val))
                add(mods, (A32, aT, UInt64, UInt32, maskT, Bool, Val))
            end
            spmods = (:mma, :sp, mods[2:end]...)
            add(spmods, (A32, aT, UInt64, A32, UInt32, Bool))
            add(spmods, (A32, aT, UInt64, A32, UInt32, maskT, Bool))
            if scale_ok
                add(spmods, (A32, aT, UInt64, A32, UInt32, Bool, Val))
                add(spmods, (A32, aT, UInt64, A32, UInt32, maskT, Bool, Val))
            end
        end
    end
    for (kind, _, _) in _TCGEN05_DENSE_KINDS, sp in (false, true),
            tmem_a in (false, true),
            (collmod, _, _) in _TCGEN05_WS_COLLECTORS
        mods = (:mma, :ws, (sp ? (:sp,) : ())..., Symbol("cta_group::1"),
                Symbol("kind::", kind),
                (collmod === nothing ? () : (collmod,))...)
        aT = tmem_a ? A32 : UInt64
        meta = sp ? (A32,) : ()
        add(mods, (A32, aT, UInt64, meta..., UInt32, Bool))
        add(mods, (A32, aT, UInt64, meta..., UInt32, Bool, UInt64))
    end
    for (kind, scale_vec, block) in _TCGEN05_MX_SCALE_VARIANTS,
            scale in (scale_vec, block), cg in 1:2
        mods = (:mma, Symbol("cta_group::", cg), Symbol("kind::", kind),
                :block_scale, scale)
        add(mods, (A32, UInt64, UInt64, UInt32, A32, A32, Bool))
        add(mods, (A32, A32, UInt64, UInt32, A32, A32, Bool))
    end
    keys = ((s.mods, s.argtypes) for s in specs)
    allunique(keys) || error("duplicate tcgen05 integer-address adapter signature")
    Tuple(specs)
end

const TCGEN05_INTEGER_ADDRESS_FORMS =
    Tuple(unique(s.mods for s in TCGEN05_INTEGER_ADDRESS_ADAPTERS))

for spec in TCGEN05_INTEGER_ADDRESS_ADAPTERS
    mods, argtypes = spec.mods, spec.argtypes
    names = [gensym(:arg) for _ in argtypes]
    decls = [:($(names[i])::$(argtypes[i])) for i in eachindex(argtypes)]
    args = [argtypes[i] <: Address ? :($(names[i]).value) : names[i]
            for i in eachindex(argtypes)]
    @eval @inline function (op::Operation{:tcgen05, $mods})($(decls...))
        op($(args...))
    end
end
