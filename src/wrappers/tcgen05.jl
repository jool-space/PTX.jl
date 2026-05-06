# Design ported from pyptx/pyptx/ptx.py `_Tcgen05`
# (https://github.com/patrick-toulme/pyptx).
# Copyright 2026 Patrick Toulmé. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Translated to Julia and adapted.

# `tcgen05.*` wrappers for Blackwell ops whose `taddr` operand is a 32-bit
# TMEM address (returned by `tcgen05.alloc`), NOT a memory pointer. Chain
# default brackets `LLVMPtr` args but treats `UInt32` as a plain `r`-constraint
# scalar — wrong for shift / cp / ld / st where taddr renders as `[%rN]`.
# `dealloc` takes the same UInt32 taddr unbracketed.
#
# Operand discipline (per pyptx + PTX 9.2 §9.7.16):
#   alloc   — chain default (LLVMPtr destination + ncols)
#   dealloc — taddr (u32, unbracketed), ncols (u32)
#   shift   — [taddr]
#   cp      — [taddr], s_desc::u64
#   ld      — multi-output NTuple{N, UInt32}, [taddr]
#   st      — [taddr], NTuple{N, UInt32}
#   mma     — deferred (8 operands incl. mask tuple + predicate)
#
# ld/st per-lane register count = `base * count` where
#   base = 1 for shape ∈ {16x64b, 32x32b}; 2 for 16x128b; 4 for 16x256b
# (PTX 9.2 §9.7.16.8.3 Table 49). count ∈ {x1..x128} with limit per_lane ≤ 128.
# Shape `.16x32bx2` excluded — needs an extra `immHalfSplitoff` immediate.

for cta in (1, 2)
    cta_part = Symbol("cta_group::", cta)
    mods = (:shift, cta_part, :down)
    asm = "tcgen05.shift.cta_group::$cta.down [\$0];"
    @eval function (::Operation{:tcgen05, $mods})(taddr::UInt32)
        Base.@inline
        @asmcall($asm, "r,~{memory}", true, Nothing, Tuple{UInt32}, taddr)
        nothing
    end
end

# Unbracketed taddr per pyptx.
for cta in (1, 2)
    cta_part = Symbol("cta_group::", cta)
    mods = (:dealloc, cta_part, :sync, :aligned, :b32)
    asm = "tcgen05.dealloc.cta_group::$cta.sync.aligned.b32 \$0, \$1;"
    @eval function (::Operation{:tcgen05, $mods})(taddr::UInt32, ncols::UInt32)
        Base.@inline
        @asmcall($asm, "r,r,~{memory}", true, Nothing,
                 Tuple{UInt32, UInt32}, taddr, ncols)
        nothing
    end
end

# Common single-warp shapes (no multicast). 64x128b / 32x128b need multicast
# qualifiers — separate wrapper wave.
for cta in (1, 2), shape in (Symbol("128x256b"), Symbol("4x256b"), Symbol("128x128b"))
    mods = (:cp, Symbol("cta_group::", cta), shape)
    asm = "tcgen05.cp.cta_group::$cta.$shape [\$0], \$1;"
    @eval function (::Operation{:tcgen05, $mods})(taddr::UInt32, s_desc::UInt64)
        Base.@inline
        @asmcall($asm, "r,l,~{memory}", true, Nothing,
                 Tuple{UInt32, UInt64}, taddr, s_desc)
        nothing
    end
end

const _TCGEN05_LDST_SHAPES = ((Symbol("16x64b"), 1), (Symbol("32x32b"), 1),
                              (Symbol("16x128b"), 2), (Symbol("16x256b"), 4))
const _TCGEN05_LDST_COUNTS = (:x1, :x2, :x4, :x8, :x16, :x32, :x64, :x128)
const _TCGEN05_COUNT_VALUE = Dict(:x1 => 1, :x2 => 2, :x4 => 4, :x8 => 8,
                                  :x16 => 16, :x32 => 32, :x64 => 64, :x128 => 128)

function _tcgen05_ld_register(shape::Symbol, count::Symbol, base::Int)
    n = base * _TCGEN05_COUNT_VALUE[count]
    n > 128 && return nothing   # NA per Table 49

    mods = (:ld, :sync, :aligned, shape, count, :b32)
    out_slots = "{" * join(("\$$i" for i in 0:n-1), ", ") * "}"
    asm = "tcgen05.ld.sync.aligned.$shape.$count.b32 $out_slots, [\$$n];"
    constraints = join(vcat(fill("=r", n), ["r", "~{memory}"]), ",")

    if n == 1
        @eval function (::Operation{:tcgen05, $mods})(taddr::UInt32)
            Base.@inline
            @asmcall($asm, $constraints, true, UInt32, Tuple{UInt32}, taddr)
        end
    else
        @eval function (::Operation{:tcgen05, $mods})(taddr::UInt32)
            Base.@inline
            @asmcall($asm, $constraints, true, NTuple{$n, UInt32},
                     Tuple{UInt32}, taddr)
        end
    end
    nothing
end

function _tcgen05_st_register(shape::Symbol, count::Symbol, base::Int)
    n = base * _TCGEN05_COUNT_VALUE[count]
    n > 128 && return nothing

    mods = (:st, :sync, :aligned, shape, count, :b32)
    src_slots = "{" * join(("\$$i" for i in 1:n), ", ") * "}"
    asm = "tcgen05.st.sync.aligned.$shape.$count.b32 [\$0], $src_slots;"
    constraints = join(vcat(["r"], fill("r", n), ["~{memory}"]), ",")

    src_args = [:(src[$i]) for i in 1:n]
    flat = fill(:UInt32, n)

    @eval function (::Operation{:tcgen05, $mods})(taddr::UInt32, src::NTuple{$n, UInt32})
        Base.@inline
        @asmcall($asm, $constraints, true, Nothing,
                 Tuple{UInt32, $(flat...)}, taddr, $(src_args...))
        nothing
    end
    nothing
end

for (shape, base) in _TCGEN05_LDST_SHAPES, count in _TCGEN05_LDST_COUNTS
    _tcgen05_ld_register(shape, count, base)
    _tcgen05_st_register(shape, count, base)
end
