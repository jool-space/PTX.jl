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

function _tcgen05_shift_register(cta::Int)
    cta_part = Symbol("cta_group::", cta)
    mods = (:shift, cta_part, :down)
    asm = "tcgen05.shift.cta_group::$cta.down [\$0];"
    @eval function (::Operation{:tcgen05, $mods})(taddr::UInt32)
        Base.@inline
        @asmcall($asm, "r,~{memory}", true, Nothing, Tuple{UInt32}, taddr)
        nothing
    end
    nothing
end

for cta in (1, 2)
    _tcgen05_shift_register(cta)
end

# Unbracketed taddr per pyptx.
function _tcgen05_dealloc_register(cta::Int)
    cta_part = Symbol("cta_group::", cta)
    mods = (:dealloc, cta_part, :sync, :aligned, :b32)
    asm = "tcgen05.dealloc.cta_group::$cta.sync.aligned.b32 \$0, \$1;"
    @eval function (::Operation{:tcgen05, $mods})(taddr::UInt32, ncols::UInt32)
        Base.@inline
        @asmcall($asm, "r,r,~{memory}", true, Nothing,
                 Tuple{UInt32, UInt32}, taddr, ncols)
        nothing
    end
    nothing
end

for cta in (1, 2)
    _tcgen05_dealloc_register(cta)
end

# Common single-warp shapes (no multicast). 64x128b / 32x128b need multicast
# qualifiers — separate wrapper wave.
function _tcgen05_cp_register(cta::Int, shape::Symbol)
    mods = (:cp, Symbol("cta_group::", cta), shape)
    asm = "tcgen05.cp.cta_group::$cta.$shape [\$0], \$1;"
    @eval function (::Operation{:tcgen05, $mods})(taddr::UInt32, s_desc::UInt64)
        Base.@inline
        @asmcall($asm, "r,l,~{memory}", true, Nothing,
                 Tuple{UInt32, UInt64}, taddr, s_desc)
        nothing
    end
    nothing
end

for cta in (1, 2), shape in (Symbol("128x256b"), Symbol("4x256b"), Symbol("128x128b"))
    _tcgen05_cp_register(cta, shape)
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

# --- alloc / relinquish / wait / commit / mma (blackwell-1) ----------------
# Added for the tcgen05 smoke + roundtrip ports (ROADMAP item 10, first wave).
# All addresses are 32-bit (TMEM taddr or .shared::cta SMEM offset) and render
# as `[$N]` with the `r` constraint — same discipline as shift/cp above, NOT
# the chain default (which would bracket an LLVMPtr with the 64-bit `l`
# constraint). Callers pass `smem_addr_u32(pointer(slot))`.

# `tcgen05.alloc` writes the allocated TMEM base into a `.shared::cta` slot.
function _tcgen05_alloc_register(cta::Int)
    mods = (:alloc, Symbol("cta_group::", cta), :sync, :aligned,
            Symbol("shared::cta"), :b32)
    asm = "tcgen05.alloc.cta_group::$cta.sync.aligned.shared::cta.b32 [\$0], \$1;"
    @eval function (::Operation{:tcgen05, $mods})(dst::UInt32, ncols::UInt32)
        Base.@inline
        @asmcall($asm, "r,r,~{memory}", true, Nothing,
                 Tuple{UInt32, UInt32}, dst, ncols)
        nothing
    end
    nothing
end

for cta in (1, 2)
    _tcgen05_alloc_register(cta)
end

# `tcgen05.relinquish_alloc_permit` — no operands.
function _tcgen05_relinquish_register(cta::Int)
    mods = (:relinquish_alloc_permit, Symbol("cta_group::", cta), :sync, :aligned)
    asm = "tcgen05.relinquish_alloc_permit.cta_group::$cta.sync.aligned;"
    @eval function (::Operation{:tcgen05, $mods})()
        Base.@inline
        @asmcall($asm, "~{memory}", true, Nothing, Tuple{})
        nothing
    end
    nothing
end

for cta in (1, 2)
    _tcgen05_relinquish_register(cta)
end

# `tcgen05.wait::ld` / `tcgen05.wait::st` — drain prior tcgen05.ld / .st
# before the registers / TMEM are reused. No operands.
function _tcgen05_wait_register(which::Symbol)
    w = Symbol("wait::", which)
    mods = (w, :sync, :aligned)
    asm = "tcgen05.wait::$which.sync.aligned;"
    @eval function (::Operation{:tcgen05, $mods})()
        Base.@inline
        @asmcall($asm, "~{memory}", true, Nothing, Tuple{})
        nothing
    end
    nothing
end

for which in (:ld, :st)
    _tcgen05_wait_register(which)
end

# `tcgen05.commit` — arrive-on-one of an mbarrier after the MMA group
# retires. Single 32-bit shared mbarrier address. The `.multicast::cluster`
# + u16 mask variant (cta_group::2 datacenter cluster GEMM) is deferred.
function _tcgen05_commit_register(cta::Int, space::Symbol)
    mods = (:commit, Symbol("cta_group::", cta),
            Symbol("mbarrier::arrive::one"), Symbol("shared::", space), :b64)
    asm = "tcgen05.commit.cta_group::$cta.mbarrier::arrive::one." *
          "shared::$space.b64 [\$0];"
    @eval function (::Operation{:tcgen05, $mods})(mbar::UInt32)
        Base.@inline
        @asmcall($asm, "r,~{memory}", true, Nothing, Tuple{UInt32}, mbar)
        nothing
    end
    nothing
end

for cta in (1, 2), space in (:cta, :cluster)
    _tcgen05_commit_register(cta, space)
end

# `tcgen05.mma` — dense F16/BF16/TF32/FP8 path (ROADMAP item 10, first
# wave). Form (PTX 9.2 §9.7.16.4, dense, ptx_version >= 8.8):
#
#   tcgen05.mma.cta_group::N.kind::K [d], a_desc, b_desc, idesc,
#                                    {m0..m_{W-1}}, p;
#
# `d` is a 32-bit TMEM accumulator address; `a_desc`/`b_desc` are 64-bit
# SMEM descriptors (`tcgen05_descriptor`); `idesc` is the 32-bit UMMA
# instruction descriptor (`tcgen05_instr_desc_f16bf16_f32`). The brace
# vector is the disable-output-lane mask — W = 4 words for cta_group::1
# (≤128 lanes), 8 for cta_group::2 (256 lanes); all-zero disables nothing,
# which is the dense CUTLASS/CuTe form. `p` is the enable-input-D /
# accumulate predicate. Deferred to a later wave: the trailing SCALE_C
# immediate, `.sp` (sparse) + `[metadata]`, `.ashift`, `.collector::a::*`,
# and the A-operand-in-TMEM variant.
function _tcgen05_mma_register(cta::Int, kind::Symbol)
    nmask = cta == 2 ? 8 : 4
    mask_slots = join(("\$$(4 + i)" for i in 0:nmask-1), ", ")
    pslot = 4 + nmask
    mods = (:mma, Symbol("cta_group::", cta), Symbol("kind::", kind))
    asm = "tcgen05.mma.cta_group::$cta.kind::$kind " *
          "[\$0], \$1, \$2, \$3, {$mask_slots}, \$$pslot;"
    # d(r, bracketed) a(l) b(l) idesc(r) nmask×mask(r) p(b)
    constraints = join(vcat(["r", "l", "l", "r"],
                            fill("r", nmask), ["b", "~{memory}"]), ",")
    masks = fill(:(UInt32(0)), nmask)
    flat  = vcat([:UInt32, :UInt64, :UInt64, :UInt32],
                 fill(:UInt32, nmask), [:Bool])
    @eval function (::Operation{:tcgen05, $mods})(
            d::UInt32, a_desc::UInt64, b_desc::UInt64,
            idesc::UInt32, enable_input_d::Bool)
        Base.@inline
        @asmcall($asm, $constraints, true, Nothing,
                 Tuple{$(flat...)},
                 d, a_desc, b_desc, idesc, $(masks...), enable_input_d)
        nothing
    end
    nothing
end

# Kinds per PTX 9.2 §9.7.16.4 (pyptx `_TCGEN05_KINDS`).
for cta in (1, 2),
    kind in (:tf32, :f16, :i8, :f8f6f4, :mxf8f6f4, :mxf4, :mxf4nvf4)
    _tcgen05_mma_register(cta, kind)
end
