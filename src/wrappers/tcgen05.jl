# Design ported from pyptx/pyptx/ptx.py `_Tcgen05`
# (https://github.com/patrick-toulme/pyptx).
# Copyright 2026 Patrick Toulmé. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Translated to Julia and adapted.

# `tcgen05.*` — mixed-route family. The notation surface is unchanged:
# taddr operands stay raw `UInt32` TMEM addresses (returned by
# `tcgen05.alloc`) and SMEM operands stay 32-bit offsets from
# `smem_addr_u32`; where a wrapper still routes to an intrinsic it retypes
# them into the address spaces the intrinsics want (`reinterpret_addrspace`
# — TMEM is addrspace(6), 32-bit in the backend datalayout, so taddr
# operands stay 32-bit registers in the emitted PTX).
#
# The management verbs — alloc, dealloc, relinquish_alloc_permit, commit,
# and the complete cp grid — are single-route convergent inline asm, like
# mbarrier and the waits/fences below. That split was retired deliberately:
#   - every one is an observable async/lifecycle effect — there is no
#     CSE/LICM for intrinsic attributes to unlock, so the tier-2 route
#     bought bookkeeping (per-intrinsic selection probes, ceiling checks,
#     name-flattening tables) and no optimization. The argmem-widen A/B on
#     B200 (branch agent/argmem-widen-b200) proved the point from the other
#     direction: widening these families' memory precision to the asm
#     route's conservative clobber left the FA/GEMM/b128 instruction
#     streams byte-identical at sm_100a;
#   - the PTX ISA 9.4 additions (.exclusive alloc/dealloc, the ::16b/::32b
#     multicast commits) have no upstream intrinsics and were already asm —
#     one route instead of two.
# Emitted-PTX deltas vs the intrinsic route, reviewed at demotion:
#   - the generic-address alloc form spells the literal (no `.shared::cta`;
#     ISel printed the qualifier explicitly — same operation, the ISA
#     requires dst to lie in the CTA shared window either way);
#   - commit keeps pyptx's `.multicast::cluster.shared::cluster` modifier
#     order (ISel reversed it; ptxas accepts both). Its `.shared::cta`
#     NOTATION still renders `.shared::cluster`: the §9.7.18.12.1 syntax
#     block admits only {.shared::cluster} on commit — no ::cta spelling
#     exists, and ptxas rejects one ("State space incorrect") — so the
#     ::cta-modified wrapper is an address-species surface (a shared::cta
#     mbar address lies inside the cluster window) and the render follows
#     the ISA, exactly as the intrinsic route printed it;
#   - pointer/static-SMEM operands materialize through mov/cvt instead of
#     folding into the operand (ptxas folds these in SASS).
#
# Intrinsic mapping for the surviving tier-2 forms (probed against llc
# 22.1.7, sm_100a):
#   - shift → shift.down.cgN: 1:1, identical spelling.
#   - ld/st → ld.<shape>.<count> / st.<shape>.<count>: identical
#     spellings. The data moves as an LLVM vector (v<N>i32), so the
#     wrappers repack to/from the notation surface's plain NTuple{N,
#     UInt32}; the i1 immarg is the .pack::16b/.unpack::16b flag,
#     surfaced as an explicit modifier (plain spellings pass Val(false)).
#     The 16x32bx2 shape carries its extra `immHalfSplitoff` i64 immarg
#     as a positional `Val(off)` operand after taddr.
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
# `tcgen05.ld.red` (load-with-reduction, PTX 8.8) has no NVVM intrinsic
# records at the pinned backend and is a single-route asm family below.
#
# Irregular families are written out literally so every intrinsic they
# stand on is greppable — test/host/conformance.jl scans for `nvvm"..."`
# literals and requires a probe for each. The regular grids (ld/st,
# dense mma) are mechanical Table expansions too large for that; they are
# generated from one spec each and record their tier-2 names in the wrapper
# registry (families :tcgen05_ldst / :tcgen05_mma_{dense,sp,ws}) that
# conformance replays with the same probe-per-name requirement plus a
# registry-equality pin.

# taddr retype for the intrinsic-routed forms (see address_space.jl).
@inline _tmem(taddr::UInt32) = reinterpret_addrspace(Val(AS.Tmem), taddr)

# tcgen05.ld returns / tcgen05.st takes LLVM vectors; the notation surface
# uses plain tuples (scalar for the single-register forms).
@inline _tc_unvec(x::UInt32) = x
@inline _tc_unvec(v::NTuple{N, VecElement{UInt32}}) where {N} =
    ntuple(i -> v[i].value, Val(N))
@inline _tc_vec(t::NTuple{N, UInt32}) where {N} =
    ntuple(i -> VecElement(t[i]), Val(N))

# Exact integer-address adapters for the reviewed tcgen05 surface. PTX 9.3
# §9.7.17 does not give the family one uniform operand convention: most forms
# bracket operand 1, `dealloc` brackets no operand, and MX block-scale MMA also
# brackets its scale descriptors plus an optional tensor-memory A operand.
# Encode the complete Julia signature, including every Address role, so an
# arity/carrier/role miss reaches the typed-wrapper-only rejection instead of
# silently shedding one marker and entering generic asm. Generic-address alloc
# and pointer commit forms need no entry because `address(::Core.LLVMPtr)` is
# identity.
#
# Adapters are emitted by the SAME enumeration that emits (or, for the
# literal families, sits next to) the primary methods: each registration
# site calls `_tcgen05_adapter!` with the exact signature it reviewed, and
# `TCGEN05_INTEGER_ADDRESS_ADAPTERS` at the bottom of this file is the
# collected result. The independent double-entry copy of this grid lives in
# test/host/address_roles.jl and pins both the 434 form set and the 1218
# signature set.
struct TCGen05IntegerAddressAdapter
    mods::Tuple{Vararg{Symbol}}
    argtypes::Tuple{Vararg{Type}}
end

const _TCGEN05_ADAPTER_SPECS = TCGen05IntegerAddressAdapter[]

# Record the adapter signature and define the Address-unwrapping method that
# forwards to the primary typed wrapper. Duplicate signatures are rejected
# when the closed inventory is sealed below.
function _tcgen05_adapter!(mods::Tuple{Vararg{Symbol}}, argtypes::Type...)
    push!(_TCGEN05_ADAPTER_SPECS,
          TCGen05IntegerAddressAdapter(mods, argtypes))
    names = [gensym(:arg) for _ in argtypes]
    decls = [:($(names[i])::$(argtypes[i])) for i in eachindex(names)]
    args = [argtypes[i] <: Address ? :($(names[i]).value) : names[i]
            for i in eachindex(names)]
    @eval @inline function (op::Operation{:tcgen05, $mods})($(decls...))
        op($(args...))
    end
    nothing
end

# --- shift / dealloc / cp -----------------------------------------------------

@inline optype"tcgen05.shift.cta_group::1.down"(taddr::UInt32) =
    ceiled(nvvm"tcgen05.shift.down.cg1", ptx"tcgen05.shift.cta_group::1.down")(
        _tmem(taddr))
@inline optype"tcgen05.shift.cta_group::2.down"(taddr::UInt32) =
    ceiled(nvvm"tcgen05.shift.down.cg2", ptx"tcgen05.shift.cta_group::2.down")(
        _tmem(taddr))

# dealloc and the complete cp grid are single-route asm (see header).
# PTX 9.3 §9.7.17.7.1 spells dealloc's taddr as a bare register; cp
# brackets its TMEM destination and takes the SMEM source as a matrix
# descriptor. Shapes carry their ISA-mandated multicast pairing
# (§9.7.17.9: `.64x128b` requires one of the `.warpx2::*` pairings,
# `.32x128b` requires `.warpx4`), each optionally with the
# `.b8x16.{b6x16_p32,b4x16_p64}` decompression pair — the same grid as the
# integer-address adapter loop below.
for cg in 1:2
    cta = Symbol("cta_group::", cg)

    mods = (:dealloc, cta, :sync, :aligned, :b32)
    ir = convergent_asm_ir(
        "tcgen05.dealloc.cta_group::$cg.sync.aligned.b32 \$0, \$1;",
        "r,r,~{memory}", Nothing, (UInt32, UInt32))
    @eval @inline function (::Operation{:tcgen05, $mods})(
            taddr::UInt32, ncols::UInt32)
        Base.llvmcall(($ir, "entry"), Nothing, Tuple{UInt32, UInt32},
                      taddr, ncols)
    end

    for shapemods in ((Symbol("128x256b"),), (Symbol("4x256b"),),
                      (Symbol("128x128b"),),
                      (Symbol("64x128b"), Symbol("warpx2::02_13")),
                      (Symbol("64x128b"), Symbol("warpx2::01_23")),
                      (Symbol("32x128b"), :warpx4)),
            fmt in ((), (:b8x16, :b6x16_p32), (:b8x16, :b4x16_p64))
        mods = (:cp, cta, shapemods..., fmt...)
        spell = join(("tcgen05", "cp", String(cta),
                      String.(shapemods)..., String.(fmt)...), ".")
        ir = convergent_asm_ir("$spell [\$0], \$1;", "r,l,~{memory}",
                               Nothing, (UInt32, UInt64))
        @eval @inline function (::Operation{:tcgen05, $mods})(
                taddr::UInt32, s_desc::UInt64)
            Base.llvmcall(($ir, "entry"), Nothing, Tuple{UInt32, UInt64},
                          taddr, s_desc)
        end
    end
end

# Integer-address adapters for the literal shift/cp methods above (dealloc
# takes a bare register per PTX 9.3 §9.7.17.7.1 — no adapter).
for cg in 1:2
    cta = Symbol("cta_group::", cg)
    _tcgen05_adapter!((:shift, cta, :down), Address{UInt32})
    for shapemods in ((Symbol("128x256b"),), (Symbol("4x256b"),),
                      (Symbol("128x128b"),),
                      (Symbol("64x128b"), Symbol("warpx2::02_13")),
                      (Symbol("64x128b"), Symbol("warpx2::01_23")),
                      (Symbol("32x128b"), :warpx4)),
            fmt in ((), (:b8x16, :b6x16_p32), (:b8x16, :b4x16_p64))
        _tcgen05_adapter!((:cp, cta, shapemods..., fmt...),
                          Address{UInt32}, UInt64)
    end
end

# --- ld / st (Table 52 expansion, generated) ------------------------------------
#
# The full grid is `shape × count × {plain, repack}` for both ld and st,
# generated from one spec (`_TCGEN05_LDST_SHAPES`) — 148 methods over 74
# tier-2 names, each with its integer-address adapter emitted by the same
# registration call. The wrapper registry (family :tcgen05_ldst) records
# every name for the conformance replay (the dense-mma standing guarantee):
# test/host/conformance.jl requires a selection probe per recorded name
# and pins the name set against the NVVM registry, so a grid edit here
# without a matching probe edit is loud.
#
# ld returns / st takes a plain NTuple{base * count, UInt32} (scalar when
# the width is 1); the intrinsics' trailing i1 immarg is the
# `.pack::16b`/`.unpack::16b` flag. 16x32bx2 — two 16x32b accesses, the
# second starting at taddr + immHalfSplitoff — carries its extra i64
# immarg as a positional `Val(off)` operand after taddr.

const _TCGEN05_LDST_SHAPES = (
    # (shape, per-lane base registers, counts, has immHalfSplitoff)
    (Symbol("16x64b"),   1, (1, 2, 4, 8, 16, 32, 64, 128), false),
    (Symbol("32x32b"),   1, (1, 2, 4, 8, 16, 32, 64, 128), false),
    (Symbol("16x128b"),  2, (1, 2, 4, 8, 16, 32, 64),      false),
    (Symbol("16x256b"),  4, (1, 2, 4, 8, 16, 32),          false),
    (Symbol("16x32bx2"), 1, (1, 2, 4, 8, 16, 32, 64, 128), true),
)

function _tcgen05_ldst_register(op::Symbol, shape::Symbol, base::Int,
                                count::Int, split::Bool, repack::Bool)
    n = base * count
    flag = repack ?
        (Symbol(op === :ld ? "pack::16b" : "unpack::16b"),) : ()
    mods = (op, :sync, :aligned, shape, Symbol("x", count), flag..., :b32)
    name = "llvm.nvvm.tcgen05.$op.$shape.x$count"
    call = wrapper_intrinsic_call(:tcgen05_ldst, :tcgen05, mods, name)
    hso_decl = split ? (:(halfsplitoff::Val),) : ()
    hso_arg  = split ? (:halfsplitoff,) : ()
    hso_T    = split ? (Val,) : ()
    if op === :ld
        @eval @inline (::Operation{:tcgen05, $mods})(
                taddr::UInt32, $(hso_decl...)) =
            _tc_unvec($call(_tmem(taddr), $(hso_arg...), Val($repack)))
        _tcgen05_adapter!(mods, Address{UInt32}, hso_T...)
    else
        src = n == 1 ? :(src[1]) : :(_tc_vec(src))
        @eval @inline (::Operation{:tcgen05, $mods})(
                taddr::UInt32, $(hso_decl...), src::NTuple{$n, UInt32}) =
            $call(_tmem(taddr), $(hso_arg...), $src, Val($repack))
        _tcgen05_adapter!(mods, Address{UInt32}, hso_T...,
                          NTuple{n, UInt32})
    end
    nothing
end

for (shape, base, counts, split) in _TCGEN05_LDST_SHAPES,
        count in counts, op in (:ld, :st), repack in (false, true)
    _tcgen05_ldst_register(op, shape, base, count, split, repack)
end

# --- ld.red (load-with-reduction, generated asm family) -------------------------
#
# `tcgen05.ld.red` (PTX 8.8 §9.7.18.8) has no NVVM intrinsic records at the
# pinned backend, so the family is single-route convergent asm under the
# family-wide sideeffect + ~{memory} + convergent nomerge contract
# (`.sync.aligned` = warp-collective, same hazard class as the mbarrier
# family). Target reality: the ISA supports ld.red on sm_110a and the
# sm_103f/sm_110f families only — NEVER sm_100 (ptxas: "Instruction
# 'tcgen05.ld.red' not supported on .target 'sm_100a'"). The ptxas legs
# assemble at sm_103f; runtime evidence needs a CC 10.3+ device.
#
# Grid: shape {32x32b, 16x32bx2} × num {x2..x128} (`.red` requires ≥ .x2) ×
# redOp {min, max} × (f32 × {∅, .abs, .NaN, .abs.NaN} ∪ {u32, s32}) =
# 168 forms. Calls return the flat tuple (data..., redval): count b32 data
# carriers (UInt32) plus the reduction result typed by the trailing dtype
# (Float32 / UInt32 / Int32). 16x32bx2 takes its immHalfSplitoff as a
# positional Val operand baked into the asm text as the ISA's immediate.

const _TCGEN05_LDRED_REDT = (f32 = Float32, u32 = UInt32, s32 = Int32)

function _tcgen05_ldred_ir(mods::Tuple{Vararg{Symbol}}, n::Int,
                           redT::Type, off)
    head = "tcgen05." * join(String.(mods), ".")
    regs = join(("\$$(k - 1)" for k in 1:n), ", ")
    tail = off === nothing ? "" : ", $off"
    asm = "$head {$regs}, \$$n, [\$$(n + 1)]$tail;"
    constraints = join(fill("=r", n), ",") * "," *
                  (redT === Float32 ? "=f" : "=r") * ",r,~{memory}"
    rt = Tuple{fill(UInt32, n)..., redT}
    convergent_asm_ir(asm, constraints, rt, (UInt32,)), rt
end

# The split-shape offset must be a PTX immediate, so its IR is generated
# per Val here (helper above — the generator-world rule).
@generated function _tcgen05_ldred_split(::Operation{:tcgen05, mods},
                                         taddr::UInt32,
                                         ::Val{off}) where {mods, off}
    off isa Integer && off >= 0 ||
        return :(throw(ArgumentError(
            "halfsplitoff must be a non-negative integer Val")))
    n = parse(Int, String(mods[6])[2:end])
    ir, rt = _tcgen05_ldred_ir(mods, n, _TCGEN05_LDRED_REDT[mods[end]],
                               Int(off))
    :(Base.llvmcall(($ir, "entry"), $rt, Tuple{UInt32}, taddr))
end

function _tcgen05_ldred_register(shape::Symbol, count::Int, redop::Symbol,
                                 variant::Tuple{Vararg{Symbol}},
                                 dtype::Symbol)
    mods = (:ld, :red, :sync, :aligned, shape, Symbol("x", count), redop,
            variant..., dtype)
    register_wrapper!(:tcgen05_ldred, :tcgen05, mods, :asm)
    if shape === Symbol("16x32bx2")
        @eval @inline (op::Operation{:tcgen05, $mods})(
                taddr::UInt32, halfsplitoff::Val) =
            _tcgen05_ldred_split(op, taddr, halfsplitoff)
        _tcgen05_adapter!(mods, Address{UInt32}, Val)
    else
        ir, rt = _tcgen05_ldred_ir(mods, count,
                                   _TCGEN05_LDRED_REDT[dtype], nothing)
        @eval @inline (::Operation{:tcgen05, $mods})(taddr::UInt32) =
            Base.llvmcall(($ir, "entry"), $rt, Tuple{UInt32}, taddr)
        _tcgen05_adapter!(mods, Address{UInt32})
    end
    nothing
end

for shape in (Symbol("32x32b"), Symbol("16x32bx2")),
        count in (2, 4, 8, 16, 32, 64, 128), redop in (:min, :max),
        (variant, dtype) in (((), :f32), ((:abs,), :f32),
                             ((Symbol("NaN"),), :f32),
                             ((:abs, Symbol("NaN")), :f32),
                             ((), :u32), ((), :s32))
    _tcgen05_ldred_register(shape, count, redop, variant, dtype)
end

# --- alloc / relinquish / wait / commit ----------------------------------------

# Single-route asm (see header). The generic-address alloc form keeps its
# exact pointer schema — the ISA permits omitting `.shared::cta` while
# still requiring `dst` to lie in the CTA shared-memory window — so a
# wrong address space, pointee, arity, or nCols carrier cannot fall
# through to generic tcgen05 rendering.
for cg in 1:2
    cta = Symbol("cta_group::", cg)
    head = "tcgen05.alloc.cta_group::$cg.sync.aligned"

    mods = (:alloc, cta, :sync, :aligned, :b32)
    ir = convergent_asm_ir("$head.b32 [\$0], \$1;", "r,r,~{memory}",
                           Nothing, (Core.LLVMPtr{UInt32, AS.Shared}, UInt32))
    @eval @inline function (::Operation{:tcgen05, $mods})(
            dst::Core.LLVMPtr{UInt32, AS.Shared}, ncols::UInt32)
        Base.llvmcall(($ir, "entry"), Nothing,
                      Tuple{Core.LLVMPtr{UInt32, AS.Shared}, UInt32},
                      dst, ncols)
    end

    mods = (:alloc, cta, :sync, :aligned, Symbol("shared::cta"), :b32)
    ir = convergent_asm_ir("$head.shared::cta.b32 [\$0], \$1;",
                           "r,r,~{memory}", Nothing, (UInt32, UInt32))
    @eval @inline function (::Operation{:tcgen05, $mods})(
            dst::UInt32, ncols::UInt32)
        Base.llvmcall(($ir, "entry"), Nothing, Tuple{UInt32, UInt32},
                      dst, ncols)
    end

    mods = (:relinquish_alloc_permit, cta, :sync, :aligned)
    ir = convergent_asm_ir(
        "tcgen05.relinquish_alloc_permit.cta_group::$cg.sync.aligned;",
        "~{memory}", Nothing, ())
    @eval @inline (::Operation{:tcgen05, $mods})() =
        Base.llvmcall(($ir, "entry"), Nothing, Tuple{})
end

# Single-route asm: under the MEMORY_WIDEN_OVERLAY the wait intrinsics
# render no memory attribute — semantically the same conservative barrier
# as `sideeffect + ~{memory}` — so an intrinsic route would buy only its
# two selection probes. The waits order ALL prior tcgen05.ld/st results
# against subsequent ordinary loads/stores, hence the full clobber.
let ir = convergent_asm_ir("tcgen05.wait::ld.sync.aligned;", "~{memory}",
                           Nothing, ())
    @eval @inline optype"tcgen05.wait::ld.sync.aligned"() =
        Base.llvmcall(($ir, "entry"), Nothing, Tuple{})
end
let ir = convergent_asm_ir("tcgen05.wait::st.sync.aligned;", "~{memory}",
                           Nothing, ())
    @eval @inline optype"tcgen05.wait::st.sync.aligned"() =
        Base.llvmcall(($ir, "entry"), Nothing, Tuple{})
end

# Specialized thread-synchronization fences are side-effecting, no-argument
# code-motion barriers (PTX 9.3 §9.7.17.11.1). Keep them as inline PTX with
# a memory clobber: LLVM 15 does not recognize the tcgen05 NVVM intrinsics and
# trusts their generated
# `readnone` declaration, which lets it delete the void calls. `sideeffect`
# retains each fence and `~{memory}` prevents tcgen05 and execution-ordering
# operations from moving across it. Exact methods still prevent an accidental
# operand from becoming a literal extra PTX operand through the generic chain.
@inline optype"tcgen05.fence::before_thread_sync"() =
    @asmcall("tcgen05.fence::before_thread_sync;", "~{memory}", true,
             Nothing, Tuple{})
@inline optype"tcgen05.fence::after_thread_sync"() =
    @asmcall("tcgen05.fence::after_thread_sync;", "~{memory}", true,
             Nothing, Tuple{})

# Single-route asm. The notation keeps both state-space spellings, but
# every form RENDERS `.shared::cluster`: the §9.7.18.12.1 syntax block
# admits only {.shared::cluster} on commit (no ::cta spelling exists;
# ptxas rejects one with "State space incorrect"), so the ::cta-modified
# wrapper is an address-species surface and the render follows the ISA —
# the same spelling the intrinsic route printed. The pre-9.4 multicast
# form keeps pyptx's multicast-first modifier order with its b16 CTA mask
# under "h".
for cg in 1:2
    cta = Symbol("cta_group::", cg)
    arrive = Symbol("mbarrier::arrive::one")
    chead = "tcgen05.commit.cta_group::$cg.mbarrier::arrive::one"

    # Pointer form (shared::cluster notation, AS 3 mbar).
    mods = (:commit, cta, arrive, Symbol("shared::cluster"), :b64)
    ir = convergent_asm_ir("$chead.shared::cluster.b64 [\$0];",
                           "r,~{memory}", Nothing,
                           (Core.LLVMPtr{UInt64, AS.Shared},))
    @eval @inline function (::Operation{:tcgen05, $mods})(
            mbar::Core.LLVMPtr{UInt64, AS.Shared})
        Base.llvmcall(($ir, "entry"), Nothing,
                      Tuple{Core.LLVMPtr{UInt64, AS.Shared}}, mbar)
    end

    # SMEM-offset forms for both state-space notations; one shared
    # `.shared::cluster` render per the section comment above.
    for space in (Symbol("shared::cta"), Symbol("shared::cluster"))
        mods = (:commit, cta, arrive, space, :b64)
        ir = convergent_asm_ir("$chead.shared::cluster.b64 [\$0];",
                               "r,~{memory}", Nothing, (UInt32,))
        @eval @inline (::Operation{:tcgen05, $mods})(mbar::UInt32) =
            Base.llvmcall(($ir, "entry"), Nothing, Tuple{UInt32}, mbar)
    end

    mods = (:commit, cta, arrive, Symbol("multicast::cluster"),
            Symbol("shared::cluster"), :b64)
    ir = convergent_asm_ir(
        "$chead.multicast::cluster.shared::cluster.b64 [\$0], \$1;",
        "r,h,~{memory}", Nothing, (UInt32, UInt16))
    @eval @inline function (::Operation{:tcgen05, $mods})(
            mbar::UInt32, mask::Integer)
        Base.llvmcall(($ir, "entry"), Nothing, Tuple{UInt32, UInt16},
                      mbar, UInt16(mask))
    end
end

# --- PTX ISA 9.4 alloc/dealloc/commit forms, asm tier -------------------------
# No NVVM intrinsics exist for any of these at 22.1.7; spelled-only until a
# CUDA 13.4+ ptxas ships. All are convergent per their form contracts, so
# they go through convergent_asm_ir like the other tcgen05 asm forms; the
# conservative ~{memory} clobber matches the family's asm tier today.
#
# `.exclusive` (sm_100f/sm_110f families; nCols up to 576 on sm_107f) claims
# sole ownership of the allocation permit: no other allocation may coexist,
# nCols may be a non-power-of-two multiple of 32, and the deallocation must
# be the matching `.exclusive` form.
#
# The 9.4 commit qualifiers follow the syntax-block canonical order — state
# space, then multicast — matching the mbarrier 9.4 forms. The 16b spelling
# is the explicit form of the pre-9.4 default (b16 mask); ::32b widens the
# CTA mask to b32 for >16-CTA clusters. `.sync_restrict::shared::read::mma::a`
# arrives on the mbarrier once all prior tcgen05.mma reads of the A matrix
# from shared memory complete — the early-SMEM-release primitive; it does
# NOT signal MMA completion.

for cg in 1:2
    cta = Symbol("cta_group::", cg)
    head = "tcgen05.alloc.exclusive.cta_group::$cg.sync.aligned"

    mods = (:alloc, :exclusive, cta, :sync, :aligned, :b32)
    ir = convergent_asm_ir("$head.b32 [\$0], \$1;", "r,r,~{memory}",
                           Nothing, (Core.LLVMPtr{UInt32, AS.Shared}, UInt32))
    @eval @inline function (::Operation{:tcgen05, $mods})(
            dst::Core.LLVMPtr{UInt32, AS.Shared}, ncols::UInt32)
        Base.llvmcall(($ir, "entry"), Nothing,
                      Tuple{Core.LLVMPtr{UInt32, AS.Shared}, UInt32},
                      dst, ncols)
    end

    mods = (:alloc, :exclusive, cta, :sync, :aligned,
            Symbol("shared::cta"), :b32)
    ir = convergent_asm_ir("$head.shared::cta.b32 [\$0], \$1;",
                           "r,r,~{memory}", Nothing, (UInt32, UInt32))
    @eval @inline function (::Operation{:tcgen05, $mods})(
            dst::UInt32, ncols::UInt32)
        Base.llvmcall(($ir, "entry"), Nothing, Tuple{UInt32, UInt32},
                      dst, ncols)
    end

    mods = (:dealloc, :exclusive, cta, :sync, :aligned, :b32)
    ir = convergent_asm_ir(
        "tcgen05.dealloc.exclusive.cta_group::$cg.sync.aligned.b32 \$0, \$1;",
        "r,r,~{memory}", Nothing, (UInt32, UInt32))
    @eval @inline function (::Operation{:tcgen05, $mods})(
            taddr::UInt32, ncols::UInt32)
        Base.llvmcall(($ir, "entry"), Nothing, Tuple{UInt32, UInt32},
                      taddr, ncols)
    end
end

for cg in 1:2
    cta = Symbol("cta_group::", cg)
    arrive = Symbol("mbarrier::arrive::one")
    cluster = Symbol("shared::cluster")
    syncres = Symbol("sync_restrict::shared::read::mma::a")
    chead = "tcgen05.commit.cta_group::$cg.mbarrier::arrive::one"

    for (mc, maskT, letter) in
            ((Symbol("multicast::cluster::16b"), UInt16, "h"),
             (Symbol("multicast::cluster::32b"), UInt32, "r"))
        mods = (:commit, cta, arrive, cluster, mc, :b64)
        ir = convergent_asm_ir("$chead.shared::cluster.$mc.b64 [\$0], \$1;",
                               "r,$letter,~{memory}", Nothing,
                               (UInt32, maskT))
        @eval @inline function (::Operation{:tcgen05, $mods})(
                mbar::UInt32, mask::Integer)
            Base.llvmcall(($ir, "entry"), Nothing, Tuple{UInt32, $maskT},
                          mbar, $maskT(mask))
        end
    end

    mods = (:commit, cta, arrive, syncres, cluster, :b64)
    ir = convergent_asm_ir(
        "$chead.sync_restrict::shared::read::mma::a.shared::cluster.b64 [\$0];",
        "r,~{memory}", Nothing, (UInt32,))
    @eval @inline function (::Operation{:tcgen05, $mods})(mbar::UInt32)
        Base.llvmcall(($ir, "entry"), Nothing, Tuple{UInt32}, mbar)
    end

    mods = (:commit, cta, arrive, syncres, cluster,
            Symbol("multicast::cluster::32b"), :b64)
    ir = convergent_asm_ir(
        "$chead.sync_restrict::shared::read::mma::a.shared::cluster" *
        ".multicast::cluster::32b.b64 [\$0], \$1;",
        "r,r,~{memory}", Nothing, (UInt32, UInt32))
    @eval @inline function (::Operation{:tcgen05, $mods})(
            mbar::UInt32, mask::Integer)
        Base.llvmcall(($ir, "entry"), Nothing, Tuple{UInt32, UInt32},
                      mbar, UInt32(mask))
    end
end

# Integer-address adapters for the literal alloc/commit methods above.
# The generic-address alloc and pointer commit forms need no entry because
# `address(::Core.LLVMPtr)` is identity.
for cg in 1:2
    cta = Symbol("cta_group::", cg)
    _tcgen05_adapter!((:alloc, cta, :sync, :aligned,
                       Symbol("shared::cta"), :b32),
                      Address{UInt32}, UInt32)
    for space in (Symbol("shared::cta"), Symbol("shared::cluster"))
        _tcgen05_adapter!((:commit, cta, Symbol("mbarrier::arrive::one"),
                           space, :b64), Address{UInt32})
    end
    _tcgen05_adapter!((:commit, cta, Symbol("mbarrier::arrive::one"),
                       Symbol("multicast::cluster"),
                       Symbol("shared::cluster"), :b64),
                      Address{UInt32}, Integer)
end

# --- dense mma (A from SMEM descriptor) ----------------------------------------
# kind immarg: 0=f16, 1=tf32, 2=f8f6f4, 3=i8; collector_usage 0 = discard
# (the ISA default, now spelled explicitly in the output).
@inline optype"tcgen05.mma.cta_group::1.kind::f16"(
        d::UInt32, a_desc::UInt64, b_desc::UInt64,
        idesc::UInt32, enable_input_d::Bool) =
    ceiled(nvvm"tcgen05.mma.shared", ptx"tcgen05.mma.cta_group::1.kind::f16")(
        _tmem(d), a_desc, b_desc, idesc, enable_input_d, Val(0), Val(1), Val(0))
@inline optype"tcgen05.mma.cta_group::2.kind::f16"(
        d::UInt32, a_desc::UInt64, b_desc::UInt64,
        idesc::UInt32, enable_input_d::Bool) =
    ceiled(nvvm"tcgen05.mma.shared", ptx"tcgen05.mma.cta_group::2.kind::f16")(
        _tmem(d), a_desc, b_desc, idesc, enable_input_d, Val(0), Val(2), Val(0))
@inline optype"tcgen05.mma.cta_group::1.kind::tf32"(
        d::UInt32, a_desc::UInt64, b_desc::UInt64,
        idesc::UInt32, enable_input_d::Bool) =
    ceiled(nvvm"tcgen05.mma.shared", ptx"tcgen05.mma.cta_group::1.kind::tf32")(
        _tmem(d), a_desc, b_desc, idesc, enable_input_d, Val(1), Val(1), Val(0))
@inline optype"tcgen05.mma.cta_group::2.kind::tf32"(
        d::UInt32, a_desc::UInt64, b_desc::UInt64,
        idesc::UInt32, enable_input_d::Bool) =
    ceiled(nvvm"tcgen05.mma.shared", ptx"tcgen05.mma.cta_group::2.kind::tf32")(
        _tmem(d), a_desc, b_desc, idesc, enable_input_d, Val(1), Val(2), Val(0))
@inline optype"tcgen05.mma.cta_group::1.kind::f8f6f4"(
        d::UInt32, a_desc::UInt64, b_desc::UInt64,
        idesc::UInt32, enable_input_d::Bool) =
    ceiled(nvvm"tcgen05.mma.shared",
           ptx"tcgen05.mma.cta_group::1.kind::f8f6f4")(
        _tmem(d), a_desc, b_desc, idesc, enable_input_d, Val(2), Val(1), Val(0))
@inline optype"tcgen05.mma.cta_group::2.kind::f8f6f4"(
        d::UInt32, a_desc::UInt64, b_desc::UInt64,
        idesc::UInt32, enable_input_d::Bool) =
    ceiled(nvvm"tcgen05.mma.shared",
           ptx"tcgen05.mma.cta_group::2.kind::f8f6f4")(
        _tmem(d), a_desc, b_desc, idesc, enable_input_d, Val(2), Val(2), Val(0))
@inline optype"tcgen05.mma.cta_group::1.kind::i8"(
        d::UInt32, a_desc::UInt64, b_desc::UInt64,
        idesc::UInt32, enable_input_d::Bool) =
    ceiled(nvvm"tcgen05.mma.shared", ptx"tcgen05.mma.cta_group::1.kind::i8")(
        _tmem(d), a_desc, b_desc, idesc, enable_input_d, Val(3), Val(1), Val(0))
@inline optype"tcgen05.mma.cta_group::2.kind::i8"(
        d::UInt32, a_desc::UInt64, b_desc::UInt64,
        idesc::UInt32, enable_input_d::Bool) =
    ceiled(nvvm"tcgen05.mma.shared", ptx"tcgen05.mma.cta_group::2.kind::i8")(
        _tmem(d), a_desc, b_desc, idesc, enable_input_d, Val(3), Val(2), Val(0))

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
# multicast. Methods are generated from one spec; the wrapper registry
# (family :tcgen05_mma_dense) records every tier-2 name for the
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
    reg(name) = wrapper_intrinsic_call(:tcgen05_mma_dense, :tcgen05,
                                       mods, name)

    # Integer-address adapters for every cell of this grid, including the
    # SMEM-A discard base cell whose primary method is one of the literal
    # methods above (the adapter forwards through ordinary dispatch).
    A32 = Address{UInt32}
    aA = tmem_a ? A32 : UInt64
    maskT = NTuple{maskN, UInt32}
    _tcgen05_adapter!(mods, A32, aA, UInt64, UInt32, Bool)
    _tcgen05_adapter!(mods, A32, aA, UInt64, UInt32, maskT, Bool)
    if scale_ok
        _tcgen05_adapter!(mods, A32, aA, UInt64, UInt32, Bool, Val)
        _tcgen05_adapter!(mods, A32, aA, UInt64, UInt32, maskT, Bool, Val)
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
    reg(name) = wrapper_intrinsic_call(:tcgen05_mma_sp, :tcgen05,
                                       mods, name)

    # Integer-address adapters, mirroring dense with the sp-meta TMEM
    # address between the B descriptor and idesc.
    A32 = Address{UInt32}
    aA = tmem_a ? A32 : UInt64
    maskT = NTuple{maskN, UInt32}
    _tcgen05_adapter!(mods, A32, aA, UInt64, A32, UInt32, Bool)
    _tcgen05_adapter!(mods, A32, aA, UInt64, A32, UInt32, maskT, Bool)
    if scale_ok
        _tcgen05_adapter!(mods, A32, aA, UInt64, A32, UInt32, Bool, Val)
        _tcgen05_adapter!(mods, A32, aA, UInt64, A32, UInt32, maskT, Bool,
                          Val)
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
    reg(name) = wrapper_intrinsic_call(:tcgen05_mma_ws, :tcgen05,
                                       mods, name)

    # Integer-address adapters: base form and trailing zero-column-mask.
    A32 = Address{UInt32}
    aA = tmem_a ? A32 : UInt64
    meta_A = sp ? (A32,) : ()
    _tcgen05_adapter!(mods, A32, aA, UInt64, meta_A..., UInt32, Bool)
    _tcgen05_adapter!(mods, A32, aA, UInt64, meta_A..., UInt32, Bool,
                      UInt64)

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
#   [d], a-desc/[a], b-desc{, [sp-meta]}, idesc, [scale-A], [scale-B],
#   enable-input-d
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

function _tcgen05_mx_register(kind::Symbol, scale::Symbol, cta_group::Int,
                              sp::Bool, coll::Union{Nothing, Symbol})
    cta_group in (1, 2) || throw(ArgumentError("invalid tcgen05 cta_group: $cta_group"))
    cg = Symbol("cta_group::", cta_group)
    kmod = Symbol("kind::", kind)
    spmods = sp ? (:sp,) : ()
    collmods = coll === nothing ? () : (coll,)
    mods = (:mma, spmods..., cg, kmod, :block_scale, scale, collmods...)
    register_wrapper!(:tcgen05_mx, :tcgen05, mods, :asm)
    head = "tcgen05.mma" * (sp ? ".sp" : "") *
           ".$cg.$kmod.block_scale.$scale" *
           (coll === nothing ? "" : ".$coll")

    # Integer-address adapters: the scale descriptors — and for `.sp` the
    # sparsity-metadata operand — are bracketed TMEM addresses, and the A
    # operand may itself be a TMEM address.
    A32 = Address{UInt32}
    meta = sp ? (A32,) : ()
    _tcgen05_adapter!(mods, A32, UInt64, UInt64, meta..., UInt32, A32, A32,
                      Bool)
    _tcgen05_adapter!(mods, A32, A32, UInt64, meta..., UInt32, A32, A32,
                      Bool)

    # A-operand species: SMEM descriptor (UInt64) versus TMEM address
    # (UInt32) — unambiguous at Julia dispatch. Operand order per
    # §9.7.18.10: [d], a, b-desc{, [sp-meta]}, idesc, [scale-A],
    # [scale-B], enable-input-d.
    for a_tmem in (false, true)
        aT = a_tmem ? UInt32 : UInt64
        slots = String["[\$0]", a_tmem ? "[\$1]" : "\$1", "\$2"]
        k = 3
        if sp
            push!(slots, "[\$3]")
            k = 4
        end
        push!(slots, "\$$k", "[\$$(k + 1)]", "[\$$(k + 2)]", "\$$(k + 3)")
        asm = head * " " * join(slots, ", ") * ";"
        constraints = "r," * (a_tmem ? "r" : "l") * ",l," *
                      (sp ? "r," : "") * "r,r,r,b,~{memory}"
        metadecl = sp ? (:(spmeta::UInt32),) : ()
        metaargs = sp ? (:spmeta,) : ()
        tt = Tuple{UInt32, aT, UInt64, (sp ? (UInt32,) : ())...,
                   UInt32, UInt32, UInt32, Bool}
        @eval @inline function (::Operation{:tcgen05, $mods})(
                d::UInt32, a::$aT, b_desc::UInt64, $(metadecl...),
                idesc::UInt32, scale_a::UInt32, scale_b::UInt32,
                enable_input_d::Bool)
            @asmcall($asm, $constraints, true, Nothing, $tt,
                     d, a, b_desc, $(metaargs...), idesc,
                     scale_a, scale_b, enable_input_d)
            nothing
        end
    end
    nothing
end

# `.sp` target gating (§9.7.18.10 support list, ptxas-verified): sparse
# block-scale with .kind::mxf4/.kind::mxf4nvf4 is a-variant-exclusive —
# family targets refuse the modifier pair ("Feature '.kind::mxf4 with .sp
# modifier' not supported on .target 'sm_100f'") — while .kind::mxf8f6f4
# assembles on both a- and f-targets. Registered uniformly; the assembler
# owns target policy, as with dense .kind::i8.
#
# collector::a mirrors the dense/sp intrinsic-tier convention: an absent
# collector is the ISA-default discard (never spelled explicitly), and
# collector::b / decompress::lut::b stay deferred.
for (kind, scale_vec, block) in _TCGEN05_MX_SCALE_VARIANTS,
        scale in (scale_vec, block), cta_group in (1, 2),
        sp in (false, true),
        coll in (nothing, Symbol("collector::a::lastuse"),
                 Symbol("collector::a::fill"), Symbol("collector::a::use"))
    _tcgen05_mx_register(kind, scale, cta_group, sp, coll)
end

# Seal the derived adapter inventory. Every signature above was emitted by
# the same enumeration that registered (or accompanies) its primary method,
# so this closed set cannot drift from the typed surface; the independent
# oracle in test/host/address_roles.jl pins its 434 forms / 1218 signatures.
const TCGEN05_INTEGER_ADDRESS_ADAPTERS = let specs = _TCGEN05_ADAPTER_SPECS
    keys = ((s.mods, s.argtypes) for s in specs)
    allunique(keys) ||
        error("duplicate tcgen05 integer-address adapter signature")
    Tuple(specs)
end

const TCGEN05_INTEGER_ADDRESS_FORMS =
    Tuple(unique(s.mods for s in TCGEN05_INTEGER_ADDRESS_ADAPTERS))
