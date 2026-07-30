# `cp.async.bulk.tensor.<N>d.*` — TMA tile copies (Hopper sm_90+) and the
# tensor prefetch grammars, single-route convergent inline asm.
# The notation surface is unchanged: exact typed wrappers own the
# `[base, {coords}]` operand encoding, qualifier rendering, and per-arity
# constraint strings.
#
# The family previously routed through the llvm.nvvm.cp.async.bulk.tensor.*
# intrinsics (tier 2). That split was retired deliberately:
#   - every form is an observable async memory effect — there is no
#     CSE/LICM for intrinsic attributes to unlock, so the tier-2 route
#     bought bookkeeping (per-intrinsic selection probes, immarg
#     flag-pair plumbing for optional qualifiers, addrspace(7)/generic
#     retypes) and no optimization. The argmem-widen A/B on B200 (branch
#     agent/argmem-widen-b200) proved the point from the other direction:
#     widening the `llvm.nvvm.cp.async.bulk` prefix's memory precision —
#     fn-level props AND per-arg readonly/writeonly — to the asm route's
#     conservative clobber left the FA/GEMM/b128 instruction streams
#     byte-identical at sm_100a;
#   - the `shared::cta` × `cta_group::2` residue was already asm (no NVVM
#     intrinsic carries both qualifiers at 22.1.7) — one route instead of
#     two.
# Emitted-PTX deltas vs the intrinsic route, reviewed at demotion:
#   - the notation is WYSIWYG again in the one spot it wasn't: the
#     cluster-destination `cta_group::2` forms render the qualifier after
#     `.<N>d` (pyptx order, matching the pre-existing asm residue), where
#     ISel rendered the §9.7.10.28.5.3 syntax-block order
#     `.mbarrier::complete_tx::bytes{.multicast::cluster}.cta_group::2`.
#     ptxas accepts both spellings (probed standalone at sm_100a, ptxas
#     13.3; pinned by the ptxas legs in test/ptxas/blackwell.jl);
#   - static-SMEM operands materialize through mov/cvt into a register
#     instead of folding into the operand as a symbol (ptxas folds these
#     in SASS), exactly as the residue always rendered.
#
# Operand/constraint schema (PTX 9.4 §9.7.10.28.5.3 and §9.7.10.28.5.5):
#   - shared-window addresses (dst, mbar, src) are `r` (32-bit window
#     offsets; same encoding the residue and pyptx always used);
#   - the tensor-map operand is a 64-bit generic address, `l`. The
#     package's TMADescriptorPtr is a *global* address typed AS.Const by
#     convention, so its raw value is already generic — the asm boundary
#     passes it as-is, never through cvta (see reinterpret_addrspace for
#     why a translation would corrupt it);
#   - tensor coordinates are `.s32` (`r`), the multicast CTA mask is the
#     default-width 16-bit `h` carrier (the ::16b/::32b sub-qualifiers are
#     not in this surface), im2col offsets are 16-bit `h`, and the
#     `.L2::cache_hint` policy is a paired 64-bit `l` operand.
#
# Every form carries the `convergent nomerge` + `~{memory}` call-site
# contract via convergent_asm_ir — the same conservative boundary the
# NVVM intrinsic records imposed (they are all marked convergent even
# though PTX imposes no collective participation rule; see the prefetch
# note below).

# Pointee-erasing retype at the asm boundary: the wrappers stay generic
# over the element type, but `Base.llvmcall` requires the exact declared
# argument types, and `_asm_lltype` spells every LLVMPtr as an i8 pointer
# anyway. Same raw ptrtoint/inttoptr bit-preservation contract as
# reinterpret_addrspace — the address space is untouched.
@generated function _tma_addr(p::Core.LLVMPtr{T, A}) where {T, A}
    spell = A == 0 ? "i8*" : "i8 addrspace($A)*"
    ir = """
        %i = ptrtoint $spell %0 to i64
        %q = inttoptr i64 %i to $spell
        ret $spell %q"""
    quote
        Base.@inline
        Base.llvmcall($ir, Core.LLVMPtr{UInt8, $A},
                      Tuple{Core.LLVMPtr{$T, $A}}, p)
    end
end

# --- Loads: shared::cluster destination --------------------------------------
# Plain, multicast::cluster (one global read lands in every CTA whose bit
# is set in the b16 mask), cta_group::2 (Blackwell 2-SM, sm_100a; both
# cluster CTAs issue the same instruction and hardware splits the read),
# and their combination. The ISA defines `shared::cta` addresses as valid
# `shared::cluster` addresses, which is exactly what the `r` constraint
# relies on — the same convention as the asm residue below.
for n in 1:5, cg2 in (false, true), mc in (false, true)
    nd = Symbol("$(n)d")
    cs = [Symbol("c", i) for i in 1:n]
    mods = (:async, :bulk, :tensor, nd,
            (cg2 ? (Symbol("cta_group::2"),) : ())...,
            Symbol("shared::cluster"), :global, :tile,
            Symbol("mbarrier::complete_tx::bytes"),
            (mc ? (Symbol("multicast::cluster"),) : ())...)
    spell = "cp." * join(String.(mods), ".")
    coordops = join(("\$$(i + 1)" for i in 1:n), ", ")
    asm = "$spell [\$0], [\$1, {$coordops}], [\$$(n + 2)]" *
          (mc ? ", \$$(n + 3);" : ";")
    constraints = join(["r"; "l"; fill("r", n); "r";
                        (mc ? ["h"] : []); "~{memory}"], ",")
    argts = (Core.LLVMPtr{UInt8, AS.Shared}, Core.LLVMPtr{UInt8, AS.Const},
             ntuple(_ -> Int32, n)..., Core.LLVMPtr{UInt8, AS.Shared},
             (mc ? (UInt16,) : ())...)
    ir = convergent_asm_ir(asm, constraints, Nothing, argts)
    coordsig = [:($c::Integer) for c in cs]
    coordvals = [:(Int32($c)) for c in cs]
    masksig = mc ? Any[:(mask::Integer)] : Any[]
    maskvals = mc ? Any[:(UInt16(mask))] : Any[]
    @eval @inline function (::Operation{:cp, $mods})(
            dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
            $(coordsig...), mbar::Core.LLVMPtr{U, AS.Shared},
            $(masksig...)) where {T, S, U}
        Base.llvmcall(($ir, "entry"), Nothing, Tuple{$(argts...)},
                      _tma_addr(dst), _tma_addr(tmap), $(coordvals...),
                      _tma_addr(mbar), $(maskvals...))
    end
end

# --- Loads: shared::cta destination (PTX 8.6) ---------------------------------

for n in 1:5
    nd = Symbol("$(n)d")
    cs = [Symbol("c", i) for i in 1:n]
    mods = (:async, :bulk, :tensor, nd, Symbol("shared::cta"), :global,
            :tile, Symbol("mbarrier::complete_tx::bytes"))
    spell = "cp." * join(String.(mods), ".")
    coordops = join(("\$$(i + 1)" for i in 1:n), ", ")
    asm = "$spell [\$0], [\$1, {$coordops}], [\$$(n + 2)];"
    constraints = join(["r"; "l"; fill("r", n); "r"; "~{memory}"], ",")
    argts = (Core.LLVMPtr{UInt8, AS.Shared}, Core.LLVMPtr{UInt8, AS.Const},
             ntuple(_ -> Int32, n)..., Core.LLVMPtr{UInt8, AS.Shared})
    ir = convergent_asm_ir(asm, constraints, Nothing, argts)
    coordsig = [:($c::Integer) for c in cs]
    coordvals = [:(Int32($c)) for c in cs]
    @eval @inline function (::Operation{:cp, $mods})(
            dst::Core.LLVMPtr{T, AS.Shared}, tmap::Core.LLVMPtr{S, AS.Const},
            $(coordsig...), mbar::Core.LLVMPtr{U, AS.Shared}) where {T, S, U}
        Base.llvmcall(($ir, "entry"), Nothing, Tuple{$(argts...)},
                      _tma_addr(dst), _tma_addr(tmap), $(coordvals...),
                      _tma_addr(mbar))
    end
end

# --- Stores: shared::cta → global, bulk_group completion ----------------------
# `[tensorMap, {coords}], [srcMem]` — the tensor map leads, per the ISA's
# shared::cta → global syntax block.

for n in 1:5
    nd = Symbol("$(n)d")
    cs = [Symbol("c", i) for i in 1:n]
    mods = (:async, :bulk, :tensor, nd, :global, Symbol("shared::cta"),
            :tile, :bulk_group)
    spell = "cp." * join(String.(mods), ".")
    coordops = join(("\$$i" for i in 1:n), ", ")
    asm = "$spell [\$0, {$coordops}], [\$$(n + 1)];"
    constraints = join(["l"; fill("r", n); "r"; "~{memory}"], ",")
    argts = (Core.LLVMPtr{UInt8, AS.Const}, ntuple(_ -> Int32, n)...,
             Core.LLVMPtr{UInt8, AS.Shared})
    ir = convergent_asm_ir(asm, constraints, Nothing, argts)
    coordsig = [:($c::Integer) for c in cs]
    coordvals = [:(Int32($c)) for c in cs]
    @eval @inline function (::Operation{:cp, $mods})(
            tmap::Core.LLVMPtr{S, AS.Const}, $(coordsig...),
            src::Core.LLVMPtr{T, AS.Shared}) where {S, T}
        Base.llvmcall(($ir, "entry"), Nothing, Tuple{$(argts...)},
                      _tma_addr(tmap), $(coordvals...), _tma_addr(src))
    end
end

# --- Prefetch: global → L2 through a tensor map -------------------------------
# No destination operand and no completion mechanism — fire-and-forget L2
# warming. This is the instruction CUTLASS's weight-prefetch mainloop
# (examples/63) stands on: a dedicated warp walks the weight tensor's
# K-tiles ahead of the TMA loads that will actually consume them.
# PTX imposes no collective participation rule; the NVVM intrinsic records
# are nevertheless `convergent`, and the asm route preserves that
# conservative optimizer boundary without claiming warp-cooperative
# semantics. PTX couples the u64 cache-policy operand to the
# `.L2::cache_hint` qualifier as an inseparable pair; the policy is only a
# performance hint and does not change weak-memory semantics.

for n in 1:5, hint in (false, true)
    nd = Symbol("$(n)d")
    cs = [Symbol("c", i) for i in 1:n]
    mods = (:async, :bulk, :prefetch, :tensor, nd, :L2, :global, :tile,
            (hint ? (Symbol("L2::cache_hint"),) : ())...)
    spell = "cp." * join(String.(mods), ".")
    coordops = join(("\$$i" for i in 1:n), ", ")
    asm = "$spell [\$0, {$coordops}]" * (hint ? ", \$$(n + 1);" : ";")
    constraints = join(["l"; fill("r", n); (hint ? ["l"] : []);
                        "~{memory}"], ",")
    argts = (Core.LLVMPtr{UInt8, AS.Const}, ntuple(_ -> Int32, n)...,
             (hint ? (UInt64,) : ())...)
    ir = convergent_asm_ir(asm, constraints, Nothing, argts)
    coordsig = [:($c::Integer) for c in cs]
    coordvals = [:(Int32($c)) for c in cs]
    hintsig = hint ? Any[:(cache_policy::UInt64)] : Any[]
    hintvals = hint ? Any[:cache_policy] : Any[]
    @eval @inline function (::Operation{:cp, $mods})(
            tmap::Core.LLVMPtr{S, AS.Const}, $(coordsig...),
            $(hintsig...)) where {S}
        Base.llvmcall(($ir, "entry"), Nothing, Tuple{$(argts...)},
                      _tma_addr(tmap), $(coordvals...), $(hintvals...))
    end
end

# Base im2col prefetch is a separate grammar island from tile prefetch. PTX
# 9.3 §9.7.9.26.5.4 admits ranks 3d..5d and requires exactly N signed tensor
# coordinates followed by N-2 signed 16-bit im2col offsets (Figures 11–15),
# the offsets rendered as a braced vector after the bracket operand.
# Keep this surface exact: the later `.im2col::w[::128]` modes and
# `.tile::gather4` have different operands and target restrictions.

for n in 3:5, hint in (false, true)
    k = n - 2
    nd = Symbol("$(n)d")
    cs = [Symbol("c", i) for i in 1:n]
    os = [Symbol("o", i) for i in 1:k]
    mods = (:async, :bulk, :prefetch, :tensor, nd, :L2, :global, :im2col,
            (hint ? (Symbol("L2::cache_hint"),) : ())...)
    spell = "cp." * join(String.(mods), ".")
    coordops = join(("\$$i" for i in 1:n), ", ")
    offsetops = join(("\$$(n + i)" for i in 1:k), ", ")
    asm = "$spell [\$0, {$coordops}], {$offsetops}" *
          (hint ? ", \$$(n + k + 1);" : ";")
    constraints = join(["l"; fill("r", n); fill("h", k);
                        (hint ? ["l"] : []); "~{memory}"], ",")
    argts = (Core.LLVMPtr{UInt8, AS.Const}, ntuple(_ -> Int32, n)...,
             ntuple(_ -> Int16, k)..., (hint ? (UInt64,) : ())...)
    ir = convergent_asm_ir(asm, constraints, Nothing, argts)
    coordsig = [:($c::Int32) for c in cs]
    offsetsig = [:($o::Int16) for o in os]
    hintsig = hint ? Any[:(cache_policy::UInt64)] : Any[]
    hintvals = hint ? Any[:cache_policy] : Any[]
    @eval @inline function (::Operation{:cp, $mods})(
            tmap::Core.LLVMPtr{UInt8, AS.Const}, $(coordsig...),
            $(offsetsig...), $(hintsig...))
        Base.llvmcall(($ir, "entry"), Nothing, Tuple{$(argts...)},
                      tmap, $(cs...), $(os...), $(hintvals...))
    end
end

# --- shared::cta × cta_group::2 ------------------------------------------------
# Predates the demotion as the family's asm residue (no NVVM intrinsic
# carried both qualifiers at 22.1.7: `g2s.cta` has no cta_group operand and
# `g2s` renders `shared::cluster`). Asm strings keep the pyptx modifier
# order (cta_group after `.<N>d`) — now the family-wide spelling.

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
