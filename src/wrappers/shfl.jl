# Warp-level register shuffle (PTX 9.2 §9.7.4) — the first family migrated
# to tier-2 intrinsic lowering: the backend
# selects the instruction from `llvm.nvvm.shfl.sync.<mode>.i32[p]`, with the
# registry supplying `convergent` (shfl is the convergence guinea pig —
# spikes/convergence.jl shows what its absence does).
#
# The notation surface is unchanged: `ptx"shfl.sync.<mode>.b32"(a, b, c,
# membermask)` in PTX operand order, with the trailing `.pred` chain
# modifier for the (data, in-range-pred) form. The intrinsic takes the
# membermask first; the reorder happens here. The pred form maps to the
# `.i32p` intrinsic's {i32, i1} aggregate return — no more pipe-operand asm.
#
# Methods are written out literally (no name-building loop) so every
# intrinsic this file stands on is greppable — test/host/conformance.jl
# scans for `nvvm"..."` literals and requires a probe for each.

const SHFL_MODES = (:up, :down, :bfly, :idx)

@inline optype"shfl.sync.idx.b32"(a::UInt32, b::UInt32,
        c::UInt32, membermask::UInt32) =
    ceiled(nvvm"shfl.sync.idx.i32", ptx"shfl.sync.idx.b32")(membermask, a, b, c)
@inline optype"shfl.sync.idx.b32.pred"(a::UInt32, b::UInt32,
        c::UInt32, membermask::UInt32) =
    ceiled(nvvm"shfl.sync.idx.i32p",
           ptx"shfl.sync.idx.b32.pred")(membermask, a, b, c)

@inline optype"shfl.sync.up.b32"(a::UInt32, b::UInt32,
        c::UInt32, membermask::UInt32) =
    ceiled(nvvm"shfl.sync.up.i32", ptx"shfl.sync.up.b32")(membermask, a, b, c)
@inline optype"shfl.sync.up.b32.pred"(a::UInt32, b::UInt32,
        c::UInt32, membermask::UInt32) =
    ceiled(nvvm"shfl.sync.up.i32p",
           ptx"shfl.sync.up.b32.pred")(membermask, a, b, c)

@inline optype"shfl.sync.down.b32"(a::UInt32, b::UInt32,
        c::UInt32, membermask::UInt32) =
    ceiled(nvvm"shfl.sync.down.i32", ptx"shfl.sync.down.b32")(membermask, a, b, c)
@inline optype"shfl.sync.down.b32.pred"(a::UInt32, b::UInt32,
        c::UInt32, membermask::UInt32) =
    ceiled(nvvm"shfl.sync.down.i32p",
           ptx"shfl.sync.down.b32.pred")(membermask, a, b, c)

@inline optype"shfl.sync.bfly.b32"(a::UInt32, b::UInt32,
        c::UInt32, membermask::UInt32) =
    ceiled(nvvm"shfl.sync.bfly.i32", ptx"shfl.sync.bfly.b32")(membermask, a, b, c)
@inline optype"shfl.sync.bfly.b32.pred"(a::UInt32, b::UInt32,
        c::UInt32, membermask::UInt32) =
    ceiled(nvvm"shfl.sync.bfly.i32p",
           ptx"shfl.sync.bfly.b32.pred")(membermask, a, b, c)
