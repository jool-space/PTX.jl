# Warp-level register shuffle (PTX 9.2 §9.7.4) — the first family migrated
# to tier-2 intrinsic lowering (DESIGN.md, "Lowering tiers"): the backend
# selects the instruction from `llvm.nvvm.shfl.sync.<mode>.i32[p]`, with the
# registry supplying `convergent` (shfl is the convergence guinea pig — see
# spikes/convergence.jl for what its absence does).
#
# The notation surface is unchanged: `ptx"shfl.sync.<mode>.b32"(a, b, c,
# membermask)` in PTX operand order, with the trailing `.pred` chain
# modifier for the (data, in-range-pred) form. The intrinsic takes the
# membermask first; the reorder happens here. The pred form maps to the
# `.i32p` intrinsic's {i32, i1} aggregate return — no more pipe-operand asm.

const SHFL_MODES = (:up, :down, :bfly, :idx)

for mode in SHFL_MODES
    data = "llvm.nvvm.shfl.sync.$mode.i32"
    pred = data * "p"
    NVVM.isintrinsic(data) && NVVM.isintrinsic(pred) ||
        error("shfl: $data[p] missing from the backend intrinsic table")
    dataop = NVVM.IntrinsicCall{Symbol(data)}()
    predop = NVVM.IntrinsicCall{Symbol(pred)}()
    mods = (:sync, mode, :b32)
    mods_pred = (:sync, mode, :b32, :pred)
    @eval begin
        @inline (::Operation{:shfl, $mods})(a::UInt32, b::UInt32, c::UInt32,
                                            membermask::UInt32) =
            $dataop(membermask, a, b, c)
        @inline (::Operation{:shfl, $mods_pred})(a::UInt32, b::UInt32, c::UInt32,
                                                 membermask::UInt32) =
            $predop(membermask, a, b, c)
    end
end
