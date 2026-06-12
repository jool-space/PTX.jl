# Golden-PTX locks, per family (DESIGN.md, "Approach"): each kernel pins a
# family's emitted instruction sequence (structural, modulo naming — see
# IR.canonicalize) so a lowering migration shows up as a reviewable git diff
# of the golden file instead of an act of faith. Regenerate deliberately:
#
#     PTX_UPDATE_GOLDEN=1 julia --project=test test/runtests.jl ptxas/golden
#
# One kernel per family, straight-line, touching every form the wrappers
# expose, at the lowest cap that supports the family.


# --- shfl: all four modes, data and data|pred forms -------------------------

function _golden_shfl!(out::CuDeviceVector{UInt32, 1})
    tid = ptx"mov.u32"(sreg"tid.x")
    mask = UInt32(0xffffffff)
    @inbounds begin
        out[tid + 1] = ptx"shfl.sync.idx.b32"(tid, tid, UInt32(0x1f), mask)
        out[tid + 33] = ptx"shfl.sync.up.b32"(tid, UInt32(1), UInt32(0), mask)
        out[tid + 65] = ptx"shfl.sync.down.b32"(tid, UInt32(1), UInt32(0x1f), mask)
        out[tid + 97] = ptx"shfl.sync.bfly.b32"(tid, UInt32(1), UInt32(0x1f), mask)
        v, p = ptx"shfl.sync.idx.b32.pred"(tid, tid, UInt32(0x1f), mask)
        out[tid + 129] = v + UInt32(p)
        v, p = ptx"shfl.sync.up.b32.pred"(tid, UInt32(1), UInt32(0), mask)
        out[tid + 161] = v + UInt32(p)
        v, p = ptx"shfl.sync.down.b32.pred"(tid, UInt32(1), UInt32(0x1f), mask)
        out[tid + 193] = v + UInt32(p)
        v, p = ptx"shfl.sync.bfly.b32.pred"(tid, UInt32(1), UInt32(0x1f), mask)
        out[tid + 225] = v + UInt32(p)
    end
    return nothing
end

@testset "golden: shfl family at sm_75" begin
    @test golden_test("shfl@sm75", _golden_shfl!,
                      Tuple{CuDeviceVector{UInt32, 1}}; cap = v"7.5")
end
