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


# --- mbarrier: ::cta forms at their cap floors -------------------------------
# Two goldens: the sm_80 subset pins the cap floor (no sm_90-only forms),
# sm_90 adds expect_tx and try_wait. Cluster-space (`shared::cluster`) sink
# forms are not in the goldens: they stay asm-tier until cluster addresses
# are modeled as addrspace(7) (see wrappers/mbarrier.jl).

function _golden_mbarrier_sm80!(out::CuDeviceVector{UInt64, 1})
    bar = CuStaticSharedArray(Int64, 1)
    mbar = pointer(bar)
    ptx"mbarrier.init.shared.b64"(mbar, UInt32(32))
    s1 = ptx"mbarrier.arrive.shared.b64"(mbar)
    s2 = ptx"mbarrier.arrive.noComplete.shared.b64"(mbar, UInt32(2))
    w1 = ptx"mbarrier.test_wait.shared.b64"(mbar, s1)
    w2 = ptx"mbarrier.test_wait.parity.shared.b64"(mbar, UInt32(0))
    ptx"mbarrier.inval.shared.b64"(mbar)
    @inbounds out[1] = s1 + s2 + UInt64(w1) + UInt64(w2)
    return nothing
end

@testset "golden: mbarrier sm_80 subset" begin
    @test golden_test("mbarrier@sm80", _golden_mbarrier_sm80!,
                      Tuple{CuDeviceVector{UInt64, 1}}; cap = v"8.0")
end

function _golden_mbarrier_sm90!(out::CuDeviceVector{UInt64, 1})
    bar = CuStaticSharedArray(Int64, 1)
    mbar = pointer(bar)
    ptx"mbarrier.init.shared.b64"(mbar, UInt32(32))
    ptx"mbarrier.expect_tx.shared.b64"(mbar, UInt32(256))
    s1 = ptx"mbarrier.arrive.expect_tx.shared.b64"(mbar, UInt32(256))
    w1 = ptx"mbarrier.try_wait.shared.b64"(mbar, s1)
    w2 = ptx"mbarrier.try_wait.parity.shared.b64"(mbar, UInt32(0))
    @inbounds out[1] = s1 + UInt64(w1) + UInt64(w2)
    return nothing
end

@testset "golden: mbarrier sm_90 forms" begin
    @test golden_test("mbarrier@sm90", _golden_mbarrier_sm90!,
                      Tuple{CuDeviceVector{UInt64, 1}}; cap = v"9.0")
end
