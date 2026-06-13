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


# --- TMA (cp.async.bulk.tensor): tile loads/stores ---------------------------
# Rank spread (1d/2d/5d) plus the qualifier axes: multicast::cluster, the
# shared::cta destination, and (at sm_100a) cta_group::2. The shared::cta ×
# cta_group::2 combination stays asm-tier (no NVVM intrinsic carries both)
# and is pinned in the sm_100a golden to prove the migration left it alone.

function _golden_tma_sm90!(tmap::Core.LLVMPtr{UInt8, PTX.AS.Const}, c::Int32)
    buf = CuStaticSharedArray(UInt16, 64)
    bar = CuStaticSharedArray(Int64, 1)
    dst = pointer(buf)
    mbar = pointer(bar)
    ptx"cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"(
        dst, tmap, c, mbar)
    ptx"cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"(
        dst, tmap, c, c, mbar)
    ptx"cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"(
        dst, tmap, c, c, c, c, c, mbar)
    ptx"cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"(
        dst, tmap, c, c, mbar, UInt16(0x3))
    ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
        dst, tmap, c, c, mbar)
    ptx"cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group"(tmap, c, dst)
    ptx"cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"(tmap, c, c, dst)
    ptx"cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group"(
        tmap, c, c, c, c, c, dst)
    return nothing
end

@testset "golden: tma tile forms at sm_90" begin
    @test golden_test("tma@sm90", _golden_tma_sm90!,
                      Tuple{Core.LLVMPtr{UInt8, PTX.AS.Const}, Int32};
                      cap = v"9.0")
end

function _golden_tma_sm100a!(tmap::Core.LLVMPtr{UInt8, PTX.AS.Const}, c::Int32)
    buf = CuStaticSharedArray(UInt16, 64)
    bar = CuStaticSharedArray(Int64, 1)
    dst = pointer(buf)
    mbar = pointer(bar)
    ptx"cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"(
        dst, tmap, c, c, mbar)
    ptx"cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"(
        dst, tmap, c, c, mbar, UInt16(0x3))
    ptx"cp.async.bulk.tensor.2d.cta_group::2.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
        dst, tmap, c, c, mbar)
    return nothing
end

@testset "golden: tma cta_group::2 forms at sm_100a" begin
    @test golden_test("tma@sm100a", _golden_tma_sm100a!,
                      Tuple{Core.LLVMPtr{UInt8, PTX.AS.Const}, Int32};
                      cap = v"10.0", feature_set = :arch)
end


# --- tcgen05: lifecycle, data movement, dense mma ----------------------------
# One straight-line kernel touching every wrapped verb: alloc/dealloc/
# relinquish, cp, shift, ld/st (scalar and multi-register shapes), waits,
# commit (cta + cluster spellings, multicast), dense mma (f16 cg1, tf32
# cg2) and one mx kind (block-scale operands are not in the notation
# surface — pinned to prove the migration leaves it on the asm tier).

function _golden_tcgen05_sm100a!(out::CuDeviceVector{UInt32, 1},
                                 s_desc::UInt64, idesc::UInt32)
    slot = CuStaticSharedArray(UInt32, 1)
    bar = CuStaticSharedArray(Int64, 1)
    slot_addr = PTX.smem_addr_u32(pointer(slot))
    mbar_addr = PTX.smem_addr_u32(pointer(bar))
    ptx"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"(
        slot_addr, UInt32(64))
    taddr = @inbounds slot[1]
    ptx"tcgen05.cp.cta_group::1.128x128b"(taddr, s_desc)
    ptx"tcgen05.mma.cta_group::1.kind::f16"(taddr, s_desc, s_desc, idesc, false)
    ptx"tcgen05.mma.cta_group::2.kind::tf32"(taddr, s_desc, s_desc, idesc, true)
    ptx"tcgen05.mma.cta_group::1.kind::mxf8f6f4"(taddr, s_desc, s_desc, idesc, false)
    ptx"tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64"(mbar_addr)
    ptx"tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64"(mbar_addr)
    ptx"tcgen05.commit.cta_group::1.mbarrier::arrive::one.multicast::cluster.shared::cluster.b64"(
        mbar_addr, UInt16(0x3))
    r1 = ptx"tcgen05.ld.sync.aligned.32x32b.x1.b32"(taddr)
    r2 = ptx"tcgen05.ld.sync.aligned.16x128b.x2.b32"(taddr)
    ptx"tcgen05.wait::ld.sync.aligned"()
    ptx"tcgen05.st.sync.aligned.32x32b.x1.b32"(taddr, (r1,))
    ptx"tcgen05.st.sync.aligned.16x128b.x2.b32"(taddr, r2)
    ptx"tcgen05.wait::st.sync.aligned"()
    ptx"tcgen05.shift.cta_group::1.down"(taddr)
    ptx"tcgen05.dealloc.cta_group::1.sync.aligned.b32"(taddr, UInt32(64))
    ptx"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"()
    @inbounds out[1] = r1 + r2[1]
    return nothing
end

@testset "golden: tcgen05 family at sm_100a" begin
    @test golden_test("tcgen05@sm100a", _golden_tcgen05_sm100a!,
                      Tuple{CuDeviceVector{UInt32, 1}, UInt64, UInt32};
                      cap = v"10.0", feature_set = :arch)
end


# --- mma.sync.aligned: the three dense data conventions ----------------------
# bf16/tf32/fp8 inputs pack into UInt32 fragments; pure-f16 inputs and f16
# accumulators move as packed UInt32 too at the notation surface (the
# intrinsics use i32 vs v2f16 internally — the migration's repack is what
# this golden pins). One kernel per accumulator/input convention at the
# Hopper floor; fp8 + kind::f8f6f4 are consumer-Blackwell (sm_120a+, the
# GB10 path) and ride a separate golden.

function _golden_mma_classic!(out::CuDeviceVector{Float32, 1},
        a1::UInt32, a2::UInt32, a3::UInt32, a4::UInt32, b1::UInt32, b2::UInt32,
        h1::UInt32, h2::UInt32, h3::UInt32)
    a4t = (a1, a2, a3, a4); b2t = (b1, b2); c = (0f0, 0f0, 0f0, 0f0)
    # bf16 inputs, f32 acc — i32 A/B, f32 C/D
    d = ptx"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"(a4t, b2t, c)
    @inbounds out[1] = d[1]
    # f16 inputs, f32 acc — intrinsic wants v2f16 A/B
    d = ptx"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"(a4t, b2t, c)
    @inbounds out[2] = d[1]
    # f16 inputs, f16 acc — v2f16 everywhere; C/D packed UInt32 at the surface
    ch = (h1, h2)
    e = ptx"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"(a4t, b2t, ch)
    @inbounds out[3] = reinterpret(Float32, e[1])
    # tf32 inputs, f32 acc, k8 — i32 A/B
    a2t = (a1, a2); b1t = (b1,)
    d = ptx"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"(a2t, b1t, c)
    @inbounds out[4] = d[1]
    return nothing
end

@testset "golden: mma dense (classic conventions) at sm_90a" begin
    @test golden_test("mma@sm90a", _golden_mma_classic!,
                      Tuple{CuDeviceVector{Float32, 1},
                            UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                            UInt32, UInt32, UInt32};
                      cap = v"9.0", feature_set = :arch)
end

# fp8 (e4m3/e5m2) and kind::f8f6f4 — consumer-Blackwell sub-byte FP.
function _golden_mma_fp8!(out::CuDeviceVector{Float32, 1},
        a1::UInt32, a2::UInt32, a3::UInt32, a4::UInt32, b1::UInt32, b2::UInt32,
        h1::UInt32, h2::UInt32)
    a2t = (a1, a2); b1t = (b1,); c = (0f0, 0f0, 0f0, 0f0)
    # k16 fp8, f32 acc — 2 A regs, 1 B reg
    d = ptx"mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32"(a2t, b1t, c)
    @inbounds out[1] = d[1]
    # k16 fp8, f16 acc — C/D v2f16 internally, packed UInt32 at surface
    ch = (h1, h2)
    e = ptx"mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e4m3.f16"(a2t, b1t, ch)
    @inbounds out[2] = reinterpret(Float32, e[1])
    # k32 kind::f8f6f4 mixed, f32 acc — 4 A regs, 2 B regs
    a4t = (a1, a2, a3, a4); b2t = (b1, b2)
    d = ptx"mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e3m2.f32"(a4t, b2t, c)
    @inbounds out[3] = d[1]
    return nothing
end

@testset "golden: mma fp8 + kind::f8f6f4 at sm_121a" begin
    @test golden_test("mma_fp8@sm121a", _golden_mma_fp8!,
                      Tuple{CuDeviceVector{Float32, 1},
                            UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                            UInt32, UInt32};
                      cap = v"12.1", feature_set = :arch)
end

# Block-scaled (mxf8f6f4/mxf4/mxf4nvf4) — the microscaling path. mxf4nvf4
# scale_vec::4X with ue8m0 stays asm-tier (no intrinsic), pinned to prove
# the migration left it alone.
function _golden_mma_scaled!(out::CuDeviceVector{Float32, 1},
        a1::UInt32, a2::UInt32, a3::UInt32, a4::UInt32, b1::UInt32, b2::UInt32,
        sa::UInt32, sb::UInt32, bid::UInt16, tid::UInt16)
    a4t = (a1, a2, a3, a4); b2t = (b1, b2); c = (0f0, 0f0, 0f0, 0f0)
    d = ptx"mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0"(
        a4t, b2t, c, sa, bid, tid, sb, bid, tid)
    @inbounds out[1] = d[1]
    d = ptx"mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0"(
        a4t, b2t, c, sa, bid, tid, sb, bid, tid)
    @inbounds out[2] = d[1]
    d = ptx"mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3"(
        a4t, b2t, c, sa, bid, tid, sb, bid, tid)
    @inbounds out[3] = d[1]
    # residue: no intrinsic for nvf4 scale_vec::4X with ue8m0 stype
    d = ptx"mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0"(
        a4t, b2t, c, sa, bid, tid, sb, bid, tid)
    @inbounds out[4] = d[1]
    return nothing
end

@testset "golden: mma block-scaled at sm_121a" begin
    @test golden_test("mma_scaled@sm121a", _golden_mma_scaled!,
                      Tuple{CuDeviceVector{Float32, 1},
                            UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                            UInt32, UInt32, UInt16, UInt16};
                      cap = v"12.1", feature_set = :arch)
end
