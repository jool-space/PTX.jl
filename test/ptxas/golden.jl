# Golden-PTX locks, per family (DESIGN.md, "Approach"): each kernel pins a
# family's emitted instruction sequence (structural, modulo naming — see
# IR.canonicalize) so a lowering migration shows up as a reviewable git diff
# of the golden file instead of an act of faith. Regenerate deliberately:
#
#     PTX_UPDATE_GOLDEN=1 julia --project=test test/runtests.jl ptxas/golden
#
# One kernel per family, straight-line, touching every form the wrappers
# expose, at the lowest cap that supports the family.
#
# Bounds-check sensitivity: run the suite via `julia --project=test
# test/runtests.jl`, not bare `Pkg.test()` — Pkg.test defaults to
# --check-bounds=yes, which overrides @inbounds inside device code and
# injects bounds-check branches into these kernels, failing the structural
# comparison. (CI sets `check_bounds: 'auto'` on julia-runtest for the same
# reason.) The goldens pin the package's real emission under normal bounds
# semantics.


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


# --- ldmatrix / stmatrix: warp-cooperative matrix ld/st -----------------------
# Locks taken ahead of the tier-2 migration (the fence recipe): every m8n8.b16
# form the wrappers expose, both state-space spellings. Results/readbacks
# thread through `out` so no call is removable.

function _golden_ldmatrix!(out::CuDeviceVector{UInt32, 1})
    buf = CuStaticSharedArray(UInt16, 256)
    addr = pointer(buf)
    r1  = ptx"ldmatrix.sync.aligned.m8n8.x1.shared.b16"(addr)
    r1t = ptx"ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16"(addr)
    r2  = ptx"ldmatrix.sync.aligned.m8n8.x2.shared.b16"(addr)
    r2t = ptx"ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16"(addr)
    r4  = ptx"ldmatrix.sync.aligned.m8n8.x4.shared.b16"(addr)
    r4t = ptx"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"(addr)
    c1  = ptx"ldmatrix.sync.aligned.m8n8.x1.shared::cta.b16"(addr)
    c1t = ptx"ldmatrix.sync.aligned.m8n8.x1.trans.shared::cta.b16"(addr)
    c2  = ptx"ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16"(addr)
    c2t = ptx"ldmatrix.sync.aligned.m8n8.x2.trans.shared::cta.b16"(addr)
    c4  = ptx"ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16"(addr)
    c4t = ptx"ldmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16"(addr)
    @inbounds out[1] = r1 + r1t + r2[1] + r2t[2] + r4[1] + r4t[4] +
                       c1 + c1t + c2[1] + c2t[2] + c4[1] + c4t[4]
    return nothing
end

@testset "golden: ldmatrix m8n8.b16 family at sm_75" begin
    @test golden_test("ldmatrix@sm75", _golden_ldmatrix!,
                      Tuple{CuDeviceVector{UInt32, 1}}; cap = v"7.5")
end

function _golden_stmatrix!(out::CuDeviceVector{UInt16, 1},
                           a::UInt32, b::UInt32, c::UInt32, d::UInt32)
    buf = CuStaticSharedArray(UInt16, 256)
    addr = pointer(buf)
    ptx"stmatrix.sync.aligned.m8n8.x1.shared.b16"(addr, a)
    ptx"stmatrix.sync.aligned.m8n8.x1.trans.shared.b16"(addr, a)
    ptx"stmatrix.sync.aligned.m8n8.x2.shared.b16"(addr, (a, b))
    ptx"stmatrix.sync.aligned.m8n8.x2.trans.shared.b16"(addr, (a, b))
    ptx"stmatrix.sync.aligned.m8n8.x4.shared.b16"(addr, (a, b, c, d))
    ptx"stmatrix.sync.aligned.m8n8.x4.trans.shared.b16"(addr, (a, b, c, d))
    ptx"stmatrix.sync.aligned.m8n8.x1.shared::cta.b16"(addr, a)
    ptx"stmatrix.sync.aligned.m8n8.x1.trans.shared::cta.b16"(addr, a)
    ptx"stmatrix.sync.aligned.m8n8.x2.shared::cta.b16"(addr, (a, b))
    ptx"stmatrix.sync.aligned.m8n8.x2.trans.shared::cta.b16"(addr, (a, b))
    ptx"stmatrix.sync.aligned.m8n8.x4.shared::cta.b16"(addr, (a, b, c, d))
    ptx"stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16"(addr, (a, b, c, d))
    @inbounds out[1] = buf[1]
    return nothing
end

@testset "golden: stmatrix m8n8.b16 family at sm_90" begin
    @test golden_test("stmatrix@sm90", _golden_stmatrix!,
                      Tuple{CuDeviceVector{UInt16, 1},
                            UInt32, UInt32, UInt32, UInt32}; cap = v"9.0")
end

# The b8 shapes (ldmatrix m16n16, stmatrix m16n8) are deliberately NOT
# locked here: every form the asm tier currently generates is
# ptxas-rejected — non-trans and m16n16-x4 are invalid modifier combos,
# and the "valid" .trans forms emit the wrong destination arity (m16n16
# is 2 regs per x1 count, the generator assumed 1). There is no working
# emission to lock; the migration rebuilds the family on the registry
# surface and introduces its golden then.


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


# --- proxy / init fences -----------------------------------------------------
# The three special fences that fence a specific memory proxy (async,
# mbarrier-init) rather than ordering generic memory — not expressible as a
# core-IR `fence`, so tier-2 intrinsics.

function _golden_fences!()
    ptx"fence.proxy.async"()
    ptx"fence.proxy.async.shared::cta"()
    ptx"fence.mbarrier_init.release.cluster"()
    return nothing
end

@testset "golden: proxy/init fences at sm_90a" begin
    @test golden_test("fences@sm90a", _golden_fences!, Tuple{};
                      cap = v"9.0", feature_set = :arch)
end


# --- generic memory fences: {sc, acq_rel} × {cta, gpu, sys} (+cluster) ------
# Tier-1 core IR: `fence <ordering> syncscope(...)` is a *semantic*
# translation of the PTX sem/scope pair, not a renaming (mapping pinned in
# wrappers/fence.jl and CONCERNS.md). Stores between the fences keep each
# fence's program position observable, so the golden pins the ordering of
# the whole sequence — a real core-IR fence pins memory ops, and nothing
# pins two adjacent fences except the accesses between them.

function _golden_fences_generic!(out::CuDeviceVector{UInt32, 1})
    @inbounds begin
        out[1] = UInt32(1)
        ptx"fence.sc.cta"()
        out[2] = UInt32(2)
        ptx"fence.sc.gpu"()
        out[3] = UInt32(3)
        ptx"fence.sc.sys"()
        out[4] = UInt32(4)
        ptx"fence.acq_rel.cta"()
        out[5] = UInt32(5)
        ptx"fence.acq_rel.gpu"()
        out[6] = UInt32(6)
        ptx"fence.acq_rel.sys"()
        out[7] = UInt32(7)
    end
    return nothing
end

# Cluster scope is sm_90+ (ISel enforces the floor loudly: "Requires SM >= 90
# and PTX >= 78"); the other six forms are pinned at the sm_70 floor above.
function _golden_fences_generic_sm90!(out::CuDeviceVector{UInt32, 1})
    @inbounds begin
        out[1] = UInt32(1)
        ptx"fence.sc.cluster"()
        out[2] = UInt32(2)
        ptx"fence.acq_rel.cluster"()
        out[3] = UInt32(3)
    end
    return nothing
end

@testset "golden: generic memory fences at sm_70" begin
    @test golden_test("fences_generic@sm70", _golden_fences_generic!,
                      Tuple{CuDeviceVector{UInt32, 1}}; cap = v"7.0")
end

@testset "golden: generic memory fences (cluster scope) at sm_90" begin
    @test golden_test("fences_generic@sm90", _golden_fences_generic_sm90!,
                      Tuple{CuDeviceVector{UInt32, 1}}; cap = v"9.0")
end


# --- vec ld/st.global: all six forms (v{2,4} × {f32,b32,b16}) ---------------
# Vectorized global load/store. The lowest-cap universal data path — no arch
# gate. Migrating from an asm `~{memory}` barrier to a tier-1 core-IR
# `load`/`store <N x T>` makes these real, optimizable, reorderable memory
# ops; the golden pins the emitted ld.global.v* / st.global.v* sequence so
# that change is a reviewable diff rather than an act of faith.

function _golden_vec_ldst!(pf::CuDeviceVector{Float32, 1},
                           pu::CuDeviceVector{UInt32, 1},
                           ph::CuDeviceVector{UInt16, 1})
    qf = pointer(pf); qu = pointer(pu); qh = pointer(ph)
    @inbounds begin
        # f32 / b32: load and store back at a distinct offset. The values flow
        # through unmodified, but the store address differs from the load
        # address, so the round-trip is a real, non-eliminable side effect.
        # (NVPTX canonicalizes float vector ld/st to the `.b32` bit spelling —
        # registers are typeless, so this is purely cosmetic vs `.f32`.)
        ptx"st.global.v4.f32"(qf + 32, ptx"ld.global.v4.f32"(qf))
        ptx"st.global.v2.f32"(qf + 64, ptx"ld.global.v2.f32"(qf + 16))
        ptx"st.global.v4.b32"(qu + 32, ptx"ld.global.v4.b32"(qu))
        ptx"st.global.v2.b32"(qu + 64, ptx"ld.global.v2.b32"(qu + 16))

        # b16: touch each lane (add 1) so the backend keeps the narrow
        # `v{2,4}.b16` form. An opaque b16 passthrough would legally coalesce
        # into a wider `.b32` access (same bytes, same transaction) — pinning
        # the per-lane form locks the wrapper's actual contract, not the
        # optimizer's packing choice.
        h4 = ptx"ld.global.v4.b16"(qh)
        ptx"st.global.v4.b16"(qh + 16,
            (ptx"add.s16"(h4[1], UInt16(1)), ptx"add.s16"(h4[2], UInt16(1)),
             ptx"add.s16"(h4[3], UInt16(1)), ptx"add.s16"(h4[4], UInt16(1))))
        h2 = ptx"ld.global.v2.b16"(qh + 8)
        ptx"st.global.v2.b16"(qh + 24,
            (ptx"add.s16"(h2[1], UInt16(1)), ptx"add.s16"(h2[2], UInt16(1))))
    end
    return nothing
end

@testset "golden: vec ld/st.global at sm_70" begin
    @test golden_test("vec_ldst@sm70", _golden_vec_ldst!,
                      Tuple{CuDeviceVector{Float32, 1},
                            CuDeviceVector{UInt32, 1},
                            CuDeviceVector{UInt16, 1}}; cap = v"7.0")
end
