# Blackwell (sm_100a) wrapper validation. Same compile-through-ptxas
# pattern as ptxas/hopper.jl: every kernel ptxas-validates without ever
# being launched, so an Ada (CC 8.9) box can validate sm_100a wrappers.
#
# Coverage: add.f32x2, sub-byte FP cvt (e2m1x2 / e2m3x2 / e3m2x2 /
# ue8m0x2). tcgen05 wrappers are not in PTX.jl yet — once they are, add
# tests here and validate at sm_100a.


# --- add.f32x2 (SIMD32-pair add) -------------------------------------------
#
# add.f32x2 packs two Float32 lanes into a UInt64 and adds element-wise.
# Introduced in PTX 8.6 / sm_100. ptxas's gating message is unambiguous:
#   error : Feature 'add.f32x2' requires .target sm_100 or higher

function _bw_add_f32x2!(out::CuDeviceVector{UInt64, 1},
                        a::UInt64, b::UInt64)
    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        @inbounds out[1] = ptx"add.f32x2"(a, b)
    end
    return nothing
end

@testset "add.f32x2 at sm_100" begin
    types = Tuple{CuDeviceVector{UInt64, 1}, UInt64, UInt64}
    @test ptxas_compiles(_bw_add_f32x2!, types;
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_bw_add_f32x2!, types;
                   cap = v"10.0", feature_set = :arch)
    @test occursin(".target sm_100a", ptx)
    @test occursin("add.f32x2", ptx)
end


# --- sub-byte FP cvt (PTX 8.6 / sm_100) ------------------------------------
#
# cvt.rn.satfinite.e2m1x2.f32 / e2m3x2 / e3m2x2 / ue8m0x2 pack two FP32
# values into a sub-byte float pair. ptxas only accepts these on sm_100+.

function _bw_cvt_e2m1x2!(out::CuDeviceVector{UInt16, 1}, x::Float32)
    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        @inbounds out[1] = ptx"cvt.rn.satfinite.e2m1x2.f32"(x, x)
    end
    return nothing
end

function _bw_cvt_e2m3x2!(out::CuDeviceVector{UInt16, 1}, x::Float32)
    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        @inbounds out[1] = ptx"cvt.rn.satfinite.e2m3x2.f32"(x, x)
    end
    return nothing
end

function _bw_cvt_e3m2x2!(out::CuDeviceVector{UInt16, 1}, x::Float32)
    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        @inbounds out[1] = ptx"cvt.rn.satfinite.e3m2x2.f32"(x, x)
    end
    return nothing
end

function _bw_cvt_ue8m0x2!(out::CuDeviceVector{UInt16, 1}, x::Float32)
    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        @inbounds out[1] = ptx"cvt.rz.satfinite.ue8m0x2.f32"(x, x)
    end
    return nothing
end

@testset "cvt sub-byte FP at sm_100" begin
    types_x16 = Tuple{CuDeviceVector{UInt16, 1}, Float32}
    @test ptxas_compiles(_bw_cvt_e2m1x2!, types_x16;
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_bw_cvt_e2m1x2!, types_x16;
                   cap = v"10.0", feature_set = :arch)
    @test occursin("cvt.rn.satfinite.e2m1x2.f32", ptx)

    @test ptxas_compiles(_bw_cvt_e2m3x2!, types_x16;
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_bw_cvt_e2m3x2!, types_x16;
                   cap = v"10.0", feature_set = :arch)
    @test occursin("cvt.rn.satfinite.e2m3x2.f32", ptx)

    @test ptxas_compiles(_bw_cvt_e3m2x2!, types_x16;
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_bw_cvt_e3m2x2!, types_x16;
                   cap = v"10.0", feature_set = :arch)
    @test occursin("cvt.rn.satfinite.e3m2x2.f32", ptx)

    @test ptxas_compiles(_bw_cvt_ue8m0x2!, types_x16;
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_bw_cvt_ue8m0x2!, types_x16;
                   cap = v"10.0", feature_set = :arch)
    @test occursin("cvt.rz.satfinite.ue8m0x2.f32", ptx)
end


# --- scaled cvt (mxfp scale) -----------------------------------------------
#
# cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 — fan-out FP8 → BF16 with a
# UE8M0 scale broadcast (mxfp pattern). PTX 8.6+ / sm_100+.

function _bw_cvt_scaled!(out::CuDeviceVector{UInt32, 1},
                          x::UInt16, scale::UInt8)
    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        @inbounds out[1] = ptx"cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2"(x, scale)
    end
    return nothing
end

@testset "cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 at sm_100" begin
    types = Tuple{CuDeviceVector{UInt32, 1}, UInt16, UInt8}
    @test ptxas_compiles(_bw_cvt_scaled!, types;
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_bw_cvt_scaled!, types;
                   cap = v"10.0", feature_set = :arch)
    @test occursin("cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2", ptx)
end


# --- FP8 → BF16 unpack cvt at sm_100a (PTX 9.2) ---------------------------
#
# `cvt.rn.bf16x2.{e4m3x2,e5m2x2}` introduced in PTX 9.2; despite the FP8
# source types being sm_89-baseline, the BF16 destination needs sm_100a+.
# Both go through the chain default — `.b16` source carrier.

function _bw_cvt_fp8_to_bf16!(out::CuDeviceVector{UInt32, 1}, x::UInt16)
    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        @inbounds out[1] = ptx"cvt.rn.bf16x2.e4m3x2"(x)
        @inbounds out[2] = ptx"cvt.rn.bf16x2.e5m2x2"(x)
    end
    return nothing
end

@testset "cvt FP8 → BF16x2 at sm_100a" begin
    types = Tuple{CuDeviceVector{UInt32, 1}, UInt16}
    @test ptxas_compiles(_bw_cvt_fp8_to_bf16!, types;
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_bw_cvt_fp8_to_bf16!, types;
                   cap = v"10.0", feature_set = :arch)
    @test occursin("cvt.rn.bf16x2.e4m3x2", ptx)
    @test occursin("cvt.rn.bf16x2.e5m2x2", ptx)
end


# --- sub-byte FP unpack cvt at sm_100a ------------------------------------
#
# `cvt.rn.f16x2.{e2m3x2,e3m2x2}` (PTX 8.6) and `cvt.rn.bf16x2.{e3m2x2,
# e2m3x2,e2m1x2}` (PTX 9.2) — the unpack mirror of the sub-byte FP packing
# wrappers in src/wrappers/cvt.jl. The .b16-source forms (e2m3x2 / e3m2x2)
# go through the chain default; e2m1x2 needs the hand-written wrapper
# because the source carrier is `.b8` (no NVPTX i8 constraint).

function _bw_cvt_fp6_unpack!(out::CuDeviceVector{UInt32, 1}, x::UInt16)
    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        @inbounds out[1] = ptx"cvt.rn.f16x2.e2m3x2"(x)
        @inbounds out[2] = ptx"cvt.rn.f16x2.e3m2x2"(x)
        @inbounds out[3] = ptx"cvt.rn.bf16x2.e2m3x2"(x)
        @inbounds out[4] = ptx"cvt.rn.bf16x2.e3m2x2"(x)
    end
    return nothing
end

@testset "cvt FP6 unpack to FP16/BF16 at sm_100a" begin
    types = Tuple{CuDeviceVector{UInt32, 1}, UInt16}
    @test ptxas_compiles(_bw_cvt_fp6_unpack!, types;
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_bw_cvt_fp6_unpack!, types;
                   cap = v"10.0", feature_set = :arch)
    @test occursin("cvt.rn.f16x2.e2m3x2",  ptx)
    @test occursin("cvt.rn.f16x2.e3m2x2",  ptx)
    @test occursin("cvt.rn.bf16x2.e2m3x2", ptx)
    @test occursin("cvt.rn.bf16x2.e3m2x2", ptx)
end

# e2m1x2 (FP4) unpack uses the hand-written wrapper. UInt16 carrier in,
# UInt32 (f16x2 / bf16x2) out; carrier shim extracts the low byte to a
# .b8 reg before the cvt — see comment header in src/wrappers/cvt.jl.

function _bw_cvt_fp4_unpack!(out::CuDeviceVector{UInt32, 1}, x::UInt16)
    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        @inbounds out[1] = ptx"cvt.rn.f16x2.e2m1x2"(x)
        @inbounds out[2] = ptx"cvt.rn.bf16x2.e2m1x2"(x)
    end
    return nothing
end

@testset "cvt FP4 (e2m1x2) unpack to FP16/BF16 at sm_100a" begin
    types = Tuple{CuDeviceVector{UInt32, 1}, UInt16}
    @test ptxas_compiles(_bw_cvt_fp4_unpack!, types;
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_bw_cvt_fp4_unpack!, types;
                   cap = v"10.0", feature_set = :arch)
    @test occursin("cvt.rn.f16x2.e2m1x2",  ptx)
    @test occursin("cvt.rn.bf16x2.e2m1x2", ptx)
end

# e2m1x2 (FP4) direct pack from packed FP16/BF16 (no f32 detour). Hand-
# written wrapper in src/wrappers/cvt.jl: UInt32 (f16x2/bf16x2) in,
# UInt16 (e2m1x2 carrier) out via a `.reg .b8 t; ... mov.b16 \$0, {t, 0};`
# shim. The pack-from-f32 form is covered above; this locks the pack-from-
# packed forms.

function _bw_cvt_fp4_pack_from_x2!(out::CuDeviceVector{UInt16, 1},
                                    x_f16x2::UInt32, x_bf16x2::UInt32)
    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        @inbounds out[1] = ptx"cvt.rn.satfinite.e2m1x2.f16x2"(x_f16x2)
        @inbounds out[2] = ptx"cvt.rn.satfinite.e2m1x2.bf16x2"(x_bf16x2)
    end
    return nothing
end

@testset "cvt FP4 (e2m1x2) pack from f16x2/bf16x2 at sm_100a" begin
    types = Tuple{CuDeviceVector{UInt16, 1}, UInt32, UInt32}
    @test ptxas_compiles(_bw_cvt_fp4_pack_from_x2!, types;
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_bw_cvt_fp4_pack_from_x2!, types;
                   cap = v"10.0", feature_set = :arch)
    @test occursin("cvt.rn.satfinite.e2m1x2.f16x2",  ptx)
    @test occursin("cvt.rn.satfinite.e2m1x2.bf16x2", ptx)
end

# `cvt.rn.bf16x2.ue8m0x2` — block-scale dtype unpacked to BF16 (PTX 8.6).
function _bw_cvt_ue8m0_unpack!(out::CuDeviceVector{UInt32, 1}, x::UInt16)
    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        @inbounds out[1] = ptx"cvt.rn.bf16x2.ue8m0x2"(x)
    end
    return nothing
end

@testset "cvt.rn.bf16x2.ue8m0x2 at sm_100a" begin
    types = Tuple{CuDeviceVector{UInt32, 1}, UInt16}
    @test ptxas_compiles(_bw_cvt_ue8m0_unpack!, types;
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_bw_cvt_ue8m0_unpack!, types;
                   cap = v"10.0", feature_set = :arch)
    @test occursin("cvt.rn.bf16x2.ue8m0x2", ptx)
end


# --- tcgen05 sink-form chain default at sm_100a --------------------------
#
# tcgen05 mixes memory-pointer-taking sinks (alloc destination, commit's
# mbarrier ptr) with TMEM-address-taking ops (shift, cp, dealloc, ld, st)
# that need a 32-bit address operand without an LLVMPtr carrier. Memory-
# pointer sinks plus the no-arg fence/wait ops route through chain default;
# the TMEM-address ops have typed wrappers in src/wrappers/tcgen05.jl,
# ported from pyptx's _Tcgen05 operand discipline.
#
# Per-prefix NO_RETURN_PREFIXES gate in src/types.jl suppresses the
# spurious `.b32`/`.b64` return-type inference for `alloc`, `commit`,
# `relinquish_alloc_permit`.
#
# LLVM corpus reference: test/corpus/external/llvm/tcgen05-{alloc,commit,
# fence}__test_*.ptx — verified call-site forms.

function _bw_tcgen05_lifecycle!(taddr_dst::Core.LLVMPtr{UInt32, PTX.AS.Shared},
                                 mbar::Core.LLVMPtr{UInt64, PTX.AS.Shared},
                                 ncols::UInt32)
    # alloc the dynamic TMEM region (writes the taddr to [taddr_dst])
    ptx"tcgen05.alloc.cta_group::1.sync.aligned.b32"(taddr_dst, ncols)
    # arrive on an mbarrier when all prior async tcgen05 ops complete
    ptx"tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"(mbar)
    # tear-down lifecycle
    ptx"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"()
    return nothing
end

@testset "tcgen05 lifecycle (alloc/commit/relinquish) at sm_100a" begin
    types = Tuple{Core.LLVMPtr{UInt32, PTX.AS.Shared},
                  Core.LLVMPtr{UInt64, PTX.AS.Shared},
                  UInt32}
    @test ptxas_compiles(_bw_tcgen05_lifecycle!, types;
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_bw_tcgen05_lifecycle!, types;
                   cap = v"10.0", feature_set = :arch)
    @test occursin("tcgen05.alloc.cta_group::1.sync.aligned.b32", ptx)
    @test occursin("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64", ptx)
    @test occursin("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned", ptx)
end

# tcgen05 fences and waits — pure no-arg ops, chain default sufficient.
function _bw_tcgen05_sync!()
    ptx"tcgen05.fence::before_thread_sync"()
    ptx"tcgen05.fence::after_thread_sync"()
    ptx"tcgen05.wait::ld.sync.aligned"()
    ptx"tcgen05.wait::st.sync.aligned"()
    return nothing
end

@testset "tcgen05 fences + waits at sm_100a" begin
    @test ptxas_compiles(_bw_tcgen05_sync!, Tuple{};
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_bw_tcgen05_sync!, Tuple{};
                   cap = v"10.0", feature_set = :arch)
    @test occursin("tcgen05.fence::before_thread_sync", ptx)
    @test occursin("tcgen05.fence::after_thread_sync",  ptx)
    @test occursin("tcgen05.wait::ld.sync.aligned",     ptx)
    @test occursin("tcgen05.wait::st.sync.aligned",     ptx)
end


# --- tcgen05 typed wrappers (TMEM-address operand ops) ------------------
#
# shift / cp / dealloc / ld / st take a 32-bit TMEM address (returned by
# alloc), not a memory pointer. Ported from pyptx/pyptx/ptx.py `_Tcgen05`.

function _bw_tcgen05_shift!(taddr::UInt32)
    ptx"tcgen05.shift.cta_group::1.down"(taddr)
    return nothing
end

function _bw_tcgen05_dealloc!(taddr::UInt32, ncols::UInt32)
    ptx"tcgen05.dealloc.cta_group::1.sync.aligned.b32"(taddr, ncols)
    return nothing
end

function _bw_tcgen05_cp!(taddr::UInt32, s_desc::UInt64)
    ptx"tcgen05.cp.cta_group::1.128x256b"(taddr, s_desc)
    return nothing
end

function _bw_tcgen05_ld_st!(taddr::UInt32)
    # 16x128b.x1 → 2 regs/lane; round-trip through a tcgen05 register file.
    r = ptx"tcgen05.ld.sync.aligned.16x128b.x1.b32"(taddr)
    ptx"tcgen05.wait::ld.sync.aligned"()
    ptx"tcgen05.st.sync.aligned.16x128b.x1.b32"(taddr, r)
    ptx"tcgen05.wait::st.sync.aligned"()
    return nothing
end

function _bw_tcgen05_ld_wide!(taddr::UInt32)
    # 16x256b.x4 → 16 regs/lane; exercises the multi-register tuple form.
    r = ptx"tcgen05.ld.sync.aligned.16x256b.x4.b32"(taddr)
    ptx"tcgen05.wait::ld.sync.aligned"()
    ptx"tcgen05.st.sync.aligned.16x256b.x4.b32"(taddr, r)
    return nothing
end

@testset "tcgen05 typed TMEM-addr wrappers at sm_100a" begin
    @test ptxas_compiles(_bw_tcgen05_shift!, Tuple{UInt32};
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_bw_tcgen05_shift!, Tuple{UInt32};
                   cap = v"10.0", feature_set = :arch)
    @test occursin("tcgen05.shift.cta_group::1.down", ptx)

    @test ptxas_compiles(_bw_tcgen05_dealloc!, Tuple{UInt32, UInt32};
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_bw_tcgen05_dealloc!, Tuple{UInt32, UInt32};
                   cap = v"10.0", feature_set = :arch)
    @test occursin("tcgen05.dealloc.cta_group::1.sync.aligned.b32", ptx)

    @test ptxas_compiles(_bw_tcgen05_cp!, Tuple{UInt32, UInt64};
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_bw_tcgen05_cp!, Tuple{UInt32, UInt64};
                   cap = v"10.0", feature_set = :arch)
    @test occursin("tcgen05.cp.cta_group::1.128x256b", ptx)

    @test ptxas_compiles(_bw_tcgen05_ld_st!, Tuple{UInt32};
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_bw_tcgen05_ld_st!, Tuple{UInt32};
                   cap = v"10.0", feature_set = :arch)
    @test occursin("tcgen05.ld.sync.aligned.16x128b.x1.b32", ptx)
    @test occursin("tcgen05.st.sync.aligned.16x128b.x1.b32", ptx)

    @test ptxas_compiles(_bw_tcgen05_ld_wide!, Tuple{UInt32};
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_bw_tcgen05_ld_wide!, Tuple{UInt32};
                   cap = v"10.0", feature_set = :arch)
    @test occursin("tcgen05.ld.sync.aligned.16x256b.x4.b32", ptx)
    @test occursin("tcgen05.st.sync.aligned.16x256b.x4.b32", ptx)
end


# --- blackwell-2: cluster / 2-SM gating wrappers --------------------------
#
# The genuinely-missing surface for the production gemm_highperf_blackwell
# 2-SM cooperative path (its 1-SM persistent path needs no new wrappers —
# already shipped in blackwell-1 + the Hopper TMA waves):
#
#   * tcgen05.commit.multicast::cluster + u16 mask — the MMA-retire arrive
#     fires on every cluster CTA's local mbarrier (mask bit set). Cluster
#     state space ONLY: ptxas rejects `.shared::cta` here ("State space
#     incorrect for instruction 'tcgen05.commit'"), so the multicast verb
#     registers `shared::cluster` only (pyptx's builder is syntactically
#     permissive about space; the assembler is not).
#   * cp.async.bulk.tensor.<N>d ... .cta_group::2 — Blackwell 2-SM
#     cooperative load. The cluster-destination forms lower via the
#     g2s.tile intrinsics, whose ISel renders `.cta_group::2` after the
#     completion mechanism (notation keeps the pyptx post-rank order; see
#     wrappers/tma.jl). The shared::cta × cta_group::2 form is asm-tier
#     and keeps the post-rank spelling — ptxas accepts both.
#
# Ported from pyptx _Tcgen05.commit(multicast=True) /
# _CpAsyncBulkTensor._tile_load(cta_group=2).

# cta_group::1 and ::2 must be in SEPARATE functions — ptxas: a function
# "uses single CTA and CTA pair granularity and that is not allowed"
# (digest §11.2: every tcgen05 op in a kernel shares one .cta_group).
function _bw_tcgen05_commit_mc1!(mbar::UInt32)
    ptx"tcgen05.commit.cta_group::1.mbarrier::arrive::one.multicast::cluster.shared::cluster.b64"(mbar, UInt16(0x3))
    return nothing
end

function _bw_tcgen05_commit_mc2!(mbar::UInt32)
    ptx"tcgen05.commit.cta_group::2.mbarrier::arrive::one.multicast::cluster.shared::cluster.b64"(mbar, UInt16(0x3))
    return nothing
end

@testset "tcgen05.commit.multicast::cluster at sm_100a" begin
    @test ptxas_compiles(_bw_tcgen05_commit_mc1!, Tuple{UInt32};
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_bw_tcgen05_commit_mc1!, Tuple{UInt32};
                   cap = v"10.0", feature_set = :arch)
    @test occursin("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64", ptx)

    @test ptxas_compiles(_bw_tcgen05_commit_mc2!, Tuple{UInt32};
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_bw_tcgen05_commit_mc2!, Tuple{UInt32};
                   cap = v"10.0", feature_set = :arch)
    @test occursin("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64", ptx)
end

function _bw_tma_cta_group2!(dst::Core.LLVMPtr{UInt16, PTX.AS.Shared},
                             tm::Core.LLVMPtr{UInt8, PTX.AS.Const},
                             mbar::Core.LLVMPtr{UInt64, PTX.AS.Shared})
    ptx"cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"(dst, tm, 0, 0, mbar)
    ptx"cp.async.bulk.tensor.2d.cta_group::2.shared::cta.global.tile.mbarrier::complete_tx::bytes"(dst, tm, 0, 0, mbar)
    ptx"cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"(dst, tm, 0, 0, mbar, UInt16(0x3))
    return nothing
end

@testset "cp.async.bulk.tensor.cta_group::2 at sm_100a" begin
    types = Tuple{Core.LLVMPtr{UInt16, PTX.AS.Shared},
                  Core.LLVMPtr{UInt8, PTX.AS.Const},
                  Core.LLVMPtr{UInt64, PTX.AS.Shared}}
    @test ptxas_compiles(_bw_tma_cta_group2!, types;
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_bw_tma_cta_group2!, types;
                   cap = v"10.0", feature_set = :arch)
    @test occursin("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2", ptx)
    @test occursin("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2", ptx)
    @test occursin("cp.async.bulk.tensor.2d.cta_group::2.shared::cta.global.tile.mbarrier::complete_tx::bytes", ptx)
end
