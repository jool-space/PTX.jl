# Blackwell (sm_100a) wrapper validation. Same compile-through-ptxas
# pattern as ptxas/hopper.jl: every kernel ptxas-validates without ever
# being launched, so an Ada (CC 8.9) box can validate sm_100a wrappers.
#
# Coverage: add.f32x2, sub-byte FP cvt (e2m1x2 / e2m3x2 / e3m2x2 /
# ue8m0x2), and tcgen05 lifecycle/data-movement/MMA wrappers.  TCGEN MX
# block-scale forms are validated on exact `a` and `f` target classes below.


# --- clusterlaunchcontrol.try_cancel address roles -------------------------
# PTX 9.3 §9.7.14.18: four grammar forms (generic/shared::cta ×
# base/multicast), all with two mandatory bracketed addresses. These are
# compile-only: meaningful execution requires a cluster launch, a live
# mbarrier phase, and 16-byte aligned response storage.

function _bw_clc_try_cancel_generic_u32!(response::UInt32, mbar::UInt32)
    ptx"clusterlaunchcontrol.try_cancel.async.mbarrier::complete_tx::bytes.b128"(
        address(response), address(mbar))
    return nothing
end

function _bw_clc_try_cancel_generic_u64!(response::UInt64, mbar::UInt64)
    ptx"clusterlaunchcontrol.try_cancel.async.mbarrier::complete_tx::bytes.b128"(
        address(response), address(mbar))
    return nothing
end

function _bw_clc_try_cancel_shared!(response::UInt32, mbar::UInt32)
    ptx"clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128"(
        address(response), address(mbar))
    return nothing
end

function _bw_clc_try_cancel_generic_multicast!(response::UInt64, mbar::UInt64)
    ptx"clusterlaunchcontrol.try_cancel.async.mbarrier::complete_tx::bytes.multicast::cluster::all.b128"(
        address(response), address(mbar))
    return nothing
end

function _bw_clc_try_cancel_shared_multicast!(response::UInt32, mbar::UInt32)
    ptx"clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128"(
        address(response), address(mbar))
    return nothing
end

@testset "clusterlaunchcontrol.try_cancel base forms at sm_100" begin
    generic_u32_types = Tuple{UInt32, UInt32}
    generic_u64_types = Tuple{UInt64, UInt64}
    shared_types = Tuple{UInt32, UInt32}
    @test ptxas_compiles(_bw_clc_try_cancel_generic_u32!, generic_u32_types;
                         cap = v"10.0", feature_set = :baseline)
    @test ptxas_compiles(_bw_clc_try_cancel_generic_u64!, generic_u64_types;
                         cap = v"10.0", feature_set = :baseline)
    @test ptxas_compiles(_bw_clc_try_cancel_shared!, shared_types;
                         cap = v"10.0", feature_set = :baseline)
    generic_u32_ptx = emit_ptx(_bw_clc_try_cancel_generic_u32!,
                               generic_u32_types;
                               cap = v"10.0", feature_set = :baseline)
    generic_u64_ptx = emit_ptx(_bw_clc_try_cancel_generic_u64!,
                               generic_u64_types;
                               cap = v"10.0", feature_set = :baseline)
    shared_ptx = emit_ptx(_bw_clc_try_cancel_shared!, shared_types;
                          cap = v"10.0", feature_set = :baseline)
    @test occursin(
        r"clusterlaunchcontrol\.try_cancel\.async\.mbarrier::complete_tx::bytes\.b128 \[%r\d+\], \[%r\d+\]",
        generic_u32_ptx)
    @test occursin(
        r"clusterlaunchcontrol\.try_cancel\.async\.mbarrier::complete_tx::bytes\.b128 \[%rd\d+\], \[%rd\d+\]",
        generic_u64_ptx)
    @test occursin(
        r"clusterlaunchcontrol\.try_cancel\.async\.shared::cta\.mbarrier::complete_tx::bytes\.b128 \[%r\d+\], \[%r\d+\]",
        shared_ptx)
end

@testset "clusterlaunchcontrol.try_cancel multicast forms at sm_100a/f" begin
    cases = (
        (_bw_clc_try_cancel_generic_multicast!, Tuple{UInt64, UInt64}, "%rd"),
        (_bw_clc_try_cancel_shared_multicast!, Tuple{UInt32, UInt32}, "%r"),
    )
    for feature_set in (:arch, :family), (f, tt, register) in cases
        @test ptxas_compiles(f, tt; cap = v"10.0", feature_set)
        ptx = emit_ptx(f, tt; cap = v"10.0", feature_set)
        @test occursin(".target sm_100" * (feature_set === :arch ? "a" : "f"), ptx)
        @test occursin(".multicast::cluster::all.b128", ptx)
        @test length(collect(eachmatch(Regex("\\[\\" * register * "\\d+\\]"), ptx))) >= 2
    end
end


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


# --- tcgen05 lifecycle wrappers at sm_100a -------------------------------
#
# tcgen05 mixes memory-pointer-taking sinks (alloc destination, commit's
# mbarrier ptr) with TMEM-address-taking ops (shift, cp, dealloc, ld, st)
# that need a 32-bit address operand without an LLVMPtr carrier. All tcgen05
# calls now require exact typed wrappers; in particular the pointer lifecycle
# forms and no-arg fences can no longer fall through to scalar rendering.
# The lifecycle verbs are single-route asm (see wrappers/tcgen05.jl): the
# generic-address alloc spelling below renders literally (the retired
# intrinsic route canonicalized it to an explicit `.shared::cta`).
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

# tcgen05 fences and waits — exact no-arg wrappers. The fences intentionally
# use side-effecting inline PTX with a memory clobber so LLVM versions that do
# not recognize their NVVM intrinsics cannot delete or move them.
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
    spellings = (
        "tcgen05.fence::before_thread_sync",
        "tcgen05.fence::after_thread_sync",
        "tcgen05.wait::ld.sync.aligned",
        "tcgen05.wait::st.sync.aligned",
    )
    positions = map(spellings) do spelling
        @test length(findall(spelling, ptx)) == 1
        findfirst(spelling, ptx)
    end
    @test all(!isnothing, positions)
    if all(!isnothing, positions)
        @test issorted(first.(positions))
    end
end


# --- tcgen05 typed wrappers (TMEM-address operand ops) ------------------
#
# shift / cp / dealloc / ld / st take a 32-bit TMEM address (returned by
# alloc), not a memory pointer. Ported from pyptx/pyptx/ptx.py `_Tcgen05`.

function _bw_tcgen05_shift!(taddr::UInt32)
    # Exercise the exact Address{UInt32} adapter rather than the bare payload
    # method; both must select the same tcgen05 intrinsic and PTX spelling.
    ptx"tcgen05.shift.cta_group::1.down"(address(taddr))
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


# --- tcgen05 MX block-scale schemas on exact a/f targets -------------------
#
# PTX 9.3 §9.7.17.10.9.1 and Table 60 expose eight modifier spellings.
# `.scale_vec::*` is architecture-specific and is compiled for both sm_100a
# and sm_110a.  The equivalent `.block16`/`.block32` aliases are compiled for
# both family targets, sm_100f and sm_110f.  These are offline PTX→ptxas
# checks only; no tcgen05 instruction is launched on the CC 12.1 GB10 runner.
#
# The cases alternate the two legal A source schemas (SMEM descriptor `ss`
# and bracketed TMEM address `ts`) and both CTA groups.  Host tests inventory
# the complete 8 modifiers × 2 CTA groups × 2 A sources surface.
const _TCGEN05_MX_PTXAS_CASES = (
    # Architecture-specific scale-vector spellings: sm_100a + sm_110a.
    (:mxf8f6f4, Symbol("scale_vec::1X"), 1, :ss, v"10.0", :arch, "sm_100a"),
    (:mxf8f6f4, Symbol("scale_vec::1X"), 2, :ts, v"11.0", :arch, "sm_110a"),
    (:mxf4,     Symbol("scale_vec::2X"), 2, :ts, v"10.0", :arch, "sm_100a"),
    (:mxf4,     Symbol("scale_vec::2X"), 1, :ss, v"11.0", :arch, "sm_110a"),
    (:mxf4nvf4, Symbol("scale_vec::2X"), 1, :ss, v"10.0", :arch, "sm_100a"),
    (:mxf4nvf4, Symbol("scale_vec::2X"), 2, :ts, v"11.0", :arch, "sm_110a"),
    (:mxf4nvf4, Symbol("scale_vec::4X"), 2, :ts, v"10.0", :arch, "sm_100a"),
    (:mxf4nvf4, Symbol("scale_vec::4X"), 1, :ss, v"11.0", :arch, "sm_110a"),

    # Family-compatible aliases: sm_100f + sm_110f.
    (:mxf8f6f4, :block32, 1, :ss, v"10.0", :family, "sm_100f"),
    (:mxf8f6f4, :block32, 2, :ts, v"11.0", :family, "sm_110f"),
    (:mxf4,     :block32, 2, :ts, v"10.0", :family, "sm_100f"),
    (:mxf4,     :block32, 1, :ss, v"11.0", :family, "sm_110f"),
    (:mxf4nvf4, :block32, 1, :ss, v"10.0", :family, "sm_100f"),
    (:mxf4nvf4, :block32, 2, :ts, v"11.0", :family, "sm_110f"),
    (:mxf4nvf4, :block16, 2, :ts, v"10.0", :family, "sm_100f"),
    (:mxf4nvf4, :block16, 1, :ss, v"11.0", :family, "sm_110f"),
)

function _tcgen05_mx_ptxas_name(kind, scale, cg, source, cap, features)
    safe_scale = replace(String(scale), ":" => "_")
    Symbol("_bw_tcgen05_mx_", kind, "_", safe_scale, "_cg", cg, "_",
           source, "_sm", cap.major, cap.minor, "_", features, "!")
end

# CUDACore 6.2.1's target database stops before sm_110, even though the
# bundled CUDA 13.3 ptxas accepts sm_110a/sm_110f.  For those two targets we
# therefore ask LLVM to emit the *identical wrapper body* at the matching
# sm_100 feature level, rewrite only the module's `.target` directive, and
# invoke ptxas directly.  The round-trip assertion makes this a deliberately
# narrow ptxas-retargeting oracle; it must not become a general PTX rewrite.
function _ptxas_retarget_sm100_to_sm110(ptx::String, source_target::String,
                                        target::String)
    @assert source_target in ("sm_100a", "sm_100f")
    @assert target == replace(source_target, "sm_100" => "sm_110")
    source_directive = ".target $source_target"
    target_directive = ".target $target"
    @assert length(findall(source_directive, ptx)) == 1

    retargeted = replace(ptx, source_directive => target_directive; count = 1)
    @assert replace(retargeted, target_directive => source_directive;
                    count = 1) == ptx

    mktempdir() do dir
        ptx_path = joinpath(dir, "retargeted.ptx")
        cubin_path = joinpath(dir, "retargeted.cubin")
        write(ptx_path, retargeted)
        cmd = `$(CUDACore.CUDA_Compiler.ptxas()) --gpu-name $target --output-file $cubin_path $ptx_path`
        log = IOBuffer()
        proc = run(pipeline(ignorestatus(cmd), stdout = log, stderr = log))
        success(proc) || error("ptxas rejected retargeted $target PTX:\n" *
                               String(take!(log)))
    end
    retargeted
end

for (kind, scale, cg, source, cap, features, _target) in
        _TCGEN05_MX_PTXAS_CASES
    mods = (:mma, Symbol("cta_group::", cg), Symbol("kind::", kind),
            :block_scale, scale)
    op = PTX.Operation{:tcgen05, mods}()
    fname = _tcgen05_mx_ptxas_name(kind, scale, cg, source, cap, features)
    if source === :ss
        @eval function $fname(d::UInt32, a_desc::UInt64, b_desc::UInt64,
                              idesc::UInt32, scale_a::UInt32, scale_b::UInt32)
            $op(address(d), a_desc, b_desc, idesc,
                address(scale_a), address(scale_b), false)
            return nothing
        end
    else
        @eval function $fname(d::UInt32, a_tmem::UInt32, b_desc::UInt64,
                              idesc::UInt32, scale_a::UInt32, scale_b::UInt32)
            $op(address(d), address(a_tmem), b_desc, idesc,
                address(scale_a), address(scale_b), false)
            return nothing
        end
    end
end

@testset "tcgen05 MX $kind.$scale cg$cg/$source at $target" for
        (kind, scale, cg, source, cap, features, target) in
            _TCGEN05_MX_PTXAS_CASES
    fname = _tcgen05_mx_ptxas_name(kind, scale, cg, source, cap, features)
    f = getfield(@__MODULE__, fname)
    a_T = source === :ss ? UInt64 : UInt32
    types = Tuple{UInt32, a_T, UInt64, UInt32, UInt32, UInt32}
    source_target = features === :arch ? "sm_100a" : "sm_100f"
    source_ptx = emit_ptx(f, types; cap = v"10.0", feature_set = features)
    if cap == v"10.0"
        @test ptxas_compiles(f, types; cap, feature_set = features)
        ptx = source_ptx
    else
        ptx = _ptxas_retarget_sm100_to_sm110(source_ptx, source_target, target)
        @test replace(ptx, ".target $target" => ".target $source_target";
                      count = 1) == source_ptx
    end
    @test occursin(".target $target", ptx)
    @test occursin("tcgen05.mma.cta_group::$cg.kind::$kind" *
                   ".block_scale.$scale", ptx)
    bracket_roles = source === :ss ?
        r"tcgen05\.mma[^;]+\[%r\d+\], %rd\d+, %rd\d+, %r\d+, \[%r\d+\], \[%r\d+\], %p\d+;" :
        r"tcgen05\.mma[^;]+\[%r\d+\], \[%r\d+\], %rd\d+, %r\d+, \[%r\d+\], \[%r\d+\], %p\d+;"
    @test occursin(bracket_roles, ptx)
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
    # Single-route asm renders the notation's multicast-first modifier
    # order (the retired intrinsic route reordered to
    # `.shared::cluster.multicast::cluster`); ptxas accepts both, pinned
    # by the ptxas_compiles legs here.
    @test ptxas_compiles(_bw_tcgen05_commit_mc1!, Tuple{UInt32};
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_bw_tcgen05_commit_mc1!, Tuple{UInt32};
                   cap = v"10.0", feature_set = :arch)
    @test occursin("tcgen05.commit.cta_group::1.mbarrier::arrive::one.multicast::cluster.shared::cluster.b64", ptx)

    @test ptxas_compiles(_bw_tcgen05_commit_mc2!, Tuple{UInt32};
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_bw_tcgen05_commit_mc2!, Tuple{UInt32};
                   cap = v"10.0", feature_set = :arch)
    @test occursin("tcgen05.commit.cta_group::2.mbarrier::arrive::one.multicast::cluster.shared::cluster.b64", ptx)
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
