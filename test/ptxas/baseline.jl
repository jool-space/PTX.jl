# Baseline-cap (sm_75..sm_89) wrapper validation — compile through ptxas
# at older capabilities than the dev box can launch via @cuda. Each test
# pins a cap at which a feature is supported, validating that the wrapper
# lowers to PTX that ptxas accepts at that target.
#
# CUDA 13's ptxas drops sm_50/sm_60/sm_70 (Maxwell/Pascal/Volta), so sm_75
# (Turing) is the floor here. `feature_set = :baseline` keeps the target
# spelled `sm_<X><Y>` (no `a` suffix); `:arch` is exercised in
# ptxas/hopper.jl and ptxas/blackwell.jl, since it's gated to sm_90+.


# --- explicit integer address roles at sm_75 -------------------------------
# PTX 9.3 §6.4.1 permits 32- or 64-bit integer/bit registers to carry byte
# addresses. Exercise every public scalar carrier through LLVM and ptxas so
# `Address` cannot grow an unvalidated constraint class.

function _baseline_address_s32!(addr::Int32)
    ptx"ld.shared.u32"(address(addr))
    return nothing
end

function _baseline_address_u32!(addr::UInt32)
    ptx"ld.shared.u32"(address(addr))
    return nothing
end

function _baseline_address_s64!(addr::Int64)
    ptx"ld.global.u32"(address(addr))
    return nothing
end

function _baseline_address_u64!(addr::UInt64)
    ptx"ld.global.u32"(address(addr))
    return nothing
end

function _baseline_address_cp_async!(dst::UInt32, src::UInt64)
    ptx"cp.async.ca.shared.global"(address(dst), address(src), Val(16))
    return nothing
end

@testset "explicit integer address carriers at sm_75" begin
    cases = (
        (_baseline_address_s32!, Tuple{Int32}, r"ld\.shared\.u32 %r\d+, \[%r\d+\]"),
        (_baseline_address_u32!, Tuple{UInt32}, r"ld\.shared\.u32 %r\d+, \[%r\d+\]"),
        (_baseline_address_s64!, Tuple{Int64}, r"ld\.global\.u32 %r\d+, \[%rd\d+\]"),
        (_baseline_address_u64!, Tuple{UInt64}, r"ld\.global\.u32 %r\d+, \[%rd\d+\]"),
    )
    for (f, tt, pattern) in cases
        @test ptxas_compiles(f, tt; cap = v"7.5")
        ptx = emit_ptx(f, tt; cap = v"7.5")
        @test occursin(pattern, ptx)
    end
end


@testset "classic cp.async integer address roles at sm_80" begin
    types = Tuple{UInt32, UInt64}
    @test ptxas_compiles(_baseline_address_cp_async!, types; cap = v"8.0")
    ptx = emit_ptx(_baseline_address_cp_async!, types; cap = v"8.0")
    @test occursin(
        r"cp\.async\.ca\.shared\.global \[%r\d+\], \[%rd\d+\], 16",
        ptx)
end

# --- f32 ALU at sm_75 (Turing baseline) -----------------------------------

function _baseline_alu_f32!(out, a::Float32, b::Float32, c::Float32)
    @inbounds out[1] = ptx"add.f32"(a, b)
    @inbounds out[2] = ptx"sub.f32"(a, b)
    @inbounds out[3] = ptx"mul.f32"(a, b)
    @inbounds out[4] = ptx"fma.rn.f32"(a, b, c)
    @inbounds out[5] = ptx"min.f32"(a, b)
    @inbounds out[6] = ptx"max.f32"(a, b)
    return nothing
end

@testset "f32 ALU at sm_75" begin
    types = Tuple{CuDeviceVector{Float32, 1}, Float32, Float32, Float32}
    @test ptxas_compiles(_baseline_alu_f32!, types; cap = v"7.5")
    ptx = emit_ptx(_baseline_alu_f32!, types; cap = v"7.5")
    @test occursin(".target sm_75", ptx)
    @test occursin("add.f32", ptx)
    @test occursin("fma.rn.f32", ptx)
end


# --- shfl.sync at sm_75 ---------------------------------------------------
#
# .sync variants of shfl/vote/bar were added in PTX ISA 6.0 / sm_70 but
# CUDA 13 ptxas requires sm_75+; this still validates the wrapper.

function _baseline_shfl_idx!(out)
    tid = ptx"mov.u32"(sreg"tid.x")
    val = ptx"shfl.sync.idx.b32"(UInt32(0xFFFFFFFF), tid, UInt32(5), UInt32(0x1F))
    @inbounds out[tid + 1] = val
    return nothing
end

@testset "shfl.sync.idx.b32 at sm_75" begin
    types = Tuple{CuDeviceVector{UInt32, 1}}
    @test ptxas_compiles(_baseline_shfl_idx!, types; cap = v"7.5")
    ptx = emit_ptx(_baseline_shfl_idx!, types; cap = v"7.5")
    @test occursin(".target sm_75", ptx)
    @test occursin("shfl.sync.idx.b32", ptx)
end


# --- cp.async at sm_80 (Ampere) --------------------------------------------
#
# cp.async was introduced in PTX 7.0 / sm_80.

function _baseline_cp_async!(dst::CuDeviceVector{UInt32, 1},
                              src::CuDeviceVector{UInt32, 1})
    smem = CuStaticSharedArray(UInt32, 32)
    tid = ptx"mov.u32"(sreg"tid.x")
    src_p = pointer(src) + Int(tid) * 4
    dst_p = pointer(smem) + Int(tid) * 4
    ptx"cp.async.ca.shared.global"(dst_p, src_p, Val(4))
    ptx"cp.async.commit_group"()
    ptx"cp.async.wait_all"()
    ptx"bar.sync"(Val(0))
    @inbounds dst[tid + 1] = smem[tid + 1]
    return nothing
end

@testset "cp.async.ca.shared.global at sm_80" begin
    types = Tuple{CuDeviceVector{UInt32, 1}, CuDeviceVector{UInt32, 1}}
    @test ptxas_compiles(_baseline_cp_async!, types; cap = v"8.0")
    ptx = emit_ptx(_baseline_cp_async!, types; cap = v"8.0")
    @test occursin(".target sm_80", ptx)
    @test occursin("cp.async.ca.shared.global", ptx)
    @test occursin("cp.async.commit_group", ptx)
    @test occursin("cp.async.wait_all", ptx)
end


# --- mma.sync bf16 at sm_80 ------------------------------------------------
#
# m16n8k16 bf16→f32 mma fragment. Same kernel shape used in gpu/exec.jl
# but here we validate the lowering at sm_80 (where bf16 mma was added).

function _baseline_mma_bf16!(out)
    a = (UInt32(0), UInt32(0), UInt32(0), UInt32(0))
    b = (UInt32(0), UInt32(0))
    c = (0f0, 0f0, 0f0, 0f0)
    d = ptx"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"(a, b, c)
    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        @inbounds out[1] = d[1]
    end
    return nothing
end

@testset "mma.sync m16n8k16 bf16 at sm_80" begin
    types = Tuple{CuDeviceVector{Float32, 1}}
    @test ptxas_compiles(_baseline_mma_bf16!, types; cap = v"8.0")
    ptx = emit_ptx(_baseline_mma_bf16!, types; cap = v"8.0")
    @test occursin(".target sm_80", ptx)
    @test occursin("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", ptx)
end


# --- FP8 → FP16 unpack cvt at sm_89 (PTX 8.1) -----------------------------
#
# Inverse of `cvt.rn.satfinite.{e4m3x2,e5m2x2}.f32` (the pack direction
# tested in test/gpu/cvt_fp8.jl). Real-world prevalence: Triton's
# matmul_tma_sm120a emits hundreds of `cvt.rn.f16x2.e5m2x2` per kernel as
# the FP8-load → FP16-mma bridge. Both f16x2-dest forms route through the
# chain default; no hand-written wrapper needed because the carrier is
# `.b16`. The bf16x2-dest unpack forms require sm_100a — see
# ptxas/blackwell.jl.

function _baseline_cvt_fp8_unpack!(out::CuDeviceVector{UInt32, 1}, x::UInt16)
    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        @inbounds out[1] = ptx"cvt.rn.f16x2.e4m3x2"(x)
        @inbounds out[2] = ptx"cvt.rn.f16x2.e5m2x2"(x)
    end
    return nothing
end

@testset "FP8 → FP16 unpack at sm_89" begin
    types = Tuple{CuDeviceVector{UInt32, 1}, UInt16}
    @test ptxas_compiles(_baseline_cvt_fp8_unpack!, types; cap = v"8.9")
    ptx = emit_ptx(_baseline_cvt_fp8_unpack!, types; cap = v"8.9")
    @test occursin("cvt.rn.f16x2.e4m3x2",  ptx)
    @test occursin("cvt.rn.f16x2.e5m2x2",  ptx)
end


# --- mbarrier pre-Hopper full cycle at sm_80 ------------------------------
#
# init / inval / arrive / arrive.noComplete / test_wait / test_wait.parity
# were added in PTX 7.0 (sm_80). Hopper-only forms (expect_tx, try_wait,
# try_wait.parity) live in ptxas/hopper.jl. One kernel covers all six pre-
# Hopper @generated bodies — previously only `which()` smoke-checked.

function _baseline_mbarrier_lifecycle!(out::CuDeviceVector{UInt64, 1},
                                        outp::CuDeviceVector{UInt32, 1})
    smem = CuStaticSharedArray(UInt64, 1)
    mbar = pointer(smem)
    tid  = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mbar, UInt32(2))
    end
    ptx"bar.sync"(Val(0))

    # Keep the compile probe semantically executable too: one elected thread
    # performs noComplete while the count is 2 (leaving 1), then the ordinary
    # arrival completes the phase. Both forms produce a u64 state token.
    if tid == UInt32(0)
        s2 = ptx"mbarrier.arrive.noComplete.shared.b64"(mbar, UInt32(1))
        s1 = ptx"mbarrier.arrive.shared.b64"(mbar)

        # Token form returns `Bool`; phase form takes a 0/1 parity bit and also
        # returns Bool. Stash both as 0/1 to keep the result alive.
        p1 = ptx"mbarrier.test_wait.shared.b64"(mbar, s1)
        p2 = ptx"mbarrier.test_wait.parity.shared.b64"(mbar, UInt32(0))

        @inbounds out[1]  = s1
        @inbounds out[2]  = s2
        @inbounds outp[1] = p1 ? UInt32(1) : UInt32(0)
        @inbounds outp[2] = p2 ? UInt32(1) : UInt32(0)
        ptx"mbarrier.inval.shared.b64"(mbar)
    end
    return nothing
end

@testset "mbarrier pre-Hopper lifecycle at sm_80" begin
    types = Tuple{CuDeviceVector{UInt64, 1}, CuDeviceVector{UInt32, 1}}
    @test ptxas_compiles(_baseline_mbarrier_lifecycle!, types; cap = v"8.0")
    ptx = emit_ptx(_baseline_mbarrier_lifecycle!, types; cap = v"8.0")
    @test occursin("mbarrier.init.shared.b64",                ptx)
    @test occursin("mbarrier.arrive.shared.b64",              ptx)
    @test occursin("mbarrier.arrive.noComplete.shared.b64",   ptx)
    @test occursin("mbarrier.test_wait.shared.b64",           ptx)
    @test occursin("mbarrier.test_wait.parity.shared.b64",    ptx)
    @test occursin("mbarrier.inval.shared.b64",               ptx)
end


# --- shfl all four modes (incl. pred-output) at sm_75 ---------------------
#
# `idx` is covered above; this adds up/down/bfly + pred-output for every
# mode. Pred form returns `(UInt32, Bool)` via the `\$0|\$1` operand.

function _baseline_shfl_all_modes!(out::CuDeviceVector{UInt32, 1})
    tid  = ptx"mov.u32"(sreg"tid.x")
    mask = UInt32(0xFFFFFFFF)
    v_up   = ptx"shfl.sync.up.b32"(  tid, UInt32(1), UInt32(0x00), mask)
    v_dn   = ptx"shfl.sync.down.b32"(tid, UInt32(1), UInt32(0x1F), mask)
    v_bfly = ptx"shfl.sync.bfly.b32"(tid, UInt32(1), UInt32(0x1F), mask)
    v_idx  = ptx"shfl.sync.idx.b32"( tid, UInt32(0), UInt32(0x1F), mask)
    @inbounds out[tid + 1] = v_up ⊻ v_dn ⊻ v_bfly ⊻ v_idx

    # Pred form — destination + in-range predicate. Stash the pred bit so
    # LLVM doesn't DCE the call.
    (vp, p) = ptx"shfl.sync.bfly.b32.pred"(tid, UInt32(1), UInt32(0x1F), mask)
    if tid == UInt32(0)
        @inbounds out[33] = p ? vp : UInt32(0xDEAD)
    end
    return nothing
end

@testset "shfl.sync all modes (+ pred form) at sm_75" begin
    types = Tuple{CuDeviceVector{UInt32, 1}}
    @test ptxas_compiles(_baseline_shfl_all_modes!, types; cap = v"7.5")
    ptx = emit_ptx(_baseline_shfl_all_modes!, types; cap = v"7.5")
    @test occursin("shfl.sync.up.b32",   ptx)
    @test occursin("shfl.sync.down.b32", ptx)
    @test occursin("shfl.sync.bfly.b32", ptx)
    @test occursin("shfl.sync.idx.b32",  ptx)
    # Pred form uses the `d|p, ...` pipe-operand syntax.
    @test occursin(r"shfl\.sync\.bfly\.b32\s+%r\d+\|%p\d+,", ptx)
end


# --- vec ld/st {v2,v4} × {f32,b32,b16} at sm_75 ---------------------------
#
# `ld.global.v{2,4}.<dt>` / `st.global.v{2,4}.<dt>` exercise the brace-
# operand renderers in src/wrappers/vec_ldst.jl. Round-trip through globals
# so the @generated body is fully expanded.

function _baseline_vec_ldst!(out_f::CuDeviceVector{Float32, 1},
                              out_u32::CuDeviceVector{UInt32, 1},
                              out_u16::CuDeviceVector{UInt16, 1})
    p_f   = pointer(out_f)
    p_u32 = pointer(out_u32)
    p_u16 = pointer(out_u16)

    # Tier-1 loads/stores are real (no `~{memory}` barrier), so a same-address
    # round-trip would be dead-store-eliminated to nothing. Store back at a
    # distinct offset to keep the access live, and touch each b16 lane so the
    # backend keeps the narrow `v{2,4}.b16` form (an opaque b16 passthrough
    # legally coalesces into a wider `.b32` access).
    ptx"st.global.v2.f32"(p_f + 32, ptx"ld.global.v2.f32"(p_f))
    ptx"st.global.v4.f32"(p_f + 64, ptx"ld.global.v4.f32"(p_f + 16))
    ptx"st.global.v2.b32"(p_u32 + 32, ptx"ld.global.v2.b32"(p_u32))
    ptx"st.global.v4.b32"(p_u32 + 64, ptx"ld.global.v4.b32"(p_u32 + 16))

    u16_2 = ptx"ld.global.v2.b16"(p_u16)
    ptx"st.global.v2.b16"(p_u16 + 16,
        (ptx"add.s16"(u16_2[1], UInt16(1)), ptx"add.s16"(u16_2[2], UInt16(1))))
    u16_4 = ptx"ld.global.v4.b16"(p_u16 + 8)
    ptx"st.global.v4.b16"(p_u16 + 32,
        (ptx"add.s16"(u16_4[1], UInt16(1)), ptx"add.s16"(u16_4[2], UInt16(1)),
         ptx"add.s16"(u16_4[3], UInt16(1)), ptx"add.s16"(u16_4[4], UInt16(1))))
    return nothing
end

@testset "vec ld/st v2/v4 × {f32,b32,b16} at sm_75" begin
    types = Tuple{CuDeviceVector{Float32, 1},
                  CuDeviceVector{UInt32, 1},
                  CuDeviceVector{UInt16, 1}}
    @test ptxas_compiles(_baseline_vec_ldst!, types; cap = v"7.5")
    ptx = emit_ptx(_baseline_vec_ldst!, types; cap = v"7.5")
    # NVPTX canonicalizes float vector ld/st to the `.b32` bit spelling
    # (registers are typeless), so the f32 forms appear as `.b32`.
    @test occursin("ld.global.v2.b32", ptx)
    @test occursin("ld.global.v4.b32", ptx)
    @test occursin("st.global.v2.b32", ptx)
    @test occursin("st.global.v4.b32", ptx)
    @test occursin("ld.global.v2.b16", ptx)
    @test occursin("ld.global.v4.b16", ptx)
    @test occursin("st.global.v2.b16", ptx)
    @test occursin("st.global.v4.b16", ptx)
end


# --- cp.async.ca size-baking N=8 / N=16 at sm_80 --------------------------
#
# The N=4 form is exercised by the `cp.async.ca.shared.global at sm_80`
# testset above. This locks N=8 and N=16 — the @generated body has separate
# size-baked asm strings for each of {4, 8, 16}.

function _baseline_cp_async_n8!(dst::CuDeviceVector{UInt64, 1},
                                 src::CuDeviceVector{UInt64, 1})
    smem = CuStaticSharedArray(UInt64, 32)
    tid  = ptx"mov.u32"(sreg"tid.x")
    src_p = pointer(src) + Int(tid) * 8
    dst_p = pointer(smem) + Int(tid) * 8
    ptx"cp.async.ca.shared.global"(dst_p, src_p, Val(8))
    ptx"cp.async.commit_group"()
    ptx"cp.async.wait_all"()
    ptx"bar.sync"(Val(0))
    @inbounds dst[tid + 1] = smem[tid + 1]
    return nothing
end

function _baseline_cp_async_n16!(dst::CuDeviceVector{UInt32, 1},
                                  src::CuDeviceVector{UInt32, 1})
    smem = CuStaticSharedArray(UInt32, 128)
    tid  = ptx"mov.u32"(sreg"tid.x")
    src_p = pointer(src) + Int(tid) * 16
    dst_p = pointer(smem) + Int(tid) * 16
    ptx"cp.async.ca.shared.global"(dst_p, src_p, Val(16))
    ptx"cp.async.cg.shared.global"(dst_p, src_p, Val(16))
    ptx"cp.async.commit_group"()
    ptx"cp.async.wait_all"()
    ptx"bar.sync"(Val(0))
    @inbounds dst[tid + 1] = smem[tid * 4 + 1]
    return nothing
end

@testset "cp.async.{ca,cg}.shared.global N=8/N=16 at sm_80" begin
    types8 = Tuple{CuDeviceVector{UInt64, 1}, CuDeviceVector{UInt64, 1}}
    @test ptxas_compiles(_baseline_cp_async_n8!, types8; cap = v"8.0")
    ptx8 = emit_ptx(_baseline_cp_async_n8!, types8; cap = v"8.0")
    @test occursin(r"cp\.async\.ca\.shared\.global \[%r\d+\], \[%rd?\d+\], 8;",
                   ptx8)

    types16 = Tuple{CuDeviceVector{UInt32, 1}, CuDeviceVector{UInt32, 1}}
    @test ptxas_compiles(_baseline_cp_async_n16!, types16; cap = v"8.0")
    ptx16 = emit_ptx(_baseline_cp_async_n16!, types16; cap = v"8.0")
    @test occursin(r"cp\.async\.ca\.shared\.global \[%r\d+\], \[%rd?\d+\], 16;",
                   ptx16)
    # cg is L2-only; size 16 is the only legal form.
    @test occursin(r"cp\.async\.cg\.shared\.global \[%r\d+\], \[%rd?\d+\], 16;",
                   ptx16)
end


# --- generic memory fences at sm_75 (tier-1 core IR) ------------------------
#
# fence.{sc,acq_rel}.{cta,gpu,sys} through the full pipeline including
# ptxas. Cluster scope is sm_90+ (ISel-enforced) and rides in
# ptxas/hopper.jl. Stores between the fences keep each one's position
# observable (a core-IR fence pins memory ops, not other fences).

function _baseline_fences_generic!(out)
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

@testset "generic memory fences at sm_75" begin
    types = Tuple{CuDeviceVector{UInt32, 1}}
    @test ptxas_compiles(_baseline_fences_generic!, types; cap = v"7.5")
    ptx = emit_ptx(_baseline_fences_generic!, types; cap = v"7.5")
    for form in ("fence.sc.cta", "fence.sc.gpu", "fence.sc.sys",
                 "fence.acq_rel.cta", "fence.acq_rel.gpu",
                 "fence.acq_rel.sys")
        @test occursin(form * ";", ptx)
    end
end


# `feature_set = :arch` is gated to sm_90+ in CUDACore (sm_89a doesn't
# exist as a target). The :arch path is exercised in ptxas/hopper.jl
# (cap=9.0) and ptxas/blackwell.jl (cap=10.0).
