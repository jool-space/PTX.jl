# Baseline-cap (sm_75..sm_89) wrapper validation — compile through ptxas
# at older capabilities than the dev box can launch via @cuda. Each test
# pins a cap at which a feature is supported, validating that the wrapper
# lowers to PTX that ptxas accepts at that target.
#
# CUDA 13's ptxas drops sm_50/sm_60/sm_70 (Maxwell/Pascal/Volta), so sm_75
# (Turing) is the floor here. `feature_set = :baseline` keeps the target
# spelled `sm_<X><Y>` (no `a` suffix); `:arch` is exercised in
# ptxas/hopper.jl and ptxas/blackwell.jl, since it's gated to sm_90+.


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


# `feature_set = :arch` is gated to sm_90+ in CUDACore (sm_89a doesn't
# exist as a target). The :arch path is exercised in ptxas/hopper.jl
# (cap=9.0) and ptxas/blackwell.jl (cap=10.0).
