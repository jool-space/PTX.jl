# REQUIRES CC 8.9
using CUDATools

# These tests close the gap I flagged in v0.0: each wrapper's @asmcall string
# is compared against the wrapper's host-side `ir_*()` golden by compiling the
# kernel for the active SM and grepping the emitted PTX. If a wrapper's asm
# string drifts from its ir_* helper, the goldens still match each other but
# the *actual* PTX no longer contains the expected instruction — caught here.

ptx_for(kernel, argtypes::Type) = sprint(io -> CUDATools.code_ptx(io, kernel, argtypes))


# --- add.f32 ----------------------------------------------------------------

function _kernel_add_f32(out)
    @inbounds out[1] = ptx"add.f32"(1.5f0, 2.5f0)
    return nothing
end

@testset "add.f32 → PTX text" begin
    asm = ptx_for(_kernel_add_f32, Tuple{CuDeviceArray{Float32,1,CUDACore.AS.Global}})
    @test occursin("add.f32", asm)
end

# --- f32 ALU expansion ------------------------------------------------------

function _kernel_alu_f32(out, a, b, c)
    @inbounds out[1] = ptx"sub.f32"(a, b)
    @inbounds out[2] = ptx"mul.f32"(a, b)
    @inbounds out[3] = ptx"fma.rn.f32"(a, b, c)
    @inbounds out[4] = ptx"min.f32"(a, b)
    @inbounds out[5] = ptx"max.f32"(a, b)
    return nothing
end

@testset "f32 ALU → PTX text" begin
    asm = ptx_for(_kernel_alu_f32,
                  Tuple{CuDeviceArray{Float32,1,CUDACore.AS.Global},
                        Float32, Float32, Float32})
    @test occursin("sub.f32",     asm)
    @test occursin("mul.f32",     asm)
    @test occursin("fma.rn.f32",  asm)
    @test occursin("min.f32",     asm)
    @test occursin("max.f32",     asm)
end

# --- mov.u32 (special register) --------------------------------------------

function _kernel_mov_tid_x(out)
    tid = ptx"mov.u32"(sreg"tid.x")
    @inbounds out[tid + 1] = tid
    return nothing
end

@testset "mov.u32 sreg → PTX text" begin
    asm = ptx_for(_kernel_mov_tid_x, Tuple{CuDeviceArray{UInt32,1,CUDACore.AS.Global}})
    @test occursin("mov.u32", asm)
    @test occursin("%tid.x", asm)
end

# --- bar.sync ---------------------------------------------------------------

function _kernel_bar_sync(out)
    ptx"bar.sync"(Val(0))
    @inbounds out[1] = 1f0
    return nothing
end

@testset "bar.sync → PTX text" begin
    asm = ptx_for(_kernel_bar_sync, Tuple{CuDeviceArray{Float32,1,CUDACore.AS.Global}})
    @test occursin("bar.sync", asm)
end

# --- atom.add.gpu.u32 -------------------------------------------------------

function _kernel_atom_add_gpu(counter)
    ptx"atom.add.gpu.u32"(pointer(counter), UInt32(1))
    return nothing
end

@testset "atom.add.gpu.u32 → PTX text" begin
    asm = ptx_for(_kernel_atom_add_gpu, Tuple{CuDeviceArray{UInt32,1,CUDACore.AS.Global}})
    @test occursin("atom.add.gpu.u32", asm)
end

# --- cp.async sync ops ------------------------------------------------------

function _kernel_cp_async_sync(out)
    ptx"cp.async.commit_group"()
    ptx"cp.async.wait_group"(Val(0))
    ptx"cp.async.wait_all"()
    @inbounds out[1] = 1f0
    return nothing
end

@testset "cp.async sync ops → PTX text" begin
    asm = ptx_for(_kernel_cp_async_sync, Tuple{CuDeviceArray{Float32,1,CUDACore.AS.Global}})
    @test occursin("cp.async.commit_group", asm)
    @test occursin("cp.async.wait_group", asm)
    @test occursin("cp.async.wait_all", asm)
end

# --- cp.async.ca.shared.global (data movement) ------------------------------

function _kernel_cp_async_ca(dst, src)
    smem = CuStaticSharedArray(Float32, 4)
    ptx"cp.async.ca.shared.global"(pointer(smem), pointer(src), Val(16))
    ptx"cp.async.commit_group"()
    ptx"cp.async.wait_group"(Val(0))
    @inbounds dst[1] = smem[1]
    return nothing
end

@testset "cp.async.ca.shared.global → PTX text" begin
    asm = ptx_for(_kernel_cp_async_ca,
                  Tuple{CuDeviceArray{Float32,1,CUDACore.AS.Global},
                        CuDeviceArray{Float32,1,CUDACore.AS.Global}})
    @test occursin("cp.async.ca.shared.global", asm)
end

# --- cp.async.cg.shared.global (L2-only data movement) ---------------------

function _kernel_cp_async_cg(dst, src)
    smem = CuStaticSharedArray(Float32, 4)
    ptx"cp.async.cg.shared.global"(pointer(smem), pointer(src), Val(16))
    ptx"cp.async.commit_group"()
    ptx"cp.async.wait_group"(Val(0))
    @inbounds dst[1] = smem[1]
    return nothing
end

@testset "cp.async.cg.shared.global → PTX text" begin
    asm = ptx_for(_kernel_cp_async_cg,
                  Tuple{CuDeviceArray{Float32,1,CUDACore.AS.Global},
                        CuDeviceArray{Float32,1,CUDACore.AS.Global}})
    @test occursin("cp.async.cg.shared.global", asm)
end

# --- ldmatrix.sync.aligned (warp-cooperative shared→register) --------------

function _kernel_ldmatrix_x1(out)
    smem = CuStaticSharedArray(UInt16, 64)
    addr = pointer(smem)
    r = ptx"ldmatrix.sync.aligned.m8n8.x1.shared.b16"(addr)
    @inbounds out[1] = r
    return nothing
end

@testset "ldmatrix.x1 → PTX text" begin
    asm = ptx_for(_kernel_ldmatrix_x1, Tuple{CuDeviceArray{UInt32,1,CUDACore.AS.Global}})
    @test occursin("ldmatrix.sync.aligned.m8n8.x1.shared.b16", asm)
end

function _kernel_ldmatrix_x2(out)
    smem = CuStaticSharedArray(UInt16, 128)
    addr = pointer(smem)
    rs = ptx"ldmatrix.sync.aligned.m8n8.x2.shared.b16"(addr)
    @inbounds out[1] = rs[1]
    @inbounds out[2] = rs[2]
    return nothing
end

@testset "ldmatrix.x2 → PTX text" begin
    asm = ptx_for(_kernel_ldmatrix_x2, Tuple{CuDeviceArray{UInt32,1,CUDACore.AS.Global}})
    @test occursin("ldmatrix.sync.aligned.m8n8.x2.shared.b16", asm)
end

function _kernel_ldmatrix_x4(out)
    smem = CuStaticSharedArray(UInt16, 256)
    addr = pointer(smem)
    rs = ptx"ldmatrix.sync.aligned.m8n8.x4.shared.b16"(addr)
    @inbounds out[1] = rs[1]
    @inbounds out[2] = rs[2]
    @inbounds out[3] = rs[3]
    @inbounds out[4] = rs[4]
    return nothing
end

@testset "ldmatrix.x4 → PTX text" begin
    asm = ptx_for(_kernel_ldmatrix_x4, Tuple{CuDeviceArray{UInt32,1,CUDACore.AS.Global}})
    @test occursin("ldmatrix.sync.aligned.m8n8.x4.shared.b16", asm)
end

function _kernel_ldmatrix_x2_trans(out)
    smem = CuStaticSharedArray(UInt16, 128)
    addr = pointer(smem)
    rs = ptx"ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16"(addr)
    @inbounds out[1] = rs[1]
    @inbounds out[2] = rs[2]
    return nothing
end

@testset "ldmatrix.x2.trans → PTX text" begin
    asm = ptx_for(_kernel_ldmatrix_x2_trans, Tuple{CuDeviceArray{UInt32,1,CUDACore.AS.Global}})
    @test occursin("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16", asm)
end

# Additional ldmatrix variants generated via the family registrar.

function _kernel_ldmatrix_x1_trans(out)
    smem = CuStaticSharedArray(UInt16, 64)
    addr = pointer(smem)
    r = ptx"ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16"(addr)
    @inbounds out[1] = r
    return nothing
end

@testset "ldmatrix.x1.trans → PTX text" begin
    asm = ptx_for(_kernel_ldmatrix_x1_trans, Tuple{CuDeviceArray{UInt32,1,CUDACore.AS.Global}})
    @test occursin("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16", asm)
end

function _kernel_ldmatrix_x4_trans(out)
    smem = CuStaticSharedArray(UInt16, 256)
    addr = pointer(smem)
    rs = ptx"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"(addr)
    @inbounds out[1] = rs[1]
    @inbounds out[2] = rs[2]
    @inbounds out[3] = rs[3]
    @inbounds out[4] = rs[4]
    return nothing
end

@testset "ldmatrix.x4.trans → PTX text" begin
    asm = ptx_for(_kernel_ldmatrix_x4_trans, Tuple{CuDeviceArray{UInt32,1,CUDACore.AS.Global}})
    @test occursin("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16", asm)
end

function _kernel_ldmatrix_x2_shared_cta(out)
    smem = CuStaticSharedArray(UInt16, 128)
    addr = pointer(smem)
    rs = ptx"ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16"(addr)
    @inbounds out[1] = rs[1]
    @inbounds out[2] = rs[2]
    return nothing
end

@testset "ldmatrix.x2.shared::cta → PTX text" begin
    asm = ptx_for(_kernel_ldmatrix_x2_shared_cta,
                  Tuple{CuDeviceArray{UInt32,1,CUDACore.AS.Global}})
    @test occursin("ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16", asm)
end

# --- mma.sync.aligned.m16n8k16 bf16 ----------------------------------------

function _kernel_mma_bf16(out)
    a = (UInt32(0), UInt32(0), UInt32(0), UInt32(0))
    b = (UInt32(0), UInt32(0))
    c = (0f0, 0f0, 0f0, 0f0)
    d = ptx"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"(a, b, c)
    @inbounds out[1] = d[1]
    @inbounds out[2] = d[2]
    @inbounds out[3] = d[3]
    @inbounds out[4] = d[4]
    return nothing
end

@testset "mma.sync.aligned.m16n8k16 bf16 → PTX text" begin
    asm = ptx_for(_kernel_mma_bf16, Tuple{CuDeviceArray{Float32,1,CUDACore.AS.Global}})
    @test occursin("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", asm)
end

# Additional shape/dtype combos — verify the generator produces correct
# wrappers for several variants without per-combo source code.

function _kernel_mma_bf16_k8(out)
    a = (UInt32(0), UInt32(0))           # m16n8k8 needs 2 a-frags
    b = (UInt32(0),)                     # and 1 b-frag
    c = (0f0, 0f0, 0f0, 0f0)
    d = ptx"mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32"(a, b, c)
    @inbounds out[1] = d[1]
    return nothing
end

@testset "mma.sync.aligned.m16n8k8 bf16 → PTX text" begin
    asm = ptx_for(_kernel_mma_bf16_k8, Tuple{CuDeviceArray{Float32,1,CUDACore.AS.Global}})
    @test occursin("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32", asm)
end

function _kernel_mma_f16_k16(out)
    a = (UInt32(0), UInt32(0), UInt32(0), UInt32(0))
    b = (UInt32(0), UInt32(0))
    c = (0f0, 0f0, 0f0, 0f0)
    d = ptx"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"(a, b, c)
    @inbounds out[1] = d[1]
    return nothing
end

@testset "mma.sync.aligned.m16n8k16 f16/f32 → PTX text" begin
    asm = ptx_for(_kernel_mma_f16_k16, Tuple{CuDeviceArray{Float32,1,CUDACore.AS.Global}})
    @test occursin("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", asm)
end

function _kernel_mma_f16_acc(out)
    # f16 accumulator → C/D are UInt32 (packed 2×f16), not Float32.
    a = (UInt32(0), UInt32(0), UInt32(0), UInt32(0))
    b = (UInt32(0), UInt32(0))
    c = (UInt32(0), UInt32(0))
    d = ptx"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"(a, b, c)
    @inbounds out[1] = d[1]
    return nothing
end

@testset "mma.sync.aligned.m16n8k16 all-f16 → PTX text" begin
    asm = ptx_for(_kernel_mma_f16_acc, Tuple{CuDeviceArray{UInt32,1,CUDACore.AS.Global}})
    @test occursin("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16", asm)
end

function _kernel_mma_e4m3(out)
    # FP8 (Ada sm_89+): m16n8k32, e4m3 inputs, f32 accumulator.
    a = (UInt32(0), UInt32(0), UInt32(0), UInt32(0))
    b = (UInt32(0), UInt32(0))
    c = (0f0, 0f0, 0f0, 0f0)
    d = ptx"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"(a, b, c)
    @inbounds out[1] = d[1]
    return nothing
end

@testset "mma.sync.aligned.m16n8k32 e4m3 → PTX text" begin
    asm = ptx_for(_kernel_mma_e4m3, Tuple{CuDeviceArray{Float32,1,CUDACore.AS.Global}})
    @test occursin("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32", asm)
end

# --- cvt family ------------------------------------------------------------

function _kernel_cvt_f16_f32(out)
    @inbounds out[1] = ptx"cvt.rn.f16.f32"(1.5f0)
    return nothing
end

@testset "cvt.rn.f16.f32 → PTX text" begin
    asm = ptx_for(_kernel_cvt_f16_f32, Tuple{CuDeviceArray{Float16,1,CUDACore.AS.Global}})
    @test occursin("cvt.rn.f16.f32", asm)
end

function _kernel_cvt_e4m3x2_f32(out)
    @inbounds out[1] = ptx"cvt.rn.satfinite.e4m3x2.f32"(1.0f0, 2.0f0)
    return nothing
end

@testset "cvt.rn.satfinite.e4m3x2.f32 (packed) → PTX text" begin
    asm = ptx_for(_kernel_cvt_e4m3x2_f32, Tuple{CuDeviceArray{UInt16,1,CUDACore.AS.Global}})
    @test occursin("cvt.rn.satfinite.e4m3x2.f32", asm)
end

# --- shfl.sync family ------------------------------------------------------

function _kernel_shfl_idx(out)
    tid = ptx"mov.u32"(sreg"tid.x")
    src = tid + UInt32(100)
    @inbounds out[Int(tid) + 1] =
        ptx"shfl.sync.idx.b32"(src, UInt32(5), UInt32(0x1F), UInt32(0xFFFFFFFF))
    return nothing
end

@testset "shfl.sync.idx.b32 → PTX text" begin
    asm = ptx_for(_kernel_shfl_idx, Tuple{CuDeviceArray{UInt32,1,CUDACore.AS.Global}})
    @test occursin("shfl.sync.idx.b32", asm)
end

function _kernel_shfl_bfly(out)
    tid = ptx"mov.u32"(sreg"tid.x")
    @inbounds out[Int(tid) + 1] =
        ptx"shfl.sync.bfly.b32"(tid, UInt32(0x10), UInt32(0x1F), UInt32(0xFFFFFFFF))
    return nothing
end

@testset "shfl.sync.bfly.b32 → PTX text" begin
    asm = ptx_for(_kernel_shfl_bfly, Tuple{CuDeviceArray{UInt32,1,CUDACore.AS.Global}})
    @test occursin("shfl.sync.bfly.b32", asm)
end

# --- fence + bar.warp.sync (handled by L0 chain default) -------------------

function _kernel_fences(out)
    ptx"fence.sc.gpu"()
    ptx"fence.acq_rel.cta"()
    ptx"fence.sc.sys"()
    @inbounds out[1] = 1f0
    return nothing
end

@testset "fence variants → PTX text" begin
    asm = ptx_for(_kernel_fences, Tuple{CuDeviceArray{Float32,1,CUDACore.AS.Global}})
    @test occursin("fence.sc.gpu",    asm)
    @test occursin("fence.acq_rel.cta", asm)
    @test occursin("fence.sc.sys",    asm)
end

function _kernel_bar_warp_sync(out)
    ptx"bar.warp.sync"(UInt32(0xFFFFFFFF))
    @inbounds out[1] = 1f0
    return nothing
end

@testset "bar.warp.sync → PTX text" begin
    asm = ptx_for(_kernel_bar_warp_sync, Tuple{CuDeviceArray{Float32,1,CUDACore.AS.Global}})
    @test occursin("bar.warp.sync", asm)
end

# --- ldmatrix.sync.aligned (warp-cooperative shared→register) --------------

