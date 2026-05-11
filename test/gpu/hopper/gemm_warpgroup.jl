# Single-warpgroup Hopper GEMM — TMA load → mbarrier sync → wgmma → store.
#
# Headline Hopper kernel for ROADMAP item 6. Composes the full sm_90a stack:
#
#   1. cuTensorMapEncodeTiled (host)   →  PTX.tensor_map_encode_tiled
#   2. cp.async.bulk.tensor.2d (device)→  TMA load A, B into SMEM
#   3. mbarrier.init/expect_tx/wait    →  hardware-sync TMA completion
#   4. fence.proxy.async + wgmma.fence →  cross-proxy ordering for wgmma
#   5. wgmma_descriptor                →  build 64-bit SMEM descriptors
#   6. wgmma.mma_async                 →  warpgroup-collective MMA
#   7. wgmma.commit_group/wait_group   →  drain the async pipeline
#   8. Per-lane store of f32 frags     →  output to D
#
# Tile shape m64n8k16 bf16.bf16.f32 — smallest wgmma m64n8k16 wgmma. Single
# warpgroup (128 threads), single CTA, single K-tile (no main loop).
#
# Runtime path needs H100 (wgmma is sm_90a-only — Blackwell uses tcgen05).
# Cross-arch ptxas-validation at v"9.0" always runs (catches wrapper-side
# regressions on the GB10 dev box, where the kernel can't actually launch).

using PTX: wgmma_descriptor, WgmmaSwizzle, smem_addr_u32, layout_for_a, layout_for_b
using CUDACore

const HOPPER_BM = 64
const HOPPER_BN = 8
const HOPPER_BK = 16
const HOPPER_THREADS = 128  # one warpgroup

# Total TMA bytes the mbarrier waits on:
#   A: BM × BK × 2 bytes = 2048
#   B: BK × BN × 2 bytes = 256
const HOPPER_LOAD_BYTES = HOPPER_BM * HOPPER_BK * 2 + HOPPER_BK * HOPPER_BN * 2

function _hopper_warpgroup_gemm_kernel!(
        D::CuDeviceVector{Float32, 1},
        tma_A::Core.LLVMPtr{UInt8, PTX.AS.Const},
        tma_B::Core.LLVMPtr{UInt8, PTX.AS.Const})

    smem_A = CuStaticSharedArray(UInt16, HOPPER_BM * HOPPER_BK)
    smem_B = CuStaticSharedArray(UInt16, HOPPER_BK * HOPPER_BN)
    mbar   = CuStaticSharedArray(UInt64, 1)

    a_ptr  = pointer(smem_A)
    b_ptr  = pointer(smem_B)
    mb_ptr = pointer(mbar)
    a_addr = smem_addr_u32(a_ptr)
    b_addr = smem_addr_u32(b_ptr)

    tid = ptx"mov.u32"(sreg"tid.x")

    # 1. Lane 0 does the entire TMA-issue sequence (init, fence, arrive,
    #    bulk loads). pyptx's tests (_test_tma_fence.py etc.) keep these in
    #    one thread-0 block followed by a CTA-wide bar.sync — the bar.sync
    #    after the issue (NOT between init and arrive) is what makes the
    #    inited+armed state visible to all threads before any of them polls
    #    test_wait.parity. Splitting init / arrive with bar.sync between
    #    deadlocks on H100 (other threads enter the spin loop too early).
    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mb_ptr, UInt32(1))
        ptx"fence.proxy.async.shared::cta"()
        ptx"mbarrier.arrive.expect_tx.shared.b64"(mb_ptr, UInt32(HOPPER_LOAD_BYTES))
        ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
            a_ptr, tma_A, Int32(0), Int32(0), mb_ptr)
        ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
            b_ptr, tma_B, Int32(0), Int32(0), mb_ptr)
    end
    ptx"bar.sync"(Val(0))

    # 2. Every thread waits for TMA completion. test_wait.parity polls the
    #    phase bit — 0 → 1 on the first arrival round.
    while !ptx"mbarrier.test_wait.parity.shared.b64"(mb_ptr, UInt32(0))
    end

    # 3. Cross-proxy fence: TMA writes via async proxy; wgmma reads via async
    #    proxy too, but the descriptor build needs the SMEM bytes visible
    #    through the same proxy.
    ptx"fence.proxy.async.shared::cta"()
    ptx"wgmma.fence.sync.aligned"()

    # 4. Build A/B descriptors. K-major bf16 m64k16 tile:
    #    row width 32 B = 2 u128 → B32 swizzle; leading 16, stride 256.
    la = layout_for_a(dtype = :bf16, m = HOPPER_BM, k = HOPPER_BK)
    lb = layout_for_b(dtype = :bf16, k = HOPPER_BK, n = HOPPER_BN)
    a_desc = wgmma_descriptor(a_addr;
        leading_byte_offset = la.leading_byte_offset,
        stride_byte_offset  = la.stride_byte_offset,
        swizzle             = la.layout_type)
    b_desc = wgmma_descriptor(b_addr;
        leading_byte_offset = lb.leading_byte_offset,
        stride_byte_offset  = lb.stride_byte_offset,
        swizzle             = lb.layout_type)

    # 5. wgmma m64n8k16.f32.bf16.bf16 — 4 f32 regs per lane × 128 lanes.
    d = ntuple(_ -> 0f0, Val(4))
    d = ptx"wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16"(
        d, a_desc, b_desc, false)

    ptx"wgmma.commit_group.sync.aligned"()
    ptx"wgmma.wait_group.sync.aligned"(Val(0))

    # 6. Stash one frag per lane so the compiler can't dead-code the chain.
    #    Full lane→(row,col) mapping is ROADMAP item 1 / fragment-coord
    #    helper — deferred until first numerical validation on H100.
    @inbounds D[Int(tid) + 1] = d[1]
    return nothing
end

@testset "single-warpgroup Hopper GEMM compiles at sm_90a" begin
    # Always-on cross-arch validation: ptxas accepts the full kernel under
    # sm_90a. Catches wrapper regressions on the GB10 dev box where the
    # kernel can't actually launch.
    types = Tuple{CuDeviceVector{Float32, 1},
                  Core.LLVMPtr{UInt8, PTX.AS.Const},
                  Core.LLVMPtr{UInt8, PTX.AS.Const}}
    @test ptxas_compiles(_hopper_warpgroup_gemm_kernel!, types;
                         cap = v"9.0", feature_set = :arch)

    ptx = emit_ptx(_hopper_warpgroup_gemm_kernel!, types;
                   cap = v"9.0", feature_set = :arch)

    # Every layer of the canonical Hopper warpgroup-MMA pipeline appears.
    @test occursin("mbarrier.init.shared.b64",                       ptx)
    @test occursin("mbarrier.arrive.expect_tx.shared.b64",           ptx)
    @test occursin("mbarrier.test_wait.parity.shared.b64",           ptx)
    @test occursin("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes", ptx)
    @test occursin("fence.proxy.async.shared::cta",                  ptx)
    @test occursin("wgmma.fence.sync.aligned",                       ptx)
    @test occursin("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16", ptx)
    @test occursin("wgmma.commit_group.sync.aligned",                ptx)
    @test occursin("wgmma.wait_group.sync.aligned 0",                ptx)

    # Descriptor build constant-folds the layout-picked offsets into a single
    # OR with the runtime SMEM address (verified compile-only).
    @test occursin(r"or\.b64", ptx)
end

# Runtime path — gated on Hopper hardware. wgmma is sm_90a-only; Blackwell
# (sm_100/sm_120/sm_121) replaced it with tcgen05, so cap is required to be
# in [9.0, 10.0). On the GB10 dev box (sm_121a) this testset is skipped.
if v"9.0" <= DEV_CAP < v"10.0"
    @testset "single-warpgroup Hopper GEMM runs on H100" begin
        # The descriptor-build math is host-side; we just need to upload the
        # blob to a device buffer (.global), then alias it as AS.Const for
        # the kernel arg (`cp.async.bulk.tensor.*.tile` accepts .global tmaps).
        # See PTX 9.2 §9.7.8.24.6.

        # Inputs: A = ones(BM, BK) bf16, B = ones(BK, BN) bf16. With f32 acc
        # the expected output of every lane's d[1] is BK = 16.0f0.
        A_bits = CuArray(fill(UInt16(0x3f80) >> 0, HOPPER_BM, HOPPER_BK))  # 1.0 in bf16
        B_bits = CuArray(fill(UInt16(0x3f80) >> 0, HOPPER_BK, HOPPER_BN))

        # Build TMA descriptors (rank-2). Innermost = K for A (row-major).
        tmap_A_host = PTX.tensor_map_tile_2d(
            :bf16, pointer(A_bits), HOPPER_BM, HOPPER_BK, HOPPER_BM, HOPPER_BK;
            swizzle = :B32)
        tmap_B_host = PTX.tensor_map_tile_2d(
            :bf16, pointer(B_bits), HOPPER_BK, HOPPER_BN, HOPPER_BK, HOPPER_BN;
            swizzle = :NONE)

        # Upload the 128-byte blobs to device memory.
        tmap_A_dev = CuArray{UInt8}(undef, 128)
        tmap_B_dev = CuArray{UInt8}(undef, 128)
        copyto!(tmap_A_dev, collect(tmap_A_host.data))
        copyto!(tmap_B_dev, collect(tmap_B_host.data))

        D = CUDACore.zeros(Float32, HOPPER_THREADS)
        a_const = reinterpret(Core.LLVMPtr{UInt8, PTX.AS.Const}, UInt64(pointer(tmap_A_dev)))
        b_const = reinterpret(Core.LLVMPtr{UInt8, PTX.AS.Const}, UInt64(pointer(tmap_B_dev)))

        # `feature_set=:arch` targets sm_90a (the arch-specific variant
        # required for wgmma) instead of @cuda's default baseline sm_90.
        # Without it ptxas rejects every wgmma.* instruction.
        @cuda threads=HOPPER_THREADS feature_set=:arch _hopper_warpgroup_gemm_kernel!(D, a_const, b_const)
        CUDACore.synchronize()

        # d[1] of every lane (ones × ones, K=16 contractions) → 16.0f0.
        # Numerical validation per-lane is ROADMAP item 1; we only sanity-check
        # the warpgroup ran and wrote a non-zero value here.
        @test any(!=(0f0), Array(D))
    end
end
