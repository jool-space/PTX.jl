# REQUIRES CC 9.0
#
# End-to-end TMA round-trip: host builds a 2D `CUtensorMap` for an 8×8 bf16
# source tensor, uploads the 128-byte descriptor blob to device memory, then
# the kernel TMA-loads the tile into SMEM via `cp.async.bulk.tensor.2d`,
# mbarrier-waits, and each thread writes one cell back to global. The output
# vector must match the input pattern byte-for-byte.
#
# This exercises every piece introduced in this branch:
#
#   1. `PTX.tensor_map_tile_2d`            — host descriptor build
#   2. `CuTensorMap.data → CuArray{UInt8}` — descriptor upload
#   3. `reinterpret(LLVMPtr{T, AS.Const}, UInt64(pointer(dev_blob)))`
#      — kernel-arg dispatch path validated end-to-end (the dispatch was
#        the unknown that gemm_warpgroup deferred numerical validation on)
#   4. `cp.async.bulk.tensor.2d.shared::cta.global.tile`  + mbarrier
#   5. `fence.proxy.async.shared::cta`     — generic↔async proxy ordering
#
# Runs on any TMA-capable device (sm_90 Hopper, sm_100/sm_120/sm_121 Blackwell).

using PTX: tensor_map_tile_2d, CuTensorMap
using CUDACore

function _tma_copy_kernel!(out::CuDeviceVector{UInt16, 1},
                           tma_src::PTX.TMADescriptorPtr)
    smem = CuStaticSharedArray(UInt16, 64)       # 8 × 8 bf16 = 128 B
    mbar = CuStaticSharedArray(UInt64, 1)
    mb_ptr = pointer(mbar)
    s_ptr  = pointer(smem)

    tid = ptx"mov.u32"(sreg"tid.x")

    # Thread 0 does the full TMA-issue sequence (init, fence, arrive, load);
    # bar.sync after makes the inited+armed state visible CTA-wide before
    # other threads enter the test_wait spin. Splitting init/arrive with a
    # bar.sync between deadlocks on H100 — pyptx's _test_tma_* kernels
    # likewise keep these together with bar.sync at the END.
    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mb_ptr, UInt32(1))
        ptx"fence.proxy.async.shared::cta"()
        ptx"mbarrier.arrive.expect_tx.shared.b64"(mb_ptr, UInt32(128))
        ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
            s_ptr, tma_src, Int32(0), Int32(0), mb_ptr)
    end
    ptx"bar.sync"(Val(0))

    while !ptx"mbarrier.test_wait.parity.shared.b64"(mb_ptr, UInt32(0))
    end

    if tid < UInt32(64)
        @inbounds out[Int(tid) + 1] = smem[Int(tid) + 1]
    end
    return nothing
end

@testset "TMA round-trip (encoder + AS.Const dispatch + TMA load)" begin
    # Distinct values per cell so any layout/dispatch bug shows up in the
    # output diff. bf16(1.0) = 0x3f80; +i moves into the mantissa region,
    # still distinct bit patterns 64 lanes deep.
    input_vals = UInt16[(0x3f80 + i) for i in 0:63]
    src = CuArray(reshape(input_vals, 8, 8))

    # 8x8 bf16 = 16 B per row → INTERLEAVE/NONE swizzle.
    tmap_host = tensor_map_tile_2d(:bf16, pointer(src), 8, 8, 8, 8; swizzle = :NONE)
    @test tmap_host isa CuTensorMap

    src_const = upload_tma_descriptor(tmap_host)
    out = CUDACore.zeros(UInt16, 64)
    @cuda threads = 128 _tma_copy_kernel!(out, src_const.ptr)
    CUDACore.synchronize()

    result = Array(out)
    @test result == input_vals
end

@testset "TMA round-trip — B32 swizzle path" begin
    # 16-element-wide row → 32 B → B32 swizzle (matches wgmma m64n*k16 A-tile).
    rows, cols = 16, 16
    input_vals = UInt16[(0x3f80 + i % 64) for i in 0:rows*cols-1]
    src = CuArray(reshape(input_vals, rows, cols))

    tmap_host = tensor_map_tile_2d(:bf16, pointer(src), rows, cols, rows, cols;
                                   swizzle = :B32)
    @test tmap_host isa CuTensorMap

    # The kernel above is shape-specific (8x8); for B32 we'd need a 16x16
    # variant. Just verify the descriptor blob differs from the NONE-swizzle
    # version for the same shape — confirms swizzle bits land in the blob.
    tmap_none = tensor_map_tile_2d(:bf16, pointer(src), rows, cols, rows, cols;
                                   swizzle = :NONE)
    @test tmap_host.data != tmap_none.data
end
