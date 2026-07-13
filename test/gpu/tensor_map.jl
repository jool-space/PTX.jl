# TEST_TARGET: requires=gpu evidence=runtime target=sm_90
#
# Smoke tests for `tensor_map_encode_tiled` — the host-side wrapper around
# CUDA driver's `cuTensorMapEncodeTiled`. Although encoding is a host-side
# call, the driver checks the *device*'s TMA support and returns
# ERROR_NOT_SUPPORTED (801) on pre-Hopper GPUs (verified on sm_89) — so
# this is gated on CC 9.0 like the ops that consume the descriptor.

using PTX: CuTensorMap, tensor_map_encode_tiled, tensor_map_tile_2d
using CUDACore

const DRIVER_HAS_TMA = CUDACore.functional() &&
                       CUDACore.runtime_version() >= v"12.0"

if !DRIVER_HAS_TMA
    @info "skipping tensor_map driver tests (needs CUDA driver >= 12.0)"
else
    @testset "2D bf16 tile" begin
        # 128×128 bf16 tensor; 64×128B = 64 cols × 2B = 128B box width
        # (the B128 swizzle alignment), 64 rows of box.
        rows, cols = 128, 128
        buf = CuArray{UInt16}(undef, rows * cols)
        tmap = tensor_map_tile_2d(:bf16, pointer(buf),
                                  rows, cols, 64, 64;
                                  swizzle = :B128)
        @test tmap isa CuTensorMap
        @test sizeof(tmap) == 128
        # Driver writes non-zero bytes into the descriptor.
        @test any(!=(0x00), tmap.data)
    end

    @testset "3D bf16 — full kwarg surface" begin
        # Pyptx's 3D layout: reshape an (H, W) bf16 matrix into
        # (64_innermost, H, W/64). 64 bf16 elements = 128 B, which is the
        # B128 swizzle line width.
        height, width = 32, 128
        elem_bytes = 2
        buf = CuArray{UInt16}(undef, height * width)
        tmap = tensor_map_encode_tiled(
            :bf16, pointer(buf),
            (64, height, width ÷ 64),
            (elem_bytes * width, 64 * elem_bytes),
            (64, height, width ÷ 64);
            swizzle = :B128, oob_fill = :NONE)
        @test tmap isa CuTensorMap
        @test any(!=(0x00), tmap.data)
    end

    @testset "rejects bad shape" begin
        # global_strides must have length N-1 = 1.
        buf = CuArray{UInt16}(undef, 16 * 16)
        @test_throws ArgumentError tensor_map_encode_tiled(
            :bf16, pointer(buf),
            (16, 16), (32, 32),  # WRONG: should be length 1
            (8, 8))
        # Rank 6 is over the TMA limit.
        @test_throws ArgumentError tensor_map_encode_tiled(
            :bf16, pointer(buf),
            (1, 1, 1, 1, 1, 1),
            (2, 2, 2, 2, 2),
            (1, 1, 1, 1, 1, 1))
    end

    @testset "distinct swizzles produce distinct blobs" begin
        # Sanity: the swizzle kwarg actually reaches the driver.
        buf = CuArray{UInt16}(undef, 128 * 128)
        no_sw  = tensor_map_tile_2d(:bf16, pointer(buf), 128, 128, 64, 64;
                                     swizzle = :NONE)
        with_sw = tensor_map_tile_2d(:bf16, pointer(buf), 128, 128, 64, 64;
                                     swizzle = :B128)
        @test no_sw.data != with_sw.data
    end
end
