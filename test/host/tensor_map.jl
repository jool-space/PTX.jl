using PTX: CuTensorMap, tensor_map_dtype_code, tensor_map_swizzle_code,
           tensor_map_interleave_code, tensor_map_l2_promotion_code,
           tensor_map_oob_fill_code

# Host-only checks for the Symbol/Type → driver-enum surface. The actual
# `cuTensorMapEncodeTiled` call lives in PTXCUDACoreExt and is exercised
# by test/gpu/tensor_map.jl (needs a CUDA context).

@testset "CuTensorMap is a 128-byte blob" begin
    tmap = CuTensorMap()
    @test sizeof(tmap) == 128
    @test sizeof(CuTensorMap) == 128
    @test length(tmap.data) == 128
    @test all(==(0x00), tmap.data)
end

@testset "Symbol → driver enum codes" begin
    # Values match the CUtensorMap{...}_enum definitions in
    # CUDACore/lib/cudadrv/libcuda.jl. Don't bump these blindly — the driver
    # reads the raw integer.
    @test tensor_map_swizzle_code(:NONE) == 0
    @test tensor_map_swizzle_code(:B32) == 1
    @test tensor_map_swizzle_code(:B64) == 2
    @test tensor_map_swizzle_code(:B128) == 3
    # Blackwell sm_100a 128B-atom variants (driver enums 4–6).
    @test tensor_map_swizzle_code(:B128_ATOM_32B) == 4
    @test tensor_map_swizzle_code(:B128_ATOM_64B) == 6

    @test tensor_map_interleave_code(:NONE) == 0
    @test tensor_map_interleave_code(:B16) == 1
    @test tensor_map_interleave_code(:B32) == 2

    @test tensor_map_l2_promotion_code(:NONE) == 0
    @test tensor_map_l2_promotion_code(:B256) == 3

    @test tensor_map_oob_fill_code(:NONE) == 0
    @test tensor_map_oob_fill_code(:NAN_REQUEST_ZERO_FMA) == 1
end

@testset "dtype mapping" begin
    # Julia types map to driver enum codes.
    @test tensor_map_dtype_code(Float32) == 7
    @test tensor_map_dtype_code(Float16) == 6
    @test tensor_map_dtype_code(Float64) == 8
    @test tensor_map_dtype_code(UInt8)   == 0
    @test tensor_map_dtype_code(UInt16)  == 1
    @test tensor_map_dtype_code(Int32)   == 3

    # Symbol form covers types without a Julia counterpart.
    @test tensor_map_dtype_code(:bf16) == 9
    @test tensor_map_dtype_code(:tf32) == 11
    @test tensor_map_dtype_code(:u4_align8b) == 13
    @test tensor_map_dtype_code(:u6_align16b) == 15
end

@testset "rejects bad keys" begin
    @test_throws ArgumentError tensor_map_swizzle_code(:B256)  # not a swizzle
    @test_throws ArgumentError tensor_map_dtype_code(:nonsense)
    @test_throws ArgumentError tensor_map_oob_fill_code(:NAN)  # spelled :NAN_REQUEST_ZERO_FMA
end

