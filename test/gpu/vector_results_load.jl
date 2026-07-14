# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc>=7.5
#
# Small end-to-end ABI proof for ordinary vector loads. The offline tier
# always compiles the exact runtime kernel; a live device additionally checks
# that the returned NTuple preserves lane order and values.

function _vector_result_load_runtime!(out::CuDeviceVector{UInt32,1},
                                      input::CuDeviceVector{UInt32,1})
    x, y = ptx"ld.global.v2.u32"(pointer(input))
    @inbounds begin
        out[1] = x
        out[2] = y
    end
    return nothing
end

@testset "vector-result ld runtime kernel compiles at sm_75" begin
    types = Tuple{CuDeviceVector{UInt32,1}, CuDeviceVector{UInt32,1}}
    @test ptxas_compiles(_vector_result_load_runtime!, types;
                         cap = v"7.5", feature_set = :baseline)
    ptx = emit_ptx(_vector_result_load_runtime!, types;
                   cap = v"7.5", feature_set = :baseline)
    @test occursin("ld.global.v2.u32", ptx)
end

if test_runtime_supported(@__FILE__)
    @testset "vector-result ld returns ordered lane values" begin
        input = CuArray(UInt32[0x01234567, 0x89abcdef])
        out = CUDACore.zeros(UInt32, 2)
        @cuda threads=1 _vector_result_load_runtime!(out, input)
        CUDACore.synchronize()
        @test Array(out) == Array(input)
    end
end
