# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc>=7.5
#
# Runtime proof of the common two-u64 carrier and low/high word order. Atom
# b128 starts at sm_90 and query-cancel needs a live cancellation response, so
# those families retain offline ptxas evidence only.

function _b128_load_store_runtime!(out::CuDeviceVector{UInt64,1},
                                   scratch::CuDeviceVector{UInt64,1},
                                   lo::UInt64, hi::UInt64)
    ptr = reinterpret(Core.LLVMPtr{UInt64,PTX.AS.Global}, pointer(scratch))
    ptx"st.global.b128"(ptr, (lo, hi))
    loaded_lo, loaded_hi = ptx"ld.global.b128"(ptr)
    @inbounds begin
        out[1] = loaded_lo
        out[2] = loaded_hi
    end
    return nothing
end

@testset "scalar-b128 runtime kernel compiles at retained sm_75" begin
    types = Tuple{CuDeviceVector{UInt64,1},CuDeviceVector{UInt64,1},UInt64,UInt64}
    @test ptxas_compiles(_b128_load_store_runtime!, types;
                         cap = v"7.5", feature_set = :baseline)
    ptx = emit_ptx(_b128_load_store_runtime!, types;
                   cap = v"7.5", feature_set = :baseline)
    @test occursin("st.global.b128", ptx)
    @test occursin("ld.global.b128", ptx)
end

if test_runtime_supported(@__FILE__)
    @testset "scalar-b128 load/store preserves both u64 words" begin
        out = CUDACore.zeros(UInt64, 2)
        scratch = CUDACore.zeros(UInt64, 2)
        lo, hi = 0x0123456789abcdef, 0xfedcba9876543210
        @cuda threads=1 _b128_load_store_runtime!(out, scratch, lo, hi)
        CUDACore.synchronize()
        @test Array(out) == UInt64[lo, hi]
        @test Array(scratch) == UInt64[lo, hi]
    end
end
