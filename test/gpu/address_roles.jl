# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=7.5
#
# Semantic check for the public integer-address role on the local Ada runner
# and later GPUs. CLC itself is compile-only: executing it safely needs a
# cluster launch and a correctly initialized completion mbarrier.

function _address_role_runtime!(out, raw::UInt64)
    @inbounds begin
        out[1] = ptx"ld.global.u32"(address(raw))
        out[2] = ptx"ld.global.u32"(address(raw + UInt64(4)))
    end
    return nothing
end

@testset "integer address roles execute" begin
    input = CuArray(UInt32[0x12345678, 0xabcdef01])
    output = CUDACore.zeros(UInt32, 2)
    raw = UInt64(pointer(input))
    @cuda threads=1 _address_role_runtime!(output, raw)
    CUDACore.synchronize()
    @test Array(output) == Array(input)
end
