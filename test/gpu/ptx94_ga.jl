# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=9.0
include(joinpath(@__DIR__, "..", "ptx94_ga_defs.jl"))

@testset "readonly scalar loads preserve their result bits" begin
    if _ptxas_isa() < v"9.4"
        @test_skip "PTX 9.4 assembler required"
    else
        value = UInt64(0x3ff00000ff80ff81)
        input = CuArray([value])
        out = CUDACore.zeros(UInt64, 56)
        @cuda threads=1 _ga_readonly_loads!(out, input)
        CUDACore.synchronize()
        expected = UInt64[
            0x81, 0xff81, 0xff80ff81, value,
            0x81, 0xff81, 0xff80ff81, value,
            0x81, 0xff81, 0xff80ff81, value,
            0xff80ff81, value,
        ]
        @test Array(out) == repeat(expected, 4)
    end
end
