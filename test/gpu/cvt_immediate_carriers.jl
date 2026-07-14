# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8
#
# Semantic representatives for the exact Julia carriers emitted by the
# ordinary-cvt transpiler. Host tests prove source-position selection; offline
# ptxas tests prove target legality; this tier checks the resulting values on
# the active device (including the project's Ada and GB10 runners).

function _cvt_immediate_runtime!(uout, sout, fout)
    @inbounds begin
        uout[1] = ptx"cvt.u16.u8"(UInt8(255))
        uout[2] = ptx"cvt.u32.u16"(UInt16(513))
        uout[3] = ptx"cvt.u64.u32"(UInt32(65539))
        sout[1] = ptx"cvt.s16.s8"(Int8(-11))
        sout[2] = ptx"cvt.s32.s16"(Int16(-1025))
        # Exact typed literals emitted after PTX §4.5.5 evaluation and §4.5.1
        # use-site truncation.
        uout[6] = ptx"cvt.u16.u8"(UInt8(0x00))
        sout[3] = ptx"cvt.s16.s8"(Int8(-1))
        uout[7] = ptx"cvt.u32.u32"(UInt32(0xffffffff))
        uout[8] = ptx"cvt.u32.u32"(UInt32(0x00000000))

        # These expressions are emitted for cross-width exact 0d/0f source
        # constants. PTX converts the literal to the declared source type at
        # use before performing the requested cvt operation.
        fout[1] = ptx"cvt.f64.f32"(
            Float32(reinterpret(Float64, 0x3ff8000000000000)))
        fout[2] = Float64(ptx"cvt.rn.f32.f64"(
            Float64(reinterpret(Float32, 0x40200000))))

        uout[4] = ptx"cvt.rn.f16x2.f32"(Float32(1.0), Float32(2.0))
        uout[5] = ptx"cvt.rn.bf16x2.f32"(Float32(3.0), Float32(4.0))
    end
    return nothing
end

@testset "ordinary cvt transpiler carriers execute with ISA semantics" begin
    uout = CUDACore.zeros(UInt64, 8)
    sout = CUDACore.zeros(Int64, 3)
    fout = CUDACore.zeros(Float64, 2)
    @cuda threads=1 _cvt_immediate_runtime!(uout, sout, fout)
    CUDACore.synchronize()

    @test Array(uout) == UInt64[
        255,
        513,
        65539,
        0x3c004000, # a=1.0 in upper f16 lane, b=2.0 in lower lane
        0x40404080, # a=3.0 in upper bf16 lane, b=4.0 in lower lane
        0,
        0xffffffff,
        0,
    ]
    @test Array(sout) == Int64[-11, -1025, -1]
    @test Array(fout) == Float64[1.5, 2.5]
end
