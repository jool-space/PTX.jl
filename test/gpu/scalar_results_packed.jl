# TEST_TARGET: requires=gpu evidence=runtime runtime=cc==12
#
# Packed 4x8 arithmetic is family-specific at sm_120f. A separate target file
# prevents CC 10.x devices from compiling these forms while GB10 (CC 12.1)
# validates lane arithmetic, saturation, relu, and packed UInt32 carriers.

function _packed_scalar_results_runtime!(out)
    @inbounds begin
        out[1] = ptx"add.u8x4"(UInt32(0x04030201), UInt32(0x08070605))
        out[2] = ptx"add.sat.u8x4"(UInt32(0xffc801fa), UInt32(0x0164020a))
        out[3] = ptx"add.s8x4.sat"(UInt32(0xce328878), UInt32(0x9c64ec14))
        out[4] = ptx"sub.sat.s8x4"(UInt32(0x0a007888), UInt32(0x1401ec14))
        out[5] = ptx"neg.s8x4"(UInt32(0x807ffe01))
        out[6] = ptx"min.relu.s8x4"(UInt32(0x886404fb), UInt32(0x8178fefd))
        out[7] = ptx"max.relu.s8x4"(UInt32(0x889c04fb), UInt32(0x8188fefd))
        out[8] = ptx"add.sat.u16x2"(UInt32(0x000afffa), UInt32(0x0014000a))

        # The ISA's min example places `.relu` after the packed type.
        out[9] = ptx"min.s16x2.relu"(UInt32(0x0014fffb), UInt32(0xfff6fffd))
        out[10] = reinterpret(UInt32, ptx"min.relu.s32"(Int32(-5), Int32(2)))
        out[11] = reinterpret(UInt32, ptx"max.relu.s32"(Int32(-5), Int32(2)))
    end
    return nothing
end

@testset "packed scalar results execute with lane semantics" begin
    out = CUDACore.zeros(UInt32, 11)
    @cuda threads=1 _packed_scalar_results_runtime!(out)
    CUDACore.synchronize()

    @test Array(out) == UInt32[
        0x0c0a0806,
        0xffff03ff,
        0x807f807f,
        0xf6ff7f80,
        0x808102ff,
        0x00640000,
        0x00000400,
        0x001effff,
        0x00000000,
        0x00000000,
        0x00000002,
    ]
end
