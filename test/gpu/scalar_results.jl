# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=7.5
#
# Ada-executable semantic representatives for fixed scalar results. The
# mixed-precision f32←f16/bf16 family starts at sm_100 and is therefore covered
# by the exact-floor ptxas tier, not this CC 8.9 runtime tier.

function _scalar_results_runtime!(uout, sout, u64out, s64out,
                                  pop64::UInt64, pop32::UInt32,
                                  dp4a_s_bits::UInt32, dp4a_u_bits::Int32,
                                  dp4a_c_bits::UInt32,
                                  dp2a_s::Int32, dp2a_b::UInt32,
                                  dp2a_c::Int32)
    @inbounds begin
        uout[1] = ptx"popc.b64"(pop64)
        uout[2] = ptx"clz.b64"(pop64)
        uout[3] = ptx"popc.b32"(pop32)
        uout[4] = ptx"clz.b32"(pop32)

        # Cross-signed Julia carriers are legal under PTX §6.1; the modifiers
        # still select signed-byte × unsigned-byte semantics and an s32 result.
        sout[1] = ptx"dp4a.s32.u32"(dp4a_s_bits, dp4a_u_bits, dp4a_c_bits)
        sout[2] = ptx"dp2a.hi.s32.u32"(dp2a_s, dp2a_b, dp2a_c)

        uout[5] = ptx"dp4a.u32.u32"(UInt32(0x04030201),
                                            UInt32(0x08070605), UInt32(9))
        uout[6] = ptx"dp2a.lo.u32.u32"(UInt32(0x00030002),
                                               UInt32(0x281e140a), UInt32(7))

        uout[7] = ptx"cvt.pack.sat.s16.s32"(Int32(40000), Int32(-40000))
        uout[8] = ptx"cvt.pack.sat.u8.s32.b32"(Int32(300), Int32(-4),
                                                       UInt32(0xabcd1234))
        uout[9] = ptx"cvt.pack.sat.u4.s32.b32"(Int32(12), Int32(-2),
                                                       UInt32(0x12345678))

        # Every `.wide` result/accumulator ABI: 16→32 and 32→64,
        # signed and unsigned, mul and mad.
        uout[10] = ptx"mul.wide.u16"(UInt16(60000), UInt16(3))
        uout[11] = ptx"mad.wide.u16"(UInt16(60000), UInt16(3), UInt32(7))
        sout[3] = ptx"mul.wide.s16"(Int16(-1234), Int16(17))
        sout[4] = ptx"mad.wide.s16"(Int16(-1234), Int16(17), Int32(-9))
        u64out[1] = ptx"mul.wide.u32"(typemax(UInt32), UInt32(2))
        u64out[2] = ptx"mad.wide.u32"(typemax(UInt32), UInt32(2), UInt64(11))
        s64out[1] = ptx"mul.wide.s32"(Int32(-2_000_000_000), Int32(3))
        s64out[2] = ptx"mad.wide.s32"(Int32(-2_000_000_000), Int32(3),
                                             Int64(17))
        uout[12] = ptx"prmt.b32.rc8"(UInt32(0x44332211), UInt32(0), UInt32(2))
    end
    return nothing
end

@testset "fixed scalar results execute with ISA semantics" begin
    pop64 = UInt64(0xf0f00000ffff0001)
    pop32 = UInt32(0x00f00000)
    dp4a_s_bits = UInt32(0x04fd02ff) # signed bytes (-1, 2, -3, 4)
    dp4a_u_bits = reinterpret(Int32, UInt32(0x08070605)) # unsigned (5,6,7,8)
    dp4a_c_bits = reinterpret(UInt32, Int32(-10))
    dp2a_s = reinterpret(Int32, UInt32(0x0003fffe)) # signed halfs (-2, 3)
    dp2a_b = UInt32(0x281e140a)                     # bytes (10,20,30,40)
    dp2a_c = Int32(-5)

    uout = CUDACore.zeros(UInt32, 12)
    sout = CUDACore.zeros(Int32, 4)
    u64out = CUDACore.zeros(UInt64, 2)
    s64out = CUDACore.zeros(Int64, 2)
    @cuda threads=1 _scalar_results_runtime!(
        uout, sout, u64out, s64out, pop64, pop32,
        dp4a_s_bits, dp4a_u_bits, dp4a_c_bits,
        dp2a_s, dp2a_b, dp2a_c)
    CUDACore.synchronize()

    @test Array(uout) == UInt32[
        count_ones(pop64),
        leading_zeros(pop64),
        count_ones(pop32),
        leading_zeros(pop32),
        79,                    # 1*5 + 2*6 + 3*7 + 4*8 + 9
        87,                    # 2*10 + 3*20 + 7
        0x7fff8000,            # clamp(40000,-40000) to signed 16-bit lanes
        0x1234ff00,            # c.low16, clamp(300,-4) to unsigned bytes
        0x345678c0,            # c.low24, clamp(12,-2) to unsigned nibbles
        180000,
        180007,
        0x33333333,            # rc8 replicates byte selector c[1:0] == 2
    ]
    @test Array(sout) == Int32[
        8,                     # (-1*5)+(2*6)+(-3*7)+(4*8)-10
        55,                    # (-2*30)+(3*40)-5, high byte pair
        -20978,
        -20987,
    ]
    @test Array(u64out) == UInt64[8589934590, 8589934601]
    @test Array(s64out) == Int64[-6000000000, -5999999983]
end
