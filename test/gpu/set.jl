# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=9.0
# The bfloat forms require CC 9.0. Packed integer set requires CC 10.7 and
# has separate offline assembly coverage; this kernel uses scalar/half forms.
function _set_semantics!(out, a::Int64, b::Int64, gate::Bool,
                          h1::Float16, h2::Float16, bf1::UInt16, bf2::UInt16,
                          fp_nan::Float32, tiny::Float32)
    @inbounds begin
        out[1] = ptx"set.lt.u32.s64"(a, b)
        out[2] = reinterpret(UInt32, ptx"set.lt.s32.s64"(a, b))
        out[3] = reinterpret(UInt32, ptx"set.lt.f32.s64"(a, b))
        out[4] = ptx"set.lt.u32.u64"(reinterpret(UInt64, a), reinterpret(UInt64, b))
        out[5] = ptx"set.lt.xor.u32.s64"(a, b, gate)
        out[6] = reinterpret(UInt32, ptx"set.lt.and.f32.s64"(a, b, !gate))
        out[7] = ptx"set.gt.or.u32.s64"(a, b, gate)
        out[8] = ptx"set.ne.u32.f32"(fp_nan, 0.0f0)
        out[9] = ptx"set.neu.u32.f32"(fp_nan, 0.0f0)
        out[10] = ptx"set.num.u32.f32"(fp_nan, 0.0f0)
        out[11] = reinterpret(UInt32, ptx"set.nan.f32.f32"(fp_nan, 0.0f0))
        out[12] = ptx"set.eq.u32.f32"(tiny, 0.0f0)
        out[13] = ptx"set.eq.ftz.u32.f32"(tiny, 0.0f0)
        out[14] = UInt32(reinterpret(UInt16, ptx"set.lt.f16.f16"(h1, h2)))
        out[15] = UInt32(ptx"set.lt.u16.f16"(h1, h2))
        out[16] = UInt32(reinterpret(UInt16, ptx"set.lt.s16.f16"(h1, h2)))
        out[17] = UInt32(ptx"set.lt.bf16.f16"(h1, h2))
        out[18] = ptx"set.lt.u32.bf16"(bf1, bf2)
        # Lane 0 compares -1 < 0 (true), lane 1 compares 2 < 1 (false).
        out[19] = ptx"set.lt.f16x2.f16x2"(UInt32(0x4000bc00), UInt32(0x3c000000))
        out[20] = ptx"set.lt.u32.f16x2"(UInt32(0x4000bc00), UInt32(0x3c000000))
        out[21] = ptx"set.lt.bf16x2.bf16x2"(UInt32(0x4000bf80), UInt32(0x3f800000))
        out[22] = ptx"set.lt.u32.bf16x2"(UInt32(0x4000bf80), UInt32(0x3f800000))
        out[23] = reinterpret(UInt32,
            ptx"set.lt.s32.bf16x2"(UInt32(0x4000bf80), UInt32(0x3f800000)))
        out[24] = ptx"set.lt.xor.u32.f16x2"(
            UInt32(0x4000bc00), UInt32(0x3c000000), gate)
        # Widening the raw bits must not retain a duplicate high half.
        out[25] = UInt32(reinterpret(UInt16, ptx"set.gt.f16.f16"(h1, h2)))
        out[26] = UInt32(reinterpret(UInt16, ptx"set.lt.and.f16.f16"(h1, h2, !gate)))
        out[27] = UInt32(reinterpret(UInt16, ptx"set.lt.f16.f16"raw(h1, h2)))
        out[28] = UInt32(reinterpret(UInt16, ptx"set.lt.f16.f32"(-1.0f0, 2.0f0)))
        out[29] = UInt32(reinterpret(UInt16, ptx"set.lt.f16.f64"(-1.0, 2.0)))
        out[30] = UInt32(reinterpret(UInt16, ptx"set.eq.f16.b64"(
            reinterpret(UInt64, a), reinterpret(UInt64, a))))
        out[31] = UInt32(reinterpret(UInt16, ptx"set.lt.f16.s64"(a, b)))
        out[32] = UInt32(reinterpret(UInt16, ptx"set.eq.xor.ftz.f16.f32"(
            tiny, 0.0f0, gate)))
    end
    nothing
end

@testset "set numeric results, NaNs, predicates, and packed lane masks" begin
    out = CUDACore.zeros(UInt32, 32)
    @cuda threads=1 _set_semantics!(out, Int64(-2), Int64(1), true,
        Float16(-1), Float16(2), UInt16(0xbf80), UInt16(0x4000),
        Float32(NaN), reinterpret(Float32, UInt32(1)))
    CUDACore.synchronize()
    expected = UInt32[
        0xffffffff, 0xffffffff, 0x3f800000, 0,
        0, 0, 0xffffffff, 0, 0xffffffff, 0, 0x3f800000,
        0, 0xffffffff, 0x3c00, 0xffff, 0xffff, 0x3f80, 0xffffffff,
        0x00003c00, 0x0000ffff, 0x00003f80, 0x0000ffff, 0x0000ffff, 0xffff0000,
        0, 0, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0,
    ]
    actual = Array(out)
    @testset "result $i" for i in eachindex(expected)
        @test actual[i] == expected[i]
    end
end
