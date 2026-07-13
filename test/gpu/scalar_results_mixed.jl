# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=10.0
#
# Mixed f32←f16/bf16 arithmetic starts at sm_100. Keep it separate from the
# Ada scalar-result suite so Ada skips cleanly while GB10 exercises rounding,
# saturation, and the three contradictory spellings printed by ISA examples.

function _mixed_scalar_results_runtime!(out, one_h::Float16,
                                        quarter_h::Float16,
                                        half_h::Float16,
                                        quarter_b::UInt16,
                                        half_b::UInt16,
                                        three_quarters_b::UInt16)
    tie = reinterpret(Float32, UInt32(0x33800000)) # 2^-24
    @inbounds begin
        out[1] = ptx"add.rn.f32.f16"(one_h, tie)
        out[2] = ptx"add.rp.f32.f16"(one_h, tie)
        out[3] = ptx"sub.rz.sat.f32.bf16"(quarter_b, Float32(0.5))

        # Exact spellings from the ISA examples, retained under
        # `:ptxas_compat` provenance because they contradict the syntax block.
        out[4] = ptx"add.rz.f32.bf16.sat"(three_quarters_b, Float32(0.5))
        out[5] = ptx"sub.rz.f32.f16.sat"(quarter_h, Float32(0.5))

        out[6] = ptx"fma.rn.f32.bf16"(half_b, quarter_b, Float32(0.5))
        out[7] = ptx"fma.rz.sat.f32.f16"(half_h, half_h, Float32(0.9))
        out[8] = ptx"fma.rz.sat.f32.f16.sat"(
            half_h, half_h, Float32(0.9))
    end
    return nothing
end

@testset "mixed scalar results execute with rounding/saturation semantics" begin
    out = CUDACore.zeros(Float32, 8)
    @cuda threads=1 _mixed_scalar_results_runtime!(
        out, Float16(1), Float16(0.25), Float16(0.5),
        bf16_bits(0.25f0), bf16_bits(0.5f0), bf16_bits(0.75f0))
    CUDACore.synchronize()

    got = Array(out)
    @test reinterpret(UInt32, got[1]) == 0x3f800000 # RN tie-to-even
    @test reinterpret(UInt32, got[2]) == 0x3f800001 # RP rounds upward
    @test got[3:end] == Float32[0, 1, 0, 0.625, 1, 1]
end
