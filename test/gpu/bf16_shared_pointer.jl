# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc>=8.0
#
# PTX ISA 9.3 §5.2.3 requires BF16 data to live in `.b16` variables.
# These patterns deliberately disagree with Float16: 0x3f80 is BF16 1.0 but
# Float16 1.875, while 0x3c00 is Float16 1.0 but BF16 0.0078125.  Keeping both
# the raw bits and their BF16 conversions observable makes a storage-type
# substitution visible rather than relying on a type name.

const _BF16_ONE_BITS = UInt16(0x3f80)
const _F16_ONE_BITS = UInt16(0x3c00)

function _bf16_shared_pointer_probe!(bits_out::CuDeviceVector{UInt16,1},
                                     values_out::CuDeviceVector{Float32,1})
    storage = CuStaticSharedArray(UInt16, 2)
    base = pointer(storage)
    ptx"st.shared.b16"(base, _BF16_ONE_BITS)
    ptx"st.shared.b16"(base + 2, _F16_ONE_BITS)

    first_bits = ptx"ld.shared.b16"(base)
    second_bits = ptx"ld.shared.b16"(base + 2)
    first_value = ptx"cvt.f32.bf16"(first_bits)
    second_value = ptx"cvt.f32.bf16"(second_bits)

    @inbounds begin
        bits_out[1] = first_bits
        bits_out[2] = second_bits
        values_out[1] = first_value
        values_out[2] = second_value
    end
    return nothing
end

@testset "BF16 shared carrier compiles at the cvt.f32.bf16 sm_80 floor" begin
    types = Tuple{CuDeviceVector{UInt16,1},CuDeviceVector{Float32,1}}
    llvm = emit_llvm(_bf16_shared_pointer_probe!, types;
                     cap = v"8.0", feature_set = :baseline)
    ptx = emit_ptx(_bf16_shared_pointer_probe!, types;
                   cap = v"8.0", feature_set = :baseline)

    @test occursin("addrspace(3)", llvm)
    @test occursin("st.shared.b16", ptx)
    @test occursin("ld.shared.b16", ptx)
    @test occursin("cvt.f32.bf16", ptx)
    @test ptxas_compiles(_bf16_shared_pointer_probe!, types;
                         cap = v"8.0", feature_set = :baseline)
end

if test_runtime_supported(@__FILE__)
    @testset "BF16 shared carrier preserves bits and BF16 interpretation" begin
        bits_out = CUDACore.zeros(UInt16, 2)
        values_out = CUDACore.zeros(Float32, 2)
        @cuda threads=1 _bf16_shared_pointer_probe!(bits_out, values_out)
        CUDACore.synchronize()

        @test Array(bits_out) == UInt16[_BF16_ONE_BITS, _F16_ONE_BITS]
        @test Array(values_out) == Float32[1.0, 0.0078125]
        @test Array(values_out) != Float32[1.875, 1.0]
    end
end
