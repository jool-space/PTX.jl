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
    storage = CuStaticSharedArray(BFloat16, 2)
    base = pointer(storage)
    ptx"st.shared.b16"(base, _BF16_ONE_BITS)
    ptx"st.shared.b16"(base + 2, _F16_ONE_BITS)

    first_bits = ptx"ld.shared.b16"(base)
    second_bits = ptx"ld.shared.b16"(base + 2)
    first_value = ptx"cvt.f32.bf16"(@inbounds storage[1])
    second_value = ptx"cvt.f32.bf16"(@inbounds storage[2])

    @inbounds begin
        bits_out[1] = first_bits
        bits_out[2] = second_bits
        values_out[1] = first_value
        values_out[2] = second_value
    end
    return nothing
end

@testset "BFloat16 shared storage compiles at the cvt.f32.bf16 sm_80 floor" begin
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
    @testset "BFloat16 shared storage preserves bits and BF16 interpretation" begin
        bits_out = CUDACore.zeros(UInt16, 2)
        values_out = CUDACore.zeros(Float32, 2)
        @cuda threads=1 _bf16_shared_pointer_probe!(bits_out, values_out)
        CUDACore.synchronize()

        @test Array(bits_out) == UInt16[_BF16_ONE_BITS, _F16_ONE_BITS]
        @test Array(values_out) == Float32[1.0, 0.0078125]
        @test Array(values_out) != Float32[1.875, 1.0]
    end
end

function _bf16_conversion_probe!(out, raw_out, input)
    i = Int(threadIdx().x)
    @inbounds begin
        out[i] = ptx"cvt.rn.bf16.f32"(input[i])
        raw_out[i] = ptx"cvt.rn.bf16.f32"raw(input[i])
    end
    return nothing
end

@testset "BFloat16 array conversion rounds ties to even" begin
    # Repetition exercises host broadcast vectorization as well as scalar ties.
    input = repeat(Float32[-0.0, 0.0, 1.00390625, 1.01171875,
                           -1.00390625, -1.01171875, Inf, -Inf, NaN], 256)
    values = BFloat16.(input)
    expected = repeat(Float32[-0.0, 0.0, 1.0, 1.015625,
                              -1.0, -1.015625, Inf, -Inf, NaN], 256)
    @test isequal(Float32.(values), expected)

    if test_runtime_supported(@__FILE__)
        input_d = CuArray(input[1:9])
        out = CuArray{BFloat16}(undef, 9)
        raw_out = similar(out)
        @cuda threads=9 _bf16_conversion_probe!(out, raw_out, input_d)
        CUDACore.synchronize()
        @test isequal(Array(out), values[1:9])
        @test isequal(Array(raw_out), values[1:9])
    end
end
