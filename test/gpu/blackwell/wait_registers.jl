# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc==10|cc==11
# Pass-through values include NaN payloads, signed zero and signed extrema.
# No asynchronous operation is pending; the wait is still a warp collective.
# The attention tests exercise the dependent tcgen05.ld consumers separately.
using PTX: wait_registers

@generated function _wr_bits_kernel!(out::CuDeviceVector{T,1},
                                     input::CuDeviceVector{T,1},
                                     op::W, ::Val{N}) where {T,W,N}
    loads = [:(input[$i]) for i in 1:N]
    stores = [:(out[$i] = result[$i]) for i in 1:N]
    quote
        @inbounds values = tuple($(loads...))
        result = wait_registers(op, values)
        # Every lane participates in the wait; one lane writes the result.
        if ptx"mov.u32"(sreg"tid.x") == UInt32(0)
            @inbounds begin
                $(stores...)
            end
        end
        nothing
    end
end

function _wr_scalar_kernel!(out, input, op)
    value = @inbounds input[1]
    result = wait_registers(op, value)
    if ptx"mov.u32"(sreg"tid.x") == UInt32(0)
        @inbounds out[1] = result
    end
    nothing
end

function _wr_empty_tuple_kernel!(out, op)
    result = wait_registers(op, ())
    lane = ptx"mov.u32"(sreg"tid.x")
    @inbounds out[lane + UInt32(1)] = result === ()
    nothing
end

@testset "empty tuples retain the selected wait without register operands" begin
    for (op, mnemonic) in (
            (ptx"tcgen05.wait::ld.sync.aligned", "tcgen05.wait::ld.sync.aligned"),
            (ptx"tcgen05.wait::st.sync.aligned", "tcgen05.wait::st.sync.aligned"))
        tt = Tuple{CuDeviceVector{Bool,1}, typeof(op)}
        ir = emit_llvm(_wr_empty_tuple_kernel!, tt; cap=v"10.0", feature_set=:arch)
        @test occursin("call void asm sideeffect \"$mnemonic;\", \"~{memory}\"()", ir)
        @test count("tcgen05.wait::", ir) == 1
        @test ptxas_compiles(_wr_empty_tuple_kernel!, tt; cap=v"10.0", feature_set=:arch)
    end
end

const _WR_BIT_CASES = (
    UInt32 => UInt32[0x00000000, 0x80000000, 0xffffffff],
    Int32 => UInt32[0x80000000, 0x7fffffff, 0xffffffff],
    Float32 => UInt32[0x7fc01234, 0x80000000, 0xff800000],
    UInt64 => UInt64[0x0000000100000002, 0xffffffffffffffff, 0],
    Int64 => UInt64[0x8000000000000000, 0x7fffffffffffffff, 0xffffffffffffffff],
    Float64 => UInt64[0x7ff8000012345678, 0x8000000000000000, 0xfff0000000000000],
)

@testset "register waits assemble with scalar, odd and wide tuples" begin
    for op in (ptx"tcgen05.wait::ld.sync.aligned", ptx"tcgen05.wait::st.sync.aligned"),
            T in (UInt32, Int32, Float32, UInt64, Int64, Float64)
        tt = Tuple{CuDeviceVector{T,1}, CuDeviceVector{T,1}, typeof(op)}
        @test ptxas_compiles(_wr_scalar_kernel!, tt; cap=v"10.0", feature_set=:arch)
        for N in (1, 3, 32, 64)
            @test ptxas_compiles(_wr_bits_kernel!, Tuple{tt.parameters...,Val{N}};
                                 cap=v"10.0", feature_set=:arch)
        end
    end
end

if test_runtime_supported(@__FILE__)
    @testset "empty tuple waits return an empty tuple in every lane" begin
        for op in (ptx"tcgen05.wait::ld.sync.aligned", ptx"tcgen05.wait::st.sync.aligned")
            out = CuArray(fill(false, 32))
            @cuda threads=32 _wr_empty_tuple_kernel!(out, op)
            CUDACore.synchronize()
            @test all(Array(out))
        end
    end

    @testset "register waits preserve exact bits" begin
        for op in (ptx"tcgen05.wait::ld.sync.aligned", ptx"tcgen05.wait::st.sync.aligned"),
                (T, bits) in _WR_BIT_CASES, N in (1, 3, 32, 64)
            expected = [bits[mod1(i, length(bits))] for i in 1:N]
            input = CuArray(reinterpret(T, expected))
            out = similar(input)
            @cuda threads=32 _wr_bits_kernel!(out, input, op, Val(N))
            CUDACore.synchronize()
            @test reinterpret(eltype(bits), Array(out)) == expected
            @cuda threads=32 _wr_scalar_kernel!(out, input, op)
            CUDACore.synchronize()
            @test reinterpret(eltype(bits), Array(out))[1] == first(expected)
        end
    end
end
