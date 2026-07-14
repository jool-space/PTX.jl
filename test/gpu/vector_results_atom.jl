# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc>=9.0
#
# Vector atomics start at sm_90. The offline tier always compiles the exact
# runtime kernel; eligible devices check both the returned old-value tuple and
# the two independently updated memory elements.
#
# multimem.ld_reduce intentionally has no runtime claim here: executing it
# requires a real multicast/multimem address, and an ordinary global pointer
# would have undefined behavior under PTX 9.3 section 9.7.9.15.

function _vector_result_atom_runtime!(old_out::CuDeviceVector{Float32,1},
                                      values::CuDeviceVector{Float32,1},
                                      add0::Float32, add1::Float32)
    old = ptx"atom.global.add.v2.f32"(
        pointer(values), (add0, add1))
    @inbounds begin
        old_out[1] = old[1]
        old_out[2] = old[2]
    end
    return nothing
end

@testset "vector-result atom runtime kernel compiles at sm_90" begin
    types = Tuple{CuDeviceVector{Float32,1}, CuDeviceVector{Float32,1},
                  Float32, Float32}
    @test ptxas_compiles(_vector_result_atom_runtime!, types;
                         cap = v"9.0", feature_set = :baseline)
    ptx = emit_ptx(_vector_result_atom_runtime!, types;
                   cap = v"9.0", feature_set = :baseline)
    @test occursin("atom.global.add.v2.f32", ptx)
end

if test_runtime_supported(@__FILE__)
    @testset "vector-result atom returns old values and updates memory" begin
        initial = Float32[1.25, -2.5]
        values = CuArray(initial)
        old_out = CUDACore.zeros(Float32, 2)
        addends = (0.75f0, 3.0f0)
        @cuda threads=1 _vector_result_atom_runtime!(
            old_out, values, addends...)
        CUDACore.synchronize()
        @test Array(old_out) == initial
        @test Array(values) == initial .+ collect(addends)
    end
end
