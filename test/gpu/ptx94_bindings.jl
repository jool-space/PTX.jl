# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=9.0
# Numeric evidence for the sm_90 subset of the PTX ISA 9.4 bindings:
# `.add.noftz.f32` atomics preserve subnormal inputs and results (the flush
# is the documented behaviour of the plain `atom.add.f32`), and the L1
# prefetch forms run alongside them.
include(joinpath(@__DIR__, "..", "ptx94_bindings_defs.jl"))

# Device-side wrappers: `pointer` on the device array yields the global
# address-space pointer the shared kernels take.
function _bind_noftz_runtime!(out::CuDeviceVector{Float32, 1},
                              target::CuDeviceVector{Float32, 1},
                              value::Float32)
    _bind_sm90_chain!(out, pointer(target), value)
    return nothing
end

function _bind_flushing_atom!(out::CuDeviceVector{Float32, 1},
                              target::CuDeviceVector{Float32, 1},
                              value::Float32)
    @inbounds out[1] = ptx"atom.global.add.f32"(pointer(target), value)
    return nothing
end

@testset "atom/red .add.noftz.f32 preserve subnormals" begin
    seed = reinterpret(Float32, 0x00000001)
    value = reinterpret(Float32, 0x00000002)
    target = CuArray([seed])
    out = CUDACore.zeros(Float32, 2)
    @cuda threads=1 _bind_noftz_runtime!(out, target, value)
    CUDACore.synchronize()
    got = Array(out)
    # Two atomics return the running sum; two reductions add silently.
    @test reinterpret(UInt32, got[1]) == 0x00000001
    @test reinterpret(UInt32, got[2]) == 0x00000003
    @test reinterpret(UInt32, Array(target)[1]) == 0x00000009

    # The non-.noftz form flushes both the subnormal input and result.
    flushed = CuArray([seed])
    @cuda threads=1 _bind_flushing_atom!(out, flushed, value)
    CUDACore.synchronize()
    @test reinterpret(UInt32, Array(flushed)[1]) == 0x00000000
end
