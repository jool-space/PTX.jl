# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=9.0
# Numeric evidence for the sm_90 subset of the PTX ISA 9.4 bindings:
# `.add.noftz.f32` atomics preserve subnormal inputs and results (the flush
# is the documented behaviour of the plain `atom.add.f32`), and the L1
# prefetch forms run alongside them.
include(joinpath(@__DIR__, "..", "ptx94_bindings_defs.jl"))
using PTX: smem_addr_u32

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

# Bulk reduction into global memory: PTX ISA 9.4 specifies the f32 add as
# non-flushing by default, with `.noftz` as the explicit spelling. Both are
# checked against subnormal operands. The 16-byte source is carved out of a
# static shared array at a 16-byte-aligned offset.
@generated function _bind_bulk_reduce_runtime!(target::CuDeviceVector{Float32, 1},
                                               value::Float32, ::Val{noftz}) where {noftz}
    mods = noftz ? (:reduce, :async, :bulk, :global, Symbol("shared::cta"),
                    :bulk_group, :add, :noftz, :f32) :
                   (:reduce, :async, :bulk, :global, Symbol("shared::cta"),
                    :bulk_group, :add, :f32)
    quote
        smem = CuStaticSharedArray(Float32, 16)
        base = pointer(smem)
        off = (16 - smem_addr_u32(base) % UInt32(16)) % UInt32(16)
        src = base + Int(off)
        tid = ptx"mov.u32"(sreg"tid.x")
        if tid < UInt32(4)
            unsafe_store!(src, value, Int(tid) + 1)
        end
        ptx"bar.sync"(Val(0))
        ptx"fence.proxy.async.shared::cta"()
        if tid == UInt32(0)
            PTX.Operation{:cp, $mods}()(pointer(target), src, UInt32(16))
            ptx"cp.async.bulk.commit_group"()
            ptx"cp.async.bulk.wait_group"(Val(0))
        end
        return nothing
    end
end

@testset "cp.reduce.async.bulk f32 add preserves subnormals" begin
    seed = reinterpret(Float32, 0x00000001)
    value = reinterpret(Float32, 0x00000002)
    for noftz in (true, false)
        target = CuArray(fill(seed, 4))
        @cuda threads=32 _bind_bulk_reduce_runtime!(target, value, Val(noftz))
        CUDACore.synchronize()
        @test reinterpret.(UInt32, Array(target)) == fill(0x00000003, 4)
    end
end
