# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc>=9.0
#
# PTX ISA 9.4 non-flushing f32 adds: `atom/red.add.noftz.f32` (§9.7.15.5)
# preserve subnormal inputs and results (the flush is the documented
# behaviour of the plain `atom.add.f32`), and the f32 add of
# cp.reduce.async.bulk is non-flushing by default with `.noftz` as the
# explicit spelling. The `prefetch.L1::32B.valid_addr` hints ride along in
# the same sm_90 chain. The offline tier assembles the chain at sm_90 and
# pins the rejection at sm_80; eligible devices check the subnormal
# arithmetic.

using PTX: smem_addr_u32

function _noftz_chain!(out::CuDeviceVector{Float32, 1},
                       g::Core.LLVMPtr{Float32, PTX.AS.Global},
                       value::Float32)
    generic = PTX.reinterpret_addrspace(Val(PTX.AS.Generic), g)
    ptx"prefetch.global.L1::32B.valid_addr"(g)
    ptx"prefetch.L1::32B.valid_addr"(generic)
    old_global = ptx"atom.global.add.noftz.f32"(g, value)
    old_generic = ptx"atom.add.noftz.f32"(generic, value)
    ptx"red.global.add.noftz.f32"(g, value)
    ptx"red.add.noftz.f32"(generic, value)
    @inbounds out[1] = old_global
    @inbounds out[2] = old_generic
    return nothing
end

const _NOFTZ_CHAIN_TYPES = Tuple{CuDeviceVector{Float32, 1},
                                 Core.LLVMPtr{Float32, PTX.AS.Global},
                                 Float32}

@testset "prefetch .L1::32B.valid_addr and .add.noftz.f32 assemble on sm_90" begin
    if _ptxas_isa() < v"9.4"
        @test_skip "PTX 9.4 assembler required"
    else
        @test ptxas_compiles(_noftz_chain!, _NOFTZ_CHAIN_TYPES; cap = v"9.0")
        ptx = emit_ptx(_noftz_chain!, _NOFTZ_CHAIN_TYPES; cap = v"9.0")
        for head in ("prefetch.global.L1::32B.valid_addr", "prefetch.L1::32B.valid_addr",
                     "atom.global.add.noftz.f32", "atom.add.noftz.f32",
                     "red.global.add.noftz.f32", "red.add.noftz.f32")
            @test occursin(head * " ", ptx)
        end
        @test ptxas_rejects(_noftz_chain!, _NOFTZ_CHAIN_TYPES; cap = v"8.0",
                            target = "sm_80")
    end
end

# Device-side wrappers: `pointer` on the device array yields the global
# address-space pointer the chain takes.
function _noftz_runtime!(out::CuDeviceVector{Float32, 1},
                         target::CuDeviceVector{Float32, 1},
                         value::Float32)
    _noftz_chain!(out, pointer(target), value)
    return nothing
end

function _flushing_atom!(out::CuDeviceVector{Float32, 1},
                         target::CuDeviceVector{Float32, 1},
                         value::Float32)
    @inbounds out[1] = ptx"atom.global.add.f32"(pointer(target), value)
    return nothing
end

# Bulk reduction into global memory, both spellings checked against subnormal
# operands. The 16-byte source is carved out of a static shared array at a
# 16-byte-aligned offset.
@generated function _bulk_reduce_runtime!(target::CuDeviceVector{Float32, 1},
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

if test_runtime_supported(@__FILE__)
    @testset "atom/red .add.noftz.f32 preserve subnormals" begin
        seed = reinterpret(Float32, 0x00000001)
        value = reinterpret(Float32, 0x00000002)
        target = CuArray([seed])
        out = CUDACore.zeros(Float32, 2)
        @cuda threads=1 _noftz_runtime!(out, target, value)
        CUDACore.synchronize()
        got = Array(out)
        # Two atomics return the running sum; two reductions add silently.
        @test reinterpret(UInt32, got[1]) == 0x00000001
        @test reinterpret(UInt32, got[2]) == 0x00000003
        @test reinterpret(UInt32, Array(target)[1]) == 0x00000009

        # The non-.noftz form flushes both the subnormal input and result.
        flushed = CuArray([seed])
        @cuda threads=1 _flushing_atom!(out, flushed, value)
        CUDACore.synchronize()
        @test reinterpret(UInt32, Array(flushed)[1]) == 0x00000000
    end

    @testset "cp.reduce.async.bulk f32 add preserves subnormals" begin
        seed = reinterpret(Float32, 0x00000001)
        value = reinterpret(Float32, 0x00000002)
        for noftz in (true, false)
            target = CuArray(fill(seed, 4))
            @cuda threads=32 _bulk_reduce_runtime!(target, value, Val(noftz))
            CUDACore.synchronize()
            @test reinterpret.(UInt32, Array(target)) == fill(0x00000003, 4)
        end
    end
end
