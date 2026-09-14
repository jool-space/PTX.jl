# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc>=9.0
#
# PTX ISA 9.4 §9.7.10.8 `ld.proxy::readonly`: every scalar type on the
# generic and global spaces, through the exact and raw wrappers. The offline
# tier assembles the kernel at its sm_90 floor and pins the rejection below
# it; eligible devices check that every load keeps its value bits live in
# global memory.

const _READONLY_LOAD_TYPES = (
    (:b8, UInt8), (:b16, UInt16), (:b32, UInt32), (:b64, UInt64),
    (:u8, UInt8), (:u16, UInt16), (:u32, UInt32), (:u64, UInt64),
    (:s8, Int8), (:s16, Int16), (:s32, Int32), (:s64, Int64),
    (:f32, Float32), (:f64, Float64),
)

@generated function _readonly_loads!(out, input)
    body = Expr(:block, :(global_ptr = pointer(input)),
        :(generic_ptr = PTX.reinterpret_addrspace(Val(PTX.AS.Generic), global_ptr)))
    i = 0
    for raw in (false, true), space in ((), (:global,)), (kind, T) in _READONLY_LOAD_TYPES
        i += 1
        mods = (space..., kind, Symbol("proxy::readonly"))
        op = raw ? PTX.RawOperation : PTX.Operation
        ptr = isempty(space) ? :generic_ptr : :global_ptr
        bits = unsigned(T === Float32 ? UInt32 : T === Float64 ? UInt64 : T)
        call = :($op{:ld, $mods}()($ptr))
        push!(body.args, :(Base.@inbounds out[$i] = UInt64(reinterpret($bits, $call))))
    end
    push!(body.args, :(return nothing))
    body
end

const _READONLY_LOAD_ARGS = Tuple{CuDeviceVector{UInt64, 1}, CuDeviceVector{UInt64, 1}}

@testset "readonly loads assemble at sm_90 and are rejected below" begin
    if _ptxas_isa() < v"9.4"
        @test_skip "PTX 9.4 assembler required"
    else
        @test ptxas_compiles(_readonly_loads!, _READONLY_LOAD_ARGS; cap = v"9.0")
        ptx = emit_ptx(_readonly_loads!, _READONLY_LOAD_ARGS; cap = v"9.0")
        for space in ("", "global."), (kind, _) in _READONLY_LOAD_TYPES
            @test occursin("ld.$space$kind.proxy::readonly", ptx)
        end
        @test_throws ErrorException ptxas_compiles(_readonly_loads!,
                                                   _READONLY_LOAD_ARGS; cap = v"8.0")
    end
end

if test_runtime_supported(@__FILE__)
    @testset "readonly scalar loads preserve their result bits" begin
        if _ptxas_isa() < v"9.4"
            @test_skip "PTX 9.4 assembler required"
        else
            value = UInt64(0x3ff00000ff80ff81)
            input = CuArray([value])
            out = CUDACore.zeros(UInt64, 56)
            @cuda threads=1 _readonly_loads!(out, input)
            CUDACore.synchronize()
            expected = UInt64[
                0x81, 0xff81, 0xff80ff81, value,
                0x81, 0xff81, 0xff80ff81, value,
                0x81, 0xff81, 0xff80ff81, value,
                0xff80ff81, value,
            ]
            @test Array(out) == repeat(expected, 4)
        end
    end
end
