# Shared numeric probe: each scalar load keeps its value live in global memory.
const _GA_LOAD_TYPES = (
    (:b8, UInt8), (:b16, UInt16), (:b32, UInt32), (:b64, UInt64),
    (:u8, UInt8), (:u16, UInt16), (:u32, UInt32), (:u64, UInt64),
    (:s8, Int8), (:s16, Int16), (:s32, Int32), (:s64, Int64),
    (:f32, Float32), (:f64, Float64),
)

@generated function _ga_readonly_loads!(out, input)
    body = Expr(:block, :(global_ptr = pointer(input)),
        :(generic_ptr = PTX.reinterpret_addrspace(Val(PTX.AS.Generic), global_ptr)))
    i = 0
    for raw in (false, true), space in ((), (:global,)), (kind, T) in _GA_LOAD_TYPES
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
