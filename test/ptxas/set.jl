# Every admitted set form keeps a stored result, partitioned by the ISA floor.
_set_partition(s) = s.ptx_version == v"9.4" ? :packed :
                    s.min_sm == v"9.0" ? :bfloat :
                    s.min_sm == v"5.3" ? :half : :general

@generated function _set_all!(out::CuDeviceVector{UInt32,1}, ::Val{P},
                              u16::UInt16, u32::UInt32, u64::UInt64,
                              f16::Float16, f32::Float32, f64::Float64,
                              gate::Bool) where {P}
    names = Dict(:b16 => :u16, :u16 => :u16, :s16 => :u16, :bf16 => :u16,
                 :b32 => :u32, :u32 => :u32, :s32 => :u32,
                 :b64 => :u64, :u64 => :u64, :s64 => :u64,
                 :f16 => :f16, :f32 => :f32, :f64 => :f64, :pred => :gate)
    body = Expr(:block)
    i = 0
    for s in PTX.SCALAR_RESULT_SCHEMAS
        s.op === :set && _set_partition(s) === P || continue
        i += 1
        args = [names[kind] for kind in s.operands]
        call = :(PTX.Operation{:set, $(s.mods)}()($(args...)))
        bits = sizeof(s.rettype) == 2 ? UInt16 : UInt32
        push!(body.args, :(Base.@inbounds out[$i] = UInt32(reinterpret($bits, $call))))
    end
    push!(body.args, :(return nothing))
    body
end

_set_types(p) = Tuple{CuDeviceVector{UInt32,1}, Val{p}, UInt16, UInt32, UInt64,
                      Float16, Float32, Float64, Bool}

@testset "every scalar and packed set form assembles" begin
    all_set = filter(s -> s.op === :set, PTX.SCALAR_RESULT_SCHEMAS)
    @test length(all_set) == 3144
    for (partition, count, cap, feature_set) in (
        (:general, 1152, v"7.5", :baseline),
        (:half, 1232, v"7.5", :baseline),
        (:bfloat, 728, v"9.0", :baseline),
        (:packed, 32, v"10.7", :family),
    )
        schemas = filter(s -> _set_partition(s) === partition, all_set)
        @test length(schemas) == count
        if partition === :packed && _ptxas_isa() < v"9.4"
            @test_skip "PTX 9.4 assembler required for packed integer set"
            continue
        end
        types = _set_types(partition)
        @test ptxas_compiles(_set_all!, types; cap, feature_set)
        emitted = emit_ptx(_set_all!, types; cap, feature_set)
        for s in schemas
            mods = s.rettype === Float16 ?
                   (s.mods[1:end-2]..., :u32, s.mods[end]) : s.mods
            @test occursin(PTX.build_head(s.op, mods), emitted)
        end
    end
end

@testset "packed set requires the sm_107 family feature" begin
    if _ptxas_isa() < v"9.4"
        @test_skip "PTX 9.4 assembler required for packed integer set"
    else
        types = _set_types(:packed)
        @test ptxas_compiles(_set_all!, types; cap = v"10.7", feature_set = :arch)
        @test occursin(".target sm_107a",
                       emit_ptx(_set_all!, types; cap = v"10.7", feature_set = :arch))
        for (cap, feature_set, target) in ((v"10.7", :baseline, "sm_107"),
                                           (v"10.0", :family, "sm_100f"),
                                           (v"12.0", :family, "sm_120f"))
            err = try
                ptxas_compiles(_set_all!, types; cap, feature_set)
                nothing
            catch caught
                caught
            end
            @test err isa ErrorException
            @test occursin("not supported on .target '$target'", sprint(showerror, err))
        end
    end
end
