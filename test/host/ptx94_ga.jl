@testset "readonly load has a result before its trailing proxy" begin
    types = (
        (:b8, UInt8), (:b16, UInt16), (:b32, UInt32), (:b64, UInt64),
        (:u8, UInt8), (:u16, UInt16), (:u32, UInt32), (:u64, UInt64),
        (:s8, Int8), (:s16, Int16), (:s32, Int32), (:s64, Int64),
        (:f32, Float32), (:f64, Float64),
    )
    for (kind, T) in types, raw in (false, true),
        (space, as) in (((), PTX.AS.Generic), ((:global,), PTX.AS.Global))
        mods = (space..., kind, Symbol("proxy::readonly"))
        spec = PTX.build_call(:ld, mods, (Core.LLVMPtr{UInt64,as},); raw)
        @test spec.rettype === T
        @test spec.asm == "ld." * join(mods, ".") * " \$0, [\$1];"
        @test startswith(spec.constraints, "=")
        @test spec.side_effects
    end
    for mods in ((Symbol("proxy::readonly"),),
                 (:global, :f16, Symbol("proxy::readonly")),
                 (:global, :pred, Symbol("proxy::readonly")))
        @test_throws ArgumentError PTX.infer_rettype(:ld, mods)
    end
end

@testset "stochastic cvt does not admit pzo" begin
    for dst in (:f16x2, :bf16x2, :e4m3x4, :e5m2x4, :e2m1x4, :e2m3x4, :e3m2x4)
        half = dst in (:f16x2, :bf16x2)
        args = half ? (Float32, Float32, UInt32) : (NTuple{4,Float32}, UInt32)
        prefix = half ? (:rs,) : (:rs, :satfinite)
        for raw in (false, true)
            @test PTX.build_call(:cvt, (prefix..., dst, :f32), args; raw).rettype !== Nothing
            @test_throws ArgumentError PTX.build_call(
                :cvt, (prefix..., :pzo, dst, :f32), args; raw)
        end
    end
    source = """
    .version 9.4
    .target sm_107f
    .address_size 64
    .entry invalid_pzo() {
        .reg .b32 r, random;
        .reg .f32 a, b;
        cvt.rs.pzo.f16x2.f32 r, a, b, random;
        ret;
    }
    """
    @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(source)
end

@testset "GA proxy fences retain compiler memory ordering" begin
    for mods in (
        (:proxy, :alias, :acquire, :sys),
        (:proxy, :alias, :release, :sys),
        (:proxy, Symbol("async::generic"), :release,
         Symbol("sync_restrict::shared::cluster::read"), :cluster),
    )
        spec = PTX.build_call(:fence, mods, ())
        @test spec.rettype === Nothing
        @test spec.asm == "fence." * join(mods, ".") * ";"
        @test spec.side_effects
        @test occursin("~{memory}", spec.constraints)
        @test !spec.convergent
    end
end
