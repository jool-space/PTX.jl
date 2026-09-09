@testset "set result ABI and predicate operands" begin
    cases = (
        (ptx"set.lt.f32.s64", (Int64, Int64), Float32, "=f,l,l"),
        (ptx"set.eq.u32.b64", (UInt64, UInt64), UInt32, "=r,l,l"),
        (ptx"set.lt.and.s32.f32", (Float32, Float32, Bool), Int32, "=r,f,f,b"),
        (ptx"set.lt.f16.f32", (Float32, Float32), Float16, "=h,f,f"),
        (ptx"set.lt.bf16.f64", (Float64, Float64), UInt16, "=h,d,d"),
        (ptx"set.lt.u16.f16", (Float16, Float16), UInt16, "=h,h,h"),
        (ptx"set.eq.xor.s16.bf16", (UInt16, UInt16, Bool), Int16, "=h,h,h,b"),
        (ptx"set.lt.f16x2.f16x2", (UInt32, UInt32), UInt32, "=r,r,r"),
        (ptx"set.lt.u32.bf16x2", (UInt32, UInt32), UInt32, "=r,r,r"),
        (ptx"set.lt.s32.bf16x2", (UInt32, UInt32), Int32, "=r,r,r"),
        (ptx"set.ge.s16x2", (UInt32, UInt32), UInt32, "=r,r,r"),
        (ptx"set.lo.u8x4", (UInt32, UInt32), UInt32, "=r,r,r"),
    )
    for (op, args, result, constraints) in cases
        name, mods = typeof(op).parameters
        @test only(Base.code_typed(op, args))[2] === result
        for raw in (false, true)
            spec = PTX.build_call(name, mods, args; raw)
            @test spec.rettype === result
            @test spec.constraints == constraints * (raw ? ",~{memory}" : "")
            @test spec.side_effects == raw
            @test spec.convergent == raw
            @test_throws ArgumentError PTX.build_call(name, mods, (args..., Bool); raw)
            @test_throws ArgumentError PTX.build_call(name, mods, args[1:end-1]; raw)
        end
    end
    @test PTX.form_contract(:set, (:lt, :f32, :s64)).effects === :pure
    @test_throws ArgumentError PTX.build_call(:set, (:lt, :and, :u32, :s32),
                                              (Int32, Int32, UInt32))
    @test_throws ArgumentError PTX.build_call(:set, (:eq, :u32, :b64), (UInt32, UInt32))
end

@testset "set grammar misses cannot fall back, including raw" begin
    for mods in (
        (), (:lt, :u32), (:eq, :f64, :f32), (:lt, :b32, :s32),
        (:lt, :b32), (:lt, :u32, :b32), (:lo, :u32, :s32),
        (:nan, :u32, :s32), (:eq, :ftz, :u32, :s64),
        (:eq, :ftz, :f16, :s32), (:eq, :ftz, :f16, :f64),
        (:eq, :ftz, :bf16, :f16), (:lo, :f16, :u32),
        (:eq, :and, :ftz, :u32, :bf16x2), (:eq, :f16, :bf16),
        (:lt, :and, :s8x4), (:lo, :s8x4), (:nan, :u8x4),
        (:lt, :ftz, :u16x2), (:lt, :sat, :u8x4), (:lt, :u32, :s8x4),
        (:and, :lt, :u32, :s32), (:lt, :u32, :s32, :and),
    )
        @test PTX.island_of(:set, mods) === PTX.ScalarLedger()
        @test_throws ArgumentError PTX.infer_rettype(:set, mods)
        for raw in (false, true)
            @test_throws ArgumentError PTX.build_call(:set, mods, (); raw)
        end
    end
end

_set_module(body) = """
.version 9.4
.target sm_107f
.address_size 64
.visible .entry set_probe() {
    .reg .b16 %h<4>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<4>;
    .reg .f32 %f;
    .reg .pred %p;
    $body
    ret;
}
"""

@testset "set transpilation preserves result type, lanes, and predicate source" begin
    out = PTX.ptx_to_julia(_set_module("""
        set.lt.and.f32.s64 %f, %rd0, -1, !%p;
        set.eq.or.u32.b32 %r0, %r1, 0, 1;
        set.lt.u16.f16 %h0, %h1, %h2;
        set.lt.u32.bf16x2 %r1, %r2, %r3;
        set.ge.s16x2 %r2, %r0, %r1;
    """))
    @test occursin("f = ptx\"set.lt.and.f32.s64\"(rd0, Int64(-1), !p)", out)
    @test occursin("r0 = ptx\"set.eq.or.u32.b32\"(r1, UInt32(0x00000000), true)", out)
    @test occursin("h0 = ptx\"set.lt.u16.f16\"(h1, h2)", out)
    @test occursin("r1 = ptx\"set.lt.u32.bf16x2\"(r2, r3)", out)
    @test occursin("r2 = ptx\"set.ge.s16x2\"(r0, r1)", out)
    for body in (
        "set.lt.f32.s64 %rd0, %rd1, %rd2;",
        "set.lt.u16.f16 %r0, %h0, %h1;",
        "set.lt.u32.bf16x2 %r0, %h0, %h1;",
        "set.lt.and.u32.s32 %r0, %r1, %r2, !%r3;",
        "set.lt.and.u32.s32 %r0, %r1, %r2, {%p};",
        "set.lt.u32.f16 %r0, 0, %h0;",
        "set.lt.u32.f16 %r0, %h0, 0f00000000;",
        "set.lt.u32.s32 %r0, %r1, !%r2;",
        "set.ge.s16x2 %h0, %r0, %r1;",
        "set.eq.u32.b64 %r0, %r1, %r2;",
        "set.lt.and.u32.s32 %r0, %r1, %r2;",
    )
        @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(_set_module(body))
    end
end

@testset "f16 set uses an integer mask before forming half one" begin
    for raw in (false, true)
        spec = PTX.build_call(:set, (:lt, :and, :ftz, :f16, :f16),
                              (Float16, Float16, Bool); raw)
        @test spec.rettype === Float16
        @test spec.asm == "{ .reg .b32 set_mask; set.lt.and.ftz.u32.f16 set_mask, \$1, \$2, \$3; " *
                          "and.b32 set_mask, set_mask, 15360; cvt.u16.u32 \$0, set_mask; }"
        @test spec.constraints == "=h,h,h,b" * (raw ? ",~{memory}" : "")
    end
end
