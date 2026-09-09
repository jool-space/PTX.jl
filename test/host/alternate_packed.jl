# PTX ISA 9.4 §§9.7.6.1–4 and §5.2.5.1: compact FP4 occupies b16;
# padded FP4, FP6, FP8, and UE8M0 occupy b32. All results are packed FP8.
@testset "alternate packed arithmetic preserves compact and padded carriers" begin
    cases = (
        (ptx"add.rn.e4m3x4.e2m1x4", (UInt16, UInt32), "=r,h,r"),
        (ptx"sub.satfinite.e5m2x4.e2m1p4x4", (UInt32, UInt32), "=r,r,r"),
        (ptx"mul.rn.satfinite.e4m3x4.e2m1x4.e2m1x4", (UInt16, UInt16), "=r,h,h"),
        (ptx"mul.e5m2x4.e2m1p4x4.ue8m0x4", (UInt32, UInt32), "=r,r,r"),
        (ptx"fma.e4m3x4.e2m1x4.e3m2x4", (UInt16, UInt32, UInt32), "=r,h,r,r"),
        (ptx"fma.rn.e4m3x4.e3m2x4.e2m1x4.satfinite",
         (UInt32, UInt16, UInt32), "=r,r,h,r"),
    )
    for (op, args, constraints) in cases
        name, mods = typeof(op).parameters
        @test only(Base.code_typed(op, args))[2] === UInt32
        for raw in (false, true)
            spec = PTX.build_call(name, mods, args; raw)
            @test spec.rettype === UInt32
            @test spec.constraints == constraints * (raw ? ",~{memory}" : "")
            @test spec.side_effects == raw
            @test spec.convergent == raw
            for i in eachindex(args)
                bad = ntuple(j -> j == i ? (args[j] === UInt16 ? UInt32 : UInt16) :
                                          args[j], length(args))
                @test_throws ArgumentError PTX.build_call(name, mods, bad; raw)
            end
            @test_throws ArgumentError PTX.build_call(name, mods, args[1:end-1]; raw)
        end
    end
end

@testset "alternate packed grammar misses are closed, including raw calls" begin
    for (op, mods) in (
        (:add, (:e4m3x4,)),
        (:add, (:rn, :e2m1x4, :e4m3x4)),
        (:add, (:rz, :e4m3x4, :e5m2x4)),
        (:add, (:rn, :sat, :e4m3x4, :e5m2x4)),
        (:sub, (:ftz, :e4m3x4, :e2m1x4)),
        (:sub, (:satfinite, :rn, :e5m2x4, :ue8m0x4)),
        (:mul, (:rn, :e4m3x4, :e2m1p4x4)),
        (:mul, (:rn, :rn, :e4m3x4, :e4m3x4, :e5m2x4)),
        (:fma, (:rn, :e4m3x4, :e2m1x4)),
        (:fma, (:rn, :e4m3x4, :e2m1x4, :e5m2x4, :e4m3x4)),
        # Undocumented postfix permutations do not inherit the seven exact
        # example compatibility entries, even when ptxas accepts them.
        (:add, (:rn, :e4m3x4, :e5m2x4, :satfinite)),
        (:mul, (:rn, :satfinite, :e5m2x4, :e3m2x4, :e2m1x4, :satfinite)),
        (:fma, (:e4m3x4, :e3m2x4, :e2m1x4, :satfinite)),
    )
        @test PTX.island_of(op, mods) === PTX.ScalarLedger()
        @test_throws ArgumentError PTX.infer_rettype(op, mods)
        for raw in (false, true)
            @test_throws ArgumentError PTX.build_call(op, mods, (); raw)
        end
    end
end

function _alternate_transpile_module(body)
    """
    .version 9.4
    .target sm_100a
    .address_size 64
    .visible .entry alternate_packed() {
        .reg .b16 %h<4>;
        .reg .b32 %r<4>;
        .reg .b64 %rd;
        .reg .f32 %f;
        $body
        ret;
    }
    """
end

@testset "alternate packed transpilation fixes each operand width" begin
    out = PTX.ptx_to_julia(_alternate_transpile_module("""
        add.rn.e4m3x4.e2m1x4 %r0, 0x2222, 0x38383838;
        mul.rn.satfinite.e5m2x4.e2m1x4.e2m1p4x4 %r1, 0x2222, 0x02020202;
        fma.e4m3x4.e2m1x4.e2m1x4 %r2, 0x2222, 0x2222, %r0;
        sub.e5m2x4.ue8m0x4 %f, %r1, %r2;
    """))
    @test occursin("r0 = ptx\"add.rn.e4m3x4.e2m1x4\"(UInt16(0x2222), UInt32(0x38383838))", out)
    @test occursin("r1 = ptx\"mul.rn.satfinite.e5m2x4.e2m1x4.e2m1p4x4\"(UInt16(0x2222), UInt32(0x02020202))", out)
    @test occursin("r2 = ptx\"fma.e4m3x4.e2m1x4.e2m1x4\"(UInt16(0x2222), UInt16(0x2222), r0)", out)
    @test occursin("f = ptx\"sub.e5m2x4.ue8m0x4\"(r1, r2)", out)

    for body in (
        "add.rn.e4m3x4.e2m1x4 %r0, %r1, %r2;",
        "add.rn.e4m3x4.e2m1p4x4 %r0, %h0, %r2;",
        "sub.rn.e4m3x4.e5m2x4 %h0, %r0, %r1;",
        "mul.e5m2x4.e2m1x4.e2m1x4 %r0, %h0, %r1;",
        "fma.rn.e4m3x4.e2m1x4.e3m2x4 %r0, %h0, %r1, %h1;",
        "fma.rn.e4m3x4.e2m1x4.e3m2x4 %rd, %h0, %r1, %r2;",
        "add.rn.e4m3x4.e2m1x4 %r0, %h0;",
        "add.rn.e4m3x4 %r0, %r1, %r2;",
    )
        @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(
            _alternate_transpile_module(body))
    end
end
