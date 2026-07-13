# Independent oracle for the audited fixed-scalar-result surface.  This test
# deliberately reconstructs the PTX grammar and result/operand facts instead
# of consuming SCALAR_RESULT_SCHEMAS to decide what should exist.

const _EXPECTED_SCALAR_SECTIONS = Dict(
    :mixed_add =>
        "ptx/9-instruction-set/9.7.5.1-mixed-precision-floating-point-instructions-add.md",
    :mixed_sub =>
        "ptx/9-instruction-set/9.7.5.2-mixed-precision-floating-point-instructions-sub.md",
    :mixed_fma =>
        "ptx/9-instruction-set/9.7.5.3-mixed-precision-floating-point-instructions-fma.md",
    :popc =>
        "ptx/9-instruction-set/9.7.1.15-integer-arithmetic-instructions-popc.md",
    :clz =>
        "ptx/9-instruction-set/9.7.1.16-integer-arithmetic-instructions-clz.md",
    :dp4a =>
        "ptx/9-instruction-set/9.7.1.24-integer-arithmetic-instructions-dp4a.md",
    :dp2a =>
        "ptx/9-instruction-set/9.7.1.25-integer-arithmetic-instructions-dp2a.md",
    :mul =>
        "ptx/9-instruction-set/9.7.1.3-integer-arithmetic-instructions-mul.md",
    :mad =>
        "ptx/9-instruction-set/9.7.1.4-integer-arithmetic-instructions-mad.md",
    :prmt =>
        "ptx/9-instruction-set/9.7.9.7-data-movement-and-conversion-instructions-prmt.md",
    :packed_add =>
        "ptx/9-instruction-set/9.7.1.1-integer-arithmetic-instructions-add.md",
    :packed_sub =>
        "ptx/9-instruction-set/9.7.1.2-integer-arithmetic-instructions-sub.md",
    :neg =>
        "ptx/9-instruction-set/9.7.1.12-integer-arithmetic-instructions-neg.md",
    :min =>
        "ptx/9-instruction-set/9.7.1.13-integer-arithmetic-instructions-min.md",
    :max =>
        "ptx/9-instruction-set/9.7.1.14-integer-arithmetic-instructions-max.md",
    :cvt =>
        "ptx/9-instruction-set/9.7.9.23-data-movement-and-conversion-instructions-cvt.pack.md",
)

function _expected_scalar_section(op, mods)
    if op in (:add, :sub, :fma) && :f32 in mods &&
       any(t -> t in (:f16, :bf16), mods)
        return _EXPECTED_SCALAR_SECTIONS[Symbol(:mixed_, op)]
    elseif op === :add
        return _EXPECTED_SCALAR_SECTIONS[:packed_add]
    elseif op === :sub
        return _EXPECTED_SCALAR_SECTIONS[:packed_sub]
    end
    _EXPECTED_SCALAR_SECTIONS[op]
end

function _expected_scalar_result_forms()
    expected = Dict{Tuple{Symbol,Tuple},NamedTuple}()
    add!(op, mods, rettype, operands, ptx_version, min_sm;
         feature_set = :baseline, provenance = :isa) = begin
        key = (op, mods)
        @assert !haskey(expected, key)
        expected[key] = (;
            rettype, operands, ptx_version, min_sm, feature_set, provenance,
            section = _expected_scalar_section(op, mods))
    end

    for op in (:add, :sub), atype in (:f16, :bf16),
        rnd in (nothing, :rn, :rz, :rm, :rp),
        prefix_sat in (false, true)
        prefix = rnd === nothing ? () : (rnd,)
        prefix_sat && (prefix = (prefix..., :sat))
        add!(op, (prefix..., :f32, atype), Float32, (atype, :f32),
             v"8.6", v"10.0")
    end
    for atype in (:f16, :bf16), rnd in (:rn, :rz, :rm, :rp),
        prefix_sat in (false, true)
        prefix = prefix_sat ? (rnd, :sat) : (rnd,)
        add!(:fma, (prefix..., :f32, atype), Float32,
             (atype, atype, :f32), v"8.6", v"10.0")
    end
    for (op, mods, operands) in (
        (:add, (:rz, :f32, :bf16, :sat), (:bf16, :f32)),
        (:sub, (:rz, :f32, :f16, :sat), (:f16, :f32)),
        (:fma, (:rz, :sat, :f32, :f16, :sat), (:f16, :f16, :f32)),
    )
        add!(op, mods, Float32, operands, v"8.6", v"10.0";
             provenance = :ptxas_compat)
    end
    for op in (:popc, :clz), width in (:b32, :b64)
        add!(op, (width,), UInt32, (width,), v"2.0", v"2.0")
    end
    for atype in (:u32, :s32), btype in (:u32, :s32)
        result = atype === :u32 && btype === :u32 ? UInt32 : Int32
        result_kind = result === UInt32 ? :u32 : :s32
        add!(:dp4a, (atype, btype), result,
             (atype, btype, result_kind), v"5.0", v"6.1")
        for mode in (:lo, :hi)
            add!(:dp2a, (mode, atype, btype), result,
                 (atype, btype, result_kind), v"5.0", v"6.1")
        end
    end
    for (itype, result, result_kind) in (
        (:u16, UInt32, :u32), (:s16, Int32, :s32),
        (:u32, UInt64, :u64), (:s32, Int64, :s64),
    )
        add!(:mul, (:wide, itype), result, (itype, itype), v"1.0", nothing)
        add!(:mad, (:wide, itype), result, (itype, itype, result_kind),
             v"1.0", nothing)
    end
    for mode in (:f4e, :b4e, :rc8, :ecl, :ecr, :rc16)
        add!(:prmt, (:b32, mode), UInt32, (:b32, :b32, :b32),
             v"2.0", v"2.0")
    end
    for packed_type in (:u16x2, :s16x2)
        add!(:add, (packed_type,), UInt32, (:b32, :b32), v"8.0", v"9.0")
        add!(:add, (:sat, packed_type), UInt32, (:b32, :b32),
             v"9.2", v"12.0"; feature_set = :family)
    end
    for packed_type in (:u8x4, :s8x4), sat in (false, true)
        mods = sat ? (:sat, packed_type) : (packed_type,)
        add!(:add, mods, UInt32, (:b32, :b32), v"9.2", v"12.0";
             feature_set = :family)
        add!(:sub, mods, UInt32, (:b32, :b32), v"9.2", v"12.0";
             feature_set = :family)
    end
    add!(:add, (:s8x4, :sat), UInt32, (:b32, :b32), v"9.2", v"12.0";
         feature_set = :family, provenance = :ptxas_compat)
    add!(:neg, (:s8x4,), UInt32, (:b32,), v"9.2", v"12.0";
         feature_set = :family)
    for op in (:min, :max)
        add!(op, (:u16x2,), UInt32, (:b32, :b32), v"8.0", v"9.0")
        for mods in ((:s16x2,), (:relu, :s16x2))
            add!(op, mods, UInt32, (:b32, :b32), v"8.0", v"9.0")
        end
        add!(op, (:u8x4,), UInt32, (:b32, :b32), v"9.2", v"12.0";
             feature_set = :family)
        for mods in ((:s8x4,), (:relu, :s8x4))
            add!(op, mods, UInt32, (:b32, :b32), v"9.2", v"12.0";
                 feature_set = :family)
        end
        add!(op, (:relu, :s32), Int32, (:s32, :s32),
             v"8.0", v"9.0")
    end
    add!(:min, (:s16x2, :relu), UInt32, (:b32, :b32),
         v"8.0", v"9.0"; provenance = :ptxas_compat)
    for convert_type in (:u16, :s16)
        add!(:cvt, (:pack, :sat, convert_type, :s32), UInt32,
             (:s32, :s32), v"6.5", v"7.2")
    end
    for convert_type in (:u8, :s8, :u4, :s4, :u2, :s2)
        min_sm = convert_type in (:u8, :s8) ? v"7.2" : v"7.5"
        add!(:cvt, (:pack, :sat, convert_type, :s32, :b32), UInt32,
             (:s32, :s32, :b32), v"6.5", min_sm)
    end
    expected
end

_scalar_test_type(kind) =
    kind === :f16  ? Float16 :
    kind === :bf16 ? UInt16 :
    kind === :f32  ? Float32 :
    kind === :u16  ? UInt16 :
    kind === :s16  ? Int16 :
    kind === :u32  ? UInt32 :
    kind === :s32  ? Int32 :
    kind === :u64  ? UInt64 :
    kind === :s64  ? Int64 :
    kind === :b32  ? UInt32 :
    kind === :b64  ? UInt64 :
    error("unknown test operand kind $kind")

_scalar_test_letter(kind) =
    kind in (:f16, :bf16, :u16, :s16) ? "h" :
    kind === :f32 ? "f" :
    kind in (:u32, :s32, :b32) ? "r" :
    kind in (:u64, :s64, :b64) ? "l" :
    error("unknown test operand kind $kind")

@testset "audited scalar result schemas" begin
    expected = _expected_scalar_result_forms()
    actual = Dict((schema.op, schema.mods) => schema
                  for schema in PTX.SCALAR_RESULT_SCHEMAS)

    @test length(expected) == 126
    @test length(actual) == 126
    @test Set(keys(actual)) == Set(keys(expected))
    @test count(key -> key[1] === :add, keys(actual)) == 30
    @test count(key -> key[1] === :sub, keys(actual)) == 25
    @test count(key -> key[1] === :fma, keys(actual)) == 17
    @test count(key -> key[1] in (:popc, :clz), keys(actual)) == 4
    @test count(key -> key[1] === :dp4a, keys(actual)) == 4
    @test count(key -> key[1] === :dp2a, keys(actual)) == 8
    @test count(key -> key[1] === :mul, keys(actual)) == 4
    @test count(key -> key[1] === :mad, keys(actual)) == 4
    @test count(key -> key[1] === :cvt, keys(actual)) == 8
    @test count(key -> key[1] === :prmt, keys(actual)) == 6
    @test count(key -> key[1] === :neg, keys(actual)) == 1
    @test count(key -> key[1] === :min, keys(actual)) == 8
    @test count(key -> key[1] === :max, keys(actual)) == 7
    @test count(schema -> schema.provenance === :isa, values(actual)) == 121
    @test count(schema -> schema.provenance === :ptxas_compat,
                values(actual)) == 5
    @test all(schema -> schema.provenance in (:isa, :ptxas_compat),
              values(actual))
    compat_keys = Set([
        (:add, (:rz, :f32, :bf16, :sat)),
        (:sub, (:rz, :f32, :f16, :sat)),
        (:fma, (:rz, :sat, :f32, :f16, :sat)),
        (:add, (:s8x4, :sat)),
        (:min, (:s16x2, :relu)),
    ])
    @test Set(key for (key, schema) in actual
              if schema.provenance === :ptxas_compat) == compat_keys

    for (key, want) in expected
        schema = actual[key]
        @test schema.rettype === want.rettype
        @test schema.operands == want.operands
        @test schema.ptx_version == want.ptx_version
        @test schema.min_sm == want.min_sm
        @test schema.feature_set === want.feature_set
        @test schema.provenance === want.provenance
        @test schema.section == want.section

        op, mods = key
        argtypes = Tuple(_scalar_test_type(kind) for kind in want.operands)
        spec = PTX.build_call(op, mods, argtypes)
        head = string(op, ".", join(mods, "."))
        operands = join(("\$" * string(i) for i in 0:length(argtypes)), ", ")
        constraints = join(["=" * PTX.constraint_letter(want.rettype);
                            _scalar_test_letter.(want.operands)...], ",")
        @test PTX.infer_rettype(op, mods) === want.rettype
        @test spec.rettype === want.rettype
        @test spec.asm == "$head $operands;"
        @test spec.constraints == constraints
        @test spec.side_effects == false
        @test spec.convergent == false
        @test PTX.lowering(PTX.Operation{op, mods}(), argtypes).rettype === want.rettype
    end
end

@testset "fixed scalar results and representative assembly" begin
    cases = [
        (ptx"add.rz.sat.f32.bf16", (UInt16, Float32), Float32,
         "add.rz.sat.f32.bf16 \$0, \$1, \$2;", "=f,h,f"),
        (ptx"sub.f32.f16", (Float16, Float32), Float32,
         "sub.f32.f16 \$0, \$1, \$2;", "=f,h,f"),
        (ptx"fma.rn.f32.f16", (Float16, Float16, Float32), Float32,
         "fma.rn.f32.f16 \$0, \$1, \$2, \$3;", "=f,h,h,f"),
        (ptx"add.rz.f32.bf16.sat", (UInt16, Float32), Float32,
         "add.rz.f32.bf16.sat \$0, \$1, \$2;", "=f,h,f"),
        (ptx"sub.rz.f32.f16.sat", (Float16, Float32), Float32,
         "sub.rz.f32.f16.sat \$0, \$1, \$2;", "=f,h,f"),
        (ptx"fma.rz.sat.f32.f16.sat", (Float16, Float16, Float32), Float32,
         "fma.rz.sat.f32.f16.sat \$0, \$1, \$2, \$3;", "=f,h,h,f"),
        (ptx"popc.b64", (UInt64,), UInt32,
         "popc.b64 \$0, \$1;", "=r,l"),
        (ptx"clz.b64", (Int64,), UInt32,
         "clz.b64 \$0, \$1;", "=r,l"),
        (ptx"dp4a.s32.u32", (Int32, UInt32, Int32), Int32,
         "dp4a.s32.u32 \$0, \$1, \$2, \$3;", "=r,r,r,r"),
        (ptx"dp2a.hi.u32.u32", (UInt32, UInt32, UInt32), UInt32,
         "dp2a.hi.u32.u32 \$0, \$1, \$2, \$3;", "=r,r,r,r"),
        (ptx"mul.wide.s16", (Int16, Int16), Int32,
         "mul.wide.s16 \$0, \$1, \$2;", "=r,h,h"),
        (ptx"mul.wide.u32", (UInt32, UInt32), UInt64,
         "mul.wide.u32 \$0, \$1, \$2;", "=l,r,r"),
        (ptx"mad.wide.s32", (Int32, Int32, Int64), Int64,
         "mad.wide.s32 \$0, \$1, \$2, \$3;", "=l,r,r,l"),
        (ptx"prmt.b32.rc8", (UInt32, UInt32, UInt32), UInt32,
         "prmt.b32.rc8 \$0, \$1, \$2, \$3;", "=r,r,r,r"),
        (ptx"add.sat.u8x4", (UInt32, UInt32), UInt32,
         "add.sat.u8x4 \$0, \$1, \$2;", "=r,r,r"),
        (ptx"add.s8x4.sat", (UInt32, UInt32), UInt32,
         "add.s8x4.sat \$0, \$1, \$2;", "=r,r,r"),
        (ptx"neg.s8x4", (UInt32,), UInt32,
         "neg.s8x4 \$0, \$1;", "=r,r"),
        (ptx"min.s16x2.relu", (UInt32, UInt32), UInt32,
         "min.s16x2.relu \$0, \$1, \$2;", "=r,r,r"),
        (ptx"max.relu.s32", (Int32, Int32), Int32,
         "max.relu.s32 \$0, \$1, \$2;", "=r,r,r"),
        (ptx"cvt.pack.sat.s16.s32", (Int32, Int32), UInt32,
         "cvt.pack.sat.s16.s32 \$0, \$1, \$2;", "=r,r,r"),
        (ptx"cvt.pack.sat.u4.s32.b32", (Int32, Int32, UInt32), UInt32,
         "cvt.pack.sat.u4.s32.b32 \$0, \$1, \$2, \$3;", "=r,r,r,r"),
    ]
    for (op, argtypes, rettype, asm, constraints) in cases
        opname, mods = typeof(op).parameters
        spec = PTX.build_call(opname, mods, argtypes)
        @test spec.rettype === rettype
        @test spec.asm == asm
        @test spec.constraints == constraints
    end

    # `.b32`/`.b64` accept every same-width scalar type under PTX §6.1.
    @test PTX.build_call(:popc, (:b32,), (Float32,)).constraints == "=r,f"
    @test PTX.build_call(:clz, (:b64,), (Float64,)).constraints == "=r,d"

    # Integer signedness is mutually compatible at a common width. The
    # modifier pair, not the Julia carrier spelling, fixes dp semantics/result.
    cross_signed = PTX.build_call(:dp4a, (:s32, :u32),
                                  (UInt32, Int32, UInt32))
    @test cross_signed.rettype === Int32
    @test cross_signed.constraints == "=r,r,r,r"
    @test PTX.build_call(:cvt, (:pack, :sat, :s16, :s32),
                         (UInt32, UInt32)).rettype === UInt32

    # PTX.jl's unsigned bit carriers exercise §6.1 bit-size compatibility.
    @test PTX.build_call(:add, (:f32, :f16),
                         (UInt16, UInt32)).constraints == "=f,h,r"

    # Integer Val operands are contextual immediates and remain usable.
    @test PTX.build_call(:dp4a, (:u32, :u32),
                         (UInt32, UInt32, Val{0})).constraints == "=r,r,r"
    @test PTX.build_call(:cvt, (:pack, :sat, :u8, :s32, :b32),
                         (Int32, Int32, Val{0})).asm ==
          "cvt.pack.sat.u8.s32.b32 \$0, \$1, \$2, 0;"
end

@testset "audited scalar carrier and grammar failures are loud" begin
    bad_args = [
        (:add, (:f32, :f16), (Int16, Float32)),
        (:add, (:f32, :bf16), (Float16, Float32)),
        (:sub, (:f32, :f16), (Float16, Float16)),
        (:fma, (:rn, :f32, :f16), (Float16, Float16)),
        (:popc, (:b64,), (UInt32,)),
        (:clz, (:b32,), (UInt64,)),
        (:dp4a, (:s32, :u32), (Float32, UInt32, Int32)),
        (:dp4a, (:s32, :u32), (Int32, UInt32, Float32)),
        (:dp2a, (:lo, :u32, :s32), (UInt32, Float32, Int32)),
        (:mul, (:wide, :u16), (UInt32, UInt16)),
        (:mad, (:wide, :s32), (Int32, Int32, Int32)),
        (:prmt, (:b32, :rc8), (UInt64, UInt32, UInt32)),
        (:add, (:sat, :u8x4), (UInt64, UInt32)),
        (:neg, (:s8x4,), (UInt64,)),
        (:min, (:relu, :s32), (Float32, Int32)),
        (:cvt, (:pack, :sat, :s16, :s32), (Float32, Int32)),
        (:cvt, (:pack, :sat, :u4, :s32, :b32), (Int32, Int32, UInt64)),
    ]
    for (op, mods, argtypes) in bad_args
        err = try
            PTX.build_call(op, mods, argtypes)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("operand", sprint(showerror, err))
        @test PTX.lowering(PTX.Operation{op, mods}(), argtypes).tier === :forbidden
        @test PTX.lowering(PTX.RawOperation{op, mods}(), argtypes).tier === :forbidden
    end

    for (op, mods, argtypes) in (
        (:popc, (:b64,), ()),
        (:fma, (:rn, :f32, :f16), (Float16, Float16)),
        (:cvt, (:pack, :sat, :s16, :s32), (Int32,)),
        (:prmt, (:b32, :rc8), (UInt32, UInt32)),
        (:add, (:sat, :u8x4), (UInt32,)),
    )
        @test PTX.lowering(PTX.Operation{op, mods}(), argtypes).tier === :forbidden
        @test PTX.lowering(PTX.RawOperation{op, mods}(), argtypes).tier === :forbidden
    end

    misses = [
        (:popc, (:u32,)),
        (:clz, (:b16,)),
        (:dp4a, (:u32, :b32)),
        (:dp2a, (:middle, :u32, :u32)),
        (:mul, (:u32, :wide)),
        (:mul, (:wide, :u64)),
        (:mad, (:wide, :s64)),
        (:prmt, ()),
        (:prmt, (:foo,)),
        (:prmt, (:f4e, :b32)),
        (:prmt, (:b32, :foo)),
        (:sub, (:u8x4, :sat)),
        (:add, (:sat, :s8x4, :sat)),
        (:min, (:s32, :relu)),
        (:max, (:s16x2, :relu)),
        (:min, (:relu, :s16x2, :relu)),
        (:cvt, (:pack, :u16, :s32)),
        (:cvt, (:rn, :pack, :u16, :s32)),
        (:add, (:sat, :rn, :f32, :f16)),
        (:add, (:rn, :f32, :f16, :sat)),
        (:fma, (:f32, :f16)),
        (:add, (:f32, :rz, :f16)),
        (:sub, (:rn, :f32, :sat, :bf16)),
        (:fma, (:bf16, :rn, :f32)),
    ]
    for (op, mods) in misses
        @test_throws ArgumentError PTX.infer_rettype(op, mods)
        @test_throws ArgumentError PTX.build_call(op, mods, ())
        @test_throws ArgumentError PTX.build_call(op, mods, (); raw = true)
        @test PTX.lowering(PTX.Operation{op, mods}(), ()).tier === :forbidden
        @test PTX.lowering(PTX.RawOperation{op, mods}(), ()).tier === :forbidden
    end

    # Exact raw forms deliberately retain the audited ABI and validation.
    raw = PTX.build_call(:popc, (:b64,), (UInt64,); raw = true)
    @test raw.rettype === UInt32
    @test raw.constraints == "=r,l,~{memory}"
    @test raw.side_effects
    @test raw.convergent
    @test_throws ArgumentError PTX.build_call(:popc, (:b64,), (UInt32,);
                                              raw = true)
    @test_throws ArgumentError PTX.build_call(:add, (:f32, :f16),
                                              (Int16, Float32); raw = true)
    @test_throws ArgumentError PTX.build_call(:prmt, (:b32, :rc8),
                                              (UInt32, UInt32); raw = true)

    # Generic safe islands are unchanged.
    @test PTX.infer_rettype(:add, (:f32,)) === Float32
    @test PTX.infer_rettype(:sub, (:f64,)) === Float64
    @test PTX.infer_rettype(:fma, (:rn, :f32)) === Float32
    @test PTX.infer_rettype(:cvt, (:u64, :u32)) === UInt64
    @test PTX.infer_rettype(:prmt, (:b32,)) === UInt32
end

@testset "pure and ordinary-cvt result ABI failures are loud" begin
    # These are accepted by some ptxas releases, but are neither canonical ISA
    # grammar nor one of the five contradictory documented example spellings.
    pure_void_misses = [
        (:add, (:s32, :sat)),
        (:abs, (:f32, :ftz)),
        (:neg, (:s32, :sat)),
    ]
    for (op, mods) in pure_void_misses
        @test_throws ArgumentError PTX.infer_rettype(op, mods)
        @test_throws ArgumentError PTX.build_call(op, mods, ())
        @test_throws ArgumentError PTX.build_call(op, mods, (); raw = true)
        @test PTX.lowering(PTX.Operation{op, mods}(), ()).tier === :forbidden
        @test PTX.lowering(PTX.RawOperation{op, mods}(), ()).tier === :forbidden
    end

    noncanonical_cvt = [
        (:f64, :bf16, :rp),
        (:bf16, :f16, :rz),
        (:rpi, :s32, :f32, :rpi),
        (:bf16x2, :e4m3x2, :scaled__n2__ue8m0, :rn),
    ]
    for mods in noncanonical_cvt
        @test_throws ArgumentError PTX.infer_rettype(:cvt, mods)
        @test_throws ArgumentError PTX.build_call(:cvt, mods, (UInt32,))
        @test_throws ArgumentError PTX.build_call(:cvt, mods, (UInt32,);
                                                  raw = true)
        @test PTX.lowering(PTX.Operation{:cvt, mods}(), (UInt32,)).tier ===
              :forbidden
        @test PTX.lowering(PTX.RawOperation{:cvt, mods}(), (UInt32,)).tier ===
              :forbidden
    end

    # The guard is scoped to reviewed pure/cvt ABIs. Raw remains a conservative
    # void escape hatch for a genuinely unregistered vendor opcode.
    vendor_raw = PTX.build_call(:vendorop, (:opaque,), (); raw = true)
    @test vendor_raw.rettype === Nothing
    @test vendor_raw.asm == "vendorop.opaque;"
    @test vendor_raw.constraints == "~{memory}"
    @test vendor_raw.side_effects
    @test vendor_raw.convergent
end

function _scalar_transpile_module(instructions;
                                  declarations = """
                                  .reg .b16 %h<4>;
                                  .reg .f32 %f<4>;
                                  .reg .b32 %r<16>;
                                  .reg .b64 %rd<4>;
                                  """)
    """
    .version 9.3
    .target sm_120f
    .address_size 64
    .visible .entry scalar_result_transpile()
    {
        $declarations
        $instructions
        ret;
    }
    """
end

@testset "transpiler consumes scalar schemas before alias lowering" begin
    source = _scalar_transpile_module("""
        add.rz.f32.f16 %f0, 1, 2;
        add.rz.f32.bf16 %f1, 3, 4;
        dp4a.s32.u32 %r0, 1, 2, 3;
        mad.wide.u32 %rd0, 4, 5, 6;
        cvt.pack.sat.u8.s32.b32 %r1, 7, 8, 0;
        prmt.b32.rc8 %r2, 9, 10, 2;
        add.sat.u8x4 %r3, 11, 12;
        min.s16x2.relu %r4, 13, 14;
    """)
    out = PTX.ptx_to_julia(source)
    parsed = Meta.parseall(out)
    @test parsed isa Expr && parsed.head == :toplevel
    @test !any(arg -> arg isa Expr && arg.head == :error, parsed.args)
    for line in (
        "f0 = ptx\"add.rz.f32.f16\"(Float16(1), Float32(2))",
        "f1 = ptx\"add.rz.f32.bf16\"(UInt16(3), Float32(4))",
        "r0 = ptx\"dp4a.s32.u32\"(Int32(1), UInt32(2), Int32(3))",
        "rd0 = ptx\"mad.wide.u32\"(UInt32(4), UInt32(5), UInt64(6))",
        "r1 = ptx\"cvt.pack.sat.u8.s32.b32\"(Int32(7), Int32(8), UInt32(0))",
        "r2 = ptx\"prmt.b32.rc8\"(UInt32(9), UInt32(10), UInt32(2))",
        "r3 = ptx\"add.sat.u8x4\"(UInt32(11), UInt32(12))",
        "r4 = ptx\"min.s16x2.relu\"(UInt32(13), UInt32(14))",
    )
        @test occursin(line, out)
    end

    # Schema grammar and arity are rejected while still in IR, not deferred to
    # evaluating the generated Julia source.
    @test_throws ArgumentError PTX.ptx_to_julia(_scalar_transpile_module(
        "add.rn.f32.f16.sat %f0, %h0, %f1;"))
    @test_throws ArgumentError PTX.ptx_to_julia(_scalar_transpile_module(
        "popc.b64 %r0, %rd0, %rd1;"))

    # Keep grammar validation ahead of shared-pointer alias absorption. If it
    # moves below `_try_alias_def!`, this malformed mixed add can be mistaken
    # for pointer arithmetic and silently disappear from the translation.
    pointer_source = _scalar_transpile_module("""
        mov.u64 %rd0, smem;
        add.f32.rz.f16 %rd1, %rd0, 1;
    """; declarations = """
        .reg .b64 %rd<2>;
        .shared .b8 smem[16];
    """)
    @test_throws ArgumentError PTX.ptx_to_julia(pointer_source)

    pure_pointer_source = _scalar_transpile_module("""
        mov.u64 %rd0, smem;
        add.s32.sat %rd1, %rd0, 1;
    """; declarations = """
        .reg .b64 %rd<2>;
        .shared .b8 smem[16];
    """)
    @test_throws ArgumentError PTX.ptx_to_julia(pure_pointer_source)

    # Ordinary cvt's noncanonical postfix examples would otherwise choose a
    # plausible but wrong result type in the transpiler as well.
    @test_throws ArgumentError PTX.ptx_to_julia(_scalar_transpile_module(
        "cvt.f64.bf16.rp %rd0, %h0;"))
end
