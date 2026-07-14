# Independent reconstruction of PTX 9.3's vector-result core cells. Optional
# memory qualifiers are tested as grammar axes below; this oracle deliberately
# does not derive its keys or counts from VECTOR_RESULT_CORE_FORMS.

const _VR_LANE_TYPE = Dict(
    :b8 => UInt8, :u8 => UInt8, :s8 => Int8,
    :b16 => UInt16, :u16 => UInt16, :s16 => Int16,
    :b32 => UInt32, :u32 => UInt32, :s32 => Int32,
    :b64 => UInt64, :u64 => UInt64, :s64 => Int64,
    :f16 => Float16, :bf16 => UInt16, :f32 => Float32, :f64 => Float64,
    :f16x2 => UInt32, :bf16x2 => UInt32,
    :e4m3 => UInt8, :e5m2 => UInt8,
    :e4m3x2 => UInt16, :e5m2x2 => UInt16,
    :e4m3x4 => UInt32, :e5m2x4 => UInt32,
)

function _expected_vector_core()
    expected = Dict{Tuple{Symbol,Tuple},NamedTuple}()
    add!(op, mods, lanes, kind, version, min_sm;
         arch = (), family = (), family_since = nothing) = begin
        key = (op, mods)
        @assert !haskey(expected, key)
        expected[key] = (; lanes, kind, lane_type = _VR_LANE_TYPE[kind],
                          version, min_sm, arch, family, family_since)
    end

    for kind in (:b8, :u8, :s8, :b16, :u16, :s16,
                 :b32, :u32, :s32, :b64, :u64, :s64, :f32, :f64)
        add!(:ld, (:v2, kind), 2, kind, v"1.0",
             kind === :f64 ? v"1.3" : nothing)
    end
    for kind in (:b8, :u8, :s8, :b16, :u16, :s16,
                 :b32, :u32, :s32, :f32)
        add!(:ld, (:v4, kind), 4, kind, v"1.0", nothing)
    end
    for kind in (:b64, :u64, :s64, :f64)
        add!(:ld, (:v4, kind), 4, kind, v"8.8", v"10.0")
    end
    for kind in (:b32, :u32, :s32, :f32)
        add!(:ld, (:v8, kind), 8, kind, v"8.8", v"10.0")
    end

    for lanes in (2, 4)
        add!(:atom, (:add, Symbol("v", lanes), :f32), lanes, :f32,
             v"8.1", v"9.0")
    end
    for lanes in (2, 4, 8), kind in (:f16, :bf16), op in (:add, :min, :max)
        add!(:atom, (op, :noftz, Symbol("v", lanes), kind), lanes, kind,
             v"8.1", v"9.0")
    end
    for lanes in (2, 4), kind in (:f16x2, :bf16x2), op in (:add, :min, :max)
        add!(:atom, (op, :noftz, Symbol("v", lanes), kind), lanes, kind,
             v"8.1", v"9.0")
    end

    shapes = (
        2 => (:f16, :f16x2, :bf16, :bf16x2, :f32,
              :e4m3x2, :e4m3x4, :e5m2x2, :e5m2x4),
        4 => (:f16, :f16x2, :bf16, :bf16x2, :f32,
              :e4m3, :e4m3x2, :e4m3x4, :e5m2, :e5m2x2, :e5m2x4),
        8 => (:f16, :bf16, :e4m3, :e4m3x2, :e5m2, :e5m2x2),
    )
    fp8 = Set((:e4m3, :e4m3x2, :e4m3x4, :e5m2, :e5m2x2, :e5m2x4))
    narrow = Set((:f16, :f16x2, :bf16, :bf16x2))
    arch = (v"10.0", v"11.0", v"12.0", v"12.1")
    family = (v"10.0", v"11.0")
    for (lanes, kinds) in shapes, kind in kinds
        target = kind in fp8 ? nothing : v"9.0"
        version = kind in fp8 ? v"8.6" : v"8.1"
        target_kw = kind in fp8 ? (; arch, family, family_since = v"8.8") : (;)
        for op in (kind === :f32 ? (:add,) : (:add, :min, :max))
            add!(:multimem, (:ld_reduce, op, Symbol("v", lanes), kind),
                 lanes, kind, version, target; target_kw...)
        end
        if kind in narrow
            for op in (:add, :min, :max)
                add!(:multimem,
                     (:ld_reduce, op, Symbol("acc::f32"), Symbol("v", lanes), kind),
                     lanes, kind, v"8.2", v"9.0")
            end
        elseif kind in fp8
            for op in (:add, :min, :max)
                add!(:multimem,
                     (:ld_reduce, op, Symbol("acc::f16"), Symbol("v", lanes), kind),
                     lanes, kind, v"8.6", nothing;
                     arch, family, family_since = v"8.8")
            end
        end
    end
    expected
end

const _VECTOR_ATOM_COMPAT_EXAMPLES = Set((
    (:global, :v8, :f16, :max, :noftz),
    (:global, :v8, :bf16, :add, :noftz),
    (:global, :v2, :f16, :add, :noftz),
    (:global, :v2, :bf16, :add, :noftz),
    (:global, :v4, :f32, :add),
    (:global, :v2, :f16x2, :min, :noftz),
    (:global, :v2, :bf16x2, :max, :noftz),
    (:global, :v2, :f32, :add),
))

@testset "audited vector-result core inventory" begin
    expected = _expected_vector_core()
    actual = Dict((form.op, form.coremods) => form
                  for form in PTX.VECTOR_RESULT_CORE_FORMS)
    @test length(expected) == 210
    @test length(actual) == 210
    @test Set(keys(actual)) == Set(keys(expected))
    @test count(k -> k[1] === :ld, keys(actual)) == 32
    @test count(k -> k[1] === :atom, keys(actual)) == 32
    @test count(k -> k[1] === :multimem, keys(actual)) == 146
    @test count(k -> k[1] === :multimem && Symbol("acc::f32") in k[2],
                keys(actual)) == 30
    @test count(k -> k[1] === :multimem && Symbol("acc::f16") in k[2],
                keys(actual)) == 42
    for (key, want) in expected
        got = actual[key]
        @test got.lanes == want.lanes
        @test got.lane_kind === want.kind
        @test got.lane_type === want.lane_type
        @test got.ptx_version == want.version
        @test got.target.min_sm == want.min_sm
        @test got.target.architectures == want.arch
        @test got.target.families == want.family
        @test got.target.family_since == want.family_since
        @test got.provenance === :isa
        @test !isempty(got.section)
    end
end

@testset "atom example compatibility quarantine" begin
    @test length(PTX.VECTOR_RESULT_COMPAT_FORMS) == 8
    recorded = Set(only(form.spellings)
                   for form in PTX.VECTOR_RESULT_COMPAT_FORMS)
    @test recorded == _VECTOR_ATOM_COMPAT_EXAMPLES
    for mods in _VECTOR_ATOM_COMPAT_EXAMPLES
        schema = PTX.vector_result_schema(:atom, mods)
        @test schema !== nothing
        @test schema.provenance === :ptxas_compat
    end

    # The ninth vector example uses a type absent from the grammar and is also
    # rejected by CUDA 13.3 ptxas with "Unknown modifier '.b16x2'".
    rejected = (:global, :v4, :b16x2, :min, :noftz)
    @test PTX.vector_result_schema(:atom, rejected) === nothing
    @test PTX.requires_vector_result_schema(:atom, rejected)
    @test_throws ArgumentError PTX.build_call(
        :atom, rejected,
        (Core.LLVMPtr{UInt8,PTX.AS.Global}, NTuple{4,UInt32}))
    # Nearby alternate orders are not inferred from the accepted examples.
    @test_throws ArgumentError PTX.build_call(
        :atom, (:global, :v4, :f32, :max),
        (Core.LLVMPtr{UInt8,PTX.AS.Global}, NTuple{4,Float32}))
end

@testset "vector qualifier grammar axes" begin
    # Five ld syntax alternatives, including the wide-only L2 eviction axis.
    for mods in (
        (:weak, :global, :cg, :v2, :u32),
        (:global, Symbol("L1::evict_first"), Symbol("L2::cache_hint"),
         Symbol("L2::128B"), :v4, :f32),
        (:volatile, :local, :v2, :b16),
        (:relaxed, :gpu, :global, Symbol("L1::evict_last"), :v2, :u32),
        (:acquire, :sys, :global, Symbol("L2::evict_last"), :v8, :u32),
    )
        @test PTX.vector_result_schema(:ld, mods) !== nothing
    end
    @test PTX.vector_result_schema(
        :ld, (:global, Symbol("L2::evict_last"), :v2, :u32)) === nothing
    @test PTX.vector_result_schema(
        :ld, (:relaxed, :sys, :local, :v2, :u32)) === nothing
    @test PTX.vector_result_schema(
        :ld, (:volatile, :const, :v2, :u32)) === nothing

    hint = Symbol("L2::cache_hint")
    atom = PTX.vector_result_schema(
        :atom, (:acq_rel, :sys, :global, :add, hint, :v2, :f32))
    @test atom !== nothing
    @test atom.cache_hint
    @test PTX.vector_result_schema(
        :atom, (:global, hint, :add, :v2, :f32)) === nothing
    @test PTX.vector_result_schema(
        :multimem, (:ld_reduce, :acquire, :gpu, :global,
                    :min, Symbol("acc::f16"), :v4, :e4m3)) !== nothing
    @test PTX.vector_result_schema(
        :multimem, (:ld_reduce, :release, :gpu, :global,
                    :add, :v2, :f16)) === nothing
end

@testset "vector call result ABI and address roles" begin
    PtrG = Core.LLVMPtr{UInt8,PTX.AS.Global}
    ld = PTX.build_call(:ld, (:global, :v2, :u32), (PtrG,))
    @test ld.rettype === NTuple{2,UInt32}
    @test ld.asm == "ld.global.v2.u32 {\$0, \$1}, [\$2];"
    @test ld.constraints == "=r,=r,l,~{memory}"
    @test ld.side_effects
    @test !ld.convergent

    atom = PTX.build_call(:atom, (:global, :add, :v4, :f32),
                          (PtrG, NTuple{4,Float32}))
    @test atom.rettype === NTuple{4,Float32}
    @test atom.asm ==
        "atom.global.add.v4.f32 {\$0, \$1, \$2, \$3}, [\$4], {\$5, \$6, \$7, \$8};"
    @test atom.constraints == "=f,=f,=f,=f,l,f,f,f,f,~{memory}"
    @test count(==('['), atom.asm) == 1

    mm = PTX.build_call(
        :multimem,
        (:ld_reduce, :relaxed, :cta, :min, Symbol("acc::f16"), :v4, :e5m2),
        (PtrG,))
    @test mm.rettype === NTuple{4,UInt8}
    @test mm.constraints == "=h,=h,=h,=h,l,~{memory}"
    @test occursin("multimem.ld_reduce.relaxed.cta.min.acc::f16.v4.e5m2", mm.asm)

    # The ISA syntax writes both the hint and policy as optional, but CUDA
    # 12.9/13.3 ptxas require a policy whenever the hint is present. The API
    # therefore fails before emitting an assembler-rejected instruction.
    hintmods = (:global, :add, Symbol("L2::cache_hint"), :v2, :f32)
    @test_throws ArgumentError PTX.build_call(
        :atom, hintmods, (PtrG, NTuple{2,Float32}))
    hinted = PTX.build_call(:atom, hintmods,
                            (PtrG, NTuple{2,Float32}, UInt64))
    @test hinted.asm ==
        "atom.global.add.L2::cache_hint.v2.f32 {\$0, \$1}, [\$2], {\$3, \$4}, \$5;"
    @test_throws ArgumentError PTX.build_call(:atom, hintmods,
                                               (PtrG, NTuple{2,Float32}, UInt32))
    ldhint = (:global, Symbol("L2::cache_hint"), :v2, :u32)
    @test_throws ArgumentError PTX.build_call(:ld, ldhint, (PtrG,))
    @test PTX.build_call(:ld, ldhint, (PtrG, UInt64)).rettype ===
        NTuple{2,UInt32}

    @test_throws ArgumentError PTX.build_call(:ld, (:global, :v2, :u32),
                                               (PtrG, UInt32))
    @test_throws ArgumentError PTX.build_call(:atom, (:global, :add, :v2, :f32),
                                               (PtrG, NTuple{4,Float32}))
    @test_throws ArgumentError PTX.build_call(:atom, (:global, :add, :v2, :f32),
                                               (PtrG, NTuple{2,UInt32}))

    raw = PTX.build_call(:ld, (:global, :v2, :u32), (PtrG,); raw = true)
    @test raw.rettype === NTuple{2,UInt32}
    @test raw.convergent
    @test_throws ArgumentError PTX.build_call(:ld, (:global, :u32, :v2),
                                               (PtrG,); raw = true)
    @test_throws ArgumentError PTX.build_call(:ld, (:global, :v2, :b128),
                                               (PtrG,); raw = true)
end

@testset "wide load sink masks" begin
    PtrG = Core.LLVMPtr{UInt8,PTX.AS.Global}
    mods = (:global, Symbol("L2::evict_last"), :v8, :u32)
    schema = PTX.vector_result_schema(:ld, mods)
    mask = (true, false, true, true, false, true, true, true)
    spec = PTX._build_vector_result_call(
        schema, (PtrG,), PTX.form_contract(:ld, mods); sink_mask = mask)
    @test spec.rettype === NTuple{6,UInt32}
    @test spec.asm ==
        "ld.global.L2::evict_last.v8.u32 {\$0, _, \$1, \$2, _, \$3, \$4, \$5}, [\$6];"
    @test spec.constraints == "=r,=r,=r,=r,=r,=r,l,~{memory}"

    @test_throws ArgumentError PTX._build_vector_result_call(
        schema, (PtrG,), PTX.form_contract(:ld, mods);
        sink_mask = ntuple(_ -> false, 8))
    @test_throws ArgumentError vector_load(
        PTX.Operation{:ld, mods}(), UInt64(0),
        Val(ntuple(_ -> false, 8)))
    @test_throws ArgumentError PTX._build_vector_result_call(
        PTX.vector_result_schema(:ld, (:global, :v4, :u32)),
        (PtrG,), PTX.form_contract(:ld, (:global, :v4, :u32));
        sink_mask = (true, false, true, true))
end

@testset "vector boundary lowering and preserved fastpaths" begin
    PtrG = Core.LLVMPtr{UInt8,PTX.AS.Global}
    generic = PTX.lowering(ptx"ld.global.v2.u32", (PtrG,))
    @test generic.tier === :chain_asm
    @test generic.rettype === NTuple{2,UInt32}
    bad = PTX.lowering(ptx"atom.global.v4.b16x2.min.noftz",
                       (PtrG, NTuple{4,UInt32}))
    @test bad.tier === :forbidden
    @test PTX.lowering(ptx"ld.global.L2::cache_hint.v2.u32",
                       (PtrG,)).tier === :forbidden
    @test_throws ArgumentError PTX.build_call(
        :ld, (:global, Symbol("L2::cache_hint"), :v2, :u32),
        (PtrG,); raw = true)
    for (lanes, kind, T) in ((2, :f32, Float32), (4, :f32, Float32),
                             (2, :b32, UInt32), (4, :b32, UInt32),
                             (2, :b16, UInt16), (4, :b16, UInt16))
        op = PTX.Operation{:ld, (:global, Symbol("v", lanes), kind)}()
        @test PTX.lowering(op, (PtrG,)).tier === :core
        @test PTX.lowering(op, (PtrG,)).rettype === NTuple{lanes,T}
    end
end

@testset "transpiler preserves vector structure and sinks" begin
    source = """
    .version 9.3
    .target sm_100
    .address_size 64
    .visible .entry vector_probe() {
      .reg .b64 %rd;
      .reg .u32 %r<8>;
      ld.global.L2::evict_last.v8.u32 {%r0, _, %r2, %r3, %r4, %r5, %r6, %r7}, [%rd];
      ret;
    }
    """
    julia = PTX.ptx_to_julia(source)
    @test occursin("(r0, r2, r3, r4, r5, r6, r7) = vector_load(", julia)
    @test occursin("Val((true, false, true, true, true, true, true, true))", julia)
    @test !occursin("_ =", julia)

    all_sink = replace(source,
        "ld.global.L2::evict_last.v8.u32 {%r0, _, %r2, %r3, %r4, %r5, %r6, %r7}, [%rd];" =>
        "ld.global.L2::evict_last.v8.u32 {_, _, _, _, _, _, _, _}, [%rd];")
    @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(all_sink)

    malformed = replace(source,
        "L2::evict_last.v8.u32 {%r0, _, %r2, %r3, %r4, %r5, %r6, %r7}" =>
        "v4.u32 {%r0, %r1}")
    @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(malformed)

    missing_policy = replace(source,
        "ld.global.L2::evict_last.v8.u32 {%r0, _, %r2, %r3, %r4, %r5, %r6, %r7}, [%rd];" =>
        "ld.global.L2::cache_hint.v8.u32 {%r0, _, %r2, %r3, %r4, %r5, %r6, %r7}, [%rd];")
    @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(missing_policy)

    # PTX permits a wider integer/bit destination for ld and extends each
    # lane (§9.4.1). The exact tuple ABI cannot preserve that width yet, so it
    # must reject instead of silently assigning narrow UInt32 lanes to u64s.
    widened = replace(source, ".reg .u32 %r<8>;" => ".reg .u64 %r<8>;")
    err = try
        PTX.ptx_to_julia(widened)
        nothing
    catch caught
        caught
    end
    @test err isa PTX.Codegen.TranspilerError
    @test err isa PTX.Codegen.TranspilerError &&
          occursin("does not lower wider", sprint(showerror, err))

    wrong_float = replace(source, ".reg .u32 %r<8>;" => ".reg .f32 %r<8>;")
    @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(wrong_float)

    wrong_address = replace(source, ".reg .b64 %rd;" => ".reg .b16 %rd;")
    @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(wrong_address)

    coordinate_address = replace(source, "[%rd];" => "[%rd, {%r0}];")
    @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(coordinate_address)

    undeclared = replace(source, "%r0, _, %r2" => "%missing, _, %r2")
    @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(undeclared)

    # General PTX same-size compatibility remains available: a bit-size load
    # may target a same-width fundamental floating-point register.
    bit_compatible = replace(source,
        ".reg .u32 %r<8>;" => ".reg .f32 %r<8>;",
        "ld.global.L2::evict_last.v8.u32 {%r0, _, %r2, %r3, %r4, %r5, %r6, %r7}, [%rd];" =>
        "ld.global.v2.b32 {%r0, %r1}, [%rd];")
    @test occursin("(r0, r1) = ptx\"ld.global.v2.b32\"(address(rd))",
                   PTX.ptx_to_julia(bit_compatible))

    atom = """
    .version 9.3
    .target sm_100
    .address_size 64
    .visible .entry vector_atom_probe() {
      .reg .b64 %rd;
      .reg .f32 %d<2>;
      .reg .u32 %a<2>;
      atom.global.add.v2.f32 {%d0, %d1}, [%rd], {%a0, %a1};
      ret;
    }
    """
    @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(atom)
    atom = replace(atom, ".reg .u32 %a<2>;" => ".reg .b32 %a<2>;")
    @test occursin("(d0, d1) = ptx\"atom.global.add.v2.f32\"",
                   PTX.ptx_to_julia(atom))
end

function _vector_atom_immediate_source(kind::Symbol, values::AbstractString)
    declaration = kind in (:f16, :bf16) ? ".reg .b16 %d<2>;" :
                  kind === :f16x2 ? ".reg .b32 %d<2>;" :
                  kind === :f32 ? ".reg .f32 %d<2>;" :
                  error("unsupported vector atom test kind $kind")
    noftz = kind === :f32 ? "" : ".noftz"
    """
    .version 9.3
    .target sm_90
    .address_size 64
    .visible .entry vector_constant_probe() {
      .reg .b64 %rd;
      $declaration
      atom.global.add$noftz.v2.$kind {%d0, %d1}, [%rd], $values;
      ret;
    }
    """
end

@testset "vector atom immediates are format-aware and fail loud" begin
    # CUDA 12.9/13.3 ptxas reject immediate lanes for half, bfloat, and
    # packed-half vector atom sources. An integer constant is not compatible
    # with f32 either. Do not reinterpret those values as bit-pattern carriers.
    for (kind, values) in ((:f16, "{0, 0}"),
                           (:f16, "{1.0, 0f40000000}"),
                           (:bf16, "{0, 0}"),
                           (:bf16, "{1.0, 0f40000000}"),
                           (:f16x2, "{0, 0}"),
                           (:f16x2, "{1.0, 0f40000000}"),
                           (:f32, "{1, 2}"))
        @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(
            _vector_atom_immediate_source(kind, values))
    end

    # f32 is the one immediate-capable vector atom format. ptxas accepts
    # exact-width 0f, exact f64 0d, and decimal floating constants; PTX applies
    # the f32 instruction-type conversion at use, so generated Julia does too.
    exact = PTX.ptx_to_julia(_vector_atom_immediate_source(
        :f32, "{0f3f800000, 0d4000000000000000}"))
    @test occursin(
        "Float32(reinterpret(Float32, 0x3f800000))", exact)
    @test occursin(
        "Float32(reinterpret(Float64, 0x4000000000000000))", exact)
    uppercase = PTX.ptx_to_julia(_vector_atom_immediate_source(
        :f32, "{0F3f800000, 0D4000000000000000}"))
    @test occursin("Float32(reinterpret(Float32, 0x3f800000))", uppercase)
    @test occursin(
        "Float32(reinterpret(Float64, 0x4000000000000000))", uppercase)
    decimal = PTX.ptx_to_julia(_vector_atom_immediate_source(
        :f32, "{-1.5, 2.0e0}"))
    @test occursin("(Float32(-1.5), Float32(2.0e0))", decimal)
end

@testset "vector cache-policy constants use the PTX integer domain" begin
    cache = """
    .version 9.3
    .target sm_90
    .address_size 64
    .visible .entry vector_cache_probe() {
      .reg .b64 %rd;
      .reg .u32 %d<2>;
      ld.global.L2::cache_hint.v2.u32 {%d0, %d1}, [%rd], -1;
      ret;
    }
    """
    cache_julia = PTX.ptx_to_julia(cache)
    @test occursin("UInt64(0xffffffffffffffff)", cache_julia)

    shifted = replace(cache, ", -1;" => ", (0xffffffff << 32);")
    shifted_julia = PTX.ptx_to_julia(shifted)
    @test occursin("UInt64(0xffffffff00000000)", shifted_julia)
end

@testset "vector atom whole-result sink preserves the operation" begin
    source = """
    .version 9.3
    .target sm_90
    .address_size 64
    .visible .entry vector_atom_sink() {
      .reg .b64 %rd;
      .reg .f32 %a<2>;
      atom.global.add.v2.f32 _, [%rd], {%a0, %a1};
      ret;
    }
    """
    julia = PTX.ptx_to_julia(source)
    call = "ptx\"atom.global.add.v2.f32\"(address(rd), (a0, a1))"
    @test occursin(call, julia)
    @test !occursin("= " * call, julia)
    @test Meta.parseall(julia).head === :toplevel

    # `_` replaces the complete atom result; it is not legal per-lane syntax.
    per_lane = replace(source, "_, [%rd]" => "{_, _}, [%rd]")
    @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(per_lane)

    PtrG = Core.LLVMPtr{UInt8,PTX.AS.Global}
    args = (PtrG, NTuple{2,Float32})
    @test PTX.build_call(:atom, (:global, :add, :v2, :f32), args).rettype ===
        NTuple{2,Float32}
    @test PTX.build_call(:atom, (:global, :add, :v2, :f32), args;
                         raw = true).rettype === NTuple{2,Float32}
end


@testset "vector declaration inventory is lexical and scope-safe" begin
    comma_packed = """
    .version 9.3
    .target sm_90
    .address_size 64
    .visible .entry comma_vector() {
      .reg .b64 rd;
      .reg .f32 d0, d1, a0, a1;
      atom.global.add.v2.f32 {d0, d1}, [rd], {a0, a1};
      ret;
    }
    """
    julia = PTX.ptx_to_julia(comma_packed)
    @test occursin(
        "(d0, d1) = ptx\"atom.global.add.v2.f32\"(address(rd), (a0, a1))", julia)

    commented = replace(comma_packed,
        ".reg .f32 d0, d1, a0, a1;" =>
        ".reg .f32 d0, /* fake, ; */ d1, a0, a1;")
    @test occursin("(d0, d1) = ptx\"atom.global.add.v2.f32\"",
                   PTX.ptx_to_julia(commented))
    spoofed = replace(commented, "{d0, d1}" => "{d0, fake}")
    @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(spoofed)

    scoped = """
    .version 9.3
    .target sm_90
    .address_size 64
    .visible .entry scoped_vector() {
      .reg .b64 rd;
      {
        .reg .f32 inner<2>;
        atom.global.add.v2.f32 _, [rd], {inner0, inner1};
      }
      atom.global.add.v2.f32 _, [rd], {inner0, inner1};
      ret;
    }
    """
    @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(scoped)
end
