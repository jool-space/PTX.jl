using PTX: Operation, RawOperation, build_call

const _IMMEDIATE_SECTIONS = (
    setmaxnreg =
        "ptx/9-instruction-set/9.7.21.5-miscellaneous-instructions-setmaxnreg.md",
    pmevent =
        "ptx/9-instruction-set/9.7.21.3-miscellaneous-instructions-pmevent.md",
)

function _expected_immediate_contracts()
    setmax_target = "sm_90a/sm_100a/sm_110a/sm_120a; PTX 8.8 also " *
        "admits sm_100f/sm_110f/sm_120f and later targets in the same family"
    [
        (op = :setmaxnreg, forms = ((:inc, :sync, :aligned, :u32),),
         operand_index = 1, minimum = 24, maximum = 256, multiple = 8,
         returns = false, side_effect = true,
         ptx_version = v"8.0", min_sm = v"9.0", feature_set = :arch,
         target_note = setmax_target, section = _IMMEDIATE_SECTIONS.setmaxnreg),
        (op = :setmaxnreg, forms = ((:dec, :sync, :aligned, :u32),),
         operand_index = 1, minimum = 24, maximum = 256, multiple = 8,
         returns = false, side_effect = true,
         ptx_version = v"8.0", min_sm = v"9.0", feature_set = :arch,
         target_note = setmax_target, section = _IMMEDIATE_SECTIONS.setmaxnreg),
        (op = :pmevent, forms = ((),), operand_index = 1,
         minimum = 0, maximum = 15, multiple = 1, returns = false,
         side_effect = true, ptx_version = v"1.4",
         min_sm = nothing, feature_set = :baseline, target_note = "all targets",
         section = _IMMEDIATE_SECTIONS.pmevent),
        (op = :pmevent, forms = ((:mask,),), operand_index = 1,
         minimum = 0, maximum = 0xffff, multiple = 1, returns = false,
         side_effect = true, ptx_version = v"3.0",
         min_sm = v"2.0", feature_set = :baseline,
         target_note = "sm_20 or higher", section = _IMMEDIATE_SECTIONS.pmevent),
    ]
end

@testset "PTX 9.3 immediate contract ledger" begin
    actual = [NamedTuple{fieldnames(typeof(contract))}(
                  Tuple(getfield(contract, field)
                        for field in fieldnames(typeof(contract))))
              for contract in PTX.IMMEDIATE_FORM_CONTRACTS]
    @test actual == _expected_immediate_contracts()
    @test length(actual) == 4

    expanded = Set((contract.op, mods) for contract in actual
                   for mods in contract.forms)
    @test expanded == Set((
        (:setmaxnreg, (:inc, :sync, :aligned, :u32)),
        (:setmaxnreg, (:dec, :sync, :aligned, :u32)),
        (:pmevent, ()), (:pmevent, (:mask,)),
    ))

    # lop3's ISA-required immLut constant (0:255) is owned by the structured
    # island as the `:imm8` operand kind — there is deliberately NO immediate
    # record for it. This oracle pins that boundary: exactly three lop3
    # forms, immLut as `:imm8` at operand 4 in each, their ISA deltas, and
    # the absence of any immediate-island resolution.
    lop3_schemas = [s for s in PTX.STRUCTURED_RESULT_SCHEMAS if s.op === :lop3]
    @test Set(s.mods for s in lop3_schemas) ==
          Set(((:b32,), (:or, :b32), (:and, :b32)))
    @test all(s.operands[4] === :imm8 for s in lop3_schemas)
    for s in lop3_schemas
        @test PTX.schema(PTX.ImmediateLedger(), s.op, s.mods) === nothing
    end
    @test Dict(s.mods => (s.ptx_version, s.min_sm) for s in lop3_schemas) ==
          Dict((:b32,) => (v"4.3", v"5.0"),
               (:or, :b32) => (v"8.2", v"7.0"),
               (:and, :b32) => (v"8.2", v"7.0"))
end

@testset "immediate direct/raw shape and exact domains" begin
    legal = (
        (:setmaxnreg, (:inc, :sync, :aligned, :u32), (24, 256)),
        (:setmaxnreg, (:dec, :sync, :aligned, :u32), (24, 256)),
        (:pmevent, (), (0, 15)),
        (:pmevent, (:mask,), (0, 0xffff)),
    )
    for (op, mods, values) in legal, value in values, raw in (false, true)
        argtypes = (Val{value},)
        spec = build_call(op, mods, argtypes; raw)
        @test spec.rettype === Nothing
        @test spec.side_effects
        @test !occursin("\$0", spec.asm)
        @test spec.asm == PTX.build_head(op, mods) * " $value;"
        @test PTX.lowering(raw ? RawOperation{op, mods}() :
                                Operation{op, mods}(), argtypes).tier === :chain_asm
    end

    @test build_call(:setmaxnreg, (:inc, :sync, :aligned, :u32),
                     (Val{24},)).convergent
    @test !build_call(:pmevent, (), (Val{0},)).convergent

    bad_forms = (
        (:setmaxnreg, ()),
        (:setmaxnreg, (:inc, :sync, :u32)),
        (:setmaxnreg, (:inc, :sync, :aligned, :s32)),
        (:setmaxnreg, (:add, :sync, :aligned, :u32)),
        (:setmaxnreg, (:sync, :aligned, :u32)),
        (:pmevent, (:u32,)),
        (:pmevent, (:mask, :u32)),
    )
    for (op, mods) in bad_forms, raw in (false, true)
        operation = raw ? RawOperation{op, mods}() : Operation{op, mods}()
        @test PTX.lowering(operation, (Val{24},)).tier === :forbidden
        @test_throws ArgumentError build_call(op, mods, (Val{24},); raw)
    end

    bad_values = (
        (:setmaxnreg, (:inc, :sync, :aligned, :u32),
         (Int32, Val{true}, Val{1.0}, Val{23}, Val{25}, Val{257})),
        (:setmaxnreg, (:dec, :sync, :aligned, :u32),
         (UInt32, Val{false}, Val{-8}, Val{248 + 1}, Val{264})),
        (:pmevent, (), (Int32, Val{true}, Val{1.0}, Val{-1}, Val{16})),
        (:pmevent, (:mask,), (UInt32, Val{false}, Val{-1}, Val{0x10000})),
    )
    for (op, mods, types) in bad_values, T in types, raw in (false, true)
        operation = raw ? RawOperation{op, mods}() : Operation{op, mods}()
        @test PTX.lowering(operation, (T,)).tier === :forbidden
        @test_throws ArgumentError build_call(op, mods, (T,); raw)
    end
    for (op, mods) in ((:setmaxnreg, (:inc, :sync, :aligned, :u32)),
                       (:pmevent, ()), (:pmevent, (:mask,))),
        argtypes in ((), (Val{1}, Val{2}))
        @test PTX.lowering(Operation{op, mods}(), argtypes).tier === :forbidden
        @test PTX.lowering(RawOperation{op, mods}(), argtypes).tier === :forbidden
        @test_throws ArgumentError build_call(op, mods, argtypes)
        @test_throws ArgumentError build_call(op, mods, argtypes; raw = true)
    end
end

function _immediate_transpile(instructions::AbstractString)
    PTX.ptx_to_julia(""".version 9.3
    .target sm_90a
    .address_size 64
    .visible .entry immediate_probe() {
      .reg .pred %p;
      .reg .u32 %r;
      $instructions
      ret;
    }
    """)
end

@testset "immediate transpiler validates before destination inference" begin
    julia = _immediate_transpile("""
        setmaxnreg.inc.sync.aligned.u32 030;
        @%p setmaxnreg.dec.sync.aligned.u32 (0x100);
        pmevent 017;
        @!%p pmevent.mask ((1 << 15) | 1);
        """)
    @test occursin("ptx\"setmaxnreg.inc.sync.aligned.u32\"(Val(24))", julia)
    @test occursin("ptx\"setmaxnreg.dec.sync.aligned.u32\"(Val(256))", julia)
    @test occursin("ptx\"pmevent\"(Val(15))", julia)
    @test occursin("ptx\"pmevent.mask\"(Val(32769))", julia)
    @test !occursin(r"= ptx\"(?:setmaxnreg|pmevent)", julia)
    @test Meta.parseall(julia).head === :toplevel

    invalid = (
        "setmaxnreg.inc.sync.aligned.u32 %r;",
        "setmaxnreg.inc.sync.aligned.u32 25;",
        "setmaxnreg.inc.sync.aligned.u32 264;",
        "setmaxnreg.inc.sync.u32 24;",
        "setmaxnreg.inc.sync.aligned.u32 24, 32;",
        "pmevent %r;",
        "pmevent -1;",
        "pmevent 16;",
        "pmevent.mask 0x10000;",
        "pmevent.mask.u32 1;",
        "pmevent;",
    )
    for instruction in invalid
        err = try
            _immediate_transpile(instruction)
            nothing
        catch caught
            caught
        end
        @test err isa PTX.Codegen.TranspilerError
        @test err.category == :schema
        @test occursin("immediate_probe", err.path)
    end
end

@testset "NVVM setmaxnreg immediate stride overlay" begin
    names = (
        "llvm.nvvm.setmaxnreg.inc.sync.aligned.u32",
        "llvm.nvvm.setmaxnreg.dec.sync.aligned.u32",
    )
    for name in names
        @test PTX.NVVM.synthesize(name, (Val{24},)).rettype === Nothing
        @test PTX.NVVM.synthesize(name, (Val{256},)).rettype === Nothing
        for T in (Val{true}, Val{23}, Val{25}, Val{255}, Val{257})
            @test_throws ErrorException PTX.NVVM.synthesize(name, (T,))
        end
    end
end
