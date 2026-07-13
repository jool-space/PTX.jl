using PTX: Operation, RawOperation, build_call

# Independent reconstruction of PTX ISA 9.3 §9.7.14.16.  This deliberately
# does not consume MBARRIER_FORM_SCHEMAS or its production helper tables: a
# grammar edit must agree with this separately reviewed form/result/operand
# inventory.
function _expected_mbarrier_forms()
    expected = Dict{Tuple,NamedTuple}()
    add!(mods, ptxmods, destination, variants, space) = begin
        @assert !haskey(expected, mods)
        expected[mods] = (; ptxmods, destination, variants, space)
    end

    local_spaces = (
        ((), :generic), ((:shared,), :cta),
        ((Symbol("shared::cta"),), :cta),
    )
    all_spaces = (local_spaces...,
                  ((Symbol("shared::cluster"),), :cluster))
    tx_pairs = ((), (:relaxed, :cta), (:relaxed, :cluster))
    arrive_pairs = ((), (:release, :cta), (:release, :cluster),
                    (:relaxed, :cta), (:relaxed, :cluster))
    wait_pairs = ((), (:acquire, :cta), (:acquire, :cluster),
                  (:relaxed, :cta), (:relaxed, :cluster))

    for layout in ((), (Symbol("layout::v0"),),
                   (Symbol("layout::v1"),)), (space, kind) in local_spaces
        mods = (:init, layout..., space..., :b64)
        add!(mods, mods, :none, ((:address, :u32),), kind)
    end
    for (space, kind) in local_spaces
        mods = (:inval, space..., :b64)
        add!(mods, mods, :none, ((:address,),), kind)
    end
    for subop in (:expect_tx, :complete_tx), sem in tx_pairs,
        (space, kind) in all_spaces
        mods = (subop, sem..., space..., :b64)
        add!(mods, mods, :none, ((:address, :u32),), kind)
    end
    for subop in (:arrive, :arrive_drop), sem in arrive_pairs,
        (space, kind) in all_spaces
        mods = (subop, sem..., space..., :b64)
        destination = kind === :cluster ? :remote_sink : :state
        add!(mods, mods, destination,
             ((:address,), (:address, :u32)), kind)
    end
    for subop in (:arrive, :arrive_drop), sem in arrive_pairs,
        (space, kind) in all_spaces
        mods = (subop, :expect_tx, sem..., space..., :b64)
        destination = kind === :cluster ? :remote_sink : :state
        add!(mods, mods, destination, ((:address, :u32),), kind)
    end
    for subop in (:arrive, :arrive_drop), sem in ((), (:release, :cta)),
        (space, kind) in local_spaces
        mods = (subop, :noComplete, sem..., space..., :b64)
        add!(mods, mods, :state, ((:address, :u32),), kind)
    end

    for wait in (:test_wait, :try_wait), sem in wait_pairs,
        (space, kind) in local_spaces
        for phase in ((), (Symbol("phase_type::primary"),))
            mods = (wait, phase..., sem..., space..., :b64)
            variants = wait === :try_wait ?
                ((:address, :u64), (:address, :u64, :u32)) :
                ((:address, :u64),)
            add!(mods, mods, :predicate, variants, kind)
        end
        for phase in ((), (Symbol("phase_type::primary"),),
                      (Symbol("phase_type::conditional"),))
            mods = (wait, :parity, phase..., sem..., space..., :b64)
            variants = wait === :try_wait ?
                ((:address, :u32), (:address, :u32, :u32)) :
                ((:address, :u32),)
            add!(mods, mods, :predicate, variants, kind)
        end
        for parity in (false, true),
            (selector, destination) in
                ((:report_pred, :report_pred), (:report, :report))
            ptxmods = parity ?
                (wait, :parity, Symbol("phase_type::primary"),
                 sem..., space..., :b64) :
                (wait, Symbol("phase_type::primary"), sem..., space..., :b64)
            mods = parity ?
                (wait, selector, :parity, Symbol("phase_type::primary"),
                 sem..., space..., :b64) :
                (wait, selector, Symbol("phase_type::primary"),
                 sem..., space..., :b64)
            source = parity ? :u32 : :u64
            variants = wait === :try_wait ?
                ((:address, source), (:address, source, :u32)) :
                ((:address, source),)
            add!(mods, ptxmods, destination, variants, kind)
        end
    end
    for layout in ((), (Symbol("layout::v0"),))
        mods = (:pending_count, layout..., :b64)
        add!(mods, mods, :count, ((:u64,),), :none)
    end
    for layout in (Symbol("layout::v0"), Symbol("layout::v1")),
        (space, kind) in (((), :generic),
                          ((Symbol("shared::cta"),), :cta))
        mods = (:check_layout, layout, space..., :b64)
        add!(mods, mods, :predicate, ((:address,),), kind)
    end
    expected
end

_mb_rettype(destination) =
    destination in (:none, :remote_sink) ? Nothing :
    destination === :state ? UInt64 :
    destination === :predicate ? Bool :
    destination === :count ? UInt32 :
    destination === :report_pred ? Tuple{Bool, Bool} :
    destination === :report ? Tuple{Bool, Bool, UInt16} :
    error("bad destination $destination")

function _mb_argtype(kind, space)
    kind === :address && return space === :generic ?
        Core.LLVMPtr{UInt64, PTX.AS.Generic} :
        Core.LLVMPtr{UInt64, PTX.AS.Shared}
    kind === :u32 && return UInt32
    kind === :u64 && return UInt64
    error("bad operand kind $kind")
end

function _mb_output(destination)
    destination === :none && return (String[], String[])
    destination === :remote_sink && return (["_"], String[])
    destination === :state && return (["\$0"], ["=l"])
    destination === :predicate && return (["\$0"], ["=b"])
    destination === :count && return (["\$0"], ["=r"])
    destination === :report_pred && return (["\$0|\$1"], ["=b", "=b"])
    destination === :report &&
        return (["\$0|\$1", "report_value"], ["=b", "=b", "=h"])
    error("bad destination $destination")
end

@testset "closed mbarrier grammar and ABI ledger" begin
    expected = _expected_mbarrier_forms()
    actual = Dict(schema.mods => schema for schema in PTX.MBARRIER_FORM_SCHEMAS)
    @test length(expected) == 404
    @test length(actual) == 404
    @test Set(keys(actual)) == Set(keys(expected))
    @test count(k -> first(k) === :init, keys(actual)) == 9
    @test count(k -> first(k) === :inval, keys(actual)) == 3
    @test count(k -> first(k) in (:expect_tx, :complete_tx), keys(actual)) == 24
    @test count(k -> first(k) in (:arrive, :arrive_drop), keys(actual)) == 92
    @test count(k -> first(k) in (:test_wait, :try_wait), keys(actual)) == 270
    @test count(k -> first(k) === :pending_count, keys(actual)) == 2
    @test count(k -> first(k) === :check_layout, keys(actual)) == 4

    for (mods, want) in expected
        schema = actual[mods]
        @test schema.ptxmods == want.ptxmods
        @test schema.destination === want.destination
        @test schema.space === want.space
        @test Tuple(v.operands for v in schema.variants) == want.variants
        @test schema.section ==
            "ptx/9-instruction-set/9.7.14.16-parallel-synchronization-and-communication-instructions-mbarrier.md"

        for operands in want.variants
            argtypes = Tuple(_mb_argtype(kind, want.space) for kind in operands)
            spec = build_call(:mbarrier, mods, argtypes)
            @test spec.rettype === _mb_rettype(want.destination)
            @test spec.side_effects
            @test !spec.convergent

            outputs, outletters = _mb_output(want.destination)
            slot = length(outletters)
            inputs = String[]
            inletters = String[]
            for kind in operands
                if kind === :address
                    push!(inputs, "[\$$slot]")
                    push!(inletters, want.space === :generic ? "l" : "r")
                else
                    push!(inputs, "\$$slot")
                    push!(inletters, kind === :u32 ? "r" : "l")
                end
                slot += 1
            end
            head = "mbarrier." * join(want.ptxmods, ".")
            all_operands = [outputs; inputs]
            expected_asm = isempty(all_operands) ? head * ";" :
                           head * " " * join(all_operands, ", ") * ";"
            want.destination === :report &&
                (expected_asm = "{ .reg .b8 report_value; " * expected_asm *
                                " mov.b16 \$2, {report_value, 0}; }")
            @test spec.asm == expected_asm
            @test spec.constraints ==
                join([outletters; inletters; "~{memory}"], ",")
        end
    end
end

@testset "mbarrier history and target provenance" begin
    schema(mods) = only(s for s in PTX.MBARRIER_FORM_SCHEMAS if s.mods == mods)
    variant(mods, arity = 1) = schema(mods).variants[arity]
    @test variant((:init, :shared, :b64)).ptx_version == v"7.0"
    @test variant((:init, :shared, :b64)).min_sm == v"8.0"
    @test variant((:init, Symbol("layout::v1"), :shared, :b64)).ptx_version == v"9.3"
    @test variant((:init, Symbol("layout::v1"), :shared, :b64)).min_sm == v"9.0"
    @test variant((:arrive, :shared, :b64), 1).min_sm == v"8.0"
    @test variant((:arrive, :shared, :b64), 2).min_sm == v"9.0"
    @test variant((:arrive, :noComplete, :shared, :b64)).min_sm == v"8.0"
    @test variant((:try_wait, :shared, :b64)).ptx_version == v"7.8"
    @test variant((:try_wait, :shared, :b64)).min_sm == v"9.0"
    @test variant((:test_wait, :report,
                   Symbol("phase_type::primary"), :shared, :b64)).ptx_version == v"9.3"
    @test variant((:pending_count, Symbol("layout::v0"), :b64)).min_sm == v"9.0"
end

@testset "mbarrier representative result shapes and exact raw" begin
    pS = Core.LLVMPtr{UInt64, PTX.AS.Shared}
    cases = (
        ((:complete_tx, :relaxed, :cta, :shared, :b64), (pS, UInt32),
         Nothing, "mbarrier.complete_tx.relaxed.cta.shared.b64 [\$0], \$1;"),
        ((:arrive, Symbol("shared::cluster"), :b64), (pS,),
         Nothing, "mbarrier.arrive.shared::cluster.b64 _, [\$0];"),
        ((:test_wait, :acquire, :cta, :shared, :b64), (pS, UInt64),
         Bool, "mbarrier.test_wait.acquire.cta.shared.b64 \$0, [\$1], \$2;"),
        ((:pending_count, :b64), (UInt64,), UInt32,
         "mbarrier.pending_count.b64 \$0, \$1;"),
        ((:test_wait, :report_pred, Symbol("phase_type::primary"), :shared, :b64),
         (pS, UInt64), Tuple{Bool, Bool},
         "mbarrier.test_wait.phase_type::primary.shared.b64 \$0|\$1, [\$2], \$3;"),
        ((:try_wait, :report, :parity, Symbol("phase_type::primary"),
          :relaxed, :cluster, :shared, :b64),
         (pS, UInt32, UInt32), Tuple{Bool, Bool, UInt16},
         "{ .reg .b8 report_value; " *
         "mbarrier.try_wait.parity.phase_type::primary.relaxed.cluster.shared.b64 " *
         "\$0|\$1, report_value, [\$3], \$4, \$5; " *
         "mov.b16 \$2, {report_value, 0}; }"),
    )
    for (mods, args, rettype, asm) in cases
        spec = build_call(:mbarrier, mods, args)
        @test spec.rettype === rettype
        @test spec.asm == asm
        @test PTX.infer_rettype(:mbarrier, mods) === rettype
        @test PTX.lowering(Operation{:mbarrier, mods}(), args).tier in
              (:chain_asm, :intrinsic, :asm)
    end

    reportmods = (:test_wait, :report, Symbol("phase_type::primary"),
                  :shared, :b64)
    rawspec = build_call(:mbarrier, reportmods, (pS, UInt64); raw = true)
    @test rawspec.rettype === Tuple{Bool, Bool, UInt16}
    @test rawspec.asm ==
        "{ .reg .b8 report_value; " *
        "mbarrier.test_wait.phase_type::primary.shared.b64 " *
        "\$0|\$1, report_value, [\$3], \$4; " *
        "mov.b16 \$2, {report_value, 0}; }"
    @test rawspec.constraints == "=b,=b,=h,r,l,~{memory}"
    @test rawspec.convergent
    @test PTX.lowering(RawOperation{:mbarrier, reportmods}(),
                       (pS, UInt64)).tier === :chain_asm
    ir = PTX.convergent_asm_ir(rawspec.asm, rawspec.constraints,
                               rawspec.rettype, rawspec.passthrough_argtypes)
    @test occursin("define { i8, i8, i16 } @entry", ir)
    @test occursin("call { i8, i8, i16 } asm sideeffect", ir)
end

@testset "mbarrier grammar and carrier misses fail loud" begin
    pS = Core.LLVMPtr{UInt64, PTX.AS.Shared}
    pG = Core.LLVMPtr{UInt64, PTX.AS.Generic}
    bad_forms = (
        (:init, :b64, :shared),
        (:expect_tx, :relaxed, :shared, :b64),
        (:complete_tx, :cta, :relaxed, :shared, :b64),
        (:arrive, :acquire, :cta, :shared, :b64),
        (:arrive, :noComplete, Symbol("shared::cluster"), :b64),
        (:test_wait, Symbol("phase_type::conditional"), :shared, :b64),
        (:test_wait, Symbol("shared::cluster"), :b64),
        (:pending_count, Symbol("layout::v1"), :b64),
        (:check_layout, Symbol("layout::v1"), :shared, :b64),
        (:test_wait, :reporting, Symbol("phase_type::primary"), :shared, :b64),
    )
    for mods in bad_forms
        @test_throws ArgumentError PTX.infer_rettype(:mbarrier, mods)
        @test_throws ArgumentError build_call(:mbarrier, mods, ())
        @test_throws ArgumentError build_call(:mbarrier, mods, (); raw = true)
        @test PTX.lowering(Operation{:mbarrier, mods}(), ()).tier === :forbidden
        @test PTX.lowering(RawOperation{:mbarrier, mods}(), ()).tier === :forbidden
    end

    carrier_misses = (
        ((:init, :shared, :b64), (pG, UInt32)),
        ((:init, :b64), (pS, UInt32)),
        ((:init, :shared, :b64), (pS, UInt64)),
        ((:pending_count, :b64), (UInt32,)),
        ((:test_wait, :shared, :b64), (pS, UInt32)),
    )
    for (i, (mods, args)) in enumerate(carrier_misses)
        @test_throws ArgumentError build_call(:mbarrier, mods, args)
        @test_throws ArgumentError build_call(:mbarrier, mods, args; raw = true)
        # Existing high-level intrinsic wrappers intentionally accept Integer
        # for u32 counts/u64 states and normalize internally. The schema pins
        # the instruction-at-a-time fallback and raw surfaces; address-space
        # mistakes and forms without that wrapper remain forbidden.
        if i in (3, 5)
            @test PTX.lowering(Operation{:mbarrier, mods}(), args).tier === :intrinsic
        else
            @test PTX.lowering(Operation{:mbarrier, mods}(), args).tier === :forbidden
        end
        @test PTX.lowering(RawOperation{:mbarrier, mods}(), args).tier === :forbidden
    end

    # Integer address carriers are required for standalone transpilation of
    # register-address PTX; the audited role, not the Julia type, supplies [].
    intaddr = build_call(:mbarrier, (:init, :b64), (UInt64, UInt32))
    @test intaddr.asm == "mbarrier.init.b64 [\$0], \$1;"
    @test intaddr.constraints == "l,r,~{memory}"
    sharedaddr = build_call(:mbarrier, (:init, :shared, :b64),
                            (UInt32, UInt32))
    @test sharedaddr.asm == "mbarrier.init.shared.b64 [\$0], \$1;"
    @test sharedaddr.constraints == "r,r,~{memory}"
end

@testset "mbarrier transpiler preserves sinks and grouped destinations" begin
    source = """
    .version 9.3
    .target sm_90
    .address_size 64

    .visible .entry mbarrier_schema_probe()
    {
        .reg .b32 %r<3>;
        .reg .b64 %rd<4>;
        .reg .b8 %status;
        .reg .pred %p<5>;
        mbarrier.init.b64 [%rd0], 1;
        mbarrier.pending_count.b64 %r0, %rd1;
        mbarrier.test_wait.phase_type::primary.b64 %p0|%p1, [%rd0], %rd1;
        mbarrier.try_wait.parity.phase_type::primary.b64 %p2|%p3, %status, [%rd0], %r1, %r2;
        mbarrier.arrive.shared::cluster.b64 _, [%r0];
        ret;
    }
    """
    julia = PTX.ptx_to_julia(source)
    @test occursin("ptx\"mbarrier.init.b64\"(rd0, UInt32(1))", julia)
    @test occursin("r0 = ptx\"mbarrier.pending_count.b64\"(rd1)", julia)
    @test occursin("(p0, p1) = ptx\"mbarrier.test_wait.report_pred.phase_type::primary.b64\"(rd0, rd1)", julia)
    @test occursin("(p2, p3, status) = ptx\"mbarrier.try_wait.report.parity.phase_type::primary.b64\"(rd0, r1, r2)", julia)
    @test occursin("ptx\"mbarrier.arrive.shared::cluster.b64\"(r0)", julia)
    @test !occursin("_ =", julia)

    for text in (
        replace(source,
                "mbarrier.test_wait.phase_type::primary.b64 %p0|%p1, [%rd0], %rd1;" =>
                "mbarrier.test_wait.phase_type::conditional.b64 %p0, [%rd0], %rd1;"),
        replace(source,
                "mbarrier.arrive.shared::cluster.b64 _, [%r0];" =>
                "mbarrier.arrive.shared::cluster.b64 %rd3, [%r0];"),
        replace(source,
                ".reg .b8 %status;" => ".reg .b64 %status;"),
    )
        @test_throws ArgumentError PTX.ptx_to_julia(text)
    end
    # The output is real Julia syntax, not merely a plausible string: parse
    # and evaluate it in an isolated module so macro expansion and tuple
    # destructuring are both exercised.
    sandbox = Module(gensym(:MBarrierTranspiler))
    Core.eval(sandbox, :(using PTX))
    Core.eval(sandbox, Meta.parseall(julia))
    @test isdefined(sandbox, :mbarrier_schema_probe)
    @test length(Base.code_lowered(getfield(sandbox, :mbarrier_schema_probe),
                                   Tuple{})) == 1
end
