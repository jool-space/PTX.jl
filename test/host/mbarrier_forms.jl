using PTX: Operation, RawOperation, build_call

# Independent reconstruction of PTX ISA 9.3 §9.7.14.16.  This deliberately
# does not consume MBARRIER_FORM_SCHEMAS or its production helper tables: a
# grammar edit must agree with this separately reviewed form/result/operand
# inventory.
function _expected_mbarrier_forms()
    expected = Dict{Tuple,NamedTuple}()
    add!(mods, ptxmods, destination, variants, space;
         provenance = :canonical) = begin
        @assert !haskey(expected, mods)
        expected[mods] = (; ptxmods, destination, variants, space, provenance)
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
        if kind !== :cluster
            sinkmods = (subop, :sink, sem..., space..., :b64)
            add!(sinkmods, mods, :sink,
                 ((:address,), (:address, :u32)), kind)
        end
    end
    for subop in (:arrive, :arrive_drop), sem in arrive_pairs,
        (space, kind) in all_spaces
        mods = (subop, :expect_tx, sem..., space..., :b64)
        destination = kind === :cluster ? :remote_sink : :state
        add!(mods, mods, destination, ((:address, :u32),), kind)
        if kind !== :cluster
            sinkmods = (subop, :sink, :expect_tx, sem..., space..., :b64)
            add!(sinkmods, mods, :sink, ((:address, :u32),), kind)
        end
    end
    for subop in (:arrive, :arrive_drop), sem in ((), (:release, :cta)),
        (space, kind) in local_spaces
        mods = (subop, :noComplete, sem..., space..., :b64)
        add!(mods, mods, :state, ((:address, :u32),), kind)
        sinkmods = (subop, :sink, :noComplete, sem..., space..., :b64)
        add!(sinkmods, mods, :sink, ((:address, :u32),), kind)
    end

    # The instruction subsection's two space-before-sem/scope examples are
    # accepted by CUDA 13.3 ptxas. They are aliases only: emitted PTX uses the
    # syntax-block order, and sink/state remain explicit ABI choices.
    compat = (:arrive_drop, Symbol("shared::cta"),
              :release, :cluster, :b64)
    canonical = (:arrive_drop, :release, :cluster,
                 Symbol("shared::cta"), :b64)
    add!(compat, canonical, :state,
         ((:address,), (:address, :u32)), :cta;
         provenance = :ptxas_compat)
    add!((:arrive_drop, :sink, Symbol("shared::cta"),
          :release, :cluster, :b64), canonical, :sink,
         ((:address,), (:address, :u32)), :cta;
         provenance = :ptxas_compat)
    compat = (:arrive_drop, :expect_tx, Symbol("shared::cta"),
              :relaxed, :cluster, :b64)
    canonical = (:arrive_drop, :expect_tx, :relaxed, :cluster,
                 Symbol("shared::cta"), :b64)
    add!(compat, canonical, :state, ((:address, :u32),), :cta;
         provenance = :ptxas_compat)
    add!((:arrive_drop, :sink, :expect_tx, Symbol("shared::cta"),
          :relaxed, :cluster, :b64), canonical, :sink,
         ((:address, :u32),), :cta; provenance = :ptxas_compat)

    # PTX ISA 9.4 multicast::cluster::32b: shared::cluster only, mask is
    # the mandatory trailing u32; multicast arrive keeps the `_` sink.
    multicast = Symbol("multicast::cluster::32b")
    cluster_space = Symbol("shared::cluster")
    for subop in (:expect_tx, :complete_tx), sem in tx_pairs
        mods = (subop, sem..., cluster_space, multicast, :b64)
        add!(mods, mods, :none, ((:address, :u32, :u32),), :cluster)
    end
    for subop in (:arrive, :arrive_drop), sem in arrive_pairs
        mods = (subop, sem..., cluster_space, multicast, :b64)
        add!(mods, mods, :remote_sink,
             ((:address, :u32), (:address, :u32, :u32)), :cluster)
        mods = (subop, :expect_tx, sem..., cluster_space, multicast, :b64)
        add!(mods, mods, :remote_sink, ((:address, :u32, :u32),), :cluster)
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
    destination in (:none, :sink, :remote_sink) ? Nothing :
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
    destination === :sink && return (["_"], String[])
    destination === :remote_sink && return (["_"], String[])
    destination === :state && return (["\$0"], ["=l"])
    destination === :predicate && return (["\$0"], ["=b"])
    destination === :count && return (["\$0"], ["=r"])
    destination === :report_pred && return (["\$0|\$1"], ["=b", "=b"])
    destination === :report &&
        return (["\$0|\$1", "report_value"], ["=b", "=b", "=h"])
    error("bad destination $destination")
end

# Independently derive each history floor from feature-introduction statements
# in §9.7.14.16. This deliberately does not consume production constructor
# tables, cross-product loops, or history helpers: the closed grammar oracle
# above and this feature oracle fail independently.
function _isa_mbarrier_floor(schema, variant)
    mods = schema.ptxmods
    op = first(mods)

    ptx, sm = if op in (:init, :inval, :arrive, :arrive_drop,
                        :test_wait, :pending_count)
        (v"7.0", v"8.0")
    elseif op === :try_wait
        (v"7.8", v"9.0")
    elseif op in (:expect_tx, :complete_tx)
        (v"8.0", v"9.0")
    elseif op === :check_layout
        (v"9.3", v"9.0")
    else
        error("unreviewed mbarrier operation: $op")
    end

    advance!(p, s = sm) = (ptx = max(ptx, p); sm = max(sm, s))

    # PTX 7.1 independently introduced result discard for mbarrier.arrive
    # and phase-parity waits. arrive_drop admitted `_` at its PTX 7.0
    # introduction (also confirmed with CUDA 13.3 ptxas at version 7.0).
    schema.destination === :sink && op === :arrive && advance!(v"7.1")
    :parity in mods && advance!(v"7.1")

    Symbol("shared::cta") in mods && advance!(v"7.8")

    # The optional count on ordinary arrive/arrive_drop is distinct from the
    # mandatory count of noComplete and txCount of expect_tx.
    if op in (:arrive, :arrive_drop) &&
       !(:noComplete in mods) && !(:expect_tx in mods) &&
       variant.operands == (:address, :u32)
        advance!(v"7.8", v"9.0")
    end

    (:expect_tx in mods || op in (:expect_tx, :complete_tx)) &&
        advance!(v"8.0", v"9.0")
    Symbol("shared::cluster") in mods && advance!(v"8.0", v"9.0")

    # Bare :cta/:cluster are synchronization scopes; state-space
    # subqualifiers are distinct Symbols containing `shared::`.
    (:cta in mods || :cluster in mods) && advance!(v"8.0")
    :cluster in mods && advance!(v"8.0", v"9.0")
    # expect_tx/complete_tx were born in 8.0 with `.relaxed`; `.relaxed` was
    # added to arrive/drop/wait later, in 8.6.
    (:relaxed in mods && !(op in (:expect_tx, :complete_tx))) &&
        advance!(v"8.6", v"9.0")

    any(m -> startswith(String(m), "layout::"), mods) &&
        advance!(v"9.3", v"9.0")
    any(m -> startswith(String(m), "phase_type::"), mods) &&
        advance!(v"9.3", v"9.0")
    schema.destination in (:report_pred, :report) &&
        advance!(v"9.3", v"9.0")
    # PTX ISA 9.4: multicast::cluster::32b is an sm_107f family feature
    # (10.7 is the lowest admitting family floor).
    Symbol("multicast::cluster::32b") in mods && advance!(v"9.4", v"10.7")

    (; ptx, sm)
end

@testset "closed mbarrier grammar and ABI ledger" begin
    expected = _expected_mbarrier_forms()
    actual = Dict(schema.mods => schema for schema in PTX.MBARRIER_FORM_SCHEMAS)
    @test length(expected) == 506
    @test length(actual) == 506
    @test Set(keys(actual)) == Set(keys(expected))
    @test count(k -> first(k) === :init, keys(actual)) == 9
    @test count(k -> first(k) === :inval, keys(actual)) == 3
    @test count(k -> first(k) in (:expect_tx, :complete_tx), keys(actual)) == 30
    @test count(k -> first(k) in (:arrive, :arrive_drop), keys(actual)) == 188
    @test count(k -> first(k) in (:test_wait, :try_wait), keys(actual)) == 270
    @test count(k -> first(k) === :pending_count, keys(actual)) == 2
    @test count(k -> first(k) === :check_layout, keys(actual)) == 4
    @test count(s -> s.provenance === :ptxas_compat, values(actual)) == 4

    for (mods, want) in expected
        schema = actual[mods]
        @test schema.ptxmods == want.ptxmods
        @test schema.destination === want.destination
        @test schema.space === want.space
        @test schema.provenance === want.provenance
        @test Tuple(v.operands for v in schema.variants) == want.variants
        @test schema.section ==
            "ptx/9-instruction-set/9.7.15.16-parallel-synchronization-and-communication-instructions-mbarrier.md"

        for operands in want.variants
            argtypes = Tuple(_mb_argtype(kind, want.space) for kind in operands)
            spec = build_call(:mbarrier, mods, argtypes)
            @test spec.rettype === _mb_rettype(want.destination)
            @test spec.side_effects
            @test spec.convergent

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
    @test variant((:arrive, :sink, :b64), 1).ptx_version == v"7.1"
    @test variant((:arrive_drop, :sink, :b64), 1).ptx_version == v"7.0"
    @test variant((:arrive, :noComplete, :shared, :b64)).min_sm == v"8.0"
    @test variant((:arrive, :sink, :noComplete, :shared, :b64)).ptx_version == v"7.1"
    @test variant((:try_wait, :shared, :b64)).ptx_version == v"7.8"
    @test variant((:try_wait, :shared, :b64)).min_sm == v"9.0"
    @test variant((:test_wait, :report,
                   Symbol("phase_type::primary"), :shared, :b64)).ptx_version == v"9.3"
    @test variant((:pending_count, Symbol("layout::v0"), :b64)).min_sm == v"9.0"
end

@testset "independent exhaustive mbarrier history floors" begin
    @test length(PTX.MBARRIER_FORM_SCHEMAS) == 506
    @test sum(length(s.variants) for s in PTX.MBARRIER_FORM_SCHEMAS) == 723
    for schema in PTX.MBARRIER_FORM_SCHEMAS, variant in schema.variants
        want = _isa_mbarrier_floor(schema, variant)
        label = "mbarrier.$(join(schema.mods, '.')) $(variant.operands)"
        @testset "$label" begin
            @test variant.ptx_version == want.ptx
            @test variant.min_sm == want.sm
        end
    end
end

@testset "every exact mbarrier wrapper is schema-backed and convergent" begin
    layout0 = Symbol("layout::v0")
    layout1 = Symbol("layout::v1")
    primary = Symbol("phase_type::primary")
    conditional = Symbol("phase_type::conditional")
    cluster = Symbol("shared::cluster")
    cta = Symbol("shared::cta")
    multicast32 = Symbol("multicast::cluster::32b")
    expected = Set((
        (:init, :shared, :b64),
        (:inval, :shared, :b64),
        (:arrive, :shared, :b64),
        (:arrive, :noComplete, :shared, :b64),
        (:arrive, :expect_tx, :shared, :b64),
        (:expect_tx, :shared, :b64),
        (:test_wait, :shared, :b64),
        (:test_wait, :parity, :shared, :b64),
        (:try_wait, :shared, :b64),
        (:try_wait, :parity, :shared, :b64),
        (:arrive, cluster, :b64),
        (:arrive, :expect_tx, cluster, :b64),
        (:init, layout0, :shared, :b64),
        (:init, layout1, :shared, :b64),
        (:check_layout, layout0, cta, :b64),
        (:check_layout, layout1, cta, :b64),
        (:test_wait, :report, primary, :shared, :b64),
        (:test_wait, :report, :parity, primary, :shared, :b64),
        (:try_wait, :report, primary, :shared, :b64),
        (:try_wait, :report, :parity, primary, :shared, :b64),
        (:test_wait, :parity, conditional, :shared, :b64),
        (:try_wait, :parity, conditional, :shared, :b64),
        # PTX ISA 9.4 cluster multicast (sm_107f, spelled-only).
        (:arrive, cluster, multicast32, :b64),
        (:arrive, :expect_tx, cluster, multicast32, :b64),
        (:expect_tx, cluster, multicast32, :b64),
    ))
    actual = Set{Tuple}()
    PTX._visit_operation_methods() do _, op, mods
        op === :mbarrier && push!(actual, mods)
    end
    @test length(expected) == 25
    @test actual == expected
    for mods in actual
        @test PTX.schema(PTX.MBarrierLedger(), :mbarrier, mods) !== nothing
    end

    # These fifteen wrappers have no NVVM spelling. Their convenience methods
    # normalize Integer inputs, then delegate back to the schema emitter. Pin
    # the actual typed body so a future hand-written @asmcall cannot silently
    # lose the call-site optimizer barrier again.
    pS = Core.LLVMPtr{UInt64, PTX.AS.Shared}
    asm_wrappers = (
        ((:arrive, cluster, :b64), (pS,)),
        ((:arrive, :expect_tx, cluster, :b64), (pS, UInt32)),
        ((:init, layout0, :shared, :b64), (pS, UInt32)),
        ((:init, layout1, :shared, :b64), (pS, Int)),
        ((:check_layout, layout0, cta, :b64), (pS,)),
        ((:check_layout, layout1, cta, :b64), (pS,)),
        ((:test_wait, :report, primary, :shared, :b64), (pS, UInt64)),
        ((:test_wait, :report, :parity, primary, :shared, :b64),
         (pS, UInt32)),
        ((:try_wait, :report, primary, :shared, :b64), (pS, Int)),
        ((:try_wait, :report, :parity, primary, :shared, :b64),
         (pS, UInt32)),
        ((:test_wait, :parity, conditional, :shared, :b64), (pS, UInt32)),
        ((:try_wait, :parity, conditional, :shared, :b64), (pS, Int)),
        ((:arrive, cluster, multicast32, :b64), (pS, UInt32)),
        ((:arrive, :expect_tx, cluster, multicast32, :b64),
         (pS, UInt32, Int)),
        ((:expect_tx, cluster, multicast32, :b64), (pS, Int, UInt32)),
    )
    @test length(asm_wrappers) == 15
    for (mods, argtypes) in asm_wrappers
        ci, rettype = first(Base.code_typed(Operation{:mbarrier, mods}(),
                                            argtypes))
        schema = PTX.schema(PTX.MBarrierLedger(), :mbarrier, mods)
        @test rettype === _mb_rettype(schema.destination)
        @test occursin("asm sideeffect", string(ci))
        @test occursin("convergent nomerge", string(ci))
    end
end

@testset "mbarrier representative result shapes and exact raw" begin
    pS = Core.LLVMPtr{UInt64, PTX.AS.Shared}
    cases = (
        ((:complete_tx, :relaxed, :cta, :shared, :b64), (pS, UInt32),
         Nothing, "mbarrier.complete_tx.relaxed.cta.shared.b64 [\$0], \$1;"),
        ((:arrive, :sink, :release, :cluster, :b64), (UInt64,),
         Nothing, "mbarrier.arrive.release.cluster.b64 _, [\$0];"),
        ((:arrive, :sink, :noComplete, :shared, :b64), (pS, UInt32),
         Nothing, "mbarrier.arrive.noComplete.shared.b64 _, [\$0], \$1;"),
        ((:arrive, Symbol("shared::cluster"), :b64), (pS,),
         Nothing, "mbarrier.arrive.shared::cluster.b64 _, [\$0];"),
        ((:arrive_drop, :sink, Symbol("shared::cta"), :release, :cluster, :b64),
         (pS, UInt32), Nothing,
         "mbarrier.arrive_drop.release.cluster.shared::cta.b64 _, [\$0], \$1;"),
        ((:arrive_drop, :expect_tx, Symbol("shared::cta"),
          :relaxed, :cluster, :b64), (pS, UInt32), UInt64,
         "mbarrier.arrive_drop.expect_tx.relaxed.cluster.shared::cta.b64 " *
         "\$0, [\$1], \$2;"),
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
        @test spec.convergent
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
        # Existing high-level exact wrappers intentionally accept Integer
        # for u32 counts/u64 states and normalize internally. The schema pins
        # the instruction-at-a-time fallback and raw surfaces; address-space
        # mistakes and forms without that wrapper remain forbidden.
        if i in (3, 5)
            @test PTX.lowering(Operation{:mbarrier, mods}(), args).tier === :asm
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

    # PTX ISA 9.4 multicast: spelled exactly as the ISA example
    # (`..._, [remoteAddr32], 0xa;`), remote sink `_`, mask trailing.
    mc = build_call(:mbarrier,
                    (:arrive, :release, :cta, Symbol("shared::cluster"),
                     Symbol("multicast::cluster::32b"), :b64),
                    (UInt32, UInt32))
    @test mc.asm ==
          "mbarrier.arrive.release.cta.shared::cluster.multicast::cluster::32b.b64 _, [\$0], \$1;"
    @test mc.constraints == "r,r,~{memory}"
    @test mc.rettype === Nothing

    # Explicit integer address roles compose with the mbarrier schema instead
    # of falling through to the generic address contract. Pin a state result,
    # a sink result, and a wait predicate; each Address must render one pair
    # of brackets and be unwrapped before reaching LLVM inline assembly.
    A32, A64 = PTX.Address{UInt32}, PTX.Address{UInt64}
    address_cases = (
        ((:arrive, :shared, :b64), (A32,),
         "mbarrier.arrive.shared.b64 \$0, [\$1];", UInt64, (UInt32,)),
        ((:arrive, :sink, :release, :cluster, :b64), (A64,),
         "mbarrier.arrive.release.cluster.b64 _, [\$0];", Nothing, (UInt64,)),
        ((:test_wait, :shared, :b64), (A32, UInt64),
         "mbarrier.test_wait.shared.b64 \$0, [\$1], \$2;", Bool,
         (UInt32, UInt64)),
    )
    for (mods, argtypes, asm, rettype, passthrough) in address_cases
        spec = build_call(:mbarrier, mods, argtypes)
        @test spec.asm == asm
        @test !occursin("[[", spec.asm)
        @test spec.rettype === rettype
        @test spec.passthrough_argtypes === passthrough
        @test spec.passthrough_unwrap_address[1]
    end
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
        mbarrier.init.b64 [%rd0+8], 1;
        mbarrier.pending_count.b64 %r0, %rd1;
        mbarrier.test_wait.phase_type::primary.b64 %p0|%p1, [%rd0], %rd1;
        mbarrier.try_wait.parity.phase_type::primary.b64 %p2|%p3, %status, [%rd0], %r1, %r2;
        mbarrier.arrive.release.cluster.b64 _, [%rd2];
        mbarrier.arrive.noComplete.shared::cta.b64 _, [%r0], %r1;
        mbarrier.arrive_drop.shared::cta.release.cluster.b64 _, [%r0], %r1;
        mbarrier.arrive.shared::cluster.b64 _, [%r0];
        {
            .reg .pred complete;
            mbarrier.try_wait.parity.shared.b64 complete, [%rd0], %r1;
        }
        ret;
    }
    """
    julia = PTX.ptx_to_julia(source)
    @test occursin("ptx\"mbarrier.init.b64\"(address(rd0), UInt32(1))", julia)
    @test occursin("ptx\"mbarrier.init.b64\"(address(rd0 + 8), UInt32(1))", julia)
    @test occursin("r0 = ptx\"mbarrier.pending_count.b64\"(rd1)", julia)
    @test occursin("(p0, p1) = ptx\"mbarrier.test_wait.report_pred.phase_type::primary.b64\"(address(rd0), rd1)", julia)
    @test occursin("(p2, p3, status) = ptx\"mbarrier.try_wait.report.parity.phase_type::primary.b64\"(address(rd0), r1, r2)", julia)
    @test occursin("ptx\"mbarrier.arrive.sink.release.cluster.b64\"(address(rd2))", julia)
    @test occursin("ptx\"mbarrier.arrive.sink.noComplete.shared::cta.b64\"(address(r0), r1)", julia)
    @test occursin("ptx\"mbarrier.arrive_drop.sink.shared::cta.release.cluster.b64\"(address(r0), r1)", julia)
    @test occursin("ptx\"mbarrier.arrive.shared::cluster.b64\"(address(r0))", julia)
    @test occursin("complete = ptx\"mbarrier.try_wait.parity.shared.b64\"(address(rd0), r1)", julia)
    @test !occursin("_ =", julia)
    @test !occursin(r"\w+\s*=\s*ptx\"mbarrier\.init", julia)

    for text in (
        replace(source,
                "mbarrier.test_wait.phase_type::primary.b64 %p0|%p1, [%rd0], %rd1;" =>
                "mbarrier.test_wait.phase_type::conditional.b64 %p0, [%rd0], %rd1;"),
        replace(source,
                "mbarrier.arrive.shared::cluster.b64 _, [%r0];" =>
                "mbarrier.arrive.shared::cluster.b64 %rd3, [%r0];"),
        replace(source,
                ".reg .b8 %status;" => ".reg .b64 %status;"),
        replace(source,
                "mbarrier.init.b64 [%rd0], 1;" =>
                "mbarrier.init.b64 [%rd0, {%r1}], 1;"),
    )
        @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(text)
    end

    for text in (
        replace(source, ".reg .pred complete;" => ".reg .b32 complete;"),
        replace(source, ".reg .pred complete;" => ""),
        replace(source,
                "mbarrier.try_wait.parity.shared.b64 complete," =>
                "mbarrier.try_wait.parity.shared.b64 _,"),
    )
        @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(text)
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
