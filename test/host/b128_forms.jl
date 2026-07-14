using PTX: B128, Address, Operation, RawOperation, build_call, b128,
           b128_form_schema

function _expected_b128_form_keys()
    keys = Set{Tuple{Symbol,Tuple}}()
    add(op, mods) = push!(keys, (op, Tuple(mods)))
    one(x) = x === nothing ? () : (x,)
    states_ld = (nothing, :const, :global, :local, :param,
                 Symbol("param::entry"), Symbol("param::func"), :shared,
                 Symbol("shared::cta"), Symbol("shared::cluster"))
    states_st = (nothing, :global, :local, :param, Symbol("param::func"),
                 :shared, Symbol("shared::cta"), Symbol("shared::cluster"))
    memory_states = (nothing, :global, :shared, Symbol("shared::cta"),
                     Symbol("shared::cluster"))
    scopes = (:cta, :cluster, :gpu, :sys)
    l1s = (nothing, Symbol("L1::evict_normal"),
           Symbol("L1::evict_unchanged"), Symbol("L1::evict_first"),
           Symbol("L1::evict_last"), Symbol("L1::no_allocate"))
    prefetches = (nothing, Symbol("L2::64B"), Symbol("L2::128B"),
                  Symbol("L2::256B"))
    hints = (false, true)

    add(:mov, (:b128,))
    for weak in (false, true), state in states_ld
        weakpart = weak ? (:weak,) : ()
        lead = (weakpart..., one(state)...)
        for cop in (nothing, :ca, :cg, :cs, :lu, :cv), hint in hints,
            prefetch in prefetches
            (hint || prefetch !== nothing) && !(state in (nothing, :global)) &&
                continue
            hintpart = hint ? (Symbol("L2::cache_hint"),) : ()
            add(:ld, (lead..., one(cop)..., hintpart...,
                      one(prefetch)..., :b128))
        end
        for l1 in l1s, hint in hints, prefetch in prefetches
            (hint || prefetch !== nothing) && !(state in (nothing, :global)) &&
                continue
            hintpart = hint ? (Symbol("L2::cache_hint"),) : ()
            add(:ld, (lead..., one(l1)..., hintpart...,
                      one(prefetch)..., :b128))
        end
    end
    for state in (nothing, :global, :local, :shared,
                  Symbol("shared::cta"), Symbol("shared::cluster")),
        prefetch in prefetches
        prefetch !== nothing && !(state in (nothing, :global)) && continue
        add(:ld, (:volatile, one(state)..., one(prefetch)..., :b128))
    end
    for sem in (:relaxed, :acquire), scope in scopes, state in memory_states,
        l1 in l1s, hint in hints, prefetch in prefetches
        (hint || prefetch !== nothing) && !(state in (nothing, :global)) &&
            continue
        hintpart = hint ? (Symbol("L2::cache_hint"),) : ()
        add(:ld, (sem, scope, one(state)..., one(l1)..., hintpart...,
                  one(prefetch)..., :b128))
    end
    for sem in (:relaxed, :acquire), state in (nothing, :global)
        add(:ld, (:mmio, sem, :sys, one(state)..., :b128))
    end
    for cop in (nothing, :ca, :cg, :cs), hint in hints,
        prefetch in prefetches
        hintpart = hint ? (Symbol("L2::cache_hint"),) : ()
        add(:ld, (:global, one(cop)..., :nc, hintpart...,
                  one(prefetch)..., :b128))
    end
    for l1 in l1s, hint in hints, prefetch in prefetches
        hintpart = hint ? (Symbol("L2::cache_hint"),) : ()
        add(:ld, (:global, :nc, one(l1)..., hintpart...,
                  one(prefetch)..., :b128))
    end
    add(:ldu, (:b128,)); add(:ldu, (:global, :b128))

    for weak in (false, true), state in states_st
        weakpart = weak ? (:weak,) : ()
        lead = (weakpart..., one(state)...)
        for cop in (nothing, :wb, :cg, :cs, :wt), hint in hints
            hint && !(state in (nothing, :global)) && continue
            hintpart = hint ? (Symbol("L2::cache_hint"),) : ()
            add(:st, (lead..., one(cop)..., hintpart..., :b128))
        end
        for l1 in l1s, hint in hints
            hint && !(state in (nothing, :global)) && continue
            hintpart = hint ? (Symbol("L2::cache_hint"),) : ()
            add(:st, (lead..., one(l1)..., hintpart..., :b128))
        end
    end
    for state in (nothing, :global, :local, :shared,
                  Symbol("shared::cta"), Symbol("shared::cluster"))
        add(:st, (:volatile, one(state)..., :b128))
    end
    for sem in (:relaxed, :release), scope in scopes, state in memory_states,
        l1 in l1s, hint in hints
        hint && !(state in (nothing, :global)) && continue
        hintpart = hint ? (Symbol("L2::cache_hint"),) : ()
        add(:st, (sem, scope, one(state)..., one(l1)..., hintpart..., :b128))
    end
    for sem in (:relaxed, :release), state in (nothing, :global)
        add(:st, (:mmio, sem, :sys, one(state)..., :b128))
    end

    for sem in (nothing, :relaxed, :acquire, :release, :acq_rel),
        scope in (nothing, scopes...), state in memory_states,
        atomop in (:exch, :cas)
        add(:atom, (one(sem)..., one(scope)..., one(state)...,
                    atomop, :b128))
        atomop === :exch && state in (nothing, :global) &&
            add(:atom, (one(sem)..., one(scope)..., one(state)...,
                        atomop, Symbol("L2::cache_hint"), :b128))
    end

    add(:clusterlaunchcontrol, (:query_cancel, :is_canceled, :pred, :b128))
    add(:clusterlaunchcontrol,
        (:query_cancel, :get_first_ctaid, :v4, :b32, :b128))
    for dim in (:x, :y, :z)
        add(:clusterlaunchcontrol,
            (:query_cancel, Symbol("get_first_ctaid::", dim), :b32, :b128))
    end
    keys
end

@testset "independent exhaustive b128 form matrix" begin
    expected = _expected_b128_form_keys()
    actual = Set(keys(PTX.B128_FORM_SCHEMAS))
    @test isempty(setdiff(actual, expected))
    @test isempty(setdiff(expected, actual))
    histogram = Dict(op => count(key -> key[1] === op, expected)
                     for op in (:mov, :ld, :ldu, :st, :atom,
                                :clusterlaunchcontrol))
    @test histogram == Dict(:mov => 1, :ld => 1528, :ldu => 2, :st => 546,
                            :atom => 300, :clusterlaunchcontrol => 5)
end

const EXPECTED_B128_ACCEPTED = (
    (:mov, (:b128,), :mov, B128, (:b128,)),
    (:ld, (:b128,), :load, B128, (:address,)),
    (:ld, (:global, :b128), :load, B128, (:address,)),
    (:ld, (:weak, :global, :ca, :b128), :load, B128, (:address,)),
    (:ld, (:volatile, :local, :b128), :load, B128, (:address,)),
    (:ld, (:acquire, :sys, :global, :b128), :load, B128, (:address,)),
    (:ld, (:global, Symbol("L2::cache_hint"), :b128), :load, B128,
     (:address, :cache_policy)),
    (:ld, (:global, Symbol("L2::256B"), :b128), :load, B128, (:address,)),
    (:ld, (:global, :nc, :b128), :load, B128, (:address,)),
    (:ld, (:global, :cg, :nc, Symbol("L2::128B"), :b128), :load, B128,
     (:address,)),
    (:ld, (:mmio, :acquire, :sys, :global, :b128), :load, B128, (:address,)),
    (:ldu, (:b128,), :load, B128, (:address,)),
    (:ldu, (:global, :b128), :load, B128, (:address,)),
    (:st, (:b128,), :store, Nothing, (:address, :b128)),
    (:st, (:global, :wb, :b128), :store, Nothing, (:address, :b128)),
    (:st, (:volatile, :local, :b128), :store, Nothing, (:address, :b128)),
    (:st, (:release, :sys, :global, :b128), :store, Nothing,
     (:address, :b128)),
    (:st, (:global, Symbol("L2::cache_hint"), :b128), :store, Nothing,
     (:address, :b128, :cache_policy)),
    (:st, (:mmio, :release, :sys, :global, :b128), :store, Nothing,
     (:address, :b128)),
    (:atom, (:global, :exch, :b128), :exch, B128, (:address, :b128)),
    (:atom, (:acq_rel, :sys, :global, :cas, :b128), :cas, B128,
     (:address, :b128, :b128)),
    (:atom, (:global, :exch, Symbol("L2::cache_hint"), :b128), :exch,
     B128, (:address, :b128, :cache_policy)),
    (:clusterlaunchcontrol, (:query_cancel, :is_canceled, :pred, :b128),
     :query_pred, Bool, (:b128,)),
    (:clusterlaunchcontrol,
     (:query_cancel, :get_first_ctaid, :v4, :b32, :b128), :query_v4,
     NTuple{4,UInt32}, (:b128,)),
    (:clusterlaunchcontrol,
     (:query_cancel, Symbol("get_first_ctaid::x"), :b32, :b128),
     :query_dim, UInt32, (:b128,)),
    (:clusterlaunchcontrol,
     (:query_cancel, Symbol("get_first_ctaid::y"), :b32, :b128),
     :query_dim, UInt32, (:b128,)),
    (:clusterlaunchcontrol,
     (:query_cancel, Symbol("get_first_ctaid::z"), :b32, :b128),
     :query_dim, UInt32, (:b128,)),
)

@testset "closed PTX 9.3 scalar-b128 grammar and carrier" begin
    @test B128 === NTuple{2,UInt64}
    @test b128(0x0123456789abcdef, 0xfedcba9876543210) ==
          (0x0123456789abcdef, 0xfedcba9876543210)
    @test b128(0x89abcdef, 0x01234567, 0x76543210, 0xfedcba98) ==
          (0x0123456789abcdef, 0xfedcba9876543210)
    for (op, mods, kind, result, operands) in EXPECTED_B128_ACCEPTED
        schema = b128_form_schema(op, mods)
        @test schema !== nothing
        @test schema.kind === kind
        @test schema.result === result
        @test schema.operands == operands
        @test startswith(schema.section, "ptx/9-instruction-set/")
    end
end

const REJECTED_B128_FORMS = (
    (:ld, (:v2, :b128)),              # vectors cannot exceed 128 bits
    (:ld, (Symbol("L2::evict_last"), :b128)), # only wide v8.b32/v4.b64
    (:ldu, (:shared, :b128)),
    (:st, (:v2, :b128)),
    (:atom, (:global, :add, :b128)),  # Table 35: exch/cas only
    (:atom, (:shared, :cas, Symbol("L2::cache_hint"), :b128)),
    (:clusterlaunchcontrol, (:query_cancel, :is_canceled, :b128)),
    (:clusterlaunchcontrol,
     (:query_cancel, Symbol("get_first_ctaid::w"), :b32, :b128)),
)

@testset "b128 grammar misses fail loud in direct, raw, and lowering" begin
    A = Address{UInt64}
    for (op, mods) in REJECTED_B128_FORMS
        args = op === :atom ? (A, B128) :
               op === :clusterlaunchcontrol ? (B128,) : (A,)
        ordinary = Operation{op,mods}()
        raw = RawOperation{op,mods}()
        @test b128_form_schema(op, mods) === nothing
        @test PTX.lowering(ordinary, args).tier === :forbidden
        @test PTX.lowering(raw, args).tier === :forbidden
        @test_throws ArgumentError build_call(op, mods, args)
        @test_throws ArgumentError build_call(op, mods, args; raw = true)
    end
end

@testset "b128 exact ABI rendering" begin
    A = Address{UInt64}
    ld = build_call(:ld, (:global, :b128), (A,))
    @test ld.rettype === B128
    @test ld.constraints == "=l,=l,l,~{memory}"
    @test occursin(".reg .b128 b128_result", ld.asm)
    @test occursin("ld.global.b128 b128_result, [\$2]", ld.asm)
    @test occursin("mov.b128 {\$0, \$1}, b128_result", ld.asm)

    st = build_call(:st, (:global, :b128), (A, B128))
    @test st.rettype === Nothing
    @test st.constraints == "l,l,l,~{memory}"
    @test occursin("mov.b128 b128_value1, {\$1, \$2}", st.asm)
    @test occursin("st.global.b128 [\$0], b128_value1", st.asm)

    atom = build_call(:atom, (:global, :cas, :b128), (A, B128, B128))
    @test atom.rettype === B128
    @test atom.constraints == "=l,=l,l,l,l,l,l,~{memory}"
    @test occursin("atom.global.cas.b128 b128_result, [\$2], b128_value1, b128_value2", atom.asm)

    query = build_call(:clusterlaunchcontrol,
        (:query_cancel, :get_first_ctaid, :v4, :b32, :b128), (B128,))
    @test query.rettype === NTuple{4,UInt32}
    @test query.constraints == "=r,=r,=r,=r,l,l,~{memory}"
    @test occursin("{\$0, \$1, \$2, \$3}, b128_value1", query.asm)

    for (op, mods, _, _, operands) in EXPECTED_B128_ACCEPTED
        argtypes = Tuple(kind === :address ? A :
                         kind === :b128 ? B128 : UInt64 for kind in operands)
        @test PTX.lowering(Operation{op,mods}(), argtypes).tier === :chain_asm
        @test PTX.lowering(RawOperation{op,mods}(), argtypes).tier === :chain_asm
    end
end

@testset "every b128 ledger cell reaches direct and raw emitters" begin
    A = Address{UInt64}
    ordered = sort!(collect(values(PTX.B128_FORM_SCHEMAS));
                    by = schema -> (string(schema.op), join(schema.mods, ".")))
    @test length(ordered) == 2382
    for schema in ordered
        argtypes = Tuple(kind === :address ? A :
                         kind === :b128 ? B128 : UInt64
                         for kind in schema.operands)
        direct_spec = build_call(schema.op, schema.mods, argtypes)
        raw_spec = build_call(schema.op, schema.mods, argtypes; raw = true)
        @test direct_spec.rettype === schema.result
        @test raw_spec.rettype === schema.result
        @test direct_spec.asm == raw_spec.asm
        @test occursin(PTX.build_head(schema.op, schema.mods), direct_spec.asm)
    end
    # Do not add lowering() reflection to this 2,382-cell loop. Each distinct
    # Operation type triggers Julia specialization; an attempted exhaustive
    # reflection sweep ran for more than six minutes. The 27 cross-family
    # accepted representatives above exercise ordinary/raw lowering, while
    # every negative grammar miss is reflected exhaustively. build_call is the
    # shared emitter used by both lowering routes and remains exhaustive here.
end

@testset "b128 transpiler closes declaration and result roles" begin
    src = """.version 8.6
    .target sm_100
    .address_size 64
    .visible .entry b128_probe()
    {
      .reg .b64 %rd<4>;
      .reg .b32 %r<5>;
      .reg .pred %p;
      mov.b128 %handle, {%rd0, %rd1};
      clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 %p, %handle;
      clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%r0, %r1, %r2, %r3}, %handle;
      ret;
    }
    """
    julia = PTX.ptx_to_julia(src)
    @test occursin("handle = ptx\"mov.b128\"((rd0, rd1))", julia)
    @test occursin("p = ptx\"clusterlaunchcontrol.query_cancel.is_canceled.pred.b128\"(handle)", julia)
    @test occursin("(r0, r1, r2, r3) = ptx\"clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128\"(handle)", julia)
    @test Meta.parseall(julia) isa Expr

    bad = replace(src, "{%rd0, %rd1}" => "{%r0, %r1}")
    @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(bad)
    unpack = replace(src,
        "mov.b128 %handle, {%rd0, %rd1};" =>
        ".reg .b128 %handle;\nmov.b128 {%rd0, %rd1}, %handle;")
    @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(unpack)
end
