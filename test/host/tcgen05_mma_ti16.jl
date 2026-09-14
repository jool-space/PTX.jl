include(joinpath(@__DIR__, "tcgen05_ptx94_mma_defs.jl"))

# Independent PTX ISA 9.4 §9.7.18.10 `.kind::ti16` oracle: the integer
# schema (no scale-input-d) with an optional B-side collector on dense and
# sparse forms, the addressed `collector::bN::op` on ws/ws.sp, and the
# dense/sp ashift restrictions. The production grid is not the source.
const _T5_TI16 = Symbol("kind::ti16")

_t5_ti16_mods(cg; sp = false, ashift = false, a = nothing, b = nothing) =
    (:mma, (sp ? (:sp,) : ())..., Symbol("cta_group::", cg), _T5_TI16,
     (ashift ? (:ashift,) : ())..., _t5_94_opt(a)..., _t5_94_opt(b)...)

_t5_ti16_ws_colls() = begin
    out = Any[nothing]
    for buf in 0:3, op in (:discard, :lastuse, :fill, :use)
        buf == 0 && op === :discard && continue
        push!(out, Symbol("collector::b$buf::$op"))
    end
    out
end

@testset "tcgen05 ti16 mma closed callable surface" begin
    expected = Set{Tuple{Vararg{Symbol}}}()
    methods_checked = 0
    for cg in (1, 2), sp in (false, true), ashift in (false, true),
            a in (ashift ? _T5_94_COLL_A[1:2] : _T5_94_COLL_A),
            b in _T5_94_COLL_B
        mods = _t5_ti16_mods(cg; sp, ashift, a, b)
        push!(expected, mods)
        maskN = cg == 1 ? 4 : 8
        meta = sp ? (UInt32,) : ()
        for a_tmem in (ashift ? (true,) : (false, true))
            aT = a_tmem ? UInt32 : UInt64
            _t5_94_check(mods, (UInt32, aT, UInt64, meta..., UInt32, Bool),
                         _t5_94_schema(; a_tmem, meta = sp))
            _t5_94_check(mods,
                         (UInt32, aT, UInt64, meta..., UInt32,
                          NTuple{maskN, UInt32}, Bool),
                         _t5_94_schema(; a_tmem, meta = sp, maskN))
            methods_checked += 2
        end
    end
    for sp in (false, true), coll in _t5_ti16_ws_colls()
        mods = (:mma, :ws, (sp ? (:sp,) : ())..., Symbol("cta_group::1"),
                _T5_TI16, _t5_94_opt(coll)...)
        push!(expected, mods)
        meta = sp ? (UInt32,) : ()
        for a_tmem in (false, true)
            aT = a_tmem ? UInt32 : UInt64
            _t5_94_check(mods, (UInt32, aT, UInt64, meta..., UInt32, Bool),
                         _t5_94_schema(; a_tmem, meta = sp))
            _t5_94_check(mods,
                         (UInt32, aT, UInt64, meta..., UInt32, Bool, UInt64),
                         _t5_94_schema(; a_tmem, meta = sp, zero_col = true))
            methods_checked += 2
        end
    end
    @test length(expected) == 128
    @test methods_checked == 448
    @test Set(PTX.wrapper_asm_forms(:tcgen05_mma_ti16)) == expected
    @test count(m -> :ws in m, expected) == 32
    @test count(m -> :sp in m && :ws ∉ m, expected) == 48
end

@testset "tcgen05 ti16 mma rejects grammar and ABI misses" begin
    base = _t5_ti16_mods(1; b = Symbol("collector::b::fill"))
    misses = (
        # No scale-input-d on the integer kinds.
        (base, (UInt32, UInt64, UInt64, UInt32, Bool, Val{5})),
        # ashift is TMEM-A only and forbids A fill/use.
        (_t5_ti16_mods(1; ashift = true), (UInt32, UInt64, UInt64, UInt32, Bool)),
        (_t5_ti16_mods(1; ashift = true, a = Symbol("collector::a::fill")),
         (UInt32, UInt32, UInt64, UInt32, Bool)),
        # The B collector follows the A collector; discard is never spelled.
        ((:mma, Symbol("cta_group::1"), _T5_TI16, Symbol("collector::b::fill"),
          Symbol("collector::a::fill")), (UInt32, UInt64, UInt64, UInt32, Bool)),
        (_t5_ti16_mods(1; b = Symbol("collector::b::discard")),
         (UInt32, UInt64, UInt64, UInt32, Bool)),
        # .kind::i8 has no B collector (§9.7.18.10 blocks 5/6).
        ((:mma, Symbol("cta_group::1"), Symbol("kind::i8"),
          Symbol("collector::b::fill")), (UInt32, UInt64, UInt64, UInt32, Bool)),
        # sp metadata is mandatory and sits before idesc.
        (_t5_ti16_mods(1; sp = true), (UInt32, UInt64, UInt64, UInt32, Bool)),
        # Mask width follows cta_group.
        (base, (UInt32, UInt64, UInt64, UInt32, NTuple{8, UInt32}, Bool)),
        (_t5_ti16_mods(2; b = Symbol("collector::b::use")),
         (UInt32, UInt64, UInt64, UInt32, NTuple{4, UInt32}, Bool)),
        # ws is cta_group::1 with the addressed collector; the dense
        # spelling is not ws grammar and ws has no mask.
        ((:mma, :ws, Symbol("cta_group::2"), _T5_TI16),
         (UInt32, UInt64, UInt64, UInt32, Bool)),
        ((:mma, :ws, Symbol("cta_group::1"), _T5_TI16,
          Symbol("collector::b::fill")), (UInt32, UInt64, UInt64, UInt32, Bool)),
        ((:mma, :ws, Symbol("cta_group::1"), _T5_TI16),
         (UInt32, UInt64, UInt64, UInt32, NTuple{4, UInt32}, Bool)),
        # Block-scale has no ti16 kind.
        ((:mma, Symbol("cta_group::1"), _T5_TI16, :block_scale, :block32),
         (UInt32, UInt64, UInt64, UInt32, UInt32, UInt32, Bool)),
    )
    for (mods, args) in misses
        _t5_94_forbidden(mods, args)
    end
end
