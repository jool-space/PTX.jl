include(joinpath(@__DIR__, "tcgen05_ptx94_mma_defs.jl"))

# Independent PTX ISA 9.4 §9.7.18.10 `.collector::b::*` oracle for the
# non-ws kinds: the float kinds (blocks 3, dense and sp) with every A
# collector, the ashift restriction, scale-input-d on f16/tf32 only, and
# the block-scale family spellings (block 4). The production grid is not
# the source.
const _T5_CB = _T5_94_COLL_B[2:end]
const _T5_CB_MX = ((:mxf8f6f4, :block32), (:mxf4, :block32),
                   (:mxf4nvf4, :block32), (:mxf4nvf4, :block16))

_t5_cb_mods(kind, cg; sp = false, ashift = false, a = nothing, b) =
    (:mma, (sp ? (:sp,) : ())..., Symbol("cta_group::", cg),
     Symbol("kind::", kind), (ashift ? (:ashift,) : ())...,
     _t5_94_opt(a)..., b)

_t5_cb_mx_mods(kind, block, cg; sp = false, a = nothing, b) =
    (:mma, (sp ? (:sp,) : ())..., Symbol("cta_group::", cg),
     Symbol("kind::", kind), :block_scale, block, _t5_94_opt(a)..., b)

@testset "tcgen05 collector::b mma closed callable surface" begin
    expected = Set{Tuple{Vararg{Symbol}}}()
    methods_checked = 0
    for kind in (:f16, :tf32, :f8f6f4), cg in (1, 2), sp in (false, true),
            ashift in (false, true),
            a in (ashift ? _T5_94_COLL_A[1:2] : _T5_94_COLL_A), b in _T5_CB
        mods = _t5_cb_mods(kind, cg; sp, ashift, a, b)
        push!(expected, mods)
        maskN = cg == 1 ? 4 : 8
        meta = sp ? (UInt32,) : ()
        scale_ok = kind in (:f16, :tf32)
        for a_tmem in (ashift ? (true,) : (false, true))
            aT = a_tmem ? UInt32 : UInt64
            _t5_94_check(mods, (UInt32, aT, UInt64, meta..., UInt32, Bool),
                         _t5_94_schema(; a_tmem, meta = sp))
            _t5_94_check(mods,
                         (UInt32, aT, UInt64, meta..., UInt32,
                          NTuple{maskN, UInt32}, Bool),
                         _t5_94_schema(; a_tmem, meta = sp, maskN))
            methods_checked += 2
            scale_ok || continue
            _t5_94_check(mods,
                         (UInt32, aT, UInt64, meta..., UInt32, Bool, Val{5}),
                         _t5_94_schema(; a_tmem, meta = sp, scale = 5))
            _t5_94_check(mods,
                         (UInt32, aT, UInt64, meta..., UInt32,
                          NTuple{maskN, UInt32}, Bool, Val{15}),
                         _t5_94_schema(; a_tmem, meta = sp, maskN, scale = 15))
            methods_checked += 2
        end
    end
    @test length(expected) == 216
    @test methods_checked == 1200

    mx_expected = Set{Tuple{Vararg{Symbol}}}()
    for (kind, block) in _T5_CB_MX, cg in (1, 2), sp in (false, true),
            a in _T5_94_COLL_A, b in _T5_CB
        mods = _t5_cb_mx_mods(kind, block, cg; sp, a, b)
        push!(mx_expected, mods)
        meta = sp ? (UInt32,) : ()
        for a_tmem in (false, true)
            aT = a_tmem ? UInt32 : UInt64
            _t5_94_check(mods,
                         (UInt32, aT, UInt64, meta..., UInt32, UInt32, UInt32,
                          Bool),
                         _t5_94_schema(; a_tmem, meta = sp, mx = true))
            methods_checked += 1
        end
    end
    @test length(mx_expected) == 192
    @test methods_checked == 1584
    @test Set(PTX.wrapper_asm_forms(:tcgen05_mma_collb)) ==
          union(expected, mx_expected)
    # The pre-9.4 block-scale inventory is untouched.
    @test length(PTX.wrapper_asm_forms(:tcgen05_mx)) == 128
end

@testset "tcgen05 collector::b mma rejects grammar and ABI misses" begin
    fill_b = Symbol("collector::b::fill")
    f16 = _t5_cb_mods(:f16, 1; b = fill_b)
    misses = (
        # .kind::i8 has no B collector.
        (_t5_cb_mods(:i8, 1; b = fill_b), (UInt32, UInt64, UInt64, UInt32, Bool)),
        # discard is the unspelled default; the B collector follows A.
        (_t5_cb_mods(:f16, 1; b = Symbol("collector::b::discard")),
         (UInt32, UInt64, UInt64, UInt32, Bool)),
        ((:mma, Symbol("cta_group::1"), Symbol("kind::f16"), fill_b,
          Symbol("collector::a::fill")), (UInt32, UInt64, UInt64, UInt32, Bool)),
        # ashift: TMEM A only, no A fill/use.
        (_t5_cb_mods(:f16, 1; ashift = true, b = fill_b),
         (UInt32, UInt64, UInt64, UInt32, Bool)),
        (_t5_cb_mods(:f16, 1; ashift = true, a = Symbol("collector::a::use"),
                     b = fill_b), (UInt32, UInt32, UInt64, UInt32, Bool)),
        # scale-input-d: f16/tf32 only, an immediate in 0:15.
        (_t5_cb_mods(:f8f6f4, 1; b = fill_b),
         (UInt32, UInt64, UInt64, UInt32, Bool, Val{5})),
        (f16, (UInt32, UInt64, UInt64, UInt32, Bool, Int64)),
        # Mask width follows cta_group and precedes enable-input-d.
        (f16, (UInt32, UInt64, UInt64, UInt32, NTuple{8, UInt32}, Bool)),
        (f16, (UInt32, UInt64, UInt64, UInt32, Bool, NTuple{4, UInt32})),
        # sp metadata is mandatory.
        (_t5_cb_mods(:f16, 1; sp = true, b = fill_b),
         (UInt32, UInt64, UInt64, UInt32, Bool)),
        # Block-scale: the family spellings only, both scale addresses
        # mandatory, no mask.
        (_t5_cb_mx_mods(:mxf8f6f4, Symbol("scale_vec::1X"), 1; b = fill_b),
         (UInt32, UInt64, UInt64, UInt32, UInt32, UInt32, Bool)),
        (_t5_cb_mx_mods(:mxf4nvf4, Symbol("scale_vec::4X"), 1; b = fill_b),
         (UInt32, UInt64, UInt64, UInt32, UInt32, UInt32, Bool)),
        (_t5_cb_mx_mods(:mxf4, :block16, 1; b = fill_b),
         (UInt32, UInt64, UInt64, UInt32, UInt32, UInt32, Bool)),
        (_t5_cb_mx_mods(:mxf8f6f4, :block32, 1; b = fill_b),
         (UInt32, UInt64, UInt64, UInt32, Bool)),
        (_t5_cb_mx_mods(:mxf8f6f4, :block32, 1; b = fill_b),
         (UInt32, UInt64, UInt64, UInt32, NTuple{4, UInt32}, UInt32, UInt32,
          Bool)),
    )
    for (mods, args) in misses
        _t5_94_forbidden(mods, args)
    end

    # An out-of-range scale immediate is rejected before any code is
    # emitted.
    op = Operation{:tcgen05, f16}()
    @test_throws ArgumentError op(UInt32(0), UInt64(0), UInt64(0), UInt32(0),
                                  false, Val(16))
    @test_throws ArgumentError op(UInt32(0), UInt64(0), UInt64(0), UInt32(0),
                                  false, Val(-1))
end
