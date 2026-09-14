include(joinpath(@__DIR__, "tcgen05_ptx94_mma_defs.jl"))

# Independent PTX ISA 9.4 §9.7.18.10 `.decompress::lut::b` oracle (syntax
# block 7): dense only, `.kind::f8f6f4` and `.kind::mxf8f6f4.block_scale`
# with the family `.block32` spelling, both collectors optional, and the
# LUT's TMEM address after the compressed B descriptor. The production
# grid is not the source.
const _T5_LUT = Symbol("decompress::lut::b")

_t5_lut_mods(cg; a = nothing, b = nothing) =
    (:mma, Symbol("cta_group::", cg), Symbol("kind::f8f6f4"), _T5_LUT,
     _t5_94_opt(a)..., _t5_94_opt(b)...)
_t5_lut_mx_mods(cg; a = nothing, b = nothing) =
    (:mma, Symbol("cta_group::", cg), Symbol("kind::mxf8f6f4"), :block_scale,
     _T5_LUT, :block32, _t5_94_opt(a)..., _t5_94_opt(b)...)

@testset "tcgen05 lut::b mma closed callable surface" begin
    expected = Set{Tuple{Vararg{Symbol}}}()
    methods_checked = 0
    for cg in (1, 2), a in _T5_94_COLL_A, b in _T5_94_COLL_B
        maskN = cg == 1 ? 4 : 8
        mods = _t5_lut_mods(cg; a, b)
        push!(expected, mods)
        for a_tmem in (false, true)
            aT = a_tmem ? UInt32 : UInt64
            _t5_94_check(mods, (UInt32, aT, UInt64, UInt32, UInt32, Bool),
                         _t5_94_schema(; a_tmem, meta = true))
            _t5_94_check(mods,
                         (UInt32, aT, UInt64, UInt32, UInt32,
                          NTuple{maskN, UInt32}, Bool),
                         _t5_94_schema(; a_tmem, meta = true, maskN))
            methods_checked += 2
        end
        mods = _t5_lut_mx_mods(cg; a, b)
        push!(expected, mods)
        for a_tmem in (false, true)
            aT = a_tmem ? UInt32 : UInt64
            _t5_94_check(mods,
                         (UInt32, aT, UInt64, UInt32, UInt32, UInt32, UInt32,
                          Bool),
                         _t5_94_schema(; a_tmem, meta = true, mx = true))
            methods_checked += 1
        end
    end
    @test length(expected) == 64
    @test methods_checked == 192
    @test Set(PTX.wrapper_asm_forms(:tcgen05_mma_lut)) == expected
end

@testset "tcgen05 lut::b mma rejects grammar and ABI misses" begin
    plain = _t5_lut_mods(1)
    misses = (
        # Dense only: no sp, no ashift, no scale-input-d.
        ((:mma, :sp, Symbol("cta_group::1"), Symbol("kind::f8f6f4"), _T5_LUT),
         (UInt32, UInt64, UInt64, UInt32, UInt32, Bool)),
        ((:mma, Symbol("cta_group::1"), Symbol("kind::f8f6f4"), _T5_LUT,
          :ashift), (UInt32, UInt32, UInt64, UInt32, UInt32, Bool)),
        (plain, (UInt32, UInt64, UInt64, UInt32, UInt32, Bool, Val{5})),
        # The LUT metadata address is mandatory.
        (plain, (UInt32, UInt64, UInt64, UInt32, Bool)),
        # Only f8f6f4 and mxf8f6f4 decompress B; the block-scale form uses
        # the family spelling only.
        ((:mma, Symbol("cta_group::1"), Symbol("kind::f16"), _T5_LUT),
         (UInt32, UInt64, UInt64, UInt32, UInt32, Bool)),
        ((:mma, Symbol("cta_group::1"), Symbol("kind::mxf4"), :block_scale,
          _T5_LUT, :block32),
         (UInt32, UInt64, UInt64, UInt32, UInt32, UInt32, UInt32, Bool)),
        ((:mma, Symbol("cta_group::1"), Symbol("kind::mxf8f6f4"), :block_scale,
          _T5_LUT, Symbol("scale_vec::1X")),
         (UInt32, UInt64, UInt64, UInt32, UInt32, UInt32, UInt32, Bool)),
        # Block-scale lut takes no mask and both scale addresses.
        (_t5_lut_mx_mods(1),
         (UInt32, UInt64, UInt64, UInt32, UInt32, NTuple{4, UInt32}, UInt32,
          UInt32, Bool)),
        (_t5_lut_mx_mods(1), (UInt32, UInt64, UInt64, UInt32, UInt32, Bool)),
        # Modifier order: lut before the collectors.
        ((:mma, Symbol("cta_group::1"), Symbol("kind::f8f6f4"),
          Symbol("collector::b::fill"), _T5_LUT),
         (UInt32, UInt64, UInt64, UInt32, UInt32, Bool)),
    )
    for (mods, args) in misses
        _t5_94_forbidden(mods, args)
    end
end
