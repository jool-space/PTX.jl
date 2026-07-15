using PTX
using PTX: Operation, lowering

# Independent PTX 9.3 §9.7.17.10.9.2 sparse-mma oracle: the dense grid
# plus the sparsity-metadata TMEM operand between the B descriptor and
# idesc. The production spec tables are intentionally not the source.
const _T5_SP_KINDS =
    ((:f16, true), (:tf32, true), (:f8f6f4, false), (:i8, false))
const _T5_SP_COLLECTORS =
    (nothing, Symbol("collector::a::lastuse"),
     Symbol("collector::a::fill"), Symbol("collector::a::use"))

_t5_sp_mods(cg, kind; ashift = false, coll = nothing) =
    (:mma, :sp, Symbol("cta_group::", cg), Symbol("kind::", kind),
     (ashift ? (:ashift,) : ())...,
     (coll === nothing ? () : (coll,))...)

@testset "tcgen05 sparse mma closed callable surface" begin
    reviewed = Set{String}()

    for (kind, scale_ok) in _T5_SP_KINDS, cg in (1, 2),
            (ci, coll) in enumerate(_T5_SP_COLLECTORS),
            (tmem_a, ashift) in ((false, false), (true, false), (true, true))
        ashift && ci > 2 && continue
        mods = _t5_sp_mods(cg, kind; ashift, coll)
        aT = tmem_a ? UInt32 : UInt64
        maskT = NTuple{cg == 1 ? 4 : 8, UInt32}
        stem = "llvm.nvvm.tcgen05.mma.sp." * (tmem_a ? "tensor" : "shared")
        sh = ashift ? ".ashift" : ""

        shapes = [
            ((UInt32, aT, UInt64, UInt32, UInt32, Bool), stem * sh),
            ((UInt32, aT, UInt64, UInt32, UInt32, maskT, Bool),
             stem * ".disable_output_lane.cg$cg" * sh),
        ]
        scale_ok && append!(shapes, [
            ((UInt32, aT, UInt64, UInt32, UInt32, Bool, Val{5}),
             stem * ".scale_d" * sh),
            ((UInt32, aT, UInt64, UInt32, UInt32, maskT, Bool, Val{5}),
             stem * ".scale_d.disable_output_lane.cg$cg" * sh),
        ])

        for (args, intrinsic) in shapes
            push!(reviewed, intrinsic)
            info = lowering(Operation{:tcgen05, mods}(), args)
            @test info.tier === :intrinsic
            @test info.rettype === Nothing
            @test intrinsic in info.intrinsics
        end
    end

    # Closed world against the registry's non-block-scale sp inventory;
    # the sp block-scale (MX) records stay outside the wrapper surface
    # like their dense counterparts.
    registry = Set(name for name in keys(PTX.NVVM.TABLE)
                   if startswith(name, "llvm.nvvm.tcgen05.mma.sp.") &&
                      !occursin("block_scale", name))
    @test reviewed == registry
    @test reviewed == Set(PTX.TCGEN05_MMA_SP_DENSE_INTRINSIC_NAMES)
    @test length(reviewed) == 18

    # ashift records restrict the collector immarg to [0, 2).
    for name in reviewed
        record = PTX.NVVM.intrinsic(name)
        hi = last(record.ranges[end])
        @test hi == (endswith(name, ".ashift") ? 2 : 4)
    end
end

@testset "tcgen05 sparse mma rejects grammar and ABI misses" begin
    f16_1 = _t5_sp_mods(1, :f16)
    misses = (
        # The sparsity-metadata operand is mandatory and is a TMEM
        # address (UInt32), not a descriptor.
        (f16_1, (UInt32, UInt64, UInt64, UInt32, Bool)),
        (f16_1, (UInt32, UInt64, UInt64, UInt64, UInt32, Bool)),
        # ... and sits between the B descriptor and idesc, not trailing.
        (f16_1, (UInt32, UInt64, UInt64, UInt32, Bool, UInt32)),
        # ashift is TMEM-A only and forbids collector fill/use.
        (_t5_sp_mods(1, :f16; ashift = true),
         (UInt32, UInt64, UInt64, UInt32, UInt32, Bool)),
        (_t5_sp_mods(1, :f16; ashift = true,
                     coll = Symbol("collector::a::fill")),
         (UInt32, UInt32, UInt64, UInt32, UInt32, Bool)),
        # scale-input-d exists only for f16/tf32.
        (_t5_sp_mods(2, :i8),
         (UInt32, UInt64, UInt64, UInt32, UInt32, Bool, Val{5})),
        # Mask width is pinned by cta_group.
        (_t5_sp_mods(1, :f16),
         (UInt32, UInt64, UInt64, UInt32, UInt32, NTuple{8, UInt32}, Bool)),
        # sp block-scale (MX) spellings are deliberately unimplemented.
        ((:mma, :sp, Symbol("cta_group::1"), Symbol("kind::mxf8f6f4"),
          :block_scale, Symbol("scale_vec::1X")),
         (UInt32, UInt64, UInt64, UInt32, UInt32, UInt32, UInt32, Bool)),
    )

    for (mods, args) in misses
        @test lowering(Operation{:tcgen05, mods}(), args).tier === :forbidden
        @test_throws ArgumentError PTX.build_call(:tcgen05, mods, args)
    end
end
