using PTX
using PTX: Operation, lowering

# Independent PTX 9.3 §9.7.17.10 dense-mma oracle: kinds with their NVVM
# immarg values and scale-input-d legality, the empirical collector enum
# (0=discard, 1=lastuse, 2=fill, 3=use), and the ashift restrictions
# (TMEM A only; collector limited to discard/lastuse). The production
# spec tables are intentionally not the source of this oracle.
const _T5_DENSE_KINDS =
    ((:f16, 0, true), (:tf32, 1, true), (:f8f6f4, 2, false), (:i8, 3, false))
const _T5_DENSE_COLLECTORS =
    ((nothing, 0), (Symbol("collector::a::lastuse"), 1),
     (Symbol("collector::a::fill"), 2), (Symbol("collector::a::use"), 3))

_t5_dense_mods(cg, kind; ashift = false, coll = nothing) =
    (:mma, Symbol("cta_group::", cg), Symbol("kind::", kind),
     (ashift ? (:ashift,) : ())...,
     (coll === nothing ? () : (coll,))...)

@testset "tcgen05 dense mma closed callable surface" begin
    reviewed = Set{String}()

    for (kind, kindval, scale_ok) in _T5_DENSE_KINDS, cg in (1, 2),
            (coll, collval) in _T5_DENSE_COLLECTORS,
            (tmem_a, ashift) in ((false, false), (true, false), (true, true))
        ashift && collval >= 2 && continue
        mods = _t5_dense_mods(cg, kind; ashift, coll)
        aT = tmem_a ? UInt32 : UInt64
        maskT = NTuple{cg == 1 ? 4 : 8, UInt32}
        stem = "llvm.nvvm.tcgen05.mma." * (tmem_a ? "tensor" : "shared")
        sh = ashift ? ".ashift" : ""

        shapes = [
            ((UInt32, aT, UInt64, UInt32, Bool), stem * sh),
            ((UInt32, aT, UInt64, UInt32, maskT, Bool),
             stem * ".disable_output_lane.cg$cg" * sh),
        ]
        scale_ok && append!(shapes, [
            ((UInt32, aT, UInt64, UInt32, Bool, Val{5}),
             stem * ".scale_d" * sh),
            ((UInt32, aT, UInt64, UInt32, maskT, Bool, Val{5}),
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

    # Closed world: the reviewed grid is exactly the registry's dense
    # inventory and the generated family's name table.
    registry = Set(name for name in keys(PTX.NVVM.TABLE)
                   if startswith(name, "llvm.nvvm.tcgen05.mma.") &&
                      !occursin(".sp.", name) && !occursin(".ws.", name) &&
                      !occursin("block_scale", name))
    @test reviewed == registry
    @test reviewed == Set(PTX.wrapper_intrinsic_names(:tcgen05_mma_dense))
    @test length(reviewed) == 18

    # The ashift-legal collector subset is pinned by the registry ranges:
    # every .ashift record restricts the collector immarg to [0, 2).
    for name in reviewed
        record = PTX.NVVM.intrinsic(name)
        hi = last(record.ranges[end])
        @test hi == (endswith(name, ".ashift") ? 2 : 4)
    end
end

@testset "tcgen05 dense mma rejects grammar and ABI misses" begin
    f16_1 = _t5_dense_mods(1, :f16)
    misses = (
        # ashift is TMEM-A only (a-desc form has no row shift).
        (_t5_dense_mods(1, :f16; ashift = true),
         (UInt32, UInt64, UInt64, UInt32, Bool)),
        # ashift forbids collector fill/use (ISA; registry range [0, 2)).
        (_t5_dense_mods(1, :f16; ashift = true,
                        coll = Symbol("collector::a::fill")),
         (UInt32, UInt32, UInt64, UInt32, Bool)),
        (_t5_dense_mods(1, :f16; ashift = true,
                        coll = Symbol("collector::a::use")),
         (UInt32, UInt32, UInt64, UInt32, Bool)),
        # Modifier order is kind{.ashift}{.collector}; collector-first is
        # not a reviewed spelling.
        ((:mma, Symbol("cta_group::1"), Symbol("kind::f16"),
          Symbol("collector::a::lastuse"), :ashift),
         (UInt32, UInt32, UInt64, UInt32, Bool)),
        # scale-input-d exists only for f16/tf32.
        (_t5_dense_mods(1, :i8),
         (UInt32, UInt64, UInt64, UInt32, Bool, Val{5})),
        (_t5_dense_mods(1, :f8f6f4),
         (UInt32, UInt64, UInt64, UInt32, Bool, Val{5})),
        # ... and must be a compile-time immediate.
        (f16_1, (UInt32, UInt64, UInt64, UInt32, Bool, Int64)),
        # The mask width is pinned by cta_group (4 vs 8 words) and sits
        # before enable-input-d.
        (f16_1, (UInt32, UInt64, UInt64, UInt32, NTuple{8, UInt32}, Bool)),
        (_t5_dense_mods(2, :f16),
         (UInt32, UInt64, UInt64, UInt32, NTuple{4, UInt32}, Bool)),
        (f16_1, (UInt32, UInt64, UInt64, UInt32, Bool, NTuple{4, UInt32})),
        # Operand carriers are pinned (no widening of the B descriptor).
        (f16_1, (UInt32, UInt64, UInt32, UInt32, Bool)),
    )

    for (mods, args) in misses
        @test lowering(Operation{:tcgen05, mods}(), args).tier === :forbidden
        @test_throws ArgumentError PTX.build_call(:tcgen05, mods, args)
    end
end
