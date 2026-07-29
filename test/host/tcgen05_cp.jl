using PTX
using PTX: Operation, lowering

# Independent PTX 9.3 §9.7.17.9 inventory: shape (with its mandatory
# multicast pairing where the ISA requires one) × optional decompression.
# The production methods are intentionally not the source of this oracle.
const _T5_CP_SHAPES = [
    ((Symbol("128x256b"),),                          "128x256b"),
    ((Symbol("4x256b"),),                            "4x256b"),
    ((Symbol("128x128b"),),                          "128x128b"),
    ((Symbol("64x128b"), Symbol("warpx2::02_13")),   "64x128b_warpx2_02_13"),
    ((Symbol("64x128b"), Symbol("warpx2::01_23")),   "64x128b_warpx2_01_23"),
    ((Symbol("32x128b"), :warpx4),                   "32x128b_warpx4"),
]
const _T5_CP_FMTS = [((), ""), ((:b8x16, :b6x16_p32), "b6x16_p32"),
                     ((:b8x16, :b4x16_p64), "b4x16_p64")]

@testset "tcgen05.cp closed callable surface" begin
    reviewed = Set{String}()

    for (shapemods, stem) in _T5_CP_SHAPES, (fmtmods, fmt) in _T5_CP_FMTS,
            cg in 1:2
        intrinsic = fmt == "" ? "llvm.nvvm.tcgen05.cp.$stem.cg$cg" :
                                "llvm.nvvm.tcgen05.cp.$stem.$fmt.cg$cg"
        push!(reviewed, intrinsic)
        mods = (:cp, Symbol("cta_group::", cg), shapemods..., fmtmods...)
        # Single-route asm since the demotion (see wrappers/tcgen05.jl):
        # every form is convergent inline asm with the full clobber.
        info = lowering(Operation{:tcgen05, mods}(), (UInt32, UInt64))
        @test info.tier === :asm
        @test info.rettype === Nothing
        @test isempty(info.intrinsics)

        # The unwrapped intrinsic records stay in the pinned registry; the
        # attribute review below is what the demotion walked away from
        # (convergent + argmem — no stronger promise than the asm route).
        record = PTX.NVVM.intrinsic(intrinsic)
        @test record.ret === ()
        @test :convergent in record.props
        @test :inaccessiblemem_or_argmemonly in record.props
        @test (1, :nocapture) in record.argattrs
    end

    # Closed world: the reviewed grid is exactly the registry's cp
    # inventory — a registry addition must force a review here.
    registry = Set(name for name in keys(PTX.NVVM.TABLE)
                   if startswith(name, "llvm.nvvm.tcgen05.cp."))
    @test reviewed == registry
    @test length(reviewed) == 36
end

@testset "tcgen05.cp rejects grammar and ABI misses" begin
    cg1 = Symbol("cta_group::1")
    misses = (
        # The ISA couples multicast to shape: 64x128b requires a warpx2
        # pairing, 32x128b requires warpx4, and the other shapes take none.
        ((:cp, cg1, Symbol("64x128b")), (UInt32, UInt64)),
        ((:cp, cg1, Symbol("32x128b")), (UInt32, UInt64)),
        ((:cp, cg1, Symbol("64x128b"), :warpx4), (UInt32, UInt64)),
        ((:cp, cg1, Symbol("32x128b"), Symbol("warpx2::02_13")),
         (UInt32, UInt64)),
        ((:cp, cg1, Symbol("128x256b"), :warpx4), (UInt32, UInt64)),
        # dst_fmt and src_fmt are an ordered pair.
        ((:cp, cg1, Symbol("128x256b"), :b6x16_p32, :b8x16),
         (UInt32, UInt64)),
        ((:cp, cg1, Symbol("128x256b"), :b8x16), (UInt32, UInt64)),
        ((:cp, cg1, Symbol("128x256b"), :b6x16_p32), (UInt32, UInt64)),
        # Multicast must precede the format pair (ISA modifier order).
        ((:cp, cg1, Symbol("64x128b"), :b8x16, :b6x16_p32,
          Symbol("warpx2::02_13")), (UInt32, UInt64)),
        # Operand carriers are pinned: TMEM address is UInt32, the SMEM
        # descriptor is UInt64.
        ((:cp, cg1, Symbol("128x256b")), (UInt64, UInt64)),
        ((:cp, cg1, Symbol("128x256b")), (UInt32, UInt32)),
        ((:cp, cg1, Symbol("128x256b")), (UInt32,)),
    )

    for (mods, args) in misses
        @test lowering(Operation{:tcgen05, mods}(), args).tier === :forbidden
        @test_throws ArgumentError PTX.build_call(:tcgen05, mods, args)
    end
end
