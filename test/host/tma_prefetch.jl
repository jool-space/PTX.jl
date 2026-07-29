using PTX
using PTX: Operation, lowering

# Independent PTX 9.3 §9.7.9.26.5.4 tile-only inventory. The production
# methods are intentionally not the source of this oracle.
const _TMA_PREFETCH_FORMS = [
    (Symbol("1d"), 1, "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.1d"),
    (Symbol("2d"), 2, "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.2d"),
    (Symbol("3d"), 3, "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.3d"),
    (Symbol("4d"), 4, "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.4d"),
    (Symbol("5d"), 5, "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.5d"),
]

_tma_prefetch_mods(rank; hint = false) =
    (:async, :bulk, :prefetch, :tensor, rank, :L2, :global, :tile,
     (hint ? (Symbol("L2::cache_hint"),) : ())...)

@testset "TMA tile-prefetch closed callable surface" begin
    tmap = PTX.TMADescriptorPtr
    reviewed = Set{String}()

    for (rank, ncoords, intrinsic) in _TMA_PREFETCH_FORMS
        push!(reviewed, intrinsic)
        coords = ntuple(_ -> Int32, ncoords)
        for hint in (false, true)
            mods = _tma_prefetch_mods(rank; hint)
            args = hint ? (tmap, coords..., UInt64) : (tmap, coords...)
            # Single-route asm since the demotion (see wrappers/tma.jl):
            # every form is convergent inline asm with the full clobber.
            info = lowering(Operation{:cp, mods}(), args)
            @test info.tier === :asm
            @test info.rettype === Nothing
            @test isempty(info.intrinsics)

            # The unwrapped intrinsic records stay in the pinned registry;
            # the attribute review below is what the demotion walked away
            # from (convergent + a readonly descriptor operand — no
            # stronger promise than the asm route).
            record = PTX.NVVM.intrinsic(intrinsic)
            @test record.ret == ()
            @test :convergent in record.props
            @test !(:nomem in record.props)
            @test record.immargs == (ncoords + 3,)
            @test (1, :readonly) in record.argattrs
        end
    end

    registry = Set(name for name in keys(PTX.NVVM.TABLE)
                   if startswith(name,
                       "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.") &&
                      !occursin("gather", name))
    @test reviewed == registry
    @test length(reviewed) == 5
end

@testset "TMA tile-prefetch rejects grammar and ABI misses" begin
    tmap = PTX.TMADescriptorPtr
    pglobal = Core.LLVMPtr{UInt8, PTX.AS.Global}
    base2 = _tma_prefetch_mods(Symbol("2d"))
    hint2 = _tma_prefetch_mods(Symbol("2d"); hint = true)

    misses = (
        # Coordinate count is exactly the declared rank.
        (base2, (tmap, Int32)),
        (base2, (tmap, Int32, Int32, Int32)),
        # Cache qualifier and u64 policy operand are an inseparable pair.
        (base2, (tmap, Int32, Int32, UInt64)),
        (hint2, (tmap, Int32, Int32)),
        (hint2, (tmap, Int32, Int32, UInt32)),
        (hint2, (tmap, Int32, Int32, Int64)),
        # The package convention requires a descriptor carrier, not an AS1
        # data pointer. The wrapper raw-retypes AS.Const to generic AS0.
        (base2, (pglobal, Int32, Int32)),
        # Rank and canonical modifier order are closed.
        ((:async, :bulk, :prefetch, :tensor, Symbol("0d"), :L2,
          :global, :tile), (tmap,)),
        ((:async, :bulk, :prefetch, :tensor, Symbol("6d"), :L2,
          :global, :tile), (tmap, Int32, Int32, Int32, Int32, Int32, Int32)),
        ((:async, :bulk, :prefetch, :tensor, Symbol("2d"), :global,
          :L2, :tile), (tmap, Int32, Int32)),
        ((:async, :bulk, :prefetch, :tensor, Symbol("2d"), :L2,
          :global, Symbol("L2::cache_hint"), :tile),
         (tmap, Int32, Int32, UInt64)),
        # Other load-mode islands stay deliberately unimplemented here.
        ((:async, :bulk, :prefetch, :tensor, Symbol("2d"), :L2,
          :global, Symbol("tile::gather4")),
         (tmap, Int32, Int32, Int32, Int32, Int32)),
        ((:async, :bulk, :prefetch, :tensor, Symbol("3d"), :L2,
          :global, Symbol("im2col::w")),
         (tmap, Int32, Int32, Int32, UInt16)),
    )

    for (mods, args) in misses
        @test lowering(Operation{:cp, mods}(), args).tier === :forbidden
        @test_throws ArgumentError PTX.build_call(:cp, mods, args)
    end
end
