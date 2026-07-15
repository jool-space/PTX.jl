using PTX
using PTX: Operation, lowering

# Independent PTX 9.3 §9.7.9.26.5.4 base-im2col inventory. Do not derive
# ranks, coordinate/offset counts, or intrinsic names from the wrapper methods.
const _TMA_IM2COL_PREFETCH_FORMS = [
    (Symbol("3d"), 3, 1,
     "llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.3d"),
    (Symbol("4d"), 4, 2,
     "llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.4d"),
    (Symbol("5d"), 5, 3,
     "llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.5d"),
]

_tma_im2col_prefetch_mods(rank; hint = false) =
    (:async, :bulk, :prefetch, :tensor, rank, :L2, :global, :im2col,
     (hint ? (Symbol("L2::cache_hint"),) : ())...)

@testset "TMA base-im2col prefetch closed callable surface" begin
    reviewed = Set{String}()

    for (rank, ncoords, noffsets, intrinsic) in _TMA_IM2COL_PREFETCH_FORMS
        push!(reviewed, intrinsic)
        coords = ntuple(_ -> Int32, ncoords)
        offsets = ntuple(_ -> Int16, noffsets)
        for hint in (false, true)
            mods = _tma_im2col_prefetch_mods(rank; hint)
            args = hint ?
                (PTX.TMADescriptorPtr, coords..., offsets..., UInt64) :
                (PTX.TMADescriptorPtr, coords..., offsets...)
            op = Operation{:cp, mods}()
            @test which(op, args).module === PTX

            info = lowering(op, args)
            @test info.tier === :intrinsic
            @test info.rettype === Nothing
            @test info.intrinsics == [intrinsic]

            record = PTX.NVVM.intrinsic(intrinsic)
            @test record.ret == ()
            @test :convergent in record.props
            @test !(:nomem in record.props)
            @test record.immargs == (2ncoords + 1,)
            @test (1, :readonly) in record.argattrs
        end
    end

    registry = Set(name for name in keys(PTX.NVVM.TABLE)
                   if startswith(name,
                       "llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.") &&
                      !occursin(".im2col.w.", name))
    @test reviewed == registry
    @test length(reviewed) == 3
end

@testset "TMA base-im2col prefetch rejects grammar and ABI misses" begin
    rank3 = Symbol("3d")
    base3 = _tma_im2col_prefetch_mods(rank3)
    hint3 = _tma_im2col_prefetch_mods(rank3; hint = true)
    pglobal = Core.LLVMPtr{UInt8, PTX.AS.Global}
    pconst16 = Core.LLVMPtr{UInt16, PTX.AS.Const}

    misses = (
        # Rank 3 requires exactly three s32 coordinates and one s16 offset.
        (base3, (PTX.TMADescriptorPtr, Int32, Int32, Int16)),
        (base3, (PTX.TMADescriptorPtr, Int32, Int32, Int32,
                 Int32, Int16)),
        (base3, (PTX.TMADescriptorPtr, UInt32, Int32, Int32, Int16)),
        (base3, (PTX.TMADescriptorPtr, Int64, Int32, Int32, Int16)),
        (base3, (PTX.TMADescriptorPtr, Int32, Int32, Int32, UInt16)),
        (base3, (PTX.TMADescriptorPtr, Int32, Int32, Int32, Int32)),
        # Cache qualifier and exact u64 policy carrier are inseparable.
        (base3, (PTX.TMADescriptorPtr, Int32, Int32, Int32,
                 Int16, UInt64)),
        (hint3, (PTX.TMADescriptorPtr, Int32, Int32, Int32, Int16)),
        (hint3, (PTX.TMADescriptorPtr, Int32, Int32, Int32,
                 Int16, UInt32)),
        # A tensor-map descriptor is the package's AS.Const UInt8 carrier.
        (base3, (pglobal, Int32, Int32, Int32, Int16)),
        (base3, (pconst16, Int32, Int32, Int32, Int16)),
        # Rank and canonical modifier order are closed.
        ((:async, :bulk, :prefetch, :tensor, Symbol("2d"), :L2,
          :global, :im2col),
         (PTX.TMADescriptorPtr, Int32, Int32)),
        ((:async, :bulk, :prefetch, :tensor, Symbol("6d"), :L2,
          :global, :im2col),
         (PTX.TMADescriptorPtr, Int32, Int32, Int32, Int32, Int32, Int32,
          Int16, Int16, Int16, Int16)),
        ((:async, :bulk, :prefetch, :tensor, rank3, :global, :L2, :im2col),
         (PTX.TMADescriptorPtr, Int32, Int32, Int32, Int16)),
        ((:async, :bulk, :prefetch, :tensor, rank3, :L2, :global,
          Symbol("L2::cache_hint"), :im2col),
         (PTX.TMADescriptorPtr, Int32, Int32, Int32, Int16, UInt64)),
        # Later modes have distinct ABIs/targets and remain deliberately absent.
        ((:async, :bulk, :prefetch, :tensor, rank3, :L2, :global,
          Symbol("im2col::w")),
         (PTX.TMADescriptorPtr, Int32, Int32, Int32, Int16, Int16)),
        ((:async, :bulk, :prefetch, :tensor, rank3, :L2, :global,
          Symbol("im2col::w::128")),
         (PTX.TMADescriptorPtr, Int32, Int32, Int32, Int16, Int16)),
        ((:async, :bulk, :prefetch, :tensor, Symbol("2d"), :L2,
          :global, Symbol("tile::gather4")),
         (PTX.TMADescriptorPtr, Int32, Int32, Int32, Int32, Int32)),
    )

    for (mods, args) in misses
        info = lowering(Operation{:cp, mods}(), args)
        @test info.tier === :forbidden
        @test_throws ArgumentError PTX.build_call(:cp, mods, args)
    end
end
