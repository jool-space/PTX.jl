using PTX: Operation

# Independent transcription of PTX 9.3 §9.7.15.5.14.  Do not derive this
# product from MMA_INT_VARIANTS or MMA_SYNC_FRAGS: a production edit must
# force an explicit review of shapes, signedness, saturation, and carriers.
const EXPECTED_DENSE_INTEGER_MMA = let rows = NamedTuple[]
    for (shape, types, n_a, n_b) in (
            (:m16n8k16, (:u8, :s8), 2, 1),
            (:m16n8k32, (:u8, :s8), 4, 2),
            (:m16n8k32, (:u4, :s4), 2, 1),
            (:m16n8k64, (:u4, :s4), 4, 2))
        for a in types, b in types, satfinite in (false, true)
            push!(rows, (; shape, a, b, satfinite, n_a, n_b))
        end
    end
    Tuple(rows)
end

function _int_mma_mods(row)
    sat = row.satfinite ? (:satfinite,) : ()
    (:sync, :aligned, row.shape, :row, :col, sat...,
     :s32, row.a, row.b, :s32)
end

@testset "dense integer mma: exact PTX 9.3 product" begin
    @test length(EXPECTED_DENSE_INTEGER_MMA) == 32
    @test length(unique(_int_mma_mods.(EXPECTED_DENSE_INTEGER_MMA))) == 32

    intrinsic_names = Set{String}()
    for row in EXPECTED_DENSE_INTEGER_MMA
        op = Operation{:mma, _int_mma_mods(row)}()
        argtypes = (NTuple{row.n_a, UInt32}, NTuple{row.n_b, UInt32},
                    NTuple{4, Int32})
        @test which(op, argtypes).module === PTX
        info = PTX.lowering(op, argtypes)
        @test info.tier === :intrinsic
        @test info.rettype === NTuple{4, Int32}
        @test length(info.intrinsics) == 1
        push!(intrinsic_names, only(info.intrinsics))
    end
    # Every PTX form has its own backend intrinsic.  This guards the less
    # obvious same-type name contraction (`.s8`, not `.s8.s8`).
    @test length(intrinsic_names) == 32
end

@testset "dense integer mma: invalid products fail loud" begin
    bad = (
        # Cross the 8-bit/4-bit shape products.
        ((:sync, :aligned, :m16n8k64, :row, :col,
          :s32, :u8, :u8, :s32),
         (NTuple{4, UInt32}, NTuple{2, UInt32}, NTuple{4, Int32})),
        ((:sync, :aligned, :m16n8k16, :row, :col,
          :s32, :u4, :u4, :s32),
         (NTuple{2, UInt32}, NTuple{1, UInt32}, NTuple{4, Int32})),
        # Only row.col, and satfinite belongs before the type quartet.
        ((:sync, :aligned, :m16n8k16, :col, :row,
          :s32, :u8, :u8, :s32),
         (NTuple{2, UInt32}, NTuple{1, UInt32}, NTuple{4, Int32})),
        ((:sync, :aligned, :m16n8k16, :row, :col,
          :s32, :u8, :u8, :s32, :satfinite),
         (NTuple{2, UInt32}, NTuple{1, UInt32}, NTuple{4, Int32})),
        # Packed A/B are UInt32, while the semantic accumulator is Int32.
        ((:sync, :aligned, :m16n8k16, :row, :col,
          :s32, :u8, :u8, :s32),
         (NTuple{2, Int32}, NTuple{1, UInt32}, NTuple{4, Int32})),
        ((:sync, :aligned, :m16n8k16, :row, :col,
          :s32, :u8, :u8, :s32),
         (NTuple{2, UInt32}, NTuple{1, UInt32}, NTuple{4, UInt32})),
    )

    for (mods, argtypes) in bad
        op = Operation{:mma, mods}()
        info = PTX.lowering(op, argtypes)
        @test info.tier === :forbidden
        @test endswith(String(which(op, argtypes).file), "entries.jl")
    end

    op = Operation{:mma, first(first(bad))}()
    args = (ntuple(_ -> UInt32(0), Val(4)),
            ntuple(_ -> UInt32(0), Val(2)),
            ntuple(_ -> Int32(0), Val(4)))
    err = try
        op(args...)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("requires an exact typed wrapper", sprint(showerror, err))
end
