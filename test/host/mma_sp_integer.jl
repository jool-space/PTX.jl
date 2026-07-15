using PTX: Operation

# Independent transcription of PTX 9.3 §9.7.15.6.2.5–.8 and §9.7.15.6.3.
# Do not derive this matrix from MMA_SP_INT_VARIANTS or MMA_SP_FRAGS: changing
# production coverage must force an explicit review of this oracle.
const EXPECTED_INTEGER_MMA_SP = let rows = NamedTuple[]
    for (shape, types, n_a, n_b, selectors) in (
            (:m16n8k32,  (:u8, :s8), 2, 2, 0:1),
            (:m16n8k64,  (:u8, :s8), 4, 4, 0:0),
            (:m16n8k64,  (:u4, :s4), 2, 2, 0:1),
            (:m16n8k128, (:u4, :s4), 4, 4, 0:0))
        for a in types, b in types, satfinite in (false, true),
                ordered in (false, true)
            push!(rows, (; shape, a, b, satfinite, ordered,
                         n_a, n_b, selectors))
        end
    end
    Tuple(rows)
end

function _expected_integer_sp_mods(row)
    variant = row.ordered ? Symbol("sp::ordered_metadata") : :sp
    sat = row.satfinite ? (:satfinite,) : ()
    (variant, :sync, :aligned, row.shape, :row, :col, sat...,
     :s32, row.a, row.b, :s32)
end

function _expected_integer_sp_intrinsic(row)
    prefix = row.ordered ? "mma.sp.ordered.metadata" : "mma.sp"
    name = "$prefix.$(row.shape).row.col" *
           (row.satfinite ? ".satfinite" : "") * ".$(row.a)"
    row.a === row.b ? "llvm.nvvm.$name" : "llvm.nvvm.$name.$(row.b)"
end

@testset "integer mma.sp: exact 64-form ABI inventory" begin
    @test length(EXPECTED_INTEGER_MMA_SP) == 64
    @test length(unique(_expected_integer_sp_mods.(EXPECTED_INTEGER_MMA_SP))) == 64
    expected_names = Set{String}()
    method_count = 0

    for row in EXPECTED_INTEGER_MMA_SP
        op = Operation{:mma, _expected_integer_sp_mods(row)}()
        name = _expected_integer_sp_intrinsic(row)
        push!(expected_names, name)

        for selector in row.selectors
            argtypes = (NTuple{row.n_a, UInt32}, NTuple{row.n_b, UInt32},
                        NTuple{4, Int32}, UInt32, Val{selector})
            method = which(op, argtypes)
            info = PTX.lowering(op, argtypes)
            @test method.module === PTX
            @test info.tier === :intrinsic
            @test info.rettype === NTuple{4, Int32}
            @test info.intrinsics == [name]
            method_count += 1
        end

        bad_types = (NTuple{row.n_a, UInt32}, NTuple{row.n_b, UInt32},
                     NTuple{4, Int32}, UInt32, Val{last(row.selectors) + 1})
        @test PTX.lowering(op, bad_types).tier === :forbidden
        @test endswith(String(which(op, bad_types).file), "inst.jl")

        intrinsic = PTX.NVVM.intrinsic(name)
        @test :nomem in intrinsic.props
        @test PTX.NVVM.is_convergent(intrinsic)
        @test PTX.NVVM.callsiteattrs(intrinsic) == "convergent nomerge"
    end

    @test method_count == 96
    @test expected_names == Set(PTX.MMA_SP_INTEGER_INTRINSIC_NAMES)
    @test count(startswith("llvm.nvvm.mma.sp.ordered.metadata"),
                expected_names) == 32
    @test count(startswith("llvm.nvvm.mma.sp.m16"), expected_names) == 32
    @test all(!occursin("tcgen05", name) for name in expected_names)
end

@testset "integer mma.sp: canonical grammar and fail-loud misses" begin
    @test PTX._mma_sp_selectors(:m16n8k64, :u4) == (0, 1)
    pair_op = ptx"mma.sp.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32"
    pair_args = (NTuple{2, UInt32}, NTuple{2, UInt32}, NTuple{4, Int32},
                 UInt32, Val{1})
    @test PTX.lowering(pair_op, pair_args).tier === :intrinsic
    pair_bad_args = (NTuple{2, UInt32}, NTuple{2, UInt32}, NTuple{4, Int32},
                     UInt32, Val{2})
    @test PTX.lowering(pair_op, pair_bad_args).tier === :forbidden

    ordered_sat = ptx"mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.u4.s32"
    @test typeof(ordered_sat).parameters[2] ==
          (Symbol("sp::ordered_metadata"), :sync, :aligned, :m16n8k64,
           :row, :col, :satfinite, :s32, :s4, :u4, :s32)

    bad = (
        # Fragment widths and carriers are semantic, not inferred from `.s32`.
        (ordered_sat, (NTuple{4, UInt32}, NTuple{2, UInt32},
                       NTuple{4, Int32}, UInt32, Val{0})),
        (ordered_sat, (NTuple{2, UInt32}, NTuple{2, UInt32},
                       NTuple{4, UInt32}, UInt32, Val{0})),
        # `satfinite` appears before the complete type quartet.
        (Operation{:mma,
            (Symbol("sp::ordered_metadata"), :sync, :aligned, :m16n8k64,
             :row, :col, :s32, :s4, :u4, :s32, :satfinite)}(),
         (NTuple{2, UInt32}, NTuple{2, UInt32}, NTuple{4, Int32},
          UInt32, Val{0})),
        # Nearby dense and tcgen05 products are not part of this surface.
        (Operation{:mma,
            (Symbol("sp::ordered_metadata"), :sync, :aligned, :m16n8k32,
             :row, :col, :s32, :u4, :u4, :s32)}(),
         (NTuple{2, UInt32}, NTuple{2, UInt32}, NTuple{4, Int32},
          UInt32, Val{0})),
    )
    for (op, argtypes) in bad
        @test PTX.lowering(op, argtypes).tier === :forbidden
        @test endswith(String(which(op, argtypes).file), "inst.jl")
    end

    args = (ntuple(_ -> UInt32(0), Val(4)),
            ntuple(_ -> UInt32(0), Val(2)),
            ntuple(_ -> Int32(0), Val(4)), UInt32(0), Val(0))
    err = try
        ordered_sat(args...)
        nothing
    catch exception
        exception
    end
    @test err isa ArgumentError
    @test occursin("requires an exact typed wrapper", sprint(showerror, err))
end
