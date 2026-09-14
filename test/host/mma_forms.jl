using PTX: Operation
using PTX.NVVM: NVVM

# Independent transcriptions of the classic warp-level mma.sync form
# inventories: single-bit (§9.7.15.5.5, .12–.14), dense integer
# (§9.7.15.5.14), sparse integer (§9.7.15.6.2.5–.8, §9.7.15.6.3), and the
# floating sp::ordered_metadata island. None of these is derived from the
# production variant tables or fragment maps: production drift must force a
# review of the complete grammar, fragments, carriers, and target floors.

# --- single-bit ------------------------------------------------------------------

# Independent transcription of PTX 9.3 §9.7.15.5.5, .12–.14 and Figures
# 62–64, 97–103. Do not derive this from MMA_B1_VARIANTS or MMA_SYNC_FRAGS:
# production drift must force review of the complete grammar, fragments, and
# the independent PTX/target floors.
const EXPECTED_B1_MMA = (
    (; shape = :m8n8k128,  bitop = :xor, n_a = 1, n_b = 1, n_cd = 2,
       ptx = v"7.0", sm = v"7.5"),
    (; shape = :m8n8k128,  bitop = :and, n_a = 1, n_b = 1, n_cd = 2,
       ptx = v"7.1", sm = v"8.0"),
    (; shape = :m16n8k128, bitop = :xor, n_a = 2, n_b = 1, n_cd = 4,
       ptx = v"7.0", sm = v"8.0"),
    (; shape = :m16n8k128, bitop = :and, n_a = 2, n_b = 1, n_cd = 4,
       ptx = v"7.1", sm = v"8.0"),
    (; shape = :m16n8k256, bitop = :xor, n_a = 4, n_b = 2, n_cd = 4,
       ptx = v"7.0", sm = v"8.0"),
    (; shape = :m16n8k256, bitop = :and, n_a = 4, n_b = 2, n_cd = 4,
       ptx = v"7.1", sm = v"8.0"),
)

_b1_mma_mods(row) =
    (:sync, :aligned, row.shape, :row, :col,
     :s32, :b1, :b1, :s32, row.bitop, :popc)

_b1_mma_intrinsic(row) =
    "llvm.nvvm.mma.$(row.bitop).popc.$(row.shape).row.col.b1"

@testset "single-bit mma: exact PTX 9.3 product and ABI" begin
    @test length(EXPECTED_B1_MMA) == 6
    @test length(unique(_b1_mma_mods.(EXPECTED_B1_MMA))) == 6
    @test Set((r.shape, r.bitop) for r in EXPECTED_B1_MMA) ==
          Set(PTX.MMA_B1_VARIANTS)
    @test PTX.wrapper_asm_forms(:mma_b1) ==
          [_b1_mma_mods(EXPECTED_B1_MMA[1])]   # m8n8k128.xor only
    @test Set(PTX.wrapper_intrinsic_names(:mma_b1)) ==
          Set(_b1_mma_intrinsic(r) for r in EXPECTED_B1_MMA[2:end])

    for row in EXPECTED_B1_MMA
        op = Operation{:mma, _b1_mma_mods(row)}()
        argtypes = (NTuple{row.n_a, UInt32}, NTuple{row.n_b, UInt32},
                    NTuple{row.n_cd, Int32})
        @test which(op, argtypes).module === PTX
        info = PTX.lowering(op, argtypes)
        @test info.rettype === NTuple{row.n_cd, Int32}
        if row.shape === :m8n8k128 && row.bitop === :xor
            # LLVM 23.1.1 cannot select its existing intrinsic at sm_75, so
            # this sole form is a typed convergent asm fallback.
            @test info.tier === :asm
            @test isempty(info.intrinsics)
            @test info.asm === nothing # direct llvmcall wrapper; PTX pinned offline
        else
            @test info.tier === :intrinsic
            @test info.intrinsics == [_b1_mma_intrinsic(row)]

            intr = NVVM.intrinsic(_b1_mma_intrinsic(row))
            # The arithmetic itself has no memory effects. The mandatory
            # sync/aligned warp rendezvous is supplied by the mma.* convergence
            # overlay at both function and call-site boundaries.
            @test intr.props == (:nomem, :nocallback)
            @test NVVM.is_convergent(intr)
            @test NVVM.callsiteattrs(intr) == "convergent nomerge"
            @test occursin("convergent nomerge", NVVM.fnattrs(intr))
        end
    end
end

@testset "single-bit mma: independent version and target floors" begin
    @test count(r -> r.ptx == v"7.0", EXPECTED_B1_MMA) == 3
    @test count(r -> r.ptx == v"7.1", EXPECTED_B1_MMA) == 3
    @test only(filter(r -> r.sm == v"7.5", EXPECTED_B1_MMA)) ==
          EXPECTED_B1_MMA[1]
    @test count(r -> r.sm == v"8.0", EXPECTED_B1_MMA) == 5
end

@testset "single-bit mma: malformed products fail loud" begin
    bad = (
        # Shape, operation, layout, and modifier order are closed.
        ((:sync, :aligned, :m8n8k256, :row, :col,
          :s32, :b1, :b1, :s32, :xor, :popc),
         (NTuple{1, UInt32}, NTuple{1, UInt32}, NTuple{2, Int32})),
        ((:sync, :aligned, :m8n8k128, :row, :col,
          :s32, :b1, :b1, :s32, :or, :popc),
         (NTuple{1, UInt32}, NTuple{1, UInt32}, NTuple{2, Int32})),
        ((:sync, :aligned, :m8n8k128, :col, :row,
          :s32, :b1, :b1, :s32, :xor, :popc),
         (NTuple{1, UInt32}, NTuple{1, UInt32}, NTuple{2, Int32})),
        ((:sync, :aligned, :m8n8k128, :row, :col,
          :s32, :b1, :b1, :s32, :popc, :xor),
         (NTuple{1, UInt32}, NTuple{1, UInt32}, NTuple{2, Int32})),
        ((:sync, :aligned, :m8n8k128, :row, :col,
          :s32, :b1, :b1, :s32, :xor),
         (NTuple{1, UInt32}, NTuple{1, UInt32}, NTuple{2, Int32})),
        # Fragment counts and signed accumulator/result carrier are exact.
        (_b1_mma_mods(EXPECTED_B1_MMA[3]),
         (NTuple{1, UInt32}, NTuple{1, UInt32}, NTuple{4, Int32})),
        (_b1_mma_mods(EXPECTED_B1_MMA[3]),
         (NTuple{2, Int32}, NTuple{1, UInt32}, NTuple{4, Int32})),
        (_b1_mma_mods(EXPECTED_B1_MMA[3]),
         (NTuple{2, UInt32}, NTuple{1, UInt32}, NTuple{4, UInt32})),
    )

    for (mods, argtypes) in bad
        op = Operation{:mma, mods}()
        info = PTX.lowering(op, argtypes)
        @test info.tier === :forbidden
        @test endswith(String(which(op, argtypes).file), "entries.jl")
    end
end

# --- dense integer ---------------------------------------------------------------

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

# --- sparse integer --------------------------------------------------------------

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
        @test endswith(String(which(op, bad_types).file), "entries.jl")

        intrinsic = PTX.NVVM.intrinsic(name)
        @test :nomem in intrinsic.props
        @test PTX.NVVM.is_convergent(intrinsic)
        @test PTX.NVVM.callsiteattrs(intrinsic) == "convergent nomerge"
    end

    @test method_count == 96
    integer_names = Set(r.intrinsic
                        for r in PTX.wrapper_records(:mma_sp, :mma_sp_ordered)
                        if :s32 in r.mods)
    @test expected_names == integer_names
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
        @test endswith(String(which(op, argtypes).file), "entries.jl")
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

# --- sp::ordered_metadata (floating) ---------------------------------------------

# Independent transcription of the ordered-metadata subset intentionally
# exposed by PTX.jl.  This is not derived from MMA_SP_FRAGS or the registration
# loops: a production edit must reconcile both sides explicitly.
const ORDERED_SP_FORMS = (
    (shape=:m16n8k16, d=:f32, a=:f16,  b=:f16,  c=:f32, na=2, nb=2, nd=4, selectors=0:3),
    (shape=:m16n8k16, d=:f16, a=:f16,  b=:f16,  c=:f16, na=2, nb=2, nd=2, selectors=0:3),
    (shape=:m16n8k16, d=:f32, a=:bf16, b=:bf16, c=:f32, na=2, nb=2, nd=4, selectors=0:3),
    (shape=:m16n8k32, d=:f32, a=:f16,  b=:f16,  c=:f32, na=4, nb=4, nd=4, selectors=0:1),
    (shape=:m16n8k32, d=:f16, a=:f16,  b=:f16,  c=:f16, na=4, nb=4, nd=2, selectors=0:1),
    (shape=:m16n8k32, d=:f32, a=:bf16, b=:bf16, c=:f32, na=4, nb=4, nd=4, selectors=0:1),
    (shape=:m16n8k8,  d=:f32, a=:tf32, b=:tf32, c=:f32, na=2, nb=2, nd=4, selectors=0:3),
    (shape=:m16n8k16, d=:f32, a=:tf32, b=:tf32, c=:f32, na=4, nb=4, nd=4, selectors=0:1),
    (shape=:m16n8k64, d=:f32, a=:e4m3, b=:e4m3, c=:f32, na=4, nb=4, nd=4, selectors=0:0),
    (shape=:m16n8k64, d=:f32, a=:e4m3, b=:e5m2, c=:f32, na=4, nb=4, nd=4, selectors=0:0),
    (shape=:m16n8k64, d=:f32, a=:e5m2, b=:e4m3, c=:f32, na=4, nb=4, nd=4, selectors=0:0),
    (shape=:m16n8k64, d=:f32, a=:e5m2, b=:e5m2, c=:f32, na=4, nb=4, nd=4, selectors=0:0),
)

ordered_mods(f) = (Symbol("sp::ordered_metadata"), :sync, :aligned,
    f.shape, :row, :col, f.d, f.a, f.b, f.c)
ordered_argtypes(f, selector) = (
    NTuple{f.na, UInt32}, NTuple{f.nb, UInt32},
    NTuple{f.nd, f.c === :f32 ? Float32 : UInt32}, UInt32, Val{selector})

@testset "mma.sp::ordered_metadata closed ABI inventory" begin
    @test length(ORDERED_SP_FORMS) == 12
    expected_names = Set{String}()

    for f in ORDERED_SP_FORMS
        op = Operation{:mma, ordered_mods(f)}()
        name = "llvm.nvvm." * PTX._mma_sp_intrinsic_name(
            f.shape, f.a, f.b, f.c; ordered = true)
        push!(expected_names, name)

        for selector in f.selectors
            argtypes = ordered_argtypes(f, selector)
            method = which(op, argtypes)
            info = PTX.lowering(op, argtypes)
            @test method.module === PTX
            @test info.tier === :intrinsic
            @test info.intrinsics == [name]
            @test info.rettype === NTuple{f.nd, f.c === :f32 ? Float32 : UInt32}
        end

        # One past the shape-specific selector domain must hit the guarded
        # generic chain, never an intrinsic whose immarg happens to reject it.
        bad_selector = last(f.selectors) + 1
        bad_types = ordered_argtypes(f, bad_selector)
        @test PTX.lowering(op, bad_types).tier === :forbidden
        @test endswith(String(which(op, bad_types).file), "entries.jl")

        intrinsic = PTX.NVVM.intrinsic(name)
        @test :nomem in intrinsic.props
        @test PTX.NVVM.is_convergent(intrinsic)
        @test PTX.NVVM.callsiteattrs(intrinsic) == "convergent nomerge"
    end

    ordered_names = Set(PTX.wrapper_intrinsic_names(:mma_sp_ordered))
    integer_names = Set(r.intrinsic
                        for r in PTX.wrapper_records(:mma_sp, :mma_sp_ordered)
                        if :s32 in r.mods)

    # This oracle independently pins the floating ordered-metadata island. The
    # production inventory also contains the separately reviewed integer forms.
    @test expected_names == setdiff(ordered_names, integer_names)
    @test length(intersect(ordered_names, integer_names)) == 32
    @test isempty(intersect(expected_names,
                            Set(PTX.wrapper_intrinsic_names(:mma_sp))))
end

@testset "mma.sp::ordered_metadata canonical spelling and fail-loud boundary" begin
    literal = ptx"mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32"
    @test typeof(literal).parameters[2] ==
          (Symbol("sp::ordered_metadata"), :sync, :aligned, :m16n8k32,
           :row, :col, :f32, :bf16, :bf16, :f32)

    # Correct spelling but a fragment-width mismatch cannot decay into the
    # scalar chain.  The same boundary covers unreviewed shapes and dtypes.
    wrong_width = (NTuple{2, UInt32}, NTuple{4, UInt32},
                   NTuple{4, Float32}, UInt32, Val{0})
    @test PTX.lowering(literal, wrong_width).tier === :forbidden

    # m16n8k256 is outside the classic sparse integer shape set in §9.7.15.6.
    unreviewed = Operation{:mma,
        (Symbol("sp::ordered_metadata"), :sync, :aligned, :m16n8k256,
         :row, :col, :s32, :u4, :u4, :s32)}()
    @test PTX.lowering(unreviewed,
        (NTuple{4, UInt32}, NTuple{4, UInt32}, NTuple{4, Int32},
         UInt32, Val{0})).tier === :forbidden

    args = (ntuple(_ -> UInt32(0), Val(2)),
            ntuple(_ -> UInt32(0), Val(4)),
            ntuple(_ -> 0f0, Val(4)), UInt32(0), Val(0))
    err = try
        literal(args...)
        nothing
    catch exception
        exception
    end
    @test err isa ArgumentError
    @test occursin("requires an exact typed wrapper", sprint(showerror, err))
end
