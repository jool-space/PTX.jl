using PTX: Operation

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
        @test endswith(String(which(op, bad_types).file), "inst.jl")

        intrinsic = PTX.NVVM.intrinsic(name)
        @test :nomem in intrinsic.props
        @test PTX.NVVM.is_convergent(intrinsic)
        @test PTX.NVVM.callsiteattrs(intrinsic) == "convergent nomerge"
    end

    @test expected_names == Set(PTX.MMA_SP_ORDERED_INTRINSIC_NAMES)
    @test isempty(intersect(expected_names, Set(PTX.MMA_SP_INTRINSIC_NAMES)))
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

    unreviewed = Operation{:mma,
        (Symbol("sp::ordered_metadata"), :sync, :aligned, :m16n8k64,
         :row, :col, :s32, :u8, :u8, :s32)}()
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
