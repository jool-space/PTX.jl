using PTX
using PTX: Operation, lowering

# Independent PTX 9.3 §9.7.15.5.15 inventory. Keep this literal rather than
# deriving it from wrappers/ldmatrix.jl: an accidental production cross-product
# must not be able to narrow its own oracle.
const _LDMATRIX_DECOMP_FORMS = [
    ((:m8n16, :x1), false, 1),
    ((:m8n16, :x2), false, 2),
    ((:m8n16, :x4), false, 4),
    ((:m16n16, :x1), true, 2),
    ((:m16n16, :x2), true, 4),
]
const _LDMATRIX_DECOMP_FORMATS = (:b6x16_p32, :b4x16_p64)

_ldmatrix_decomp_mods(shape_count, trans, space, src_fmt) =
    (:sync, :aligned, shape_count..., (trans ? (:trans,) : ())...,
     space, :b8x16, src_fmt)

@testset "ldmatrix optional-decompression typed surface" begin
    pS8 = Core.LLVMPtr{UInt8, PTX.AS.Shared}
    expected_intrinsics = Set{String}()

    for (shape_count, trans, nout) in _LDMATRIX_DECOMP_FORMS,
        src_fmt in _LDMATRIX_DECOMP_FORMATS
        mods = _ldmatrix_decomp_mods(shape_count, trans, :shared, src_fmt)
        suffix = join((shape_count..., (trans ? (:trans,) : ())...,
                       :b8x16, src_fmt), ".")
        intr = "llvm.nvvm.ldmatrix.sync.aligned.$suffix"
        push!(expected_intrinsics, intr)

        op = Operation{:ldmatrix, mods}()
        info = lowering(op, (pS8,))
        expected_type = nout == 1 ? UInt32 : NTuple{nout, UInt32}
        @test info.tier === :intrinsic
        @test info.rettype === expected_type
        # Optimized reflection may also report the canonical `.p3` overload;
        # the unmangled registry name must remain present.
        @test intr in info.intrinsics
        @test PTX.NVVM.isintrinsic(intr)
        props = PTX.NVVM.intrinsic(intr).props
        @test :readmem in props
        @test :argmemonly in props
        @test :convergent in props

        cta_mods = _ldmatrix_decomp_mods(
            shape_count, trans, Symbol("shared::cta"), src_fmt)
        cta_info = lowering(Operation{:ldmatrix, cta_mods}(), (pS8,))
        @test cta_info.tier === :asm
        @test cta_info.rettype === expected_type
        ci, _ = only(Base.code_typed(Operation{:ldmatrix, cta_mods}(), (pS8,)))
        typed = string(ci)
        @test occursin("ldmatrix.sync.aligned.$(shape_count[1]).$(shape_count[2])" *
                       (trans ? ".trans" : "") *
                       ".shared::cta.b8x16.$src_fmt", typed)
        @test occursin("convergent nomerge", typed)
    end

    registry_names = Set(name for name in keys(PTX.NVVM.TABLE)
                         if startswith(name, "llvm.nvvm.ldmatrix.") &&
                            (endswith(name, ".b8x16.b6x16_p32") ||
                             endswith(name, ".b8x16.b4x16_p64")))
    @test expected_intrinsics == registry_names
    @test length(expected_intrinsics) == 10
end

@testset "ldmatrix decompression rejects grammar and ABI misses" begin
    pS8 = Core.LLVMPtr{UInt8, PTX.AS.Shared}
    pG8 = Core.LLVMPtr{UInt8, PTX.AS.Global}
    invalid = (
        # m8n16 does not admit .trans; m16n16 requires it.
        (:sync, :aligned, :m8n16, :x1, :trans, :shared,
         :b8x16, :b6x16_p32),
        (:sync, :aligned, :m16n16, :x1, :shared,
         :b8x16, :b6x16_p32),
        # m16n16 has no x4 form; decompression has no m8n8 shape.
        (:sync, :aligned, :m16n16, :x4, :trans, :shared,
         :b8x16, :b4x16_p64),
        (:sync, :aligned, :m8n8, :x1, :shared,
         :b8x16, :b4x16_p64),
        # Formats are an ordered pair, not independently permutable tokens.
        (:sync, :aligned, :m8n16, :x1, :shared,
         :b4x16_p64, :b8x16),
        (:sync, :aligned, :m8n16, :x1, :shared,
         :b8x16, :b5x16_p48),
    )
    for mods in invalid
        info = lowering(Operation{:ldmatrix, mods}(), (pS8,))
        @test info.tier === :forbidden
        @test_throws ArgumentError PTX.build_call(:ldmatrix, mods, (pS8,))
    end

    # A non-shared pointer misses the exact method and cannot fall through to
    # a scalar-result guess. Generic-address ldmatrix is legal PTX, but is not
    # represented by this typed surface.
    mods = (:sync, :aligned, :m8n16, :x1, :shared,
            :b8x16, :b4x16_p64)
    @test lowering(Operation{:ldmatrix, mods}(), (pG8,)).tier === :forbidden
    @test_throws ArgumentError PTX.build_call(:ldmatrix, mods, (pG8,))
end
