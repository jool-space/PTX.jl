using PTX
using PTX: Operation, lowering

# Independent PTX 9.3 §9.7.17.8 Table 52 inventory (shape, per-count base
# registers, legal counts, needs-immHalfSplitoff). The production methods
# are intentionally not the source of this oracle.
const _T5_LDST_SHAPES = [
    (Symbol("16x64b"),   1, (1, 2, 4, 8, 16, 32, 64, 128), false),
    (Symbol("32x32b"),   1, (1, 2, 4, 8, 16, 32, 64, 128), false),
    (Symbol("16x128b"),  2, (1, 2, 4, 8, 16, 32, 64),      false),
    (Symbol("16x256b"),  4, (1, 2, 4, 8, 16, 32),          false),
    (Symbol("16x32bx2"), 1, (1, 2, 4, 8, 16, 32, 64, 128), true),
]

_t5_mods(op, shape, c; repack = false) =
    (op, :sync, :aligned, shape, Symbol("x", c),
     (repack ? (Symbol(op === :ld ? "pack::16b" : "unpack::16b"),) : ())...,
     :b32)

@testset "tcgen05 ld/st closed callable surface" begin
    reviewed = Set{String}()

    for (shape, base, counts, split) in _T5_LDST_SHAPES, c in counts,
            repack in (false, true)
        n = base * c
        splitargs = split ? (Val{8},) : ()
        for op in (:ld, :st)
            intrinsic = "llvm.nvvm.tcgen05.$op.$shape.x$c"
            push!(reviewed, intrinsic)
            mods = _t5_mods(op, shape, c; repack)
            args = op === :ld ? (UInt32, splitargs...) :
                                (UInt32, splitargs..., NTuple{n, UInt32})
            info = lowering(Operation{:tcgen05, mods}(), args)
            @test info.tier === :intrinsic
            @test intrinsic in info.intrinsics
            if op === :ld
                @test info.rettype === (n == 1 ? UInt32 : NTuple{n, UInt32})
            else
                @test info.rettype === Nothing
            end

            record = PTX.NVVM.intrinsic(intrinsic)
            @test :convergent in record.props
            if op === :ld
                @test record.ret ===
                      (n == 1 ? (:i32,) : (Symbol("v$(n)i32"),))
                @test record.immargs == (split ? (2, 3) : (2,))
            else
                @test record.ret === ()
                @test record.immargs == (split ? (2, 4) : (3,))
            end
        end
    end

    # Closed world: the reviewed Table-52 grid is exactly the registry's
    # ld/st inventory — a registry addition (e.g. ld.red records at a
    # backend bump) must force a review here.
    registry = Set(name for name in keys(PTX.NVVM.TABLE)
                   if startswith(name, "llvm.nvvm.tcgen05.ld.") ||
                      startswith(name, "llvm.nvvm.tcgen05.st."))
    @test reviewed == registry
    @test length(reviewed) == 74
end

@testset "tcgen05 ld/st rejects grammar and ABI misses" begin
    b2 = Symbol("16x32bx2")
    misses = (
        # Table 52 NA cells: per-lane register count would exceed 128.
        (_t5_mods(:ld, Symbol("16x128b"), 128), (UInt32,)),
        (_t5_mods(:st, Symbol("16x256b"), 64),
         (UInt32, NTuple{256, UInt32})),
        # pack is a load-side qualifier and unpack a store-side one.
        ((:ld, :sync, :aligned, Symbol("16x64b"), :x2,
          Symbol("unpack::16b"), :b32), (UInt32,)),
        ((:st, :sync, :aligned, Symbol("16x64b"), :x2,
          Symbol("pack::16b"), :b32), (UInt32, NTuple{2, UInt32})),
        # Canonical modifier order: shape.num.pack.b32.
        ((:ld, :sync, :aligned, Symbol("16x64b"), Symbol("pack::16b"),
          :x2, :b32), (UInt32,)),
        # immHalfSplitoff is mandatory for 16x32bx2, compile-time only,
        # and illegal elsewhere.
        (_t5_mods(:ld, b2, 2), (UInt32,)),
        (_t5_mods(:ld, b2, 2), (UInt32, Int64)),
        (_t5_mods(:st, b2, 2), (UInt32, NTuple{2, UInt32})),
        (_t5_mods(:ld, Symbol("16x64b"), 2), (UInt32, Val{8})),
        # Data tuple width is pinned by shape×count.
        (_t5_mods(:st, Symbol("16x128b"), 2), (UInt32, NTuple{2, UInt32})),
        (_t5_mods(:st, b2, 2), (UInt32, Val{8}, NTuple{4, UInt32})),
        # Load-with-reduction (PTX 9.1+) has no NVVM records at the pinned
        # backend and stays deliberately outside the wrapper surface.
        ((:ld, :red, :sync, :aligned, Symbol("32x32b"), :x2, :min, :f32),
         (Float32, UInt32)),
    )

    for (mods, args) in misses
        @test lowering(Operation{:tcgen05, mods}(), args).tier === :forbidden
        @test_throws ArgumentError PTX.build_call(:tcgen05, mods, args)
    end
end
