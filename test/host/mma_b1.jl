using PTX: Operation
using PTX.NVVM: NVVM

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
    @test PTX.MMA_B1_ASM_FORMS == [(:m8n8k128, :xor)]
    @test Set(PTX.MMA_B1_INTRINSIC_NAMES) ==
          Set(_b1_mma_intrinsic(r) for r in EXPECTED_B1_MMA[2:end])

    for row in EXPECTED_B1_MMA
        op = Operation{:mma, _b1_mma_mods(row)}()
        argtypes = (NTuple{row.n_a, UInt32}, NTuple{row.n_b, UInt32},
                    NTuple{row.n_cd, Int32})
        @test which(op, argtypes).module === PTX
        info = PTX.lowering(op, argtypes)
        @test info.rettype === NTuple{row.n_cd, Int32}
        if row.shape === :m8n8k128 && row.bitop === :xor
            # LLVM 22.1.7 cannot select its existing intrinsic at sm_75, so
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
        @test endswith(String(which(op, argtypes).file), "inst.jl")
    end
end
