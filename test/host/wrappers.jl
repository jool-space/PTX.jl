using PTX: format_call, build_call, Operation

# Goldens: byte-exact PTX-text for each typed-chain call site, without
# touching LLVM or the GPU. Tests that the chain's @generated body would
# produce the right asm — same string the runtime call emits.

@testset "f32 ALU" begin
    @test format_call(ptx"add.f32",    Tuple{Float32, Float32})           == "add.f32 \$0, \$1, \$2;"
    @test format_call(ptx"sub.f32",    Tuple{Float32, Float32})           == "sub.f32 \$0, \$1, \$2;"
    @test format_call(ptx"mul.f32",    Tuple{Float32, Float32})           == "mul.f32 \$0, \$1, \$2;"
    @test format_call(ptx"fma.rn.f32", Tuple{Float32, Float32, Float32})  == "fma.rn.f32 \$0, \$1, \$2, \$3;"
    @test format_call(ptx"min.f32",    Tuple{Float32, Float32})           == "min.f32 \$0, \$1, \$2;"
    @test format_call(ptx"max.f32",    Tuple{Float32, Float32})           == "max.f32 \$0, \$1, \$2;"

    # Pure ALU should not carry the memory clobber or side_effects=true.
    spec = build_call(:add, (:f32,), (Float32, Float32))
    @test spec.side_effects == false
    @test !occursin("memory", spec.constraints)
end

@testset "mov.u32 (special register)" begin
    # `sreg"..."` bakes the verbatim PTX token into the asm, no input slot.
    @test format_call(ptx"mov.u32", Tuple{typeof(sreg"%tid.x")})            == "mov.u32 \$0, %tid.x;"
    @test format_call(ptx"mov.u32", Tuple{typeof(sreg"%ntid.y")})           == "mov.u32 \$0, %ntid.y;"
    @test format_call(ptx"mov.u32", Tuple{typeof(sreg"%dynamic_smem_size")}) == "mov.u32 \$0, %dynamic_smem_size;"
    # Register-copy form: integer arg becomes \$1.
    @test format_call(ptx"mov.u32", Tuple{UInt32})                          == "mov.u32 \$0, \$1;"

    # Reading a special register is observable.
    @test build_call(:mov, (:u32,), (typeof(sreg"%tid.x"),)).side_effects == true
end

@testset "mov.u32 sreg NVVM whitelist" begin
    # Invariant-per-thread sregs route through `llvm.nvvm.read.ptx.sreg.*`
    # so LLVM can CSE the reads and propagate range metadata.
    @test haskey(PTX.NVVM_SREG_U32, Symbol("%tid.x"))
    @test haskey(PTX.NVVM_SREG_U32, Symbol("%ntid.y"))
    @test haskey(PTX.NVVM_SREG_U32, Symbol("%ctaid.z"))
    @test haskey(PTX.NVVM_SREG_U32, Symbol("%laneid"))
    @test haskey(PTX.NVVM_SREG_U32, Symbol("%warpsize"))
    @test haskey(PTX.NVVM_SREG_U32, Symbol("%cluster_ctarank"))
    @test haskey(PTX.NVVM_SREG_U32, Symbol("%lanemask_eq"))

    # PTX-asm underscores become dots in the LLVM intrinsic name.
    @test PTX.NVVM_SREG_U32[Symbol("%cluster_ctarank")] == "cluster.ctarank"
    @test PTX.NVVM_SREG_U32[Symbol("%lanemask_eq")]     == "lanemask.eq"

    # Volatile sregs deliberately stay on the asm path so `~{memory}` blocks
    # CSE. Per PTX spec these may return different values on each read.
    @test !haskey(PTX.NVVM_SREG_U32, Symbol("%clock"))
    @test !haskey(PTX.NVVM_SREG_U32, Symbol("%clock64"))
    @test !haskey(PTX.NVVM_SREG_U32, Symbol("%activemask"))
    @test !haskey(PTX.NVVM_SREG_U32, Symbol("%smid"))
    @test !haskey(PTX.NVVM_SREG_U32, Symbol("%warpid"))
    @test !haskey(PTX.NVVM_SREG_U32, Symbol("%dynamic_smem_size"))

    # `format_call` reflects the asm fallback unchanged — the NVVM path only
    # kicks in at runtime through the @generated dispatch.
    @test format_call(ptx"mov.u32", Tuple{typeof(sreg"%tid.x")}) ==
        "mov.u32 \$0, %tid.x;"
end

@testset "bar.sync" begin
    # Integer-Val bakes immediate `0` / `15` into the asm.
    @test format_call(ptx"bar.sync", Tuple{Val{0}})  == "bar.sync 0;"
    @test format_call(ptx"bar.sync", Tuple{Val{15}}) == "bar.sync 15;"
    # Register form.
    @test format_call(ptx"bar.sync", Tuple{UInt32})  == "bar.sync \$0;"

    # Sync-group opcode → side_effects=true + memory clobber.
    spec = build_call(:bar, (:sync,), (Val{0},))
    @test spec.side_effects == true
    @test occursin("~{memory}", spec.constraints)
end

@testset "atom.add.gpu.u32" begin
    @test format_call(ptx"atom.add.gpu.u32",
                      Tuple{Core.LLVMPtr{UInt32, PTX.AS.Global}, UInt32}) ==
        "atom.add.gpu.u32 \$0, [\$1], \$2;"

    # Memory-group opcode → side_effects=true + memory clobber.
    spec = build_call(:atom, (:add, :gpu, :u32),
                      (Core.LLVMPtr{UInt32, PTX.AS.Global}, UInt32))
    @test spec.side_effects == true
    @test occursin("~{memory}", spec.constraints)
end

@testset "cp.async sync ops" begin
    @test format_call(ptx"cp.async.commit_group", Tuple{})         == "cp.async.commit_group;"
    @test format_call(ptx"cp.async.wait_group",   Tuple{Val{0}})   == "cp.async.wait_group 0;"
    @test format_call(ptx"cp.async.wait_group",   Tuple{Val{3}})   == "cp.async.wait_group 3;"
    @test format_call(ptx"cp.async.wait_all",     Tuple{})         == "cp.async.wait_all;"
end

@testset "setp single-pred" begin
    # setp's terminal modifier (.s32, .f32, ...) is the *compare* type, not
    # the result type. The result is always a predicate (Bool / i1).
    @test format_call(ptx"setp.eq.s32", Tuple{Int32, Int32})     == "setp.eq.s32 \$0, \$1, \$2;"
    @test format_call(ptx"setp.lt.f32", Tuple{Float32, Float32}) == "setp.lt.f32 \$0, \$1, \$2;"
    @test format_call(ptx"setp.ge.u32", Tuple{UInt32, UInt32})   == "setp.ge.u32 \$0, \$1, \$2;"

    # Result constraint is `=b` (i1 → Bool); inputs use their own letters.
    spec = build_call(:setp, (:eq, :s32), (Int32, Int32))
    @test spec.rettype == Bool
    @test startswith(spec.constraints, "=b,")
    @test spec.side_effects == false

    spec = build_call(:setp, (:lt, :f32), (Float32, Float32))
    @test spec.constraints == "=b,f,f"

    # Bool input → `b` constraint (e.g. setp.eq.pred for pred-input compare).
    @test build_call(:setp, (:eq, :s32), (Int32, Int32)).constraints == "=b,r,r"
end

@testset "vote.sync predicate-output" begin
    # vote.sync.{all,any,uni}.pred: pred in, pred out.
    @test format_call(ptx"vote.sync.all.pred", Tuple{Bool, UInt32}) ==
          "vote.sync.all.pred \$0, \$1, \$2;"
    @test format_call(ptx"vote.sync.any.pred", Tuple{Bool, UInt32}) ==
          "vote.sync.any.pred \$0, \$1, \$2;"
    @test format_call(ptx"vote.sync.uni.pred", Tuple{Bool, UInt32}) ==
          "vote.sync.uni.pred \$0, \$1, \$2;"

    spec = build_call(:vote, (:sync, :all, :pred), (Bool, UInt32))
    @test spec.rettype == Bool
    @test spec.constraints == "=b,b,r,~{memory}"     # warp-collective → nonpure
    @test spec.side_effects == true

    # vote.sync.ballot.b32: pred in, u32 mask out (one bit per lane).
    @test format_call(ptx"vote.sync.ballot.b32", Tuple{Bool, UInt32}) ==
          "vote.sync.ballot.b32 \$0, \$1, \$2;"
    spec = build_call(:vote, (:sync, :ballot, :b32), (Bool, UInt32))
    @test spec.rettype == UInt32
    @test spec.constraints == "=r,b,r,~{memory}"
    @test spec.side_effects == true
end

@testset "PTX.AS module" begin
    @test PTX.AS.Generic == 0
    @test PTX.AS.Global  == 1
    @test PTX.AS.Shared  == 3
    @test PTX.AS.Const   == 4
    @test PTX.AS.Local   == 5
    @test PTX.AS.Param   == 101
end

@testset "constraint_letter" begin
    # NVPTX register-class letters per src/types.jl. Wrong letters → ptxas
    # rejects with "Invalid register class" or silent miscompile (e.g. an i8
    # value passed via "r" is zero-extended into the high bits).
    @test PTX.constraint_letter(Float64) == "d"
    @test PTX.constraint_letter(Float32) == "f"
    @test PTX.constraint_letter(Float16) == "h"

    # i8 has no native NVPTX register; both signed and unsigned are bridged
    # through i16.
    @test PTX.constraint_letter(Int8)   == "h"
    @test PTX.constraint_letter(UInt8)  == "h"

    @test PTX.constraint_letter(Int16)  == "h"
    @test PTX.constraint_letter(UInt16) == "h"
    @test PTX.constraint_letter(Int32)  == "r"
    @test PTX.constraint_letter(UInt32) == "r"
    @test PTX.constraint_letter(Int64)  == "l"
    @test PTX.constraint_letter(UInt64) == "l"
    @test PTX.constraint_letter(Bool)   == "b"

    # LLVMPtr always uses "l" regardless of address space — NVPTX represents
    # non-zero AS pointers as 64-bit at the LLVM IR level even when the
    # underlying PTX address is 32-bit.
    @test PTX.constraint_letter(Core.LLVMPtr{Float32, PTX.AS.Generic}) == "l"
    @test PTX.constraint_letter(Core.LLVMPtr{Float32, PTX.AS.Global})  == "l"
    @test PTX.constraint_letter(Core.LLVMPtr{Float32, PTX.AS.Shared})  == "l"
    @test PTX.constraint_letter(Core.LLVMPtr{UInt8,   PTX.AS.Const})   == "l"
    @test PTX.constraint_letter(Core.LLVMPtr{UInt64,  PTX.AS.Local})   == "l"

    # Indirect coverage via build_call — confirms the letter flows into the
    # constraint string for both i8 inputs and LLVMPtr inputs.
    @test build_call(:add, (:s16,), (Int8, Int8)).constraints  == "=h,h,h"
    @test build_call(:add, (:u16,), (UInt8, UInt8)).constraints == "=h,h,h"
    @test build_call(:ld, (:global, :u32),
                     (Core.LLVMPtr{UInt32, PTX.AS.Global},)).constraints ==
        "=r,l,~{memory}"
end

@testset "Tier 1 chain entries (mad / lop3 / prmt / redux / match / membar / etc.)" begin
    # `mad` — pure ALU, no memory clobber.
    @test format_call(ptx"mad.lo.s32", Tuple{Int32, Int32, Int32}) ==
          "mad.lo.s32 \$0, \$1, \$2, \$3;"
    spec = build_call(:mad, (:lo, :s32), (Int32, Int32, Int32))
    @test spec.side_effects == false
    @test !occursin("~{memory}", spec.constraints)
    @test spec.rettype == Int32

    # `lop3` — pure 3-input bitwise lookup; immediate baked via Val.
    @test format_call(ptx"lop3.b32", Tuple{UInt32, UInt32, UInt32, Val{0xC0}}) ==
          "lop3.b32 \$0, \$1, \$2, \$3, 192;"
    spec = build_call(:lop3, (:b32,), (UInt32, UInt32, UInt32, Val{0xC0}))
    @test spec.side_effects == false
    @test spec.rettype == UInt32

    # `prmt` — pure byte permutation.
    @test format_call(ptx"prmt.b32", Tuple{UInt32, UInt32, UInt32}) ==
          "prmt.b32 \$0, \$1, \$2, \$3;"
    spec = build_call(:prmt, (:b32,), (UInt32, UInt32, UInt32))
    @test spec.side_effects == false

    # `redux.sync.add.u32` — warp-collective; needs `~{memory}` to block CSE.
    @test format_call(ptx"redux.sync.add.u32", Tuple{UInt32, UInt32}) ==
          "redux.sync.add.u32 \$0, \$1, \$2;"
    spec = build_call(:redux, (:sync, :add, :u32), (UInt32, UInt32))
    @test spec.side_effects == true
    @test occursin("~{memory}", spec.constraints)

    # `match.any.sync.b32` — warp-collective.
    spec = build_call(:match, (:any, :sync, :b32), (UInt32, UInt32))
    @test spec.side_effects == true
    @test occursin("~{memory}", spec.constraints)

    # `membar.gl` — memory barrier.
    @test format_call(ptx"membar.gl", Tuple{}) == "membar.gl;"
    spec = build_call(:membar, (:gl,), ())
    @test spec.side_effects == true
    @test occursin("~{memory}", spec.constraints)

    # sm_90 cluster intrinsics — observable cross-CTA visibility.
    spec = build_call(:mapa, (Symbol("shared::cluster",), :u32), (UInt32, UInt32))
    @test spec.side_effects == true
    spec = build_call(:getctarank, (Symbol("shared::cluster",), :u32), (UInt32,))
    @test spec.side_effects == true

    # `griddepcontrol.wait` — inter-launch dependency, needs barrier.
    spec = build_call(:griddepcontrol, (:wait,), ())
    @test spec.side_effects == true
end

@testset "stmatrix.sync.aligned hand-written wrapper" begin
    # m8n8.x4.shared.b16 — canonical 4-fragment store (Ada+).
    spec = PTX.stmatrix_spec(:m8n8, :x4, false, :shared, :b16)
    @test spec.n_in == 4
    @test spec.asm ==
        "stmatrix.sync.aligned.m8n8.x4.shared.b16 " *
        "[\$0], {\$1, \$2, \$3, \$4};"
    @test spec.constraints == "r,r,r,r,r,~{memory}"

    # x1 — scalar-input form (no NTuple wrapper).
    spec1 = PTX.stmatrix_spec(:m8n8, :x1, false, :shared, :b16)
    @test spec1.n_in == 1
    @test spec1.asm ==
        "stmatrix.sync.aligned.m8n8.x1.shared.b16 [\$0], {\$1};"
    @test spec1.constraints == "r,r,~{memory}"

    # x2 with .trans and .shared::cta state-space.
    spec2 = PTX.stmatrix_spec(:m8n8, :x2, true, :shared_cta, :b16)
    @test spec2.asm ==
        "stmatrix.sync.aligned.m8n8.x2.trans.shared::cta.b16 " *
        "[\$0], {\$1, \$2};"

    # Hopper m16n8.b8 — declared but unreachable on Ada.
    spec_h = PTX.stmatrix_spec(:m16n8, :x4, false, :shared, :b8)
    @test occursin("m16n8.x4.shared.b8", spec_h.asm)

    # Methods registered for representative variants.
    @test which(Operation{:stmatrix, (:sync, :aligned, :m8n8, :x4, :shared, :b16)}(),
                (Core.LLVMPtr{UInt16, PTX.AS.Shared}, NTuple{4, UInt32})).module == PTX
    @test which(Operation{:stmatrix, (:sync, :aligned, :m8n8, :x1, :shared, :b16)}(),
                (Core.LLVMPtr{UInt16, PTX.AS.Shared}, UInt32)).module == PTX
end

@testset "mbarrier wrapper (tier-2 intrinsic lowering)" begin
    # Second migrated family (DESIGN.md): the ten ::cta forms lower through
    # llvm.nvvm.mbarrier.* — legacy *.shared intrinsics where the form is
    # sm_80 (the scoped count-form arrive can't ISel below sm_90), scoped
    # *.scope.cta.space.cta for parity waits (no legacy exists) and the
    # sm_90 forms. Goldens: test/golden/mbarrier@sm{80,90}.ptx. The
    # cluster-space sink forms stay asm-tier pending AS-7 modeling.
    pS = Core.LLVMPtr{UInt64, PTX.AS.Shared}
    cases = [
        ((:init, :shared, :b64),                (pS, UInt32), Nothing,
            "llvm.nvvm.mbarrier.init.shared"),
        ((:inval, :shared, :b64),               (pS,),        Nothing,
            "llvm.nvvm.mbarrier.inval.shared"),
        ((:arrive, :shared, :b64),              (pS,),        UInt64,
            "llvm.nvvm.mbarrier.arrive.shared"),
        ((:arrive, :noComplete, :shared, :b64), (pS, UInt32), UInt64,
            "llvm.nvvm.mbarrier.arrive.noComplete.shared"),
        ((:arrive, :expect_tx, :shared, :b64),  (pS, UInt32), UInt64,
            "llvm.nvvm.mbarrier.arrive.expect.tx.scope.cta.space.cta"),
        ((:expect_tx, :shared, :b64),           (pS, UInt32), Nothing,
            "llvm.nvvm.mbarrier.expect.tx.scope.cta.space.cta"),
        ((:test_wait, :shared, :b64),           (pS, UInt64), Bool,
            "llvm.nvvm.mbarrier.test.wait.shared"),
        ((:test_wait, :parity, :shared, :b64),  (pS, UInt32), Bool,
            "llvm.nvvm.mbarrier.test.wait.parity.scope.cta.space.cta"),
        ((:try_wait, :shared, :b64),            (pS, UInt64), Bool,
            "llvm.nvvm.mbarrier.try.wait.scope.cta.space.cta"),
        ((:try_wait, :parity, :shared, :b64),   (pS, UInt32), Bool,
            "llvm.nvvm.mbarrier.try.wait.parity.scope.cta.space.cta"),
    ]
    for (mods, argts, rettype, intr) in cases
        # the intrinsic is registered and convergent
        @test PTX.NVVM.isintrinsic(intr)
        @test :convergent in PTX.NVVM.intrinsic(intr).props
        # the method dispatches and routes to that intrinsic
        op = Operation{:mbarrier, mods}()
        @test which(op, argts).module == PTX
        ci, rt = first(Base.code_typed(op, argts))
        @test rt === rettype
        @test occursin(intr, string(ci))
    end

    # integer-flexible signatures still convert (count::Integer etc.)
    ci, rt = first(Base.code_typed(Operation{:mbarrier, (:init, :shared, :b64)}(),
                                   (pS, Int)))
    @test rt === Nothing

    # cluster-space sink forms remain on the asm tier
    for (mods, argts, asm) in [
        ((:arrive, Symbol("shared::cluster"), :b64), (pS,),
            "mbarrier.arrive.shared::cluster.b64 _, ["),
        ((:arrive, :expect_tx, Symbol("shared::cluster"), :b64), (pS, UInt32),
            "mbarrier.arrive.expect_tx.shared::cluster.b64 _, ["),
    ]
        ci, _ = first(Base.code_typed(Operation{:mbarrier, mods}(), argts))
        @test occursin(asm, string(ci))
    end
end


@testset "wgmma sync ops" begin
    # wgmma.fence/commit_group/wait_group flow through the chain default.
    @test format_call(ptx"wgmma.fence.sync.aligned",        Tuple{})       == "wgmma.fence.sync.aligned;"
    @test format_call(ptx"wgmma.commit_group.sync.aligned", Tuple{})       == "wgmma.commit_group.sync.aligned;"
    @test format_call(ptx"wgmma.wait_group.sync.aligned",   Tuple{Val{0}}) == "wgmma.wait_group.sync.aligned 0;"
    @test format_call(ptx"wgmma.wait_group.sync.aligned",   Tuple{Val{2}}) == "wgmma.wait_group.sync.aligned 2;"

    # `:wgmma` is in NONPURE_OPCODES → memory clobber + side_effects.
    spec = build_call(:wgmma, (:fence, :sync, :aligned), ())
    @test spec.side_effects == true
    @test occursin("~{memory}", spec.constraints)
end

@testset "wgmma.mma_async hand-written wrapper" begin
    # bf16 m64n8k16, f32 acc: nd = N/2 = 4, constraint letter `f`.
    spec = PTX.wgmma_mma_async_spec(:f32, :bf16, :bf16, 8, 16, true)
    @test spec.nd == 4
    @test spec.d_let == "f"
    @test spec.asm ==
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 " *
        "{\$0, \$1, \$2, \$3}, \$4, \$5, \$6, 1, 1, 0, 0;"
    @test spec.constraints == "=f,=f,=f,=f,l,l,b,0,1,2,3,~{memory}"

    # f16 input, f32 acc: same shape, different ab-dtype.
    spec_f16 = PTX.wgmma_mma_async_spec(:f32, :f16, :f16, 8, 16, true)
    @test spec_f16.asm ==
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 " *
        "{\$0, \$1, \$2, \$3}, \$4, \$5, \$6, 1, 1, 0, 0;"

    # tf32 (k=8) has no transpose → only `1, 1` baked.
    spec_tf32 = PTX.wgmma_mma_async_spec(:f32, :tf32, :tf32, 8, 8, false)
    @test spec_tf32.asm ==
        "wgmma.mma_async.sync.aligned.m64n8k8.f32.tf32.tf32 " *
        "{\$0, \$1, \$2, \$3}, \$4, \$5, \$6, 1, 1;"

    # FP8 e4m3 with f32 acc, k=32 (Hopper FP8 GEMM).
    spec_e4m3 = PTX.wgmma_mma_async_spec(:f32, :e4m3, :e4m3, 16, 32, false)
    @test spec_e4m3.asm ==
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 " *
        "{\$0, \$1, \$2, \$3, \$4, \$5, \$6, \$7}, \$8, \$9, \$10, 1, 1;"

    # f16 accumulator: nd = N/4, constraint letter `r` (UInt32 packed f16x2).
    spec_f16acc = PTX.wgmma_mma_async_spec(:f16, :f16, :f16, 16, 16, true)
    @test spec_f16acc.nd == 4               # N=16, /4 = 4 packed regs
    @test spec_f16acc.d_let == "r"
    @test spec_f16acc.asm ==
        "wgmma.mma_async.sync.aligned.m64n16k16.f16.f16.f16 " *
        "{\$0, \$1, \$2, \$3}, \$4, \$5, \$6, 1, 1, 0, 0;"
    @test spec_f16acc.constraints == "=r,=r,=r,=r,l,l,b,0,1,2,3,~{memory}"

    # Largest covered shape: m64n256k16 f32-acc → 128 d-regs.
    spec_big = PTX.wgmma_mma_async_spec(:f32, :bf16, :bf16, 256, 16, true)
    @test spec_big.nd == 128
    @test occursin("m64n256k16.f32.bf16.bf16", spec_big.asm)
    @test occursin("{\$0, \$1, \$2, ", spec_big.asm)
    @test occursin(", \$126, \$127}", spec_big.asm)
    @test endswith(spec_big.constraints, ",126,127,~{memory}")

    # m64n256k16 f16-acc → 64 packed regs.
    spec_big_f16 = PTX.wgmma_mma_async_spec(:f16, :f16, :f16, 256, 16, true)
    @test spec_big_f16.nd == 64

    # Methods registered for representative variants.
    @test which(Operation{:wgmma, (:mma_async, :sync, :aligned, :m64n256k16, :f32, :bf16, :bf16)}(),
                (NTuple{128, Float32}, UInt64, UInt64, Bool)).module == PTX

    @test which(Operation{:wgmma, (:mma_async, :sync, :aligned, :m64n16k16, :f16, :f16, :f16)}(),
                (NTuple{4, UInt32}, UInt64, UInt64, Bool)).module == PTX

    @test which(Operation{:wgmma, (:mma_async, :sync, :aligned, :m64n16k32, :f32, :e4m3, :e4m3)}(),
                (NTuple{8, Float32}, UInt64, UInt64, Bool)).module == PTX
end

@testset "wgmma.mma_async — full _WGMMA_VARIANTS coverage" begin
    # Per-variant goldens for every entry of `_WGMMA_VARIANTS` in
    # src/wrappers/wgmma.jl. The encoder is cross-checked against pyptx's
    # `_Wgmma.mma_async()` (pyptx/pyptx/ptx.py:817-910) — same modifier
    # ordering, same imm tail (`scaleD, scaleA[, transA, transB]`), same
    # tied-operand convention. Corpus reference:
    # pyptx/tests/corpus/wgmma_simple.ptx (m64n256k16.f32.e4m3.e4m3) and
    # wgmma_gemm_tile.ptx (m64n128k16.f32.bf16.bf16).
    #
    # The five "popular" combos are spot-checked above; this set fills the
    # remaining 7 of 12 variants × representative N at each end of the grid.

    # FP8 e5m2 with f32 acc — k=32, no transpose imms. N=8 → 4 d-regs.
    spec = PTX.wgmma_mma_async_spec(:f32, :e5m2, :e5m2, 8, 32, false)
    @test spec.nd == 4
    @test spec.d_let == "f"
    @test spec.asm ==
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e5m2.e5m2 " *
        "{\$0, \$1, \$2, \$3}, \$4, \$5, \$6, 1, 1;"
    @test spec.constraints == "=f,=f,=f,=f,l,l,b,0,1,2,3,~{memory}"

    # f16 acc + e4m3 inputs — packed f16x2 d-regs (nd = N/4), no trans imms.
    spec = PTX.wgmma_mma_async_spec(:f16, :e4m3, :e4m3, 16, 32, false)
    @test spec.nd == 4              # N=16, /4 = 4 packed regs
    @test spec.d_let == "r"
    @test spec.asm ==
        "wgmma.mma_async.sync.aligned.m64n16k32.f16.e4m3.e4m3 " *
        "{\$0, \$1, \$2, \$3}, \$4, \$5, \$6, 1, 1;"
    @test spec.constraints == "=r,=r,=r,=r,l,l,b,0,1,2,3,~{memory}"

    # f16 acc + e5m2 inputs — same shape, different ab-dtype.
    spec = PTX.wgmma_mma_async_spec(:f16, :e5m2, :e5m2, 16, 32, false)
    @test spec.asm ==
        "wgmma.mma_async.sync.aligned.m64n16k32.f16.e5m2.e5m2 " *
        "{\$0, \$1, \$2, \$3}, \$4, \$5, \$6, 1, 1;"

    # Integer accumulator: s32 with all four (s8/u8) × (s8/u8) ab-pairs.
    # PTX 9.2 §9.7.14.5.7: integer wgmma has only `scale-d` (no scale-a/b
    # or trans imms) — asm tail is just `, $scale_d;` with no trailing
    # `1, 1`. nd = N/2, constraint letter `r` (Int32 d-regs).
    for (dt_a, dt_b) in ((:s8, :s8), (:u8, :u8), (:s8, :u8), (:u8, :s8))
        spec = PTX.wgmma_mma_async_spec(:s32, dt_a, dt_b, 8, 32, false)
        @test spec.nd == 4
        @test spec.d_let == "r"
        @test spec.asm ==
            "wgmma.mma_async.sync.aligned.m64n8k32.s32.$dt_a.$dt_b " *
            "{\$0, \$1, \$2, \$3}, \$4, \$5, \$6;"
        @test spec.constraints == "=r,=r,=r,=r,l,l,b,0,1,2,3,~{memory}"
    end

    # Largest s32 shape (m64n256k32) — 128 d-regs, locks the brace-count loop.
    spec = PTX.wgmma_mma_async_spec(:s32, :s8, :s8, 256, 32, false)
    @test spec.nd == 128
    @test occursin("m64n256k32.s32.s8.s8", spec.asm)
    @test occursin(", \$126, \$127}", spec.asm)
    @test endswith(spec.asm, ", \$128, \$129, \$130;")
    @test endswith(spec.constraints, ",126,127,~{memory}")

    # Methods registered for the new variants — locks dispatch.
    @test which(Operation{:wgmma,
            (:mma_async, :sync, :aligned, :m64n8k32, :f32, :e5m2, :e5m2)}(),
        (NTuple{4, Float32}, UInt64, UInt64, Bool)).module == PTX
    @test which(Operation{:wgmma,
            (:mma_async, :sync, :aligned, :m64n16k32, :f16, :e4m3, :e4m3)}(),
        (NTuple{4, UInt32}, UInt64, UInt64, Bool)).module == PTX
    @test which(Operation{:wgmma,
            (:mma_async, :sync, :aligned, :m64n8k32, :s32, :s8, :u8)}(),
        (NTuple{4, Int32}, UInt64, UInt64, Bool)).module == PTX
end

@testset "wgmma_descriptor packing" begin
    using PTX: wgmma_descriptor, WgmmaSwizzle

    # Encoder cross-reference: pyptx's `_Wgmma.make_descriptor()`
    # (pyptx/pyptx/ptx.py:1022-1144) implements the same 14-bit field layout
    # (start_addr / leading_offset / stride_offset / base_offset / swizzle).
    # The two encoders are independent; tests below validate the bit layout
    # directly against PTX 9.2 §9.7.14.5.

    # Empty descriptor: all zeros.
    @test wgmma_descriptor(UInt32(0); leading_byte_offset = 0,
                                       stride_byte_offset = 0) === UInt64(0)

    # Address-only: bits [13:0] = (addr & 0x3FFFF) >> 4.
    # 0x100 → 0x10. Sits in low 14 bits.
    @test wgmma_descriptor(UInt32(0x100); leading_byte_offset = 0,
                                          stride_byte_offset = 0) ===
          UInt64(0x10)

    # Address truncated past 18 bits: 0x40000 (bit 18) masks to 0 → 0.
    @test wgmma_descriptor(UInt32(0x40000); leading_byte_offset = 0,
                                            stride_byte_offset = 0) ===
          UInt64(0)

    # Leading byte offset: 16 → encoded `1`, lands at bit 16.
    d = wgmma_descriptor(UInt32(0); leading_byte_offset = 16,
                                     stride_byte_offset = 0)
    @test d === UInt64(1) << 16

    # Stride byte offset: 128 → encoded `8`, lands at bit 32.
    d = wgmma_descriptor(UInt32(0); leading_byte_offset = 0,
                                     stride_byte_offset = 128)
    @test d === UInt64(8) << 32

    # Swizzle mode: 128B (= 1) → bits [63:62].
    d = wgmma_descriptor(UInt32(0); leading_byte_offset = 0,
                                     stride_byte_offset = 0,
                                     swizzle = WgmmaSwizzle.B128)
    @test d === UInt64(1) << 62

    # base_offset: 5 → bits [51:49].
    d = wgmma_descriptor(UInt32(0); leading_byte_offset = 0,
                                     stride_byte_offset = 0,
                                     base_offset = 5)
    @test d === UInt64(5) << 49

    # Combined: canonical bf16 16×8 tile descriptor.
    #   addr = 0x200 → encoded 0x20 at [13:0]
    #   leading = 16 → encoded 1 at [29:16]
    #   stride  = 128 → encoded 8 at [45:32]
    #   swizzle = NONE
    d = wgmma_descriptor(UInt32(0x200); leading_byte_offset = 16,
                                         stride_byte_offset = 128)
    expected = UInt64(0x20) | (UInt64(1) << 16) | (UInt64(8) << 32)
    @test d === expected

    # Swizzle mode constants match PTX encoding.
    @test WgmmaSwizzle.NONE == 0
    @test WgmmaSwizzle.B128 == 1
    @test WgmmaSwizzle.B64  == 2
    @test WgmmaSwizzle.B32  == 3
end

@testset "tcgen05_instr_desc_f16bf16_f32 packing" begin
    using PTX: tcgen05_instr_desc_f16bf16_f32

    # Goldens cross-checked against pyptx's
    # `test_make_instr_desc_f16bf16_f32` cases (m=128, n=256, K-major).
    @test tcgen05_instr_desc_f16bf16_f32(m = 128, n = 256, ab_dtype = :bf16) ===
          UInt32(0x08400490)
    @test tcgen05_instr_desc_f16bf16_f32(m = 128, n = 256, ab_dtype = :f16) ===
          UInt32(0x08400010)
    @test tcgen05_instr_desc_f16bf16_f32(m = 128, n = 256, ab_dtype = :f16,
                                         scale_a = -1) ===
          UInt32(0x08402010)
    @test tcgen05_instr_desc_f16bf16_f32(m = 128, n = 256, ab_dtype = :tf32) ===
          UInt32(0x08400910)

    # MN-major flips bit 15 (A) / 16 (B).
    base = tcgen05_instr_desc_f16bf16_f32(m = 128, n = 256, ab_dtype = :bf16)
    @test tcgen05_instr_desc_f16bf16_f32(m = 128, n = 256, ab_dtype = :bf16,
                                         a_major = :MN) ===
          (base | (UInt32(1) << 15))
    @test tcgen05_instr_desc_f16bf16_f32(m = 128, n = 256, ab_dtype = :bf16,
                                         a_major = :MN, b_major = :MN) ===
          (base | (UInt32(1) << 15) | (UInt32(1) << 16))

    # M and N field placement: m=64 → bits[27:24]=4, m=256 → 16.
    # n=8 → bits[22:17]=1, n=128 → 16.
    @test tcgen05_instr_desc_f16bf16_f32(m = 64,  n = 256, ab_dtype = :bf16) ===
          UInt32((4  << 24) | (32 << 17) | 0x490)
    @test tcgen05_instr_desc_f16bf16_f32(m = 256, n = 256, ab_dtype = :bf16) ===
          UInt32((16 << 24) | (32 << 17) | 0x490)
    @test tcgen05_instr_desc_f16bf16_f32(m = 128, n = 8,   ab_dtype = :bf16) ===
          UInt32((8  << 24) | (1  << 17) | 0x490)
    @test tcgen05_instr_desc_f16bf16_f32(m = 128, n = 128, ab_dtype = :bf16) ===
          UInt32((8  << 24) | (16 << 17) | 0x490)

    # sparse / saturate / max_shift bits.
    @test tcgen05_instr_desc_f16bf16_f32(m = 128, n = 256, ab_dtype = :bf16,
                                         sparse = true) ===
          UInt32(base | (UInt32(1) << 2))
    @test tcgen05_instr_desc_f16bf16_f32(m = 128, n = 256, ab_dtype = :bf16,
                                         saturate = true) ===
          UInt32(base | (UInt32(1) << 3))
    @test tcgen05_instr_desc_f16bf16_f32(m = 128, n = 256, ab_dtype = :bf16,
                                         max_shift = 3) ===
          UInt32(base | (UInt32(3) << 30))

    # Range checks.
    @test_throws ArgumentError tcgen05_instr_desc_f16bf16_f32(
        m = 96, n = 256, ab_dtype = :bf16)              # m must be 64/128/256
    @test_throws ArgumentError tcgen05_instr_desc_f16bf16_f32(
        m = 128, n = 12, ab_dtype = :bf16)              # n must be multiple of 8
    @test_throws ArgumentError tcgen05_instr_desc_f16bf16_f32(
        m = 128, n = 264, ab_dtype = :bf16)             # n must be ≤ 256
    @test_throws ArgumentError tcgen05_instr_desc_f16bf16_f32(
        m = 128, n = 256, ab_dtype = :f64)              # bad dtype
    @test_throws ArgumentError tcgen05_instr_desc_f16bf16_f32(
        m = 128, n = 256, ab_dtype = :bf16, a_major = :T)  # bad major
    @test_throws ArgumentError tcgen05_instr_desc_f16bf16_f32(
        m = 128, n = 256, ab_dtype = :bf16, scale_a = 2)   # bad scale
    @test_throws ArgumentError tcgen05_instr_desc_f16bf16_f32(
        m = 128, n = 256, ab_dtype = :bf16, max_shift = 4) # bad max_shift
end

@testset "tcgen05_descriptor packing" begin
    using PTX: tcgen05_descriptor, BlackwellLayout

    # Field-isolation tests pass version=0 to suppress the default-1 bit
    # (pyptx parity: descriptor()'s version defaults to 1).

    # Empty descriptor: all zeros.
    @test tcgen05_descriptor(UInt32(0); leading_bytes = 0,
                                         stride_bytes = 0,
                                         version = 0) === UInt64(0)

    # Address: 14-bit field at bits [13:0], encoded as (addr & 0x3FFF0) >> 4.
    # 0x100 → 0x10. Sits in low 14 bits.
    @test tcgen05_descriptor(UInt32(0x100); leading_bytes = 0,
                                            stride_bytes = 0,
                                            version = 0) ===
          UInt64(0x10)
    # 0x40000 (bit 18) is outside the 14-bit window → 0.
    @test tcgen05_descriptor(UInt32(0x40000); leading_bytes = 0,
                                              stride_bytes = 0,
                                              version = 0) ===
          UInt64(0)

    # Leading / stride byte fields.
    @test tcgen05_descriptor(UInt32(0); leading_bytes = 16,
                                         stride_bytes = 0,
                                         version = 0) ===
          UInt64(1) << 16
    @test tcgen05_descriptor(UInt32(0); leading_bytes = 0,
                                         stride_bytes = 1024,
                                         version = 0) ===
          UInt64(64) << 32

    # Layout-type at bit 61 (3 bits — distinct from wgmma's 2-bit field).
    @test tcgen05_descriptor(UInt32(0); leading_bytes = 0,
                                         stride_bytes = 0,
                                         version = 0,
                                         swizzle = BlackwellLayout.B128) ===
          UInt64(BlackwellLayout.B128) << 61
    @test tcgen05_descriptor(UInt32(0); leading_bytes = 0,
                                         stride_bytes = 0,
                                         version = 0,
                                         swizzle = BlackwellLayout.B64) ===
          UInt64(BlackwellLayout.B64)  << 61

    # version (default 1) → bit 46. Pass version=0 to clear it.
    @test tcgen05_descriptor(UInt32(0); leading_bytes = 0, stride_bytes = 0,
                                         version = 0) === UInt64(0)
    @test tcgen05_descriptor(UInt32(0); leading_bytes = 0, stride_bytes = 0,
                                         version = 1) === UInt64(1) << 46
    @test tcgen05_descriptor(UInt32(0); leading_bytes = 0, stride_bytes = 0,
                                         version = 3) === UInt64(3) << 46

    # base_offset / lbo_mode.
    @test tcgen05_descriptor(UInt32(0); leading_bytes = 0, stride_bytes = 0,
                                         version = 0, base_offset = 5) ===
          UInt64(5) << 49
    @test tcgen05_descriptor(UInt32(0); leading_bytes = 0, stride_bytes = 0,
                                         version = 0, lbo_mode = 1) ===
          UInt64(1) << 52

    # The pyptx-equivalent BLACKWELL_MASKED_DESC_B128 constant:
    # leading=16, stride=1024, swizzle=B128, version=1.
    @test tcgen05_descriptor(UInt32(0); leading_bytes = 16,
                                         stride_bytes = 1024,
                                         swizzle = BlackwellLayout.B128) ===
          UInt64(0x4000404000010000)

    # Range checks.
    @test_throws ArgumentError tcgen05_descriptor(
        UInt32(0); leading_bytes = 8, stride_bytes = 0)         # not 16-aligned
    @test_throws ArgumentError tcgen05_descriptor(
        UInt32(0); leading_bytes = 0, stride_bytes = 24)        # not 16-aligned
    @test_throws ArgumentError tcgen05_descriptor(
        UInt32(0); leading_bytes = 0, stride_bytes = 0, version = 4)
    @test_throws ArgumentError tcgen05_descriptor(
        UInt32(0); leading_bytes = 0, stride_bytes = 0, base_offset = 8)
    @test_throws ArgumentError tcgen05_descriptor(
        UInt32(0); leading_bytes = 0, stride_bytes = 0, lbo_mode = 2)

    # BlackwellLayout constants match PTX encoding (3-bit, non-consecutive).
    @test BlackwellLayout.NONE         == 0
    @test BlackwellLayout.B128_BASE32B == 1
    @test BlackwellLayout.B128         == 2
    @test BlackwellLayout.B64          == 4
    @test BlackwellLayout.B32          == 6
end

@testset "tma (cp.async.bulk.tensor) wrapper (tier-2 intrinsic lowering)" begin
    # Third migrated family. Cluster-destination loads lower through
    # llvm.nvvm.cp.async.bulk.tensor.g2s.tile.<N>d — multicast and
    # cta_group are immarg-selected qualifiers on the same intrinsic;
    # shared::cta loads through g2s.cta.tile.<N>d (PTX 8.6 ISel floor);
    # stores through s2g.tile.<N>d. The wrapper retypes dst → addrspace(7)
    # and tmap → generic raw (reinterpret_addrspace — never addrspacecast,
    # see address_space.jl). Goldens: test/golden/tma@sm{90,100a}.ptx.
    # shared::cta × cta_group::2 stays asm-tier (no intrinsic carries both).

    pSh = Core.LLVMPtr{Float32, PTX.AS.Shared}
    pTm = Core.LLVMPtr{UInt8, PTX.AS.Const}
    pMb = Core.LLVMPtr{UInt64, PTX.AS.Shared}
    cluster = Symbol("shared::cluster")
    cta     = Symbol("shared::cta")
    cg2     = Symbol("cta_group::2")
    mc      = Symbol("multicast::cluster")
    cmpl    = Symbol("mbarrier::complete_tx::bytes")

    for n in 1:5
        nd = Symbol("$(n)d")
        coords = ntuple(_ -> Int32, n)

        # the intrinsics the wrappers stand on: registered and convergent
        for name in ("llvm.nvvm.cp.async.bulk.tensor.g2s.tile.$(n)d",
                     "llvm.nvvm.cp.async.bulk.tensor.g2s.cta.tile.$(n)d",
                     "llvm.nvvm.cp.async.bulk.tensor.s2g.tile.$(n)d")
            @test PTX.NVVM.isintrinsic(name)
            @test :convergent in PTX.NVVM.intrinsic(name).props
        end

        # cluster loads (plain / multicast / cta_group::2 / both) all route
        # to g2s.tile.<N>d, differing only in immargs and the mask operand
        for (mods, argtypes) in (
                ((:async, :bulk, :tensor, nd, cluster, :global, :tile, cmpl),
                 (pSh, pTm, coords..., pMb)),
                ((:async, :bulk, :tensor, nd, cluster, :global, :tile, cmpl, mc),
                 (pSh, pTm, coords..., pMb, UInt16)),
                ((:async, :bulk, :tensor, nd, cg2, cluster, :global, :tile, cmpl),
                 (pSh, pTm, coords..., pMb)),
                ((:async, :bulk, :tensor, nd, cg2, cluster, :global, :tile, cmpl, mc),
                 (pSh, pTm, coords..., pMb, UInt16)))
            @test which(Operation{:cp, mods}(), argtypes).module == PTX
            ci, rt = first(Base.code_typed(Operation{:cp, mods}(), argtypes))
            @test rt === Nothing
            @test occursin("g2s.tile.$(n)d", string(ci))
        end

        # shared::cta load → g2s.cta.tile.<N>d
        mods = (:async, :bulk, :tensor, nd, cta, :global, :tile, cmpl)
        @test which(Operation{:cp, mods}(),
                    (pSh, pTm, coords..., pMb)).module == PTX
        ci, rt = first(Base.code_typed(Operation{:cp, mods}(),
                                       (pSh, pTm, coords..., pMb)))
        @test rt === Nothing
        @test occursin("g2s.cta.tile.$(n)d", string(ci))

        # store → s2g.tile.<N>d
        mods = (:async, :bulk, :tensor, nd, :global, cta, :tile, :bulk_group)
        @test which(Operation{:cp, mods}(),
                    (pTm, coords..., pSh)).module == PTX
        ci, rt = first(Base.code_typed(Operation{:cp, mods}(),
                                       (pTm, coords..., pSh)))
        @test rt === Nothing
        @test occursin("s2g.tile.$(n)d", string(ci))

        # residue: shared::cta × cta_group::2 stays asm-tier (match
        # truncated before the operand brackets — CodeInfo escapes `$`)
        mods = (:async, :bulk, :tensor, nd, cg2, cta, :global, :tile, cmpl)
        ci, _ = first(Base.code_typed(Operation{:cp, mods}(),
                                      (pSh, pTm, coords..., pMb)))
        @test occursin("cp.async.bulk.tensor.$(n)d.cta_group::2.shared::cta" *
                       ".global.tile.mbarrier::complete_tx::bytes [", string(ci))
    end
end

@testset "proxy/init fences (tier-2 intrinsic lowering)" begin
    # The three proxy/init fences route to llvm.nvvm.fence.* intrinsics
    # (a core-IR `fence` can't express a proxy fence). Generic memory
    # fences stay asm-tier — see wrappers/fence.jl.
    for (mods, intr) in (
            ((:proxy, :async), "llvm.nvvm.fence.proxy.async"),
            ((:proxy, :async, Symbol("shared::cta")),
             "llvm.nvvm.fence.proxy.async.shared_cta"),
            ((:mbarrier_init, :release, :cluster),
             "llvm.nvvm.fence.mbarrier_init.release.cluster"))
        @test PTX.NVVM.isintrinsic(intr)
        @test which(Operation{:fence, mods}(), ()).module == PTX
        ci, rt = first(Base.code_typed(Operation{:fence, mods}(), ()))
        @test rt === Nothing
        @test occursin(intr, string(ci))
    end
end

@testset "tcgen05 wrapper (tier-2 intrinsic lowering)" begin
    # Fourth migrated family — see wrappers/tcgen05.jl for the mapping.
    # The notation surface keeps raw UInt32 taddr/SMEM-offset operands;
    # bodies must route to the intrinsic literals. mx mma kinds stay
    # asm-tier (block-scale operands are not in the notation surface).
    cg1 = Symbol("cta_group::1")
    cg2 = Symbol("cta_group::2")

    for (mods, argts, intr) in (
            ((:shift, cg1, :down), (UInt32,), "tcgen05.shift.down.cg1"),
            ((:dealloc, cg2, :sync, :aligned, :b32), (UInt32, UInt32),
             "tcgen05.dealloc.cg2"),
            ((:cp, cg1, Symbol("128x128b")), (UInt32, UInt64),
             "tcgen05.cp.128x128b.cg1"),
            ((:alloc, cg1, :sync, :aligned, Symbol("shared::cta"), :b32),
             (UInt32, UInt32), "tcgen05.alloc.shared.cg1"),
            ((:relinquish_alloc_permit, cg1, :sync, :aligned), (),
             "tcgen05.relinq.alloc.permit.cg1"),
            ((Symbol("wait::ld"), :sync, :aligned), (), "tcgen05.wait.ld"),
            ((:commit, cg1, Symbol("mbarrier::arrive::one"),
              Symbol("shared::cta"), :b64), (UInt32,),
             "tcgen05.commit.shared.cg1"),
            ((:commit, cg1, Symbol("mbarrier::arrive::one"),
              Symbol("multicast::cluster"), Symbol("shared::cluster"), :b64),
             (UInt32, UInt16), "tcgen05.commit.mc.shared.cg1"))
        @test which(Operation{:tcgen05, mods}(), argts).module == PTX
        ci, rt = first(Base.code_typed(Operation{:tcgen05, mods}(), argts))
        @test rt === Nothing
        @test occursin(intr, string(ci))
    end

    # ld/st: the full Table-49 grid — arity, inferred return, intrinsic
    for (shape, base) in (("16x64b", 1), ("32x32b", 1),
                          ("16x128b", 2), ("16x256b", 4)),
        c in (1, 2, 4, 8, 16, 32, 64, 128)

        n = base * c
        n > 128 && continue
        sh = Symbol(shape)
        cnt = Symbol("x$c")
        ld = Operation{:tcgen05, (:ld, :sync, :aligned, sh, cnt, :b32)}()
        ci, rt = first(Base.code_typed(ld, (UInt32,)))
        @test rt === (n == 1 ? UInt32 : NTuple{n, UInt32})
        @test occursin("tcgen05.ld.$shape.x$c", string(ci))
        st = Operation{:tcgen05, (:st, :sync, :aligned, sh, cnt, :b32)}()
        ci2, rt2 = first(Base.code_typed(st, (UInt32, NTuple{n, UInt32})))
        @test rt2 === Nothing
        @test occursin("tcgen05.st.$shape.x$c", string(ci2))
    end

    # dense mma kinds route to mma.shared (immarg-selected kind/cta_group)
    for kind in ("f16", "tf32", "f8f6f4", "i8"), cg in (cg1, cg2)
        mods = (:mma, cg, Symbol("kind::$kind"))
        ci, rt = first(Base.code_typed(Operation{:tcgen05, mods}(),
                                       (UInt32, UInt64, UInt64, UInt32, Bool)))
        @test rt === Nothing
        @test occursin("tcgen05.mma.shared", string(ci))
    end

    # residue: mx kinds stay asm-tier (match truncated at the bracket)
    for kind in ("mxf8f6f4", "mxf4", "mxf4nvf4")
        mods = (:mma, cg1, Symbol("kind::$kind"))
        ci, _ = first(Base.code_typed(Operation{:tcgen05, mods}(),
                                      (UInt32, UInt64, UInt64, UInt32, Bool)))
        @test occursin("tcgen05.mma.cta_group::1.kind::$kind [", string(ci))
    end
end

@testset "tuple args → braced operand groups" begin
    # Homogeneous tuple types render as `{$N, $N+1, ...}` and contribute one
    # LLVM input slot per lane. Used by ldmatrix / mma / .rs cvt forms where
    # PTX requires a register-vector operand. Hand-written wrappers like
    # `wrappers/stmatrix.jl` build this manually; the chain default now does
    # too via `render_arg(::Type{<:Tuple})`.

    # Hypothetical fma form taking a 4-lane tuple of f32s as the first
    # operand. Not a real PTX op — we just want to exercise the braced
    # rendering. Real ops with this shape (mma.sync, ldmatrix, etc.) have
    # hand-written wrappers; this confirms the chain default also works.
    spec = build_call(:fakeop, (:f32,), (NTuple{4, Float32},))
    @test spec.asm == "fakeop.f32 \$0, {\$1, \$2, \$3, \$4};"
    @test spec.rettype === Float32
    @test spec.constraints == "=f,f,f,f,f"
    @test spec.passthrough_argtypes === (Float32, Float32, Float32, Float32)
    @test spec.passthrough_indices === ((1, 1), (1, 2), (1, 3), (1, 4))

    # Mix scalar and tuple args — slot numbering interleaves correctly.
    spec = build_call(:fakeop, (:f32,), (Float32, NTuple{2, UInt32}, Float32))
    @test spec.asm == "fakeop.f32 \$0, \$1, {\$2, \$3}, \$4;"
    @test spec.constraints == "=f,f,r,r,f"
    @test spec.passthrough_indices ===
          ((1, nothing), (2, 1), (2, 2), (3, nothing))

    # NTuple{1, T} is still tuple-shaped — emits `{$N}` (single-element
    # brace group). Real PTX accepts this; e.g. `stmatrix.x1` uses
    # `{$1}` not `$1` per the spec.
    spec = build_call(:fakeop, (), (NTuple{1, UInt32},))
    @test spec.asm == "fakeop {\$0};"
    @test spec.constraints == "r"
    @test spec.passthrough_indices === ((1, 1),)

    # Heterogeneous tuple is rejected — there's no single constraint
    # letter that works.
    @test_throws ErrorException build_call(:fakeop, (),
                                            (Tuple{Float32, Int32},))

    # Empty tuple is rejected — no operand mapping.
    @test_throws ErrorException build_call(:fakeop, (), (Tuple{},))
end

@testset "setp dual-pred hand-written wrapper" begin
    # `setp.<cmp>.<dt> $0|$1, $2, $3;` produces (Bool, Bool). The `$0|$1`
    # pipe form is distinct from single-pred (which flows through the chain).

    spec = PTX.setp_dual_spec(:eq, :s32, Int32)
    @test spec.asm == "setp.eq.s32 \$0|\$1, \$2, \$3;"
    @test spec.constraints == "=b,=b,r,r"
    @test spec.rettype === Tuple{Bool, Bool}

    # Float compare — `f` constraint letter on inputs.
    spec = PTX.setp_dual_spec(:lt, :f32, Float32)
    @test spec.asm == "setp.lt.f32 \$0|\$1, \$2, \$3;"
    @test spec.constraints == "=b,=b,f,f"

    # 16-bit dtype — `h` constraint.
    spec = PTX.setp_dual_spec(:ne, :u16, UInt16)
    @test spec.constraints == "=b,=b,h,h"

    # 64-bit float — `d` constraint.
    spec = PTX.setp_dual_spec(:ge, :f64, Float64)
    @test spec.constraints == "=b,=b,d,d"

    # Methods registered on chain singletons for representative variants.
    @test which(Operation{:setp, (:dual, :eq, :s32)}(),
                (Int32, Int32)).module == PTX
    @test which(Operation{:setp, (:dual, :lt, :f32)}(),
                (Float32, Float32)).module == PTX
    @test which(Operation{:setp, (:dual, :ge, :u64)}(),
                (UInt64, UInt64)).module == PTX
end

@testset "cp.async data hand-written wrapper" begin
    # cp.async.ca.shared.global accepts size 4/8/16; cp.async.cg requires 16.
    # Shared destination uses `r` (32-bit) constraint, not `l` — see comment
    # in src/wrappers/cp_async.jl.

    spec = PTX.cp_async_spec(:ca, 16)
    @test spec.asm == "cp.async.ca.shared.global [\$0], [\$1], 16;"
    @test spec.constraints == "r,l,~{memory}"
    @test spec.rettype === Nothing

    spec = PTX.cp_async_spec(:ca, 8)
    @test spec.asm == "cp.async.ca.shared.global [\$0], [\$1], 8;"

    spec = PTX.cp_async_spec(:ca, 4)
    @test spec.asm == "cp.async.ca.shared.global [\$0], [\$1], 4;"

    spec = PTX.cp_async_spec(:cg, 16)
    @test spec.asm == "cp.async.cg.shared.global [\$0], [\$1], 16;"
    @test spec.constraints == "r,l,~{memory}"

    # Size validation — :ca rejects anything outside {4,8,16}, :cg requires 16.
    @test_throws ErrorException PTX.cp_async_spec(:ca, 0)
    @test_throws ErrorException PTX.cp_async_spec(:ca, 7)
    @test_throws ErrorException PTX.cp_async_spec(:ca, 32)
    @test_throws ErrorException PTX.cp_async_spec(:cg, 4)
    @test_throws ErrorException PTX.cp_async_spec(:cg, 8)
    @test_throws ErrorException PTX.cp_async_spec(:cg, 32)

    # Bad qualifier.
    @test_throws ErrorException PTX.cp_async_spec(:bogus, 16)

    # Methods registered on chain singletons.
    @test which(Operation{:cp, (:async, :ca, :shared, :global)}(),
                (Core.LLVMPtr{Float32, PTX.AS.Shared},
                 Core.LLVMPtr{Float32, PTX.AS.Global},
                 Val{16})).module == PTX
    @test which(Operation{:cp, (:async, :cg, :shared, :global)}(),
                (Core.LLVMPtr{Float32, PTX.AS.Shared},
                 Core.LLVMPtr{Float32, PTX.AS.Global},
                 Val{16})).module == PTX

    # @generated catch path — when `cp_async_spec(:ca/:cg, N)` throws (bad N),
    # the wrapper's try/catch (src/wrappers/cp_async.jl:24-28, 44-48) re-emits
    # `:(error(...))` as the function body. Constructing pointers via bitcast
    # so we can invoke from host without a kernel.
    let
        op_ca = Operation{:cp, (:async, :ca, :shared, :global)}()
        op_cg = Operation{:cp, (:async, :cg, :shared, :global)}()
        p_s = Base.bitcast(Core.LLVMPtr{Float32, PTX.AS.Shared}, UInt(0))
        p_g = Base.bitcast(Core.LLVMPtr{Float32, PTX.AS.Global}, UInt(0))

        # ca rejects N=7 (not in {4,8,16}) — message bubbles up via showerror.
        @test_throws ErrorException op_ca(p_s, p_g, Val(7))
        @test_throws ErrorException op_ca(p_s, p_g, Val(0))
        @test_throws ErrorException op_ca(p_s, p_g, Val(32))

        # cg rejects N≠16.
        @test_throws ErrorException op_cg(p_s, p_g, Val(4))
        @test_throws ErrorException op_cg(p_s, p_g, Val(8))
    end

    # @generated body expansion via `code_typed` — triggers the `try cp_async_spec
    # catch e ... end` branch on the success side, hitting the body lines that
    # build the @asmcall quote. Each registered (qual, N) pair lands the
    # size-baked opcode + constraint string in the lowered IR.
    pS = Core.LLVMPtr{Float32, PTX.AS.Shared}
    pG = Core.LLVMPtr{Float32, PTX.AS.Global}
    for (qual, n) in [(:ca, 4), (:ca, 8), (:ca, 16), (:cg, 16)]
        op = Operation{:cp, (:async, qual, :shared, :global)}()
        ci, _ = first(Base.code_typed(op, (pS, pG, Val{n})))
        s = string(ci)
        # Size baked into the asm tail.
        @test occursin("cp.async.$qual.shared.global", s)
        @test occursin(", $n;",                        s)
        @test occursin("r,l,~{memory}",                s)
    end
end

@testset "shfl.sync wrapper (tier-2 intrinsic lowering)" begin
    # shfl.sync.<mode>.b32 d, a, b, c, mask;        — UInt32 output
    # shfl.sync.<mode>.b32 d|p, a, b, c, mask;      — (UInt32, Bool) output
    # First migrated family (DESIGN.md): the notation surface is unchanged
    # but lowering goes through llvm.nvvm.shfl.sync.<mode>.i32[p] — the
    # registry supplies convergent, the .i32p aggregate replaces the
    # pipe-operand asm, and the wrapper reorders (a, b, c, mask) to the
    # intrinsic's mask-first convention. Golden: test/golden/shfl@sm75.ptx.

    for mode in (:up, :down, :bfly, :idx)
        # the intrinsics the wrapper stands on are in the backend table,
        # marked convergent
        for name in ("llvm.nvvm.shfl.sync.$mode.i32",
                     "llvm.nvvm.shfl.sync.$mode.i32p")
            @test PTX.NVVM.isintrinsic(name)
            @test :convergent in PTX.NVVM.intrinsic(name).props
        end

        # methods registered for both forms; bodies route to the intrinsic
        # callable rather than @asmcall
        m = which(Operation{:shfl, (:sync, mode, :b32)}(),
                  (UInt32, UInt32, UInt32, UInt32))
        @test m.module == PTX
        ci, rt = first(Base.code_typed(Operation{:shfl, (:sync, mode, :b32)}(),
                                       (UInt32, UInt32, UInt32, UInt32)))
        @test rt === UInt32
        @test occursin("shfl.sync.$mode.i32", string(ci))

        mp = which(Operation{:shfl, (:sync, mode, :b32, :pred)}(),
                   (UInt32, UInt32, UInt32, UInt32))
        @test mp.module == PTX
        ci, rt = first(Base.code_typed(Operation{:shfl, (:sync, mode, :b32, :pred)}(),
                                       (UInt32, UInt32, UInt32, UInt32)))
        @test rt === Tuple{UInt32, Bool}
        @test occursin("shfl.sync.$mode.i32p", string(ci))
    end
end

@testset "vec ld/st.global tier-1 wrapper" begin
    # ld.global.v{2,4}.{f32,b32,b16} / st.global.v{2,4}.{f32,b32,b16} lower to
    # core LLVM IR (`load`/`store <N x T>`), not asm — no `~{memory}` barrier,
    # so the backend can fold offsets / reorder / CSE them. The surface is
    # `NTuple{N, T}`, so the body repacks between the LLVM vector type
    # `<N x T>` (what load/store want) and the array type `[N x T]` (how Julia
    # represents the homogeneous tuple).

    # v4.f32 load: load <4 x float>, then unpack into the [4 x float] tuple.
    ir = PTX.vec_ld_ir(4, :f32, 16)
    @test occursin("load <4 x float>, ptr addrspace(1) %0, align 16", ir)
    @test occursin("extractelement <4 x float> %v, i32 3", ir)
    @test occursin("insertvalue [4 x float]", ir)
    @test occursin("ret [4 x float]", ir)
    @test !occursin("~{memory}", ir)   # no optimization barrier

    # v2.b32 → i32 elements, align 8.
    ir = PTX.vec_ld_ir(2, :b32, 8)
    @test occursin("load <2 x i32>, ptr addrspace(1) %0, align 8", ir)

    # v4.b16 → i16 elements, align 8.
    ir = PTX.vec_ld_ir(4, :b16, 8)
    @test occursin("load <4 x i16>, ptr addrspace(1) %0, align 8", ir)

    # v4.f32 store: unpack the [4 x float] arg (%1), build <4 x float>, store
    # to the pointer arg (%0).
    ir = PTX.vec_st_ir(4, :f32, 16)
    @test occursin("extractvalue [4 x float] %1, 3", ir)
    @test occursin("insertelement <4 x float>", ir)
    @test occursin("store <4 x float> %v3, ptr addrspace(1) %0, align 16", ir)
    @test occursin("ret void", ir)

    # v2.b16 store, align 4.
    ir = PTX.vec_st_ir(2, :b16, 4)
    @test occursin("store <2 x i16> %v1, ptr addrspace(1) %0, align 4", ir)

    # Methods registered for representative variants.
    @test which(Operation{:ld, (:global, :v4, :f32)}(),
                (Core.LLVMPtr{Float32, PTX.AS.Global},)).module == PTX
    @test which(Operation{:st, (:global, :v2, :b16)}(),
                (Core.LLVMPtr{UInt16, PTX.AS.Global},
                 NTuple{2, UInt16})).module == PTX
    @test which(Operation{:ld, (:global, :v2, :b32)}(),
                (Core.LLVMPtr{UInt32, PTX.AS.Global},)).module == PTX
    @test which(Operation{:st, (:global, :v4, :f32)}(),
                (Core.LLVMPtr{Float32, PTX.AS.Global},
                 NTuple{4, Float32})).module == PTX

    # vec ld is `@eval @generated function ...`, so `code_typed` triggers body
    # expansion at host. Every variant bakes its alignment as N*sizeof(T).
    for (n, dt, T) in PTX._VEC_LDST_VARIANTS
        op = Operation{:ld, (:global, Symbol("v", n), dt)}()
        ci, _ = first(Base.code_typed(op,
            (Core.LLVMPtr{T, PTX.AS.Global},)))
        s = string(ci)
        @test occursin("load <$n x", s)
        @test occursin("align $(n * sizeof(T))", s)
    end
end

@testset "wgmma.mma_async dispatch surface (registered vs unregistered)" begin
    # The wgmma generator registers methods only for the cross-product
    # `_WGMMA_VARIANTS × _WGMMA_NS` (PTX 9.2 §9.7.14.5). For unsupported
    # combinations — e.g. bf16/bf16 with k=8 (only valid for tf32), N=7, or
    # N=264 — no specific method exists on `Operation{parts}`. The `Vararg`
    # chain default in `inst.jl` then catches the call and would emit a
    # garbage asm string (wrong rettype, wrong operand count) that crashes
    # in LLVM register allocation. Tests here lock in the registered surface
    # so a regression that drops a (n,k,dtype) combo from the table is
    # caught at host time, not via a confusing LLVM error at code-gen.

    # Registered set: for each valid (dtype_d, dtype_a, dtype_b, k) from
    # `_WGMMA_VARIANTS`, a method exists for every N in `_WGMMA_NS`.
    registered_variants = (
        # f32 acc → NTuple{N/2, Float32}
        (:f32, :bf16, :bf16, 16),
        (:f32, :f16,  :f16,  16),
        (:f32, :tf32, :tf32, 8),
        (:f32, :e4m3, :e4m3, 32),
        (:f32, :e5m2, :e5m2, 32),
        # f16 acc → NTuple{N/4, UInt32}
        (:f16, :f16,  :f16,  16),
        (:f16, :e4m3, :e4m3, 32),
        (:f16, :e5m2, :e5m2, 32),
        # s32 acc → NTuple{N/2, Int32}
        (:s32, :s8, :s8, 32),
        (:s32, :u8, :u8, 32),
        (:s32, :s8, :u8, 32),
        (:s32, :u8, :s8, 32),
    )

    for (dt_d, dt_a, dt_b, k) in registered_variants
        for n in (8, 64, 128, 256)               # spot-check across the N grid
            shape = Symbol("m64n", n, "k", k)
            mods = (:mma_async, :sync, :aligned, shape, dt_d, dt_a, dt_b)
            d_J = dt_d === :f32 ? Float32 : (dt_d === :f16 ? UInt32 : Int32)
            nd  = dt_d === :f16 ? n >>> 2 : n >>> 1
            d_T = NTuple{nd, d_J}
            @test which(Operation{:wgmma, mods}(),
                        (d_T, UInt64, UInt64, Bool)).module == PTX
        end
    end

    # bf16 with k=8 has no registration (k=8 is tf32-only in the table).
    mods_bad_k = (:mma_async, :sync, :aligned,
                  :m64n8k8, :f32, :bf16, :bf16)
    bad_k_method = which(Operation{:wgmma, mods_bad_k}(),
                         (NTuple{4, Float32}, UInt64, UInt64, Bool))
    # Falls through to the chain default in inst.jl, NOT a wgmma-specific
    # method. The chain default's source file is inst.jl (the @generated
    # `(::Operation{op, mods})(args::Vararg{Any,N})`).
    @test endswith(string(bad_k_method.file), "inst.jl")

    # N=7 (not step-by-8) and N=264 (over cap) — same chain-default fallthrough.
    mods_bad_n = (:mma_async, :sync, :aligned,
                  :m64n7k16, :f32, :bf16, :bf16)
    @test endswith(string(which(Operation{:wgmma, mods_bad_n}(),
                                 (NTuple{4, Float32}, UInt64, UInt64, Bool)).file),
                   "inst.jl")
end

@testset "cvt convention (chain-driven)" begin
    # cvt's grammar is `cvt.<modifiers...>.<dst>.<src>`, so dst is parts[end-1]
    # and src is parts[end]. The chain default infers rettype from dst, builds
    # constraints from runtime argtypes — no per-(dst,src) registration.

    # Scalar widening: f32 ← f16 (no rmode permitted on widening per spec).
    @test format_call(ptx"cvt.f32.f16", Tuple{Float16}) == "cvt.f32.f16 \$0, \$1;"
    spec = build_call(:cvt, (:f32, :f16), (Float16,))
    @test spec.rettype === Float32
    @test spec.constraints == "=f,h"
    @test spec.side_effects == false

    # Scalar narrowing: f16 ← f32 with rn rounding.
    @test format_call(ptx"cvt.rn.f16.f32", Tuple{Float32}) == "cvt.rn.f16.f32 \$0, \$1;"
    spec = build_call(:cvt, (:rn, :f16, :f32), (Float32,))
    @test spec.rettype === Float16
    @test spec.constraints == "=h,f"

    # Packed pack: e4m3x2 ← (f32, f32). Two inputs, UInt16 output (i16 carrier).
    @test format_call(ptx"cvt.rn.satfinite.e4m3x2.f32", Tuple{Float32, Float32}) ==
          "cvt.rn.satfinite.e4m3x2.f32 \$0, \$1, \$2;"
    spec = build_call(:cvt, (:rn, :satfinite, :e4m3x2, :f32), (Float32, Float32))
    @test spec.rettype === UInt16
    @test spec.constraints == "=h,f,f"

    # Packed pack: f16x2 ← (f32, f32). Output is UInt32 (i32 carrier).
    spec = build_call(:cvt, (:rn, :f16x2, :f32), (Float32, Float32))
    @test spec.rettype === UInt32
    @test spec.constraints == "=r,f,f"

    # Packed unpack: f16x2 ← e4m3x2. UInt16 → UInt32. Single input, single output.
    @test format_call(ptx"cvt.rn.f16x2.e4m3x2", Tuple{UInt16}) ==
          "cvt.rn.f16x2.e4m3x2 \$0, \$1;"
    spec = build_call(:cvt, (:rn, :f16x2, :e4m3x2), (UInt16,))
    @test spec.rettype === UInt32
    @test spec.constraints == "=r,h"

    # Packed unpack: bf16x2 ← e5m2x2.
    spec = build_call(:cvt, (:rn, :bf16x2, :e5m2x2), (UInt16,))
    @test spec.rettype === UInt32
    @test spec.constraints == "=r,h"

    # Direct FP16x2 → FP8x2 down-cast (no f32 detour).
    @test format_call(ptx"cvt.rn.satfinite.e4m3x2.f16x2", Tuple{UInt32}) ==
          "cvt.rn.satfinite.e4m3x2.f16x2 \$0, \$1;"
    spec = build_call(:cvt, (:rn, :satfinite, :e4m3x2, :f16x2), (UInt32,))
    @test spec.rettype === UInt16
    @test spec.constraints == "=h,r"

    # `.relu` modifier composes for free — no registration needed.
    @test format_call(ptx"cvt.rn.relu.f16.f32", Tuple{Float32}) ==
          "cvt.rn.relu.f16.f32 \$0, \$1;"
    spec = build_call(:cvt, (:rn, :relu, :f16, :f32), (Float32,))
    @test spec.rettype === Float16
    @test spec.constraints == "=h,f"

    # `.ftz` modifier composes for free.
    @test format_call(ptx"cvt.rzi.ftz.s32.f32", Tuple{Float32}) ==
          "cvt.rzi.ftz.s32.f32 \$0, \$1;"
    spec = build_call(:cvt, (:rzi, :ftz, :s32, :f32), (Float32,))
    @test spec.rettype === Int32
    @test spec.constraints == "=r,f"

    # Integer cvt: u64 ← u32 (widening). Previously broken — chain default
    # used last(parts) = :u32 and returned UInt32 with =r,r constraints.
    @test format_call(ptx"cvt.u64.u32", Tuple{UInt32}) == "cvt.u64.u32 \$0, \$1;"
    spec = build_call(:cvt, (:u64, :u32), (UInt32,))
    @test spec.rettype === UInt64
    @test spec.constraints == "=l,r"

    # tf32: i32 carrier, satfinite optional per spec.
    @test format_call(ptx"cvt.rn.satfinite.tf32.f32", Tuple{Float32}) ==
          "cvt.rn.satfinite.tf32.f32 \$0, \$1;"
    spec = build_call(:cvt, (:rn, :satfinite, :tf32, :f32), (Float32,))
    @test spec.rettype === UInt32
    @test spec.constraints == "=r,f"

    # bf16: i16 carrier.
    spec = build_call(:cvt, (:rn, :bf16, :f32), (Float32,))
    @test spec.rettype === UInt16
    @test spec.constraints == "=h,f"

    # `.scaled::n2::ue8m0` modifier — munged through `__` → `::`. Trailing
    # scale-factor operand is just a regular UInt16 input.
    @test format_call(ptx"cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2",
                      Tuple{UInt16, UInt16}) ==
          "cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 \$0, \$1, \$2;"
    spec = build_call(:cvt, (:rn, :scaled__n2__ue8m0, :bf16x2, :e4m3x2),
                      (UInt16, UInt16))
    @test spec.rettype === UInt32
    @test spec.constraints == "=r,h,h"

    # cvt is a pure ALU op — no memory clobber, no side effects.
    spec = build_call(:cvt, (:rn, :f16, :f32), (Float32,))
    @test spec.side_effects == false
    @test !occursin("memory", spec.constraints)

    # cvta is unaffected — uses the standard "rettype = last modifier" rule.
    @test format_call(ptx"cvta.to.global.u32", Tuple{Core.LLVMPtr{Float32, PTX.AS.Global}}) ==
          "cvta.to.global.u32 \$0, \$1;"
    spec = build_call(:cvta, (:to, :global, :u32),
                      (Core.LLVMPtr{Float32, PTX.AS.Global},))
    @test spec.rettype === UInt32
end


# Re-invoke private `_*_register` helpers from host so their bodies show in
# coverage reports. The module-load for-loops register every method during
# precompile, but precompile-time line execution doesn't generate coverage
# data — only runtime does. Each call here re-defines an existing method
# (silenced via `redirect_stderr`) and we follow up with `which()` to assert
# dispatch is intact afterward. These pull `mma.jl`, `mma_scaled.jl`,
# `ldmatrix.jl`, `tcgen05.jl` from 0% host coverage into the meaningful
# range, and lift the partially-covered helpers in stmatrix/tma/vec_ldst/
# wgmma to fully covered.
@testset "register helpers — re-invoke for coverage + dispatch sanity" begin

    # `redirect_stderr(devnull) do ... end` swallows the
    # "Method definition ... overwritten on the same line" notices that
    # `--warn-overwrite=yes` (set by Pkg.test) emits when @eval re-defines
    # an existing method.
    silent(f) = redirect_stderr(devnull) do; f(); end

    # ---- _mma_register (migrated to tier-2; now 5-arg, row.col only) ----
    silent() do
        PTX._mma_register(:m16n8k16, :f32, :bf16, :bf16, :f32)
        PTX._mma_register(:m16n8k16, :f16, :f16, :f16, :f16)
        PTX._mma_register(:m16n8k32, :f32, :e4m3, :e4m3, :f32; kind = :f8f6f4)
        # Asm-tier fallback path: kind::f8f6f4 has no intrinsic at m16n8k16.
        PTX._mma_register(:m16n8k16, :f32, :e4m3, :e4m3, :f32; kind = :f8f6f4)
        # Early-return path: shape/dtype not in MMA_SYNC_FRAGS.
        @test PTX._mma_register(:m99n99k99, :f32, :bf16, :bf16, :f32) === nothing
    end
    @test which(Operation{:mma,
            (:sync, :aligned, :m16n8k16, :row, :col, :f32, :bf16, :bf16, :f32)}(),
        (NTuple{4, UInt32}, NTuple{2, UInt32}, NTuple{4, Float32})).module == PTX
    @test which(Operation{:mma,
            (:sync, :aligned, Symbol("kind::f8f6f4"),
             :m16n8k32, :row, :col, :f32, :e4m3, :e4m3, :f32)}(),
        (NTuple{4, UInt32}, NTuple{2, UInt32}, NTuple{4, Float32})).module == PTX

    # ---- _mma_scaled_register ----
    silent() do
        PTX._mma_scaled_register(:mxf8f6f4, Symbol("1X"),
            :m16n8k32, :row, :col, :f32, :e4m3, :e4m3, :f32, :ue8m0)
        # Early-return path: shape/dtype not in MMA_SCALED_FRAGS.
        @test PTX._mma_scaled_register(:mxf4, Symbol("1X"),
            :m99n99k99, :row, :col, :f32, :e2m1, :e2m1, :f32, :ue8m0) === nothing
    end

    # ---- _ldmatrix_register ----
    silent() do
        PTX._ldmatrix_register(:m8n8, :x4, false, :shared, :b16)
        PTX._ldmatrix_register(:m8n8, :x1, true,  :shared_cta, :b16)
        PTX._ldmatrix_register(:m16n16, :x1, false, :shared, :b8)
    end
    @test which(Operation{:ldmatrix,
            (:sync, :aligned, :m8n8, :x4, :shared, :b16)}(),
        (Core.LLVMPtr{UInt16, PTX.AS.Shared},)).module == PTX

    # ---- _stmatrix_register (also coverage for n_in==1 vs n_in>1 branches) ----
    silent() do
        PTX._stmatrix_register(:m8n8, :x1, false, :shared, :b16)       # n_in=1 path
        PTX._stmatrix_register(:m8n8, :x4, true,  :shared_cta, :b16)   # n_in>1 + trans
    end
    @test which(Operation{:stmatrix,
            (:sync, :aligned, :m8n8, :x1, :shared, :b16)}(),
        (Core.LLVMPtr{UInt16, PTX.AS.Shared}, UInt32)).module == PTX

    # (tma migrated to tier-2 literal methods — no register helper left;
    # dispatch is asserted in its own testset above)

    # ---- _vec_ld_register / _vec_st_register (tier-1; 3-arg) ----
    silent() do
        PTX._vec_ld_register(2, :f32, Float32)
        PTX._vec_ld_register(4, :b32, UInt32)
        PTX._vec_st_register(4, :f32, Float32)
        PTX._vec_st_register(2, :b16, UInt16)
    end

    # ---- _wgmma_mma_async_register ----
    silent() do
        PTX._wgmma_mma_async_register(:f32, :bf16, :bf16, 8,  16, true)
        PTX._wgmma_mma_async_register(:f16, :f16,  :f16,  16, 16, true)
        PTX._wgmma_mma_async_register(:s32, :s8,   :s8,   8,  32, false)
    end

    # (tcgen05 migrated to tier-2 literal methods — no register helpers
    # left; dispatch is asserted in its own testset above)

    # ---- _setp_dual_register ----
    silent() do
        PTX._setp_dual_register(:eq, :s32, Int32)
        PTX._setp_dual_register(:lt, :f32, Float32)
        PTX._setp_dual_register(:ne, :u16, UInt16)
    end
    @test which(Operation{:setp, (:dual, :eq, :s32)}(),
                (Int32, Int32)).module == PTX

    # ---- shfl: no _register helper since the tier-2 migration (methods
    # come from a top-level loop in wrappers/shfl.jl); dispatch sanity only
    @test which(Operation{:shfl, (:sync, :idx, :b32)}(),
                (UInt32, UInt32, UInt32, UInt32)).module == PTX
    @test which(Operation{:shfl, (:sync, :bfly, :b32, :pred)}(),
                (UInt32, UInt32, UInt32, UInt32)).module == PTX
end
