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
    # PTX 9.3 uses WARP_SZ (an immediate), not a %warpsize special register.
    # The public sreg macro normalizes that legacy spelling to Val(32), so it
    # must not leak into either PTX or NVVM-sreg inventories.
    @test !haskey(PTX.NVVM_SREG_U32, Symbol("%warpsize"))
    @test !("%warpsize" in PTX.IR.SPECIAL_REGS)
    @test format_call(ptx"mov.u32", Tuple{typeof(sreg"%warpsize")}) ==
        "mov.u32 \$0, 32;"
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

@testset "bar/barrier wrapper (tier-2 intrinsic lowering)" begin
    # Migrated family (the fence recipe): bar.{sync,arrive}, bar.warp.sync
    # and the barrier.{sync,arrive}{.aligned} spellings lower through
    # llvm.nvvm.barrier.cta.* / llvm.nvvm.bar.warp.sync for Val and
    # UInt32/Int32 operands (golden: test/golden/barrier@sm75.ptx).
    # bar.* rides the .aligned intrinsics — PTX §9.7.12.1 defines
    # bar ≡ barrier.aligned.
    cases = [
        (:bar, (:sync,), (Val{0},),
            "llvm.nvvm.barrier.cta.sync.aligned.all"),
        (:bar, (:sync,), (UInt32,),
            "llvm.nvvm.barrier.cta.sync.aligned.all"),
        (:bar, (:sync,), (Val{1}, Val{128}),
            "llvm.nvvm.barrier.cta.sync.aligned.count"),
        (:bar, (:sync,), (UInt32, UInt32),
            "llvm.nvvm.barrier.cta.sync.aligned.count"),
        (:bar, (:warp, :sync), (UInt32,),
            "llvm.nvvm.bar.warp.sync"),
        (:bar, (:arrive,), (Val{2}, Val{128}),
            "llvm.nvvm.barrier.cta.arrive.aligned.count"),
        (:barrier, (:sync,), (Val{3},),
            "llvm.nvvm.barrier.cta.sync.all"),
        (:barrier, (:sync,), (Val{4}, Val{128}),
            "llvm.nvvm.barrier.cta.sync.count"),
        (:barrier, (:sync, :aligned), (Val{5},),
            "llvm.nvvm.barrier.cta.sync.aligned.all"),
        (:barrier, (:sync, :aligned), (Val{6}, Val{128}),
            "llvm.nvvm.barrier.cta.sync.aligned.count"),
        (:barrier, (:arrive,), (Val{7}, Val{128}),
            "llvm.nvvm.barrier.cta.arrive.count"),
        (:barrier, (:arrive, :aligned), (Val{8}, Val{128}),
            "llvm.nvvm.barrier.cta.arrive.aligned.count"),
    ]
    for (opsym, mods, argts, intr) in cases
        # the intrinsic is registered and convergent
        @test PTX.NVVM.isintrinsic(intr)
        @test :convergent in PTX.NVVM.intrinsic(intr).props
        # the method dispatches and routes to that intrinsic
        op = Operation{opsym, mods}()
        @test which(op, argts).module == PTX
        ci, rt = first(Base.code_typed(op, argts))
        @test rt === Nothing
        @test occursin(intr, string(ci))
    end

    # Wider integers stay on the asm-tier chain fallback unchanged — the
    # frozen transpiler emits `ptx"bar.sync"(0)` with Int literals, and
    # that path keeps its rendering and its convergent nomerge asm.
    ci, _ = first(Base.code_typed(Operation{:bar, (:sync,)}(), (Int64,)))
    s = string(ci)
    @test occursin("bar.sync", s)
    @test occursin("convergent nomerge", s)

    # Renderer + registry contract for the fallback path, unchanged:
    @test format_call(ptx"bar.sync", Tuple{Val{0}})  == "bar.sync 0;"
    @test format_call(ptx"bar.sync", Tuple{Val{15}}) == "bar.sync 15;"
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
    # NVPTX register-class letters per src/ledgers/types.jl. Wrong letters → ptxas
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

    # `mapa` — pure address computation, kept observable (undeletable) but
    # with no memory clobber: the chain now matches the typed wrapper's
    # long-standing sideeffect-without-~{memory} rendering.
    spec = build_call(:mapa, (Symbol("shared::cluster",), :u32), (UInt32, UInt32))
    @test spec.side_effects == true
    @test !occursin("~{memory}", spec.constraints)
    # getctarank deliberately stays at the full-clobber default pending its
    # own review.
    spec = build_call(:getctarank, (Symbol("shared::cluster",), :u32), (UInt32,))
    @test spec.side_effects == true
    @test occursin("~{memory}", spec.constraints)

    # `griddepcontrol.wait` — inter-launch dependency, needs barrier.
    spec = build_call(:griddepcontrol, (:wait,), ())
    @test spec.side_effects == true
end

@testset "clmad chain entry (PTX 9.3)" begin
    # `clmad.{lo,hi}.u64 d, a, b, c;` — pure carryless multiply-add. No
    # hand-written wrapper: last modifier `.u64` flows through DTYPE_RETTYPE,
    # `:clmad` is registered _PURE in src/ledgers/forms.jl (pure ALU, no memory).
    # PTX 9.3 / sm_80+. The chain doesn't enforce arity — ptxas rejects
    # wrong-arity calls.
    @test format_call(ptx"clmad.lo.u64", Tuple{UInt64, UInt64, UInt64}) ==
          "clmad.lo.u64 \$0, \$1, \$2, \$3;"
    @test format_call(ptx"clmad.hi.u64", Tuple{UInt64, UInt64, UInt64}) ==
          "clmad.hi.u64 \$0, \$1, \$2, \$3;"

    spec = build_call(:clmad, (:lo, :u64), (UInt64, UInt64, UInt64))
    @test spec.rettype == UInt64
    @test spec.side_effects == false
    @test !occursin("~{memory}", spec.constraints)
    @test spec.constraints == "=l,l,l,l"
end

@testset "cp.async.bulk family chain entries (.sem/.scope, PTX 9.3)" begin
    # PTX 9.3 added `.sem`/`.scope` to `cp.async.bulk`, `cp.reduce.async.bulk`,
    # and `multimem.cp.{async,reduce}.bulk`. The new forms append a terminal
    # `.type` (.b128 for cp.async.bulk; .b32/.b64/.f16/.u64/etc. for the reduce
    # variants). DTYPE_RETTYPE has entries for most of those, so the chain
    # default would misread the terminal as a return slot without the
    # `(:async, :bulk)` / `(:reduce, :async, :bulk)` / `(:cp,)` _MEMSINK
    # overrides in src/ledgers/forms.jl.

    # `cp.async.bulk` with .sem/.scope + .b128 (sm_100+ global→shared::cta).
    asm = format_call(
        ptx"cp.async.bulk.relaxed.gpu.shared::cta.global.mbarrier::complete_tx::bytes.b128",
        Tuple{Core.LLVMPtr{UInt8, PTX.AS.Shared},
              Core.LLVMPtr{UInt8, PTX.AS.Global},
              UInt32,
              Core.LLVMPtr{UInt64, PTX.AS.Shared}})
    @test asm ==
        "cp.async.bulk.relaxed.gpu.shared::cta.global.mbarrier::complete_tx::bytes.b128 " *
        "[\$0], [\$1], \$2, [\$3];"

    spec = build_call(:cp,
        (:async, :bulk, :relaxed, :gpu, Symbol("shared::cta"), :global,
         Symbol("mbarrier::complete_tx::bytes"), :b128),
        (Core.LLVMPtr{UInt8, PTX.AS.Shared},
         Core.LLVMPtr{UInt8, PTX.AS.Global},
         UInt32,
         Core.LLVMPtr{UInt64, PTX.AS.Shared}))
    @test spec.rettype === Nothing
    @test spec.side_effects == true
    @test occursin("~{memory}", spec.constraints)

    # `cp.reduce.async.bulk` with .sem/.scope + .add.u64 (sm_90+, .sem/.scope new in 9.3).
    asm = format_call(
        ptx"cp.reduce.async.bulk.relaxed.cluster.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u64",
        Tuple{Core.LLVMPtr{UInt64, PTX.AS.Shared},
              Core.LLVMPtr{UInt64, PTX.AS.Shared},
              UInt32,
              Core.LLVMPtr{UInt64, PTX.AS.Shared}})
    @test asm ==
        "cp.reduce.async.bulk.relaxed.cluster.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u64 " *
        "[\$0], [\$1], \$2, [\$3];"

    spec = build_call(:cp,
        (:reduce, :async, :bulk, :relaxed, :cluster, Symbol("shared::cluster"),
         Symbol("shared::cta"), Symbol("mbarrier::complete_tx::bytes"), :add, :u64),
        (Core.LLVMPtr{UInt64, PTX.AS.Shared},
         Core.LLVMPtr{UInt64, PTX.AS.Shared},
         UInt32,
         Core.LLVMPtr{UInt64, PTX.AS.Shared}))
    @test spec.rettype === Nothing

    # `multimem.cp.async.bulk` with .sem/.scope (9.3, sm_100+). Multicast version
    # of cp.async.bulk; same operand shape, opcode is :multimem.
    asm = format_call(
        ptx"multimem.cp.async.bulk.relaxed.sys.shared::cluster.global.mbarrier::complete_tx::bytes.b128",
        Tuple{Core.LLVMPtr{UInt8, PTX.AS.Shared},
              Core.LLVMPtr{UInt8, PTX.AS.Global},
              UInt32,
              Core.LLVMPtr{UInt64, PTX.AS.Shared}})
    @test asm ==
        "multimem.cp.async.bulk.relaxed.sys.shared::cluster.global.mbarrier::complete_tx::bytes.b128 " *
        "[\$0], [\$1], \$2, [\$3];"

    spec = build_call(:multimem,
        (:cp, :async, :bulk, :relaxed, :sys, Symbol("shared::cluster"), :global,
         Symbol("mbarrier::complete_tx::bytes"), :b128),
        (Core.LLVMPtr{UInt8, PTX.AS.Shared},
         Core.LLVMPtr{UInt8, PTX.AS.Global},
         UInt32,
         Core.LLVMPtr{UInt64, PTX.AS.Shared}))
    @test spec.rettype === Nothing

    # The `(:cp,)` prefix must NOT shadow `(:st,)` / `(:red,)`. Spot-check
    # that st/red still resolve correctly.
    spec = build_call(:multimem, (:st, :async, :release, :gpu, :global, :u32),
        (Core.LLVMPtr{UInt32, PTX.AS.Global}, UInt32))
    @test spec.rettype === Nothing
end

@testset "multimem chain entries (PTX 9.3 async forms)" begin
    # `multimem.st.async.<sem>.<scope>{.ss}.<type> [a], b;` — bracketed
    # multimem address + register value, no return. Routes through the
    # chain default: `:multimem` is _MEM in src/ledgers/forms.jl with `(:st,)` /
    # `(:red,)` _MEMSINK overrides so the terminal dtype isn't misread as
    # a return slot.
    @test format_call(ptx"multimem.st.async.release.gpu.global.u32",
                      Tuple{Core.LLVMPtr{UInt32, PTX.AS.Global}, UInt32}) ==
          "multimem.st.async.release.gpu.global.u32 [\$0], \$1;"
    @test format_call(ptx"multimem.st.async.release.sys.f64",
                      Tuple{Core.LLVMPtr{Float64, PTX.AS.Global}, Float64}) ==
          "multimem.st.async.release.sys.f64 [\$0], \$1;"

    # `multimem.red.async.<sem>.<scope>{.ss}.<op>.<type> [a], b;`
    @test format_call(ptx"multimem.red.async.release.gpu.global.add.u32",
                      Tuple{Core.LLVMPtr{UInt32, PTX.AS.Global}, UInt32}) ==
          "multimem.red.async.release.gpu.global.add.u32 [\$0], \$1;"
    @test format_call(ptx"multimem.red.async.release.gpu.global.add.u64",
                      Tuple{Core.LLVMPtr{UInt64, PTX.AS.Global}, UInt64}) ==
          "multimem.red.async.release.gpu.global.add.u64 [\$0], \$1;"

    # Memory-group opcode → side_effects=true + memory clobber.
    spec = build_call(:multimem, (:st, :async, :release, :gpu, :global, :u32),
                      (Core.LLVMPtr{UInt32, PTX.AS.Global}, UInt32))
    @test spec.rettype == Nothing
    @test spec.side_effects == true
    @test occursin("~{memory}", spec.constraints)

    spec = build_call(:multimem, (:red, :async, :release, :gpu, :global, :add, :u32),
                      (Core.LLVMPtr{UInt32, PTX.AS.Global}, UInt32))
    @test spec.rettype == Nothing
    @test spec.side_effects == true

    # Pre-9.3 forms (sm_90+, PTX 8.1): `multimem.st` / `multimem.red` without
    # `.async`. Same override coverage.
    @test format_call(ptx"multimem.st.relaxed.sys.global.u32",
                      Tuple{Core.LLVMPtr{UInt32, PTX.AS.Global}, UInt32}) ==
          "multimem.st.relaxed.sys.global.u32 [\$0], \$1;"
    @test format_call(ptx"multimem.red.relaxed.sys.global.add.u32",
                      Tuple{Core.LLVMPtr{UInt32, PTX.AS.Global}, UInt32}) ==
          "multimem.red.relaxed.sys.global.add.u32 [\$0], \$1;"

    # `multimem.ld_reduce` (sm_90+) IS a load — terminal `.type` drives the
    # rettype as usual; the sink overrides only catch st/red/cp.
    @test format_call(ptx"multimem.ld_reduce.relaxed.sys.global.add.u32",
                      Tuple{Core.LLVMPtr{UInt32, PTX.AS.Global}}) ==
          "multimem.ld_reduce.relaxed.sys.global.add.u32 \$0, [\$1];"
    spec = build_call(:multimem, (:ld_reduce, :relaxed, :sys, :global, :add, :u32),
                      (Core.LLVMPtr{UInt32, PTX.AS.Global},))
    @test spec.rettype == UInt32
end

@testset "ldmatrix/stmatrix wrapper (tier-2 intrinsic lowering)" begin
    # Migrated family: the plain-`.shared`
    # and b8 forms
    # lower through llvm.nvvm.{ld,st}matrix.* (state space carried by the
    # pointer's address space); the explicit `shared::cta` spellings ride
    # convergent_asm_ir. Goldens: test/golden/{ldmatrix@sm75,stmatrix@sm90,
    # ldst_matrix_b8@sm100a}.ptx.
    pSh  = Core.LLVMPtr{UInt16, PTX.AS.Shared}
    pSh8 = Core.LLVMPtr{UInt8, PTX.AS.Shared}
    T2 = NTuple{2, UInt32}; T4 = NTuple{4, UInt32}
    cases = [
        (:ldmatrix, (:sync, :aligned, :m8n8, :x1, :shared, :b16),
            (pSh,), UInt32, "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.b16"),
        (:ldmatrix, (:sync, :aligned, :m8n8, :x2, :trans, :shared, :b16),
            (pSh,), T2, "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.trans.b16"),
        (:ldmatrix, (:sync, :aligned, :m8n8, :x4, :shared, :b16),
            (pSh,), T4, "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16"),
        # b8: TWO regs per count step — the arity the old asm generator got
        # wrong (it assumed 1; ptxas rejected every b8 form it emitted).
        (:ldmatrix, (:sync, :aligned, :m16n16, :x1, :trans, :shared, :b8),
            (pSh8,), T2, "llvm.nvvm.ldmatrix.sync.aligned.m16n16.x1.trans.b8"),
        (:ldmatrix, (:sync, :aligned, :m16n16, :x2, :trans, :shared, :b8),
            (pSh8,), T4, "llvm.nvvm.ldmatrix.sync.aligned.m16n16.x2.trans.b8"),
        (:stmatrix, (:sync, :aligned, :m8n8, :x1, :shared, :b16),
            (pSh, UInt32), Nothing, "llvm.nvvm.stmatrix.sync.aligned.m8n8.x1.b16"),
        (:stmatrix, (:sync, :aligned, :m8n8, :x2, :trans, :shared, :b16),
            (pSh, T2), Nothing, "llvm.nvvm.stmatrix.sync.aligned.m8n8.x2.trans.b16"),
        (:stmatrix, (:sync, :aligned, :m8n8, :x4, :shared, :b16),
            (pSh, T4), Nothing, "llvm.nvvm.stmatrix.sync.aligned.m8n8.x4.b16"),
        (:stmatrix, (:sync, :aligned, :m16n8, :x4, :trans, :shared, :b8),
            (pSh8, T4), Nothing, "llvm.nvvm.stmatrix.sync.aligned.m16n8.x4.trans.b8"),
    ]
    for (opsym, mods, argts, rettype, intr) in cases
        # the intrinsic is registered and convergent
        @test PTX.NVVM.isintrinsic(intr)
        @test :convergent in PTX.NVVM.intrinsic(intr).props
        # the method dispatches and routes to that intrinsic
        op = Operation{opsym, mods}()
        @test which(op, argts).module == PTX
        ci, rt = first(Base.code_typed(op, argts))
        @test rt === rettype
        @test occursin(intr, string(ci))
    end

    # `shared::cta` spellings stay asm-tier, with convergent nomerge on the
    # call (the IR-level tripwire — llc ignores the attribute and goldens
    # can't observe its loss).
    for (opsym, mods, argts, asm) in [
        (:ldmatrix, (:sync, :aligned, :m8n8, :x1, Symbol("shared::cta"), :b16),
            (pSh,), "ldmatrix.sync.aligned.m8n8.x1.shared::cta.b16"),
        (:ldmatrix, (:sync, :aligned, :m8n8, :x4, :trans, Symbol("shared::cta"), :b16),
            (pSh,), "ldmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16"),
        (:stmatrix, (:sync, :aligned, :m8n8, :x1, Symbol("shared::cta"), :b16),
            (pSh, UInt32), "stmatrix.sync.aligned.m8n8.x1.shared::cta.b16"),
        (:stmatrix, (:sync, :aligned, :m8n8, :x4, :trans, Symbol("shared::cta"), :b16),
            (pSh, T4), "stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16"),
    ]
        ci, _ = first(Base.code_typed(Operation{opsym, mods}(), argts))
        s = string(ci)
        @test occursin(asm, s)
        @test occursin("convergent nomerge", s)
    end

    # The ptxas-invalid b8 forms (non-trans, m16n16 x4) no longer have
    # dedicated methods — they fall through to the chain default like any
    # unregistered form.
    for (opsym, mods, argts) in [
        (:ldmatrix, (:sync, :aligned, :m16n16, :x1, :shared, :b8), (pSh8,)),
        (:ldmatrix, (:sync, :aligned, :m16n16, :x4, :trans, :shared, :b8), (pSh8,)),
        (:stmatrix, (:sync, :aligned, :m16n8, :x1, :shared, :b8), (pSh8, UInt32)),
    ]
        m = which(Operation{opsym, mods}(), argts)
        @test !occursin("m16n16", string(m.sig)) && !occursin("m16n8", string(m.sig))
    end
end

@testset "mbarrier wrapper (single-route asm lowering)" begin
    # The whole family lowers through the mbarrier_forms schema to
    # convergent inline asm — the ten former tier-2 forms were demoted
    # (observable sync effects gain nothing from intrinsic attributes, and
    # the 9.3 report forms, cluster-space sinks, and 9.4 multicast forms
    # have no intrinsics anyway). Every call site carries the family-wide
    # sideeffect + ~{memory} + convergent nomerge contract. Goldens:
    # test/golden/mbarrier@sm{80,90}.ptx.
    pS = Core.LLVMPtr{UInt64, PTX.AS.Shared}
    cases = [
        ((:init, :shared, :b64),                (pS, UInt32), Nothing,
            "mbarrier.init.shared.b64 ["),
        ((:inval, :shared, :b64),               (pS,),        Nothing,
            "mbarrier.inval.shared.b64 ["),
        ((:arrive, :shared, :b64),              (pS,),        UInt64,
            "mbarrier.arrive.shared.b64 "),
        ((:arrive, :noComplete, :shared, :b64), (pS, UInt32), UInt64,
            "mbarrier.arrive.noComplete.shared.b64 "),
        ((:arrive, :expect_tx, :shared, :b64),  (pS, UInt32), UInt64,
            "mbarrier.arrive.expect_tx.shared.b64 "),
        ((:expect_tx, :shared, :b64),           (pS, UInt32), Nothing,
            "mbarrier.expect_tx.shared.b64 ["),
        ((:test_wait, :shared, :b64),           (pS, UInt64), Bool,
            "mbarrier.test_wait.shared.b64 "),
        ((:test_wait, :parity, :shared, :b64),  (pS, UInt32), Bool,
            "mbarrier.test_wait.parity.shared.b64 "),
        ((:try_wait, :shared, :b64),            (pS, UInt64), Bool,
            "mbarrier.try_wait.shared.b64 "),
        ((:try_wait, :parity, :shared, :b64),   (pS, UInt32), Bool,
            "mbarrier.try_wait.parity.shared.b64 "),
    ]
    for (mods, argts, rettype, asm) in cases
        op = Operation{:mbarrier, mods}()
        @test which(op, argts).module == PTX
        ci, rt = first(Base.code_typed(op, argts))
        code = string(ci)
        @test rt === rettype
        @test occursin(asm, code)
        # single-route contract: no intrinsic call, full asm effect set
        @test !occursin("llvm.nvvm.mbarrier", code)
        @test occursin("~{memory}", code)
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

@testset "mbarrier PTX 9.3 extensions (layout / phase_type / report)" begin
    # Asm tier by necessity (no NVVM intrinsics at 22.1.7). Prefix-matching
    # on the @generated body's asm head plus rettype pins; `:report` is the
    # audited synthetic report selector for the predicate pair + value.
    pS = Core.LLVMPtr{UInt64, PTX.AS.Shared}
    cases = [
        # mods, argts, rettype, asm head
        ((:init, Symbol("layout::v0"), :shared, :b64), (pS, UInt32), Nothing,
            "mbarrier.init.layout::v0.shared.b64 ["),
        ((:init, Symbol("layout::v1"), :shared, :b64), (pS, Int), Nothing,
            "mbarrier.init.layout::v1.shared.b64 ["),
        ((:check_layout, Symbol("layout::v0"), Symbol("shared::cta"), :b64), (pS,), Bool,
            "mbarrier.check_layout.layout::v0.shared::cta.b64 "),
        ((:check_layout, Symbol("layout::v1"), Symbol("shared::cta"), :b64), (pS,), Bool,
            "mbarrier.check_layout.layout::v1.shared::cta.b64 "),
        # grouped report forms: (waitComplete, reportPredicate, reportValue)
        ((:test_wait, :report, Symbol("phase_type::primary"), :shared, :b64),
            (pS, UInt64), Tuple{Bool, Bool, UInt16},
            "mbarrier.test_wait.phase_type::primary.shared.b64 "),
        ((:test_wait, :report, :parity, Symbol("phase_type::primary"), :shared, :b64),
            (pS, UInt32), Tuple{Bool, Bool, UInt16},
            "mbarrier.test_wait.parity.phase_type::primary.shared.b64 "),
        ((:try_wait, :report, Symbol("phase_type::primary"), :shared, :b64),
            (pS, UInt64), Tuple{Bool, Bool, UInt16},
            "mbarrier.try_wait.phase_type::primary.shared.b64 "),
        ((:try_wait, :report, :parity, Symbol("phase_type::primary"), :shared, :b64),
            (pS, UInt32), Tuple{Bool, Bool, UInt16},
            "mbarrier.try_wait.parity.phase_type::primary.shared.b64 "),
        # conditional-phase parity waits (both layouts; single output)
        ((:test_wait, :parity, Symbol("phase_type::conditional"), :shared, :b64),
            (pS, UInt32), Bool,
            "mbarrier.test_wait.parity.phase_type::conditional.shared.b64 "),
        ((:try_wait, :parity, Symbol("phase_type::conditional"), :shared, :b64),
            (pS, UInt32), Bool,
            "mbarrier.try_wait.parity.phase_type::conditional.shared.b64 "),
    ]
    for (mods, argts, rettype, asm) in cases
        op = Operation{:mbarrier, mods}()
        @test which(op, argts).module == PTX
        ci, rt = first(Base.code_typed(op, argts))
        @test rt === rettype
        s = string(ci)
        @test occursin(asm, s)
        @test occursin("~{memory}", s)
    end

    # Full report shape: predicate pair plus a b8 temporary packed into the
    # low byte of an NVPTX-compatible UInt16 carrier.
    ci, _ = first(Base.code_typed(
        Operation{:mbarrier, (:test_wait, :report, Symbol("phase_type::primary"),
                              :shared, :b64)}(), (pS, UInt64)))
    @test occursin("\\\$0|\\\$1, report_value", string(ci))
    @test occursin("mov.b16 \\\$2, {report_value, 0}", string(ci))
    @test occursin("=b,=b,=h,r,l", string(ci))
end

@testset "fabric.* hand-written wrappers (PTX 9.3, sm_100+)" begin
    # Zero-arg lifecycle ops — typed wrappers in fabric.jl. Each check
    # confirms the @generated body baked the verbatim asm string, and the
    # constraints check pins the `~{memory}` clobber (these are observable
    # cross-GPU ops, must not be reordered around mbarrier submit/wait).
    #
    # `:fabric` is deliberately NOT in the form registry: the CFT handle
    # operand `[leId, off]` has no chain-default rendering, so unimplemented
    # fabric forms must error at the blessing boundary, not render wrong asm.

    pS  = Core.LLVMPtr{UInt64, PTX.AS.Shared}
    pS8 = Core.LLVMPtr{UInt8,  PTX.AS.Shared}

    # Match opcode + qualifier prefix only (operand `[$N]` chunks are escape-
    # hostile inside the lowered LLVM IR string — mirrors the mbarrier
    # @generated body-expansion testset's prefix-only style).
    cases = [
        # mods, argts, expected_asm_prefix, expected_constraints
        ((:submit,), (), "fabric.submit;", "~{memory}"),
        ((:submit, Symbol("op_restrict::fetching")), (),
            "fabric.submit.op_restrict::fetching;", "~{memory}"),
        ((:wait, Symbol("sync_restrict::reads")), (),
            "fabric.wait.sync_restrict::reads;", "~{memory}"),

        ((:try_get, :async, Symbol("shared::cta"),
          Symbol("mbarrier::complete_tx::bytes"),
          Symbol("mbarrier::report::fabric"),
          :relaxed, :sys, :b128),
            (pS8, UInt32, UInt64, UInt32, pS),
            "fabric.try_get.async.shared::cta" *
              ".mbarrier::complete_tx::bytes.mbarrier::report::fabric" *
              ".relaxed.sys.b128",
            "r,r,l,r,r,~{memory}"),

        # try_put basic. `complete_tx::16B` is the put-side completion mechanism
        # (count per 16-byte chunk), distinct from try_get's `complete_tx::bytes`
        # (count = exact bytes).
        ((:try_put, :async, Symbol("shared::cta"),
          Symbol("mbarrier::complete_tx::16B"),
          Symbol("mbarrier::report::fabric"),
          :relaxed, :sys, :b128),
            (UInt32, UInt64, pS8, UInt32, pS),
            "fabric.try_put.async.shared::cta" *
              ".mbarrier::complete_tx::16B.mbarrier::report::fabric" *
              ".relaxed.sys.b128",
            "r,l,r,r,r,~{memory}"),

        # try_put .multimem: `.multimem` inserted after `.async`.
        ((:try_put, :async, :multimem, Symbol("shared::cta"),
          Symbol("mbarrier::complete_tx::16B"),
          Symbol("mbarrier::report::fabric"),
          :relaxed, :sys, :b128),
            (UInt32, UInt64, pS8, UInt32, pS),
            "fabric.try_put.async.multimem.shared::cta" *
              ".mbarrier::complete_tx::16B.mbarrier::report::fabric" *
              ".relaxed.sys.b128",
            "r,l,r,r,r,~{memory}"),
    ]
    for (mods, argts, expected_asm, expected_cons) in cases
        op = Operation{:fabric, mods}()
        @test which(op, argts).module == PTX
        ci, rt = first(Base.code_typed(op, argts))
        @test rt === Nothing
        s = string(ci)
        @test occursin(expected_asm,  s)
        @test occursin(expected_cons, s)
    end

    # Unimplemented fabric forms die at the blessing boundary (no registry
    # entry), steering to the wrappers or the raw tier.
    @test_throws ErrorException build_call(:fabric, (:try_red,), (pS8,))
end


@testset "wgmma sync ops" begin
    # wgmma.fence/commit_group/wait_group flow through the chain default.
    @test format_call(ptx"wgmma.fence.sync.aligned",        Tuple{})       == "wgmma.fence.sync.aligned;"
    @test format_call(ptx"wgmma.commit_group.sync.aligned", Tuple{})       == "wgmma.commit_group.sync.aligned;"
    @test format_call(ptx"wgmma.wait_group.sync.aligned",   Tuple{Val{0}}) == "wgmma.wait_group.sync.aligned 0;"
    @test format_call(ptx"wgmma.wait_group.sync.aligned",   Tuple{Val{2}}) == "wgmma.wait_group.sync.aligned 2;"

    # `:wgmma` is registered nonpure (src/ledgers/forms.jl) → memory clobber + side_effects.
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

@testset "collective asm forms carry `convergent`" begin
    # wgmma.mma_async and the mma.sync asm fallbacks are warp(group)-
    # collective: `sideeffect` alone permits jump-threading duplication of
    # the call site across a divergent branch — the active_mask miscompile
    # class. The wrappers emit llvmcall IR with a `convergent` call-site
    # attribute group (src/dsl/convergent_asm.jl, convergent_asm_ir). This attribute binds in
    # the *in-process* middle end only — llc ignores it — so no ptxas/golden
    # test can catch its loss; this IR-level pin is the tripwire.
    # (The optimizer-shape counterpart lives in test/ptxas/hopper.jl.)
    unescape(s) = replace(s, "\\\"" => "\"")

    # wgmma: all three scale_d variants of a representative form.
    wg = Operation{:wgmma, (:mma_async, :sync, :aligned,
                            :m64n8k16, :f32, :bf16, :bf16)}()
    for argts in ((NTuple{4, Float32}, UInt64, UInt64, Bool),
                  (NTuple{4, Float32}, UInt64, UInt64, Val{true}),
                  (NTuple{4, Float32}, UInt64, UInt64, Val{false}))
        ci, rt = first(Base.code_typed(wg, argts))
        @test rt === NTuple{4, Float32}
        s = unescape(string(ci))
        @test occursin("asm sideeffect", s)
        @test occursin("convergent", s)
    end

    # mma asm fallback: kind::f8f6f4 at m16n8k16 (no intrinsic at 22.1.7).
    mma_fb = Operation{:mma, (:sync, :aligned, Symbol("kind::f8f6f4"),
                              :m16n8k16, :row, :col,
                              :f32, :e4m3, :e4m3, :f32)}()
    ci, rt = first(Base.code_typed(mma_fb,
        (NTuple{2, UInt32}, NTuple{1, UInt32}, NTuple{4, Float32})))
    @test rt === NTuple{4, Float32}
    s = unescape(string(ci))
    @test occursin("asm sideeffect", s)
    @test occursin("convergent", s)

    # mma_scaled asm fallback: mxf4nvf4 scale_vec::4X ue8m0 (no intrinsic
    # at 22.1.7). aeff3ee converted wgmma + dense mma but missed this file
    # — found and fixed during B4; this pin keeps it fixed.
    mma_sc_fb = Operation{:mma, (:sync, :aligned, Symbol("kind::mxf4nvf4"),
                                 :block_scale, Symbol("scale_vec::4X"),
                                 :m16n8k64, :row, :col,
                                 :f32, :e2m1, :e2m1, :f32, :ue8m0)}()
    ci, rt = first(Base.code_typed(mma_sc_fb,
        (NTuple{4, UInt32}, NTuple{2, UInt32}, NTuple{4, Float32},
         UInt32, UInt16, UInt16, UInt32, UInt16, UInt16)))
    @test rt === NTuple{4, Float32}
    s = unescape(string(ci))
    @test occursin("asm sideeffect", s)
    @test occursin("convergent nomerge", s)

    # Tier-2 mma (dense + scaled): upstream props lack IntrConvergent (the
    # whole generated `llvm.nvvm.mma.` surface is IntrNoMem only at 22.1.7).
    # The emission overlay must put `convergent nomerge` on the call site;
    # retaining it on the declaration is harmless but insufficient alone.
    mma_t2 = Operation{:mma, (:sync, :aligned, :m16n8k16, :row, :col,
                              :f32, :bf16, :bf16, :f32)}()
    ci, rt = first(Base.code_typed(mma_t2,
        (NTuple{4, UInt32}, NTuple{2, UInt32}, NTuple{4, Float32})))
    @test rt === NTuple{4, Float32}
    s = unescape(string(ci))
    @test occursin("convergent nomerge", s)

    # Contrast: a pure chain-default form (cvt) must stay unattributed and
    # side-effect-free — CSE/DCE of pure conversions is intended.
    spec = build_call(:cvt, (:rn, :f16, :f32), (Float32,))
    @test spec.side_effects == false
    @test !occursin("~{memory}", spec.constraints)
end

@testset "wgmma.mma_async — floating/integer variant coverage" begin
    # Per-variant goldens for every entry of the separate dtype inventories in
    # src/wrappers/wgmma.jl. The encoder is cross-checked against pyptx's
    # `_Wgmma.mma_async()` (pyptx/pyptx/ptx.py:817-910) — same modifier
    # ordering, same imm tail (`scaleD, scaleA[, transA, transB]`), same
    # tied-operand convention. Corpus reference:
    # pyptx/tests/corpus/wgmma_simple.ptx (m64n256k16.f32.e4m3.e4m3) and
    # wgmma_gemm_tile.ptx (m64n128k16.f32.bf16.bf16).
    #
    # The five "popular" combos are spot-checked above; this set fills the
    # remaining variants at representative N values.

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
    # PTX 9.3 §9.7.16.5.2: integer wgmma has only `scale-d` (no scale-a/b
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

    # Largest legal dense s32 shape (m64n224k32) — 112 d-regs.  Integer
    # WGMMA has a narrower N grid than floating-point WGMMA.
    spec = PTX.wgmma_mma_async_spec(:s32, :s8, :s8, 224, 32, false)
    @test spec.nd == 112
    @test occursin("m64n224k32.s32.s8.s8", spec.asm)
    @test occursin(", \$110, \$111}", spec.asm)
    @test endswith(spec.asm, ", \$112, \$113, \$114;")
    @test endswith(spec.constraints, ",110,111,~{memory}")

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
    # directly against PTX 9.3 §9.7.16.5.

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

    # M and N field placement: m=32 → bits[28:24]=2, m=256 → 16.
    # n=8 → bits[22:17]=1, n=128 → 16.
    @test tcgen05_instr_desc_f16bf16_f32(m = 32,  n = 256, ab_dtype = :bf16) ===
          UInt32((2  << 24) | (32 << 17) | 0x490)
    @test tcgen05_instr_desc_f16bf16_f32(m = 64,  n = 256, ab_dtype = :bf16) ===
          UInt32((4  << 24) | (32 << 17) | 0x490)
    @test tcgen05_instr_desc_f16bf16_f32(m = 256, n = 256, ab_dtype = :bf16) ===
          UInt32((16 << 24) | (32 << 17) | 0x490)
    @test tcgen05_instr_desc_f16bf16_f32(m = 128, n = 8,   ab_dtype = :bf16) ===
          UInt32((8  << 24) | (1  << 17) | 0x490)
    @test tcgen05_instr_desc_f16bf16_f32(m = 128, n = 128, ab_dtype = :bf16) ===
          UInt32((8  << 24) | (16 << 17) | 0x490)

    # sparse / max_shift bits. Saturation is integer-only in Table 45 and is
    # deliberately absent from this floating-point builder.
    @test tcgen05_instr_desc_f16bf16_f32(m = 128, n = 256, ab_dtype = :bf16,
                                         sparse = true) ===
          UInt32(base | (UInt32(1) << 2))
    @test_throws MethodError tcgen05_instr_desc_f16bf16_f32(
        m = 128, n = 256, ab_dtype = :bf16, saturate = true)
    @test tcgen05_instr_desc_f16bf16_f32(m = 128, n = 256, ab_dtype = :bf16,
                                         max_shift = 3) ===
          UInt32(base | (UInt32(3) << 30))

    # Range checks.
    @test_throws ArgumentError tcgen05_instr_desc_f16bf16_f32(
        m = 96, n = 256, ab_dtype = :bf16)           # m must be 32/64/128/256
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

    # PTX Table 43 fixes bits 46–48 at 0b001 in every valid descriptor.
    fixed = UInt64(1) << 46

    # An otherwise-empty descriptor still carries the fixed constant.
    @test tcgen05_descriptor(UInt32(0); leading_bytes = 0,
                                         stride_bytes = 0) === fixed

    # Address: 14-bit field at bits [13:0], encoded as (addr & 0x3FFF0) >> 4.
    # 0x100 → 0x10. Sits in low 14 bits.
    @test tcgen05_descriptor(UInt32(0x100); leading_bytes = 0,
                                            stride_bytes = 0) ===
          fixed | UInt64(0x10)
    # Values outside the 18-bit input window are rejected, not truncated.
    @test_throws ArgumentError tcgen05_descriptor(
        UInt32(0x40000); leading_bytes = 0, stride_bytes = 0)

    # Leading / stride byte fields.
    @test tcgen05_descriptor(UInt32(0); leading_bytes = 16,
                                         stride_bytes = 0) ===
          fixed | (UInt64(1) << 16)
    @test tcgen05_descriptor(UInt32(0); leading_bytes = 0,
                                         stride_bytes = 1024) ===
          fixed | (UInt64(64) << 32)

    # Layout-type at bit 61 (3 bits — distinct from wgmma's 2-bit field).
    @test tcgen05_descriptor(UInt32(0); leading_bytes = 0,
                                         stride_bytes = 0,
                                         swizzle = BlackwellLayout.B128) ===
          fixed | (UInt64(BlackwellLayout.B128) << 61)
    @test tcgen05_descriptor(UInt32(0); leading_bytes = 0,
                                         stride_bytes = 0,
                                         swizzle = BlackwellLayout.B64) ===
          fixed | (UInt64(BlackwellLayout.B64) << 61)

    # The former caller-controlled `version` keyword is gone: this is a fixed
    # ISA constant, not a version field.
    @test tcgen05_descriptor(UInt32(0); leading_bytes = 0,
                                         stride_bytes = 0) === fixed
    @test_throws MethodError tcgen05_descriptor(
        UInt32(0); leading_bytes = 0, stride_bytes = 0, version = 1)

    # base_offset / lbo_mode. Absolute mode requires 128B swizzling with 16B
    # atomicity and base_offset=0 (PTX §9.7.17.3.1.2).
    @test tcgen05_descriptor(UInt32(0); leading_bytes = 0, stride_bytes = 0,
                                         base_offset = 5) ===
          fixed | (UInt64(5) << 49)
    @test tcgen05_descriptor(UInt32(0); leading_bytes = 0, stride_bytes = 0,
                                         swizzle = BlackwellLayout.B128,
                                         lbo_mode = 1) ===
          fixed | (UInt64(1) << 52) |
          (UInt64(BlackwellLayout.B128) << 61)

    # The pyptx-equivalent BLACKWELL_MASKED_DESC_B128 constant:
    # leading=16, stride=1024, swizzle=B128, fixed bits=0b001.
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
    # fences are tier-1 core IR — see the next testset.
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

@testset "fabric proxy fences (PTX 9.3, asm tier)" begin
    # `fence.proxy.<to::from>.alias.<sem>.sys;` — no NVVM intrinsics at
    # 22.1.7, so unlike the proxy/init fences above these stay on the asm
    # tier: sideeffect + `~{memory}`, not convergent.
    for dir in (Symbol("generic::fabric"), Symbol("fabric::generic"),
                Symbol("fabric::fabric")),
        sem in (:acquire, :release)

        mods = (:proxy, dir, :alias, sem, :sys)
        op = Operation{:fence, mods}()
        @test which(op, ()).module == PTX
        ci, rt = first(Base.code_typed(op, ()))
        @test rt === Nothing
        s = string(ci)
        @test occursin("fence.proxy.$dir.alias.$sem.sys;", s)
        @test occursin("~{memory}", s)
    end
end

@testset "generic memory fences (tier-1 core IR)" begin
    # fence.{sc,acq_rel}.{cta,cluster,gpu,sys} lower to a core-IR
    # `fence <ordering> syncscope(...)` — the PTX↔LLVM mapping is pinned
    # in wrappers/fence.jl (sc↔seq_cst, acq_rel↔acq_rel; cta↔"block",
    # cluster↔"cluster", gpu↔"device", sys↔system default). Each body must
    # be the fence instruction, not asm.
    for (sem, ordering) in ((:sc, "seq_cst"), (:acq_rel, "acq_rel"))
        for (scope, syncscope) in ((:cta,     "syncscope(\"block\") "),
                                   (:cluster, "syncscope(\"cluster\") "),
                                   (:gpu,     "syncscope(\"device\") "),
                                   (:sys,     ""))
            mods = (sem, scope)
            @test which(Operation{:fence, mods}(), ()).module == PTX
            ci, rt = first(Base.code_typed(Operation{:fence, mods}(), ()))
            @test rt === Nothing
            # CodeInfo printing escapes the quotes inside the llvmcall IR
            # string; unescape before matching the syncscope clause.
            s = replace(string(ci), "\\\"" => "\"")
            @test occursin("fence $syncscope$ordering", s)
            @test !occursin("asm", s)   # tier 1: real ordering op, no asm
        end
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

    # ld/st: the full Table-52 grid (plain and pack/unpack variants) —
    # arity, inferred return, intrinsic. 16x32bx2 threads its
    # immHalfSplitoff immediate as a Val operand after taddr.
    for (shape, base) in (("16x64b", 1), ("32x32b", 1),
                          ("16x128b", 2), ("16x256b", 4),
                          ("16x32bx2", 1)),
        c in (1, 2, 4, 8, 16, 32, 64, 128),
        repack in (false, true)

        n = base * c
        n > 128 && continue
        sh = Symbol(shape)
        cnt = Symbol("x$c")
        split = shape == "16x32bx2" ? (Val{8},) : ()
        ldflag = repack ? (Symbol("pack::16b"),) : ()
        stflag = repack ? (Symbol("unpack::16b"),) : ()
        ld = Operation{:tcgen05, (:ld, :sync, :aligned, sh, cnt,
                                  ldflag..., :b32)}()
        ci, rt = first(Base.code_typed(ld, (UInt32, split...)))
        @test rt === (n == 1 ? UInt32 : NTuple{n, UInt32})
        @test occursin("tcgen05.ld.$shape.x$c", string(ci))
        st = Operation{:tcgen05, (:st, :sync, :aligned, sh, cnt,
                                  stflag..., :b32)}()
        ci2, rt2 = first(Base.code_typed(st, (UInt32, split...,
                                              NTuple{n, UInt32})))
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

    # MX kinds use the distinct seven-operand block-scale schema.  A complete
    # closed-world inventory lives in host/matrix_api_safety.jl; this keeps a
    # representative asm-tier inference pin beside the dense intrinsic pins.
    mxmods = (:mma, cg1, Symbol("kind::mxf8f6f4"), :block_scale,
              Symbol("scale_vec::1X"))
    ci, rt = first(Base.code_typed(Operation{:tcgen05, mxmods}(),
        (UInt32, UInt64, UInt64, UInt32, UInt32, UInt32, Bool)))
    @test rt === Nothing
    @test occursin(
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X",
        string(ci))
end

@testset "tuple args → braced operand groups" begin
    # Homogeneous tuple types render as `{$N, $N+1, ...}` and contribute one
    # LLVM input slot per lane. Used by .rs cvt forms and any chain-default
    # op where PTX requires a register-vector operand (ldmatrix / stmatrix /
    # mma once built this by hand on the asm tier; all three are tier-2
    # now). The chain default does it via `render_arg(::Type{<:Tuple})`.
    # :fakeop is deliberately unregistered — rendering is exercised with an
    # explicit permissive contract (the registry gate itself is tested in
    # host/inst.jl).
    fake = PTX.FormContract(effects = :pure)

    # Hypothetical fma form taking a 4-lane tuple of f32s as the first
    # operand. Not a real PTX op — we just want to exercise the braced
    # rendering. Real ops with this shape (mma.sync, ldmatrix, etc.) have
    # hand-written wrappers; this confirms the chain default also works.
    spec = build_call(:fakeop, (:f32,), (NTuple{4, Float32},); contract = fake)
    @test spec.asm == "fakeop.f32 \$0, {\$1, \$2, \$3, \$4};"
    @test spec.rettype === Float32
    @test spec.constraints == "=f,f,f,f,f"
    @test spec.passthrough_argtypes === (Float32, Float32, Float32, Float32)
    @test spec.passthrough_indices === ((1, 1), (1, 2), (1, 3), (1, 4))

    # Mix scalar and tuple args — slot numbering interleaves correctly.
    spec = build_call(:fakeop, (:f32,), (Float32, NTuple{2, UInt32}, Float32); contract = fake)
    @test spec.asm == "fakeop.f32 \$0, \$1, {\$2, \$3}, \$4;"
    @test spec.constraints == "=f,f,r,r,f"
    @test spec.passthrough_indices ===
          ((1, nothing), (2, 1), (2, 2), (3, nothing))

    # NTuple{1, T} is still tuple-shaped — emits `{$N}` (single-element
    # brace group). Real PTX accepts this; e.g. `stmatrix.x1` uses
    # `{$1}` not `$1` per the spec.
    spec = build_call(:fakeop, (), (NTuple{1, UInt32},); contract = fake)
    @test spec.asm == "fakeop {\$0};"
    @test spec.constraints == "r"
    @test spec.passthrough_indices === ((1, 1),)

    # Heterogeneous tuple is rejected — there's no single constraint
    # letter that works.
    @test_throws ErrorException build_call(:fakeop, (),
                                            (Tuple{Float32, Int32},); contract = fake)

    # Empty tuple is rejected — no operand mapping.
    @test_throws ErrorException build_call(:fakeop, (), (Tuple{},); contract = fake)
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
    # First migrated family: the notation surface is unchanged
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

    # v4.f32 load: bitcast the i8 ABI pointer to pointer-to-vector (typed
    # spelling — Julia ≤ 1.11 device context parses typed only; the ≥ 1.12
    # opaque upgrade folds the cast away), load <4 x float>, unpack into
    # the [4 x float] tuple.
    ir = PTX.vec_ld_ir(4, :f32, 16)
    @test occursin("bitcast i8 addrspace(1)* %0 to <4 x float> addrspace(1)*", ir)
    @test occursin("load <4 x float>, <4 x float> addrspace(1)* %vp, align 16", ir)
    @test occursin("extractelement <4 x float> %v, i32 3", ir)
    @test occursin("insertvalue [4 x float]", ir)
    @test occursin("ret [4 x float]", ir)
    @test !occursin("~{memory}", ir)   # no optimization barrier

    # v2.b32 → i32 elements, align 8.
    ir = PTX.vec_ld_ir(2, :b32, 8)
    @test occursin("load <2 x i32>, <2 x i32> addrspace(1)* %vp, align 8", ir)

    # v4.b16 → i16 elements, align 8.
    ir = PTX.vec_ld_ir(4, :b16, 8)
    @test occursin("load <4 x i16>, <4 x i16> addrspace(1)* %vp, align 8", ir)

    # v4.f32 store: unpack the [4 x float] arg (%1), build <4 x float>, store
    # to the pointer arg (%0, bitcast to pointer-to-vector).
    ir = PTX.vec_st_ir(4, :f32, 16)
    @test occursin("extractvalue [4 x float] %1, 3", ir)
    @test occursin("insertelement <4 x float>", ir)
    @test occursin("bitcast i8 addrspace(1)* %0 to <4 x float> addrspace(1)*", ir)
    @test occursin("store <4 x float> %v3, <4 x float> addrspace(1)* %sp, align 16", ir)
    @test occursin("ret void", ir)

    # v2.b16 store, align 4.
    ir = PTX.vec_st_ir(2, :b16, 4)
    @test occursin("store <2 x i16> %v1, <2 x i16> addrspace(1)* %sp, align 4", ir)

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
    # The wgmma generator registers floating and integer variants against
    # separate N grids (PTX 9.3 §9.7.16.5.2). For unsupported
    # combinations — e.g. bf16/bf16 with k=8 (only valid for tf32), N=7, or
    # N=264 — no specific method exists on `Operation{parts}`. Dispatch still
    # reaches the shared `Vararg` method in `src/dsl/entries.jl`, but its typed-wrapper
    # boundary must reject the call before it can render garbage asm (wrong
    # rettype and operand count). Tests here lock in the registered surface so
    # a dropped (n,k,dtype) combo fails at host time with an actionable error.

    # Registered set: spot-check each valid dtype tuple across its own grid.
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
        ns = dt_d === :s32 ? (8, 64, 128, 224) : (8, 64, 128, 256)
        for n in ns
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
    # Resolves to the shared method in src/dsl/entries.jl, NOT a wgmma-specific method,
    # but reflection classifies that generic path as forbidden.
    @test endswith(string(bad_k_method.file), "entries.jl")
    @test PTX.lowering(Operation{:wgmma, mods_bad_k}(),
                       (NTuple{4, Float32}, UInt64, UInt64, Bool)).tier === :forbidden

    # N=7 (not step-by-8) and N=264 (over cap) — same chain-default fallthrough.
    mods_bad_n = (:mma_async, :sync, :aligned,
                  :m64n7k16, :f32, :bf16, :bf16)
    @test endswith(string(which(Operation{:wgmma, mods_bad_n}(),
                                 (NTuple{4, Float32}, UInt64, UInt64, Bool)).file),
                   "entries.jl")
    @test PTX.lowering(Operation{:wgmma, mods_bad_n}(),
                       (NTuple{4, Float32}, UInt64, UInt64, Bool)).tier === :forbidden

    # N=40 belongs to the floating grid but is not a dense integer shape.
    mods_bad_int_n = (:mma_async, :sync, :aligned,
                      :m64n40k32, :s32, :s8, :s8)
    @test endswith(string(which(Operation{:wgmma, mods_bad_int_n}(),
                                 (NTuple{20, Int32}, UInt64, UInt64, Bool)).file),
                   "entries.jl")
    @test PTX.lowering(Operation{:wgmma, mods_bad_int_n}(),
                       (NTuple{20, Int32}, UInt64, UInt64, Bool)).tier === :forbidden
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
        PTX._mma_int_register(:m16n8k16, :s8, :u8, true)
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
    @test which(Operation{:mma,
            (:sync, :aligned, :m16n8k16, :row, :col, :satfinite,
             :s32, :s8, :u8, :s32)}(),
        (NTuple{2, UInt32}, NTuple{1, UInt32}, NTuple{4, Int32})).module == PTX
    # f64 convention: Float64 A/B/C/D fragments.
    @test which(Operation{:mma,
            (:sync, :aligned, :m8n8k4, :row, :col, :f64, :f64, :f64, :f64)}(),
        (NTuple{1, Float64}, NTuple{1, Float64}, NTuple{2, Float64})).module == PTX

    # ---- _mma_sp_register ----
    silent() do
        # Re-register (idempotent bookkeeping) + early-return path.
        PTX._mma_sp_register(:m16n8k32, :f32, :bf16, :bf16, :f32)
        PTX._mma_sp_register(:m16n8k32, :f32, :bf16, :bf16, :f32;
                             ordered = true)
        PTX._mma_sp_register(:m16n8k64, :s32, :u4, :s4, :s32;
                             satfinite = true)
        PTX._mma_sp_register(:m16n8k64, :s32, :s4, :u4, :s32;
                             ordered = true)
        @test PTX._mma_sp_register(:m99n99k99, :f32, :bf16, :bf16, :f32) === nothing
        @test PTX._mma_sp_register(:m16n8k32, :f32, :bf16, :bf16, :f32;
                                   satfinite = true) === nothing
    end
    @test which(Operation{:mma,
            (:sp, :sync, :aligned, :m16n8k32, :row, :col, :f32, :bf16, :bf16, :f32)}(),
        (NTuple{4, UInt32}, NTuple{4, UInt32}, NTuple{4, Float32},
         UInt32, Val{0})).module == PTX
    @test which(Operation{:mma,
            (Symbol("sp::ordered_metadata"), :sync, :aligned, :m16n8k32,
             :row, :col, :f32, :bf16, :bf16, :f32)}(),
        (NTuple{4, UInt32}, NTuple{4, UInt32}, NTuple{4, Float32},
         UInt32, Val{1})).module == PTX
    @test which(Operation{:mma,
            (:sp, :sync, :aligned, :m16n8k64, :row, :col, :satfinite,
             :s32, :u4, :s4, :s32)}(),
        (NTuple{2, UInt32}, NTuple{2, UInt32}, NTuple{4, Int32},
         UInt32, Val{1})).module == PTX
    @test which(Operation{:mma,
            (Symbol("sp::ordered_metadata"), :sync, :aligned, :m16n8k64,
             :row, :col, :s32, :s4, :u4, :s32)}(),
        (NTuple{2, UInt32}, NTuple{2, UInt32}, NTuple{4, Int32},
         UInt32, Val{1})).module == PTX

    # ---- _mma_scaled_register ----
    silent() do
        PTX._mma_scaled_register(:mxf8f6f4, Symbol("1X"),
            :m16n8k32, :row, :col, :f32, :e4m3, :e4m3, :f32, :ue8m0)
        # Early-return path: shape/dtype not in MMA_SCALED_FRAGS.
        @test PTX._mma_scaled_register(:mxf4, Symbol("1X"),
            :m99n99k99, :row, :col, :f32, :e2m1, :e2m1, :f32, :ue8m0) === nothing
    end

    # (ldmatrix/stmatrix migrated to tier-2 literal methods; the ::cta asm
    # forms are built by include-time loops — no register helper left.
    # Dispatch sanity lives in the tier-2 wrapper testset above.)

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

    # ---- shfl: no _register helper since the tier-2 migration (methods
    # come from a top-level loop in wrappers/shfl.jl); dispatch sanity only
    @test which(Operation{:shfl, (:sync, :idx, :b32)}(),
                (UInt32, UInt32, UInt32, UInt32)).module == PTX
    @test which(Operation{:shfl, (:sync, :bfly, :b32, :pred)}(),
                (UInt32, UInt32, UInt32, UInt32)).module == PTX
end
