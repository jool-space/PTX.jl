using PTX: Operation, RawOperation, build_call, format_call
using InteractiveUtils: code_llvm

@noinline function _tcgen05_thread_fence_optimizer_probe()
    ptx"tcgen05.fence::before_thread_sync"()
    ptx"tcgen05.fence::after_thread_sync"()
    return nothing
end

# Independent closed-world oracle. Do not derive this from
# TYPED_WRAPPER_ONLY_RULES: adding or broadening a production rule must force a
# test review of the public fail-loud boundary.
const EXPECTED_TYPED_ONLY_RULES = Set([
    (:mma,      (),            nothing),
    (:ldmatrix, (),            nothing),
    (:wgmma,    (:mma_async,), nothing),
    (:tcgen05,  (),            nothing),
    (:shfl,     (),            :pred),
])

@testset "typed-wrapper-only boundary: closed-world rule inventory" begin
    actual = Set((r.op, r.prefix, r.marker) for r in PTX.TYPED_WRAPPER_ONLY_RULES)
    @test actual == EXPECTED_TYPED_ONLY_RULES
    @test length(PTX.TYPED_WRAPPER_ONLY_RULES) == 5

    positives = (
        (:mma, (:sync, :aligned, :m16n8k16, :row, :col,
                :f32, :f16, :f16, :f32)),
        (:mma, (:sp, :sync, :aligned, :m16n8k32, :row, :col,
                :f32, :bf16, :bf16, :f32)),
        (:ldmatrix, (:sync, :aligned, :m8n16, :x1, :shared,
                     :b8x16, :b6x16_p32)),
        (:wgmma, (:mma_async, :sync, :aligned,
                  :m64n8k16, :f32, :bf16, :bf16)),
        (:tcgen05, (:ld, :sync, :aligned, Symbol("16x64b"), :x2, :b32)),
        (:shfl, (:sync, :idx, :b32, :pred)),
    )
    for (op, mods) in positives
        @test PTX.requires_typed_wrapper(op, mods)
        @test PTX.typed_wrapper_only_rule(op, mods) !== nothing
    end

    # Prefix/token near misses must not expand the boundary accidentally.
    for (op, mods) in (
        (:wgmma, (:fence, :sync, :aligned)),
        (:wgmma, (:mma_asyncish, :sync, :aligned)),
        (:setp, (:duality, :eq, :s32)),
        (:shfl, (:sync, :idx, :b32, :predicate)),
        (:fabric, (:try_get, Symbol("mbarrier::report::fabric"))),
        (:tcgen050, (:ld, :sync, :aligned, :b32)),
    )
        @test !PTX.requires_typed_wrapper(op, mods)
        @test PTX.typed_wrapper_only_rule(op, mods) === nothing
    end
end

@testset "typed-wrapper-only boundary: generic misses fail before rendering" begin
    misses = (
        # Grouped matrix destination collapsed to one scalar before this fix.
        (Operation{:mma, (:sync, :aligned, :m16n8k16, :row, :col,
                          :f32, :f16, :f16, :f32)}(),
         (NTuple{4, Int32}, NTuple{2, UInt32}, NTuple{4, Float32})),
        # Illegal shape/type product used to invent a UInt16 result.
        (Operation{:wgmma, (:mma_async, :sync, :aligned,
                            :m64n7k16, :f32, :bf16, :bf16)}(),
         (NTuple{4, Float32}, UInt64, UInt64, Bool)),
        # ldmatrix has shape-dependent grouped results and requires a
        # reviewed shared-address carrier rather than scalar inference.
        (Operation{:ldmatrix, (:sync, :aligned, :m8n16, :x1, :shared,
                               :b8x16, :b6x16_p32)}(),
         (Int32,)),
        # tcgen05.ld.x2 must return two registers, not a scalar b32.
        (Operation{:tcgen05, (:ld, :sync, :aligned,
                              Symbol("16x64b"), :x2, :b32)}(),
         (Int32,)),
        # The marker token is an internal grouped-result selector, not a
        # modifier that the scalar formatter may print literally.
        (Operation{:shfl, (:sync, :idx, :b32, :pred)}(),
         (Int32, Int32, Int32, Int32)),
    )

    for (op, argts) in misses
        info = PTX.lowering(op, argts)
        @test info.tier === :forbidden
        @test info.asm === nothing
        @test endswith(String(info.method.file), "inst.jl")

        opsym, mods = typeof(op).parameters
        err = try
            build_call(opsym, mods, argts)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("requires an exact typed wrapper", sprint(showerror, err))
        @test occursin("raw", sprint(showerror, err))
        @test_throws ArgumentError format_call(op, Tuple{argts...})
    end

    # Exercise the shared @generated dispatch itself, not just its pure
    # renderer/reflection helpers. Every rule gets a concrete unmatched call;
    # the error must be raised while generating the body, before LLVM sees asm.
    dispatch_misses = (
        (Operation{:mma, (:sync, :aligned, :m16n8k16, :row, :col,
                          :f32, :f16, :f16, :f32)}(),
         (ntuple(_ -> Int32(0), Val(4)),
          ntuple(_ -> UInt32(0), Val(2)),
          ntuple(_ -> 0.0f0, Val(4)))),
        (Operation{:wgmma, (:mma_async, :sync, :aligned,
                            :m64n7k16, :f32, :bf16, :bf16)}(),
         (ntuple(_ -> 0.0f0, Val(4)), UInt64(0), UInt64(0), false)),
        (Operation{:tcgen05, (:ld, :sync, :aligned,
                              Symbol("16x64b"), :x2, :b32)}(), (Int32(0),)),
        (Operation{:shfl, (:sync, :idx, :b32, :pred)}(),
         (Int32(0), Int32(0), Int32(0), Int32(0))),
    )
    for (op, args) in dispatch_misses
        @test endswith(String(which(op, Tuple{typeof.(args)...}).file), "inst.jl")
        err = try
            op(args...)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("requires an exact typed wrapper", sprint(showerror, err))
    end
end

@testset "typed-wrapper-only boundary: exact methods remain authoritative" begin
    pS = Core.LLVMPtr{UInt64, PTX.AS.Shared}
    exact = (
        (Operation{:mma, (:sync, :aligned, :m16n8k16, :row, :col,
                          :f32, :bf16, :bf16, :f32)}(),
         (NTuple{4, UInt32}, NTuple{2, UInt32}, NTuple{4, Float32})),
        (Operation{:wgmma, (:mma_async, :sync, :aligned,
                            :m64n8k16, :f32, :bf16, :bf16)}(),
         (NTuple{4, Float32}, UInt64, UInt64, Bool)),
        (Operation{:tcgen05, (:shift, Symbol("cta_group::1"), :down)}(),
         (UInt32,)),
        (Operation{:tcgen05, (:alloc, Symbol("cta_group::1"),
                              :sync, :aligned, :b32)}(),
         (Core.LLVMPtr{UInt32, PTX.AS.Shared}, UInt32)),
        (Operation{:tcgen05, (:commit, Symbol("cta_group::1"),
                              Symbol("mbarrier::arrive::one"),
                              Symbol("shared::cluster"), :b64)}(),
         (Core.LLVMPtr{UInt64, PTX.AS.Shared},)),
        (Operation{:tcgen05, (Symbol("fence::before_thread_sync"),)}(), ()),
        (Operation{:shfl, (:sync, :idx, :b32, :pred)}(),
         (UInt32, UInt32, UInt32, UInt32)),
        (Operation{:mbarrier, (:test_wait, :report,
                               Symbol("phase_type::primary"), :shared, :b64)}(),
         (pS, UInt64)),
        # The report wrapper intentionally accepts any Integer state and
        # normalizes it to the required u64 carrier inside the method.
        (Operation{:mbarrier, (:test_wait, :report,
                               Symbol("phase_type::primary"), :shared, :b64)}(),
         (pS, Int32)),
    )

    for (op, argts) in exact
        @test which(op, argts).module === PTX
        info = PTX.lowering(op, argts)
        @test info.tier in (:intrinsic, :asm)
        @test info.tier !== :forbidden
    end
end

@testset "typed-wrapper-only boundary: exact tcgen05 lifecycle surfaces" begin
    # Independent manifest of the six formerly-generic calls used by the
    # Blackwell ptxas tier. Do not derive this from wrapper methods: every
    # opcode spelling, carrier schema, and lowering tier is reviewed here.
    cg1 = Symbol("cta_group::1")
    cg2 = Symbol("cta_group::2")
    pS32 = Core.LLVMPtr{UInt32, PTX.AS.Shared}
    pS64 = Core.LLVMPtr{UInt64, PTX.AS.Shared}
    pG32 = Core.LLVMPtr{UInt32, PTX.AS.Global}
    pG64 = Core.LLVMPtr{UInt64, PTX.AS.Global}

    surfaces = (
        ((:alloc, cg1, :sync, :aligned, :b32), (pS32, UInt32),
         :intrinsic, "llvm.nvvm.tcgen05.alloc.shared.cg1",
         ((UInt32, UInt32), (pG32, UInt32), (pS64, UInt32),
          (pS32, Int32), (pS32,), (pS32, UInt32, UInt32))),
        ((:alloc, cg2, :sync, :aligned, :b32), (pS32, UInt32),
         :intrinsic, "llvm.nvvm.tcgen05.alloc.shared.cg2",
         ((UInt32, UInt32), (pG32, UInt32), (pS64, UInt32),
          (pS32, Int32), (pS32,), (pS32, UInt32, UInt32))),
        ((:commit, cg1, Symbol("mbarrier::arrive::one"),
          Symbol("shared::cluster"), :b64), (pS64,),
         :intrinsic, "llvm.nvvm.tcgen05.commit.shared.cg1",
         ((UInt64,), (pG64,), (pS32,), (), (pS64, UInt32))),
        ((:commit, cg2, Symbol("mbarrier::arrive::one"),
          Symbol("shared::cluster"), :b64), (pS64,),
         :intrinsic, "llvm.nvvm.tcgen05.commit.shared.cg2",
         ((UInt64,), (pG64,), (pS32,), (), (pS64, UInt32))),
        ((Symbol("fence::before_thread_sync"),), (),
         :asm, "tcgen05.fence::before_thread_sync;",
         ((UInt32,), (pS32,), (UInt32, UInt32))),
        ((Symbol("fence::after_thread_sync"),), (),
         :asm, "tcgen05.fence::after_thread_sync;",
         ((UInt32,), (pS32,), (UInt32, UInt32))),
    )

    @test length(surfaces) == 6
    for (mods, good_argts, expected_tier, evidence, bad_argtypes) in surfaces
        op = Operation{:tcgen05, mods}()
        @test which(op, good_argts).module === PTX
        info = PTX.lowering(op, good_argts)
        @test info.tier === expected_tier
        @test info.rettype === Nothing
        if expected_tier === :intrinsic
            @test info.intrinsics == [evidence]
        else
            @test isempty(info.intrinsics)
        end

        for argts in bad_argtypes
            @test endswith(String(which(op, argts).file), "inst.jl")
            bad = PTX.lowering(op, argts)
            @test bad.tier === :forbidden
            @test bad.asm === nothing
            @test isempty(bad.intrinsics)
            @test_throws ArgumentError build_call(:tcgen05, mods, argts)
            @test_throws ArgumentError format_call(op, Tuple{argts...})
        end
    end
end

@testset "typed-wrapper-only boundary: tcgen05 fences survive optimization" begin
    llvm = sprint() do io
        code_llvm(io, _tcgen05_thread_fence_optimizer_probe, Tuple{};
                  optimize = true, raw = true, debuginfo = :none,
                  dump_module = true)
    end
    for spelling in ("tcgen05.fence::before_thread_sync;",
                     "tcgen05.fence::after_thread_sync;")
        @test length(findall(spelling, llvm)) == 1
    end
    @test length(collect(eachmatch(r"asm sideeffect", llvm))) == 2
    @test length(findall("~{memory}", llvm)) == 2
end

@testset "typed-wrapper-only boundary: raw is an explicit structural escape" begin
    # RAW_CONTRACT is conservative (sideeffect + memory clobber + convergent).
    # It remains available because these boundaries are structural. This is
    # deliberately different from CC.CF, whose hidden state forbids raw too.
    cases = (
        (:mma, (:sync, :aligned, :m16n8k16, :row, :col,
                :f32, :f16, :f16, :f32),
         (NTuple{4, UInt32}, NTuple{2, UInt32}, NTuple{4, Float32})),
        (:wgmma, (:mma_async, :sync, :aligned,
                  :m64n7k16, :f32, :bf16, :bf16),
         (NTuple{4, Float32}, UInt64, UInt64, Bool)),
        (:tcgen05, (:ld, :sync, :aligned, Symbol("16x64b"), :x2, :b32),
         (Int32,)),
        (:shfl, (:sync, :idx, :b32, :pred),
         (Int32, Int32, Int32, Int32)),
    )

    for (op, mods, argts) in cases
        spec = build_call(op, mods, argts; raw = true)
        @test spec.side_effects
        @test spec.convergent
        @test occursin("~{memory}", spec.constraints)
        @test startswith(spec.asm, PTX.build_head(op, mods))

        raw = RawOperation{op, mods}()
        info = PTX.lowering(raw, argts)
        @test info.tier === :chain_asm
        @test info.asm == spec.asm
    end

    # Ordinary scalar chains and the intentionally generic WGMMA sync forms
    # remain unchanged.
    @test PTX.lowering(ptx"add.f32", (Float32, Float32)).tier === :chain_asm
    @test format_call(ptx"add.f32", Tuple{Float32, Float32}) ==
        "add.f32 \$0, \$1, \$2;"
    @test PTX.lowering(ptx"wgmma.fence.sync.aligned", ()).tier === :chain_asm
    @test format_call(ptx"wgmma.fence.sync.aligned", Tuple{}) ==
        "wgmma.fence.sync.aligned;"

    @test PTX.lowering(ptx"add.cc.u32"raw,
                       (UInt32, UInt32)).tier === :forbidden

    # A FormContract value is not authorization to cross the typed boundary:
    # tcgen05's ordinary contract is bit-identical to RAW_CONTRACT. Raw mode is
    # a distinct signal, and cannot be combined with a supplied contract.
    tcmods = (:ld, :sync, :aligned, Symbol("16x64b"), :x2, :b32)
    tcargs = (Int32,)
    @test PTX.form_contract(:tcgen05, tcmods) === PTX.RAW_CONTRACT
    @test_throws ArgumentError build_call(:tcgen05, tcmods, tcargs;
                                          contract = PTX.form_contract(:tcgen05, tcmods))
    @test_throws ArgumentError build_call(:tcgen05, tcmods, tcargs;
                                          raw = true, contract = PTX.RAW_CONTRACT)
end
