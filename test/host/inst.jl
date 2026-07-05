using PTX: build_call, format_call, Operation, SpecialReg, Chain, @mod_str

@testset "ptx\"...\" string macro" begin
    # Returns an Operation singleton parameterized by opcode and modifier tuple.
    @test typeof(ptx"add.f32") == Operation{:add, (:f32,)}
    @test typeof(ptx"cp.async") == Operation{:cp, (:async,)}

    # Bare opcode → empty modifier tuple.
    @test typeof(ptx"ret") == Operation{:ret, ()}

    # `::` goes in verbatim — segment is one Symbol with `::` in its name.
    @test typeof(ptx"shared::cta.b16") ==
          Operation{Symbol("shared::cta"), (:b16,)}
    @test typeof(ptx"mbarrier::complete_tx::bytes") ==
          Operation{Symbol("mbarrier::complete_tx::bytes"), ()}

    # Digit-leading modifiers — no escape needed.
    @test typeof(ptx"cp.async.bulk.tensor.3d") ==
          Operation{:cp, (:async, :bulk, :tensor, Symbol("3d"))}
    @test typeof(ptx"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32") ==
          Operation{:mma, (:sync, :aligned, :m16n8k32, :row, :col,
                           :f32, :e4m3, :e4m3, :f32)}

    # Empty / malformed chains caught at expansion.
    @test_throws LoadError @eval ptx""
    @test_throws LoadError @eval ptx"add..f32"
    @test_throws LoadError @eval ptx".add.f32"
end

@testset "ptx\"...\" string macro: \$ interpolation" begin
    dt = "u32"
    @test ptx"mov.$dt" === ptx"mov.u32"

    # Symbol values interpolate via Base string conversion.
    sym = :f32
    @test ptx"add.$sym" === ptx"add.f32"

    # Interpolated value containing dots produces multiple modifier parts.
    sp = "shared::cta"
    @test ptx"st.$sp.b32" === Operation{:st, (Symbol("shared::cta"), :b32)}()

    # $(...) form for non-identifier expressions.
    @test ptx"add.$(:f64)" === ptx"add.f64"

    # Identifier as opcode — first segment after split is the opcode.
    op = "mov"
    @test ptx"$op.u32" === ptx"mov.u32"

    # Errors at expansion / runtime split.
    @test_throws LoadError @eval ptx"mov.\$"
    let x = "u32"
        @test_throws ErrorException ptx"mov..$x"
    end

    # Glued `$(expr)` mid-segment with a value-known interp folds to the same
    # singleton the literal form produces. Critical for device kernels:
    # `@inline f(p, ::Val{N}) = ptx"...x$(N)..."(p)` must lower to a single
    # asm site with no runtime split/Symbol/concat. We assert (a) singleton
    # equivalence and (b) `@inferred` returns the concrete singleton type —
    # i.e. constant propagation through `*` saw all params.
    let glue(N) = ptx"ldmatrix.sync.aligned.m8n8.x$(N).shared.b16"
        @test glue(4) === ptx"ldmatrix.sync.aligned.m8n8.x4.shared.b16"
        @test glue(2) === ptx"ldmatrix.sync.aligned.m8n8.x2.shared.b16"
        @test glue(1) === ptx"ldmatrix.sync.aligned.m8n8.x1.shared.b16"
    end
    let f(::Val{N}) where {N} = ptx"foo.x$(N).bar"
        @test @inferred(f(Val(7))) === ptx"foo.x7.bar"
    end
    # Bare `$expr` between dots, value with no `.` → single Symbol modifier.
    let g(s::Symbol) = ptx"add.$s"
        @test g(:f16) === ptx"add.f16"
    end
end

@testset "mod\"...\" string macro" begin
    @test mod"f32" === Chain{(:f32,)}()
    @test mod"row.col.f32" === Chain{(:row, :col, :f32)}()
    # Empty form is the no-op chain.
    @test mod"" === Chain{()}()
    # Interpolation rejected.
    @test_throws LoadError @eval mod"f.\$x"
    # Malformed dotted form rejected.
    @test_throws LoadError @eval mod".f32"
    @test_throws LoadError @eval mod"f32."
    @test_throws LoadError @eval mod"a..b"
end

@testset "Operation/Chain `*` composition" begin
    # Operation * Chain extends modifiers; result is a singleton at compile time.
    @test ptx"mma.sync.aligned" * mod"m16n8k16" * mod"row.col" ===
          ptx"mma.sync.aligned.m16n8k16.row.col"

    # Chain * Chain.
    @test mod"row.col" * mod"f32.f16" === mod"row.col.f32.f16"

    # Operation * Symbol (helper-style, plain Symbol on right).
    @test ptx"mov" * :u32 === ptx"mov.u32"
    @test mod"" * :f32 === mod"f32"

    # mod"" is the right (and left) identity for *.
    op = ptx"mov.u32"
    @test op * mod"" === op
    @test mod"" * mod"" === mod""
    @test mod"f32" * mod"" === mod"f32"
    @test mod"" * mod"f32" === mod"f32"

    # Operation * Operation is intentionally undefined.
    @test_throws MethodError ptx"mov.u32" * ptx"add.f32"
end

@testset "build_call" begin
    spec = build_call(:cp, (:async, :bulk, :tensor, Symbol("3d"),
                            Symbol("shared::cta"), :global),
                      (UInt64, UInt64, Int32, UInt64))
    @test spec.asm == "cp.async.bulk.tensor.3d.shared::cta.global \$0, \$1, \$2, \$3;"
    @test spec.rettype === Nothing
    @test spec.side_effects == true   # cp prefix → nonpure
end

@testset "build_call: side-effecting opcodes must never classify pure" begin
    # The chain default's failure mode for a forgotten NONPURE entry is a
    # miscompile, not slowness (CONCERNS.md, "the chain default is
    # permissive by default"). These are the gaps found 2026-07-04 — each
    # was pure + clobber-free before.

    # multimem.st writes memory: nonpure, void (the dtype suffix is the
    # value being written), bracketed address.
    spec = build_call(:multimem, (:st, :relaxed, :sys, :global, :u32),
                      (Core.LLVMPtr{UInt32, 1}, UInt32))
    @test spec.side_effects == true
    @test spec.rettype === Nothing
    @test spec.asm == "multimem.st.relaxed.sys.global.u32 [\$0], \$1;"
    @test endswith(spec.constraints, "~{memory}")

    # multimem.ld_reduce returns a value — the trailing-dtype rule still
    # fires — but is nonpure and brackets its pointer.
    spec = build_call(:multimem,
                      (:ld_reduce, :relaxed, :sys, :global, :add, :u32),
                      (Core.LLVMPtr{UInt32, 1},))
    @test spec.side_effects == true
    @test spec.rettype === UInt32
    @test spec.asm ==
          "multimem.ld_reduce.relaxed.sys.global.add.u32 \$0, [\$1];"

    # nanosleep.u32: `.u32` is the duration operand's width, not a return —
    # a phantom $0 output makes ptxas reject with "Arguments mismatch".
    spec = build_call(:nanosleep, (:u32,), (UInt32,))
    @test spec.side_effects == true
    @test spec.rettype === Nothing
    @test spec.asm == "nanosleep.u32 \$0;"

    # trap / brkpt / pmevent: void control ops. Pure classification made
    # them legal to reorder and left them alive only by DCE conservatism.
    @test build_call(:trap, (), ()).side_effects == true
    @test build_call(:brkpt, (), ()).side_effects == true
    @test build_call(:pmevent, (), (Val{0},)).side_effects == true

    # Cache-control ops: nonpure + bracketed memory operand.
    spec = build_call(:discard, (:global, :L2,), (Core.LLVMPtr{UInt8, 1}, Val{128}))
    @test spec.side_effects == true
    @test spec.asm == "discard.global.L2 [\$0], 128;"
end

@testset "sreg whitelist targets exist in the backend registry" begin
    # The mov.u32-from-sreg fast path emits a tier-2 IntrinsicCall; a name
    # missing from the registry would error at first use. (The in-process
    # LLVM need not know these — see the NVVM_SREG_U32 comment.)
    for suffix in values(PTX.NVVM_SREG_U32)
        @test PTX.NVVM.isintrinsic("llvm.nvvm.read.ptx.sreg." * suffix)
    end
end

@testset "sreg\"...\" string macro + SpecialReg render" begin
    # Macro produces SpecialReg{Symbol("%name")}() for both naked and
    # %-prefixed input forms.
    @test sreg"tid.x"  === SpecialReg{Symbol("%tid.x")}()
    @test sreg"%tid.x" === SpecialReg{Symbol("%tid.x")}()
    @test sreg"cluster_ctarank" === SpecialReg{Symbol("%cluster_ctarank")}()
    @test sreg"lanemask_eq"     === SpecialReg{Symbol("%lanemask_eq")}()

    # Empty rejected at expansion.
    @test_throws LoadError @eval sreg""

    # Chain emits the symbol verbatim — preserves underscores in real names.
    @test build_call(:mov, (:u32,), (typeof(sreg"%tid.x"),)).asm ==
          "mov.u32 \$0, %tid.x;"
    @test build_call(:mov, (:u32,), (typeof(sreg"%cluster_ctarank"),)).asm ==
          "mov.u32 \$0, %cluster_ctarank;"
    @test build_call(:mov, (:u32,), (typeof(sreg"%lanemask_eq"),)).asm ==
          "mov.u32 \$0, %lanemask_eq;"
    # Mixed (real underscore + real dot) — only verbatim path gets it right.
    @test build_call(:mov, (:u32,), (typeof(sreg"%cluster_ctaid.x"),)).asm ==
          "mov.u32 \$0, %cluster_ctaid.x;"

    # Reading any SpecialReg forces side_effects=true.
    @test build_call(:mov, (:u32,), (typeof(sreg"%tid.x"),)).side_effects == true
end

@testset "infer_rettype + registry returns gate" begin
    # Sink forms carry `returns=false` in the form registry (src/forms.jl),
    # suppressing the trailing-dtype rule. `st.global.b32` ends in :b32 but
    # the value is being *written*, not returned — so rettype must be Nothing.
    @test PTX.infer_rettype(:st, (:global, :b32)) === Nothing
    @test PTX.infer_rettype(:red, (:global, :add, :u32)) === Nothing
    @test PTX.infer_rettype(:setmaxnreg, (:inc, :sync, :aligned, :u32)) === Nothing
    @test PTX.infer_rettype(:tensormap, (:replace, :tile, :global_address, :b1024, :b32)) === Nothing

    # Prefix overrides — tcgen05 mixes ops; only the three sink prefixes
    # carry returns=false.
    @test PTX.infer_rettype(:tcgen05,
        (:alloc, Symbol("cta_group::1"), :sync, :aligned, :b32)) === Nothing
    @test PTX.infer_rettype(:tcgen05,
        (:commit, Symbol("cta_group::1"), Symbol("mbarrier::arrive::one"),
         Symbol("shared::cluster"), :b64)) === Nothing
    @test PTX.infer_rettype(:tcgen05,
        (:relinquish_alloc_permit, Symbol("cta_group::1"), :sync, :aligned)) === Nothing

    # Negative — `:tcgen05, :ld, ...` has no returns=false override; the
    # trailing dtype rule fires and the rettype is the carrier of `:b32` → UInt32.
    @test PTX.infer_rettype(:tcgen05,
        (:ld, :sync, :aligned, Symbol("16x128b"), :x1, :b32)) === UInt32

    # First-segment match alone isn't enough — `(:tcgen05, :alloc)` shouldn't
    # match a chain that starts with a different second segment. The loop must
    # check every position; if it terminated early on op-match, this would
    # incorrectly return Nothing.
    @test PTX.infer_rettype(:tcgen05,
        (:dealloc, Symbol("cta_group::1"), :sync, :aligned, :b32)) === UInt32

    # nanosleep / multimem sinks suppress the trailing-dtype rule;
    # multimem.ld_reduce keeps it.
    @test PTX.infer_rettype(:nanosleep, (:u32,)) === Nothing
    @test PTX.infer_rettype(:multimem, (:st, :relaxed, :sys, :global, :u32)) === Nothing
    @test PTX.infer_rettype(:multimem, (:red, :relaxed, :sys, :global, :add, :u32)) === Nothing
    @test PTX.infer_rettype(:multimem,
        (:ld_reduce, :relaxed, :sys, :global, :add, :u32)) === UInt32

    # Direct check of the registry lookup: longest-prefix override wins,
    # non-matching prefixes fall back to the family default.
    @test PTX.form_contract(:st, (:global, :b32)).returns === false
    @test PTX.form_contract(:tcgen05,
        (:alloc, Symbol("cta_group::1"), :sync, :aligned, :b32)).returns === false
    @test PTX.form_contract(:tcgen05,
        (:dealloc, Symbol("cta_group::1"))).returns === true
    @test PTX.form_contract(:add, (:f32,)).returns === true
    @test PTX.form_contract(:add, (:f32,)).pure === true
    # Unregistered opcode → nothing (the chain default errors on it).
    @test PTX.form_contract(:frobnicate, ()) === nothing
end

@testset "blessing boundary: unregistered chains error, raw tier opts in" begin
    # The chain default refuses opcodes absent from the form registry — the
    # optimizer promises must be reviewed per form, not guessed (the old
    # permissive default's failure mode was a miscompile, not slowness).
    err = try
        PTX.build_call(:frobnicate, (:x2,), (UInt32,)); ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("form registry", err)
    @test occursin("ptx\"...\"raw", err)
    # Calling the Operation errors the same way (generator-time).
    @test_throws Exception PTX.Operation{:frobnicate, (:x2,)}()(UInt32(1))

    # ptx"..."raw gets RAW_CONTRACT: sideeffect + clobber + convergent,
    # pointer operands bracketed, trailing-dtype return inference.
    r = ptx"frobnicate.global.u32"raw
    @test r isa PTX.RawOperation{:frobnicate, (:global, :u32)}
    spec = PTX.build_call(:frobnicate, (:global, :u32),
                          (Core.LLVMPtr{UInt32, 1}, UInt32);
                          contract = PTX.RAW_CONTRACT)
    @test spec.side_effects == true
    @test spec.convergent == true
    @test endswith(spec.constraints, "~{memory}")
    @test spec.asm == "frobnicate.global.u32 \$0, [\$1], \$2;"
    @test spec.rettype === UInt32

    # Raw composes like Operation.
    @test ptx"frobnicate.x2"raw * :u32 ===
          PTX.RawOperation{:frobnicate, (:x2, :u32)}()
    @test ptx"frobnicate"raw * mod"x2.u32" ===
          PTX.RawOperation{:frobnicate, (:x2, :u32)}()

    # Raw is static-only and validates its flag.
    @test_throws LoadError @eval ptx"frobnicate.$x"raw
    @test_throws LoadError @eval ptx"frobnicate.x"nosuchflag
end

@testset "collective chain forms carry convergent nomerge (registry)" begin
    # Registered convergent families route through convergent_asm_ir — the
    # IR-level tripwire (llc ignores the attribute; goldens can't observe
    # its loss). This retires the pre-registry residual where chain-default
    # collectives got sideeffect but NOT convergent.
    for (op, argts) in [
        (ptx"vote.sync.ballot.b32", (Bool, UInt32)),
        (ptx"bar.sync",             (Val{0},)),
        (ptx"redux.sync.add.s32",   (Int32, UInt32)),
        (ptx"activemask.b32",       ()),
    ]
        ci, _ = first(Base.code_typed(op, argts))
        @test occursin("convergent nomerge", string(ci))
    end
    # Non-collective side effects stay on plain @asmcall (no attr group).
    ci, _ = first(Base.code_typed(ptx"membar.gl", ()))
    @test !occursin("convergent", string(ci))
end

@testset "property notation: composition + completion" begin
    # Dot chains compose in the type domain exactly like `*` — same
    # singleton, device-safe folding.
    @test ptx"cvt".rn.f32.f16 === ptx"cvt.rn.f32.f16"
    @test ptx"add".f32 === ptx"add.f32"
    @test ptx"st".var"shared::cta".b32 ===
          Operation{:st, (Symbol("shared::cta"), :b32)}()
    @test mod"row".col === mod"row.col"
    @test ptx"frobnicate.x2"raw.u32 ===
          PTX.RawOperation{:frobnicate, (:x2, :u32)}()
    let f() = ptx"cvt".rn.f32.f16
        @test @inferred(f()) === ptx"cvt.rn.f32.f16"
    end

    # propertynames suggests continuations: registry override prefixes...
    @test :st in propertynames(ptx"multimem")
    @test :alloc in propertynames(ptx"tcgen05")
    # ...and NVVM intrinsic-table segments (best effort).
    @test :cluster in propertynames(ptx"barrier")
    @test :arrive in propertynames(ptx"barrier".cluster)
end
