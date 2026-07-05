using PTX: NVVM
using PTX.NVVM: Intrinsic, intrinsic, isintrinsic, matching, overloaded,
                llvmtype, ptr, slot, anyptr, anyint, anyfloat, TABLE,
                synthesize, IntrinsicCall, @nvvm_str

# The registry's contract (DESIGN.md, "The registry"): the committed table
# is the backend's intrinsic surface, queryable, with no silent gaps. The
# extraction itself is conformance-checked against the llc binary's name
# table at generation time (gen/extract_intrinsics.sh); these tests pin the
# Julia-side representation against independently hand-verified facts from
# the original llc experiments and the validation spikes.

@testset "table shape" begin
    # exact agreement with the 22.1.7 llc name table, established at
    # extraction; a regenerated table that drifts in count means tblgen
    # skew (see CONCERNS.md, "Obtaining llvm-tblgen")
    @test length(TABLE) == 2569
    @test NVVM.BACKEND_LLVM_VERSION == v"22.1.7"
    @test all(k == i.name for (k, i) in TABLE)
end

@testset "hand-verified signatures" begin
    # mbarrier: verified by the original llc experiment and the convergence
    # spike's pipeline
    mb = intrinsic("llvm.nvvm.mbarrier.arrive.expect.tx.scope.cta.space.cta")
    @test mb.ret == (:i64,)
    @test mb.params == (ptr(3), :i32)
    @test :convergent in mb.props
    @test !overloaded(mb)

    # ldmatrix: verified on hardware by spikes/aggregate_return.jl;
    # overloaded on the pointer address space → mangled callsite name
    ld = intrinsic("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16")
    @test ld.ret == (:i32, :i32, :i32, :i32)
    @test ld.params == (anyptr,)
    @test overloaded(ld)
    @test (1, :readonly) in ld.argattrs && (1, :nocapture) in ld.argattrs

    # activemask: the convergence spike's subject
    am = intrinsic("llvm.nvvm.activemask")
    @test am.ret == (:i32,) && isempty(am.params)
    @test :convergent in am.props

    # tcgen05.mma.tensor: immediate operands with ranges and .td names —
    # the metadata that drives immarg validation and generated docs
    tc = intrinsic("llvm.nvvm.tcgen05.mma.tensor")
    @test tc.immargs == (6, 7, 8)
    @test (6, 0, 4) in tc.ranges        # kind ∈ [0, 4)
    @test (7, 1, 3) in tc.ranges        # cta_group ∈ [1, 3)
    @test (6, :kind) in tc.argnames && (7, :cta_group) in tc.argnames

    # sreg range metadata lands on the return slot (position 0)
    tid = intrinsic("llvm.nvvm.read.ptx.sreg.tid.x")
    @test (0, 0, 1024) in tid.ranges
    @test (0, :noundef) in tid.argattrs

    # overload-slot matching: atomics repeat slot 0 for the value operand
    at = intrinsic("llvm.nvvm.atomic.add.gen.f.cta")
    @test at.ret == (anyfloat,)
    @test at.params == (anyptr, slot(0))
    @test overloaded(at)
end

@testset "block-scaled mma is in the table" begin
    # the GB10 sm_121a MXFP8 path — the reason tier 2 pays for itself
    @test isintrinsic("llvm.nvvm.mma.block.scale.m16n8k32.row.col.mxf8f6f4.f32.e4m3.e4m3.f32.ue8m0")
    @test length(matching("llvm.nvvm.mma.block.scale.")) == 54
    # wgmma.mma_async is NOT upstream; only its fences are — the asm tier
    # stays load-bearing for it
    @test isempty(matching("llvm.nvvm.wgmma.mma_async"))
    @test isintrinsic("llvm.nvvm.wgmma.commit_group.sync.aligned")
end

@testset "llvmtype rendering" begin
    @test llvmtype(:i32) == "i32"
    @test llvmtype(:f16) == "half"
    @test llvmtype(:bf16) == "bfloat"
    @test llvmtype(:v2f16) == "<2 x half>"
    @test llvmtype(:v2bf16) == "<2 x bfloat>"
    @test llvmtype(:v4i32) == "<4 x i32>"
    @test llvmtype(:v128i32) == "<128 x i32>"
    # Typed spellings by design — parse on Julia ≤ 1.11's typed-pointer
    # device context, auto-upgrade to opaque on ≥ 1.12 (see NVVM.llvmtype).
    @test llvmtype(ptr(0)) == "i8*"
    @test llvmtype(ptr(3)) == "i8 addrspace(3)*"
    @test llvmtype(ptr(6)) == "i8 addrspace(6)*"   # tensor memory
    @test_throws ErrorException llvmtype(:notatype)
    @test_throws ErrorException llvmtype(anyptr)   # unbound overload slot
    @test_throws ErrorException llvmtype(slot(0))

    # every token in the table renders (or is an overload token by design):
    # no entry can reach emission and then discover an unmapped type
    for i in values(TABLE), t in (i.ret..., i.params...)
        if t isa Symbol || t isa NVVM.PtrTok
            @test llvmtype(t) isa String
        end
    end
end

@testset "miss errors carry suggestions" begin
    @test !isintrinsic("llvm.nvvm.mbarrier.arrive.expect.tx.scope.gpu.space.cta")
    err = try
        intrinsic("llvm.nvvm.mbarrier.arrive.expect.tx.scope.gpu.space.cta")
        nothing
    catch e
        sprint(showerror, e)
    end
    @test occursin("not in the backend's intrinsic table", err)
    @test occursin("llvm.nvvm.mbarrier.arrive.expect.tx.scope.cta.space.cta", err)

    err = try
        intrinsic("llvm.foo.bar"); nothing
    catch e
        sprint(showerror, e)
    end
    @test occursin("no registered names share a prefix", err)
end

# --- Synthesis (emit.jl) ----------------------------------------------------
#
# Host-side checks of the llvmcall IR the @generated path splices. The same
# IR was validated against the real 22.1.7 llc during development (every
# case below selected its instruction); the ptxas/ and gpu/ tiers re-prove
# that continuously through the actual pipeline. Here we pin the *text*: the
# declaration, the attribute groups, mangling, glue, and repacking.

@testset "synthesize: plain signature, convergent attrs" begin
    s = synthesize("llvm.nvvm.mbarrier.arrive.expect.tx.scope.cta.space.cta",
                   (Core.LLVMPtr{Int64,3}, UInt32))
    @test occursin("declare i64 @\"llvm.nvvm.mbarrier.arrive.expect.tx.scope.cta.space.cta\"(i8 addrspace(3)*, i32) #0", s.ir)
    @test occursin("attributes #0 = { convergent nomerge nounwind nocallback }", s.ir)
    @test occursin("attributes #1 = { alwaysinline }", s.ir)
    @test s.rettype == UInt64
    @test s.tupletype == Tuple{Core.LLVMPtr{Int64,3}, UInt32}
    @test s.runtime == [1, 2]
end

@testset "synthesize: mangling and aggregate repack (ldmatrix)" begin
    # Typed-pointer LLVMs (≤ 16 / Julia ≤ 1.11) mangle pointer overloads
    # with the pointee; opaque with the address space alone.
    psuf = Base.libllvm_version < v"17" ? "i8" : ""

    s = synthesize("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16",
                   (Core.LLVMPtr{UInt16,3},))
    @test occursin("@\"llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3$psuf\"", s.ir)
    @test occursin("memory(argmem: read)", s.ir)
    @test occursin("i8 addrspace(3)* readonly nocapture", s.ir)
    @test occursin("insertvalue [4 x i32]", s.ir)
    @test s.rettype == NTuple{4,UInt32}

    # the address space drives the suffix
    s = synthesize("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16",
                   (Core.LLVMPtr{UInt16,0},))
    @test occursin(".b16.p0$psuf\"", s.ir)
end

@testset "synthesize: two-slot mangle in canonical order (atomic)" begin
    psuf = Base.libllvm_version < v"17" ? "i8" : ""
    # ret slot (f32) precedes the pointer slot (p1) — the order llc's
    # remangler normalizes to (CONCERNS.md, mangling)
    s = synthesize("llvm.nvvm.atomic.add.gen.f.cta",
                   (Core.LLVMPtr{Float32,1}, Float32))
    @test occursin("@\"llvm.nvvm.atomic.add.gen.f.cta.f32.p1$psuf\"", s.ir)
    @test occursin("nocapture", s.ir)
    @test s.rettype == Float32

    # the value argument binds the ret slot — Float64 flips it to .f64
    s = synthesize("llvm.nvvm.atomic.add.gen.f.cta",
                   (Core.LLVMPtr{Float64,1}, Float64))
    @test occursin(".f64.p1$psuf\"", s.ir)
    @test s.rettype == Float64
end

@testset "synthesize: heterogeneous struct return (shfl pred variant)" begin
    s = synthesize("llvm.nvvm.shfl.sync.idx.i32p", (UInt32, UInt32, UInt32, UInt32))
    @test occursin("call { i32, i1 }", s.ir)              # intrinsic side
    @test occursin("define { i32, i8 } @entry", s.ir)     # Julia ABI side
    @test occursin("zext i1", s.ir)
    @test s.rettype == Tuple{UInt32,Bool}
end

@testset "synthesize: Bool <-> i1 glue (vote)" begin
    s = synthesize("llvm.nvvm.vote.all.sync", (UInt32, Bool))
    @test occursin("trunc i8 %a1 to i1", s.ir)            # param glue
    @test occursin("zext i1 %r to i8", s.ir)              # return glue
    @test occursin("define i8 @entry(i32 %a0, i8 %a1)", s.ir)
    @test s.rettype == Bool
end

@testset "synthesize: immediate operands" begin
    s = synthesize("llvm.nvvm.tcgen05.mma.shared",
                   (Core.LLVMPtr{UInt32,6}, UInt64, UInt64, UInt32, Bool,
                    Val{2}, Val{1}, Val{0}))
    # immargs spliced as literals, not entry parameters
    @test occursin("i32 2, i32 1, i32 0)", s.ir)
    @test s.runtime == [1, 2, 3, 4, 5]
    @test occursin("define void @entry(i8 addrspace(6)* %a0, i64 %a1, i64 %a2, i32 %a3, i8 %a4)", s.ir)

    # errors: missing Val, out-of-range, named operand label from the .td
    bad(args) = try synthesize("llvm.nvvm.tcgen05.mma.shared", args); ""
                catch e sprint(showerror, e) end
    @test occursin("immediate operand: pass Val(x)",
                   bad((Core.LLVMPtr{UInt32,6}, UInt64, UInt64, UInt32, Bool,
                        UInt32, Val{1}, Val{0})))
    @test occursin("outside the legal range [0, 3]",
                   bad((Core.LLVMPtr{UInt32,6}, UInt64, UInt64, UInt32, Bool,
                        Val{9}, Val{1}, Val{0})))
    named = try synthesize("llvm.nvvm.tcgen05.mma.tensor",
                           (Core.LLVMPtr{UInt32,6}, Core.LLVMPtr{UInt32,6},
                            UInt64, UInt32, Bool, UInt32, Val{1}, Val{0})); ""
            catch e sprint(showerror, e) end
    @test occursin("`kind`", named)   # operand name surfaces in the error
end

@testset "synthesize: rejections" begin
    msg(name, args) = try synthesize(name, args); ""
                      catch e sprint(showerror, e) end
    @test occursin("expects 4 arguments, got 3",
                   msg("llvm.nvvm.shfl.sync.idx.i32", (UInt32, UInt32, UInt32)))
    @test occursin("expected UInt32/Int32, got Float32",
                   msg("llvm.nvvm.shfl.sync.idx.i32", (UInt32, Float32, UInt32, UInt32)))
    @test occursin("expected Core.LLVMPtr{T,6}",
                   msg("llvm.nvvm.tcgen05.mma.shared",
                       (Core.LLVMPtr{UInt32,3}, UInt64, UInt64, UInt32, Bool,
                        Val{2}, Val{1}, Val{0})))
    # ldu's loaded-value type is a return-only overload slot — not inferable
    # from arguments (explicit slot binding is future work, if ever needed)
    @test occursin("appears only in the return type",
                   msg("llvm.nvvm.ldu.global.i", (Core.LLVMPtr{UInt32,1}, UInt32)))
    @test occursin("metadata-typed",
                   msg("llvm.nvvm.texsurf.handle", (Core.LLVMPtr{UInt8,1},)))
end

@testset "synthesize: pointer return placeholder" begin
    s = synthesize("llvm.nvvm.mapa.shared.cluster", (Core.LLVMPtr{Float32,3}, UInt32))
    @test s.rettype == Core.LLVMPtr{UInt8,7}
    @test occursin("declare i8 addrspace(7)*", s.ir)
end

@testset "nvvm\"\" macro" begin
    op = nvvm"llvm.nvvm.activemask"
    @test op isa IntrinsicCall{Symbol("llvm.nvvm.activemask")}
    @test nvvm"activemask" === op           # prefix is implied
    @test sprint(show, op) == "nvvm\"llvm.nvvm.activemask\""
    # unknown names die at macro expansion, with suggestions
    @test_throws LoadError @eval nvvm"llvm.nvvm.activemask.sync.warp"
end
