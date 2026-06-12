using PTX: NVVM
using PTX.NVVM: Intrinsic, intrinsic, isintrinsic, matching, overloaded,
                llvmtype, ptr, slot, anyptr, anyint, anyfloat, TABLE

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
    @test llvmtype(ptr(0)) == "ptr"
    @test llvmtype(ptr(3)) == "ptr addrspace(3)"
    @test llvmtype(ptr(6)) == "ptr addrspace(6)"   # tensor memory
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
