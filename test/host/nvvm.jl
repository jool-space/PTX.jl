using PTX: NVVM
using PTX.NVVM: Intrinsic, intrinsic, isintrinsic, matching, overloaded,
                llvmtype, ptr, slot, anyptr, anyint, anyfloat, TABLE,
                synthesize, IntrinsicCall, @nvvm_str
using InteractiveUtils: code_llvm

# Host-optimizer contract probe. Both branches deliberately call the same
# recognized NVVM intrinsic with identical operands; without call-site
# convergence LLVM merges them before the branch. This is not executed on a
# GPU: divergent WMMA would itself violate PTX's uniform-warp requirement.
@noinline function _wmma_convergence_probe(c::Bool, a::Float64, b::Float64,
                                           x::Float64, y::Float64)
    if c
        r = nvvm"wmma.m8n8k4.mma.row.col.f64"(a, b, x, y)
        return r[1] + 1.0
    else
        r = nvvm"wmma.m8n8k4.mma.row.col.f64"(a, b, x, y)
        return r[2] + 2.0
    end
end

const SIDE_EFFECTING_NOMEM_EXPECTED = (
    "llvm.nvvm.griddepcontrol.launch.dependents",
    "llvm.nvvm.griddepcontrol.wait",
    "llvm.nvvm.nanosleep",
    "llvm.nvvm.pm.event.mask",
    "llvm.nvvm.setmaxnreg.dec.sync.aligned.u32",
    "llvm.nvvm.setmaxnreg.inc.sync.aligned.u32",
    "llvm.nvvm.tcgen05.fence.after.thread.sync",
    "llvm.nvvm.tcgen05.fence.before.thread.sync",
)

# The eight IntrNoMem + IntrHasSideEffects records all return void. Keep one
# direct tier-2 call to every record in a single host-only optimizer probe:
# these calls are inspected as LLVM and never executed. On an in-process LLVM
# that does not recognize a newer NVVM intrinsic, the declaration emitted by
# PTX.jl is its only semantic contract.
@noinline function _nomem_sideeffects_optimizer_probe(delay::UInt32)
    nvvm"griddepcontrol.launch.dependents"()
    nvvm"griddepcontrol.wait"()
    nvvm"nanosleep"(delay)
    nvvm"pm.event.mask"(Val(1))
    nvvm"setmaxnreg.dec.sync.aligned.u32"(Val(24))
    nvvm"setmaxnreg.inc.sync.aligned.u32"(Val(32))
    nvvm"tcgen05.fence.after.thread.sync"()
    nvvm"tcgen05.fence.before.thread.sync"()
    return nothing
end

# Host-only optimizer probe for return-position registry metadata. Cluster
# nctaid is deliberately newer than the in-process LLVM carried by the oldest
# supported Julia, so the synthesized declaration/call contract remains
# load-bearing instead of relying on intrinsic canonicalization.
@noinline _nvvm_return_contract_probe() =
    nvvm"read.ptx.sreg.cluster.nctaid.x"()

const RETURN_RANGE_EXPECTED = let
    names = String[]
    for family in ("cluster.ctaid", "cluster.nctaid", "clusterid", "ctaid",
                   "nclusterid", "nctaid", "ntid", "tid"),
        component in ("w", "x", "y", "z")
        push!(names, "llvm.nvvm.read.ptx.sreg.$family.$component")
    end
    append!(names, ("llvm.nvvm.read.ptx.sreg.laneid",
                    "llvm.nvvm.read.ptx.sreg.warpsize"))
    Set(names)
end

const RETURN_NOUNDEF_EXPECTED = let
    names = collect(RETURN_RANGE_EXPECTED)
    append!(names, (
        "llvm.nvvm.internal.addrspace.wrap",
        "llvm.nvvm.is_explicit_cluster",
        "llvm.nvvm.read.ptx.sreg.aggr_smem_size",
        "llvm.nvvm.read.ptx.sreg.clock",
        "llvm.nvvm.read.ptx.sreg.clock64",
        "llvm.nvvm.read.ptx.sreg.cluster.ctarank",
        "llvm.nvvm.read.ptx.sreg.cluster.nctarank",
        "llvm.nvvm.read.ptx.sreg.dynamic_smem_size",
        "llvm.nvvm.read.ptx.sreg.globaltimer",
        "llvm.nvvm.read.ptx.sreg.globaltimer.lo",
        "llvm.nvvm.read.ptx.sreg.gridid",
        "llvm.nvvm.read.ptx.sreg.nsmid",
        "llvm.nvvm.read.ptx.sreg.nwarpid",
        "llvm.nvvm.read.ptx.sreg.smid",
        "llvm.nvvm.read.ptx.sreg.total_smem_size",
        "llvm.nvvm.read.ptx.sreg.warpid",
    ))
    append!(names, ("llvm.nvvm.read.ptx.sreg.envreg$i" for i in 0:31))
    append!(names, ("llvm.nvvm.read.ptx.sreg.lanemask.$suffix"
                    for suffix in ("eq", "ge", "gt", "le", "lt")))
    append!(names, ("llvm.nvvm.read.ptx.sreg.pm$i" for i in 0:3))
    Set(names)
end

# The registry's contract: the committed table
# is the backend's intrinsic surface, queryable, with no silent gaps. The
# extraction itself is conformance-checked against the llc binary's name
# table at generation time (gen/extract_intrinsics.sh); these tests pin the
# Julia-side representation against independently hand-verified facts from
# the original llc experiments and the validation spikes.

@testset "table shape" begin
    # exact agreement with the 22.1.7 llc name table, established at
    # extraction; a regenerated table that drifts in count means tblgen
    # skew (gen/ must run tblgen at the backend's exact version)
    @test length(TABLE) == 2569
    @test NVVM.BACKEND_LLVM_VERSION == v"22.1.7"
    @test all(k == i.name for (k, i) in TABLE)
end

@testset "return range and noundef inventory is closed" begin
    ranged = Set(i.name for i in values(TABLE)
                 if any(entry -> entry[1] == 0, i.ranges))
    noundef = Set(i.name for i in values(TABLE)
                  if (0, :noundef) in i.argattrs)

    @test ranged == RETURN_RANGE_EXPECTED
    @test noundef == RETURN_NOUNDEF_EXPECTED
    @test length(ranged) == 34
    @test length(noundef) == 91
    @test ranged ⊆ noundef

    # Every current range is one scalar-i32 interval. The emitter deliberately
    # fails closed if a future backend introduces another return shape.
    for name in ranged
        i = intrinsic(name)
        @test i.ret == (:i32,)
        @test count(entry -> entry[1] == 0, i.ranges) == 1
    end
end

@testset "side-effecting no-memory inventory is closed" begin
    # Independent reviewed inventory of every IntrinsicsNVVM.td record whose
    # properties combine IntrNoMem with IntrHasSideEffects. Do not derive the
    # expected names from an emitter constant: a registry regeneration must
    # force a semantic review before this optimizer boundary can broaden.
    expected = Set(SIDE_EFFECTING_NOMEM_EXPECTED)
    actual = Set(i.name for i in values(TABLE)
                 if :nomem in i.props && :sideeffects in i.props)

    @test actual == expected
    @test length(actual) == 8
    @test all(isempty(intrinsic(name).ret) for name in expected)

    for name in expected
        i = intrinsic(name)
        @test NVVM.memory_attr(i.props) === nothing
        attrs = NVVM.fnattrs(i)
        @test !occursin(r"\breadnone\b", attrs)
        @test !occursin("memory(none)", attrs)
    end

    # Control: ordinary pure no-memory arithmetic keeps the precise attribute.
    pure = intrinsic("llvm.nvvm.add.rn.f")
    expected_nomem = Base.libllvm_version < v"16" ? "readnone" : "memory(none)"
    @test NVVM.memory_attr(pure.props) == expected_nomem
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
    @test occursin(r"call i64 @\"llvm\.nvvm\.mbarrier\.arrive\.expect\.tx\.scope\.cta\.space\.cta\"\([^\n]+\) #2", s.ir)
    @test occursin("attributes #0 = { convergent nomerge nounwind nocallback }", s.ir)
    @test occursin("attributes #1 = { alwaysinline convergent }", s.ir)
    @test occursin("attributes #2 = { convergent nomerge }", s.ir)
    @test s.rettype == UInt64
    @test s.tupletype == Tuple{Core.LLVMPtr{Int64,3}, UInt32}
    @test s.runtime == [1, 2]
end

@testset "synthesize: stored return contracts" begin
    ranged = intrinsic("llvm.nvvm.read.ptx.sreg.tid.x")
    s = synthesize(ranged.name, ())
    @test occursin(
        "declare noundef i32 @\"llvm.nvvm.read.ptx.sreg.tid.x\"() #0", s.ir)
    @test occursin(
        "call noundef i32 @\"llvm.nvvm.read.ptx.sreg.tid.x\"(), !range !0", s.ir)
    @test occursin("!0 = !{ i32 0, i32 1024 }", s.ir)

    # A noundef-only return gets the inline return attribute without inventing
    # range metadata. A result with neither property remains untouched.
    noundef_only = synthesize("llvm.nvvm.read.ptx.sreg.aggr_smem_size", ())
    @test occursin("declare noundef i32", noundef_only.ir)
    @test occursin("call noundef i32", noundef_only.ir)
    @test !occursin("!range", noundef_only.ir)

    control = synthesize("llvm.nvvm.activemask", ())
    @test !occursin("declare noundef", control.ir)
    @test !occursin("call noundef", control.ir)
    @test !occursin("!range", control.ir)

    # Exhaust every callable inventory member, not just representatives. The
    # sole exception has a return-only overload slot and is already rejected
    # by the tier-2 ABI before any declaration can be generated.
    for name in setdiff(RETURN_NOUNDEF_EXPECTED,
                        Set(("llvm.nvvm.internal.addrspace.wrap",)))
        member = intrinsic(name)
        emitted = synthesize(name, ()).ir
        @test occursin("declare noundef ", emitted)
        @test occursin("call noundef ", emitted)
        if name in RETURN_RANGE_EXPECTED
            _, lo, hi = only(entry for entry in member.ranges
                             if entry[1] == 0)
            @test occursin("!range !0", emitted)
            @test occursin("!0 = !{ i32 $lo, i32 $hi }", emitted)
        else
            @test !occursin("!range", emitted)
        end
    end
    wrap_error = try
        synthesize("llvm.nvvm.internal.addrspace.wrap",
                   (Core.LLVMPtr{UInt8,0},))
        ""
    catch err
        sprint(showerror, err)
    end
    @test occursin("appears only in the return type", wrap_error)

    # Version and registry-shape negatives exercise the fail-closed helper
    # without requiring unsupported LLVM parsers in CI.
    contract = NVVM._return_contract(ranged, "i32"; llvm_version=v"18")
    @test contract.call_suffix == ", !range !0"
    @test contract.metadata == "!0 = !{ i32 0, i32 1024 }"
    @test_throws ErrorException NVVM._return_contract(
        ranged, "i32"; llvm_version=v"14")

    void_noundef = Intrinsic("test.void.noundef", (), (), (), (), (), (),
                             ((0, :noundef),))
    @test_throws ErrorException NVVM._return_contract(void_noundef, "void")
    unknown_attr = Intrinsic("test.unknown.return.attr", (:i32,), (), (), (),
                             (), (), ((0, :readonly),))
    @test_throws ErrorException NVVM._return_contract(unknown_attr, "i32")
    float_range = Intrinsic("test.float.range", (:f32,), (), (), (),
                            ((0, 0, 1),), (), ())
    @test_throws ErrorException NVVM._return_contract(float_range, "float")
    split_range = Intrinsic("test.split.range", (:i32,), (), (), (),
                            ((0, 0, 2), (0, 4, 6)), (), ())
    @test_throws ErrorException NVVM._return_contract(split_range, "i32")
end

@testset "optimized host LLVM retains stored return contracts" begin
    llvm = sprint() do io
        code_llvm(io, _nvvm_return_contract_probe, Tuple{};
                  optimize=true, raw=true, debuginfo=:none, dump_module=true)
    end
    name = "@llvm.nvvm.read.ptx.sreg.cluster.nctaid.x"
    calls = [String(line) for line in eachline(IOBuffer(llvm))
             if occursin(" call ", line) && occursin(name, line)]
    @test length(calls) == 1
    if length(calls) == 1
        call = only(calls)
        @test occursin(r"\bcall\s+noundef\s+i32\b", call)
        range_ref = match(r"!range !([0-9]+)", call)
        @test range_ref !== nothing
        if range_ref !== nothing
            node = "!$(range_ref.captures[1]) = !{i32 1, i32 -2147483648}"
            # LLVM prints i32 2^31 canonically as its signed spelling.
            @test any(line -> strip(line) == node,
                      eachline(IOBuffer(llvm)))
        end
    end
end

@testset "synthesize: WMMA convergence is a call-site contract" begin
    # These representatives cover the emitter's scalar, aggregate, and void
    # branches and WMMA's read/none/write declaration-memory classes.
    cases = [
        ("llvm.nvvm.wmma.m8n8k4.load.a.col.f64",
         (Core.LLVMPtr{UInt8,0},), 1),
        ("llvm.nvvm.wmma.m8n8k4.mma.row.col.f64",
         (Float64, Float64, Float64, Float64), 2),
        ("llvm.nvvm.wmma.m8n8k4.store.d.col.f64",
         (Core.LLVMPtr{UInt8,0}, Float64, Float64), 0),
    ]
    for (name, argtypes, nret) in cases
        i = intrinsic(name)
        s = synthesize(name, argtypes)
        calls = [String(line) for line in eachline(IOBuffer(s.ir))
                 if occursin(" call ", line) && occursin("@\"$name", line)]

        @test length(i.ret) == nret
        @test length(calls) == 1
        if length(calls) == 1
            @test endswith(strip(only(calls)), "#2")
        end
        @test NVVM.callsiteattrs(i) == "convergent nomerge"
        @test occursin("attributes #0 = { $(NVVM.fnattrs(i)) }", s.ir)
        @test occursin("attributes #1 = { alwaysinline convergent }", s.ir)
        @test occursin("attributes #2 = { convergent nomerge }", s.ir)
    end

    # `tcgen05.mma` has single-thread issue semantics and must not inherit the
    # WMMA overlay merely because its name also contains `mma`.
    tcgen = synthesize("llvm.nvvm.tcgen05.mma.shared",
        (Core.LLVMPtr{UInt32,6}, UInt64, UInt64, UInt32, Bool,
         Val{2}, Val{1}, Val{0}))
    @test occursin("attributes #1 = { alwaysinline }", tcgen.ir)
    @test !occursin("attributes #2", tcgen.ir)
end

@testset "optimized WMMA calls retain their branch-local convergence sites" begin
    llvm = sprint() do io
        code_llvm(io, _wmma_convergence_probe,
                  Tuple{Bool,Float64,Float64,Float64,Float64};
                  optimize=true, raw=true, debuginfo=:none, dump_module=true)
    end
    name = "@llvm.nvvm.wmma.m8n8k4.mma.row.col.f64"
    calls = [String(line) for line in eachline(IOBuffer(llvm))
             if occursin(" call ", line) && occursin(name, line)]

    # The unfixed emitter produces one hoisted call here on LLVM 18. Every
    # retained call must reference a group carrying both optimizer barriers;
    # checking an unrelated module-level group would be a false positive.
    @test length(calls) == 2
    groups = Dict{String,String}()
    for line in eachline(IOBuffer(llvm))
        m = match(r"^attributes #([0-9]+) = \{([^}]*)\}", strip(line))
        m === nothing || (groups[m.captures[1]] = m.captures[2])
    end
    for call in calls
        m = match(r" #([0-9]+)(?:,|$)", strip(call))
        @test m !== nothing
        m === nothing && continue
        attrs = get(groups, m.captures[1], "")
        @test occursin(r"\bconvergent\b", attrs)
        @test occursin(r"\bnomerge\b", attrs)
    end
end

@testset "side-effecting no-memory calls survive optimization" begin
    llvm = sprint() do io
        code_llvm(io, _nomem_sideeffects_optimizer_probe, Tuple{UInt32};
                  optimize=true, raw=true, debuginfo=:none, dump_module=true)
    end
    # Exactly one call to each closed-world member must remain. Looking only
    # for declarations would reproduce the old false-positive: the optimizer
    # retained declarations after deleting the observable void calls.
    for name in SIDE_EFFECTING_NOMEM_EXPECTED
        calls = [line for line in eachline(IOBuffer(llvm))
                 if occursin(" call ", line) && occursin("@$name", line)]
        @test length(calls) == 1
    end
end

@testset "synthesize: mangling and aggregate repack (ldmatrix)" begin
    # Typed-pointer LLVMs (≤ 16 / Julia ≤ 1.11) mangle pointer overloads
    # with the pointee; opaque with the address space alone.
    psuf = Base.libllvm_version < v"17" ? "i8" : ""
    # LLVM 15 (Julia 1.10) predates memory(...) — legacy spelling there.
    argmem_read = Base.libllvm_version < v"16" ?
        "argmemonly readonly" : "memory(argmem: read)"

    s = synthesize("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16",
                   (Core.LLVMPtr{UInt16,3},))
    @test occursin("@\"llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3$psuf\"", s.ir)
    @test occursin(argmem_read, s.ir)
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
    # remangler normalizes to
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
