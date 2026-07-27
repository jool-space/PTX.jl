using PTX: Operation
using PTX.NVVM: NVVM

const EXPECTED_TENSORMAP_FENCE_SCOPES = (:cta, :cluster, :gpu, :sys)
const EXPECTED_TENSORMAP_FENCES = let forms = NamedTuple[]
    for sem in (:acquire, :release), scope in EXPECTED_TENSORMAP_FENCE_SCOPES
        intr = "llvm.nvvm.fence.proxy.tensormap_generic.$sem.$scope"
        push!(forms, (; sem, scope, intr))
    end
    Tuple(forms)
end

_tmap_fence_mods(sem, scope) =
    (:proxy, Symbol("tensormap::generic"), sem, scope)

@testset "tensormap proxy fences: exact PTX 9.3 surface" begin
    @test length(EXPECTED_TENSORMAP_FENCES) == 8
    @test length(unique(f.intr for f in EXPECTED_TENSORMAP_FENCES)) == 8
    p0 = Core.LLVMPtr{UInt8, PTX.AS.Generic}

    for form in EXPECTED_TENSORMAP_FENCES
        op = Operation{:fence, _tmap_fence_mods(form.sem, form.scope)}()
        argtypes = form.sem === :acquire ? (p0, Val{128}) : ()
        @test NVVM.isintrinsic(form.intr)
        @test which(op, argtypes).module === PTX
        info = PTX.lowering(op, argtypes)
        @test info.tier === :intrinsic
        @test info.rettype === Nothing
        @test info.intrinsics == [form.intr]
    end
end

@testset "tensormap proxy fences: optimizer contracts" begin
    p0 = Core.LLVMPtr{UInt8, PTX.AS.Generic}
    for scope in EXPECTED_TENSORMAP_FENCE_SCOPES
        acquire = NVVM.intrinsic(
            "llvm.nvvm.fence.proxy.tensormap_generic.acquire.$scope")
        release = NVVM.intrinsic(
            "llvm.nvvm.fence.proxy.tensormap_generic.release.$scope")

        # Acquire names the descriptor range; release deliberately leaves
        # memory effects unspecified.  Both choices make a void fence
        # observable to LLVM, while neither fence is collective.
        @test acquire.props == (:nocallback, :argmemonly)
        @test acquire.immargs == (2,)
        @test acquire.ranges == ((2, 128, 129),)
        @test release.props == (:nocallback,)
        @test !NVVM.is_convergent(acquire)
        @test !NVVM.is_convergent(release)

        acquire_ir = NVVM.synthesize(acquire.name, (p0, Val{128})).ir
        release_ir = NVVM.synthesize(release.name, ()).ir
        if Base.libllvm_version < v"16"
            @test occursin("argmemonly", acquire_ir)
        else
            @test occursin("memory(argmem: readwrite)", acquire_ir)
        end
        @test !occursin("memory(none)", acquire_ir)
        @test !occursin("readnone", acquire_ir)
        @test !occursin("memory(", release_ir)
        @test !occursin("readnone", release_ir)
        @test !occursin("convergent", acquire_ir)
        @test !occursin("convergent", release_ir)
    end
end

@testset "tensormap proxy fences: malformed ABIs fail loud" begin
    p0 = Core.LLVMPtr{UInt8, PTX.AS.Generic}
    p1 = Core.LLVMPtr{UInt8, PTX.AS.Global}
    p3 = Core.LLVMPtr{UInt8, PTX.AS.Shared}

    misses = (
        # Acquire requires a generic address and exactly literal 128.
        (_tmap_fence_mods(:acquire, :gpu), (p0, Val{64})),
        (_tmap_fence_mods(:acquire, :gpu), (p0, UInt32)),
        (_tmap_fence_mods(:acquire, :gpu), (p0,)),
        (_tmap_fence_mods(:acquire, :gpu), (p1, Val{128})),
        (_tmap_fence_mods(:acquire, :gpu), (p3, Val{128})),
        # Release has no address or size operands.
        (_tmap_fence_mods(:release, :gpu), (p0, Val{128})),
        # Closed scope set and canonical modifier ordering/direction.
        ((:proxy, Symbol("tensormap::generic"), :acquire, :gl),
         (p0, Val{128})),
        ((:proxy, :acquire, Symbol("tensormap::generic"), :gpu),
         (p0, Val{128})),
        ((:acquire, Symbol("tensormap::generic"), :proxy, :gpu),
         (p0, Val{128})),
        ((:proxy, Symbol("generic::tensormap"), :acquire, :gpu),
         (p0, Val{128})),
    )

    for (mods, argtypes) in misses
        op = Operation{:fence, mods}()
        info = PTX.lowering(op, argtypes)
        @test info.tier === :forbidden
        @test endswith(String(which(op, argtypes).file), "entries.jl")
    end

    bad = Operation{:fence, _tmap_fence_mods(:acquire, :gpu)}()
    err = try
        bad(reinterpret(p0, UInt64(0)), Val(64))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("requires an exact typed wrapper", sprint(showerror, err))
end
