using PTX: format_call, build_call, Operation

# PTX ISA 9.5 additions that PTX.jl binds: rendering and lowering contracts.
# Assembly evidence lives in ptxas/ptx_isa_9_5.jl.

@testset "PTX 9.5 locality special registers" begin
    for name in ("%locality_domain_id", "%locality_domain_supplemental_sm")
        sreg = PTX.SpecialReg{Symbol(name)}
        @test format_call(ptx"mov.u32", Tuple{sreg}) == "mov.u32 \$0, $name;"
        ci, rt = first(PTX.device_code_typed(ptx"mov.u32", (sreg,)))
        @test rt === UInt32
    end
end

@testset "PTX 9.5 fabric submit groups (asm tier)" begin
    ci, rt = first(PTX.device_code_typed(ptx"fabric.submit.commit_group", ()))
    @test rt === Nothing
    s = string(ci)
    @test occursin("fabric.submit.commit_group;", s)
    @test occursin("~{memory}", s)

    for n in (0, 1, 7)
        op = ptx"fabric.wait.sync_restrict::read"
        @test which(op, (Val{n},)).module == PTX
        ci, rt = first(PTX.device_code_typed(op, (Val{n},)))
        @test rt === Nothing
        s = string(ci)
        @test occursin("fabric.wait.sync_restrict::read $n;", s)
        @test occursin("~{memory}", s)
    end
    @test_throws ErrorException ptx"fabric.wait.sync_restrict::read"(Val(-1))
    @test_throws ErrorException ptx"fabric.wait.sync_restrict::read"(Val(:all))
end

@testset "PTX 9.5 tensormap.query.alloc_class_large (typed wrappers)" begin
    for (ss, as, constraint) in (((),                       PTX.AS.Generic, "l"),
                                 ((:global,),               PTX.AS.Global,  "l"),
                                 ((Symbol("shared::cta"),), PTX.AS.Shared,  "l"))
        mods = (:query, :alloc_class_large, ss..., :b1024)
        op = Operation{:tensormap, mods}()
        argtypes = (Core.LLVMPtr{UInt8, as},)
        @test which(op, argtypes).module == PTX
        ci, rt = first(PTX.device_code_typed(op, argtypes))
        @test rt === Bool
        s = string(ci)
        @test occursin("tensormap." * join(mods, ".") * " \\\$0, [\\\$1];", s)
        @test occursin("=b,$constraint,~{memory}", s)
    end

    # Unwrapped state spaces and carriers must not fall back to the
    # tensormap.replace sink contract, which would drop the destination.
    misses = (
        ((:query, :alloc_class_large, :const, :b1024),
         (Core.LLVMPtr{UInt8, PTX.AS.Const},)),
        ((:query, :alloc_class_large, :param, :b1024),
         (Core.LLVMPtr{UInt8, PTX.AS.Generic},)),
        ((:query, :alloc_class_large, :global, :b1024),
         (Core.LLVMPtr{UInt8, PTX.AS.Shared},)),
    )
    for (mods, argtypes) in misses
        op = Operation{:tensormap, mods}()
        @test PTX.lowering(op, argtypes).tier === :forbidden
        err = try
            op(reinterpret(only(argtypes), UInt64(0)))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("requires an exact typed wrapper", sprint(showerror, err))
    end
end

@testset "PTX 9.5 chain forms" begin
    for f in (:sin, :cos)
        op = Operation{f, (:approx, Symbol("2pi"), :ftz, :f32)}()
        @test format_call(op, Tuple{Float32}) == "$f.approx.2pi.ftz.f32 \$0, \$1;"
        @test build_call(f, (:approx, Symbol("2pi"), :ftz, :f32),
                         (Float32,)).rettype === Float32
    end

    mods = (:alias, Symbol("sync_restrict::read"), :release, :sys)
    spec = build_call(:fence, mods, ())
    @test spec.asm == "fence.alias.sync_restrict::read.release.sys;"
    @test spec.rettype === Nothing
    @test spec.side_effects
    @test occursin("~{memory}", spec.constraints)
    @test !spec.convergent

    @test format_call(ptx"cp.async.bulk.wait_group.sync_restrict::proxy::async",
                      Tuple{Val{3}}) ==
          "cp.async.bulk.wait_group.sync_restrict::proxy::async 3;"

    # tensormap.replace .alloc_class stays a chain sink.
    pG = Core.LLVMPtr{UInt8, PTX.AS.Global}
    @test format_call(ptx"tensormap.replace.tile.alloc_large.global.b1024",
                      Tuple{pG}) ==
          "tensormap.replace.tile.alloc_large.global.b1024 [\$0];"
    @test format_call(ptx"tensormap.replace.tile.global_address.alloc_same.global.b1024.b64",
                      Tuple{pG, UInt64}) ==
          "tensormap.replace.tile.global_address.alloc_same.global.b1024.b64 [\$0], \$1;"
end
