using PTX
using PTX.NVVM: NVVM, synthesize
include(joinpath(@__DIR__, "..", "nvptx_backend_defs.jl"))

@testset "backend errors retain diagnostics and allow subsequent compilation" begin
    ir = "define void @probe() { ret void }"
    for (cpu, isa, diagnostic) in (("sm_999", "+ptx88", "processor"),
                                  ("sm_80", "+ptx99", "PTX"))
        result = compile_nvvm_ir(ir, cpu, isa)
        @test !result.ok
        @test occursin(diagnostic, result.diagnostics)
    end
    # Exercise parse failure as well as target validation.
    result = compile_nvvm_ir("invalid LLVM IR", "sm_80", "+ptx70")
    @test !result.ok
    @test !isempty(result.diagnostics)
    result = compile_nvvm_ir(ir, "sm_80", "+ptx70")
    @test result.ok
    @test occursin(".target sm_80", result.ptx)
end

@testset "tcgen05 ld.red registry and newly introduced vector types" begin
    expected = Set{String}()
    p6 = Core.LLVMPtr{UInt32,6}
    for shape in ("32x32b", "16x32bx2"), n in (2, 4, 8, 16, 32, 64, 128),
            dtype in ("f32", "i32")
        name = "llvm.nvvm.tcgen05.ld.red.$shape.x$n.$dtype"
        push!(expected, name)
        record = NVVM.intrinsic(name)
        @test record.ret == (Symbol("v$n$dtype"), Symbol(dtype))
        @test record.props == (:convergent, :argmemonly)
        split = shape == "16x32bx2"
        @test record.immargs == (dtype == "f32" ?
            (split ? (2, 3, 4, 5) : (2, 3, 4)) : (split ? (2, 3) : (2,)))
        args = (p6, (split ? (Val{8},) : ())..., Val{0},
                (dtype == "f32" ? (Val{false}, Val{false}) : ())...)
        emitted = synthesize(name, args)
        result = compile_nvvm_ir(emitted.ir, "sm_103f", "+ptx88")
        result.ok || @info "ld.red probe failed" name result.diagnostics
        @test result.ok
        ptx_type = dtype == "i32" ? "u32" : "f32"
        @test occursin("tcgen05.ld.red.sync.aligned.$shape.x$n.min.$ptx_type",
                       result.ptx)
    end
    @test Set(NVVM.matching("llvm.nvvm.tcgen05.ld.red.")) == expected
    @test length(expected) == 28
end

@testset "removed NVVM names fail explicitly" begin
    for name in ("llvm.nvvm.atomic.add.gen.f.cta",
                 "llvm.nvvm.fma.rn.ftz.bf16",
                 "llvm.nvvm.internal.addrspace.wrap")
        @test !NVVM.isintrinsic(name)
        @test_throws ErrorException NVVM.intrinsic(name)
    end
end

@testset "FMA contraction matches the executable backend policy" begin
    # Keeping the product as a second result prevents LLVM from folding the
    # whole pair into one instruction. The contraction option must then
    # control whether ptxas may fuse the emitted mul/add.
    ir = """
    define { float, float } @probe(float %a, float %b, float %c) {
        %m = fmul float %a, %b
        %s = fadd float %m, %c
        %r = insertvalue { float, float } poison, float %m, 0
        %r2 = insertvalue { float, float } %r, float %s, 1
        ret { float, float } %r2
    }
    """
    # Switch the option both ways in one process: backend options must not
    # leak from the preceding compilation.
    for allow in (true, false, true)
        result = compile_nvvm_ir(ir, "sm_80", "+ptx70"; fma_contraction = allow)
        @test result.ok
        modifier = allow ? "" : ".rn"
        @test occursin("mul$modifier.f32", result.ptx)
        @test occursin("add$modifier.f32", result.ptx)
        @test !occursin("fma.", result.ptx)
    end
end

@testset "tensor-map replacement dimension bounds" begin
    p0 = Core.LLVMPtr{UInt8,0}
    for (suffix, value_type) in (("box.dim", UInt32), ("element.stride", UInt32),
                                 ("global.dim", UInt32), ("global.stride", UInt64))
        name = "llvm.nvvm.tensormap.replace.$suffix"
        @test NVVM.intrinsic(name).ranges == ((2, 0, 5),)
        for dim in (0, 4)
            @test synthesize(name, (p0, Val{dim}, value_type)).rettype === Nothing
        end
        for dim in (-1, 5)
            @test_throws ErrorException synthesize(name, (p0, Val{dim}, value_type))
        end
    end
end
