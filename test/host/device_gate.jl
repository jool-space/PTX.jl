using PTX: DeviceOnlyError

@testset "host calls throw instead of reaching the host backend" begin
    # One spelling per emission path; any of them reaching LLVM would abort
    # this worker rather than fail the test.
    @test_throws DeviceOnlyError ptx"add.f32"(1f0, 2f0)               # chain asm
    @test_throws DeviceOnlyError ptx"mov.u32"(ptx"%tid.x")            # NVVM intrinsic
    @test_throws DeviceOnlyError ptx"fence.sc.gpu"()                  # core IR
    @test_throws DeviceOnlyError ptx"tcgen05.fence::before_thread_sync"()  # @asmcall
    @test_throws DeviceOnlyError ptx"ld.global.v4.f32"(
        reinterpret(Core.LLVMPtr{Float32, PTX.AS.Global}, C_NULL))    # core IR load

    err = try
        ptx"add.f32"(1f0, 2f0)
    catch e
        e
    end
    @test err.what == "add.f32 \$0, \$1, \$2;"
    msg = sprint(showerror, err)
    @test occursin("cannot run on the host", msg)
    @test occursin("PTX.on_device() = true", msg)
end

# Probed through a caller: reflection on `on_device` itself resolves the
# top-level method in the native table on Julia 1.10.
_gate_probe() = PTX.on_device()

@testset "the gate resolves to the device body under CUDA.jl" begin
    @test !_gate_probe()
    ci, _ = only(PTX.device_code_typed(_gate_probe, ()))
    @test occursin("return true", string(ci))

    ci, rt = only(PTX.device_code_typed(ptx"add.f32", (Float32, Float32)))
    @test rt === Float32
    @test occursin("add.f32", string(ci))
    @test !occursin("device_only_error", string(ci))

    ci, rt = only(Base.code_typed(ptx"add.f32", (Float32, Float32)))
    @test rt === Union{}
    @test occursin("device_only_error", string(ci))
    @test !occursin("asm", string(ci))
end
