# Host inspection must work with no selected CUDA compiler artifact.
function _host_codegen_probe!(out::CuDeviceVector{UInt32,1})
    @inbounds out[1] = ptx"mov.u32"(sreg"tid.x")
    nothing
end

@testset "toolkit-free host code generation" begin
    tt = Tuple{CuDeviceVector{UInt32,1}}
    for (cap, feature_set, sm) in ((v"8.9", :baseline, "sm_89"),
                                  (v"10.0", :arch, "sm_100a"),
                                  (v"10.0", :family, "sm_100f"))
        job = _host_target_job(_host_codegen_probe!, tt; cap, feature_set)
        @test job.config.kernel
        @test !job.config.libraries
        ptx = emit_host_ptx(_host_codegen_probe!, tt; cap, feature_set)
        @test occursin(".target $sm", ptx)
        @test occursin("%tid.x", ptx)
        ir = emit_host_llvm(_host_codegen_probe!, tt; cap, feature_set)
        @test occursin("store i32", ir)
    end
    @test_throws ArgumentError _host_target_job(_host_codegen_probe!, tt; cap = v"99.0")
end
