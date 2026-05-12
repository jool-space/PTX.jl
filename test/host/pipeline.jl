# Host tests for cursor arithmetic on `Pipeline{N_STAGES}`. The full
# kernel-emission tests for `BarrierArray` / `pipeline_init!` /
# `barrier_*` verbs live under `test/ptxas/hopper.jl` (they need the
# NVPTX backend + ptxas).

using PTX: Pipeline, pipeline_stage, pipeline_phase, pipeline_cursor

@testset "pipeline_cursor arithmetic (N=2)" begin
    P = Pipeline{2}
    # k=0 → (stage 0, phase 0). k=1 → (stage 1, phase 0).
    # k=2 → (stage 0, phase 1). k=3 → (stage 1, phase 1).
    # k=4 → (stage 0, phase 0) — wraps.
    for (k, exp_stage, exp_phase) in [(0, 0, 0), (1, 1, 0),
                                       (2, 0, 1), (3, 1, 1),
                                       (4, 0, 0), (5, 1, 0)]
        @test pipeline_stage(P, Int32(k)) == Int32(exp_stage)
        @test pipeline_phase(P, Int32(k)) == UInt32(exp_phase)
        s, p = pipeline_cursor(P, Int32(k))
        @test s == Int32(exp_stage)
        @test p == UInt32(exp_phase)
    end
end

@testset "pipeline_cursor arithmetic (N=4)" begin
    P = Pipeline{4}
    # log2(4) = 2 → phase flips every 4 iters.
    @test pipeline_phase(P, Int32(0)) == UInt32(0)
    @test pipeline_phase(P, Int32(3)) == UInt32(0)
    @test pipeline_phase(P, Int32(4)) == UInt32(1)
    @test pipeline_phase(P, Int32(7)) == UInt32(1)
    @test pipeline_phase(P, Int32(8)) == UInt32(0)

    @test pipeline_stage(P, Int32(0)) == Int32(0)
    @test pipeline_stage(P, Int32(3)) == Int32(3)
    @test pipeline_stage(P, Int32(4)) == Int32(0)
    @test pipeline_stage(P, Int32(5)) == Int32(1)
end

@testset "Pipeline rejects non-power-of-two N_STAGES" begin
    # @generated phase function asserts ispow2(N). The error surfaces
    # at compile time of the first phase call — exercise that here.
    @test_throws AssertionError pipeline_phase(Pipeline{3}, Int32(0))
end
