# Compile-touch shard: wgmma. The sweep engine, batching strategy, and the
# reason this evidence exists at all live in test/setup.jl
# (compile_touch_sweep). The surface is split across compile_touch_* files so
# the parallel runner spreads GPUCompiler cost over workers; this shard owns
# `op === :wgmma`. Together the shards must cover every wrapper opcode exactly once.

@testset "compile-touch (wgmma): every wrapper method compiles" begin
    result = compile_touch_sweep(op -> op === :wgmma)
    isempty(result.failures) ||
        foreach(f -> println("TOUCH FAILURE: ", f), result.failures)
    isempty(result.unsynthesized) ||
        foreach(u -> println("TOUCH UNSYNTHESIZED: ", u), result.unsynthesized)
    @test isempty(result.failures)
    # Argument synthesis must keep up with the wrapper surface: a method the
    # sweep cannot even construct types for is a hole in the sweep, not a
    # pass. Curate _touch_tv/_touch_argtypes when this fires.
    @test isempty(result.unsynthesized)
    # Regression floor: the sweep must actually be sweeping. Update when
    # the wrapper surface grows or shrinks deliberately.
    @test result.touched >= 1600
end
