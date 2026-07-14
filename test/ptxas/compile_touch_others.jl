# Compile-touch shard: others. The sweep engine, batching strategy, and the
# reason this evidence exists at all live in test/setup.jl
# (compile_touch_sweep). The surface is split across compile_touch_* files so
# the parallel runner spreads GPUCompiler cost over workers; this shard owns
# `!(op in (:wgmma, :tcgen05, :mma))`. Together the shards must cover every wrapper opcode exactly once.

@testset "compile-touch (others): every wrapper method compiles" begin
    # Optional-decompression ldmatrix has a Blackwell architecture-family
    # floor. Pin the routing explicitly because backend versions differ in
    # whether they reject these intrinsics immediately when given sm_90.
    @test _touch_target(:ldmatrix,
        (:sync, :aligned, :m8n16, :x1, :shared,
         :b8x16, :b6x16_p32)) == (v"10.0", :arch)
    @test _touch_target(:ldmatrix,
        (:sync, :aligned, :m16n16, :x2, :trans, Symbol("shared::cta"),
         :b8x16, :b4x16_p64)) == (v"10.0", :arch)

    result = compile_touch_sweep(op -> !(op in (:wgmma, :tcgen05, :mma)))
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
    @test result.touched >= 190
end
