using PTX.MBarriers: BarrierSet, BarrierGroup, barrier_bytes, barrier_offset

# An FA-kernel-shaped arena: heterogeneous groups, multi-dimensional
# shapes, mixed arrival counts — the shape a real warp-specialized
# synchronization plan has. The expected offsets below are hand-computed
# from the declaration (8 bytes per barrier, groups in declaration
# order), NOT derived from the implementation.
const _BS_FA = (q          = ((2,),   1),
                kv_full    = ((4,),   1),
                kv_free    = ((4,),   1),
                s_full     = ((2,),   1),
                p_q        = ((2, 4), 128),
                o_resc     = ((2, 2), 128),
                o_done     = ((),     1),
                stats_free = ((2,),   128),
                pv_done    = ((2,),   1),
                q_free     = ((),     1),
                epi        = ((),     128))

@testset "BarrierSet layout arithmetic" begin
    T = BarrierSet{_BS_FA}
    @test barrier_offset(T, :q)          == 0
    @test barrier_offset(T, :kv_full)    == 16
    @test barrier_offset(T, :kv_free)    == 48
    @test barrier_offset(T, :s_full)     == 80
    @test barrier_offset(T, :p_q)        == 96
    @test barrier_offset(T, :o_resc)     == 160
    @test barrier_offset(T, :o_done)     == 192
    @test barrier_offset(T, :stats_free) == 200
    @test barrier_offset(T, :pv_done)    == 216
    @test barrier_offset(T, :q_free)     == 232
    @test barrier_offset(T, :epi)        == 240
    @test barrier_bytes(T) == 248
    @test_throws ErrorException barrier_offset(T, :nope)
end

@testset "BarrierSet accessor arithmetic" begin
    base = reinterpret(Core.LLVMPtr{UInt64, PTX.AS.Shared}, UInt64(0))
    bars = BarrierSet{_BS_FA}(base)
    addr(p) = reinterpret(UInt64, p)

    # ()-shaped groups hand back the raw pointer; shaped groups index
    # row-major, 0-based, one index per dimension.
    @test addr(bars.o_done) == 192
    @test addr(bars.q_free) == 232
    @test bars.p_q isa BarrierGroup{(2, 4)}
    @test addr(bars.kv_full[0]) == 16
    @test addr(bars.kv_full[3]) == 16 + 3 * 8
    @test addr(bars.p_q[0, 0])  == 96
    @test addr(bars.p_q[1, 2])  == 96 + (1 * 4 + 2) * 8
    @test addr(bars.o_resc[1, 1]) == 160 + (1 * 2 + 1) * 8
    @test addr(bars.base) == 0
    @test propertynames(bars) == (keys(_BS_FA)..., :base)

    # Index arity mismatches fail loud rather than guessing a layout.
    @test_throws ErrorException bars.p_q[1]
    @test_throws ErrorException bars.kv_full[1, 0]
    @test_throws ErrorException bars.missing_group
end
