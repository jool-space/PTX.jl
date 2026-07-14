# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=9.0
#
# elect.sync starts at sm_90, so it is isolated from the Ada-compatible
# structured-result runtime test.  The ISA promises a deterministic leader,
# not a particular lane: assert the specified cross-lane invariants only.

function _structured_elect_runtime!(leaders, selected)
    lane = ptx"mov.u32"(sreg"tid.x")
    leader, is_leader = ptx"elect.sync"(UInt32(0x0000ffff))
    @inbounds begin
        leaders[lane + UInt32(1)] = leader
        selected[lane + UInt32(1)] = is_leader
    end
    return nothing
end

@testset "elect.sync grouped result executes with ISA semantics" begin
    leaders = CUDACore.zeros(UInt32, 16)
    selected = CUDACore.zeros(Bool, 16)
    @cuda threads=16 _structured_elect_runtime!(leaders, selected)
    CUDACore.synchronize()

    host_leaders = Array(leaders)
    host_selected = Array(selected)
    @test all(==(first(host_leaders)), host_leaders)
    @test first(host_leaders) < UInt32(16)
    @test count(identity, host_selected) == 1
    @test host_selected[Int(first(host_leaders)) + 1]
end
