# Offline compiler evidence at each retained legal target partition. CUDA 13
# has retired sm_20/sm_50/sm_70, so sm_75 is the earliest ptxas target for
# pmevent and both lop3 language floors. setmaxnreg is architecture-specific
# and is compiled at its exact sm_90a floor. No kernel is loaded onto a GPU.

function _immediate_baseline!(out::CuDeviceVector{UInt32,1},
                              a::UInt32, b::UInt32, c::UInt32, gate::Bool)
    ptx"pmevent"(Val(15))
    ptx"pmevent.mask"(Val(0x8001))
    out[1] = ptx"lop3.b32"(a, b, c, Val(0x96))
    value, pred = ptx"lop3.or.b32"(a, b, c, Val(0xe8), gate)
    out[2] = value + UInt32(pred)
    return nothing
end

function _immediate_sm90a!()
    ptx"setmaxnreg.dec.sync.aligned.u32"(Val(24))
    ptx"setmaxnreg.inc.sync.aligned.u32"(Val(256))
    return nothing
end

@testset "immediate forms compile at retained legal floors" begin
    baseline_types = Tuple{CuDeviceVector{UInt32,1},UInt32,UInt32,UInt32,Bool}
    @test ptxas_compiles(_immediate_baseline!, baseline_types; cap = v"7.5")
    baseline = emit_ptx(_immediate_baseline!, baseline_types; cap = v"7.5")
    for head in ("pmevent 15", "pmevent.mask 32769", "lop3.b32",
                 "lop3.or.b32")
        @test occursin(head, baseline)
    end

    @test ptxas_compiles(_immediate_sm90a!, Tuple{};
                         cap = v"9.0", feature_set = :arch)
    hopper = emit_ptx(_immediate_sm90a!, Tuple{};
                      cap = v"9.0", feature_set = :arch)
    @test occursin("setmaxnreg.dec.sync.aligned.u32 24", hopper)
    @test occursin("setmaxnreg.inc.sync.aligned.u32 256", hopper)
end
