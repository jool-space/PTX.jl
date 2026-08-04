# PTX.Warps.warp_reduce — the butterfly ladder is pinned structurally
# against emitted PTX (host tier: NVPTX emission only, no ptxas, no GPU).
# The expected offsets and c-operands below are hand-computed from the
# documented contract (descending XOR offsets W/2..1, CUDA's
# ((32-W)<<8)|0x1F width encoding), not derived from the implementation.

using PTX.Warps: warp_reduce

function _wr_full_kernel!(out::CuDeviceVector{Float32, 1})
    tid = ptx"mov.u32"(sreg"tid.x")
    v = warp_reduce((a, b) -> ptx"max.f32"(a, b), Float32(tid))
    @inbounds out[1] = v
    return nothing
end

function _wr_seg4_kernel!(out::CuDeviceVector{Float32, 1})
    tid = ptx"mov.u32"(sreg"tid.x")
    v = warp_reduce((a, b) -> ptx"add.f32"(a, b), Float32(tid), Val(4))
    @inbounds out[1] = v
    return nothing
end

# The exact hand-rolled ladder warp_reduce replaced in the norm-family
# kernels — kept here as the byte-equivalence oracle.
@inline function _wr_hand_ladder(v::Float32)
    full = UInt32(0xFFFFFFFF)
    seg  = UInt32(0x1F)
    Base.@nexprs 5 i -> begin
        offset = UInt32(1) << UInt32(5 - i)        # 16, 8, 4, 2, 1
        u      = reinterpret(UInt32, v)
        u_par  = ptx"shfl.sync.bfly.b32"(u, offset, seg, full)
        v      = ptx"max.f32"(v, reinterpret(Float32, u_par))
    end
    v
end
function _wr_hand_kernel!(out::CuDeviceVector{Float32, 1})
    tid = ptx"mov.u32"(sreg"tid.x")
    v = _wr_hand_ladder(Float32(tid))
    @inbounds out[1] = v
    return nothing
end

const _WR_TT = Tuple{CuDeviceVector{Float32, 1}}

@testset "warp_reduce: full-warp ladder structure" begin
    ptx = emit_ptx(_wr_full_kernel!, _WR_TT; cap = v"8.9")
    shfls = collect(eachmatch(r"shfl\.sync\.bfly\.b32 \t%r\d+, %r\d+, (\d+), (\d+), -1;", ptx))
    @test length(shfls) == 5
    @test [m.captures[1] for m in shfls] == ["16", "8", "4", "2", "1"]
    @test all(m -> m.captures[2] == "31", shfls)
    @test length(collect(eachmatch(r"max\.f32", ptx))) == 5
end

@testset "warp_reduce: 4-lane segments" begin
    ptx = emit_ptx(_wr_seg4_kernel!, _WR_TT; cap = v"8.9")
    shfls = collect(eachmatch(r"shfl\.sync\.bfly\.b32 \t%r\d+, %r\d+, (\d+), (\d+), -1;", ptx))
    @test length(shfls) == 2
    @test [m.captures[1] for m in shfls] == ["2", "1"]
    # ((32 - 4) << 8) | 0x1F
    @test all(m -> m.captures[2] == "7199", shfls)
end

@testset "warp_reduce: byte-identical to the hand-rolled ladder" begin
    norm(p) = replace(split(p, ".visible .entry")[2],
                      r"%(r|rd|f|p)\d+" => s"%\1",
                      r"_wr_\w+_kernel" => "_wr_kernel")
    a = norm(emit_ptx(_wr_full_kernel!, _WR_TT; cap = v"8.9"))
    b = norm(emit_ptx(_wr_hand_kernel!, _WR_TT; cap = v"8.9"))
    @test a == b
end

@testset "warp_reduce: refusals" begin
    @test_throws ErrorException warp_reduce(max, 1.0f0, Val(3))
    @test_throws ErrorException warp_reduce(max, 1.0f0, Val(1))
    @test_throws ErrorException warp_reduce(max, 1.0f0, Val(64))
    @test_throws ErrorException warp_reduce(max, 1.0, Val(32))   # Float64
end
