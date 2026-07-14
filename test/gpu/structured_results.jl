# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=7.5
#
# Runtime semantics for the structured-result families available on Ada and
# every later baseline target.  The exhaustive spelling/type matrix lives in
# ptxas/structured_results.jl; this tier checks the result grouping itself:
# general setp's compare/complement pair, packed-half lane predicates, lop3's
# data/predicate pair, and match's mask/predicate pair.

function _structured_results_runtime!(out, packed_a::UInt32,
                                      packed_b::UInt32)
    lane = ptx"mov.u32"(sreg"tid.x")
    i = lane + UInt32(1)
    c = lane < UInt32(24)

    scalar = ptx"setp.lt.u32"(lane, UInt32(16))
    cmp, complement = ptx"setp.dual.eq.and.u32"(
        lane & UInt32(1), UInt32(0), c)
    half_scalar = ptx"setp.gt.f16"(Float16(lane), Float16(7))
    lane0, lane1 = ptx"setp.eq.f16x2"(packed_a, packed_b)

    parity = ptx"lop3.b32"(
        lane, UInt32(0xf0f0f0f0), UInt32(0xaaaaaaaa), Val(0x96))
    selected, selected_pred = ptx"lop3.or.b32"(
        lane, UInt32(3), UInt32(1), Val(0x80), c)

    membermask = UInt32(0xffffffff)
    matching = ptx"match.any.sync.b32"(lane & UInt32(3), membermask)
    all_mask, all_pred = ptx"match.all.sync.b64.pred"(
        UInt64(7), membermask)
    no_mask, no_pred = ptx"match.all.sync.b32.pred"(
        lane & UInt32(1), membermask)

    @inbounds begin
        out[i, 1] = UInt32(scalar)
        out[i, 2] = UInt32(cmp)
        out[i, 3] = UInt32(complement)
        out[i, 4] = UInt32(half_scalar)
        out[i, 5] = UInt32(lane0)
        out[i, 6] = UInt32(lane1)
        out[i, 7] = parity
        out[i, 8] = selected
        out[i, 9] = UInt32(selected_pred)
        out[i, 10] = matching
        out[i, 11] = all_mask
        out[i, 12] = UInt32(all_pred)
        out[i, 13] = no_mask
        out[i, 14] = UInt32(no_pred)
    end
    return nothing
end

_pack_f16x2(lo::Float16, hi::Float16) =
    UInt32(reinterpret(UInt16, lo)) |
    (UInt32(reinterpret(UInt16, hi)) << 16)

@testset "structured-result ABI executes with ISA semantics" begin
    packed_a = _pack_f16x2(Float16(1), Float16(2))
    packed_b = _pack_f16x2(Float16(1), Float16(3))
    out = CUDACore.zeros(UInt32, 32, 14)
    @cuda threads=32 _structured_results_runtime!(out, packed_a, packed_b)
    CUDACore.synchronize()
    actual = Array(out)

    match_masks = UInt32[0x11111111, 0x22222222,
                         0x44444444, 0x88888888]
    for lane in UInt32(0):UInt32(31)
        i = Int(lane) + 1
        c = lane < UInt32(24)
        @test actual[i, 1] == UInt32(lane < UInt32(16))
        @test actual[i, 2] == UInt32(iszero(lane & UInt32(1)) && c)
        @test actual[i, 3] == UInt32(!iszero(lane & UInt32(1)) && c)
        @test actual[i, 4] == UInt32(lane > UInt32(7))
        @test actual[i, 5] == UInt32(1)
        @test actual[i, 6] == UInt32(0)
        @test actual[i, 7] ==
              xor(xor(lane, UInt32(0xf0f0f0f0)), UInt32(0xaaaaaaaa))
        selected = lane & UInt32(1)
        @test actual[i, 8] == selected
        @test actual[i, 9] == UInt32(!iszero(selected) || c)
        @test actual[i, 10] == match_masks[Int(lane & UInt32(3)) + 1]
        @test actual[i, 11] == UInt32(0xffffffff)
        @test actual[i, 12] == UInt32(1)
        @test actual[i, 13] == UInt32(0)
        @test actual[i, 14] == UInt32(0)
    end
end
