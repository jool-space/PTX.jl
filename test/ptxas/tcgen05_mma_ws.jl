# Exact-floor compiler evidence for PTX 9.3 §9.7.17.10.9.3/.4
# weight-stationary tcgen05.mma.ws: addressed B-side collector buffers,
# the trailing runtime zero-column-mask descriptor, and the ws.sp
# metadata operand. Compile-only; ws is cta_group::1-only so a single
# surface function suffices.

function _t5_mma_ws!(d::UInt32, a_t::UInt32, a_desc::UInt64,
                     b_desc::UInt64, meta::UInt32, idesc::UInt32,
                     e::Bool, zcm::UInt64)
    ptx"tcgen05.mma.ws.cta_group::1.kind::f16"(d, a_desc, b_desc, idesc, e)
    ptx"tcgen05.mma.ws.cta_group::1.kind::tf32.collector::b1::fill"(
        d, a_t, b_desc, idesc, e)
    ptx"tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::use"(
        d, a_desc, b_desc, idesc, e, zcm)
    ptx"tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b3::lastuse"(
        d, a_t, b_desc, idesc, e, zcm)
    ptx"tcgen05.mma.ws.sp.cta_group::1.kind::f16"(
        d, a_desc, b_desc, meta, idesc, e)
    ptx"tcgen05.mma.ws.sp.cta_group::1.kind::tf32.collector::b1::use"(
        d, a_t, b_desc, meta, idesc, e, zcm)
    return nothing
end

@testset "tcgen05 ws mma at the sm_100a floor" begin
    types = Tuple{UInt32, UInt32, UInt64, UInt64, UInt32, UInt32, Bool,
                  UInt64}
    @test ptxas_compiles(_t5_mma_ws!, types;
                         cap = v"10.0", feature_set = :arch)

    ptx = emit_ptx(_t5_mma_ws!, types; cap = v"10.0", feature_set = :arch)
    @test occursin(".target sm_100a", ptx)
    @test occursin(
        r"tcgen05\.mma\.ws\.cta_group::1\.kind::f16\.collector::b0::discard \[%r\d+\], %rd\d+, %rd\d+, %r\d+, %p\d+;",
        ptx)
    @test occursin(
        r"tcgen05\.mma\.ws\.cta_group::1\.kind::tf32\.collector::b1::fill \[%r\d+\], \[%r\d+\], %rd\d+, %r\d+, %p\d+;",
        ptx)
    @test occursin(
        r"tcgen05\.mma\.ws\.cta_group::1\.kind::i8\.collector::b2::use \[%r\d+\], %rd\d+, %rd\d+, %r\d+, %p\d+, %rd\d+;",
        ptx)
    @test occursin(
        r"tcgen05\.mma\.ws\.cta_group::1\.kind::f8f6f4\.collector::b3::lastuse \[%r\d+\], \[%r\d+\], %rd\d+, %r\d+, %p\d+, %rd\d+;",
        ptx)
    @test occursin(
        r"tcgen05\.mma\.ws\.sp\.cta_group::1\.kind::f16\.collector::b0::discard \[%r\d+\], %rd\d+, %rd\d+, \[%r\d+\], %r\d+, %p\d+;",
        ptx)
    @test occursin(
        r"tcgen05\.mma\.ws\.sp\.cta_group::1\.kind::tf32\.collector::b1::use \[%r\d+\], \[%r\d+\], %rd\d+, \[%r\d+\], %r\d+, %p\d+, %rd\d+;",
        ptx)
end
