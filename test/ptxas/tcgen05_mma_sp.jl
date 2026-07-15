# Exact-floor compiler evidence for PTX 9.3 §9.7.17.10.9.2 sparse
# tcgen05.mma.sp: the sparsity-metadata TMEM operand position plus the
# dense grid's qualifiers and optional operands. Compile-only; the
# surface splits by cta_group (ptxas rejects mixing ::1/::2).

function _t5_mma_sp_cg1!(d::UInt32, a_t::UInt32, a_desc::UInt64,
                         b_desc::UInt64, meta::UInt32, idesc::UInt32,
                         e::Bool, m::NTuple{4, UInt32})
    ptx"tcgen05.mma.sp.cta_group::1.kind::f16"(d, a_desc, b_desc, meta,
                                               idesc, e)
    ptx"tcgen05.mma.sp.cta_group::1.kind::tf32.collector::a::use"(
        d, a_t, b_desc, meta, idesc, e)
    ptx"tcgen05.mma.sp.cta_group::1.kind::i8.ashift"(
        d, a_t, b_desc, meta, idesc, e)
    ptx"tcgen05.mma.sp.cta_group::1.kind::f8f6f4"(
        d, a_desc, b_desc, meta, idesc, m, e)
    ptx"tcgen05.mma.sp.cta_group::1.kind::tf32"(
        d, a_desc, b_desc, meta, idesc, e, Val(5))
    ptx"tcgen05.mma.sp.cta_group::1.kind::f16.collector::a::lastuse"(
        d, a_t, b_desc, meta, idesc, m, e, Val(9))
    return nothing
end

function _t5_mma_sp_cg2!(d::UInt32, a_t::UInt32, a_desc::UInt64,
                         b_desc::UInt64, meta::UInt32, idesc::UInt32,
                         e::Bool, m::NTuple{8, UInt32})
    ptx"tcgen05.mma.sp.cta_group::2.kind::f16"(d, a_t, b_desc, meta,
                                               idesc, e)
    ptx"tcgen05.mma.sp.cta_group::2.kind::tf32.collector::a::fill"(
        d, a_desc, b_desc, meta, idesc, m, e)
    return nothing
end

@testset "tcgen05 sparse mma at the sm_100a floor" begin
    t1 = Tuple{UInt32, UInt32, UInt64, UInt64, UInt32, UInt32, Bool,
               NTuple{4, UInt32}}
    t2 = Tuple{UInt32, UInt32, UInt64, UInt64, UInt32, UInt32, Bool,
               NTuple{8, UInt32}}
    @test ptxas_compiles(_t5_mma_sp_cg1!, t1;
                         cap = v"10.0", feature_set = :arch)
    @test ptxas_compiles(_t5_mma_sp_cg2!, t2;
                         cap = v"10.0", feature_set = :arch)

    ptx1 = emit_ptx(_t5_mma_sp_cg1!, t1; cap = v"10.0", feature_set = :arch)
    ptx2 = emit_ptx(_t5_mma_sp_cg2!, t2; cap = v"10.0", feature_set = :arch)
    @test occursin(".target sm_100a", ptx1)

    # sp-meta renders bracketed between the B descriptor and idesc.
    @test occursin(
        r"tcgen05\.mma\.sp\.cta_group::1\.kind::f16\.collector::a::discard \[%r\d+\], %rd\d+, %rd\d+, \[%r\d+\], %r\d+, %p\d+;",
        ptx1)
    @test occursin(
        r"tcgen05\.mma\.sp\.cta_group::1\.kind::tf32\.collector::a::use \[%r\d+\], \[%r\d+\], %rd\d+, \[%r\d+\], %r\d+, %p\d+;",
        ptx1)
    @test occursin(
        r"tcgen05\.mma\.sp\.cta_group::1\.kind::i8\.collector::a::discard\.ashift \[%r\d+\], \[%r\d+\], %rd\d+, \[%r\d+\]",
        ptx1)
    @test occursin(
        r"tcgen05\.mma\.sp\.cta_group::1\.kind::f8f6f4\.collector::a::discard \[%r\d+\], %rd\d+, %rd\d+, \[%r\d+\], %r\d+, \{%r\d+, %r\d+, %r\d+, %r\d+\}, %p\d+;",
        ptx1)
    @test occursin(
        r"tcgen05\.mma\.sp\.cta_group::1\.kind::tf32\.collector::a::discard \[%r\d+\], %rd\d+, %rd\d+, \[%r\d+\], %r\d+, %p\d+, 5;",
        ptx1)
    @test occursin(
        r"tcgen05\.mma\.sp\.cta_group::1\.kind::f16\.collector::a::lastuse \[%r\d+\], \[%r\d+\], %rd\d+, \[%r\d+\], %r\d+, \{%r\d+, %r\d+, %r\d+, %r\d+\}, %p\d+, 9;",
        ptx1)

    @test occursin(
        r"tcgen05\.mma\.sp\.cta_group::2\.kind::f16\.collector::a::discard \[%r\d+\], \[%r\d+\], %rd\d+, \[%r\d+\], %r\d+, %p\d+;",
        ptx2)
    @test occursin(
        r"tcgen05\.mma\.sp\.cta_group::2\.kind::tf32\.collector::a::fill \[%r\d+\], %rd\d+, %rd\d+, \[%r\d+\], %r\d+, \{(%r\d+, ){7}%r\d+\}, %p\d+;",
        ptx2)
end
