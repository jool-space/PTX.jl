# Exact-floor compiler evidence for PTX 9.3 §9.7.17.10 dense tcgen05.mma
# completion: TMEM-A operand form, collector usage, .ashift, the
# disable-output-lane mask vector, and the trailing scale-input-d
# immediate. Compile-only; runtime evidence needs datacenter Blackwell.
# ptxas rejects mixing .cta_group::1 and ::2 in one function, so the
# surface splits by group.

function _t5_mma_dense_cg1!(d::UInt32, a_t::UInt32, a_desc::UInt64,
                            b_desc::UInt64, idesc::UInt32, e::Bool,
                            m::NTuple{4, UInt32})
    ptx"tcgen05.mma.cta_group::1.kind::f16"(d, a_t, b_desc, idesc, e)
    ptx"tcgen05.mma.cta_group::1.kind::f16.collector::a::fill"(
        d, a_desc, b_desc, idesc, e)
    ptx"tcgen05.mma.cta_group::1.kind::tf32.collector::a::use"(
        d, a_t, b_desc, idesc, e)
    ptx"tcgen05.mma.cta_group::1.kind::i8.ashift"(d, a_t, b_desc, idesc, e)
    ptx"tcgen05.mma.cta_group::1.kind::f16.ashift.collector::a::lastuse"(
        d, a_t, b_desc, idesc, m, e)
    ptx"tcgen05.mma.cta_group::1.kind::f8f6f4"(d, a_desc, b_desc, idesc, m, e)
    ptx"tcgen05.mma.cta_group::1.kind::tf32"(d, a_desc, b_desc, idesc, e, Val(5))
    ptx"tcgen05.mma.cta_group::1.kind::f16.collector::a::lastuse"(
        d, a_t, b_desc, idesc, m, e, Val(9))
    return nothing
end

function _t5_mma_dense_cg2!(d::UInt32, a_t::UInt32, a_desc::UInt64,
                            b_desc::UInt64, idesc::UInt32, e::Bool,
                            m::NTuple{8, UInt32})
    ptx"tcgen05.mma.cta_group::2.kind::f16"(d, a_t, b_desc, idesc, e)
    ptx"tcgen05.mma.cta_group::2.kind::tf32.collector::a::fill"(
        d, a_desc, b_desc, idesc, m, e)
    ptx"tcgen05.mma.cta_group::2.kind::f16.ashift"(
        d, a_t, b_desc, idesc, e, Val(3))
    return nothing
end

@testset "tcgen05 dense mma completion at the sm_100a floor" begin
    t1 = Tuple{UInt32, UInt32, UInt64, UInt64, UInt32, Bool,
               NTuple{4, UInt32}}
    t2 = Tuple{UInt32, UInt32, UInt64, UInt64, UInt32, Bool,
               NTuple{8, UInt32}}
    @test ptxas_compiles(_t5_mma_dense_cg1!, t1;
                         cap = v"10.0", feature_set = :arch)
    @test ptxas_compiles(_t5_mma_dense_cg2!, t2;
                         cap = v"10.0", feature_set = :arch)

    ptx1 = emit_ptx(_t5_mma_dense_cg1!, t1; cap = v"10.0",
                    feature_set = :arch)
    ptx2 = emit_ptx(_t5_mma_dense_cg2!, t2; cap = v"10.0",
                    feature_set = :arch)
    @test occursin(".target sm_100a", ptx1)

    # TMEM-A brackets vs SMEM descriptor, collector rendering, and ISel's
    # collector-before-ashift spelling.
    @test occursin(
        r"tcgen05\.mma\.cta_group::1\.kind::f16\.collector::a::discard \[%r\d+\], \[%r\d+\], %rd\d+, %r\d+, %p\d+;",
        ptx1)
    @test occursin(
        r"tcgen05\.mma\.cta_group::1\.kind::f16\.collector::a::fill \[%r\d+\], %rd\d+, %rd\d+, %r\d+, %p\d+;",
        ptx1)
    @test occursin(
        r"tcgen05\.mma\.cta_group::1\.kind::tf32\.collector::a::use \[%r\d+\], \[%r\d+\]",
        ptx1)
    @test occursin(
        r"tcgen05\.mma\.cta_group::1\.kind::i8\.collector::a::discard\.ashift \[%r\d+\], \[%r\d+\]",
        ptx1)
    # mask vector before the enable predicate; scale immediate after it
    @test occursin(
        r"tcgen05\.mma\.cta_group::1\.kind::f16\.ashift\.collector::a::lastuse \[%r\d+\], \[%r\d+\], %rd\d+, %r\d+, \{%r\d+, %r\d+, %r\d+, %r\d+\}, %p\d+;",
        ptx1)
    @test occursin(
        r"tcgen05\.mma\.cta_group::1\.kind::f8f6f4\.collector::a::discard \[%r\d+\], %rd\d+, %rd\d+, %r\d+, \{%r\d+, %r\d+, %r\d+, %r\d+\}, %p\d+;",
        ptx1)
    @test occursin(
        r"tcgen05\.mma\.cta_group::1\.kind::tf32\.collector::a::discard \[%r\d+\], %rd\d+, %rd\d+, %r\d+, %p\d+, 5;",
        ptx1)
    @test occursin(
        r"tcgen05\.mma\.cta_group::1\.kind::f16\.collector::a::lastuse \[%r\d+\], \[%r\d+\], %rd\d+, %r\d+, \{%r\d+, %r\d+, %r\d+, %r\d+\}, %p\d+, 9;",
        ptx1)

    # cta_group::2: 8-word mask and the scale immediate on the ashift form
    @test occursin(
        r"tcgen05\.mma\.cta_group::2\.kind::tf32\.collector::a::fill \[%r\d+\], %rd\d+, %rd\d+, %r\d+, \{(%r\d+, ){7}%r\d+\}, %p\d+;",
        ptx2)
    @test occursin(
        r"tcgen05\.mma\.cta_group::2\.kind::f16\.collector::a::discard\.ashift \[%r\d+\], \[%r\d+\], %rd\d+, %r\d+, %p\d+, 3;",
        ptx2)
end
