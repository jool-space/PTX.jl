# Exact-floor compiler evidence for the tcgen05.mma surface. PTX 9.3
# §9.7.17.10: dense completion (TMEM-A operand form, collector usage,
# .ashift, the disable-output-lane mask vector, the trailing scale-input-d
# immediate), sparse `.sp` (the sparsity-metadata TMEM operand position),
# weight-stationary `.ws` (addressed B-side collector buffers, the trailing
# zero-column-mask descriptor), and the MX block-scale family with its
# family-target gating. PTX ISA 9.4 §9.7.18.10: `.kind::ti16`, the
# `.collector::b::*` forms, and `.decompress::lut::b`, assembled on the
# sm_107 family and rejected on every other Blackwell target. Compile-only;
# runtime evidence needs datacenter Blackwell. ptxas rejects mixing
# .cta_group::1 and ::2 in one function, so every surface splits by group.

# --- dense (PTX 9.3 §9.7.17.10) ------------------------------------------------

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
        r"tcgen05\.mma\.cta_group::1\.kind::i8\.ashift\.collector::a::discard \[%r\d+\], \[%r\d+\]",
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
        r"tcgen05\.mma\.cta_group::2\.kind::f16\.ashift\.collector::a::discard \[%r\d+\], \[%r\d+\], %rd\d+, %r\d+, %p\d+, 3;",
        ptx2)
end

# --- sparse (PTX 9.3 §9.7.17.10.9.2) ------------------------------------------

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
        r"tcgen05\.mma\.sp\.cta_group::1\.kind::i8\.ashift\.collector::a::discard \[%r\d+\], \[%r\d+\], %rd\d+, \[%r\d+\]",
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

# --- weight-stationary (PTX 9.3 §9.7.17.10.9.3/.4) ------------------------------

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

# --- MX block-scale, dense and sp -------------------------------------------------
# The a-variant-exclusive gating of the sparse mxf4 kinds follows the
# §9.7.18.10 support list: the family architectures exclude `.kind::i8`,
# `.kind::mxf4nvf4`, and `.kind::mxf4` for `.sp`, and ptxas refuses the
# modifier pair, not the target. Every surface here stays on cta_group::1.

function _t5_mx_dense!(out::Core.LLVMPtr{UInt32, 1}, d::UInt32,
                       adesc::UInt64, bdesc::UInt64, idesc::UInt32,
                       sa::UInt32, sb::UInt32)
    ptx"tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"(
        d, adesc, bdesc, idesc, sa, sb, false)
    ptx"tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X"(
        d, adesc, bdesc, idesc, sa, sb, false)
    # TMEM-A species of the third kind.
    ptx"tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"(
        d, d, bdesc, idesc, sa, sb, false)
    # collector::a usage (absent collector = the ISA-default discard).
    ptx"tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::fill"(
        d, adesc, bdesc, idesc, sa, sb, false)
    ptx"tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::use"(
        d, adesc, bdesc, idesc, sa, sb, false)
    ptx"st.global.b32"(out, d)
    return nothing
end

function _t5_mx_sp!(out::Core.LLVMPtr{UInt32, 1}, d::UInt32,
                    adesc::UInt64, bdesc::UInt64, meta::UInt32,
                    idesc::UInt32, sa::UInt32, sb::UInt32)
    ptx"tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"(
        d, adesc, bdesc, meta, idesc, sa, sb, false)
    ptx"tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.scale_vec::2X"(
        d, adesc, bdesc, meta, idesc, sa, sb, false)
    # TMEM-A species of the third kind.
    ptx"tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"(
        d, d, bdesc, meta, idesc, sa, sb, false)
    # collector::a on a sparse form.
    ptx"tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::lastuse"(
        d, adesc, bdesc, meta, idesc, sa, sb, false)
    ptx"st.global.b32"(out, d)
    return nothing
end

function _t5_mx_sp_family!(out::Core.LLVMPtr{UInt32, 1}, d::UInt32,
                           adesc::UInt64, bdesc::UInt64, meta::UInt32,
                           idesc::UInt32, sa::UInt32, sb::UInt32)
    ptx"tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32"(
        d, adesc, bdesc, meta, idesc, sa, sb, false)
    ptx"tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::fill"(
        d, adesc, bdesc, meta, idesc, sa, sb, false)
    ptx"st.global.b32"(out, d)
    return nothing
end

function _t5_mx_sp_mxf4_family!(out::Core.LLVMPtr{UInt32, 1}, d::UInt32,
                                adesc::UInt64, bdesc::UInt64, meta::UInt32,
                                idesc::UInt32, sa::UInt32, sb::UInt32)
    ptx"tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32"(
        d, adesc, bdesc, meta, idesc, sa, sb, false)
    ptx"st.global.b32"(out, d)
    return nothing
end

const _T5_MX_DENSE_TT = Tuple{Core.LLVMPtr{UInt32, 1}, UInt32, UInt64,
                              UInt64, UInt32, UInt32, UInt32}
const _T5_MX_SP_TT = Tuple{Core.LLVMPtr{UInt32, 1}, UInt32, UInt64, UInt64,
                           UInt32, UInt32, UInt32, UInt32}

@testset "tcgen05 MX block-scale (dense + sp) at the sm_100a floor" begin
    @test ptxas_compiles(_t5_mx_dense!, _T5_MX_DENSE_TT;
                         cap = v"10.0", feature_set = :arch)
    @test ptxas_compiles(_t5_mx_sp!, _T5_MX_SP_TT;
                         cap = v"10.0", feature_set = :arch)

    ptx = emit_ptx(_t5_mx_sp!, _T5_MX_SP_TT; cap = v"10.0",
                   feature_set = :arch)
    @test occursin(".target sm_100a", ptx)
    # The sparsity-metadata TMEM address renders bracketed, between the B
    # descriptor and the idesc; both A species keep the schema.
    @test occursin(r"tcgen05\.mma\.sp\.cta_group::1\.kind::mxf8f6f4\.block_scale\.scale_vec::1X \[%r\d+\], %rd\d+, %rd\d+, \[%r\d+\], %r\d+, \[%r\d+\], \[%r\d+\], %p\d+;",
                   ptx)
    @test occursin(r"tcgen05\.mma\.sp\.cta_group::1\.kind::mxf4nvf4\.block_scale\.scale_vec::4X \[%r\d+\], \[%r\d+\], %rd\d+, \[%r\d+\], %r\d+, \[%r\d+\], \[%r\d+\], %p\d+;",
                   ptx)
    # collector::a renders after the scale qualifier, before the operands.
    @test occursin(r"tcgen05\.mma\.sp\.cta_group::1\.kind::mxf4\.block_scale\.scale_vec::2X\.collector::a::lastuse \[%r\d+\],",
                   ptx)
end

@testset "tcgen05 MX sp: family-target gating (§9.7.18.10)" begin
    # mxf8f6f4 sp assembles on family targets with the block spelling.
    @test ptxas_compiles(_t5_mx_sp_family!, _T5_MX_SP_TT;
                         cap = v"10.0", feature_set = :family)

    # The sparse mxf4 kinds are a-variant-exclusive: same spelling, family
    # target, refused at the feature level.
    err = try
        ptxas_compiles(_t5_mx_sp_mxf4_family!, _T5_MX_SP_TT;
                       cap = v"10.0", feature_set = :family)
        nothing
    catch e
        sprint(showerror, e)
    end
    @test err isa String
    @test occursin("Feature '.kind::mxf4 with .sp modifier' not supported " *
                   "on .target 'sm_100f'", err)

    # The a-variant accepts the very same kernel.
    @test ptxas_compiles(_t5_mx_sp_mxf4_family!, _T5_MX_SP_TT;
                         cap = v"10.0", feature_set = :arch)
end

# --- PTX ISA 9.4 forms: ti16, collector::b, and lut::b on sm_107 ---------------
# Every form is a side-effecting asm call, so nothing is eliminated; the
# trailing store keeps the TMEM address live.

function _t5_mma_ti16_cg1!(out::Core.LLVMPtr{UInt32, 1}, d::UInt32,
                           adesc::UInt64, bdesc::UInt64, meta::UInt32,
                           idesc::UInt32, zcm::UInt64)
    mask = (UInt32(0), UInt32(0), UInt32(0), UInt32(0))
    ptx"tcgen05.mma.cta_group::1.kind::ti16"(d, adesc, bdesc, idesc, false)
    ptx"tcgen05.mma.cta_group::1.kind::ti16.collector::b::fill"(
        d, adesc, bdesc, idesc, mask, false)
    ptx"tcgen05.mma.cta_group::1.kind::ti16.collector::a::fill.collector::b::use"(
        d, d, bdesc, idesc, false)
    ptx"tcgen05.mma.cta_group::1.kind::ti16.ashift.collector::a::lastuse.collector::b::lastuse"(
        d, d, bdesc, idesc, mask, false)
    ptx"tcgen05.mma.sp.cta_group::1.kind::ti16.collector::b::fill"(
        d, adesc, bdesc, meta, idesc, false)
    ptx"tcgen05.mma.sp.cta_group::1.kind::ti16.ashift.collector::b::use"(
        d, d, bdesc, meta, idesc, mask, false)
    ptx"tcgen05.mma.ws.cta_group::1.kind::ti16"(d, adesc, bdesc, idesc, false)
    ptx"tcgen05.mma.ws.cta_group::1.kind::ti16.collector::b1::fill"(
        d, d, bdesc, idesc, false, zcm)
    ptx"tcgen05.mma.ws.sp.cta_group::1.kind::ti16.collector::b3::lastuse"(
        d, adesc, bdesc, meta, idesc, false)
    ptx"st.global.b32"(out, d)
    return nothing
end

function _t5_mma_ti16_cg2!(out::Core.LLVMPtr{UInt32, 1}, d::UInt32,
                           adesc::UInt64, bdesc::UInt64, meta::UInt32,
                           idesc::UInt32)
    mask = ntuple(_ -> UInt32(0), Val(8))
    ptx"tcgen05.mma.cta_group::2.kind::ti16.collector::b::lastuse"(
        d, adesc, bdesc, idesc, false)
    ptx"tcgen05.mma.cta_group::2.kind::ti16.collector::a::use.collector::b::fill"(
        d, adesc, bdesc, idesc, mask, false)
    ptx"tcgen05.mma.sp.cta_group::2.kind::ti16.ashift"(
        d, d, bdesc, meta, idesc, false)
    ptx"st.global.b32"(out, d)
    return nothing
end

function _t5_mma_collb_cg1!(out::Core.LLVMPtr{UInt32, 1}, d::UInt32,
                            adesc::UInt64, bdesc::UInt64, meta::UInt32,
                            idesc::UInt32, sa::UInt32, sb::UInt32)
    mask = (UInt32(0), UInt32(0), UInt32(0), UInt32(0))
    ptx"tcgen05.mma.cta_group::1.kind::f16.collector::b::fill"(
        d, adesc, bdesc, idesc, false)
    ptx"tcgen05.mma.cta_group::1.kind::f16.collector::a::fill.collector::b::use"(
        d, adesc, bdesc, idesc, mask, false, Val(5))
    ptx"tcgen05.mma.cta_group::1.kind::tf32.ashift.collector::a::lastuse.collector::b::lastuse"(
        d, d, bdesc, idesc, false, Val(1))
    ptx"tcgen05.mma.cta_group::1.kind::f8f6f4.collector::b::use"(
        d, d, bdesc, idesc, mask, false)
    ptx"tcgen05.mma.sp.cta_group::1.kind::f16.collector::b::fill"(
        d, adesc, bdesc, meta, idesc, false)
    ptx"tcgen05.mma.sp.cta_group::1.kind::tf32.ashift.collector::b::use"(
        d, d, bdesc, meta, idesc, mask, false, Val(15))
    ptx"tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::b::fill"(
        d, adesc, bdesc, idesc, sa, sb, false)
    ptx"tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::use.collector::b::lastuse"(
        d, d, bdesc, idesc, sa, sb, false)
    ptx"tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::fill.collector::b::use"(
        d, adesc, bdesc, meta, idesc, sa, sb, false)
    ptx"st.global.b32"(out, d)
    return nothing
end

# The sparse mxf4 kinds are a-variant-exclusive, so their B-collector forms
# assemble on sm_107a only.
function _t5_mma_collb_avariant!(out::Core.LLVMPtr{UInt32, 1}, d::UInt32,
                                 adesc::UInt64, bdesc::UInt64, meta::UInt32,
                                 idesc::UInt32, sa::UInt32, sb::UInt32)
    ptx"tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::fill.collector::b::fill"(
        d, adesc, bdesc, meta, idesc, sa, sb, false)
    ptx"tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::b::use"(
        d, d, bdesc, meta, idesc, sa, sb, false)
    ptx"st.global.b32"(out, d)
    return nothing
end

function _t5_mma_lut_cg1!(out::Core.LLVMPtr{UInt32, 1}, d::UInt32,
                          adesc::UInt64, bdesc::UInt64, lut::UInt32,
                          idesc::UInt32, sa::UInt32, sb::UInt32)
    mask = (UInt32(0), UInt32(0), UInt32(0), UInt32(0))
    ptx"tcgen05.mma.cta_group::1.kind::f8f6f4.decompress::lut::b"(
        d, adesc, bdesc, lut, idesc, false)
    ptx"tcgen05.mma.cta_group::1.kind::f8f6f4.decompress::lut::b.collector::a::fill.collector::b::use"(
        d, d, bdesc, lut, idesc, mask, false)
    ptx"tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.decompress::lut::b.block32"(
        d, adesc, bdesc, lut, idesc, sa, sb, false)
    ptx"tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.decompress::lut::b.block32.collector::b::lastuse"(
        d, d, bdesc, lut, idesc, sa, sb, false)
    ptx"st.global.b32"(out, d)
    return nothing
end

const _T5_TI16_CG1_TT = Tuple{Core.LLVMPtr{UInt32, 1}, UInt32, UInt64,
                              UInt64, UInt32, UInt32, UInt64}
const _T5_TI16_CG2_TT = Tuple{Core.LLVMPtr{UInt32, 1}, UInt32, UInt64,
                              UInt64, UInt32, UInt32}

@testset "tcgen05 ti16, collector::b, and lut::b mma assemble on sm_107" begin
    if _ptxas_isa() < v"9.4"
        @test_skip "PTX 9.4 assembler required"
    else
        family = ((_t5_mma_ti16_cg1!, _T5_TI16_CG1_TT),
                  (_t5_mma_ti16_cg2!, _T5_TI16_CG2_TT),
                  (_t5_mma_collb_cg1!, _T5_MX_SP_TT),
                  (_t5_mma_lut_cg1!, _T5_MX_SP_TT))
        for (kernel, tt) in family, feature_set in (:family, :arch)
            @test ptxas_compiles(kernel, tt; cap = v"10.7", feature_set)
        end
        @test ptxas_compiles(_t5_mma_collb_avariant!, _T5_MX_SP_TT;
                             cap = v"10.7", feature_set = :arch)

        ptx = emit_ptx(_t5_mma_ti16_cg1!, _T5_TI16_CG1_TT;
                       cap = v"10.7", feature_set = :family)
        @test occursin(".target sm_107f", ptx)
        # The operand schema per form: sp metadata bracketed after B, the
        # mask before enable-input-d, the ws zero-column mask last.
        @test occursin(r"tcgen05\.mma\.cta_group::1\.kind::ti16 \[%r\d+\], %rd\d+, %rd\d+, %r\d+, %p\d+;",
                       ptx)
        @test occursin(r"tcgen05\.mma\.cta_group::1\.kind::ti16\.collector::b::fill \[%r\d+\], %rd\d+, %rd\d+, %r\d+, \{%r\d+, %r\d+, %r\d+, %r\d+\}, %p\d+;",
                       ptx)
        @test occursin(r"tcgen05\.mma\.cta_group::1\.kind::ti16\.ashift\.collector::a::lastuse\.collector::b::lastuse \[%r\d+\], \[%r\d+\], %rd\d+, %r\d+, \{%r\d+, %r\d+, %r\d+, %r\d+\}, %p\d+;",
                       ptx)
        @test occursin(r"tcgen05\.mma\.sp\.cta_group::1\.kind::ti16\.collector::b::fill \[%r\d+\], %rd\d+, %rd\d+, \[%r\d+\], %r\d+, %p\d+;",
                       ptx)
        @test occursin(r"tcgen05\.mma\.ws\.cta_group::1\.kind::ti16\.collector::b1::fill \[%r\d+\], \[%r\d+\], %rd\d+, %r\d+, %p\d+, %rd\d+;",
                       ptx)
        @test occursin(r"tcgen05\.mma\.ws\.sp\.cta_group::1\.kind::ti16\.collector::b3::lastuse \[%r\d+\], %rd\d+, %rd\d+, \[%r\d+\], %r\d+, %p\d+;",
                       ptx)

        ptx = emit_ptx(_t5_mma_collb_cg1!, _T5_MX_SP_TT;
                       cap = v"10.7", feature_set = :family)
        # scale-input-d is a trailing immediate; the block-scale forms keep
        # the two scale addresses.
        @test occursin(r"tcgen05\.mma\.cta_group::1\.kind::f16\.collector::a::fill\.collector::b::use \[%r\d+\], %rd\d+, %rd\d+, %r\d+, \{%r\d+, %r\d+, %r\d+, %r\d+\}, %p\d+, 5;",
                       ptx)
        @test occursin(r"tcgen05\.mma\.cta_group::1\.kind::tf32\.ashift\.collector::a::lastuse\.collector::b::lastuse \[%r\d+\], \[%r\d+\], %rd\d+, %r\d+, %p\d+, 1;",
                       ptx)
        @test occursin(r"tcgen05\.mma\.sp\.cta_group::1\.kind::tf32\.ashift\.collector::b::use \[%r\d+\], \[%r\d+\], %rd\d+, \[%r\d+\], %r\d+, \{%r\d+, %r\d+, %r\d+, %r\d+\}, %p\d+, 15;",
                       ptx)
        @test occursin(r"tcgen05\.mma\.sp\.cta_group::1\.kind::mxf8f6f4\.block_scale\.block32\.collector::a::fill\.collector::b::use \[%r\d+\], %rd\d+, %rd\d+, \[%r\d+\], %r\d+, \[%r\d+\], \[%r\d+\], %p\d+;",
                       ptx)

        ptx = emit_ptx(_t5_mma_lut_cg1!, _T5_MX_SP_TT;
                       cap = v"10.7", feature_set = :family)
        @test occursin(r"tcgen05\.mma\.cta_group::1\.kind::f8f6f4\.decompress::lut::b \[%r\d+\], %rd\d+, %rd\d+, \[%r\d+\], %r\d+, %p\d+;",
                       ptx)
        @test occursin(r"tcgen05\.mma\.cta_group::1\.kind::mxf8f6f4\.block_scale\.decompress::lut::b\.block32\.collector::b::lastuse \[%r\d+\], \[%r\d+\], %rd\d+, \[%r\d+\], %r\d+, \[%r\d+\], \[%r\d+\], %p\d+;",
                       ptx)

        for (kernel, tt) in family,
                (cap, feature_set, target) in ((v"10.0", :family, "sm_100f"),
                                               (v"10.3", :arch, "sm_103a"),
                                               (v"12.0", :family, "sm_120f"))
            @test ptxas_rejects(kernel, tt; cap, feature_set, target)
        end
        # The sparse mxf4 kinds refuse the family target even on sm_107.
        @test ptxas_rejects(_t5_mma_collb_avariant!, _T5_MX_SP_TT; cap = v"10.7",
                            feature_set = :family, target = "sm_107f")
    end
end
