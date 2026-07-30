# Exact-floor compiler evidence for the tcgen05 MX block-scale family,
# dense and `.sp`: assembly at sm_100a, the family-target story for
# `.kind::mxf8f6f4`, and the a-variant-exclusive gating of the sparse
# mxf4 kinds (§9.7.18.10 support list: the family architectures exclude
# `.kind::i8`/`.kind::mxf4nvf4`/`.kind::mxf4` for `.sp` — ptxas refuses
# the modifier pair, not the target). ptxas rejects mixing
# cta_group::1/::2 in one function, so every surface here stays on
# cta_group::1.

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
