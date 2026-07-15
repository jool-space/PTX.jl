# Exact-floor compiler evidence for PTX 9.3 §9.7.17.8 tcgen05.ld/st:
# the pack::16b/unpack::16b qualifier rendering and the 16x32bx2 shape's
# immHalfSplitoff immediate operand. Compile-only (emitted PTX + ptxas);
# runtime TMEM evidence needs datacenter Blackwell and lives in gpu/.

function _t5_ldst_surface!(out::Core.LLVMPtr{UInt32, 1}, taddr::UInt32)
    s = UInt32(0)

    v = ptx"tcgen05.ld.sync.aligned.16x64b.x2.b32"(taddr)
    s += v[1]
    ptx"tcgen05.st.sync.aligned.16x64b.x2.b32"(taddr, v)
    p = ptx"tcgen05.ld.sync.aligned.16x64b.x2.pack::16b.b32"(taddr)
    s += p[1]
    ptx"tcgen05.st.sync.aligned.16x64b.x2.unpack::16b.b32"(taddr, p)

    v32 = ptx"tcgen05.ld.sync.aligned.32x32b.x2.pack::16b.b32"(taddr)
    s += v32[1]
    ptx"tcgen05.st.sync.aligned.32x32b.x2.unpack::16b.b32"(taddr, v32)

    v128 = ptx"tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32"(taddr)
    s += v128[1]
    ptx"tcgen05.st.sync.aligned.16x128b.x2.unpack::16b.b32"(taddr, v128)

    v256 = ptx"tcgen05.ld.sync.aligned.16x256b.x2.pack::16b.b32"(taddr)
    s += v256[1]
    ptx"tcgen05.st.sync.aligned.16x256b.x2.unpack::16b.b32"(taddr, v256)

    # Two distinct split offsets pin the immediate's rendering position.
    b2 = ptx"tcgen05.ld.sync.aligned.16x32bx2.x2.b32"(taddr, Val(8))
    s += b2[1]
    ptx"tcgen05.st.sync.aligned.16x32bx2.x2.b32"(taddr, Val(8), b2)
    b2p = ptx"tcgen05.ld.sync.aligned.16x32bx2.x2.pack::16b.b32"(taddr, Val(16))
    s += b2p[1]
    ptx"tcgen05.st.sync.aligned.16x32bx2.x2.unpack::16b.b32"(taddr, Val(16), b2p)

    ptx"tcgen05.wait::ld.sync.aligned"()
    ptx"tcgen05.wait::st.sync.aligned"()
    ptx"st.global.b32"(out, s)
    return nothing
end

@testset "tcgen05 ld/st pack and 16x32bx2 at the sm_100a floor" begin
    types = Tuple{Core.LLVMPtr{UInt32, 1}, UInt32}
    @test ptxas_compiles(_t5_ldst_surface!, types;
                         cap = v"10.0", feature_set = :arch)

    ptx = emit_ptx(_t5_ldst_surface!, types; cap = v"10.0",
                   feature_set = :arch)
    @test occursin(".target sm_100a", ptx)

    # Plain vs pack/unpack: the qualifier sits between .num and .b32.
    @test occursin(r"tcgen05\.ld\.sync\.aligned\.16x64b\.x2\.b32", ptx)
    @test occursin(r"tcgen05\.st\.sync\.aligned\.16x64b\.x2\.b32", ptx)
    for shape in ("16x64b", "32x32b", "16x128b", "16x256b")
        @test occursin(
            Regex("tcgen05\\.ld\\.sync\\.aligned\\.$shape\\.x2\\.pack::16b\\.b32"),
            ptx)
        @test occursin(
            Regex("tcgen05\\.st\\.sync\\.aligned\\.$shape\\.x2\\.unpack::16b\\.b32"),
            ptx)
    end

    # 16x32bx2: immHalfSplitoff renders as a trailing immediate on ld and
    # between the address and the data vector on st.
    @test occursin(
        r"tcgen05\.ld\.sync\.aligned\.16x32bx2\.x2\.b32 \{[^}]+\}, \[%r\d+\], 8;",
        ptx)
    @test occursin(
        r"tcgen05\.st\.sync\.aligned\.16x32bx2\.x2\.b32 \[%r\d+\], 8, \{[^}]+\};",
        ptx)
    @test occursin(
        r"tcgen05\.ld\.sync\.aligned\.16x32bx2\.x2\.pack::16b\.b32 \{[^}]+\}, \[%r\d+\], 16;",
        ptx)
    @test occursin(
        r"tcgen05\.st\.sync\.aligned\.16x32bx2\.x2\.unpack::16b\.b32 \[%r\d+\], 16, \{[^}]+\};",
        ptx)
end
