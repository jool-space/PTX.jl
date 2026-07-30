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

# --- ld.red: sm_103f assembles; sm_100 refuses the instruction ---------------
# The ISA's ld.red support list is sm_110a plus the sm_103f/sm_110f
# families — no sm_100 entry at all (and ptxas 13.3 has no sm_110a target),
# so the family floor here is deliberately sm_103f, not the tcgen05
# baseline sm_100a used above.

function _t5_ldred_surface!(out::Core.LLVMPtr{UInt32, 1}, taddr::UInt32)
    s = UInt32(0)
    fsum = 0.0f0

    r2 = ptx"tcgen05.ld.red.sync.aligned.32x32b.x2.min.f32"(taddr)
    s += r2[1]; fsum += r2[3]
    r4 = ptx"tcgen05.ld.red.sync.aligned.32x32b.x4.max.abs.NaN.f32"(taddr)
    s += r4[1]; fsum += r4[5]
    ru = ptx"tcgen05.ld.red.sync.aligned.32x32b.x2.max.u32"(taddr)
    s += ru[3]

    # Two distinct split offsets pin the immediate's rendering position.
    rs = ptx"tcgen05.ld.red.sync.aligned.16x32bx2.x2.min.s32"(taddr, Val(8))
    s += rs[3] % UInt32
    rb = ptx"tcgen05.ld.red.sync.aligned.16x32bx2.x4.max.NaN.f32"(taddr,
                                                                  Val(16))
    fsum += rb[5]

    ptx"tcgen05.wait::ld.sync.aligned"()
    ptx"st.global.b32"(out, s + reinterpret(UInt32, fsum))
    return nothing
end

@testset "tcgen05.ld.red at the sm_103f family floor" begin
    types = Tuple{Core.LLVMPtr{UInt32, 1}, UInt32}
    @test ptxas_compiles(_t5_ldred_surface!, types;
                         cap = v"10.3", feature_set = :family)

    ptx = emit_ptx(_t5_ldred_surface!, types; cap = v"10.3",
                   feature_set = :family)
    @test occursin(".target sm_103f", ptx)
    @test occursin("tcgen05.ld.red.sync.aligned.32x32b.x2.min.f32", ptx)
    @test occursin("tcgen05.ld.red.sync.aligned.32x32b.x4.max.abs.NaN.f32",
                   ptx)
    @test occursin("tcgen05.ld.red.sync.aligned.32x32b.x2.max.u32", ptx)
    @test occursin(r"tcgen05\.ld\.red\.sync\.aligned\.16x32bx2\.x2\.min\.s32 \{%r\d+, %r\d+\}, %r\d+, \[%r\d+\], 8;",
                   ptx)
    # (the f32 redval lives in a .b32 %r register — the backend's unified
    # register file — and ptxas accepts it under the `=f` constraint)
    @test occursin(r"tcgen05\.ld\.red\.sync\.aligned\.16x32bx2\.x4\.max\.NaN\.f32 \{%r\d+, %r\d+, %r\d+, %r\d+\}, %r\d+, \[%r\d+\], 16;",
                   ptx)

    # Feature-level negative: the instruction, not the target, is refused
    # on sm_100 — guards against ever widening this floor.
    err = try
        ptxas_compiles(_t5_ldred_surface!, types;
                       cap = v"10.0", feature_set = :arch)
        nothing
    catch e
        sprint(showerror, e)
    end
    @test err isa String
    @test occursin("Instruction 'tcgen05.ld.red' not supported on .target " *
                   "'sm_100a'", err)
end
