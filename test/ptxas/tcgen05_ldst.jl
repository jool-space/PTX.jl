# Exact-floor compiler evidence for the tcgen05 TMEM data-movement surface:
# PTX 9.3 §9.7.17.8 ld/st (the pack::16b/unpack::16b qualifier rendering and
# the 16x32bx2 shape's immHalfSplitoff immediate operand), §9.7.17.9 cp (the
# multicast-mandatory shapes and the optional b8x16 decompression pair),
# ld.red at its sm_103f family floor, and the sm_107a ld{.red}.spcompress
# family. Compile-only (emitted PTX + ptxas); runtime TMEM evidence needs
# datacenter Blackwell and lives in gpu/. ptxas rejects mixing .cta_group::1
# and ::2 in one function, so the cp surface splits by group.

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

# --- cp: multicast shapes and decompression at the sm_100a floor -------------

function _t5_cp_surface_cg1!(taddr::UInt32, s_desc::UInt64)
    ptx"tcgen05.cp.cta_group::1.128x128b"(taddr, s_desc)
    ptx"tcgen05.cp.cta_group::1.64x128b.warpx2::02_13"(taddr, s_desc)
    ptx"tcgen05.cp.cta_group::1.32x128b.warpx4"(taddr, s_desc)
    ptx"tcgen05.cp.cta_group::1.128x256b.b8x16.b6x16_p32"(taddr, s_desc)
    ptx"tcgen05.cp.cta_group::1.4x256b.b8x16.b4x16_p64"(taddr, s_desc)
    return nothing
end

function _t5_cp_surface_cg2!(taddr::UInt32, s_desc::UInt64)
    ptx"tcgen05.cp.cta_group::2.64x128b.warpx2::01_23"(taddr, s_desc)
    ptx"tcgen05.cp.cta_group::2.32x128b.warpx4.b8x16.b6x16_p32"(taddr, s_desc)
    return nothing
end

@testset "tcgen05.cp multicast and decompression at the sm_100a floor" begin
    types = Tuple{UInt32, UInt64}
    @test ptxas_compiles(_t5_cp_surface_cg1!, types;
                         cap = v"10.0", feature_set = :arch)
    @test ptxas_compiles(_t5_cp_surface_cg2!, types;
                         cap = v"10.0", feature_set = :arch)

    ptx1 = emit_ptx(_t5_cp_surface_cg1!, types; cap = v"10.0",
                    feature_set = :arch)
    ptx2 = emit_ptx(_t5_cp_surface_cg2!, types; cap = v"10.0",
                    feature_set = :arch)
    @test occursin(".target sm_100a", ptx1)
    for (ptx, spell) in ((ptx1, "cta_group::1.128x128b"),
                         (ptx1, "cta_group::1.64x128b.warpx2::02_13"),
                         (ptx1, "cta_group::1.32x128b.warpx4"),
                         (ptx1, "cta_group::1.128x256b.b8x16.b6x16_p32"),
                         (ptx1, "cta_group::1.4x256b.b8x16.b4x16_p64"),
                         (ptx2, "cta_group::2.64x128b.warpx2::01_23"),
                         (ptx2, "cta_group::2.32x128b.warpx4.b8x16.b6x16_p32"))
        @test occursin(
            Regex("tcgen05\\.cp\\." * replace(spell, "." => "\\.") *
                  "\\s+\\[%r\\d+\\], %rd\\d+;"),
            ptx)
    end
end

# --- ld{.red}.spcompress: sm_107a assembles; sm_107f and sm_100a refuse -------
# Every registered form, results folded so each load stays live.

@generated function _t5_ldspc_surface!(out::CuDeviceVector{UInt32, 1},
                                       taddr::UInt32)
    body = Expr(:block, :(acc = UInt32(0)), :(facc = 0.0f0))
    for mods in PTX.wrapper_asm_forms(:tcgen05_ldspc)
        red = mods[2] === :red
        push!(body.args, :(r = PTX.Operation{:tcgen05, $mods}()(taddr)))
        push!(body.args, :(acc ⊻= r[1][1] ⊻ r[2][1]))
        red && push!(body.args, :(facc += r[3]))
    end
    push!(body.args, :(ptx"tcgen05.wait::ld.sync.aligned"()))
    push!(body.args, :(Base.@inbounds out[1] = acc ⊻ reinterpret(UInt32, facc)))
    push!(body.args, :(return nothing))
    body
end

@testset "tcgen05.ld.spcompress at the sm_107a floor" begin
    types = Tuple{CuDeviceVector{UInt32, 1}, UInt32}
    if _ptxas_isa() < v"9.4"
        @test_skip "PTX 9.4 assembler required"
    else
        @test ptxas_compiles(_t5_ldspc_surface!, types;
                             cap = v"10.7", feature_set = :arch)
        ptx = emit_ptx(_t5_ldspc_surface!, types;
                       cap = v"10.7", feature_set = :arch)
        @test occursin(".target sm_107a", ptx)
        for mods in PTX.wrapper_asm_forms(:tcgen05_ldspc)
            @test occursin(PTX.build_head(:tcgen05, mods) * " {", ptx)
        end
        @test occursin(r"tcgen05\.ld\.red\.spcompress\.sync\.aligned\.32x32b\.x4\.max\.sp::2:4\.abs\.NaN\.f32\.b2 \{%r\d+\}, \{%r\d+, %r\d+\}, %r\d+, \[%r\d+\];",
                       ptx)
        # The `.sp::2:4` qualifier is a-variant-exclusive: the family target
        # and the sm_100a root both refuse it.
        @test ptxas_rejects(_t5_ldspc_surface!, types; cap = v"10.7",
                            feature_set = :family, target = "sm_107f")
        @test ptxas_rejects(_t5_ldspc_surface!, types; cap = v"10.0",
                            feature_set = :arch, target = "sm_100a")
    end
end
