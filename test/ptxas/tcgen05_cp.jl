# Exact-floor compiler evidence for PTX 9.3 §9.7.17.9 tcgen05.cp: the
# multicast-mandatory shapes and the optional b8x16 decompression pair.
# Compile-only (emitted PTX + ptxas); runtime TMEM evidence needs
# datacenter Blackwell and lives in gpu/. ptxas rejects mixing
# .cta_group::1 and ::2 in one function, so the surface splits by group.

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
