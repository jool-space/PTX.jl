# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc==10|cc==11|cc==12
#
# PTX 9.3 §9.7.15.5.15: optional 4-/6-bit to 8-bit decompression is a
# Blackwell-family ldmatrix feature, and PTX ISA 9.4 adds the sign-extending
# .s8.s4 form. Offline compilation pins the complete typed form matrix at
# each admitting target; eligible live devices additionally check the
# decompression semantics.

function _ldmatrix_decompression_surface!(out::CuDeviceVector{UInt32,1})
    buf = CuStaticSharedArray(UInt8, 512)
    addr = pointer(buf)

    a1 = ptx"ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b4x16_p64"(addr)
    a2 = ptx"ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64"(addr)
    a4 = ptx"ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64"(addr)
    b1 = ptx"ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b6x16_p32"(addr)
    b2 = ptx"ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b6x16_p32"(addr)
    b4 = ptx"ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b6x16_p32"(addr)
    c1 = ptx"ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16.b4x16_p64"(addr)
    c2 = ptx"ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8x16.b4x16_p64"(addr)
    d1 = ptx"ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16.b6x16_p32"(addr)
    d2 = ptx"ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8x16.b6x16_p32"(addr)

    e1 = ptx"ldmatrix.sync.aligned.m8n16.x1.shared::cta.b8x16.b4x16_p64"(addr)
    e2 = ptx"ldmatrix.sync.aligned.m8n16.x2.shared::cta.b8x16.b4x16_p64"(addr)
    e4 = ptx"ldmatrix.sync.aligned.m8n16.x4.shared::cta.b8x16.b4x16_p64"(addr)
    f1 = ptx"ldmatrix.sync.aligned.m8n16.x1.shared::cta.b8x16.b6x16_p32"(addr)
    f2 = ptx"ldmatrix.sync.aligned.m8n16.x2.shared::cta.b8x16.b6x16_p32"(addr)
    f4 = ptx"ldmatrix.sync.aligned.m8n16.x4.shared::cta.b8x16.b6x16_p32"(addr)
    g1 = ptx"ldmatrix.sync.aligned.m16n16.x1.trans.shared::cta.b8x16.b4x16_p64"(addr)
    g2 = ptx"ldmatrix.sync.aligned.m16n16.x2.trans.shared::cta.b8x16.b4x16_p64"(addr)
    h1 = ptx"ldmatrix.sync.aligned.m16n16.x1.trans.shared::cta.b8x16.b6x16_p32"(addr)
    h2 = ptx"ldmatrix.sync.aligned.m16n16.x2.trans.shared::cta.b8x16.b6x16_p32"(addr)

    plain = a1 + a2[1] + a2[2] + a4[1] + a4[4] +
        b1 + b2[1] + b2[2] + b4[1] + b4[4] +
        c1[1] + c1[2] + c2[1] + c2[4] +
        d1[1] + d1[2] + d2[1] + d2[4]
    cta = e1 + e2[1] + e2[2] + e4[1] + e4[4] +
        f1 + f2[1] + f2[2] + f4[1] + f4[4] +
        g1[1] + g1[2] + g2[1] + g2[4] +
        h1[1] + h1[2] + h2[1] + h2[4]
    @inbounds out[1] = plain + cta
    return nothing
end

@testset "ldmatrix decompression ptxas matrix at Blackwell family roots" begin
    types = Tuple{CuDeviceVector{UInt32,1}}
    # CUDACore's PTX 9.3 target vocabulary currently exposes the 10.x and
    # 12.x roots. The ISA also names sm_110a; backend selection for every
    # intrinsic is separately pinned at sm_100a by host conformance probes.
    for cap in (v"10.0", v"12.0")
        @test ptxas_compiles(_ldmatrix_decompression_surface!, types;
                             cap, feature_set = :arch)
        ptx = emit_ptx(_ldmatrix_decompression_surface!, types;
                       cap, feature_set = :arch)
        @test count("ldmatrix.sync.aligned.m8n16", ptx) == 12
        @test count("ldmatrix.sync.aligned.m16n16", ptx) == 8
        @test occursin(".shared::cta.b8x16.b4x16_p64", ptx)
        @test occursin(".shared::cta.b8x16.b6x16_p32", ptx)
    end
end

function _ldmatrix_decompression_runtime!(out::CuDeviceVector{UInt32,1})
    packed4 = CuStaticSharedArray(UInt8, 512)
    packed6 = CuStaticSharedArray(UInt8, 512)
    lane = Int(ptx"mov.u32"(sreg"tid.x"))
    base = lane * 16
    @inbounds for i in 0:15
        # 16 packed elements followed by the format-named row padding:
        # 8 payload bytes + 8 padding for b4; 12 + 4 for b6.
        packed4[base + i + 1] = i < 8 ? UInt8(0xff) : UInt8(0)
        packed6[base + i + 1] = i < 12 ? UInt8(0xff) : UInt8(0)
    end
    sync_threads()

    addr4 = pointer(packed4) + base
    addr6 = pointer(packed6) + base
    a = ptx"ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b4x16_p64"(addr4)
    b = ptx"ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b6x16_p32"(addr6)
    c = ptx"ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16.b4x16_p64"(addr4)
    d = ptx"ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16.b6x16_p32"(addr6)
    ok = a == UInt32(0x0f0f0f0f) && b == UInt32(0x3f3f3f3f) &&
         c[1] == UInt32(0x0f0f0f0f) && c[2] == UInt32(0x0f0f0f0f) &&
         d[1] == UInt32(0x3f3f3f3f) && d[2] == UInt32(0x3f3f3f3f)
    @inbounds out[lane + 1] = UInt32(ok)
    return nothing
end

if test_runtime_supported(@__FILE__)
    @testset "ldmatrix b4/b6 decompression semantics" begin
        out = CUDACore.zeros(UInt32, 32)
        @cuda threads=32 _ldmatrix_decompression_runtime!(out)
        CUDACore.synchronize()
        @test all(Array(out) .== UInt32(1))
    end
end

# --- .s8.s4 sign-extending decompression (PTX ISA 9.4) -------------------------
# Signed 4-bit elements are expanded to signed 8-bit during the load. Every
# byte of shared memory carries the same nibble pair, so each lane's fragment
# must be that nibble sign-extended four times regardless of the fragment
# layout; three fills cover a negative, a positive, and the most negative
# nibble. The form assembles on sm_90a and the sm_100f/sm_110f/sm_120f
# families; baseline sm_90 is not an admitting target.

function _ldmatrix_s4_runtime!(out::CuDeviceVector{UInt32, 1}, fill::UInt8)
    buf = CuStaticSharedArray(UInt8, 512)
    lane = Int(ptx"mov.u32"(sreg"tid.x"))
    @inbounds for i in 0:15
        buf[lane * 16 + i + 1] = fill
    end
    sync_threads()
    addr = pointer(buf) + lane * 16
    a = ptx"ldmatrix.sync.aligned.m8n16.x1.shared.s8.s4"(addr)
    b = ptx"ldmatrix.sync.aligned.m8n16.x2.shared.s8.s4"(addr)
    c = ptx"ldmatrix.sync.aligned.m8n16.x4.shared.s8.s4"(addr)
    d = ptx"ldmatrix.sync.aligned.m8n16.x1.shared::cta.s8.s4"(addr)
    e = ptx"ldmatrix.sync.aligned.m8n16.x2.shared::cta.s8.s4"(addr)
    f = ptx"ldmatrix.sync.aligned.m8n16.x4.shared::cta.s8.s4"(addr)
    base = lane * 14
    @inbounds begin
        out[base + 1] = a
        out[base + 2] = b[1]; out[base + 3] = b[2]
        out[base + 4] = c[1]; out[base + 5] = c[2]
        out[base + 6] = c[3]; out[base + 7] = c[4]
        out[base + 8] = d
        out[base + 9] = e[1]; out[base + 10] = e[2]
        out[base + 11] = f[1]; out[base + 12] = f[2]
        out[base + 13] = f[3]; out[base + 14] = f[4]
    end
    return nothing
end

const _LDMATRIX_S4_TYPES = Tuple{CuDeviceVector{UInt32, 1}, UInt8}

@testset "ldmatrix .m8n16 .s8.s4 assembles on sm_90a and the sm_100f+ families" begin
    if _ptxas_isa() < v"9.4"
        @test_skip "PTX 9.4 assembler required"
    else
        for (cap, feature_set) in ((v"9.0", :arch), (v"10.0", :family),
                                   (v"12.0", :family), (v"12.1", :arch))
            @test ptxas_compiles(_ldmatrix_s4_runtime!, _LDMATRIX_S4_TYPES;
                                 cap, feature_set)
        end
        ptx = emit_ptx(_ldmatrix_s4_runtime!, _LDMATRIX_S4_TYPES;
                       cap = v"9.0", feature_set = :arch)
        for count in ("x1", "x2", "x4"), space in ("shared", "shared::cta")
            @test occursin("ldmatrix.sync.aligned.m8n16.$count.$space.s8.s4 ", ptx)
        end
        @test ptxas_rejects(_ldmatrix_s4_runtime!, _LDMATRIX_S4_TYPES;
                            cap = v"9.0", target = "sm_90")
        @test ptxas_rejects(_ldmatrix_s4_runtime!, _LDMATRIX_S4_TYPES;
                            cap = v"8.0", target = "sm_80")
    end
end

if test_runtime_supported(@__FILE__)
    @testset "ldmatrix .s8.s4 sign-extends 4-bit elements" begin
        for (fill, expected) in ((0xff, 0xffffffff),   # -1 -> 0xff
                                 (0x77, 0x07070707),   #  7 -> 0x07
                                 (0x88, 0xf8f8f8f8))   # -8 -> 0xf8
            out = CUDACore.zeros(UInt32, 32 * 14)
            @cuda threads=32 _ldmatrix_s4_runtime!(out, UInt8(fill))
            CUDACore.synchronize()
            @test all(Array(out) .== UInt32(expected))
        end
    end
end
