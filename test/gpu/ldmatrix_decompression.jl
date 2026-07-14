# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc==10|cc==11|cc==12
#
# PTX 9.3 §9.7.15.5.15: optional 4-/6-bit to 8-bit decompression is a
# Blackwell-family ldmatrix feature. Offline compilation pins the complete
# typed form matrix at each architecture-specific family root; eligible live
# devices additionally check nonzero decompression semantics.

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
