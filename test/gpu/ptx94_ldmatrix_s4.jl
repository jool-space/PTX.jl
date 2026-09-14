# TEST_TARGET: requires=gpu evidence=runtime runtime=cc==10|cc==11|cc==12
# PTX ISA 9.4 `ldmatrix .m8n16 .s8.s4`: signed 4-bit elements are expanded
# to signed 8-bit during the load. Every byte of shared memory carries the
# same nibble pair, so each lane's fragment must be that nibble
# sign-extended four times regardless of the fragment layout; three fills
# cover a negative, a positive, and the most negative nibble.

function _bind_ldmatrix_s4_runtime!(out::CuDeviceVector{UInt32, 1},
                                    fill::UInt8)
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

@testset "ldmatrix .s8.s4 sign-extends 4-bit elements" begin
    for (fill, expected) in ((0xff, 0xffffffff),   # -1 -> 0xff
                             (0x77, 0x07070707),   #  7 -> 0x07
                             (0x88, 0xf8f8f8f8))   # -8 -> 0xf8
        out = CUDACore.zeros(UInt32, 32 * 14)
        @cuda threads=32 _bind_ldmatrix_s4_runtime!(out, UInt8(fill))
        CUDACore.synchronize()
        @test all(Array(out) .== UInt32(expected))
    end
end
