# REQUIRES CC 10.0
import CUDACore: CUDACompilerJob, GPUCompiler
using CUDACore.LLVM

pack_f32x2(lo::Float32, hi::Float32) =
    (UInt64(reinterpret(UInt32, hi)) << 32) |
     UInt64(reinterpret(UInt32, lo))

unpack_lo(p::UInt64) = reinterpret(Float32, UInt32(p & 0xFFFFFFFF))
unpack_hi(p::UInt64) = reinterpret(Float32, UInt32(p >> 32))

function _add_f32x2!(out::AbstractArray{UInt64},
                     xs::AbstractArray{UInt64},
                     ys::AbstractArray{UInt64})
    tid  = ptx"mov.u32"(sreg"tid.x")
    bid  = ptx"mov.u32"(sreg"ctaid.x")
    ntid = ptx"mov.u32"(sreg"ntid.x")
    i = Int(bid) * Int(ntid) + Int(tid) + 1
    if i <= length(xs)
        a = @inbounds xs[i]
        b = @inbounds ys[i]
        @inbounds out[i] = ptx"add.f32x2"(a, b)
    end
    return nothing
end

@testset "add.f32x2 ≡ scalar pair-wise add" begin
    n  = 256
    xs = UInt64[pack_f32x2(Float32(2i),     Float32(2i + 1))   for i in 1:n]
    ys = UInt64[pack_f32x2(Float32(0.5i), Float32(-0.25i))   for i in 1:n]

    out_d = CUDACore.zeros(UInt64, n)
    threads = min(n, 256); blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks _add_f32x2!(out_d, CuArray(xs), CuArray(ys))
    CUDACore.synchronize()

    expected = UInt64[pack_f32x2(unpack_lo(xs[i]) + unpack_lo(ys[i]),
                                  unpack_hi(xs[i]) + unpack_hi(ys[i]))
                       for i in 1:n]
    @test Array(out_d) == expected
end
