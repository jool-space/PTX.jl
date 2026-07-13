# TEST_TARGET: requires=gpu evidence=runtime target=sm_89
#
# mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 on Ada (sm_89+).
# Single warp, single block.
#
# m16n8k32 FP8 fragment layout (PTX ISA 9.2 §9.7.14.5).
# Per lane T:
#   group    = T >> 2    (0..7)
#   in_group = T & 0x3   (0..3)
# A (16×32 e4m3 row-major), NTuple{4, UInt32}, 4 e4m3 packed per UInt32
# (low byte = lowest column index):
#   a[1] = A[group,     4*in_group + 0..3]
#   a[2] = A[group + 8, 4*in_group + 0..3]
#   a[3] = A[group,     4*in_group + 16..19]
#   a[4] = A[group + 8, 4*in_group + 16..19]
# B (32×8 e4m3 col-major), NTuple{2, UInt32}:
#   b[1] = B[4*in_group + 0..3,   group]
#   b[2] = B[4*in_group + 16..19, group]
# D (16×8 f32), NTuple{4, Float32} same as bf16/f16 case:
#   d[1] = D[group,     2*in_group + 0]
#   d[2] = D[group,     2*in_group + 1]
#   d[3] = D[group + 8, 2*in_group + 0]
#   d[4] = D[group + 8, 2*in_group + 1]

# --- bit-packed correctness ------------------------------------------------
#
# e4m3(1.0): bias 7, exp=7 → 0_0111_000 = 0x38. Four packed in a UInt32
# = 0x38383838. With A=B=1.0 every output element is a 32-term dot product
# of 1*1 → 32.0 exactly.

function _exec_mma_e4m3_bitpacked!(out)
    one4 = UInt32(0x3838_3838)
    a = (one4, one4, one4, one4)
    b = (one4, one4)
    c = (0f0, 0f0, 0f0, 0f0)
    d = ptx"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"(a, b, c)
    tid = ptx"mov.u32"(sreg"tid.x")
    off = Int(tid) * 4
    @inbounds out[off + 1] = d[1]
    @inbounds out[off + 2] = d[2]
    @inbounds out[off + 3] = d[3]
    @inbounds out[off + 4] = d[4]
    return nothing
end

@testset "mma.sync e4m3 (A=B=1.0, C=0 → D==32.0)" begin
    out = CUDACore.zeros(Float32, 32 * 4)
    @cuda threads=32 _exec_mma_e4m3_bitpacked!(out)
    CUDACore.synchronize()
    @test all(Array(out) .== 32.0f0)
end

# --- Float32 → e4m3 fragments via cvt --------------------------------------
#
# cvt.rn.satfinite.e4m3x2.f32(a, b) packs `a` in HIGH byte, `b` in LOW byte
# (verified by exec.jl test). To get low-to-high byte ordering [e0, e1, e2, e3]
# in a UInt32, pack pairs (e1, e0) and (e3, e2) into UInt16 then concat.

@inline function pack4_e4m3(e0::Float32, e1::Float32, e2::Float32, e3::Float32)
    p01 = ptx"cvt.rn.satfinite.e4m3x2.f32"(e1, e0)   # LO byte = e0
    p23 = ptx"cvt.rn.satfinite.e4m3x2.f32"(e3, e2)   # LO byte = e2
    UInt32(p01) | (UInt32(p23) << 16)
end

function _exec_mma_e4m3_from_f32!(D, A, B)
    tid = ptx"mov.u32"(sreg"tid.x")
    g  = Int(tid >> UInt32(2))
    ig = Int(tid & UInt32(0x3))

    # A row-major flat: A[row, col] at row*32 + col + 1 (1-indexed)
    a1 = pack4_e4m3(@inbounds(A[g*32     + 4*ig + 1]),
                    @inbounds(A[g*32     + 4*ig + 2]),
                    @inbounds(A[g*32     + 4*ig + 3]),
                    @inbounds(A[g*32     + 4*ig + 4]))
    a2 = pack4_e4m3(@inbounds(A[(g+8)*32 + 4*ig + 1]),
                    @inbounds(A[(g+8)*32 + 4*ig + 2]),
                    @inbounds(A[(g+8)*32 + 4*ig + 3]),
                    @inbounds(A[(g+8)*32 + 4*ig + 4]))
    a3 = pack4_e4m3(@inbounds(A[g*32     + 4*ig + 17]),
                    @inbounds(A[g*32     + 4*ig + 18]),
                    @inbounds(A[g*32     + 4*ig + 19]),
                    @inbounds(A[g*32     + 4*ig + 20]))
    a4 = pack4_e4m3(@inbounds(A[(g+8)*32 + 4*ig + 17]),
                    @inbounds(A[(g+8)*32 + 4*ig + 18]),
                    @inbounds(A[(g+8)*32 + 4*ig + 19]),
                    @inbounds(A[(g+8)*32 + 4*ig + 20]))

    # B col-major flat: B[row, col] at col*32 + row + 1
    b1 = pack4_e4m3(@inbounds(B[g*32 + 4*ig + 1]),
                    @inbounds(B[g*32 + 4*ig + 2]),
                    @inbounds(B[g*32 + 4*ig + 3]),
                    @inbounds(B[g*32 + 4*ig + 4]))
    b2 = pack4_e4m3(@inbounds(B[g*32 + 4*ig + 17]),
                    @inbounds(B[g*32 + 4*ig + 18]),
                    @inbounds(B[g*32 + 4*ig + 19]),
                    @inbounds(B[g*32 + 4*ig + 20]))

    a = (a1, a2, a3, a4)
    b = (b1, b2)
    c = (0f0, 0f0, 0f0, 0f0)
    d = ptx"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"(a, b, c)

    @inbounds D[g*8     + 2*ig + 1] = d[1]
    @inbounds D[g*8     + 2*ig + 2] = d[2]
    @inbounds D[(g+8)*8 + 2*ig + 1] = d[3]
    @inbounds D[(g+8)*8 + 2*ig + 2] = d[4]
    return nothing
end

# Pack a Float32 matrix row-major into a flat Vector{Float32}.
pack_row(M::AbstractMatrix{Float32}) = vec(permutedims(M))
# Column-major is Julia default.
pack_col(M::AbstractMatrix{Float32}) = vec(M)

@testset "mma.sync e4m3 from Float32 (A=B=2.0 → D==128.0)" begin
    A_f32 = fill(2.0f0, 16, 32)
    B_f32 = fill(2.0f0, 32, 8)
    A = CuArray(pack_row(A_f32))
    B = CuArray(pack_col(B_f32))
    D = CUDACore.zeros(Float32, 16 * 8)
    @cuda threads=32 _exec_mma_e4m3_from_f32!(D, A, B)
    CUDACore.synchronize()
    @test all(Array(D) .== 128.0f0)
end

@testset "mma.sync e4m3 from Float32: non-trivial A, B" begin
    # Small integers 1..4 are exactly representable in e4m3, so the 32-term
    # dot product matches the f32 reference exactly.
    A_f32 = Float32[mod(i + j, 4) + 1 for i in 1:16, j in 1:32]
    B_f32 = Float32[mod(i - j, 4) + 1 for i in 1:32, j in 1:8]
    A = CuArray(pack_row(A_f32))
    B = CuArray(pack_col(B_f32))
    D = CUDACore.zeros(Float32, 16 * 8)
    @cuda threads=32 _exec_mma_e4m3_from_f32!(D, A, B)
    CUDACore.synchronize()
    Dgot = reshape(Array(D), 8, 16)'   # row-major flat → 16×8
    Dref = A_f32 * B_f32
    @test Dgot == Dref
end

# --- E5M2 mirror -----------------------------------------------------------
#
# e5m2(1.0): bias 15, exp=15 → 0_01111_00 = 0x3C. Packed × 4 = 0x3C3C3C3C.
# e5m2(2.0): exp=16 → 0_10000_00 = 0x40. Packed × 4 = 0x40404040.

function _exec_mma_e5m2_bitpacked!(out)
    one4 = UInt32(0x3C3C_3C3C)
    a = (one4, one4, one4, one4)
    b = (one4, one4)
    c = (0f0, 0f0, 0f0, 0f0)
    d = ptx"mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32"(a, b, c)
    tid = ptx"mov.u32"(sreg"tid.x")
    off = Int(tid) * 4
    @inbounds out[off + 1] = d[1]
    @inbounds out[off + 2] = d[2]
    @inbounds out[off + 3] = d[3]
    @inbounds out[off + 4] = d[4]
    return nothing
end

@testset "mma.sync e5m2 (A=B=1.0, C=0 → D==32.0)" begin
    out = CUDACore.zeros(Float32, 32 * 4)
    @cuda threads=32 _exec_mma_e5m2_bitpacked!(out)
    CUDACore.synchronize()
    @test all(Array(out) .== 32.0f0)
end

@inline function pack4_e5m2(e0::Float32, e1::Float32, e2::Float32, e3::Float32)
    p01 = ptx"cvt.rn.satfinite.e5m2x2.f32"(e1, e0)
    p23 = ptx"cvt.rn.satfinite.e5m2x2.f32"(e3, e2)
    UInt32(p01) | (UInt32(p23) << 16)
end

function _exec_mma_e5m2_from_f32!(D, A, B)
    tid = ptx"mov.u32"(sreg"tid.x")
    g  = Int(tid >> UInt32(2))
    ig = Int(tid & UInt32(0x3))

    a1 = pack4_e5m2(@inbounds(A[g*32     + 4*ig + 1]),
                    @inbounds(A[g*32     + 4*ig + 2]),
                    @inbounds(A[g*32     + 4*ig + 3]),
                    @inbounds(A[g*32     + 4*ig + 4]))
    a2 = pack4_e5m2(@inbounds(A[(g+8)*32 + 4*ig + 1]),
                    @inbounds(A[(g+8)*32 + 4*ig + 2]),
                    @inbounds(A[(g+8)*32 + 4*ig + 3]),
                    @inbounds(A[(g+8)*32 + 4*ig + 4]))
    a3 = pack4_e5m2(@inbounds(A[g*32     + 4*ig + 17]),
                    @inbounds(A[g*32     + 4*ig + 18]),
                    @inbounds(A[g*32     + 4*ig + 19]),
                    @inbounds(A[g*32     + 4*ig + 20]))
    a4 = pack4_e5m2(@inbounds(A[(g+8)*32 + 4*ig + 17]),
                    @inbounds(A[(g+8)*32 + 4*ig + 18]),
                    @inbounds(A[(g+8)*32 + 4*ig + 19]),
                    @inbounds(A[(g+8)*32 + 4*ig + 20]))

    b1 = pack4_e5m2(@inbounds(B[g*32 + 4*ig + 1]),
                    @inbounds(B[g*32 + 4*ig + 2]),
                    @inbounds(B[g*32 + 4*ig + 3]),
                    @inbounds(B[g*32 + 4*ig + 4]))
    b2 = pack4_e5m2(@inbounds(B[g*32 + 4*ig + 17]),
                    @inbounds(B[g*32 + 4*ig + 18]),
                    @inbounds(B[g*32 + 4*ig + 19]),
                    @inbounds(B[g*32 + 4*ig + 20]))

    a = (a1, a2, a3, a4)
    b = (b1, b2)
    c = (0f0, 0f0, 0f0, 0f0)
    d = ptx"mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32"(a, b, c)

    @inbounds D[g*8     + 2*ig + 1] = d[1]
    @inbounds D[g*8     + 2*ig + 2] = d[2]
    @inbounds D[(g+8)*8 + 2*ig + 1] = d[3]
    @inbounds D[(g+8)*8 + 2*ig + 2] = d[4]
    return nothing
end

@testset "mma.sync e5m2 from Float32 (A=B=2.0 → D==128.0)" begin
    A_f32 = fill(2.0f0, 16, 32)
    B_f32 = fill(2.0f0, 32, 8)
    A = CuArray(pack_row(A_f32))
    B = CuArray(pack_col(B_f32))
    D = CUDACore.zeros(Float32, 16 * 8)
    @cuda threads=32 _exec_mma_e5m2_from_f32!(D, A, B)
    CUDACore.synchronize()
    @test all(Array(D) .== 128.0f0)
end

@testset "mma.sync e5m2 from Float32: non-trivial A, B" begin
    # Powers of two (1, 2, 4, 8) are exactly representable in e5m2; their
    # 32-term dot products fit in f32 without rounding.
    A_f32 = Float32[2^mod(i + j, 4) for i in 1:16, j in 1:32]
    B_f32 = Float32[2^mod(i - j, 4) for i in 1:32, j in 1:8]
    A = CuArray(pack_row(A_f32))
    B = CuArray(pack_col(B_f32))
    D = CUDACore.zeros(Float32, 16 * 8)
    @cuda threads=32 _exec_mma_e5m2_from_f32!(D, A, B)
    CUDACore.synchronize()
    Dgot = reshape(Array(D), 8, 16)'
    Dref = A_f32 * B_f32
    @test Dgot == Dref
end
