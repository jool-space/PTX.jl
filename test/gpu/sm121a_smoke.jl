# REQUIRES CC ==12.1
#
# Hardware smoke test for the consumer-Blackwell sub-byte FP tensor core.
# Mirrors `_exec_mma_bf16_smoke!` at test/gpu/exec.jl:158 — A=B=0, identity C,
# every lane should observe d == c. Verifies that the sm_121a-specific
# `mma.sync.aligned.kind::f8f6f4` op compiles, launches, and that the I/O
# register layout matches NVPTX's expectations on real silicon.
#
# Exact-match cap gate (`==12.1`): the `kind::f8f6f4` mma is arch-specific
# (locked to one CC), so this file deliberately does not try to run on
# sm_100a / sm_103a / sm_120a. Phase 2 will introduce per-arch test files
# as those targets come online.

function _sm121a_mma_kind_f8f6f4_smoke!(out)
    a = (UInt32(0), UInt32(0), UInt32(0), UInt32(0))
    b = (UInt32(0), UInt32(0))
    tid = ptx"mov.u32"(sreg"tid.x")
    base = Float32(tid) * 4f0
    c = (base, base + 1f0, base + 2f0, base + 3f0)
    d = ptx"mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32"(a, b, c)
    off = Int(tid) * 4
    @inbounds out[off + 1] = d[1]
    @inbounds out[off + 2] = d[2]
    @inbounds out[off + 3] = d[3]
    @inbounds out[off + 4] = d[4]
    return nothing
end

@testset "mma.sync.aligned.kind::f8f6f4 m16n8k32 e4m3 smoke (a=b=0 → d == c)" begin
    out = CUDACore.zeros(Float32, 32 * 4)
    @cuda threads=32 _sm121a_mma_kind_f8f6f4_smoke!(out)
    CUDACore.synchronize()
    got = Array(out)
    expected = [Float32(t) * 4f0 + Float32(j) for t in 0:31 for j in 0:3]
    @test got == expected
end

# Real correctness: A=B=1.0 (e4m3 1.0 bit pattern 0x38 packed × 4 = 0x38383838),
# C=0. Every output element of the 16×8 result is a 32-term dot product of
# 1·1 → 32.0. Uniform fragments mean lane→element layout doesn't matter for
# this check — only that the op flows data end-to-end.

function _sm121a_mma_kind_f8f6f4_correct!(out)
    one4 = UInt32(0x3838_3838)
    a = (one4, one4, one4, one4)
    b = (one4, one4)
    c = (0f0, 0f0, 0f0, 0f0)
    d = ptx"mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32"(a, b, c)
    tid = ptx"mov.u32"(sreg"tid.x")
    off = Int(tid) * 4
    @inbounds out[off + 1] = d[1]
    @inbounds out[off + 2] = d[2]
    @inbounds out[off + 3] = d[3]
    @inbounds out[off + 4] = d[4]
    return nothing
end

@testset "mma.sync.aligned.kind::f8f6f4 e4m3 correctness (A=B=1.0, C=0 → D==32.0)" begin
    out = CUDACore.zeros(Float32, 32 * 4)
    @cuda threads=32 _sm121a_mma_kind_f8f6f4_correct!(out)
    CUDACore.synchronize()
    @test all(Array(out) .== 32.0f0)
end

# m16n8k16 mirror — same uniform-1.0 pattern, K halves so D == 16.0.
function _sm121a_mma_kind_f8f6f4_k16_correct!(out)
    one4 = UInt32(0x3838_3838)
    a = (one4, one4)
    b = (one4,)
    c = (0f0, 0f0, 0f0, 0f0)
    d = ptx"mma.sync.aligned.kind::f8f6f4.m16n8k16.row.col.f32.e4m3.e4m3.f32"(a, b, c)
    tid = ptx"mov.u32"(sreg"tid.x")
    off = Int(tid) * 4
    @inbounds out[off + 1] = d[1]
    @inbounds out[off + 2] = d[2]
    @inbounds out[off + 3] = d[3]
    @inbounds out[off + 4] = d[4]
    return nothing
end

@testset "mma.sync.aligned.kind::f8f6f4 m16n8k16 e4m3 correctness (A=B=1.0 → D==16.0)" begin
    out = CUDACore.zeros(Float32, 32 * 4)
    @cuda threads=32 _sm121a_mma_kind_f8f6f4_k16_correct!(out)
    CUDACore.synchronize()
    @test all(Array(out) .== 16.0f0)
end

# --- block-scaled correctness ---------------------------------------------
#
# Bit-pattern table for unit values (per IEEE-style encodings used by PTX):
#   e4m3(1.0)  = 0x38 (exp=7, mant=0; bias 7)
#   e2m1(1.0)  = 0x2  (4-bit nibble: sign=0, exp=01, mant=0; bias 1)
#   ue8m0(1.0) = 0x7F (exp=127; bias 127, no mantissa)
#   ue4m3(1.0) = 0x38 (exp=7, mant=0; bias 7)
#
# Block-scaled mma reads the per-lane scale values via the (byte-id, thread-id)
# selectors. With every byte of scale-a-data/scale-b-data set to 1.0 across
# every lane, any selector picks a 1.0 — the result is exactly A·B per K.

function _sm121a_mma_kind_mxf8f6f4_correct!(out)
    one4 = UInt32(0x3838_3838)               # e4m3(1.0) × 4
    a = (one4, one4, one4, one4)
    b = (one4, one4)
    c = (0f0, 0f0, 0f0, 0f0)
    sa = UInt32(0x7F7F_7F7F)                 # ue8m0(1.0) × 4
    sb = UInt32(0x7F7F_7F7F)
    d = ptx"mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0"(
        a, b, c, sa, UInt16(0), UInt16(0), sb, UInt16(0), UInt16(0))
    tid = ptx"mov.u32"(sreg"tid.x")
    off = Int(tid) * 4
    @inbounds out[off + 1] = d[1]
    @inbounds out[off + 2] = d[2]
    @inbounds out[off + 3] = d[3]
    @inbounds out[off + 4] = d[4]
    return nothing
end

@testset "mma.sync.aligned.kind::mxf8f6f4 e4m3 ue8m0 correctness (scale=1 → D==32.0)" begin
    out = CUDACore.zeros(Float32, 32 * 4)
    @cuda threads=32 _sm121a_mma_kind_mxf8f6f4_correct!(out)
    CUDACore.synchronize()
    @test all(Array(out) .== 32.0f0)
end

function _sm121a_mma_kind_mxf4_correct!(out)
    one8 = UInt32(0x2222_2222)               # e2m1(1.0) × 8 (4-bit nibbles)
    a = (one8, one8, one8, one8)
    b = (one8, one8)
    c = (0f0, 0f0, 0f0, 0f0)
    sa = UInt32(0x7F7F_7F7F)                 # ue8m0(1.0) × 4
    sb = UInt32(0x7F7F_7F7F)
    d = ptx"mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0"(
        a, b, c, sa, UInt16(0), UInt16(0), sb, UInt16(0), UInt16(0))
    tid = ptx"mov.u32"(sreg"tid.x")
    off = Int(tid) * 4
    @inbounds out[off + 1] = d[1]
    @inbounds out[off + 2] = d[2]
    @inbounds out[off + 3] = d[3]
    @inbounds out[off + 4] = d[4]
    return nothing
end

@testset "mma.sync.aligned.kind::mxf4 e2m1 ue8m0 correctness (scale=1 → D==64.0)" begin
    out = CUDACore.zeros(Float32, 32 * 4)
    @cuda threads=32 _sm121a_mma_kind_mxf4_correct!(out)
    CUDACore.synchronize()
    @test all(Array(out) .== 64.0f0)
end

function _sm121a_mma_kind_mxf4nvf4_correct!(out)
    one8 = UInt32(0x2222_2222)               # e2m1(1.0) × 8
    a = (one8, one8, one8, one8)
    b = (one8, one8)
    c = (0f0, 0f0, 0f0, 0f0)
    sa = UInt32(0x3838_3838)                 # ue4m3(1.0) × 4
    sb = UInt32(0x3838_3838)
    d = ptx"mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3"(
        a, b, c, sa, UInt16(0), UInt16(0), sb, UInt16(0), UInt16(0))
    tid = ptx"mov.u32"(sreg"tid.x")
    off = Int(tid) * 4
    @inbounds out[off + 1] = d[1]
    @inbounds out[off + 2] = d[2]
    @inbounds out[off + 3] = d[3]
    @inbounds out[off + 4] = d[4]
    return nothing
end

@testset "mma.sync.aligned.kind::mxf4nvf4 e2m1 ue4m3 correctness (scale=1 → D==64.0)" begin
    out = CUDACore.zeros(Float32, 32 * 4)
    @cuda threads=32 _sm121a_mma_kind_mxf4nvf4_correct!(out)
    CUDACore.synchronize()
    @test all(Array(out) .== 64.0f0)
end
