# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=10.0
using Microfloats
using Microfloats: NanOnlyAllOnes, IEEE, SAT, OVF

# Sub-byte / MX FP cvt parity (sm_100+).
#
# Several of these ops (e2m3x2, e3m2x2, ue8m0x2, scaled cvts) are
# arch-specific — ptxas only accepts them under the `a` target suffix
# (e.g. `.target sm_121a`). By default `@cuda` selects the arch-specific
# feature set, so both the `.target` directive and the `--gpu-name`
# ptxas flag carry the `a` suffix matching the live device.

# Microfloats reference types — most sub-byte FP types are `FiniteOnly` so
# their default policy is SAT; ue8m0 needs an explicit SAT twin.
@microfloat E8M0_SAT      sign=0 exponent=8 significand=0 nonfinite=NanOnlyAllOnes overflow=SAT
@microfloat E4M3FN_SAT_X4 exponent=4 significand=3 nonfinite=NanOnlyAllOnes overflow=SAT
@microfloat E5M2_SAT_X4   exponent=5 significand=2 overflow=SAT

function fp_test_values()
    base = Float32[
        0f0, -0f0,
        1f0, -1f0, 2f0, -2f0, 0.5f0, -0.5f0,
        1.25f0, 1.5f0, 1.75f0, 3f0, 6f0,
        1.0625f0, 1.125f0, 1.375f0, 1.625f0, 1.875f0,
        0.125f0, 0.25f0, 0.375f0,
        0.03125f0, 0.0625f0, 0.01f0, 0.001f0,
        7.5f0, 17f0, 100f0, 300f0,
        10f0, -10f0, 50f0, -50f0,
        500f0, 1000f0, 70000f0, -70000f0,
    ]
    extra = Float32[]
    for k in -16:16
        push!(extra, Float32(2.0^k))
        push!(extra, Float32(-2.0^k))
        push!(extra, Float32(1.5 * 2.0^k))
    end
    vcat(base, extra)
end

_mf4(x)              = reinterpret(UInt8, Float4_E2M1FN(x)) & 0x0F
_mf6_e2m3(x)         = reinterpret(UInt8, Float6_E2M3FN(x)) & 0x3F
_mf6_e3m2(x)         = reinterpret(UInt8, Float6_E3M2FN(x)) & 0x3F
_mf8_e4m3_sat(x)     = reinterpret(UInt8, E4M3FN_SAT_X4(x))
_mf8_e5m2_sat(x)     = reinterpret(UInt8, E5M2_SAT_X4(x))
_mf8_ue8m0_rz(x)     = reinterpret(UInt8, E8M0_SAT(x, RoundToZero))

# --- e2m1x2: hand-wrapped via wrappers/cvt.jl (NVPTX has no i8 constraint),
#     output is UInt16 with the packed nibbles in the low byte. ---
function _cvt_e2m1x2!(out::AbstractArray{UInt16}, xs::AbstractArray{Float32})
    tid  = ptx"mov.u32"(sreg"tid.x")
    bid  = ptx"mov.u32"(sreg"ctaid.x")
    ntid = ptx"mov.u32"(sreg"ntid.x")
    i = Int(bid) * Int(ntid) + Int(tid) + 1
    if i <= length(xs)
        x = @inbounds xs[i]
        @inbounds out[i] = ptx"cvt.rn.satfinite.e2m1x2.f32"(x, x)
    end
    return nothing
end

@testset "cvt.rn.satfinite.e2m1x2 ≡ Float4_E2M1FN" begin
    xs = fp_test_values()
    n  = length(xs)
    out_d = CUDACore.zeros(UInt16, n)
    threads = min(n, 256); blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks _cvt_e2m1x2!(out_d, CuArray(xs))
    CUDACore.synchronize()
    # Wrapper packs the .b8 cvt result into the low byte of a UInt16; high
    # byte is zero by the `mov.b16 dst, {t, 0}` form.
    expected = UInt16[(_mf4(x) << 4) | _mf4(x) for x in xs]
    @test Array(out_d) == expected
end

# N.B. `cvt.rn.satfinite.e2m1x4.f32` (and the e4m3x4 / e5m2x4 four-lane FP8
# packs from f32) do not appear to exist as PTX ops. NVIDIA's `cuda_fp4.h`
# defines `__nv_fp4x4_storage_t` only as a type alias for `__nv_fp8x2_storage_t`
# and exposes no `_to_fp4x4` conversion; same story across `cuda_fp8.h` /
# CCCL / pyptx — every direct float-to-FP4/MX cvt builtin is `_to_fp4` (×1)
# or `_to_fp4x2` (×2). ptxas's "Arguments mismatch for instruction 'cvt'"
# on the four-input form is consistent with "no such overload of cvt." The
# idiomatic four-lane pack is two `e2m1x2.f32` + `mov.b32 d, {a, b}`. Tests
# elided pending a PTX ISA reference that confirms a true x4 form (PTX 9.0+
# may have introduced one — re-add if so).

# --- e2m3x2 / e3m2x2: chain default. .b16 dest, byte-aligned per lane. ---
function _cvt_e2m3x2!(out::AbstractArray{UInt16}, xs::AbstractArray{Float32})
    tid  = ptx"mov.u32"(sreg"tid.x")
    bid  = ptx"mov.u32"(sreg"ctaid.x")
    ntid = ptx"mov.u32"(sreg"ntid.x")
    i = Int(bid) * Int(ntid) + Int(tid) + 1
    if i <= length(xs)
        x = @inbounds xs[i]
        @inbounds out[i] = ptx"cvt.rn.satfinite.e2m3x2.f32"(x, x)
    end
    return nothing
end

@testset "cvt.rn.satfinite.e2m3x2 ≡ Float6_E2M3FN" begin
    xs = fp_test_values()
    n  = length(xs)
    out_d = CUDACore.zeros(UInt16, n)
    threads = min(n, 256); blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks _cvt_e2m3x2!(out_d, CuArray(xs))
    CUDACore.synchronize()
    expected = UInt16[let b = UInt16(_mf6_e2m3(x))
                          (b << 8) | b
                      end
                      for x in xs]
    @test Array(out_d) == expected
end

function _cvt_e3m2x2!(out::AbstractArray{UInt16}, xs::AbstractArray{Float32})
    tid  = ptx"mov.u32"(sreg"tid.x")
    bid  = ptx"mov.u32"(sreg"ctaid.x")
    ntid = ptx"mov.u32"(sreg"ntid.x")
    i = Int(bid) * Int(ntid) + Int(tid) + 1
    if i <= length(xs)
        x = @inbounds xs[i]
        @inbounds out[i] = ptx"cvt.rn.satfinite.e3m2x2.f32"(x, x)
    end
    return nothing
end

@testset "cvt.rn.satfinite.e3m2x2 ≡ Float6_E3M2FN" begin
    xs = fp_test_values()
    n  = length(xs)
    out_d = CUDACore.zeros(UInt16, n)
    threads = min(n, 256); blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks _cvt_e3m2x2!(out_d, CuArray(xs))
    CUDACore.synchronize()
    expected = UInt16[let b = UInt16(_mf6_e3m2(x))
                          (b << 8) | b
                      end
                      for x in xs]
    @test Array(out_d) == expected
end

# N.B. e4m3x4 / e5m2x4 four-lane FP8 packs from f32 — same story as e2m1x4.
# Tests elided pending PTX ISA confirmation.

# --- ue8m0x2: only .rz / .rp rounding are legal; .rn is rejected. ---
function _cvt_ue8m0x2!(out::AbstractArray{UInt16}, xs::AbstractArray{Float32})
    tid  = ptx"mov.u32"(sreg"tid.x")
    bid  = ptx"mov.u32"(sreg"ctaid.x")
    ntid = ptx"mov.u32"(sreg"ntid.x")
    i = Int(bid) * Int(ntid) + Int(tid) + 1
    if i <= length(xs)
        x = @inbounds xs[i]
        @inbounds out[i] = ptx"cvt.rz.satfinite.ue8m0x2.f32"(x, x)
    end
    return nothing
end

@testset "cvt.rz.satfinite.ue8m0x2 ≡ Float8_E8M0FNU(SAT, RoundToZero)" begin
    # `x >= 0` lets -0.0 through (-0.0 == 0.0); E8M0_SAT rejects negatives
    # because BFloat16 in the conversion path preserves the sign bit.
    xs = filter(x -> !signbit(x) && isfinite(x), fp_test_values())
    n  = length(xs)
    out_d = CUDACore.zeros(UInt16, n)
    threads = min(n, 256); blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks _cvt_ue8m0x2!(out_d, CuArray(xs))
    CUDACore.synchronize()
    expected = UInt16[let b = UInt16(_mf8_ue8m0_rz(x))
                          (b << 8) | b
                      end
                      for x in xs]
    @test Array(out_d) == expected
end

# --- scaled::n2::ue8m0  e4m3x2 → bf16x2 with packed scale; scale=0x7F
#     (= biased exp 127 → 2^0 = 1) must match the unscaled cvt path. ---
function _cvt_scaled!(out::AbstractArray{UInt32}, xs::AbstractArray{UInt16}, scale::UInt16)
    tid  = ptx"mov.u32"(sreg"tid.x")
    bid  = ptx"mov.u32"(sreg"ctaid.x")
    ntid = ptx"mov.u32"(sreg"ntid.x")
    i = Int(bid) * Int(ntid) + Int(tid) + 1
    if i <= length(xs)
        x = @inbounds xs[i]
        @inbounds out[i] = ptx"cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2"(x, scale)
    end
    return nothing
end

function _cvt_unscaled!(out::AbstractArray{UInt32}, xs::AbstractArray{UInt16})
    tid  = ptx"mov.u32"(sreg"tid.x")
    bid  = ptx"mov.u32"(sreg"ctaid.x")
    ntid = ptx"mov.u32"(sreg"ntid.x")
    i = Int(bid) * Int(ntid) + Int(tid) + 1
    if i <= length(xs)
        x = @inbounds xs[i]
        @inbounds out[i] = ptx"cvt.rn.bf16x2.e4m3x2"(x)
    end
    return nothing
end

@testset "cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 (scale=1) ≡ unscaled" begin
    xs = UInt16[(UInt16(b) << 8) | UInt16(b) for b in UInt8.(0:255)]
    n  = length(xs)
    scaled_d   = CUDACore.zeros(UInt32, n)
    unscaled_d = CUDACore.zeros(UInt32, n)
    threads = min(n, 256); blocks = cld(n, threads)
    # `n2::ue8m0` packs *per-lane* scales into the UInt16 — high byte scales
    # the high bf16 lane, low byte scales the low one — so set both bytes.
    @cuda threads=threads blocks=blocks _cvt_scaled!(scaled_d, CuArray(xs), UInt16(0x7F7F))
    @cuda threads=threads blocks=blocks _cvt_unscaled!(unscaled_d, CuArray(xs))
    CUDACore.synchronize()
    @test Array(scaled_d) == Array(unscaled_d)
end

# --- sub-byte FP unpack (FP6/FP4 → FP16x2) round-trip --------------------
#
# Pack f32 → e*x2 with the existing wrappers, then unpack via the new
# `cvt.rn.f16x2.e*x2` paths. Both halves of the f16x2 result must match
# Float16(Microfloat(x)) since pack(x, x) puts the same value in both
# slots. e2m3x2 / e3m2x2 use the chain default (.b16 source); e2m1x2 uses
# the hand-written wrapper in src/wrappers/cvt.jl (.b8 source carrier).

@inline _f16x2_lo(p::UInt32) = reinterpret(Float16, UInt16(p        & 0xFFFF))
@inline _f16x2_hi(p::UInt32) = reinterpret(Float16, UInt16((p >> 16) & 0xFFFF))

function _cvt_e2m3_unpack_roundtrip!(out::AbstractArray{UInt32}, xs::AbstractArray{Float32})
    tid  = ptx"mov.u32"(sreg"tid.x")
    bid  = ptx"mov.u32"(sreg"ctaid.x")
    ntid = ptx"mov.u32"(sreg"ntid.x")
    i = Int(bid) * Int(ntid) + Int(tid) + 1
    if i <= length(xs)
        x = @inbounds xs[i]
        packed = ptx"cvt.rn.satfinite.e2m3x2.f32"(x, x)
        @inbounds out[i] = ptx"cvt.rn.f16x2.e2m3x2"(packed)
    end
    return nothing
end

@testset "cvt.rn.f16x2.e2m3x2 round-trip ≡ Float16(Float6_E2M3FN)" begin
    xs = fp_test_values()
    n = length(xs)
    out_d = CUDACore.zeros(UInt32, n)
    threads = min(n, 256); blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks _cvt_e2m3_unpack_roundtrip!(out_d, CuArray(xs))
    CUDACore.synchronize()
    got = Array(out_d)
    expected = Float16[Float16(Float6_E2M3FN(x)) for x in xs]
    @test all(_f16x2_lo.(got) .=== expected)
    @test all(_f16x2_hi.(got) .=== expected)
end

function _cvt_e3m2_unpack_roundtrip!(out::AbstractArray{UInt32}, xs::AbstractArray{Float32})
    tid  = ptx"mov.u32"(sreg"tid.x")
    bid  = ptx"mov.u32"(sreg"ctaid.x")
    ntid = ptx"mov.u32"(sreg"ntid.x")
    i = Int(bid) * Int(ntid) + Int(tid) + 1
    if i <= length(xs)
        x = @inbounds xs[i]
        packed = ptx"cvt.rn.satfinite.e3m2x2.f32"(x, x)
        @inbounds out[i] = ptx"cvt.rn.f16x2.e3m2x2"(packed)
    end
    return nothing
end

@testset "cvt.rn.f16x2.e3m2x2 round-trip ≡ Float16(Float6_E3M2FN)" begin
    xs = fp_test_values()
    n = length(xs)
    out_d = CUDACore.zeros(UInt32, n)
    threads = min(n, 256); blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks _cvt_e3m2_unpack_roundtrip!(out_d, CuArray(xs))
    CUDACore.synchronize()
    got = Array(out_d)
    expected = Float16[Float16(Float6_E3M2FN(x)) for x in xs]
    @test all(_f16x2_lo.(got) .=== expected)
    @test all(_f16x2_hi.(got) .=== expected)
end

function _cvt_e2m1_unpack_roundtrip!(out::AbstractArray{UInt32}, xs::AbstractArray{Float32})
    tid  = ptx"mov.u32"(sreg"tid.x")
    bid  = ptx"mov.u32"(sreg"ctaid.x")
    ntid = ptx"mov.u32"(sreg"ntid.x")
    i = Int(bid) * Int(ntid) + Int(tid) + 1
    if i <= length(xs)
        x = @inbounds xs[i]
        packed = ptx"cvt.rn.satfinite.e2m1x2.f32"(x, x)
        @inbounds out[i] = ptx"cvt.rn.f16x2.e2m1x2"(packed)
    end
    return nothing
end

@testset "cvt.rn.f16x2.e2m1x2 round-trip ≡ Float16(Float4_E2M1FN)" begin
    xs = fp_test_values()
    n = length(xs)
    out_d = CUDACore.zeros(UInt32, n)
    threads = min(n, 256); blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks _cvt_e2m1_unpack_roundtrip!(out_d, CuArray(xs))
    CUDACore.synchronize()
    got = Array(out_d)
    expected = Float16[Float16(Float4_E2M1FN(x)) for x in xs]
    @test all(_f16x2_lo.(got) .=== expected)
    @test all(_f16x2_hi.(got) .=== expected)
end
