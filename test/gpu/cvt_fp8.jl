# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.9
using Microfloats
using Microfloats: NanOnlyAllOnes, SAT, OVF

# Compare PTX cvt.rn.satfinite.{e4m3,e5m2}x2.f32 against the host-side
# Microfloats conversion for the matching policy.
#
# PTX's `.e4m3` is the OCP E4M3FN format (NanOnlyAllOnes — no Inf, NaN at
# 0x7F/0xFF). PTX's `.e5m2` is IEEE-style E5M2 (Inf and NaN). The `.satfinite`
# modifier clamps overflow to ±floatmax (matching Microfloats' SAT policy);
# without it, overflow goes to NaN (E4M3FN) or ±Inf (E5M2), matching OVF.
#
# Microfloats ships Float8_E4M3FN (OVF) and Float8_E5M2 (OVF) by default.
# We declare SAT twins here for the satfinite path. (Once Microfloats grows
# device-side conversion overrides, the kernel could call its routines
# directly — for now we just verify bit-for-bit agreement.)

@microfloat E4M3FN_SAT exponent=4 significand=3 nonfinite=NanOnlyAllOnes overflow=SAT
@microfloat E5M2_SAT   exponent=5 significand=2 overflow=SAT

# PTX-side conversion kernel: each thread converts xs[i] via the packed
# x2 cvt with both lanes set to xs[i], then writes the low byte (= e4m3/e5m2
# of xs[i]). The packed form is the only one available on sm_89; sm_120+
# may add a true scalar form.
function _cvt_e4m3_satfinite!(out::AbstractArray{UInt8}, xs::AbstractArray{Float32})
    tid  = ptx"mov.u32"(sreg"tid.x")
    bid  = ptx"mov.u32"(sreg"ctaid.x")
    ntid = ptx"mov.u32"(sreg"ntid.x")
    i = Int(bid) * Int(ntid) + Int(tid) + 1
    if i <= length(xs)
        x = @inbounds xs[i]
        @inbounds out[i] = UInt8(ptx"cvt.rn.satfinite.e4m3x2.f32"(x, x) & 0xFF)
    end
    return nothing
end

function _cvt_e5m2_satfinite!(out::AbstractArray{UInt8}, xs::AbstractArray{Float32})
    tid  = ptx"mov.u32"(sreg"tid.x")
    bid  = ptx"mov.u32"(sreg"ctaid.x")
    ntid = ptx"mov.u32"(sreg"ntid.x")
    i = Int(bid) * Int(ntid) + Int(tid) + 1
    if i <= length(xs)
        x = @inbounds xs[i]
        @inbounds out[i] = UInt8(ptx"cvt.rn.satfinite.e5m2x2.f32"(x, x) & 0xFF)
    end
    return nothing
end

# Run a Float32 → FP8 conversion through PTX and Microfloats, return both
# byte vectors for comparison.
function ptx_vs_microfloat(::Type{T}, xs::Vector{Float32}) where {T<:Microfloat}
    n = length(xs)
    xs_d = CuArray(xs)
    out_d = CUDACore.zeros(UInt8, n)
    threads = min(n, 256)
    blocks  = cld(n, threads)
    if T === E4M3FN_SAT
        @cuda threads=threads blocks=blocks _cvt_e4m3_satfinite!(out_d, xs_d)
    elseif T === E5M2_SAT
        @cuda threads=threads blocks=blocks _cvt_e5m2_satfinite!(out_d, xs_d)
    else
        error("unsupported $T")
    end
    CUDACore.synchronize()
    ptx_bytes = Array(out_d)
    mf_bytes  = UInt8[reinterpret(UInt8, T(x)) for x in xs]
    return ptx_bytes, mf_bytes
end

# A small but representative sweep: zero, exact small values, sign,
# subnormal range, normal range, rounding-tie cases, overflow.
function fp8_test_values()
    base = Float32[
        0f0, -0f0,
        1f0, -1f0, 2f0, -2f0, 0.5f0, -0.5f0,
        # Round-to-even ties for E4M3 (3 mantissa bits): mid-rungs in [1,2]
        1.0625f0, 1.1875f0, 1.3125f0, 1.4375f0,
        # Round-to-even ties for E5M2 (2 mantissa bits): mid-rungs in [1,2]
        1.125f0, 1.375f0, 1.625f0, 1.875f0,
        # Subnormal-ish for E4M3 (smin_normal = 2^-6 ≈ 0.0156)
        0.001f0, 0.005f0, 0.01f0, 0.02f0,
        # Subnormal-ish for E5M2 (smin_normal = 2^-14 ≈ 6.1e-5)
        1f-5, 5f-5, 1f-4,
        # In-range normals
        7.5f0, 17.0f0, 100.0f0, 300.0f0,
        # E4M3 overflow (floatmax = 448), E5M2 still in-range
        500.0f0, -500.0f0, 1000.0f0, 10000.0f0,
        # E5M2 overflow (floatmax = 57344)
        70000.0f0, -70000.0f0,
    ]
    # Plus a deterministic sweep across many magnitudes.
    extra = Float32[]
    for k in -16:16
        push!(extra, Float32(2.0^k))
        push!(extra, Float32(-2.0^k))
        push!(extra, Float32(1.5 * 2.0^k))
        push!(extra, Float32(1.25 * 2.0^k))
    end
    vcat(base, extra)
end

@testset "PTX cvt.rn.satfinite.e4m3 ≡ Microfloats Float8_E4M3FN(SAT)" begin
    xs = fp8_test_values()
    ptx_bytes, mf_bytes = ptx_vs_microfloat(E4M3FN_SAT, xs)
    @test ptx_bytes == mf_bytes
end

@testset "PTX cvt.rn.satfinite.e5m2 ≡ Microfloats Float8_E5M2(SAT)" begin
    xs = fp8_test_values()
    ptx_bytes, mf_bytes = ptx_vs_microfloat(E5M2_SAT, xs)
    @test ptx_bytes == mf_bytes
end


# --- FP8 → FP16 unpack (chain default, sm_89) -----------------------------
#
# Inverse of the pack tested above — the form Triton emits hundreds of times
# per FP8-mma kernel. Round-trip: f32 → e4m3x2 (existing pack wrapper) →
# UInt16 carrier (both bytes hold the value) → f16x2 (chain default unpack)
# → split into two Float16 lanes. Each Float16 lane should equal
# Float16(Float8_E4M3FN(x)) — Microfloats handles the f8→f16 promotion.

function _cvt_e4m3_roundtrip!(out::AbstractArray{UInt32}, xs::AbstractArray{Float32})
    tid  = ptx"mov.u32"(sreg"tid.x")
    bid  = ptx"mov.u32"(sreg"ctaid.x")
    ntid = ptx"mov.u32"(sreg"ntid.x")
    i = Int(bid) * Int(ntid) + Int(tid) + 1
    if i <= length(xs)
        x = @inbounds xs[i]
        packed = ptx"cvt.rn.satfinite.e4m3x2.f32"(x, x)
        @inbounds out[i] = ptx"cvt.rn.f16x2.e4m3x2"(packed)
    end
    return nothing
end

function _cvt_e5m2_roundtrip!(out::AbstractArray{UInt32}, xs::AbstractArray{Float32})
    tid  = ptx"mov.u32"(sreg"tid.x")
    bid  = ptx"mov.u32"(sreg"ctaid.x")
    ntid = ptx"mov.u32"(sreg"ntid.x")
    i = Int(bid) * Int(ntid) + Int(tid) + 1
    if i <= length(xs)
        x = @inbounds xs[i]
        packed = ptx"cvt.rn.satfinite.e5m2x2.f32"(x, x)
        @inbounds out[i] = ptx"cvt.rn.f16x2.e5m2x2"(packed)
    end
    return nothing
end

# Split a UInt32 holding f16x2 into two Float16 lanes.
@inline _f16x2_lo(p::UInt32) = reinterpret(Float16, UInt16(p        & 0xFFFF))
@inline _f16x2_hi(p::UInt32) = reinterpret(Float16, UInt16((p >> 16) & 0xFFFF))

@testset "PTX cvt.rn.f16x2.e4m3x2 round-trip ≡ Float16(Float8_E4M3FN(SAT))" begin
    xs = fp8_test_values()
    n = length(xs)
    out_d = CUDACore.zeros(UInt32, n)
    threads = min(n, 256); blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks _cvt_e4m3_roundtrip!(out_d, CuArray(xs))
    CUDACore.synchronize()
    got = Array(out_d)
    expected = Float16[Float16(E4M3FN_SAT(x)) for x in xs]
    @test all(_f16x2_lo.(got) .=== expected)
    @test all(_f16x2_hi.(got) .=== expected)
end

@testset "PTX cvt.rn.f16x2.e5m2x2 round-trip ≡ Float16(Float8_E5M2(SAT))" begin
    xs = fp8_test_values()
    n = length(xs)
    out_d = CUDACore.zeros(UInt32, n)
    threads = min(n, 256); blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks _cvt_e5m2_roundtrip!(out_d, CuArray(xs))
    CUDACore.synchronize()
    got = Array(out_d)
    expected = Float16[Float16(E5M2_SAT(x)) for x in xs]
    @test all(_f16x2_lo.(got) .=== expected)
    @test all(_f16x2_hi.(got) .=== expected)
end
