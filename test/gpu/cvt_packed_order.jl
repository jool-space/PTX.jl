# REQUIRES CC 8.9
#
# Probe the operand ordering of PTX packed-cvt instructions
# (`cvt.rn.{bf16x2,f16x2,satfinite.e4m3x2,satfinite.e5m2x2}.f32`).
#
# Background: prior empirical evidence (see commit 4129d49 and
# test/gpu/hopper/gemm_pc_tma_store.jl L198-216) suggests that
# `cvt.rn.bf16x2.f32 d, a, b` places the SECOND source in the LOW
# 16 bits of `d`, despite PTX ISA 9.2 §9.7.6.4 saying the first source
# maps to the lower half. This file probes whether that swap is
# bf16-specific or whether the rest of the packed-cvt family
# (f16x2, e4m3x2, e5m2x2) exhibits the same behavior.
#
# Method: for each packed cvt, run a kernel with distinct sentinel
# Float32 inputs `a` and `b`, write the resulting packed UInt32 (or
# UInt16 for fp8) to global memory, then dequantize the low and high
# halves on the host. Compare against the two input sentinels to
# determine which input ended up in which half.

# Two distinct, easy-to-recognize sentinels with separable bit patterns.
const A_SENTINEL = 1.5f0    # bf16 bits = 0x3FC0, f16 bits = 0x3E00
const B_SENTINEL = 3.25f0   # bf16 bits = 0x4050, f16 bits = 0x4280

# ── bf16x2 kernel ─────────────────────────────────────────────────────────

function _cvt_bf16x2_kernel!(out::AbstractArray{UInt32})
    out[1] = ptx"cvt.rn.bf16x2.f32"(A_SENTINEL, B_SENTINEL)
    return nothing
end

# ── f16x2 kernel ──────────────────────────────────────────────────────────

function _cvt_f16x2_kernel!(out::AbstractArray{UInt32})
    out[1] = ptx"cvt.rn.f16x2.f32"(A_SENTINEL, B_SENTINEL)
    return nothing
end

# ── e4m3x2 kernel (packed UInt16, written into low 16 of a UInt32) ────────

function _cvt_e4m3x2_kernel!(out::AbstractArray{UInt32})
    p = ptx"cvt.rn.satfinite.e4m3x2.f32"(A_SENTINEL, B_SENTINEL)
    out[1] = UInt32(p)
    return nothing
end

# ── e5m2x2 kernel ─────────────────────────────────────────────────────────

function _cvt_e5m2x2_kernel!(out::AbstractArray{UInt32})
    p = ptx"cvt.rn.satfinite.e5m2x2.f32"(A_SENTINEL, B_SENTINEL)
    out[1] = UInt32(p)
    return nothing
end

# ── Host-side dequant helpers ─────────────────────────────────────────────

# bf16_bits / bf16_to_f32 are defined in test/setup.jl.

# Expected bit patterns for the sentinels under each encoding:
const A_BF16 = bf16_bits(A_SENTINEL)        # 0x3FC0
const B_BF16 = bf16_bits(B_SENTINEL)        # 0x4050
const A_F16  = reinterpret(UInt16, Float16(A_SENTINEL))   # 0x3E00
const B_F16  = reinterpret(UInt16, Float16(B_SENTINEL))   # 0x4280

# E4M3FN encoding of 1.5  = 0b0_0111_100 = 0x3C
# E4M3FN encoding of 3.25 = 0b0_1000_101 = 0x45
# (sign=0, exp_biased=7=0b0111, mantissa=100 for 1.5)
# (sign=0, exp_biased=8=0b1000, mantissa=101 for 3.25)
const A_E4M3 = UInt8(0x3C)
const B_E4M3 = UInt8(0x45)
# E5M2 encoding of 1.5  = 0b0_01111_10 = 0x3E
# E5M2 encoding of 3.25 = 0b0_10000_10 = 0x42  (1.5 not 1.625 since only 2 mant bits)
# 3.25 in binary = 1.101 × 2^1, mantissa rounds to "10" → 1.10 × 2^1 = 3.0
# Actually with RNE: 1.101 → 1.10 (round down at tie? No, the tie-break to
# even rounds 1.101 to 1.10) — we'll just print what the GPU emits and
# match against the bit pattern below, not by dequant.
const A_E5M2 = UInt8(0x3E)
const B_E5M2 = UInt8(0x42)

# Tag-naming helper for the report column.
function which_input(low::Any, high::Any, a_pat::Any, b_pat::Any)
    if low == a_pat && high == b_pat
        return ("first", "second")     # matches PTX docs
    elseif low == b_pat && high == a_pat
        return ("second", "first")     # swapped vs docs
    else
        return ("unknown(low=$(low))", "unknown(high=$(high))")
    end
end

# ── Run only if a Blackwell-or-Hopper-class device is available ───────────

if DEV_CAP >= v"8.9"
    @testset "Packed cvt operand ordering probe (sm_$(DEV_CAP.major)$(DEV_CAP.minor))" begin

        # bf16x2
        out_d = CUDACore.zeros(UInt32, 1)
        @cuda threads=1 blocks=1 _cvt_bf16x2_kernel!(out_d)
        CUDACore.synchronize()
        bf16_packed = Array(out_d)[1]
        bf16_lo = UInt16(bf16_packed        & 0xFFFF)
        bf16_hi = UInt16((bf16_packed >> 16) & 0xFFFF)
        bf16_lo_f = bf16_to_f32(bf16_lo)
        bf16_hi_f = bf16_to_f32(bf16_hi)
        bf16_who = which_input(bf16_lo, bf16_hi, A_BF16, B_BF16)
        @info "cvt.rn.bf16x2.f32" packed=string(bf16_packed, base=16, pad=8) lo_bits=string(bf16_lo, base=16, pad=4) hi_bits=string(bf16_hi, base=16, pad=4) lo_f32=bf16_lo_f hi_f32=bf16_hi_f low_is=bf16_who[1] high_is=bf16_who[2]
        @test bf16_who[1] in ("first", "second")
        @test bf16_who[2] in ("first", "second")
        @test bf16_who[1] != bf16_who[2]

        # f16x2
        out_d = CUDACore.zeros(UInt32, 1)
        @cuda threads=1 blocks=1 _cvt_f16x2_kernel!(out_d)
        CUDACore.synchronize()
        f16_packed = Array(out_d)[1]
        f16_lo = UInt16(f16_packed        & 0xFFFF)
        f16_hi = UInt16((f16_packed >> 16) & 0xFFFF)
        f16_lo_f = Float32(reinterpret(Float16, f16_lo))
        f16_hi_f = Float32(reinterpret(Float16, f16_hi))
        f16_who = which_input(f16_lo, f16_hi, A_F16, B_F16)
        @info "cvt.rn.f16x2.f32" packed=string(f16_packed, base=16, pad=8) lo_bits=string(f16_lo, base=16, pad=4) hi_bits=string(f16_hi, base=16, pad=4) lo_f32=f16_lo_f hi_f32=f16_hi_f low_is=f16_who[1] high_is=f16_who[2]
        @test f16_who[1] in ("first", "second")
        @test f16_who[2] in ("first", "second")
        @test f16_who[1] != f16_who[2]

        # e4m3x2 (only the low 16 bits of out[1] hold the packed pair: lo byte/hi byte = two e4m3 lanes)
        out_d = CUDACore.zeros(UInt32, 1)
        @cuda threads=1 blocks=1 _cvt_e4m3x2_kernel!(out_d)
        CUDACore.synchronize()
        e4m3_packed = UInt16(Array(out_d)[1] & 0xFFFF)
        e4m3_lo = UInt8(e4m3_packed        & 0xFF)
        e4m3_hi = UInt8((e4m3_packed >> 8) & 0xFF)
        e4m3_who = which_input(e4m3_lo, e4m3_hi, A_E4M3, B_E4M3)
        @info "cvt.rn.satfinite.e4m3x2.f32" packed=string(e4m3_packed, base=16, pad=4) lo_byte=string(e4m3_lo, base=16, pad=2) hi_byte=string(e4m3_hi, base=16, pad=2) low_is=e4m3_who[1] high_is=e4m3_who[2]
        @test e4m3_who[1] in ("first", "second")
        @test e4m3_who[2] in ("first", "second")
        @test e4m3_who[1] != e4m3_who[2]

        # e5m2x2
        out_d = CUDACore.zeros(UInt32, 1)
        @cuda threads=1 blocks=1 _cvt_e5m2x2_kernel!(out_d)
        CUDACore.synchronize()
        e5m2_packed = UInt16(Array(out_d)[1] & 0xFFFF)
        e5m2_lo = UInt8(e5m2_packed        & 0xFF)
        e5m2_hi = UInt8((e5m2_packed >> 8) & 0xFF)
        e5m2_who = which_input(e5m2_lo, e5m2_hi, A_E5M2, B_E5M2)
        @info "cvt.rn.satfinite.e5m2x2.f32" packed=string(e5m2_packed, base=16, pad=4) lo_byte=string(e5m2_lo, base=16, pad=2) hi_byte=string(e5m2_hi, base=16, pad=2) low_is=e5m2_who[1] high_is=e5m2_who[2]
        @test e5m2_who[1] in ("first", "second")
        @test e5m2_who[2] in ("first", "second")
        @test e5m2_who[1] != e5m2_who[2]
    end
end

# Operand order findings (sm_121a, CC 12.1; CUDA 13.0 ptxas; GB10):
#
#   cvt.rn.bf16x2.f32                → low = second, high = first
#   cvt.rn.f16x2.f32                 → low = second, high = first
#   cvt.rn.satfinite.e4m3x2.f32      → low = second, high = first
#   cvt.rn.satfinite.e5m2x2.f32      → low = second, high = first
#
# All four packed cvt forms place the SECOND source operand in the LOW
# half of the result and the FIRST source operand in the HIGH half,
# i.e. the SAME swap previously seen for bf16x2 alone (commit 4129d49).
# This contradicts PTX ISA 9.2 §9.7.6.4, which states the first source
# maps to the lower-half lane. Verified by inspecting the emitted PTX
# directly: `cvt.rn.bf16x2.f32 %r3, %f1, %f2;` with %f1 = 1.5f0 and
# %f2 = 3.25f0 produces %r3 = 0x3FC04050 — bf16(3.25)=0x4050 in low,
# bf16(1.5)=0x3FC0 in high. So the swap is in the hardware/ptxas
# semantics, not in the Julia/LLVM lowering.
#
# Workaround when wrappers register one of these packed cvts: either
# swap the Julia-side argument order in the wrapper (so callers can
# pass `(low_input, high_input)` and get back the natural ordering),
# or — for the bf16/f16 cases — route through scalar
# `cvt.rn.{bf16,f16}.f32` + `mov.b32 dst, {lo, hi}` where the brace
# pair has unambiguous PTX ordering. The fp8 packed cvt currently has
# no scalar alternative on this arch, so a Julia-side argument swap is
# the only option for e4m3x2/e5m2x2.
