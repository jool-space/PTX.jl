# TEST_TARGET: requires=toolkit evidence=ptxas
# Parametric wgmma m64n*k* sweep — every wgmma shape pyptx ports against.
# Each entry: (dtype_d, dtype_a, dtype_b, N, K, has_trans). The wgmma chain
# generates one method per combo; this validates each lowers cleanly under
# sm_90a ptxas, with the right N → d-vec arity (N/2 for f32, N/4 for f16,
# N/2 for s32 accumulator).
#
# Source of truth: PTX 9.3 §9.7.16.5 + PTX.jl's separate floating/integer
# variant and N inventories in src/wrappers/wgmma.jl. Each variant goes
# through a host-side `wgmma_descriptor`
# build with the canonical layout picked by `wgmma_layout`, so the two
# helper layers compose end-to-end here.

using PTX: wgmma_descriptor, layout_for_a, layout_for_mn_major, WgmmaSwizzle

# (acc, ab, ab, N, K, has_trans, d_jltype, d_per_lane)
const WGMMA_CASES = [
    # bf16 inputs: N ∈ {8, 16, 64, 128, 256}, K = 16, f32 acc → N/2 d-regs.
    (:f32, :bf16, :bf16, 8,   16, true,  Float32, 4),
    (:f32, :bf16, :bf16, 16,  16, true,  Float32, 8),
    (:f32, :bf16, :bf16, 64,  16, true,  Float32, 32),
    (:f32, :bf16, :bf16, 128, 16, true,  Float32, 64),
    (:f32, :bf16, :bf16, 256, 16, true,  Float32, 128),

    # f16 inputs, f32 acc.
    (:f32, :f16,  :f16,  8,   16, true,  Float32, 4),
    (:f32, :f16,  :f16,  64,  16, true,  Float32, 32),
    (:f32, :f16,  :f16,  128, 16, true,  Float32, 64),

    # f16 inputs, f16 acc → N/4 d-regs (packed UInt32).
    (:f16, :f16,  :f16,  16,  16, true,  UInt32,  4),
    (:f16, :f16,  :f16,  64,  16, true,  UInt32,  16),
    (:f16, :f16,  :f16,  128, 16, true,  UInt32,  32),

    # tf32 inputs (k8), f32 acc.
    (:f32, :tf32, :tf32, 8,   8,  false, Float32, 4),
    (:f32, :tf32, :tf32, 16,  8,  false, Float32, 8),

    # FP8 inputs (k32), f32 acc, no trans. Same-dtype only — mixed e4m3/e5m2
    # is valid PTX 9.3 but not yet registered in PTX.jl's floating inventory.
    (:f32, :e4m3, :e4m3, 16,  32, false, Float32, 8),
    (:f32, :e4m3, :e4m3, 64,  32, false, Float32, 32),
    (:f32, :e5m2, :e5m2, 128, 32, false, Float32, 64),

    # FP8 inputs, f16 acc → N/4 d-regs.
    (:f16, :e4m3, :e4m3, 64,  32, false, UInt32,  16),

    # Integer wgmma (s8/u8 × s8/u8, k32, s32 acc → N/2 d-regs).
    (:s32, :s8,   :s8,   8,   32, false, Int32,   4),
    (:s32, :u8,   :u8,   64,  32, false, Int32,   32),
    (:s32, :s8,   :u8,   128, 32, false, Int32,   64),
    (:s32, :u8,   :s8,   224, 32, false, Int32,   112),
]

# wgmma reads N input scalars (input frags) per lane = N/4 UInt32 packed.
# But the chain wrapper takes only the d (acc) tuple + descriptors — input
# scaling is via descriptors, so the call signature is uniform.

# Pre-generate one kernel per combo at file scope so world-age advances
# before the @testset runs. Each kernel is a tiny wrapper that fires the
# Operation{:wgmma, mods}() dispatch with concrete type params at parse time.
_wgmma_kern_name(d_ty, a_ty, b_ty, n, k) =
    Symbol("_wgmma_kern_", d_ty, "_", a_ty, "_", b_ty, "_n", n, "k", k)

for case in WGMMA_CASES
    d_ty, a_ty, b_ty, n, k, _has_trans, d_jltype, d_per_lane = case
    mods = (:mma_async, :sync, :aligned,
            Symbol("m64n", n, "k", k),
            d_ty, a_ty, b_ty)
    op = PTX.Operation{:wgmma, mods}()
    fname = _wgmma_kern_name(d_ty, a_ty, b_ty, n, k)

    @eval function $(fname)(out::CuDeviceVector{$d_jltype, 1},
                            a_desc::UInt64, b_desc::UInt64)
        d = ntuple(_ -> zero($d_jltype), Val($d_per_lane))
        d = $op(d, a_desc, b_desc, false)
        @inbounds out[1] = d[1]
        return nothing
    end
end

@testset "wgmma sweep: $(case[1]).$(case[2]).$(case[3]) m64n$(case[4])k$(case[5])" for
    case in WGMMA_CASES

    d_ty, a_ty, b_ty, n, k, _has_trans, d_jltype, _d_per_lane = case
    f = getfield(@__MODULE__, _wgmma_kern_name(d_ty, a_ty, b_ty, n, k))
    types = Tuple{CuDeviceVector{d_jltype, 1}, UInt64, UInt64}

    @test ptxas_compiles(f, types; cap = v"9.0", feature_set = :arch)
    ptx = emit_ptx(f, types; cap = v"9.0", feature_set = :arch)
    @test occursin("wgmma.mma_async.sync.aligned.m64n$(n)k$(k).$(d_ty).$(a_ty).$(b_ty)", ptx)
end

@testset "wgmma_layout + wgmma_descriptor integration" begin
    # The two host helpers should compose into a valid 64-bit descriptor for
    # the canonical bf16 m64k16 tile. Verifies values shipped to the device.
    la = layout_for_a(dtype = :bf16, m = 64, k = 16)
    @test la.layout_type == WgmmaSwizzle.B32
    @test la.leading_byte_offset == 16
    @test la.stride_byte_offset == 256

    # Plug into the descriptor builder. SMEM addr 0 → just the metadata bits.
    desc = wgmma_descriptor(UInt32(0);
        leading_byte_offset = la.leading_byte_offset,
        stride_byte_offset  = la.stride_byte_offset,
        swizzle             = la.layout_type)

    # swizzle field (bits 63:62) == 3 (B32).
    @test (desc >> 62) & 0x3 == 3
    # stride field (bits 45:32) = 256 >> 4 = 16.
    @test (desc >> 32) & 0x3FFF == 16
    # leading field (bits 29:16) = 16 >> 4 = 1.
    @test (desc >> 16) & 0x3FFF == 1

    # B-side (degenerate N=8, MN-major): INTERLEAVE, LBO = 128, SBO trivial.
    lb = layout_for_mn_major(dtype = :bf16, k = 16, n = 8)
    @test lb.layout_type == WgmmaSwizzle.NONE
    @test lb.leading_byte_offset == 128

    desc_b = wgmma_descriptor(UInt32(0);
        leading_byte_offset = lb.leading_byte_offset,
        stride_byte_offset  = lb.stride_byte_offset,
        swizzle             = lb.layout_type)
    @test (desc_b >> 62) & 0x3 == 0           # NONE
    @test (desc_b >> 16) & 0x3FFF == 8        # 128 >> 4 = 8
end

@testset "wgmma_layout — every dtype × shape combination that ships" begin
    # Spot-check the layout picker hits every entry in the layout family
    # table (NONE, B32, B64, B128) for at least one (dtype, K) pair, plus
    # the multi-stripe path (K > 128 B / elem_bytes).
    @test layout_for_a(dtype = :bf16, m = 64, k = 8).layout_type   == WgmmaSwizzle.NONE
    @test layout_for_a(dtype = :bf16, m = 64, k = 16).layout_type  == WgmmaSwizzle.B32
    @test layout_for_a(dtype = :bf16, m = 64, k = 32).layout_type  == WgmmaSwizzle.B64
    @test layout_for_a(dtype = :bf16, m = 64, k = 64).layout_type  == WgmmaSwizzle.B128
    @test layout_for_a(dtype = :bf16, m = 64, k = 128).layout_type == WgmmaSwizzle.B128

    # FP8 (1 B/elem) covers the same row-byte widths at 8× the K.
    @test layout_for_a(dtype = :e4m3, m = 64, k = 32).layout_type  == WgmmaSwizzle.B32
    @test layout_for_a(dtype = :e5m2, m = 64, k = 128).layout_type == WgmmaSwizzle.B128

    # tf32 (4 B/elem) — K=4 hits INTERLEAVE, K=8 → B32.
    @test layout_for_a(dtype = :tf32, m = 64, k = 4).layout_type == WgmmaSwizzle.NONE
    @test layout_for_a(dtype = :tf32, m = 64, k = 8).layout_type == WgmmaSwizzle.B32
end
