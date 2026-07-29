# tcgen05 idesc-bit probes for the non-f16 kinds (EVIDENCE queue item:
# "validate i8/f8f6f4/MX instruction-descriptor helpers — only f16bf16_f32
# exists"). Before growing real builders, this probes the Table 51 bit
# interpretations for .kind::i8 (s8×s8→s32) and .kind::f8f6f4 (e4m3×e4m3→f32)
# with HAND-ENCODED descriptors and exact uniform-ones references. A green
# runtime leg here is the evidence the future `tcgen05_instr_desc_i8` /
# `_f8f6f4` builders cite; MX (block-scaled) kinds need the scale-tmem
# operands and are out of scope for this probe.
#
# Structure mirrors test/gpu/blackwell/tcgen05_mma_probe.jl: 128×256 tile,
# uniform fill (swizzle-invariant, so staging skips the permutation while
# descriptors still declare B32), single-thread MMA + commit, mbarrier wait,
# 32-lane × 64-column TMEM readback window.
#
# Both kinds use 1-byte containers, so dense K per instruction is 32 (not
# f16's 16): A = 128×32 = 4096 B, B = 256×32 = 8192 B — the same byte
# counts as the f16 probe, with 32-byte rows → BlackwellLayout.B32 and
# stride_bytes = 8*K = 256.
#
# idesc encodings (PTX ISA 9.4 §9.7.18.4 Table 51; bits not mentioned = 0):
#   .kind::i8, s8×s8→s32, m=128 n=256, K-major, no saturate:
#     (2 << 4)  d-format = S32
#     (1 << 7)  atype    = s8       (u8 = 0)
#     (1 << 10) btype    = s8
#     (32 << 17) n >> 3
#     (8 << 24)  m >> 4
#     = 0x084004A0.  negate (13/14) and transpose (15/16) MUST be 0 for i8.
#   .kind::f8f6f4, e4m3×e4m3→f32, same shape:
#     (1 << 4)  d-format = F32
#     (0 << 7)  atype    = E4M3     (E5M2=1, E2M3=3, E3M2=4, E2M1=5)
#     (0 << 10) btype    = E4M3
#     (32 << 17) | (8 << 24)
#     = 0x08400010.  Bit 29 (K=64) stays 0 — that encoding is sm_107f+.
#
# Run:   julia --project=test experiments/b200-queue/idesc_probe.jl
# Legs:  emit + ptxas run anywhere; runtime needs CC 10.x/11.x.

using Test, Random
using PTX, CUDACore, CUDATools
using PTX: smem_addr_u32, tcgen05_descriptor, BlackwellLayout, tmem_lane_addr
using PTX.MBarriers: barrier_try_wait
# TARGET GATE (learned on the 2026-07-29 B300 session): `.kind::i8` is
# a-variant-exclusive — §9.7.18.10 lists sm_100a/sm_101a/sm_110a and the
# family-extension note excludes i8 by name; ptxas 13.3 rejects it on
# sm_100f/sm_103a/sm_103f. A CC 10.3 B300 cannot run tcgen05 integer MMA at
# all: the external llc's "Cannot select llvm.nvvm.tcgen05.mma.shared" at
# sm_103a was CORRECT arch dispatch, not a bug (asm-routing it merely moves
# the same refusal to ptxas). The i8 legs below run at sm_100a only and the
# i8 runtime case needs a CC 10.0 device (B200); e4m3 runs on 10.0 and 10.3.

include(joinpath(dirname(dirname(@__DIR__)), "test", "setup.jl"))

const IDP_BM = 128
const IDP_BN = 256
const IDP_BK = 32                       # 1-byte containers → dense K = 32
const IDP_A_BYTES = IDP_BM * IDP_BK     # 4096
const IDP_B_BYTES = IDP_BN * IDP_BK     # 8192
const IDP_AB_WORDS = (IDP_A_BYTES + IDP_B_BYTES) ÷ 4
const IDP_STRIDE = 8 * IDP_BK           # 256-byte 8-row stride, B32 swizzle

const IDP_IDESC_I8   = UInt32(2 << 4) | UInt32(1 << 7) | UInt32(1 << 10) |
                       UInt32((IDP_BN >> 3) << 17) | UInt32((IDP_BM >> 4) << 24)
const IDP_IDESC_E4M3 = UInt32(1 << 4) |
                       UInt32((IDP_BN >> 3) << 17) | UInt32((IDP_BM >> 4) << 24)
@assert IDP_IDESC_I8   == 0x084004A0
@assert IDP_IDESC_E4M3 == 0x08400010

const IDP_FILL_S8   = UInt32(0x01010101)  # four packed s8 ones
const IDP_FILL_E4M3 = UInt32(0x38383838)  # four packed e4m3 1.0

# Shared prologue/epilogue; the mma spelling and output element type differ
# per kind, so the two kernels are spelled out (an experiment file — clarity
# over deduplication).

function _idp_i8_kernel!(O::CuDeviceVector{Int32, 1})
    smem      = CuStaticSharedArray(UInt32, IDP_AB_WORDS)
    mbar      = CuStaticSharedArray(UInt64, 1)
    tmem_slot = CuStaticSharedArray(UInt32, 1)

    a_addr    = smem_addr_u32(pointer(smem))
    b_addr    = a_addr + UInt32(IDP_A_BYTES)
    bar_addr  = smem_addr_u32(pointer(mbar))
    slot_addr = smem_addr_u32(pointer(tmem_slot))
    mb_ptr    = pointer(mbar)

    tid  = ptx"mov.u32"(sreg"tid.x")
    nthr = ptx"mov.u32"(sreg"ntid.x")
    lane = tid & UInt32(31)

    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mb_ptr, UInt32(1))
    end
    if tid < UInt32(32)
        ptx"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"(
            slot_addr, UInt32(512))
    end

    i = tid
    while i < UInt32(IDP_AB_WORDS)
        @inbounds smem[Int(i) + 1] = IDP_FILL_S8
        i += nthr
    end
    ptx"bar.sync"(Val(0))

    tmem   = @inbounds tmem_slot[1]
    a_desc = tcgen05_descriptor(a_addr; leading_bytes = 16,
                                stride_bytes = IDP_STRIDE,
                                swizzle = BlackwellLayout.B32)
    b_desc = tcgen05_descriptor(b_addr; leading_bytes = 16,
                                stride_bytes = IDP_STRIDE,
                                swizzle = BlackwellLayout.B32)

    if tid == UInt32(0)
        # pred = false → D = A·B (overwrite, no accumulate).
        ptx"tcgen05.mma.cta_group::1.kind::i8"(
            tmem, a_desc, b_desc, IDP_IDESC_I8, false)
        ptx"tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"(
            bar_addr)
    end
    barrier_try_wait(mb_ptr, UInt32(0))

    tmem_addr = tmem_lane_addr(tmem, lane)
    if tid < UInt32(32)
        dst = ptx"tcgen05.ld.sync.aligned.32x32b.x64.b32"(tmem_addr)
        ptx"tcgen05.wait::ld.sync.aligned"()
        base = Int(lane) * 64
        for c in 1:64
            @inbounds O[base + c] = reinterpret(Int32, dst[c])
        end
    end

    if tid < UInt32(32)
        ptx"tcgen05.dealloc.cta_group::1.sync.aligned.b32"(tmem, UInt32(512))
        ptx"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"()
    end
    return nothing
end

function _idp_e4m3_kernel!(O::CuDeviceVector{Float32, 1})
    smem      = CuStaticSharedArray(UInt32, IDP_AB_WORDS)
    mbar      = CuStaticSharedArray(UInt64, 1)
    tmem_slot = CuStaticSharedArray(UInt32, 1)

    a_addr    = smem_addr_u32(pointer(smem))
    b_addr    = a_addr + UInt32(IDP_A_BYTES)
    bar_addr  = smem_addr_u32(pointer(mbar))
    slot_addr = smem_addr_u32(pointer(tmem_slot))
    mb_ptr    = pointer(mbar)

    tid  = ptx"mov.u32"(sreg"tid.x")
    nthr = ptx"mov.u32"(sreg"ntid.x")
    lane = tid & UInt32(31)

    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mb_ptr, UInt32(1))
    end
    if tid < UInt32(32)
        ptx"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"(
            slot_addr, UInt32(512))
    end

    i = tid
    while i < UInt32(IDP_AB_WORDS)
        @inbounds smem[Int(i) + 1] = IDP_FILL_E4M3
        i += nthr
    end
    ptx"bar.sync"(Val(0))

    tmem   = @inbounds tmem_slot[1]
    a_desc = tcgen05_descriptor(a_addr; leading_bytes = 16,
                                stride_bytes = IDP_STRIDE,
                                swizzle = BlackwellLayout.B32)
    b_desc = tcgen05_descriptor(b_addr; leading_bytes = 16,
                                stride_bytes = IDP_STRIDE,
                                swizzle = BlackwellLayout.B32)

    if tid == UInt32(0)
        ptx"tcgen05.mma.cta_group::1.kind::f8f6f4"(
            tmem, a_desc, b_desc, IDP_IDESC_E4M3, false)
        ptx"tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"(
            bar_addr)
    end
    barrier_try_wait(mb_ptr, UInt32(0))

    tmem_addr = tmem_lane_addr(tmem, lane)
    if tid < UInt32(32)
        dst = ptx"tcgen05.ld.sync.aligned.32x32b.x64.b32"(tmem_addr)
        ptx"tcgen05.wait::ld.sync.aligned"()
        base = Int(lane) * 64
        for c in 1:64
            @inbounds O[base + c] = reinterpret(Float32, dst[c])
        end
    end

    if tid < UInt32(32)
        ptx"tcgen05.dealloc.cta_group::1.sync.aligned.b32"(tmem, UInt32(512))
        ptx"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"()
    end
    return nothing
end

# caps = compile targets for the emit/ptxas legs; runs(cc) = runtime gate.
# i8 is sm_100a-exclusive (see the target-gate note above).
const _IDP_CASES = (
    ("kind::i8  s8×s8→s32   ", _idp_i8_kernel!, Tuple{CuDeviceVector{Int32, 1}},
     Int32, Int32(IDP_BK), "tcgen05.mma.cta_group::1.kind::i8",
     (v"10.0",), cc -> cc.major == 10 && cc.minor == 0,
     "kind::i8 is sm_100a-exclusive — needs a CC 10.0 device (B200)"),
    ("kind::f8f6f4 e4m3→f32 ", _idp_e4m3_kernel!, Tuple{CuDeviceVector{Float32, 1}},
     Float32, Float32(IDP_BK), "tcgen05.mma.cta_group::1.kind::f8f6f4",
     (v"10.0", v"10.3"), cc -> v"10.0" <= cc < v"12.0",
     "needs a CC 10.x/11.x device"),
)

_target_name(cap) = "sm_$(cap.major * 10 + cap.minor)a"

println("=" ^ 72)
println("[emit] mma spelling and hand-encoded idesc reach the PTX")
for (label, f, tt, _, _, needle, caps, _, _) in _IDP_CASES, cap in caps
    ptx = emit_ptx(f, tt; cap, feature_set = :arch)
    println("  ", _target_name(cap), " ", label,
            ": mma spelled = ", occursin(needle, ptx),
            ", alloc/commit present = ",
            occursin("tcgen05.alloc", ptx) && occursin("tcgen05.commit", ptx))
end

println("=" ^ 72)
println("[ptxas] assembly")
for (label, f, tt, _, _, _, caps, _, _) in _IDP_CASES, cap in caps
    accepted, err = try
        ptxas_compiles(f, tt; cap, feature_set = :arch), ""
    catch e
        false, sprint(showerror, e)
    end
    println("  ", _target_name(cap), " ", label, ": ",
            accepted ? "ACCEPTED" : "REJECTED")
    accepted || println(err)
end

println("=" ^ 72)
devcap = try
    CUDACore.functional() ? CUDACore.capability(CUDACore.device()) : nothing
catch
    nothing
end
if devcap === nothing
    println("[runtime] SKIPPED — no functional device")
else
    println("[runtime] device = ", CUDACore.name(CUDACore.device()), " CC ", devcap)
    for (label, f, _, elt, expected, _, _, runs, skipmsg) in _IDP_CASES
        if !runs(devcap)
            println("  ", label, ": SKIP — ", skipmsg, " (found CC $devcap)")
            continue
        end
        O = CUDACore.zeros(elt, 32 * 64)
        @cuda blocks = 1 threads = 128 f(O)
        CUDACore.synchronize()
        got = Array(O)
        ok = all(==(expected), got)
        println("  ", label, ": ", ok ? "PASS (all $expected)" :
                "FAIL — unique values: $(unique(got)[1:min(end, 8)])")
    end
end
println("=" ^ 72)
println("done — record the runtime verdict in EVIDENCE.toml / the idesc-builder PR")
