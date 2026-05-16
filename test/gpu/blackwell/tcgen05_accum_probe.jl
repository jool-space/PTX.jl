# Blackwell tcgen05 accumulation probe. Ported from pyptx
# examples/blackwell/tcgen05_accum_probe.py (default build: num_mmas=4,
# advance_descs=True, scale_c=None, split_after=None).
#
# Run the same 128×256×16 f16-kind UMMA four times against all-ones bf16
# A/B tiles, then read back the first 32×64 accumulator subtile:
#
#   step 0:  enable_input_d = false → D  = A·B          (= 16)
#   step k>0: enable_input_d = true → D += A·B          (+16 each)
#   → every element = 4·16 = 64.0 exactly.
#
# This exercises the *accumulate predicate* of the blackwell-1
# tcgen05.mma wrapper — confirming `enable_input_d::Bool` is the
# D += A·B / D = A·B selector (CUTLASS's scaleC/accumulate). No new
# wrappers; scale_c immediate / split_after variants stay deferred (the
# default pyptx path uses neither).
#
# advance_descs (pyptx default True): each step nudges the SMEM
# descriptor by 2·step (the descriptor-stage-step pattern). Uniform
# all-ones tiles make this numerically inert but it keeps the
# add-on-descriptor path in the port. Uniform 0x3F803F80 fill is
# swizzle-invariant → linear fill, swizzle math skipped (see mma_probe).
#
# Static SMEM (< 48 KiB). Always ptxas-validates cross-arch at sm_100a;
# runtime gated on datacenter Blackwell [10.0, 11.0) — B300 = sm_103a.

using PTX: smem_addr_u32, tcgen05_descriptor, tcgen05_instr_desc_f16bf16_f32,
           BlackwellLayout, tmem_lane_addr
using CUDACore

const TAP_BM      = 128
const TAP_BN      = 256
const TAP_BK      = 16
const TAP_A_BYTES = TAP_BM * TAP_BK * 2
const TAP_B_BYTES = TAP_BN * TAP_BK * 2
const TAP_AB_WORDS = (TAP_A_BYTES + TAP_B_BYTES) ÷ 4
const TAP_PACKED_BF16_ONE = UInt32(0x3F803F80)
const TAP_STRIDE  = TAP_BK * 16
const TAP_NUM_MMAS = 4

function _tcgen05_accum_probe_kernel!(O::CuDeviceVector{Float32, 1})
    smem      = CuStaticSharedArray(UInt32, TAP_AB_WORDS)
    mbar      = CuStaticSharedArray(UInt64, 1)
    tmem_slot = CuStaticSharedArray(UInt32, 1)

    a_ptr     = pointer(smem)
    a_addr    = smem_addr_u32(a_ptr)
    b_addr    = a_addr + UInt32(TAP_A_BYTES)
    mb_ptr    = pointer(mbar)
    bar_addr  = smem_addr_u32(mb_ptr)
    slot_addr = smem_addr_u32(pointer(tmem_slot))

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

    w = Int(tid) + 1
    while w <= TAP_AB_WORDS
        @inbounds smem[w] = TAP_PACKED_BF16_ONE
        w += Int(nthr)
    end
    ptx"bar.sync"(Val(0))

    tmem   = @inbounds tmem_slot[1]
    idesc  = tcgen05_instr_desc_f16bf16_f32(; m = TAP_BM, n = TAP_BN,
                                            ab_dtype = :bf16)
    a_desc = tcgen05_descriptor(a_addr; leading_bytes = 16,
                                stride_bytes = TAP_STRIDE,
                                swizzle = BlackwellLayout.B32)
    b_desc = tcgen05_descriptor(b_addr; leading_bytes = 16,
                                stride_bytes = TAP_STRIDE,
                                swizzle = BlackwellLayout.B32)

    if tid == UInt32(0)
        for step in 0:(TAP_NUM_MMAS - 1)
            da = a_desc + UInt64(2 * step)        # advance_descs (inert here)
            db = b_desc + UInt64(2 * step)
            # step 0 → p=false (D = A·B); step>0 → p=true (D += A·B).
            ptx"tcgen05.mma.cta_group::1.kind::f16"(tmem, da, db, idesc,
                                                    step != 0)
        end
        ptx"tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"(
            bar_addr)
    end

    # parity 0 — see tcgen05_mma_probe.jl (pyptx uses 1; blackwell-1's
    # B300-verified kernels use 0 for the init(1)+single-commit pattern).
    while !ptx"mbarrier.try_wait.parity.shared.b64"(mb_ptr, UInt32(0))
    end

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

@testset "tcgen05 accumulation probe compiles at sm_100a" begin
    types = Tuple{CuDeviceVector{Float32, 1}}
    @test ptxas_compiles(_tcgen05_accum_probe_kernel!, types;
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_tcgen05_accum_probe_kernel!, types;
                   cap = v"10.0", feature_set = :arch)
    # The 4× accumulation (D += A·B) is verified by the runtime ==64
    # check below — the loop may stay rolled (1 textual mma) or unroll.
    @test occursin("tcgen05.mma.cta_group::1.kind::f16", ptx)
    @test occursin("tcgen05.ld.sync.aligned.32x32b.x64.b32", ptx)
end

# Datacenter-Blackwell only [10.0, 11.0); see tcgen05_smoke.jl rationale.
if v"10.0" <= DEV_CAP < v"11.0"
    @testset "tcgen05 accumulation probe (B300 runtime, 4·A·B = 64)" begin
        O = CUDACore.zeros(Float32, 32 * 64)
        @cuda blocks=1 threads=128 feature_set=:arch _tcgen05_accum_probe_kernel!(O)
        CUDACore.synchronize()
        got = reshape(Array(O), 64, 32)
        @test got == fill(64.0f0, 64, 32)
    end
end
