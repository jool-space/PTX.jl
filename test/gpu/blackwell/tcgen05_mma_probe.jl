# Blackwell tcgen05 single-MMA numeric probe. Ported from pyptx
# examples/blackwell/tcgen05_mma_probe.py.
#
# Fill A (128×16 bf16) and B (256×16 bf16) SMEM tiles with bf16 ones, run
# ONE 128×256×16 f16-kind UMMA (enable_input_d = false → D = A·B, no
# accumulate), read back the first 32×64 accumulator subtile. Each element
# is sum_{k=1..16} 1·1 = 16.0 exactly.
#
# This is the numeric counterpart to blackwell-1's smoke tier: it proves
# tcgen05.mma actually computes A·B (not just "did it launch"), on the
# wrapper surface landed in blackwell-1 — no new wrappers.
#
# The pyptx source writes the operand fill through a K-major 32B-swizzle
# permutation; a uniform 0x3F803F80 (two packed bf16 ones) fill is
# swizzle-invariant, so a plain linear fill is byte-for-byte identical
# and lets the probe skip the swizzle index arithmetic (the descriptors
# still declare 32B swizzle, so the MMA reads the tiles correctly).
#
# A|B|bar|slot = 12320 B < 48 KiB → static SMEM, no dynamic opt-in.
# Always ptxas-validates cross-arch at sm_100a; runtime gated on
# datacenter Blackwell [10.0, 11.0) — B300 (sm_103a) is the cloud target.

using PTX: smem_addr_u32, tcgen05_descriptor, tcgen05_instr_desc_f16bf16_f32,
           BlackwellLayout, tmem_lane_addr
using PTX.MBarriers: barrier_try_wait
using CUDACore

const TMP_BM      = 128
const TMP_BN      = 256
const TMP_BK      = 16
const TMP_A_BYTES = TMP_BM * TMP_BK * 2     # 4096
const TMP_B_BYTES = TMP_BN * TMP_BK * 2     # 8192
const TMP_AB_WORDS = (TMP_A_BYTES + TMP_B_BYTES) ÷ 4
const TMP_PACKED_BF16_ONE = UInt32(0x3F803F80)
# K-major operand row = BK*2 = 32 B → 32B swizzle; stride = BK*16.
const TMP_STRIDE  = TMP_BK * 16             # 256

function _tcgen05_mma_probe_kernel!(O::CuDeviceVector{Float32, 1})
    smem      = CuStaticSharedArray(UInt32, TMP_AB_WORDS)
    mbar      = CuStaticSharedArray(UInt64, 1)
    tmem_slot = CuStaticSharedArray(UInt32, 1)

    a_ptr     = pointer(smem)
    a_addr    = smem_addr_u32(a_ptr)
    b_addr    = a_addr + UInt32(TMP_A_BYTES)
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

    # Uniform bf16-ones fill (swizzle-invariant — see header).
    w = Int(tid) + 1
    while w <= TMP_AB_WORDS
        @inbounds smem[w] = TMP_PACKED_BF16_ONE
        w += Int(nthr)
    end
    ptx"bar.sync"(Val(0))

    tmem   = @inbounds tmem_slot[1]
    idesc  = tcgen05_instr_desc_f16bf16_f32(; m = TMP_BM, n = TMP_BN,
                                            ab_dtype = :bf16)
    a_desc = tcgen05_descriptor(a_addr; leading_bytes = 16,
                                stride_bytes = TMP_STRIDE,
                                swizzle = BlackwellLayout.B32)
    b_desc = tcgen05_descriptor(b_addr; leading_bytes = 16,
                                stride_bytes = TMP_STRIDE,
                                swizzle = BlackwellLayout.B32)

    if tid == UInt32(0)
        # pred_operand = false → p = 0 → D = A·B (overwrite, no accumulate).
        ptx"tcgen05.mma.cta_group::1.kind::f16"(tmem, a_desc, b_desc, idesc, false)
        ptx"tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"(
            bar_addr)
    end

    # pyptx waits try_wait.parity … 1 here; blackwell-1's mma_only (same
    # mbarrier.init(1) + single tcgen05.commit pattern) is B300-verified
    # with parity 0, so we use 0 for consistency with the green kernels.
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

@testset "tcgen05 single-MMA probe compiles at sm_100a" begin
    types = Tuple{CuDeviceVector{Float32, 1}}
    @test ptxas_compiles(_tcgen05_mma_probe_kernel!, types;
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_tcgen05_mma_probe_kernel!, types;
                   cap = v"10.0", feature_set = :arch)
    @test occursin("tcgen05.mma.cta_group::1.kind::f16", ptx)
    @test occursin("tcgen05.ld.sync.aligned.32x32b.x64.b32", ptx)
    @test occursin("tcgen05.wait::ld.sync.aligned", ptx)
end

# Datacenter-Blackwell only [10.0, 11.0); see tcgen05_smoke.jl rationale.
if v"10.0" <= DEV_CAP < v"11.0"
    @testset "tcgen05 single-MMA probe (B300 runtime, A·B = 16)" begin
        O = CUDACore.zeros(Float32, 32 * 64)
        @cuda blocks=1 threads=128 _tcgen05_mma_probe_kernel!(O)
        CUDACore.synchronize()
        got = reshape(Array(O), 64, 32)          # row-major (col, row)
        @test got == fill(16.0f0, 64, 32)
    end
end
