# Blackwell tcgen05 smoke kernels — alloc / mma / ld isolated from the full
# GEMM path. Ported from pyptx examples/blackwell/tcgen05_smoke.py
# (build_alloc_only / build_mma_only / build_ld_only).
#
# Each kernel writes a constant marker to O[1] so a successful launch is
# observable without a numeric reference: alloc → 11, mma → 22, ld → 33.
# These are bring-up shake-out tests for the tcgen05 wrapper surface
# (alloc, dealloc, relinquish_alloc_permit, mma, commit, wait::ld, ld) —
# the Blackwell analog of Hopper's gemm_warpgroup smoke kernel. Numeric
# TMEM data integrity is covered separately by tcgen05_roundtrip.jl.
#
# Always validates cross-arch (sm_100a ptxas) — runs on the sm_89 dev box
# where the kernels cannot launch. Runtime launch is gated on real
# Blackwell datacenter hardware (CC >= 10.0); B300 (sm_103a) is the cloud
# target. SMEM layout mirrors pyptx exactly (A | B | mbar | tmem_slot)
# and exceeds 48 KiB, so the kernels use dynamic SMEM with an explicit
# MAX_DYNAMIC_SHARED_SIZE_BYTES opt-in (same idiom as
# gemm_highperf_hopper.jl).

using PTX: smem_addr_u32, tcgen05_descriptor, tcgen05_instr_desc_f16bf16_f32,
           BlackwellLayout
using PTX.MBarriers: barrier_try_wait
using CUDACore

# pyptx tcgen05_smoke.py SMEM map (bytes).
const TCS_A_BYTES   = 128 * 64 * 2
const TCS_B_BYTES   = 256 * 64 * 2
const TCS_BAR_OFF   = TCS_A_BYTES + TCS_B_BYTES
const TCS_SLOT_OFF  = TCS_BAR_OFF + 16
const TCS_SMEM      = TCS_SLOT_OFF + 16

# --- alloc_only -----------------------------------------------------------
# Allocate 512 TMEM columns, read the returned base back out of the
# .shared::cta slot, free it, relinquish the per-CTA alloc permit.
function _tcgen05_alloc_only_kernel!(O::CuDeviceVector{Float32, 1})
    tmem_slot = CuDynamicSharedArray(UInt32, 1, TCS_SLOT_OFF)
    slot_addr = smem_addr_u32(pointer(tmem_slot))

    tid = ptx"mov.u32"(sreg"tid.x")
    if tid < UInt32(32)
        ptx"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"(
            slot_addr, UInt32(512))
    end
    ptx"bar.sync"(Val(0))

    tmem = @inbounds tmem_slot[1]
    if tid < UInt32(32)
        ptx"tcgen05.dealloc.cta_group::1.sync.aligned.b32"(tmem, UInt32(512))
        ptx"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"()
    end
    if tid == UInt32(0)
        @inbounds O[1] = 11.0f0
    end
    return nothing
end

# --- mma_only -------------------------------------------------------------
# One dense f16-kind UMMA over all-zero B128-swizzled operand tiles, then
# commit → mbarrier try_wait::parity drains the async MMA before teardown.
# pyptx writes the zero fill through a 128B swizzle permutation; an
# all-zero fill is swizzle-invariant, so a plain linear zeroing is
# byte-for-byte identical and lets the smoke test skip the swizzle index
# arithmetic (the descriptors still declare B128 swizzle).
function _tcgen05_mma_only_kernel!(O::CuDeviceVector{Float32, 1})
    smem      = CuDynamicSharedArray(UInt32, (TCS_A_BYTES + TCS_B_BYTES) ÷ 4, 0)
    mbar      = CuDynamicSharedArray(UInt64, 1, TCS_BAR_OFF)
    tmem_slot = CuDynamicSharedArray(UInt32, 1, TCS_SLOT_OFF)

    a_ptr     = pointer(smem)
    a_addr    = smem_addr_u32(a_ptr)
    b_addr    = a_addr + UInt32(TCS_A_BYTES)
    mb_ptr    = pointer(mbar)
    bar_addr  = smem_addr_u32(mb_ptr)
    slot_addr = smem_addr_u32(pointer(tmem_slot))

    tid  = ptx"mov.u32"(sreg"tid.x")
    nthr = ptx"mov.u32"(sreg"ntid.x")

    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mb_ptr, UInt32(1))
    end
    if tid < UInt32(32)
        ptx"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"(
            slot_addr, UInt32(512))
    end

    # Cooperative all-zero fill of the A | B operand region.
    w = Int(tid) + 1
    N = length(smem)
    while w <= N
        @inbounds smem[w] = UInt32(0)
        w += Int(nthr)
    end
    ptx"bar.sync"(Val(0))

    tmem   = @inbounds tmem_slot[1]
    idesc  = tcgen05_instr_desc_f16bf16_f32(; m = 128, n = 256, ab_dtype = :bf16)
    a_desc = tcgen05_descriptor(a_addr; leading_bytes = 16,
                                stride_bytes = 1024, swizzle = BlackwellLayout.B128)
    b_desc = tcgen05_descriptor(b_addr; leading_bytes = 16,
                                stride_bytes = 1024, swizzle = BlackwellLayout.B128)

    if tid == UInt32(0)
        # enable_input_d = true → predicate operand p = 1 (matches pyptx's
        # pred_operand=True path; D = A·B with the all-zero tiles → 0).
        ptx"tcgen05.mma.cta_group::1.kind::f16"(tmem, a_desc, b_desc, idesc, true)
        ptx"tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"(
            bar_addr)
    end

    barrier_try_wait(mb_ptr, UInt32(0))

    if tid < UInt32(32)
        ptx"tcgen05.dealloc.cta_group::1.sync.aligned.b32"(tmem, UInt32(512))
        ptx"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"()
    end
    if tid == UInt32(0)
        @inbounds O[1] = 22.0f0
    end
    return nothing
end

# --- ld_only --------------------------------------------------------------
# Allocate, then a single 32x32b.x1 TMEM load (uninitialized TMEM — the
# value is discarded; this isolates the tcgen05.ld addressing path).
function _tcgen05_ld_only_kernel!(O::CuDeviceVector{Float32, 1})
    tmem_slot = CuDynamicSharedArray(UInt32, 1, TCS_SLOT_OFF)
    slot_addr = smem_addr_u32(pointer(tmem_slot))

    tid = ptx"mov.u32"(sreg"tid.x")
    if tid < UInt32(32)
        ptx"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"(
            slot_addr, UInt32(512))
    end
    ptx"bar.sync"(Val(0))

    tmem = @inbounds tmem_slot[1]
    v = ptx"tcgen05.ld.sync.aligned.32x32b.x1.b32"(tmem)
    ptx"tcgen05.wait::ld.sync.aligned"()

    if tid < UInt32(32)
        ptx"tcgen05.dealloc.cta_group::1.sync.aligned.b32"(tmem, UInt32(512))
        ptx"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"()
    end
    if tid == UInt32(0)
        # Fold `v` into the marker so the load can't be DCE'd, but keep the
        # observable value the fixed 33 sentinel (v is uninitialized TMEM).
        @inbounds O[1] = 33.0f0 + 0.0f0 * Float32(v)
    end
    return nothing
end

@testset "tcgen05 smoke kernels compile at sm_100a" begin
    # Always-on cross-arch validation: ptxas accepts the full tcgen05
    # wrapper surface under sm_100a. Catches wrapper regressions on the
    # sm_89 dev box where these kernels cannot launch.
    types = Tuple{CuDeviceVector{Float32, 1}}

    @test ptxas_compiles(_tcgen05_alloc_only_kernel!, types;
                         cap = v"10.0", feature_set = :arch)
    @test ptxas_compiles(_tcgen05_mma_only_kernel!, types;
                         cap = v"10.0", feature_set = :arch)
    @test ptxas_compiles(_tcgen05_ld_only_kernel!, types;
                         cap = v"10.0", feature_set = :arch)

    ptx = emit_ptx(_tcgen05_mma_only_kernel!, types;
                   cap = v"10.0", feature_set = :arch)
    @test occursin("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32", ptx)
    @test occursin("tcgen05.mma.cta_group::1.kind::f16", ptx)
    @test occursin("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64", ptx)
    @test occursin("tcgen05.dealloc.cta_group::1.sync.aligned.b32", ptx)
    @test occursin("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned", ptx)

    ptx_ld = emit_ptx(_tcgen05_ld_only_kernel!, types;
                      cap = v"10.0", feature_set = :arch)
    @test occursin("tcgen05.ld.sync.aligned.32x32b.x1.b32", ptx_ld)
    @test occursin("tcgen05.wait::ld.sync.aligned", ptx_ld)
end

# Runtime path — gated on Blackwell *datacenter* hardware. tcgen05 ships
# on sm_100/sm_103 (B100/B200/B300, CC 10.x); consumer Blackwell
# (sm_120/sm_121, CC 12.x) dropped it. The datacenter-Blackwell window is
# [10.0, 11.0): admits B300 (sm_103a, CC 10.3) while skipping the GB10
# dev box (sm_121a) and the sm_89 box.
if v"10.0" <= DEV_CAP < v"11.0"
    @testset "tcgen05 smoke kernels (B300 runtime)" begin
        for (kernel!, marker) in (
                (_tcgen05_alloc_only_kernel!, 11.0f0),
                (_tcgen05_mma_only_kernel!,   22.0f0),
                (_tcgen05_ld_only_kernel!,    33.0f0))
            O = CUDACore.zeros(Float32, 1)
            kern = @cuda launch=false feature_set=:arch kernel!(O)
            attrs = CUDACore.attributes(kern.fun)
            attrs[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = TCS_SMEM
            kern(O; blocks = 1, threads = 128, shmem = TCS_SMEM)
            CUDACore.synchronize()
            @test Array(O)[1] == marker
        end
    end
end
