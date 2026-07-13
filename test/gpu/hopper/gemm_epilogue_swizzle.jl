# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc==9.0
# Int8 GEMM with SMEM-staged vectorized epilogue — ported from
# cutlass/examples/50_hopper_gemm_with_epilogue_swizzle (NVIDIA CUTLASS,
# BSD-3-Clause).
#
# The CUTLASS example hand-assembles a custom collective epilogue
# (sm70_epilogue_vectorized) instead of the default: an s8×s8→s32 GEMM
# whose s32 accumulators are converted to the s8 output type, staged
# through a swizzled SMEM tile, and only then written to global memory as
# wide vectorized stores — the wgmma fragment scatter never touches gmem
# directly. What's ported:
#
#   - s8×s8→s32 integer wgmma:
#     wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8 (first integer-wgmma
#     kernel in this corpus; integer wgmma takes only scale-d — no
#     scale-a/b or trans immediates, PTX ISA §9.7.14.5.7).
#   - Saturating epilogue conversion s32 → s8 (LinearCombination with
#     alpha, beta = 0, clamp to [-128, 127]) — integer end-to-end, so the
#     runtime test asserts EXACT equality against the host reference.
#   - SMEM-staged epilogue: per-lane s8 frags land in a swizzled SMEM
#     tile; after bar.sync each of 32 lanes takes one 16-byte physical
#     chunk and issues a single st.global.v4.b32 — 16 logically
#     consecutive output bytes per store instead of 4 scattered
#     byte-stores per lane.
#   - The swizzle: 16-B chunk q sits at physical chunk q ⊻ ((q >> 3) & 3)
#     — an involution keyed on bits the XOR doesn't touch, so the reader
#     recovers the logical index from the physical one (the same
#     layout-functor contract cute's Swizzle<> encodes; both sides of the
#     SMEM round-trip agree on the mapping and gmem coalescing is
#     preserved since the permutation stays within the tile). At this
#     tile size (512 B) bank conflicts are noise — what's ported is the
#     pattern, documented, not a perf claim.
#
# Not ported: the 8-stage pipeline / cluster-multicast mainloop config
# (pipelining is gemm_pc_pipeline.jl's brick; multicast is
# tma_multicast_cluster.jl's) — the mainloop here is the single-warpgroup
# TMA→mbarrier→wgmma brick from gemm_warpgroup.jl with a K-loop.

using PTX: layout_for_a, wgmma_descriptor, smem_addr_u32, tensor_map_tile_2d,
           step_desc
using CUDACore
using Random

const EPS_BM = 64
const EPS_BN = 8
const EPS_BK = 32                               # s8 wgmma K-step
const EPS_THREADS = 128                         # one warpgroup
const EPS_LOAD_BYTES = EPS_BM * EPS_BK + EPS_BK * EPS_BN
const EPS_D_BYTES = EPS_BM * EPS_BN             # 512 B output tile
const EPS_CHUNKS  = EPS_D_BYTES ÷ 16            # 32 × 16-B chunks

# The chunk-level swizzle involution. Keyed on bits 3-4, XORs bits 0-1 —
# key bits are untouched by the XOR, so applying it twice is the identity
# and the epilogue reader can recover a chunk's logical index from its
# physical one.
@inline eps_swizzle(q::UInt32) = q ⊻ ((q >> UInt32(3)) & UInt32(3))

function _eps_gemm_kernel!(
        D::CuDeviceVector{UInt32, 1},           # 64×8 s8 tile as 128 words
        tma_A::PTX.TMADescriptorPtr,
        tma_B::PTX.TMADescriptorPtr,
        K::Int32,
        alpha::Int32)

    smem_A = CuStaticSharedArray(UInt8, EPS_BM * EPS_BK)
    smem_B = CuStaticSharedArray(UInt8, EPS_BK * EPS_BN)
    smem_D = CuStaticSharedArray(UInt8, EPS_D_BYTES)
    mbar   = CuStaticSharedArray(UInt64, 1)

    a_ptr  = pointer(smem_A)
    b_ptr  = pointer(smem_B)
    mb_ptr = pointer(mbar)
    a_addr = smem_addr_u32(a_ptr)
    b_addr = smem_addr_u32(b_ptr)

    tid = ptx"mov.u32"(sreg"tid.x")

    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mb_ptr, UInt32(1))
        ptx"fence.proxy.async.shared::cta"()
    end
    ptx"bar.sync"(Val(0))

    la = layout_for_a(dtype = 1, m = EPS_BM, k = EPS_BK)   # elem_bytes = 1
    lb = layout_for_a(dtype = 1, m = EPS_BN, k = EPS_BK)
    a_desc = wgmma_descriptor(a_addr;
        leading_byte_offset = la.leading_byte_offset,
        stride_byte_offset  = la.stride_byte_offset,
        swizzle             = la.layout_type)
    b_desc = wgmma_descriptor(b_addr;
        leading_byte_offset = lb.leading_byte_offset,
        stride_byte_offset  = lb.stride_byte_offset,
        swizzle             = lb.layout_type)

    d = ntuple(_ -> Int32(0), Val(4))
    num_k_tiles = K >> Int32(5)                 # K / 32

    # Single SMEM buffer, serialized K-loop (no ring): TMA → wait → wgmma
    # → next tile. Phase alternates 0/1/0/… as the single mbarrier cycles.
    @inbounds for k_iter in Int32(0):(num_k_tiles - Int32(1))
        if tid == UInt32(0)
            ptx"mbarrier.arrive.expect_tx.shared.b64"(mb_ptr, UInt32(EPS_LOAD_BYTES))
            k_off = k_iter * Int32(EPS_BK)
            ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                a_ptr, tma_A, k_off, Int32(0), mb_ptr)
            ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                b_ptr, tma_B, k_off, Int32(0), mb_ptr)
        end
        phase = UInt32(k_iter & Int32(1))
        while !ptx"mbarrier.test_wait.parity.shared.b64"(mb_ptr, phase)
        end

        ptx"fence.proxy.async.shared::cta"()
        ptx"wgmma.fence.sync.aligned"()
        d = ptx"wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8"(
            d, a_desc, b_desc, true)
        ptx"wgmma.commit_group.sync.aligned"()
        ptx"wgmma.wait_group.sync.aligned"(Val(0))
        # The wait_group drains wgmma's SMEM reads, so the next TMA may
        # overwrite the buffer — but only after ALL threads pass it:
        # bar.sync before thread 0 re-arms.
        ptx"bar.sync"(Val(0))
    end

    # ── Epilogue: saturate s32 → s8, stage into swizzled SMEM ──────────
    # LinearCombination with integer compute: clamp(alpha·acc, s8 range).
    wid      = tid >> UInt32(5)
    lane     = tid & UInt32(31)
    frag_row = (wid << UInt32(4)) + (lane >> UInt32(2))
    frag_col = (lane & UInt32(3)) << UInt32(1)

    @inbounds for (i, dr) in ((1, UInt32(0)), (2, UInt32(0)),
                              (3, UInt32(8)), (4, UInt32(8)))
        r = frag_row + dr
        c = frag_col + UInt32(iseven(i) ? 1 : 0)
        v = clamp(alpha * d[i], Int32(-128), Int32(127)) % Int8
        logical  = r * UInt32(EPS_BN) + c
        q        = logical >> UInt32(4)
        physical = eps_swizzle(q) << UInt32(4) | (logical & UInt32(15))
        smem_D[physical + UInt32(1)] = reinterpret(UInt8, v)
    end
    ptx"bar.sync"(Val(0))

    # ── Vectorized drain: one st.global.v4.b32 per 16-B chunk ──────────
    # Lane p (< 32) reads PHYSICAL chunk p; the involution gives back the
    # LOGICAL chunk it holds, which fixes the gmem address.
    if tid < UInt32(EPS_CHUNKS)
        p32 = reinterpret(Core.LLVMPtr{UInt32, 3}, pointer(smem_D))
        base = Int(tid) * 4
        # Explicit alignment: the reinterpret from the byte array leaves
        # LLVM assuming align 1, which scalarizes the loads to ld.shared.b8.
        vec = (unsafe_load(p32, base + 1, Val(4)), unsafe_load(p32, base + 2, Val(4)),
               unsafe_load(p32, base + 3, Val(4)), unsafe_load(p32, base + 4, Val(4)))
        logical_chunk = eps_swizzle(tid)
        pd = pointer(D)
        ptx"st.global.v4.b32"(pd + Int(logical_chunk) * 16, vec)
    end

    return nothing
end

# ── Cross-arch ptxas validation ────────────────────────────────────────

@testset "epilogue-swizzle int8 GEMM compiles at sm_90a" begin
    types = Tuple{CuDeviceVector{UInt32, 1},
                  PTX.TMADescriptorPtr, PTX.TMADescriptorPtr, Int32, Int32}
    @test ptxas_compiles(_eps_gemm_kernel!, types;
                         cap = v"9.0", feature_set = :arch)

    ptx = emit_ptx(_eps_gemm_kernel!, types; cap = v"9.0", feature_set = :arch)
    @test occursin("wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8", ptx)
    @test occursin("st.global.v4.b32", ptx)
    @test occursin("st.shared.b8", ptx)         # the staging writes
    @test occursin("ld.shared.b32", ptx)        # the vectorized-drain reads
    @test occursin("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes", ptx)
end

# ── Runtime — Hopper hardware ──────────────────────────────────────────

if test_runtime_supported(@__FILE__)
    @testset "epilogue-swizzle int8 GEMM (exact integer, K=128)" begin
        rng = MersenneTwister(0x50)
        K_test = 128                            # 4 K-iters
        # Values small enough that alpha·acc explores both the linear and
        # the saturated range of s8 (K=128 × |a·b| ≤ 49 ≫ 127 would
        # saturate everything at ±7 — keep operands in [-3, 3] and alpha
        # small so a healthy mix of outputs clamps and doesn't).
        A_i = rand(rng, Int8.(-3:3), EPS_BM, K_test)
        B_i = rand(rng, Int8.(-3:3), K_test, EPS_BN)

        A_packed = Array{UInt8}(undef, K_test, EPS_BM)
        B_packed = Array{UInt8}(undef, K_test, EPS_BN)
        for m in 1:EPS_BM, k in 1:K_test
            A_packed[k, m] = reinterpret(UInt8, A_i[m, k])
        end
        for k in 1:K_test, n in 1:EPS_BN
            B_packed[k, n] = reinterpret(UInt8, B_i[k, n])
        end
        A_d = CuArray(A_packed)
        B_d = CuArray(B_packed)

        tmap_A = tensor_map_tile_2d(:u8, pointer(A_d),
            EPS_BM, K_test, EPS_BM, EPS_BK; swizzle = :B32)
        tmap_B = tensor_map_tile_2d(:u8, pointer(B_d),
            EPS_BN, K_test, EPS_BN, EPS_BK; swizzle = :B32)
        A = upload_tma_descriptor(tmap_A)
        B = upload_tma_descriptor(tmap_B)

        alpha = Int32(1)
        D_dev = CUDACore.zeros(UInt32, EPS_D_BYTES ÷ 4)
        @cuda threads = EPS_THREADS _eps_gemm_kernel!(
            D_dev, A.ptr, B.ptr, Int32(K_test), alpha)
        CUDACore.synchronize()

        D_bytes = reinterpret(Int8, Array(D_dev))   # row-major (M, N), N-fast
        D_got = Array{Int}(undef, EPS_BM, EPS_BN)
        for m in 1:EPS_BM, n in 1:EPS_BN
            D_got[m, n] = Int(D_bytes[(m - 1) * EPS_BN + n])
        end

        # Integer path end-to-end → exact.
        acc_ref = Int.(A_i) * Int.(B_i)
        D_ref = clamp.(Int(alpha) .* acc_ref, -128, 127)
        @test D_got == D_ref
        # The clamp must actually be exercised from both sides.
        @test any(==(127), D_ref) && any(==(-128), D_ref)
    end
end
