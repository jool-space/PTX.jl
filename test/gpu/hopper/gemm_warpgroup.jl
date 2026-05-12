# Single-warpgroup Hopper GEMM — TMA load → mbarrier sync → wgmma → store.
#
# Headline Hopper kernel for ROADMAP item 6. Composes the full sm_90a stack:
#
#   1. cuTensorMapEncodeTiled (host)   →  PTX.tensor_map_encode_tiled
#   2. cp.async.bulk.tensor.2d (device)→  TMA load A, B into SMEM
#   3. mbarrier.init/expect_tx/wait    →  hardware-sync TMA completion
#   4. fence.proxy.async + wgmma.fence →  cross-proxy ordering for wgmma
#   5. wgmma_descriptor                →  build 64-bit SMEM descriptors
#   6. wgmma.mma_async                 →  warpgroup-collective MMA
#   7. wgmma.commit_group/wait_group   →  drain the async pipeline
#   8. Per-lane store of f32 frags     →  output to D
#
# Tile shape m64n8k16 bf16.bf16.f32 — smallest wgmma m64n8k16 wgmma. Single
# warpgroup (128 threads), single CTA, single K-tile (no main loop).
#
# Runtime path needs H100 (wgmma is sm_90a-only — Blackwell uses tcgen05).
# Cross-arch ptxas-validation at v"9.0" always runs (catches wrapper-side
# regressions on the GB10 dev box, where the kernel can't actually launch).

using PTX: wgmma_descriptor, WgmmaSwizzle, smem_addr_u32, layout_for_a
using CUDACore
using Random

const HOPPER_BM = 64
const HOPPER_BN = 8
const HOPPER_BK = 16
const HOPPER_THREADS = 128  # one warpgroup

# Total TMA bytes the mbarrier waits on:
#   A: BM × BK × 2 bytes = 2048
#   B: BK × BN × 2 bytes = 256
const HOPPER_LOAD_BYTES = HOPPER_BM * HOPPER_BK * 2 + HOPPER_BK * HOPPER_BN * 2

function _hopper_warpgroup_gemm_kernel!(
        D::CuDeviceVector{Float32, 1},
        tma_A::Core.LLVMPtr{UInt8, PTX.AS.Const},
        tma_B::Core.LLVMPtr{UInt8, PTX.AS.Const})

    smem_A = CuStaticSharedArray(UInt16, HOPPER_BM * HOPPER_BK)
    smem_B = CuStaticSharedArray(UInt16, HOPPER_BK * HOPPER_BN)
    mbar   = CuStaticSharedArray(UInt64, 1)

    a_ptr  = pointer(smem_A)
    b_ptr  = pointer(smem_B)
    mb_ptr = pointer(mbar)
    a_addr = smem_addr_u32(a_ptr)
    b_addr = smem_addr_u32(b_ptr)

    tid = ptx"mov.u32"(sreg"tid.x")

    # 1. Lane 0 does the entire TMA-issue sequence (init, fence, arrive,
    #    bulk loads). pyptx's tests (_test_tma_fence.py etc.) keep these in
    #    one thread-0 block followed by a CTA-wide bar.sync — the bar.sync
    #    after the issue (NOT between init and arrive) is what makes the
    #    inited+armed state visible to all threads before any of them polls
    #    test_wait.parity. Splitting init / arrive with bar.sync between
    #    deadlocks on H100 (other threads enter the spin loop too early).
    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mb_ptr, UInt32(1))
        ptx"fence.proxy.async.shared::cta"()
        ptx"mbarrier.arrive.expect_tx.shared.b64"(mb_ptr, UInt32(HOPPER_LOAD_BYTES))
        ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
            a_ptr, tma_A, Int32(0), Int32(0), mb_ptr)
        ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
            b_ptr, tma_B, Int32(0), Int32(0), mb_ptr)
    end
    ptx"bar.sync"(Val(0))

    # 2. Every thread waits for TMA completion. test_wait.parity polls the
    #    phase bit — 0 → 1 on the first arrival round.
    while !ptx"mbarrier.test_wait.parity.shared.b64"(mb_ptr, UInt32(0))
    end

    # 3. Cross-proxy fence: TMA writes via async proxy; wgmma reads via async
    #    proxy too, but the descriptor build needs the SMEM bytes visible
    #    through the same proxy.
    ptx"fence.proxy.async.shared::cta"()
    ptx"wgmma.fence.sync.aligned"()

    # 4. Build A/B descriptors. Both tiles are K-fast in SMEM (so wgmma's
    #    trans_b=0 default reads B correctly as col-major / K-fast); both
    #    use the K-major canonical formula (`layout_for_a`). Row width =
    #    16 bf16 = 32 B → B32 swizzle, LBO=16, SBO=256.
    #    Earlier code used `layout_for_mn_major` (MN-major / N-fast canonical) for
    #    B, which mismatches the wrapper's baked trans_b=0 and would silently
    #    miscompute on non-uniform inputs. See grouped_gemm.jl for the same
    #    fix applied at port-time. The wgmma_descriptor builder doesn't
    #    distinguish A vs B operands — it just packs offsets into a UInt64.
    la = layout_for_a(dtype = :bf16, m = HOPPER_BM, k = HOPPER_BK)
    lb = layout_for_a(dtype = :bf16, m = HOPPER_BN, k = HOPPER_BK)
    a_desc = wgmma_descriptor(a_addr;
        leading_byte_offset = la.leading_byte_offset,
        stride_byte_offset  = la.stride_byte_offset,
        swizzle             = la.layout_type)
    b_desc = wgmma_descriptor(b_addr;
        leading_byte_offset = lb.leading_byte_offset,
        stride_byte_offset  = lb.stride_byte_offset,
        swizzle             = lb.layout_type)

    # 5. wgmma m64n8k16.f32.bf16.bf16 — 4 f32 regs per lane × 128 lanes.
    d = ntuple(_ -> 0f0, Val(4))
    d = ptx"wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16"(
        d, a_desc, b_desc, false)

    ptx"wgmma.commit_group.sync.aligned"()
    ptx"wgmma.wait_group.sync.aligned"(Val(0))

    # 6. Epilogue. m64n8k16 wgmma f32 layout (PTX 9.2 §9.7.14.5.5):
    #    warp w ∈ 0..3 owns rows [16w, 16w+16); lane l within a warp:
    #      frag_row = 16w + (l >> 2),    frag_col = 2 * (l & 3)
    #    4 f32 frags per lane at (frag_row, frag_col), (frag_row, frag_col+1),
    #    (frag_row+8, frag_col), (frag_row+8, frag_col+1). v2.f32 stores
    #    two consecutive cols at the same row in one transaction.
    wid      = tid >> UInt32(5)
    lane     = tid & UInt32(31)
    frag_row = (wid << UInt32(4)) + (lane >> UInt32(2))
    frag_col = (lane & UInt32(3)) << UInt32(1)
    pd       = pointer(D)
    off_a    = (frag_row * UInt32(HOPPER_BN) + frag_col) * UInt32(4)
    off_b    = ((frag_row + UInt32(8)) * UInt32(HOPPER_BN) + frag_col) * UInt32(4)
    ptx"st.global.v2.f32"(pd + Int(off_a), (d[1], d[2]))
    ptx"st.global.v2.f32"(pd + Int(off_b), (d[3], d[4]))
    return nothing
end

@testset "single-warpgroup Hopper GEMM compiles at sm_90a" begin
    # Always-on cross-arch validation: ptxas accepts the full kernel under
    # sm_90a. Catches wrapper regressions on the GB10 dev box where the
    # kernel can't actually launch.
    types = Tuple{CuDeviceVector{Float32, 1},
                  Core.LLVMPtr{UInt8, PTX.AS.Const},
                  Core.LLVMPtr{UInt8, PTX.AS.Const}}
    @test ptxas_compiles(_hopper_warpgroup_gemm_kernel!, types;
                         cap = v"9.0", feature_set = :arch)

    ptx = emit_ptx(_hopper_warpgroup_gemm_kernel!, types;
                   cap = v"9.0", feature_set = :arch)

    # Every layer of the canonical Hopper warpgroup-MMA pipeline appears.
    @test occursin("mbarrier.init.shared.b64",                       ptx)
    @test occursin("mbarrier.arrive.expect_tx.shared.b64",           ptx)
    @test occursin("mbarrier.test_wait.parity.shared.b64",           ptx)
    @test occursin("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes", ptx)
    @test occursin("fence.proxy.async.shared::cta",                  ptx)
    @test occursin("wgmma.fence.sync.aligned",                       ptx)
    @test occursin("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16", ptx)
    @test occursin("wgmma.commit_group.sync.aligned",                ptx)
    @test occursin("wgmma.wait_group.sync.aligned 0",                ptx)
    @test occursin("st.global.v2.f32",                               ptx)

    # Descriptor build constant-folds the layout-picked offsets into a single
    # OR with the runtime SMEM address (verified compile-only).
    @test occursin(r"or\.b64", ptx)
end

# Runtime path — gated on Hopper hardware. wgmma is sm_90a-only; Blackwell
# (sm_100/sm_120/sm_121) replaced it with tcgen05, so cap is required to be
# in [9.0, 10.0). On the GB10 dev box (sm_121a) this testset is skipped.
if v"9.0" <= DEV_CAP < v"10.0"
    bf16_bits(x::Float32) = UInt16((reinterpret(UInt32, x) + UInt32(0x8000)) >> 16)
    bf16_to_f32(b::UInt16) = reinterpret(Float32, UInt32(b) << 16)

    function _warpgroup_gemm_cpu_ref(A::Array{Float32, 2}, B::Array{Float32, 2})
        M, K = size(A); _, N = size(B)
        Ab = bf16_to_f32.(bf16_bits.(A))
        Bb = bf16_to_f32.(bf16_bits.(B))
        D = zeros(Float32, M, N)
        for m in 1:M, n in 1:N, k in 1:K
            D[m, n] += Ab[m, k] * Bb[k, n]
        end
        D
    end

    @testset "single-warpgroup Hopper GEMM (random bf16, K-fast layout)" begin
        # Random non-uniform inputs — exercises the layout fix (both A and B
        # K-fast in SMEM, both descriptors via layout_for_a). Uniform 1.0s
        # would have masked the previous N-fast B / trans_b=0 mismatch.
        rng = MersenneTwister(0x501a)
        A_f32 = Float32.(randn(rng, HOPPER_BM, HOPPER_BK)) .* 0.1f0
        B_f32 = Float32.(randn(rng, HOPPER_BK, HOPPER_BN)) .* 0.1f0

        # Pack to bf16. Memory layout = K-fast for both:
        #   A: Julia (K, M) col-major → K innermost
        #   B: Julia (K, N) col-major → K innermost
        A_packed = Array{UInt16}(undef, HOPPER_BK, HOPPER_BM)
        B_packed = Array{UInt16}(undef, HOPPER_BK, HOPPER_BN)
        for m in 1:HOPPER_BM, k in 1:HOPPER_BK
            A_packed[k, m] = bf16_bits(A_f32[m, k])
        end
        for n in 1:HOPPER_BN, k in 1:HOPPER_BK
            B_packed[k, n] = bf16_bits(B_f32[k, n])
        end
        A_d = CuArray(A_packed)
        B_d = CuArray(B_packed)

        # TMA descriptors. Both K-innermost (matching K-fast memory).
        # Row width = 16 bf16 = 32 B → B32 swizzle for both.
        tmap_A = PTX.tensor_map_tile_2d(:bf16, pointer(A_d),
            HOPPER_BM, HOPPER_BK, HOPPER_BM, HOPPER_BK; swizzle = :B32)
        tmap_B = PTX.tensor_map_tile_2d(:bf16, pointer(B_d),
            HOPPER_BN, HOPPER_BK, HOPPER_BN, HOPPER_BK; swizzle = :B32)

        tmap_A_dev = CuArray{UInt8}(undef, 128); copyto!(tmap_A_dev, collect(tmap_A.data))
        tmap_B_dev = CuArray{UInt8}(undef, 128); copyto!(tmap_B_dev, collect(tmap_B.data))
        a_const = reinterpret(Core.LLVMPtr{UInt8, PTX.AS.Const}, UInt64(pointer(tmap_A_dev)))
        b_const = reinterpret(Core.LLVMPtr{UInt8, PTX.AS.Const}, UInt64(pointer(tmap_B_dev)))

        # Output D allocated as (BN, BM) col-major = BN-innermost. The
        # kernel writes (frag_row * BN + frag_col) byte offsets — N-innermost
        # in global memory, matching the per-lane v2.f32 stride.
        D = CUDACore.zeros(Float32, HOPPER_BM * HOPPER_BN)
        @cuda threads = HOPPER_THREADS feature_set = :arch _hopper_warpgroup_gemm_kernel!(
            D, a_const, b_const)
        CUDACore.synchronize()

        D_packed = reshape(Array(D), HOPPER_BN, HOPPER_BM)
        D_got = Array{Float32}(undef, HOPPER_BM, HOPPER_BN)
        for m in 1:HOPPER_BM, n in 1:HOPPER_BN
            D_got[m, n] = D_packed[n, m]
        end

        D_ref = _warpgroup_gemm_cpu_ref(A_f32, B_f32)
        @test isapprox(D_got, D_ref; rtol = 1e-3, atol = 1e-3)
    end
end
