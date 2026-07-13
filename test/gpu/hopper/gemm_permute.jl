# TEST_TARGET: requires=toolkit evidence=mixed target=sm_90a
# GEMM with fused output permutation — ported from
# cutlass/examples/53_hopper_gemm_permute (NVIDIA CUTLASS, BSD-3-Clause).
#
# The CUTLASS example fuses tensor-mode permutations into a Hopper GEMM
# so no separate reorder pass (and no intermediate gmem round-trip) is
# needed — the PyTorch motivator being
#
#   D = torch.mm(A, B).view(M/D1, D1, D2, N/D2).permute(0, 2, 1, 3)
#
# The fusion mechanism on Hopper is the TMA descriptor: the epilogue
# stores through a tensor map whose multi-mode strides ENCODE the
# permutation, so the "reorder" is free address math in the TMA unit.
# The port does exactly that with the M-mode split (D2 = 1 case):
#
#   D[m, n], m folded as (m1, m0) = (m ÷ 16, m mod 16)
#   output memory layout = permuted modes (m0, m1):
#     D_perm[((m0·4 + m1)·8 + n)] = D[m, n]
#
# The kernel computes the standard 64×8 wgmma tile, stages it in SMEM
# (n-fast, the natural epilogue layout), and thread 0 issues ONE rank-3
# `cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group` whose
# dims walk SMEM in (n, m0, m1) order while the global strides jump in
# permuted (m0-major) order — note the strides are deliberately
# NON-MONOTONIC across ranks (128 B for dim1, 32 B for dim2): mode
# transposition is exactly a stride reordering, which the TMA driver
# accepts as long as each stride is 16 B-aligned.
#
# Not ported: A/B-side gather permutations (the input-side analog; the
# corpus covers input-side index indirection in gemm_gather_scatter.jl),
# batched D2 > 1 shapes (same mechanism, one more rank), and the
# CollectiveBuilder plumbing.

using PTX: layout_for_a, wgmma_descriptor, smem_addr_u32,
           tensor_map_encode_tiled, tensor_map_tile_2d
using CUDACore
using Random

const PRM_BM = 64
const PRM_BN = 8
const PRM_BK = 16
const PRM_M0 = 16                    # inner logical M-mode
const PRM_M1 = 4                     # outer logical M-mode (M = M0·M1)
const PRM_THREADS = 128
const PRM_LOAD_BYTES = PRM_BM * PRM_BK * 2 + PRM_BK * PRM_BN * 2

function _prm_gemm_kernel!(
        tma_D::PTX.TMADescriptorPtr,
        tma_A::PTX.TMADescriptorPtr,
        tma_B::PTX.TMADescriptorPtr,
        K::Int32)

    smem_A = CuStaticSharedArray(UInt16, PRM_BM * PRM_BK)
    smem_B = CuStaticSharedArray(UInt16, PRM_BK * PRM_BN)
    smem_D = CuStaticSharedArray(Float32, PRM_BM * PRM_BN)
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

    la = layout_for_a(dtype = :bf16, m = PRM_BM, k = PRM_BK)
    lb = layout_for_a(dtype = :bf16, m = PRM_BN, k = PRM_BK)
    a_desc = wgmma_descriptor(a_addr;
        leading_byte_offset = la.leading_byte_offset,
        stride_byte_offset  = la.stride_byte_offset,
        swizzle             = la.layout_type)
    b_desc = wgmma_descriptor(b_addr;
        leading_byte_offset = lb.leading_byte_offset,
        stride_byte_offset  = lb.stride_byte_offset,
        swizzle             = lb.layout_type)

    d = ntuple(_ -> 0f0, Val(4))
    num_k_tiles = K >> Int32(4)

    @inbounds for k_iter in Int32(0):(num_k_tiles - Int32(1))
        if tid == UInt32(0)
            ptx"mbarrier.arrive.expect_tx.shared.b64"(mb_ptr, UInt32(PRM_LOAD_BYTES))
            k_off = k_iter * Int32(PRM_BK)
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
        d = ptx"wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16"(
            d, a_desc, b_desc, true)
        ptx"wgmma.commit_group.sync.aligned"()
        ptx"wgmma.wait_group.sync.aligned"(Val(0))
        ptx"bar.sync"(Val(0))
    end

    # ── Epilogue: stage frags in SMEM (natural n-fast layout) ──────────
    wid      = tid >> UInt32(5)
    lane     = tid & UInt32(31)
    frag_row = (wid << UInt32(4)) + (lane >> UInt32(2))
    frag_col = (lane & UInt32(3)) << UInt32(1)
    @inbounds begin
        base_a = frag_row * UInt32(PRM_BN) + frag_col
        base_b = (frag_row + UInt32(8)) * UInt32(PRM_BN) + frag_col
        smem_D[base_a + UInt32(1)] = d[1]
        smem_D[base_a + UInt32(2)] = d[2]
        smem_D[base_b + UInt32(1)] = d[3]
        smem_D[base_b + UInt32(2)] = d[4]
    end
    ptx"bar.sync"(Val(0))
    # Generic-proxy SMEM writes must be visible to the async proxy before
    # the bulk store reads them (same fence as tma_epilogue.jl).
    ptx"fence.proxy.async.shared::cta"()

    # ── Fused permute: ONE rank-3 TMA store through permuted strides ───
    if tid == UInt32(0)
        ptx"cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"(
            tma_D, Int32(0), Int32(0), Int32(0), pointer(smem_D))
        ptx"cp.async.bulk.commit_group"()
        ptx"cp.async.bulk.wait_group.read"(Val(0))
    end
    return nothing
end

# ── Cross-arch ptxas validation ────────────────────────────────────────

@testset "permute-epilogue GEMM compiles at sm_90a" begin
    types = Tuple{PTX.TMADescriptorPtr, PTX.TMADescriptorPtr,
                  PTX.TMADescriptorPtr, Int32}
    @test ptxas_compiles(_prm_gemm_kernel!, types;
                         cap = v"9.0", feature_set = :arch)

    ptx = emit_ptx(_prm_gemm_kernel!, types; cap = v"9.0", feature_set = :arch)
    @test occursin("cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group", ptx)
    @test occursin("cp.async.bulk.commit_group", ptx)
    @test occursin("cp.async.bulk.wait_group.read", ptx)
    @test occursin("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16", ptx)
end

# ── Runtime — Hopper hardware ──────────────────────────────────────────

if test_runtime_supported(@__FILE__)
    @testset "GEMM + fused (m1, m0) → (m0, m1) output permute (K=32)" begin
        rng = MersenneTwister(0x53)
        K_test = 32
        A_f32 = randn(rng, Float32, PRM_BM, K_test) .* 0.1f0
        B_f32 = randn(rng, Float32, K_test, PRM_BN) .* 0.1f0

        A_packed = Array{UInt16}(undef, K_test, PRM_BM)
        B_packed = Array{UInt16}(undef, K_test, PRM_BN)
        for m in 1:PRM_BM, k in 1:K_test
            A_packed[k, m] = bf16_bits(A_f32[m, k])
        end
        for k in 1:K_test, n in 1:PRM_BN
            B_packed[k, n] = bf16_bits(B_f32[k, n])
        end
        A_d = CuArray(A_packed)
        B_d = CuArray(B_packed)

        tmap_A = tensor_map_tile_2d(:bf16, pointer(A_d),
            PRM_BM, K_test, PRM_BM, PRM_BK; swizzle = :B32)
        tmap_B = tensor_map_tile_2d(:bf16, pointer(B_d),
            PRM_BN, K_test, PRM_BN, PRM_BK; swizzle = :B32)

        # Output map: dims walk SMEM as (n, m0, m1); strides land each
        # element at its PERMUTED home ((m0·M1 + m1)·N + n)·4 B. dim1
        # stride (128 B) > dim2 stride (32 B) — the mode transposition.
        D_dev = CUDACore.zeros(Float32, PRM_BM * PRM_BN)
        tmap_D = tensor_map_encode_tiled(:f32, pointer(D_dev),
            (PRM_BN, PRM_M0, PRM_M1),
            (PRM_M1 * PRM_BN * 4, PRM_BN * 4),
            (PRM_BN, PRM_M0, PRM_M1);
            swizzle = :NONE)

        A = upload_tma_descriptor(tmap_A)
        B = upload_tma_descriptor(tmap_B)
        Dd = upload_tma_descriptor(tmap_D)

        @cuda threads = PRM_THREADS _prm_gemm_kernel!(
            Dd.ptr, A.ptr, B.ptr, Int32(K_test))
        CUDACore.synchronize()

        D_flat = Array(D_dev)
        D_ref = bf16_gemm_ref(A_f32, B_f32)

        # Logical row m folds as (m1, m0) = (m ÷ M0, m mod M0); the fused
        # store must have placed D[m, n] at the transposed-mode offset.
        ok = true
        for m in 0:(PRM_BM - 1), n in 0:(PRM_BN - 1)
            m1, m0 = divrem(m, PRM_M0)
            got = D_flat[(m0 * PRM_M1 + m1) * PRM_BN + n + 1]
            ok &= isapprox(got, D_ref[m + 1, n + 1]; rtol = 1e-3, atol = 1e-3)
        end
        @test ok
    end
end
