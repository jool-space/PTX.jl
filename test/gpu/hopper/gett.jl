# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc==9.0
# GETT — Generalized Tensor times Tensor contraction — ported from
# cutlass/examples/51_hopper_gett (NVIDIA CUTLASS, BSD-3-Clause).
#
# The CUTLASS example's thesis: every Hopper GEMM is a GETT in disguise.
# Tensor contraction modes fold into four semantic groups (M-modes shared
# by A and D, N-modes shared by B and D, K-modes contracted away, batch
# modes everywhere), and CuTe's multi-mode strides let the SAME mainloop
# run a contraction — only the layouts change, "the lexical spelling of
# the kernels remains the same".
#
# The port makes the identical point with the TMA descriptor as the
# layout carrier: the contraction
#
#   D[(m0, m1), n] = Σ_k  A[k, m0, m1] · B[k, n]
#
# runs on the UNMODIFIED single-warpgroup wgmma mainloop of
# gemm_warpgroup.jl. The only change is host-side: A's tensor map is
# rank-3 (multi-mode M = m0 × m1, k innermost), encoded with
# tensor_map_encode_tiled instead of the 2D convenience wrapper, and the
# kernel's A-load is the 3d TMA form with a (k, m0, m1) coordinate
# triple. The M-mode split is real, not cosmetic: the box covers all of
# m0 but only HALF of m1, so the grid walks the outer M-mode (2 CTAs,
# each contracting a different m1-slice) — the "batched-GEMM-that-isn't"
# shape from the example. TMA's box-fill order (dim0 innermost) lays
# (m1_local, m0) rows contiguously in SMEM, so the wgmma descriptor is
# byte-identical to the plain 2D GEMM's.
#
# Not ported: multi-mode N/K/batch (same mechanism, more ranks — rank-3
# on one operand demonstrates the layout-only claim), and the
# CollectiveBuilder major-mode detection (no builder here; the K-major
# choice is explicit).

using PTX: layout_for_a, wgmma_descriptor, smem_addr_u32,
           tensor_map_encode_tiled, tensor_map_tile_2d
using CUDACore
using Random

const GETT_M0 = 32                   # inner M-mode (contiguous after k)
const GETT_M1 = 4                    # outer M-mode (grid-split)
const GETT_M1_BOX = 2                # m1 per CTA → CTA tile M = 64
const GETT_BM = GETT_M0 * GETT_M1_BOX
const GETT_BN = 8
const GETT_BK = 16
const GETT_THREADS = 128
const GETT_LOAD_BYTES = GETT_BM * GETT_BK * 2 + GETT_BK * GETT_BN * 2

function _gett_kernel!(
        D::CuDeviceVector{Float32, 1},
        tma_A::PTX.TMADescriptorPtr,
        tma_B::PTX.TMADescriptorPtr,
        K::Int32)

    smem_A = CuStaticSharedArray(UInt16, GETT_BM * GETT_BK)
    smem_B = CuStaticSharedArray(UInt16, GETT_BK * GETT_BN)
    mbar   = CuStaticSharedArray(UInt64, 1)

    a_ptr  = pointer(smem_A)
    b_ptr  = pointer(smem_B)
    mb_ptr = pointer(mbar)
    a_addr = smem_addr_u32(a_ptr)
    b_addr = smem_addr_u32(b_ptr)

    tid    = ptx"mov.u32"(sreg"tid.x")
    cta    = ptx"mov.u32"(sreg"ctaid.x")
    m1_off = Int32(cta) * Int32(GETT_M1_BOX)

    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mb_ptr, UInt32(1))
        ptx"fence.proxy.async.shared::cta"()
    end
    ptx"bar.sync"(Val(0))

    la = layout_for_a(dtype = :bf16, m = GETT_BM, k = GETT_BK)
    lb = layout_for_a(dtype = :bf16, m = GETT_BN, k = GETT_BK)
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
            ptx"mbarrier.arrive.expect_tx.shared.b64"(mb_ptr, UInt32(GETT_LOAD_BYTES))
            k_off = k_iter * Int32(GETT_BK)
            # Rank-3 A: coordinate is (k, m0, m1) — the GETT surface.
            # m0 spans its full extent (coord 0), m1 is grid-split.
            ptx"cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                a_ptr, tma_A, k_off, Int32(0), m1_off, mb_ptr)
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
        ptx"bar.sync"(Val(0))               # drain before re-arming the buffer
    end

    # Standard m64n8 f32 frag epilogue; each CTA owns a 64×8 slab of D.
    wid      = tid >> UInt32(5)
    lane     = tid & UInt32(31)
    frag_row = (wid << UInt32(4)) + (lane >> UInt32(2))
    frag_col = (lane & UInt32(3)) << UInt32(1)
    pd       = pointer(D) + Int(cta) * GETT_BM * GETT_BN * 4
    off_a    = (frag_row * UInt32(GETT_BN) + frag_col) * UInt32(4)
    off_b    = ((frag_row + UInt32(8)) * UInt32(GETT_BN) + frag_col) * UInt32(4)
    ptx"st.global.v2.f32"(pd + Int(off_a), (d[1], d[2]))
    ptx"st.global.v2.f32"(pd + Int(off_b), (d[3], d[4]))
    return nothing
end

# ── Cross-arch ptxas validation ────────────────────────────────────────

@testset "GETT kernel compiles at sm_90a" begin
    types = Tuple{CuDeviceVector{Float32, 1},
                  PTX.TMADescriptorPtr, PTX.TMADescriptorPtr, Int32}
    @test ptxas_compiles(_gett_kernel!, types;
                         cap = v"9.0", feature_set = :arch)

    ptx = emit_ptx(_gett_kernel!, types; cap = v"9.0", feature_set = :arch)
    @test occursin("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes", ptx)
    @test occursin("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes", ptx)
    @test occursin("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16", ptx)
end

# ── Runtime — Hopper hardware ──────────────────────────────────────────

if test_runtime_supported(@__FILE__)
    @testset "GETT D[(m0,m1),n] = Σk A[k,m0,m1]·B[k,n] (K=64, 2 CTAs)" begin
        rng = MersenneTwister(0x9e77)
        K_test = 64
        # A as a genuine rank-3 tensor, k innermost — Julia col-major
        # (K, M0, M1) gives exactly that.
        A_f32 = randn(rng, Float32, K_test, GETT_M0, GETT_M1) .* 0.1f0
        B_f32 = randn(rng, Float32, K_test, GETT_BN) .* 0.1f0

        A_packed = bf16_bits.(A_f32)
        B_packed = bf16_bits.(B_f32)
        A_d = CuArray(A_packed)
        B_d = CuArray(B_packed)

        # Rank-3 multi-mode M: dims innermost-first (k, m0, m1), byte
        # strides for the outer two modes. Box covers full k-tile and m0
        # but half of m1 → the grid supplies the m1 coordinate.
        tmap_A = tensor_map_encode_tiled(:bf16, pointer(A_d),
            (K_test, GETT_M0, GETT_M1),
            (K_test * 2, K_test * GETT_M0 * 2),
            (GETT_BK, GETT_M0, GETT_M1_BOX);
            swizzle = :B32)
        tmap_B = tensor_map_tile_2d(:bf16, pointer(B_d),
            GETT_BN, K_test, GETT_BN, GETT_BK; swizzle = :B32)
        A = upload_tma_descriptor(tmap_A)
        B = upload_tma_descriptor(tmap_B)

        n_ctas = GETT_M1 ÷ GETT_M1_BOX
        D_dev = CUDACore.zeros(Float32, n_ctas * GETT_BM * GETT_BN)
        @cuda threads = GETT_THREADS blocks = n_ctas _gett_kernel!(
            D_dev, A.ptr, B.ptr, Int32(K_test))
        CUDACore.synchronize()

        # CTA c, SMEM row r ↦ modes m0 = r mod 32, m1 = 2c + r ÷ 32
        # (TMA box-fill order: k innermost, then m0, then m1).
        D_flat = Array(D_dev)
        D_got = Array{Float32}(undef, GETT_M0, GETT_M1, GETT_BN)
        for c in 0:(n_ctas - 1), r in 0:(GETT_BM - 1), n in 0:(GETT_BN - 1)
            m0 = r % GETT_M0
            m1 = c * GETT_M1_BOX + r ÷ GETT_M0
            D_got[m0 + 1, m1 + 1, n + 1] = D_flat[c * GETT_BM * GETT_BN + r * GETT_BN + n + 1]
        end

        # Reference contraction over the rank-3 A, bf16-rounded inputs.
        Ab = bf16_to_f32.(A_packed)
        Bb = bf16_to_f32.(B_packed)
        D_ref = zeros(Float32, GETT_M0, GETT_M1, GETT_BN)
        for m0 in 1:GETT_M0, m1 in 1:GETT_M1, n in 1:GETT_BN, k in 1:K_test
            D_ref[m0, m1, n] += Ab[k, m0, m1] * Bb[k, n]
        end
        @test isapprox(D_got, D_ref; rtol = 1e-3, atol = 1e-3)
    end
end
