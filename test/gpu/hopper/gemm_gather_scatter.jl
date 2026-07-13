# TEST_TARGET: requires=toolkit evidence=mixed target=sm_90a
# Gather → GEMM → scatter fusion — ported from
# cutlass/examples/52_hopper_gather_scatter_fusion (NVIDIA CUTLASS,
# BSD-3-Clause).
#
# The CUTLASS example fuses a row-gather of A (through an index vector)
# before the GEMM and a row-scatter of D (through another index vector)
# after it, into a single kernel — no separate permute passes, and the
# problem shape is the shape AFTER gather / BEFORE scatter. Gather always
# runs along a strided dimension (rows of a row-major matrix) so the
# contiguous K-dim loads stay vectorized. Notably CUTLASS does NOT use TMA
# for the gathered operand — its mainloop falls back to cp.async — because
# TMA boxes can't take per-row indices.
#
# What's ported, on the single-warpgroup gemm_warpgroup.jl skeleton
# (BM=64, BN=8, BK=16 bf16, one K-tile):
#
#   - A-row gather fused into the mainloop: the warpgroup cooperatively
#     copies row `gather_idx[r]` of the A source into SMEM tile row r —
#     one 16-byte flit (8 bf16) per thread — writing through the B32
#     swizzle pattern by hand so the tile is byte-identical to what a
#     full-tile B32 TMA load would have produced, and the standard B32
#     wgmma descriptor reads it unchanged. (Per-row 1×BK TMA boxes were
#     tried first and die with CUDA_ERROR_MISALIGNED_ADDRESS: a swizzled
#     TMA requires its SMEM destination aligned to the swizzle atom, and
#     odd rows land at 32-B-aligned addresses. Same reason CUTLASS's
#     gather collective is cp.async-based.)
#     The B32 placement rule, probed empirically on H100 (matches cute's
#     Swizzle<1,4,3>): 16-B flit c of row r lands at flit c ^ ((r>>2)&1);
#     bytes within a flit stay in order.
#   - D-row scatter fused into the epilogue: each lane's two frag rows
#     store to output rows `scatter_idx[frag_row]` / `scatter_idx[frag_row+8]`
#     instead of frag_row directly. Rows not named by scatter_idx are
#     never written.
#
# B is a normal full-tile TMA load (its own mbarrier). The generic-proxy
# gather stores need the same fence.proxy.async.shared::cta before wgmma
# that the TMA path needs — wgmma reads SMEM through the async proxy
# either way.
#
# Not ported: gather along K/N and the C operand. CUTLASS 52 checks
# against a host reference built with the same index vectors; so does
# this test, plus an untouched-rows-stay-zero check pinning the
# scatter's write-set.

using PTX: wgmma_descriptor, smem_addr_u32, layout_for_a, tensor_map_tile_2d
using CUDACore
using Random

const GS_BM      = 64          # gathered rows fed to the GEMM
const GS_BN      = 8
const GS_BK      = 16
const GS_M_TOTAL = 128         # A source rows (gather domain)
const GS_M_OUT   = 96          # D rows (scatter range)
const GS_THREADS = 128         # one warpgroup

# Only B comes through the mbarrier: BK × BN × 2 bytes.
const GS_B_BYTES = GS_BK * GS_BN * 2

function _gs_gemm_kernel!(
        D::CuDeviceVector{Float32, 1},
        A_src::CuDeviceVector{UInt16, 1},
        gather_idx::CuDeviceVector{Int32, 1},
        scatter_idx::CuDeviceVector{Int32, 1},
        tma_B::PTX.TMADescriptorPtr)

    smem_A = CuStaticSharedArray(UInt16, GS_BM * GS_BK)
    smem_B = CuStaticSharedArray(UInt16, GS_BK * GS_BN)
    mbar   = CuStaticSharedArray(UInt64, 1)

    a_ptr  = pointer(smem_A)
    b_ptr  = pointer(smem_B)
    mb_ptr = pointer(mbar)
    a_addr = smem_addr_u32(a_ptr)
    b_addr = smem_addr_u32(b_ptr)

    tid = ptx"mov.u32"(sreg"tid.x")

    # Thread 0: init the mbarrier + issue B's TMA (gemm_warpgroup.jl issue
    # block, minus A).
    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mb_ptr, UInt32(1))
        ptx"fence.proxy.async.shared::cta"()
        ptx"mbarrier.arrive.expect_tx.shared.b64"(mb_ptr, UInt32(GS_B_BYTES))
        ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
            b_ptr, tma_B, Int32(0), Int32(0), mb_ptr)
    end

    # Cooperative fused gather: thread t moves one 16-B flit — tile row
    # r = t>>1, flit c = t&1 — from source row gather_idx[r] into the SMEM
    # tile, applying the B32 placement (flit c ^ ((r>>2)&1)) so the tile
    # matches a B32 TMA load bit-for-bit.
    r = Int(tid >> UInt32(1))
    c = Int(tid & UInt32(1))
    @inbounds begin
        src_row  = Int(gather_idx[r + 1])
        c_sw     = xor(c, (r >> 2) & 1)
        src_base = src_row * GS_BK + c * 8         # u16 index into A_src
        dst_base = r * GS_BK + c_sw * 8            # u16 index into smem_A
        for j in 1:8
            smem_A[dst_base + j] = A_src[src_base + j]
        end
    end

    # Publish the gather (generic proxy) CTA-wide, then wait for B's TMA.
    ptx"bar.sync"(Val(0))
    while !ptx"mbarrier.test_wait.parity.shared.b64"(mb_ptr, UInt32(0))
    end

    # Cross-proxy fence covers BOTH the TMA-written B tile and the
    # st.shared-written A tile before wgmma reads them via the async proxy.
    ptx"fence.proxy.async.shared::cta"()
    ptx"wgmma.fence.sync.aligned"()

    # Both tiles K-fast; K-major canonical descriptors (trans_b=0 — see
    # gemm_warpgroup.jl). The gathered A tile is indistinguishable from a
    # full-tile B32 TMA load by the time wgmma reads it.
    la = layout_for_a(dtype = :bf16, m = GS_BM, k = GS_BK)
    lb = layout_for_a(dtype = :bf16, m = GS_BN, k = GS_BK)
    a_desc = wgmma_descriptor(a_addr;
        leading_byte_offset = la.leading_byte_offset,
        stride_byte_offset  = la.stride_byte_offset,
        swizzle             = la.layout_type)
    b_desc = wgmma_descriptor(b_addr;
        leading_byte_offset = lb.leading_byte_offset,
        stride_byte_offset  = lb.stride_byte_offset,
        swizzle             = lb.layout_type)

    d = ntuple(_ -> 0f0, Val(4))
    d = ptx"wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16"(
        d, a_desc, b_desc, false)

    ptx"wgmma.commit_group.sync.aligned"()
    ptx"wgmma.wait_group.sync.aligned"(Val(0))

    # Epilogue with fused row-scatter: the m64n8 f32 frag layout puts lane
    # frags at rows (frag_row, frag_row+8); both redirect through
    # scatter_idx before computing the global offset.
    wid      = tid >> UInt32(5)
    lane     = tid & UInt32(31)
    frag_row = (wid << UInt32(4)) + (lane >> UInt32(2))
    frag_col = (lane & UInt32(3)) << UInt32(1)
    pd       = pointer(D)
    row_a    = @inbounds scatter_idx[Int(frag_row) + 1]
    row_b    = @inbounds scatter_idx[Int(frag_row) + 9]
    off_a    = (UInt32(row_a) * UInt32(GS_BN) + frag_col) * UInt32(4)
    off_b    = (UInt32(row_b) * UInt32(GS_BN) + frag_col) * UInt32(4)
    ptx"st.global.v2.f32"(pd + Int(off_a), (d[1], d[2]))
    ptx"st.global.v2.f32"(pd + Int(off_b), (d[3], d[4]))
    return nothing
end

# ── Cross-arch ptxas validation ────────────────────────────────────────

@testset "gather-GEMM-scatter fusion compiles at sm_90a" begin
    types = Tuple{CuDeviceVector{Float32, 1}, CuDeviceVector{UInt16, 1},
                  CuDeviceVector{Int32, 1}, CuDeviceVector{Int32, 1},
                  PTX.TMADescriptorPtr}
    @test ptxas_compiles(_gs_gemm_kernel!, types;
                         cap = v"9.0", feature_set = :arch)

    ptx = emit_ptx(_gs_gemm_kernel!, types; cap = v"9.0", feature_set = :arch)
    @test occursin("mbarrier.arrive.expect_tx.shared.b64",                ptx)
    @test occursin("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes", ptx)
    @test occursin("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16", ptx)
    # The manual gather: global loads of A flow into shared stores.
    @test occursin("st.shared", ptx)
end

# ── Runtime — Hopper hardware ──────────────────────────────────────────

if test_runtime_supported(@__FILE__)
    @testset "gather-GEMM-scatter (gather 64 of 128 rows, scatter into 96)" begin
        rng = MersenneTwister(0x9a77e5)

        A_src = Float32.(randn(rng, GS_M_TOTAL, GS_BK)) .* 0.1f0
        B_f32 = Float32.(randn(rng, GS_BK, GS_BN)) .* 0.1f0

        # Index vectors, 0-based for the device. Gather: 64 distinct rows
        # of the 128-row source. Scatter: 64 distinct rows in [0, 96).
        gather_idx  = Int32.(shuffle(rng, 0:(GS_M_TOTAL - 1))[1:GS_BM])
        scatter_idx = Int32.(shuffle(rng, 0:(GS_M_OUT - 1))[1:GS_BM])

        # Pack the FULL A source K-fast (the kernel gathers rows out of
        # it); B packs as usual.
        A_packed = Array{UInt16}(undef, GS_BK, GS_M_TOTAL)
        B_packed = Array{UInt16}(undef, GS_BK, GS_BN)
        for m in 1:GS_M_TOTAL, k in 1:GS_BK
            A_packed[k, m] = bf16_bits(A_src[m, k])
        end
        for n in 1:GS_BN, k in 1:GS_BK
            B_packed[k, n] = bf16_bits(B_f32[k, n])
        end
        A_d = CuArray(vec(A_packed))
        B_d = CuArray(B_packed)

        tmap_B = tensor_map_tile_2d(:bf16, pointer(B_d),
            GS_BN, GS_BK, GS_BN, GS_BK; swizzle = :B32)
        B = upload_tma_descriptor(tmap_B)

        gidx_d = CuArray(gather_idx)
        sidx_d = CuArray(scatter_idx)

        # D covers the full scatter range; unscattered rows must stay 0.
        D = CUDACore.zeros(Float32, GS_M_OUT * GS_BN)
        @cuda threads = GS_THREADS _gs_gemm_kernel!(
            D, A_d, gidx_d, sidx_d, B.ptr)
        CUDACore.synchronize()

        D_packed = reshape(Array(D), GS_BN, GS_M_OUT)
        D_got = Array{Float32}(undef, GS_M_OUT, GS_BN)
        for m in 1:GS_M_OUT, n in 1:GS_BN
            D_got[m, n] = D_packed[n, m]
        end

        # Reference: gather host-side with the same indices, GEMM, scatter.
        A_gathered = A_src[gather_idx .+ 1, :]
        D_tile = bf16_gemm_ref(A_gathered, B_f32)
        D_ref = zeros(Float32, GS_M_OUT, GS_BN)
        for m in 1:GS_BM
            D_ref[scatter_idx[m] + 1, :] = D_tile[m, :]
        end
        @test isapprox(D_got, D_ref; rtol = 1e-3, atol = 1e-3)

        # The scatter's write-set is exactly scatter_idx: every row NOT in
        # it must still be all-zero.
        untouched = setdiff(0:(GS_M_OUT - 1), scatter_idx)
        @test all(iszero, D_got[untouched .+ 1, :])
    end
end
