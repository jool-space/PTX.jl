# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc==9.0
# Hopper GEMM with fused top-K + softmax epilogue — ported from
# cutlass/examples/61_hopper_gemm_with_topk_and_softmax (NVIDIA CUTLASS,
# BSD-3-Clause).
#
# The CUTLASS example fuses the `LinCombTopKSoftmaxCol<2>` EVT node into a
# warp-specialized Hopper GEMM epilogue: per output row (fusion is over the
# N dimension), keep only the top-2 values, replace them with their softmax
# weights (normalized over the kept set only), and zero everything else:
#
#   m1 = max(row), m2 = second-max(row), sum = exp(m1-m1) + exp(m2-m1)
#   y_j = x_j < m2 ? 0 : exp(x_j - m1) / sum
#
# The example's assumption 3 (tile N ≥ problem N) is what makes the fusion
# a register-level reduction — one CTA sees the whole row. This port keeps
# that: BN = N = 8, so a row's 8 values live in one 4-lane wgmma frag group
# (lane t of the group owns cols {2t, 2t+1}), and top-2 reduces with two
# shfl.sync.bfly rounds (offsets 1, 2 — XOR on lane bits 0/1 stays inside
# the group). The denominator needs NO further reduction: the kept set is
# {m1, m2}, so sum = 1 + exp(m2 - m1), computable in every lane (this is
# exactly how the .cu host reference forms it).
#
# Ported: bf16 GEMM (TMA → mbarrier → wgmma m64n8k16, the gemm_warpgroup.jl
# brick), alpha pre-scale (beta = 0, C = void in the example), top-2 +
# softmax fused in-register before the store. exp goes through
# ex2.approx.f32 (~3 ULP; the example itself budgets for reduction error).
# Not ported: warp-specialized producer/consumer split (covered by
# gemm_pc_pipeline.jl), fp16 A/B (example uses half_t; corpus convention is
# bf16 — same wgmma family, different operand dtype), TopK = 4 variant.
#
# Tie caveat (mirrors the .cu comment verbatim): the formulation only works
# when the top-K elements are NOT repeated — after reduction there is no
# way to tell repeated elements apart. Device excludes *all* instances of
# m1 when forming m2; the host reference excludes one. With continuous
# random inputs, ties have measure zero.

using PTX: wgmma_descriptor, smem_addr_u32, layout_for_a
using CUDACore
using Random

const TKS_BM = 64
const TKS_BN = 8
const TKS_BK = 16
const TKS_THREADS = 128                 # one warpgroup
const TKS_LOAD_BYTES = TKS_BM * TKS_BK * 2 + TKS_BK * TKS_BN * 2
const TKS_LOG2E = 1.4426950408889634f0

# Butterfly max over the 4-lane frag group (lane id bits 0/1). All 32 lanes
# participate (full membermask); XOR offsets 1 and 2 never leave the group.
@inline function _tks_group4_max(v::Float32)
    full = UInt32(0xFFFFFFFF)
    seg  = UInt32(0x1F)
    Base.@nexprs 2 i -> begin
        u     = reinterpret(UInt32, v)
        u_par = ptx"shfl.sync.bfly.b32"(u, UInt32(i), seg, full)
        v     = ptx"max.f32"(v, reinterpret(Float32, u_par))
    end
    v
end

# Top-2 + softmax over one output row, given this lane's two column values.
# Returns the two post-fusion values in place.
@inline function _tks_top2_softmax(x1::Float32, x2::Float32)
    m1 = _tks_group4_max(max(x1, x2))
    # m2: group max excluding m1. A lane holding m1 candidates its other
    # value. (Excludes every instance of m1 — see tie caveat in header.)
    c1 = x1 == m1 ? -Inf32 : x1
    c2 = x2 == m1 ? -Inf32 : x2
    m2 = _tks_group4_max(max(c1, c2))
    # Kept set is {m1, m2} → denominator directly, no reduction:
    denom = 1f0 + ptx"ex2.approx.f32"((m2 - m1) * TKS_LOG2E)
    inv   = ptx"rcp.approx.f32"(denom)
    y1 = x1 < m2 ? 0f0 : ptx"ex2.approx.f32"((x1 - m1) * TKS_LOG2E) * inv
    y2 = x2 < m2 ? 0f0 : ptx"ex2.approx.f32"((x2 - m1) * TKS_LOG2E) * inv
    (y1, y2)
end

function _tks_gemm_kernel!(
        D::CuDeviceVector{Float32, 1},
        tma_A::PTX.TMADescriptorPtr,
        tma_B::PTX.TMADescriptorPtr,
        alpha::Float32)

    smem_A = CuStaticSharedArray(UInt16, TKS_BM * TKS_BK)
    smem_B = CuStaticSharedArray(UInt16, TKS_BK * TKS_BN)
    mbar   = CuStaticSharedArray(UInt64, 1)

    a_ptr  = pointer(smem_A)
    b_ptr  = pointer(smem_B)
    mb_ptr = pointer(mbar)
    a_addr = smem_addr_u32(a_ptr)
    b_addr = smem_addr_u32(b_ptr)

    tid = ptx"mov.u32"(sreg"tid.x")

    # TMA issue from lane 0, CTA-wide publish after (see gemm_warpgroup.jl
    # for why the bar.sync must come after the whole issue block).
    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mb_ptr, UInt32(1))
        ptx"fence.proxy.async.shared::cta"()
        ptx"mbarrier.arrive.expect_tx.shared.b64"(mb_ptr, UInt32(TKS_LOAD_BYTES))
        ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
            a_ptr, tma_A, Int32(0), Int32(0), mb_ptr)
        ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
            b_ptr, tma_B, Int32(0), Int32(0), mb_ptr)
    end
    ptx"bar.sync"(Val(0))

    while !ptx"mbarrier.test_wait.parity.shared.b64"(mb_ptr, UInt32(0))
    end

    ptx"fence.proxy.async.shared::cta"()
    ptx"wgmma.fence.sync.aligned"()

    # Both tiles K-fast; K-major canonical layout for both descriptors.
    la = layout_for_a(dtype = :bf16, m = TKS_BM, k = TKS_BK)
    lb = layout_for_a(dtype = :bf16, m = TKS_BN, k = TKS_BK)
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

    # ── Fused epilogue: alpha scale → top-2 + softmax per row ──────────
    # Frag layout (m64n8k16 f32): lane owns (frag_row, frag_col{,+1}) and
    # (frag_row+8, frag_col{,+1}); its 4-lane group covers each row's 8 cols.
    x1 = d[1] * alpha
    x2 = d[2] * alpha
    x3 = d[3] * alpha
    x4 = d[4] * alpha
    y1, y2 = _tks_top2_softmax(x1, x2)      # row frag_row
    y3, y4 = _tks_top2_softmax(x3, x4)      # row frag_row + 8

    wid      = tid >> UInt32(5)
    lane     = tid & UInt32(31)
    frag_row = (wid << UInt32(4)) + (lane >> UInt32(2))
    frag_col = (lane & UInt32(3)) << UInt32(1)
    pd       = pointer(D)
    off_a    = (frag_row * UInt32(TKS_BN) + frag_col) * UInt32(4)
    off_b    = ((frag_row + UInt32(8)) * UInt32(TKS_BN) + frag_col) * UInt32(4)
    ptx"st.global.v2.f32"(pd + Int(off_a), (y1, y2))
    ptx"st.global.v2.f32"(pd + Int(off_b), (y3, y4))
    return nothing
end

# ── Cross-arch ptxas validation ────────────────────────────────────────

@testset "GEMM + top-2 softmax fusion compiles at sm_90a" begin
    types = Tuple{CuDeviceVector{Float32, 1},
                  PTX.TMADescriptorPtr, PTX.TMADescriptorPtr, Float32}
    @test ptxas_compiles(_tks_gemm_kernel!, types;
                         cap = v"9.0", feature_set = :arch)

    ptx = emit_ptx(_tks_gemm_kernel!, types; cap = v"9.0", feature_set = :arch)
    @test occursin("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16", ptx)
    @test occursin("shfl.sync.bfly.b32", ptx)
    @test occursin("ex2.approx.f32",     ptx)
    @test occursin("rcp.approx.f32",     ptx)
    @test occursin("mbarrier.arrive.expect_tx.shared.b64", ptx)
end

# ── Runtime — Hopper hardware ──────────────────────────────────────────

# Host reference with the .cu's exact semantics: top-2 per row, softmax
# normalized over the kept pair, zeros elsewhere. Excludes ONE instance of
# the max when finding m2 (the device excludes all — ties are measure-zero
# with random input, see header).
function _tks_ref(X::Matrix{Float32})
    M, N = size(X)
    Y = zeros(Float32, M, N)
    for i in 1:M
        j1 = argmax(view(X, i, :))
        m1 = X[i, j1]
        m2 = maximum(X[i, j] for j in 1:N if j != j1)
        s  = 1f0 + exp(m2 - m1)
        for j in 1:N
            Y[i, j] = X[i, j] < m2 ? 0f0 : exp(X[i, j] - m1) / s
        end
    end
    Y
end

if test_runtime_supported(@__FILE__)
    @testset "GEMM + top-2 softmax (random bf16, all 64 rows)" begin
        rng = MersenneTwister(0x707c)
        A_f32 = Float32.(randn(rng, TKS_BM, TKS_BK)) .* 0.5f0
        B_f32 = Float32.(randn(rng, TKS_BK, TKS_BN)) .* 0.5f0
        alpha = 1.25f0

        # K-fast packing for both operands (gemm_warpgroup.jl convention).
        A_packed = Array{UInt16}(undef, TKS_BK, TKS_BM)
        B_packed = Array{UInt16}(undef, TKS_BK, TKS_BN)
        for m in 1:TKS_BM, k in 1:TKS_BK
            A_packed[k, m] = bf16_bits(A_f32[m, k])
        end
        for n in 1:TKS_BN, k in 1:TKS_BK
            B_packed[k, n] = bf16_bits(B_f32[k, n])
        end
        A_d = CuArray(A_packed)
        B_d = CuArray(B_packed)

        tmap_A = PTX.tensor_map_tile_2d(:bf16, pointer(A_d),
            TKS_BM, TKS_BK, TKS_BM, TKS_BK; swizzle = :B32)
        tmap_B = PTX.tensor_map_tile_2d(:bf16, pointer(B_d),
            TKS_BN, TKS_BK, TKS_BN, TKS_BK; swizzle = :B32)
        A = upload_tma_descriptor(tmap_A)
        B = upload_tma_descriptor(tmap_B)

        D = CUDACore.zeros(Float32, TKS_BM * TKS_BN)
        @cuda threads = TKS_THREADS _tks_gemm_kernel!(D, A.ptr, B.ptr, alpha)
        CUDACore.synchronize()

        D_packed = reshape(Array(D), TKS_BN, TKS_BM)
        D_got = Array{Float32}(undef, TKS_BM, TKS_BN)
        for m in 1:TKS_BM, n in 1:TKS_BN
            D_got[m, n] = D_packed[n, m]
        end

        X_ref = alpha .* bf16_gemm_ref(A_f32, B_f32)
        D_ref = _tks_ref(X_ref)
        @test isapprox(D_got, D_ref; rtol = 1e-3, atol = 1e-3)

        # Structural checks: every row keeps exactly 2 nonzeros summing to 1.
        for m in 1:TKS_BM
            @test count(!iszero, D_got[m, :]) == 2
            @test isapprox(sum(D_got[m, :]), 1f0; atol = 1e-3)
        end
    end
end
