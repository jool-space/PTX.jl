# Uniform-shape grouped GEMM (Blackwell) — algorithmic port of
# pyptx/examples/blackwell/grouped_gemm.py.
#
# G problems sharing (M,K)×(K,N) → (M,N): C[g] = A[g]·B[g]. Grid is
# (N/BN, M/BM, G); ctaid.z picks the group, folded into the TMA coords.
#
# Scope mirrors the Hopper grouped_gemm.jl "v0": single-stage (no
# warp-specialized 3-stage ring — that structure is shared with
# gemm_highperf_blackwell and is the deferred ROADMAP-11 piece). This
# composes two already-proven things:
#   * the Hopper grouped_gemm TMA host descriptor + mbarrier + phase-XOR
#     pipeline (validated on H100), and
#   * gemm_experimental's faithful tcgen05 compute (alloc → K-loop of
#     accumulating tcgen05.mma → commit → TMEM-ld epilogue).
#
# B300 access is gone, so this is validated by always-on cross-arch
# ptxas at sm_100a + faithful pyptx port + a bf16 CPU reference behind
# the runtime gate. Every mbarrier parity is pyptx-faithful (the
# deviation that bit gemm_experimental is deliberately NOT repeated):
# load barrier uses the phase-XOR-per-ktile trick (Hopper-proven),
# bar_mma uses parity 0 (pyptx grouped_gemm line 222).
#
# v0 tile: BM=128, BN=64, BK=64 (K-major 128B swizzle, K = multiple of
# 64). SMEM 24616 B < 48 KiB → static, no dynamic opt-in.

using PTX: smem_addr_u32, tcgen05_descriptor, tcgen05_instr_desc_f16bf16_f32,
           BlackwellLayout, tensor_map_tile_2d, CuTensorMap
using CUDACore
using Random

const GGB_BM = 128
const GGB_BN = 64               # = N (N % 64 == 0, N ≤ 256 → grid X = 1)
const GGB_BK = 64               # K-tile; 4 × k16 tcgen05.mma per ktile
const GGB_THREADS = 128
const GGB_A_STAGE = GGB_BM * GGB_BK * 2     # 16384
const GGB_B_STAGE = GGB_BN * GGB_BK * 2     # 8192
const GGB_LOAD_BYTES = GGB_A_STAGE + GGB_B_STAGE
const GGB_STRIDE = GGB_BK * 16              # 1024 (tcgen05 descriptor stride)

function _grouped_gemm_bw_kernel!(
        C::CuDeviceVector{Float32, 1},
        tma_A::PTX.TMADescriptorPtr,
        tma_B::PTX.TMADescriptorPtr,
        M::Int32, N::Int32, K::Int32)

    smem_A    = CuStaticSharedArray(UInt16, GGB_BM * GGB_BK)
    smem_B    = CuStaticSharedArray(UInt16, GGB_BN * GGB_BK)
    bar_load  = CuStaticSharedArray(UInt64, 1)
    bar_mma   = CuStaticSharedArray(UInt64, 1)
    tmem_slot = CuStaticSharedArray(UInt32, 1)

    a_ptr     = pointer(smem_A)
    b_ptr     = pointer(smem_B)
    bl_ptr    = pointer(bar_load)
    bm_ptr    = pointer(bar_mma)
    a_addr    = smem_addr_u32(a_ptr)
    b_addr    = smem_addr_u32(b_ptr)
    slot_addr = smem_addr_u32(pointer(tmem_slot))

    tid    = ptx"mov.u32"(sreg"tid.x")
    cta_n  = ptx"mov.u32"(sreg"ctaid.x")    # N tile (= 0, grid X = 1)
    cta_m  = ptx"mov.u32"(sreg"ctaid.y")    # M tile
    group  = ptx"mov.u32"(sreg"ctaid.z")    # which problem

    # Flattened-view bases. A:(G*M,K) row m_outer; B_T:(G*N,K) row n_outer;
    # C:(G*M,N) row-major, col base = cta_n*BN (N axis shared across groups).
    m_outer  = group * UInt32(M) + cta_m * UInt32(GGB_BM)
    n_outer  = group * UInt32(N) + cta_n * UInt32(GGB_BN)
    col_base = cta_n * UInt32(GGB_BN)

    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(bl_ptr, UInt32(1))
        ptx"mbarrier.init.shared.b64"(bm_ptr, UInt32(1))
    end
    if tid < UInt32(32)
        ptx"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"(
            slot_addr, UInt32(512))
    end
    ptx"bar.sync"(Val(0))

    tmem   = @inbounds tmem_slot[1]
    idesc  = tcgen05_instr_desc_f16bf16_f32(; m = GGB_BM, n = GGB_BN,
                                            ab_dtype = :bf16)
    a_desc0 = tcgen05_descriptor(a_addr; leading_bytes = 16,
                                 stride_bytes = GGB_STRIDE,
                                 swizzle = BlackwellLayout.B128)
    b_desc0 = tcgen05_descriptor(b_addr; leading_bytes = 16,
                                 stride_bytes = GGB_STRIDE,
                                 swizzle = BlackwellLayout.B128)

    # K-loop: TMA A/B tile → wait → 4 accumulating tcgen05.mma. The
    # accumulate predicate is false only on the very first MMA of the
    # very first ktile (pyptx is_first = ki==0 && kk==0); true after,
    # so the full K accumulates into one TMEM tile.
    phase = UInt32(0)
    k_off = Int32(0)
    while k_off < K
        if tid == UInt32(0)
            ptx"fence.proxy.async.shared::cta"()
            ptx"mbarrier.arrive.expect_tx.shared.b64"(bl_ptr,
                UInt32(GGB_LOAD_BYTES))
            ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                a_ptr, tma_A, k_off, reinterpret(Int32, m_outer), bl_ptr)
            ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                b_ptr, tma_B, k_off, reinterpret(Int32, n_outer), bl_ptr)
        end
        ptx"bar.sync"(Val(0))

        while !ptx"mbarrier.test_wait.parity.shared.b64"(bl_ptr, phase)
        end
        ptx"fence.proxy.async.shared::cta"()

        if tid == UInt32(0)
            first_ktile = k_off == Int32(0)
            for kk in 0:3
                da = a_desc0 + UInt64(2 * kk)
                db = b_desc0 + UInt64(2 * kk)
                is_first = first_ktile & (kk == 0)
                ptx"tcgen05.mma.cta_group::1.kind::f16"(tmem, da, db, idesc,
                                                        !is_first)
            end
        end

        k_off += Int32(GGB_BK)
        phase ⊻= UInt32(1)
    end

    if tid == UInt32(0)
        ptx"tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"(
            bm_ptr)
    end
    # bar_mma: single init + single commit → parity 0 (pyptx-faithful).
    while !ptx"mbarrier.try_wait.parity.shared.b64"(bm_ptr, UInt32(0))
    end

    # Epilogue: per-thread TMEM row → BN f32 to C[(m_outer+tid), col_base..].
    row     = m_outer + tid
    d_elem  = UInt64(row) * UInt64(N) + UInt64(col_base)
    d_ptr   = pointer(C) + Int(d_elem) * 4
    tmem_ad = tmem + ((tid << UInt32(16)) & UInt32(0x03E00000))

    out = ntuple(c -> ptx"tcgen05.ld.sync.aligned.32x32b.x1.b32"(
                          tmem_ad + UInt32(c - 1)), Val(GGB_BN))
    ptx"tcgen05.wait::ld.sync.aligned"()

    for vec in 0:(GGB_BN ÷ 4 - 1)
        b = 4 * vec
        ptx"st.global.v4.b32"(d_ptr + vec * 16,
            (out[b + 1], out[b + 2], out[b + 3], out[b + 4]))
    end

    if tid < UInt32(32)
        ptx"tcgen05.dealloc.cta_group::1.sync.aligned.b32"(tmem, UInt32(512))
        ptx"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"()
    end
    return nothing
end

@testset "grouped_gemm (Blackwell) compiles at sm_100a" begin
    types = Tuple{CuDeviceVector{Float32, 1},
                  PTX.TMADescriptorPtr, PTX.TMADescriptorPtr,
                  Int32, Int32, Int32}
    @test ptxas_compiles(_grouped_gemm_bw_kernel!, types;
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_grouped_gemm_bw_kernel!, types;
                   cap = v"10.0", feature_set = :arch)
    @test occursin("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32", ptx)
    @test occursin("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes", ptx)
    @test occursin("mbarrier.test_wait.parity.shared.b64", ptx)
    @test occursin("tcgen05.mma.cta_group::1.kind::f16", ptx)
    @test occursin("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64", ptx)
    @test occursin("tcgen05.ld.sync.aligned.32x32b.x1.b32", ptx)
    @test occursin("st.global.v4.b32", ptx)
    @test occursin("%ctaid.z", ptx)             # Z-grid grouping plumbed
end

# Datacenter-Blackwell only [10.0, 11.0); see tcgen05_smoke.jl rationale.
# (No hardware currently available — kept ready + faithful.)
if v"10.0" <= DEV_CAP < v"11.0"
    function _ggb_cpu_ref(A3::Array{Float32, 3}, B3::Array{Float32, 3})
        G, M, K = size(A3); _, _, N = size(B3)
        Ab = bf16_to_f32.(bf16_bits.(A3))
        Bb = bf16_to_f32.(bf16_bits.(B3))
        C = zeros(Float32, G, M, N)
        for g in 1:G, m in 1:M, n in 1:N, k in 1:K
            @inbounds C[g, m, n] += Ab[g, m, k] * Bb[g, k, n]
        end
        return C
    end

    function _ggb_pack_A(A3)         # (G,M,K) → (K, G*M) col-major, K-fast
        G, M, K = size(A3)
        out = Array{UInt16}(undef, K, G * M)
        for g in 1:G, m in 1:M, k in 1:K
            out[k, (g - 1) * M + m] = bf16_bits(A3[g, m, k])
        end
        out
    end
    function _ggb_pack_B(B3)         # (G,K,N) → (K, G*N) col-major, K-fast
        G, K, N = size(B3)
        out = Array{UInt16}(undef, K, G * N)
        for g in 1:G, k in 1:K, n in 1:N
            out[k, (g - 1) * N + n] = bf16_bits(B3[g, k, n])
        end
        out
    end

    function _run_ggb(G, M, N, K; rtol = 5e-2, atol = 5e-2)
        @assert M % GGB_BM == 0 && N == GGB_BN && K % GGB_BK == 0
        rng = MersenneTwister(G * 1009 + M * 997 + N * 17 + K)
        A3 = Float32.(randn(rng, G, M, K)) .* 0.1f0
        B3 = Float32.(randn(rng, G, K, N)) .* 0.1f0
        A_d = CuArray(_ggb_pack_A(A3))
        B_d = CuArray(_ggb_pack_B(B3))

        tmap_A = tensor_map_tile_2d(:bf16, pointer(A_d), G * M, K, GGB_BM, GGB_BK;
                                    swizzle = :B128)
        tmap_B = tensor_map_tile_2d(:bf16, pointer(B_d), G * N, K, GGB_BN, GGB_BK;
                                    swizzle = :B128)
        a_const = upload_tma_descriptor(tmap_A)
        b_const = upload_tma_descriptor(tmap_B)

        C_d = CUDACore.zeros(Float32, G * M * N)
        grid = (N ÷ GGB_BN, M ÷ GGB_BM, G)
        @cuda threads=GGB_THREADS blocks=grid feature_set=:arch _grouped_gemm_bw_kernel!(
            C_d, a_const.ptr, b_const.ptr, Int32(M), Int32(N), Int32(K))
        CUDACore.synchronize()

        C_packed = reshape(Array(C_d), N, G * M)   # (G*M,N) row-major → (N,G*M)
        C_got = Array{Float32}(undef, G, M, N)
        for g in 1:G, m in 1:M, n in 1:N
            C_got[g, m, n] = C_packed[n, (g - 1) * M + m]
        end
        ≈(C_got, _ggb_cpu_ref(A3, B3); rtol = rtol, atol = atol)
    end

    @testset "grouped GEMM (BW) G=$G M=$M N=$N K=$K" for (G, M, N, K) in [
            (1, 128, 64, 64),
            (2, 128, 64, 64),
            (4, 128, 64, 128),     # K=128 → 2 ktiles (phase-XOR + accumulate)
        ]
        @test _run_ggb(G, M, N, K)
    end
end
