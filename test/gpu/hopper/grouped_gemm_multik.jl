# TEST_TARGET: requires=toolkit evidence=mixed target=sm_90a
# Multi-K grouped GEMM (Hopper) — pyptx's actual perf-tuned config.
#
# Same MoE-shape algorithm as grouped_gemm.jl, but each K-iter loads
# `tile_k = 64` K-elements per operand in ONE TMA, then issues
# `wgmma_k_iters = 4` wgmma.mma_async sub-calls within the iter, each
# advancing the SMEM descriptor by `kk * 32` bytes (16 K-elements × 2B).
# This amortizes mbarrier + TMA overhead across 4× the K-work.
#
# Exercises the descriptor-byte-offset trick: the wgmma_descriptor is a
# UInt64 with the SMEM address in bits [13:0] (post-`>>4`), so advancing
# by N bytes (N multiple of 16) is just `desc + UInt64(N >> 4)`.
#
# A K-fast (K, G*M) col-major, B K-fast (K, G*N) col-major, C N-fast (N, G*M)
# col-major — identical layout convention to grouped_gemm.jl.

using PTX: layout_for_a, wgmma_descriptor, smem_addr_u32,
           tensor_map_tile_2d, CuTensorMap
using CUDACore
using Random

const GGM_BM           = 64
const GGM_BN           = 8
const GGM_TILE_K       = 64       # bytes loaded per TMA = K-elements per iter
const GGM_WGMMA_K      = 16       # wgmma m64n8k16 hard-coded k
const GGM_WGMMA_KITERS = GGM_TILE_K ÷ GGM_WGMMA_K     # 4 sub-calls per iter
const GGM_THREADS      = 128

# Per-iter TMA bytes: full A and B tiles.
const GGM_LOAD_BYTES = GGM_BM * GGM_TILE_K * 2 + GGM_TILE_K * GGM_BN * 2

# K-stride between wgmma sub-calls, in 16-byte u128 units (= descriptor low-bits delta).
# 16 K-elements × 2 B/elem = 32 B = 2 u128.
const GGM_KSTRIDE_U128 = UInt64((GGM_WGMMA_K * 2) >> 4)

function _grouped_gemm_multik_kernel!(
        C::CuDeviceVector{Float32, 1},
        tma_A::PTX.TMADescriptorPtr,
        tma_B::PTX.TMADescriptorPtr,
        M::Int32, N::Int32, K::Int32)

    smem_A = CuStaticSharedArray(UInt16, GGM_BM * GGM_TILE_K)        # 64 × 64 bf16 = 8192 B
    smem_B = CuStaticSharedArray(UInt16, GGM_TILE_K * GGM_BN)        # 64 × 8  bf16 = 1024 B
    mbar   = CuStaticSharedArray(UInt64, 1)

    a_ptr  = pointer(smem_A)
    b_ptr  = pointer(smem_B)
    mb_ptr = pointer(mbar)
    a_addr = smem_addr_u32(a_ptr)
    b_addr = smem_addr_u32(b_ptr)

    tid   = ptx"mov.u32"(sreg"tid.x")
    cta_x = ptx"mov.u32"(sreg"ctaid.x")
    cta_y = ptx"mov.u32"(sreg"ctaid.y")
    cta_z = ptx"mov.u32"(sreg"ctaid.z")

    m_outer = cta_y * UInt32(GGM_BM) + cta_z * UInt32(M)
    n_base  = cta_x * UInt32(GGM_BN)
    n_outer = n_base + cta_z * UInt32(N)

    # tile_k = 64 → row_bytes = 64 × 2 = 128 → B128 swizzle for both A and B.
    # LBO = 16 (single-stripe K-major), SBO = 8 × 128 = 1024.
    la = layout_for_a(dtype = :bf16, m = GGM_BM, k = GGM_TILE_K)
    lb = layout_for_a(dtype = :bf16, m = GGM_BN, k = GGM_TILE_K)
    a_desc_base = wgmma_descriptor(a_addr;
        leading_byte_offset = la.leading_byte_offset,
        stride_byte_offset  = la.stride_byte_offset,
        swizzle             = la.layout_type)
    b_desc_base = wgmma_descriptor(b_addr;
        leading_byte_offset = lb.leading_byte_offset,
        stride_byte_offset  = lb.stride_byte_offset,
        swizzle             = lb.layout_type)

    d = ntuple(_ -> 0f0, Val(4))

    # mbarrier init ONCE outside the loop — re-initializing every iter resets
    # the phase indicator to 0 and breaks the `phase ^= 1` parity tracking
    # whenever K > tile_k (i.e., the loop runs more than once).
    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mb_ptr, UInt32(1))
    end
    ptx"bar.sync"(Val(0))

    phase = UInt32(0)
    k_off = Int32(0)
    while k_off < K
        if tid == UInt32(0)
            ptx"fence.proxy.async.shared::cta"()
            ptx"mbarrier.arrive.expect_tx.shared.b64"(mb_ptr, UInt32(GGM_LOAD_BYTES))
            ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                a_ptr, tma_A, k_off, reinterpret(Int32, m_outer), mb_ptr)
            ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                b_ptr, tma_B, k_off, reinterpret(Int32, n_outer), mb_ptr)
        end
        ptx"bar.sync"(Val(0))

        while !ptx"mbarrier.test_wait.parity.shared.b64"(mb_ptr, phase)
        end

        ptx"fence.proxy.async.shared::cta"()
        ptx"wgmma.fence.sync.aligned"()
        # Issue WGMMA_KITERS wgmma sub-calls per K-tile, each advancing the
        # SMEM address by 32 bytes (= 16 K-elements) in K-fast layout. The
        # loop is over a compile-time const range so LLVM unrolls it. We
        # always accumulate (scale_d=true) — mathematically equivalent to
        # pyptx's "scale=false on first, true after" given d is zero-init.
        @inbounds for kk in 0:(GGM_WGMMA_KITERS - 1)
            offset = UInt64(kk) * GGM_KSTRIDE_U128
            d = ptx"wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16"(
                d, a_desc_base + offset, b_desc_base + offset, true)
        end
        ptx"wgmma.commit_group.sync.aligned"()
        ptx"wgmma.wait_group.sync.aligned"(Val(0))

        k_off += Int32(GGM_TILE_K)
        phase ⊻= UInt32(1)
    end

    # Same epilogue as grouped_gemm.jl: m64n8k16 wgmma f32 output layout.
    wid      = tid >> UInt32(5)
    lane     = tid & UInt32(31)
    frag_row = (wid << UInt32(4)) + (lane >> UInt32(2))
    frag_col = (lane & UInt32(3)) << UInt32(1)
    abs_row  = frag_row + m_outer
    abs_col  = frag_col + n_base

    pc    = pointer(C)
    off_a = (abs_row * UInt32(N) + abs_col) * UInt32(4)
    off_b = ((abs_row + UInt32(8)) * UInt32(N) + abs_col) * UInt32(4)
    ptx"st.global.v2.f32"(pc + Int(off_a), (d[1], d[2]))
    ptx"st.global.v2.f32"(pc + Int(off_b), (d[3], d[4]))
    return nothing
end

# --- Cross-arch ptxas validation -----------------------------------------

@testset "grouped_gemm_multik kernel compiles at sm_90a" begin
    types = Tuple{CuDeviceVector{Float32, 1},
                  PTX.TMADescriptorPtr,
                  PTX.TMADescriptorPtr,
                  Int32, Int32, Int32}
    @test ptxas_compiles(_grouped_gemm_multik_kernel!, types;
                         cap = v"9.0", feature_set = :arch)

    ptx = emit_ptx(_grouped_gemm_multik_kernel!, types;
                   cap = v"9.0", feature_set = :arch)
    @test occursin("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16", ptx)
    # Four wgmma sub-calls per K-iter (loop body unrolled via @generated).
    @test count(==("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16"),
                [m.match for m in eachmatch(
                    r"wgmma\.mma_async\.sync\.aligned\.m64n8k16\.f32\.bf16\.bf16", ptx)]) >= 4
    @test occursin("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes", ptx)
    @test occursin("mbarrier.test_wait.parity.shared.b64", ptx)
end

# --- Runtime — H100 only -------------------------------------------------

if test_runtime_supported(@__FILE__)
    function _multik_cpu_ref(A3::Array{Float32, 3}, B3::Array{Float32, 3})
        G, M, K = size(A3)
        _, _, N = size(B3)
        Ab = bf16_to_f32.(bf16_bits.(A3))
        Bb = bf16_to_f32.(bf16_bits.(B3))
        C = zeros(Float32, G, M, N)
        for g in 1:G, m in 1:M, n in 1:N
            s = 0f0
            for k in 1:K
                s += Ab[g, m, k] * Bb[g, k, n]
            end
            C[g, m, n] = s
        end
        C
    end

    function _multik_pack_A(A3)
        G, M, K = size(A3)
        out = Array{UInt16}(undef, K, G * M)
        for g in 1:G, m in 1:M, k in 1:K
            out[k, (g - 1) * M + m] = bf16_bits(A3[g, m, k])
        end
        out
    end

    function _multik_pack_B(B3)
        G, K, N = size(B3)
        out = Array{UInt16}(undef, K, G * N)
        for g in 1:G, k in 1:K, n in 1:N
            out[k, (g - 1) * N + n] = bf16_bits(B3[g, k, n])
        end
        out
    end

    function _run_grouped_gemm_multik(G::Int, M::Int, N::Int, K::Int;
                                      rtol = 1e-3, atol = 1e-3)
        @assert M % GGM_BM == 0 "M must be divisible by BM=$GGM_BM"
        @assert N % GGM_BN == 0 "N must be divisible by BN=$GGM_BN"
        @assert K % GGM_TILE_K == 0 "K must be divisible by tile_k=$GGM_TILE_K"

        rng = MersenneTwister(G * 1009 + M * 997 + N * 17 + K + 31)
        A3 = Float32.(randn(rng, G, M, K)) .* 0.1f0
        B3 = Float32.(randn(rng, G, K, N)) .* 0.1f0

        A_d = CuArray(_multik_pack_A(A3))
        B_d = CuArray(_multik_pack_B(B3))

        # TMA: tile_k=64 with bf16 → 128B row width → B128 swizzle.
        tmap_A = tensor_map_tile_2d(:bf16, pointer(A_d), G * M, K,
                                    GGM_BM, GGM_TILE_K; swizzle = :B128)
        tmap_B = tensor_map_tile_2d(:bf16, pointer(B_d), G * N, K,
                                    GGM_BN, GGM_TILE_K; swizzle = :B128)

        a_const = upload_tma_descriptor(tmap_A)
        b_const = upload_tma_descriptor(tmap_B)

        C_d = CUDACore.zeros(Float32, G * M * N)
        grid = (N ÷ GGM_BN, M ÷ GGM_BM, G)
        @cuda threads = GGM_THREADS blocks = grid _grouped_gemm_multik_kernel!(
            C_d, a_const.ptr, b_const.ptr, Int32(M), Int32(N), Int32(K))
        CUDACore.synchronize()

        C_packed = reshape(Array(C_d), N, G * M)
        C_got = Array{Float32}(undef, G, M, N)
        for g in 1:G, m in 1:M, n in 1:N
            C_got[g, m, n] = C_packed[n, (g - 1) * M + m]
        end

        C_ref = _multik_cpu_ref(A3, B3)
        ≈(C_got, C_ref; rtol = rtol, atol = atol)
    end

    @testset "grouped GEMM multi-K G=$G M=$M N=$N K=$K" for (G, M, N, K) in [
            (1, 64, 8, 64),
            (2, 64, 8, 64),
            (4, 128, 8, 128),
        ]
        @test _run_grouped_gemm_multik(G, M, N, K)
    end
end
