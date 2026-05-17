# Production Blackwell tcgen05 GEMM — algorithmic port of pyptx
# examples/blackwell/gemm_highperf_blackwell.py `build_gemm` (the
# warp-specialized 1-SM core; `build_gemm_persistent` just wraps a
# tile-scheduler around this same body — the next thin layer).
#
# Structure (faithful to pyptx build_gemm):
#   * Warp-specialized: warp 0 lane 0 (tid 0) issues the TMA loads; warp
#     1 lane 0 (tid 32) dispatches tcgen05.mma; all 128 threads run the
#     TMEM→regs→st.global.v4 epilogue.
#   * STAGES-deep SMEM ring: bar_load[s] (producer→MMA, "tile ready"),
#     bar_consumed[s] (MMA→producer, "slot free"). Single bar_mma commit
#     after the last K-tile.
#   * Accumulate predicate false only on the very first MMA (ki==0 &&
#     kk==0); the whole K accumulates into one TMEM tile.
#
# Every mbarrier parity is copied verbatim from pyptx (memory
# blackwell-parity-faithful): producer consumed-wait
# ((ki÷S - 1)&1), MMA load-wait ((ki÷S)&1), epilogue bar_mma 0. The
# suspending `try_wait.parity` form is the pyptx idiom →
# `MBarriers.barrier_try_wait`.
#
# v0 scope vs pyptx build_gemm: the serpentine grid rasterization
# (`grid_minor_dim` / `grid_tile_width` / group serpentine) is a pure
# L2-locality perf optimization with zero effect on numerics — skipped
# here (simple tile_m=ctaid.y, tile_n=ctaid.x), parked as a follow-up
# (see memory gb10_l2_swizzle). Persistent scheduling is the other
# deferred layer. Both are perf, not correctness.
#
# SMEM ≈ 192 KiB (STAGES=4) ≫ 48 KiB → dynamic SMEM + the explicit
# opt-in launch (FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES + shmem=),
# the idiom pyptx ports drop (memory dynamic_smem_optin); modelled on
# gemm_highperf_hopper.jl.

using PTX: smem_addr_u32, tcgen05_descriptor, tcgen05_instr_desc_f16bf16_f32,
           BlackwellLayout, tensor_map_tile_2d, tmem_lane_addr,
           tmem_warp_band_lane
using PTX.MBarriers: BarrierArray, barrier_init, barrier_arrive,
                     barrier_arrive_expect_tx, barrier_try_wait
using CUDACore
using Random

const GHB_BM      = 128
const GHB_BN      = 256
const GHB_BK      = 64                       # K-tile; 4 × k16 tcgen05.mma
const GHB_KPI     = 16
const GHB_MMAS    = GHB_BK ÷ GHB_KPI         # 4
const GHB_STAGES  = 4                        # SMEM ring depth (pyptx STAGES)
const GHB_THREADS = 128
const GHB_A_STAGE = GHB_BM * GHB_BK * 2      # 16384 B
const GHB_B_STAGE = GHB_BN * GHB_BK * 2      # 32768 B
const GHB_STRIDE  = GHB_BK * 16              # 1024 (tcgen05 desc stride)

# Dynamic-SMEM byte offsets (mirror pyptx SMEM_* layout).
const GHB_OFF_A   = 0
const GHB_OFF_B   = GHB_OFF_A + GHB_STAGES * GHB_A_STAGE
const GHB_OFF_BL  = GHB_OFF_B + GHB_STAGES * GHB_B_STAGE
const GHB_OFF_BC  = GHB_OFF_BL + GHB_STAGES * 8
const GHB_OFF_BM  = GHB_OFF_BC + GHB_STAGES * 8
const GHB_OFF_TS  = GHB_OFF_BM + 8
const GHB_SMEM    = GHB_OFF_TS + 16          # ≈ 196696 B

function _ghb_gemm_kernel!(
        D::CuDeviceVector{Float32, 1},
        tma_A::PTX.TMADescriptorPtr,
        tma_B::PTX.TMADescriptorPtr,
        M::Int32, N::Int32, K::Int32)

    smem_A    = CuDynamicSharedArray(UInt16, GHB_STAGES * GHB_BM * GHB_BK, GHB_OFF_A)
    smem_B    = CuDynamicSharedArray(UInt16, GHB_STAGES * GHB_BN * GHB_BK, GHB_OFF_B)
    bar_load  = CuDynamicSharedArray(UInt64, GHB_STAGES, GHB_OFF_BL)
    bar_cons  = CuDynamicSharedArray(UInt64, GHB_STAGES, GHB_OFF_BC)
    bar_mma   = CuDynamicSharedArray(UInt64, 1, GHB_OFF_BM)
    tmem_slot = CuDynamicSharedArray(UInt32, 1, GHB_OFF_TS)

    a_base    = pointer(smem_A)
    b_base    = pointer(smem_B)
    bl        = BarrierArray{GHB_STAGES}(pointer(bar_load))
    bc        = BarrierArray{GHB_STAGES}(pointer(bar_cons))
    bm_ptr    = pointer(bar_mma)
    bm_addr   = smem_addr_u32(bm_ptr)
    slot_addr = smem_addr_u32(pointer(tmem_slot))

    tid    = ptx"mov.u32"(sreg"tid.x")
    tile_n = ptx"mov.u32"(sreg"ctaid.x")     # v0: no serpentine swizzle
    tile_m = ptx"mov.u32"(sreg"ctaid.y")
    m_base = tile_m << UInt32(7)             # * BM (128)
    n_base = tile_n << UInt32(8)             # * BN (256)

    is_tma = tid == UInt32(0)
    is_mma = tid == UInt32(32)

    if tid == UInt32(0)
        for s in 0:(GHB_STAGES - 1)
            barrier_init(bl[s], 1)
            barrier_init(bc[s], 1)
        end
        barrier_init(bm_ptr, 1)
        ptx"fence.proxy.async.shared::cta"()
    end
    if tid < UInt32(32)
        ptx"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"(
            slot_addr, UInt32(512))
    end
    ptx"bar.sync"(Val(0))

    tmem  = @inbounds tmem_slot[1]
    idesc = tcgen05_instr_desc_f16bf16_f32(; m = GHB_BM, n = GHB_BN,
                                           ab_dtype = :bf16)
    kiters = UInt32(K ÷ Int32(GHB_BK))

    # ── Producer (tid 0): ring-buffered TMA loads ──
    if is_tma
        ki = UInt32(0)
        while ki < kiters
            slot   = ki & UInt32(GHB_STAGES - 1)
            a_ptr  = a_base + Int(slot) * GHB_A_STAGE
            b_ptr  = b_base + Int(slot) * GHB_B_STAGE
            mbar_l = bl[slot]
            if ki >= UInt32(GHB_STAGES)
                # slot reuse: wait MMA done with the previous occupant.
                barrier_try_wait(bc[slot], ((ki >> 2) - UInt32(1)) & UInt32(1))
            end
            barrier_arrive_expect_tx(mbar_l, GHB_A_STAGE + GHB_B_STAGE)
            kcoord = reinterpret(Int32, ki * UInt32(GHB_BK))
            ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                a_ptr, tma_A, kcoord, reinterpret(Int32, m_base), mbar_l)
            ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                b_ptr, tma_B, kcoord, reinterpret(Int32, n_base), mbar_l)
            ki += UInt32(1)
        end
    end

    # ── MMA dispatcher (tid 32): wait loads, issue accumulating MMAs ──
    if is_mma
        ki = UInt32(0)
        while ki < kiters
            slot = ki & UInt32(GHB_STAGES - 1)
            barrier_try_wait(bl[slot], (ki >> 2) & UInt32(1))
            a_addr = smem_addr_u32(a_base) + UInt32(slot) * UInt32(GHB_A_STAGE)
            b_addr = smem_addr_u32(b_base) + UInt32(slot) * UInt32(GHB_B_STAGE)
            da0 = tcgen05_descriptor(a_addr; leading_bytes = 16,
                                     stride_bytes = GHB_STRIDE,
                                     swizzle = BlackwellLayout.B128)
            db0 = tcgen05_descriptor(b_addr; leading_bytes = 16,
                                     stride_bytes = GHB_STRIDE,
                                     swizzle = BlackwellLayout.B128)
            for kk in 0:(GHB_MMAS - 1)
                da = da0 + UInt64(2 * kk)
                db = db0 + UInt64(2 * kk)
                is_first = (ki == UInt32(0)) & (kk == 0)
                ptx"tcgen05.mma.cta_group::1.kind::f16"(tmem, da, db, idesc,
                                                        !is_first)
            end
            barrier_arrive(bc[slot])
            ki += UInt32(1)
        end
        ptx"tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"(
            bm_addr)
    end

    # ── Epilogue (all 128 threads): wait MMA, TMEM → gmem ──
    barrier_try_wait(bm_ptr, UInt32(0))

    row    = m_base + tid
    d_elem = UInt64(row) * UInt64(N) + UInt64(n_base)
    d_ptr  = pointer(D) + Int(d_elem) * 4
    tmem_a = tmem_lane_addr(tmem, tmem_warp_band_lane(tid))

    for chunk in 0:(GHB_BN ÷ 128 - 1)
        coff = chunk * 128
        out  = ptx"tcgen05.ld.sync.aligned.32x32b.x128.b32"(
                   tmem_a + UInt32(coff))
        ptx"tcgen05.wait::ld.sync.aligned"()
        for vec in 0:(128 ÷ 4 - 1)
            b = 4 * vec
            ptx"st.global.v4.b32"(d_ptr + (coff + b) * 4,
                (out[b + 1], out[b + 2], out[b + 3], out[b + 4]))
        end
    end

    if tid < UInt32(32)
        ptx"tcgen05.dealloc.cta_group::1.sync.aligned.b32"(tmem, UInt32(512))
        ptx"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"()
    end
    return nothing
end

@testset "gemm_highperf_blackwell compiles at sm_100a" begin
    types = Tuple{CuDeviceVector{Float32, 1},
                  PTX.TMADescriptorPtr, PTX.TMADescriptorPtr,
                  Int32, Int32, Int32}
    @test ptxas_compiles(_ghb_gemm_kernel!, types;
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_ghb_gemm_kernel!, types;
                   cap = v"10.0", feature_set = :arch)
    @test occursin("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32", ptx)
    @test occursin("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes", ptx)
    @test occursin("mbarrier.arrive.expect_tx.shared.b64", ptx)
    @test occursin("mbarrier.try_wait.parity.shared.b64", ptx)
    @test occursin("tcgen05.mma.cta_group::1.kind::f16", ptx)
    @test occursin("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64", ptx)
    @test occursin("tcgen05.ld.sync.aligned.32x32b.x128.b32", ptx)
    @test occursin("st.global.v4.b32", ptx)
end

# Datacenter-Blackwell only [10.0, 11.0); see tcgen05_smoke.jl rationale.
if v"10.0" <= DEV_CAP < v"11.0"
    function _ghb_cpu_ref(A::Matrix{Float32}, B::Matrix{Float32})
        Ab = bf16_to_f32.(bf16_bits.(A))
        Bb = bf16_to_f32.(bf16_bits.(B))
        Ab * Bb
    end

    function _run_ghb(M, N, K; rtol = 5e-2, atol = 5e-2)
        @assert M % GHB_BM == 0 && N % GHB_BN == 0 && K % GHB_BK == 0
        rng = MersenneTwister(M * 9176 + N * 131 + K)
        A = Float32.(randn(rng, M, K)) .* 0.1f0      # (M,K) row-major logical
        B = Float32.(randn(rng, K, N)) .* 0.1f0      # (K,N)

        # A tile (M,K) K-major; B_T is (N,K) K-major (B transposed), both
        # bf16, K-fast — identical packing to grouped_gemm.
        A_pk = Array{UInt16}(undef, K, M)
        for m in 1:M, k in 1:K
            @inbounds A_pk[k, m] = bf16_bits(A[m, k])
        end
        B_pk = Array{UInt16}(undef, K, N)
        for k in 1:K, n in 1:N
            @inbounds B_pk[k, n] = bf16_bits(B[k, n])
        end
        A_d = CuArray(A_pk)
        B_d = CuArray(B_pk)

        tmap_A = tensor_map_tile_2d(:bf16, pointer(A_d), M, K, GHB_BM, GHB_BK;
                                    swizzle = :B128)
        tmap_B = tensor_map_tile_2d(:bf16, pointer(B_d), N, K, GHB_BN, GHB_BK;
                                    swizzle = :B128)
        a_const = upload_tma_descriptor(tmap_A)
        b_const = upload_tma_descriptor(tmap_B)

        D_d = CUDACore.zeros(Float32, M * N)
        grid = (N ÷ GHB_BN, M ÷ GHB_BM, 1)

        kern = @cuda launch=false feature_set=:arch _ghb_gemm_kernel!(
            D_d, a_const.ptr, b_const.ptr, Int32(M), Int32(N), Int32(K))
        attrs = CUDACore.attributes(kern.fun)
        attrs[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = GHB_SMEM
        kern(D_d, a_const.ptr, b_const.ptr, Int32(M), Int32(N), Int32(K);
             blocks = grid, threads = GHB_THREADS, shmem = GHB_SMEM)
        CUDACore.synchronize()

        D_got = permutedims(reshape(Array(D_d), N, M))   # (M,N) row-major
        ≈(D_got, _ghb_cpu_ref(A, B); rtol = rtol, atol = atol)
    end

    @testset "gemm_highperf_blackwell M=$M N=$N K=$K" for (M, N, K) in [
            (128, 256, 64),       # 1 tile, 1 K-iter (< STAGES, no reuse)
            (128, 256, 512),      # 1 tile, 8 K-iters (ring slot reuse)
            (256, 512, 256),      # 2×2 tiles, 4 K-iters
        ]
        @test _run_ghb(M, N, K)
    end
end


# ── Persistent variant — pyptx `build_gemm_persistent` ───────────────────
#
# The proven `build_gemm` body wrapped in a persistent tile-scheduler:
# grid is (num_ctas,1,1); each CTA strides `tile_linear = cta_x +
# iter*num_ctas` over the (M/BM)×(N/BN) tile grid until it runs past
# total_tiles. TMEM is alloc'd ONCE and reused for every tile (the
# accumulator is re-zeroed per tile via the ki==0&&kk==0 predicate).
#
# Faithful to pyptx EXCEPT the thread count: pyptx declares
# `block=(256,1,1)` but its epilogue does `row = m_base + tid`
# unguarded — tid≥128 writes outside the BM=128 tile (OOB). The
# warp-specialized pipeline only uses tid 0 / 32, so 256 buys nothing
# for correctness; we keep the **B200-proven 128-thread block** from
# `build_gemm`. Everything else is verbatim, including the non-obvious
# per-tile mbarrier re-init + the `slot_use_index` parity scheme
# (`supt = 1 + (k_iters-1-slot)÷STAGES`, `sui = iter*supt + ki÷STAGES`)
# and the `tcgen05.fence::after_thread_sync` after the consumed-wait.
# That init-vs-phase interaction is the suspect area if a B200 run
# hangs (memory blackwell-parity-faithful: replicate, let hardware
# judge — do NOT pre-"fix" it; that is the deviation that bit
# gemm_experimental). v0 also skips the serpentine rasterization.

function _ghb_persistent_kernel!(
        D::CuDeviceVector{Float32, 1},
        tma_A::PTX.TMADescriptorPtr,
        tma_B::PTX.TMADescriptorPtr,
        M::Int32, N::Int32, K::Int32)

    smem_A    = CuDynamicSharedArray(UInt16, GHB_STAGES * GHB_BM * GHB_BK, GHB_OFF_A)
    smem_B    = CuDynamicSharedArray(UInt16, GHB_STAGES * GHB_BN * GHB_BK, GHB_OFF_B)
    bar_load  = CuDynamicSharedArray(UInt64, GHB_STAGES, GHB_OFF_BL)
    bar_cons  = CuDynamicSharedArray(UInt64, GHB_STAGES, GHB_OFF_BC)
    bar_mma   = CuDynamicSharedArray(UInt64, 1, GHB_OFF_BM)
    tmem_slot = CuDynamicSharedArray(UInt32, 1, GHB_OFF_TS)

    a_base    = pointer(smem_A)
    b_base    = pointer(smem_B)
    bl        = BarrierArray{GHB_STAGES}(pointer(bar_load))
    bc        = BarrierArray{GHB_STAGES}(pointer(bar_cons))
    bm_ptr    = pointer(bar_mma)
    bm_addr   = smem_addr_u32(bm_ptr)
    slot_addr = smem_addr_u32(pointer(tmem_slot))

    tid    = ptx"mov.u32"(sreg"tid.x")
    cta_x  = ptx"mov.u32"(sreg"ctaid.x")
    nctas  = ptx"mov.u32"(sreg"nctaid.x")
    is_tma = tid == UInt32(0)
    is_mma = tid == UInt32(32)

    # TMEM allocated ONCE, reused across every tile this CTA processes.
    if tid < UInt32(32)
        ptx"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"(
            slot_addr, UInt32(512))
    end
    ptx"bar.sync"(Val(0))
    tmem  = @inbounds tmem_slot[1]
    idesc = tcgen05_instr_desc_f16bf16_f32(; m = GHB_BM, n = GHB_BN,
                                           ab_dtype = :bf16)

    tile_n_iters = UInt32(N ÷ Int32(GHB_BN))
    total_tiles  = UInt32(M ÷ Int32(GHB_BM)) * tile_n_iters
    kiters       = UInt32(K ÷ Int32(GHB_BK))

    tile_iter = UInt32(0)
    while true
        tile_linear = cta_x + tile_iter * nctas
        tile_linear >= total_tiles && break

        tile_m = tile_linear ÷ tile_n_iters         # v0: no serpentine
        tile_n = tile_linear % tile_n_iters
        m_base = tile_m << UInt32(7)
        n_base = tile_n << UInt32(8)

        if tid == UInt32(0)
            for s in 0:(GHB_STAGES - 1)
                barrier_init(bl[s], 1)
                barrier_init(bc[s], 1)
            end
            barrier_init(bm_ptr, 1)
            ptx"fence.proxy.async.shared::cta"()
        end
        ptx"bar.sync"(Val(0))

        if is_tma
            ki = UInt32(0)
            while ki < kiters
                slot = ki & UInt32(GHB_STAGES - 1)
                supt = UInt32(1) + ((kiters - UInt32(1) - slot) >> 2)
                sui  = tile_iter * supt + (ki >> 2)
                a_ptr  = a_base + Int(slot) * GHB_A_STAGE
                b_ptr  = b_base + Int(slot) * GHB_B_STAGE
                mbar_l = bl[slot]
                if sui > UInt32(0)
                    barrier_try_wait(bc[slot], (sui - UInt32(1)) & UInt32(1))
                    ptx"tcgen05.fence::after_thread_sync"()
                end
                barrier_arrive_expect_tx(mbar_l, GHB_A_STAGE + GHB_B_STAGE)
                kcoord = reinterpret(Int32, ki * UInt32(GHB_BK))
                ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                    a_ptr, tma_A, kcoord, reinterpret(Int32, m_base), mbar_l)
                ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                    b_ptr, tma_B, kcoord, reinterpret(Int32, n_base), mbar_l)
                ki += UInt32(1)
            end
        end

        if is_mma
            ki = UInt32(0)
            while ki < kiters
                slot = ki & UInt32(GHB_STAGES - 1)
                supt = UInt32(1) + ((kiters - UInt32(1) - slot) >> 2)
                sui  = tile_iter * supt + (ki >> 2)
                barrier_try_wait(bl[slot], sui & UInt32(1))
                a_addr = smem_addr_u32(a_base) + UInt32(slot) * UInt32(GHB_A_STAGE)
                b_addr = smem_addr_u32(b_base) + UInt32(slot) * UInt32(GHB_B_STAGE)
                da0 = tcgen05_descriptor(a_addr; leading_bytes = 16,
                                         stride_bytes = GHB_STRIDE,
                                         swizzle = BlackwellLayout.B128)
                db0 = tcgen05_descriptor(b_addr; leading_bytes = 16,
                                         stride_bytes = GHB_STRIDE,
                                         swizzle = BlackwellLayout.B128)
                for kk in 0:(GHB_MMAS - 1)
                    da = da0 + UInt64(2 * kk)
                    db = db0 + UInt64(2 * kk)
                    is_first = (ki == UInt32(0)) & (kk == 0)
                    ptx"tcgen05.mma.cta_group::1.kind::f16"(tmem, da, db, idesc,
                                                            !is_first)
                end
                barrier_arrive(bc[slot])
                ki += UInt32(1)
            end
            ptx"tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"(
                bm_addr)
        end

        barrier_try_wait(bm_ptr, UInt32(0))

        row    = m_base + tid
        d_elem = UInt64(row) * UInt64(N) + UInt64(n_base)
        d_ptr  = pointer(D) + Int(d_elem) * 4
        tmem_a = tmem_lane_addr(tmem, tmem_warp_band_lane(tid))
        for chunk in 0:(GHB_BN ÷ 128 - 1)
            coff = chunk * 128
            out  = ptx"tcgen05.ld.sync.aligned.32x32b.x128.b32"(
                       tmem_a + UInt32(coff))
            ptx"tcgen05.wait::ld.sync.aligned"()
            for vec in 0:(128 ÷ 4 - 1)
                b = 4 * vec
                ptx"st.global.v4.b32"(d_ptr + (coff + b) * 4,
                    (out[b + 1], out[b + 2], out[b + 3], out[b + 4]))
            end
        end
        ptx"bar.sync"(Val(0))
        tile_iter += UInt32(1)
    end

    if tid < UInt32(32)
        ptx"tcgen05.dealloc.cta_group::1.sync.aligned.b32"(tmem, UInt32(512))
        ptx"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"()
    end
    return nothing
end

@testset "gemm_highperf_blackwell persistent compiles at sm_100a" begin
    types = Tuple{CuDeviceVector{Float32, 1},
                  PTX.TMADescriptorPtr, PTX.TMADescriptorPtr,
                  Int32, Int32, Int32}
    @test ptxas_compiles(_ghb_persistent_kernel!, types;
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_ghb_persistent_kernel!, types;
                   cap = v"10.0", feature_set = :arch)
    @test occursin("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32", ptx)
    @test occursin("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes", ptx)
    @test occursin("tcgen05.fence::after_thread_sync", ptx)
    @test occursin("mbarrier.try_wait.parity.shared.b64", ptx)
    @test occursin("tcgen05.mma.cta_group::1.kind::f16", ptx)
    @test occursin("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64", ptx)
    @test occursin("tcgen05.ld.sync.aligned.32x32b.x128.b32", ptx)
    @test occursin("%nctaid.x", ptx)             # persistent grid stride
end

# Datacenter-Blackwell only [10.0, 11.0). Reuses `_ghb_cpu_ref` from the
# core block above (same gate condition → defined when this runs).
if v"10.0" <= DEV_CAP < v"11.0"
    function _run_ghb_persistent(M, N, K, num_ctas; rtol = 5e-2, atol = 5e-2)
        @assert M % GHB_BM == 0 && N % GHB_BN == 0 && K % GHB_BK == 0
        rng = MersenneTwister(M * 5557 + N * 89 + K * 7 + num_ctas)
        A = Float32.(randn(rng, M, K)) .* 0.1f0
        B = Float32.(randn(rng, K, N)) .* 0.1f0
        A_pk = Array{UInt16}(undef, K, M)
        for m in 1:M, k in 1:K
            @inbounds A_pk[k, m] = bf16_bits(A[m, k])
        end
        B_pk = Array{UInt16}(undef, K, N)
        for k in 1:K, n in 1:N
            @inbounds B_pk[k, n] = bf16_bits(B[k, n])
        end
        A_d = CuArray(A_pk)
        B_d = CuArray(B_pk)

        tmap_A = tensor_map_tile_2d(:bf16, pointer(A_d), M, K, GHB_BM, GHB_BK;
                                    swizzle = :B128)
        tmap_B = tensor_map_tile_2d(:bf16, pointer(B_d), N, K, GHB_BN, GHB_BK;
                                    swizzle = :B128)
        a_const = upload_tma_descriptor(tmap_A)
        b_const = upload_tma_descriptor(tmap_B)

        D_d = CUDACore.zeros(Float32, M * N)
        kern = @cuda launch=false feature_set=:arch _ghb_persistent_kernel!(
            D_d, a_const.ptr, b_const.ptr, Int32(M), Int32(N), Int32(K))
        attrs = CUDACore.attributes(kern.fun)
        attrs[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = GHB_SMEM
        kern(D_d, a_const.ptr, b_const.ptr, Int32(M), Int32(N), Int32(K);
             blocks = (num_ctas, 1, 1), threads = GHB_THREADS, shmem = GHB_SMEM)
        CUDACore.synchronize()

        D_got = permutedims(reshape(Array(D_d), N, M))
        ≈(D_got, _ghb_cpu_ref(A, B); rtol = rtol, atol = atol)
    end

    # num_ctas < total_tiles forces multiple tiles/CTA → exercises the
    # persistent loop, per-tile re-init, and the slot_use_index parity.
    @testset "ghb persistent M=$M N=$N K=$K ctas=$C" for (M, N, K, C) in [
            (256, 512, 256, 1),   # 4 tiles, 1 CTA → 4 tiles serially
            (256, 512, 128, 2),   # 4 tiles, 2 CTAs → 2 tiles each
            (128, 256, 512, 1),   # 1 tile, 8 K-iters (persistent body)
        ]
        @test _run_ghb_persistent(M, N, K, C)
    end
end
