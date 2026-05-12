# Full warp-specialized Hopper GEMM. Port of
# pyptx/examples/hopper/gemm_highperf_hopper.py. Composes every Hopper
# brick validated in this directory + adds the three remaining pieces:
#
#   - Hilbert tile scheduler (host helper + device SMEM-resident fetch)
#   - Persistent kernel pattern (cluster iterates tiles until sentinel)
#   - Tile-K inner loop (WGMMA_KITERS wgmma sub-calls per K-tile load)
#
# This commit takes BN=128 / BK_TILE=64 / N_STAGES=2 (vs pyptx's
# 256/64/3) — half the column width and one less pipeline stage, but
# the same wgmma shape family and the same K-loop structure. Trading
# one octave of perf for an order of magnitude less generated PTX
# in the epilogue; bumping to BN=256 / N_STAGES=3 is mechanical
# (change two consts + scale the @nexprs unroll factor).
#
# Configuration:
#   BM_CTA = 128, BN = 128, BK_TILE = 64, WGMMA_K = 16, N_STAGES = 2
#   CLUSTERS = 2 (cluster_m=2, cluster_n=1)
#   384 threads per CTA = 1 producer + 2 consumer warpgroups
#   m64n128k16 wgmma — 64 f32 frags per lane
#   WGMMA_KITERS = 4 wgmma sub-calls per K-tile load
#   Dynamic SMEM ~97 KB per CTA
#
# Hilbert schedule: each cluster receives a contiguous segment of
# `SPACE_LEN` u32 tile-indices `(m_tile << 16) | n_tile`. The kernel
# loads its segment into SMEM at startup, then each consumer/producer
# iterates `tile_iter = 0, 1, ...` reading the SMEM segment until it
# sees the sentinel `0xffffffff` (== -1 cast to u32).

using PTX: layout_for_a, wgmma_descriptor, smem_addr_u32, tensor_map_tile_2d,
           bf16x2_pack, step_desc
using PTX.MBarriers: BarrierArray, Pipeline, pipeline_init!, pipeline_cursor,
                     barrier_wait, barrier_arrive_expect_tx, barrier_arrive_cluster
using CUDACore
using CUDACore: cluster_arrive, cluster_wait
using Random

# ── Shape constants ────────────────────────────────────────────────────
const GHH_BM_CTA       = 128
const GHH_BN           = 128
const GHH_BK_TILE      = 64        # K-elements loaded per stage
const GHH_WGMMA_K      = 16
const GHH_WGMMA_KITERS = GHH_BK_TILE ÷ GHH_WGMMA_K         # 4
const GHH_N_STAGES     = 2
const GHH_NUM_CONSUMERS = 2
const GHH_CLUSTERS     = 2
const GHH_B_WG_M       = GHH_BM_CTA ÷ GHH_NUM_CONSUMERS    # 64
const GHH_THREADS      = 128 * (1 + GHH_NUM_CONSUMERS)     # 384
const GHH_BM_CLUSTER   = GHH_BM_CTA * GHH_CLUSTERS         # 256

# Hilbert schedule
const GHH_SPACE_LEN    = 128                                # max tiles/cluster
const GHH_SENTINEL     = UInt32(0xffffffff)                # marks end of cluster's list

# Per-stage SMEM bytes
const GHH_A_STAGE_BYTES = GHH_BM_CTA * GHH_BK_TILE * 2      # 16384
const GHH_B_STAGE_BYTES = GHH_BK_TILE * GHH_BN * 2          # 16384
const GHH_C_BYTES       = GHH_BM_CTA * GHH_BN * 2           # 32768
const GHH_LOAD_BYTES    = GHH_A_STAGE_BYTES + GHH_B_STAGE_BYTES  # 32768 per CTA
const GHH_K_STEP_BYTES  = GHH_WGMMA_K * 2                   # 32 (16 K-elem × 2 B/elem)

# Per-consumer A-offset within stage's A tile
const GHH_A_CONSUMER_BYTES = GHH_B_WG_M * GHH_BK_TILE * 2   # 8192
const GHH_EMPTY_COUNT = GHH_NUM_CONSUMERS * GHH_CLUSTERS    # 4

# Dynamic SMEM layout (byte offsets into the kernel's shmem block).
const GHH_OFF_A      = 0
const GHH_OFF_B      = GHH_OFF_A + GHH_N_STAGES * GHH_A_STAGE_BYTES
const GHH_OFF_C      = GHH_OFF_B + GHH_N_STAGES * GHH_B_STAGE_BYTES
const GHH_OFF_FULL   = GHH_OFF_C + GHH_C_BYTES
const GHH_OFF_EMPTY  = GHH_OFF_FULL  + GHH_N_STAGES * 8
const GHH_OFF_SCHED  = GHH_OFF_EMPTY + GHH_N_STAGES * 8
const GHH_TOTAL_SMEM = GHH_OFF_SCHED + GHH_SPACE_LEN * 4

# ── Hilbert schedule host helper ───────────────────────────────────────
# Maps Hilbert curve index `d` on an n×n grid to (x, y). n must be a
# power of 2. Mirrors pyptx/examples/hopper/gemm_highperf_hopper.py:_d2xy.
function _ghh_d2xy(n::Int, d::Int)
    x = 0; y = 0
    t = d
    s = 1
    while s < n
        rx = 1 & (t >> 1)
        ry = 1 & (t ⊻ rx)
        if ry == 0
            if rx == 1
                x = s - 1 - x
                y = s - 1 - y
            end
            x, y = y, x
        end
        x += s * rx
        y += s * ry
        t >>= 2
        s <<= 1
    end
    (x, y)
end

# Build the Hilbert schedule buffer. Each cluster receives a contiguous
# `SPACE_LEN` slot; tiles are assigned round-robin in Hilbert order.
# Slots beyond the cluster's tile count are filled with the sentinel.
function create_hilbert_schedule(m_tiles::Int, n_tiles::Int, num_clusters::Int;
                                  space_len::Int = GHH_SPACE_LEN)
    dim = 1 << ((max(m_tiles, n_tiles) - 1) |> x -> ndigits(x; base=2))
    @assert dim >= max(m_tiles, n_tiles)
    pos = [UInt32[] for _ in 1:num_clusters]
    core = 1
    total_tiles = 0
    for i in 0:(dim*dim - 1)
        x, y = _ghh_d2xy(dim, i)
        if x < m_tiles && y < n_tiles
            length(pos[core]) < space_len ||
                throw(ArgumentError("Hilbert schedule exceeded SPACE_LEN at cluster $(core-1)"))
            push!(pos[core], (UInt32(x) << 16) | UInt32(y))
            total_tiles += 1
            core = core % num_clusters + 1
        end
    end
    @assert total_tiles == m_tiles * n_tiles

    sched = fill(GHH_SENTINEL, num_clusters * space_len)
    for (i, entries) in enumerate(pos)
        base = (i - 1) * space_len
        for (j, e) in enumerate(entries)
            sched[base + j] = e
        end
    end
    sched
end

# ── Kernel ─────────────────────────────────────────────────────────────

function _ghh_gemm_kernel!(
        C::CuDeviceVector{UInt16, 1},
        tma_A::PTX.TMADescriptorPtr,
        tma_B::PTX.TMADescriptorPtr,
        tma_C::PTX.TMADescriptorPtr,
        sched::CuDeviceVector{UInt32, 1},
        M::Int32, N::Int32, K::Int32)

    # Dynamic SMEM: A / B ring buffers, C epilogue tile, mbarriers,
    # Hilbert schedule slice. Offsets are byte counts into the dynamic
    # block; CuDynamicSharedArray takes byte offsets.
    smem_A      = CuDynamicSharedArray(UInt16, GHH_N_STAGES * GHH_BM_CTA * GHH_BK_TILE, GHH_OFF_A)
    smem_B      = CuDynamicSharedArray(UInt16, GHH_N_STAGES * GHH_BK_TILE * GHH_BN, GHH_OFF_B)
    smem_C      = CuDynamicSharedArray(UInt16, GHH_BM_CTA * GHH_BN, GHH_OFF_C)
    mbar_full   = CuDynamicSharedArray(UInt64, GHH_N_STAGES, GHH_OFF_FULL)
    mbar_empty  = CuDynamicSharedArray(UInt64, GHH_N_STAGES, GHH_OFF_EMPTY)
    smem_sched  = CuDynamicSharedArray(UInt32, GHH_SPACE_LEN, GHH_OFF_SCHED)

    a_base_ptr    = pointer(smem_A)
    b_base_ptr    = pointer(smem_B)
    c_base_ptr    = pointer(smem_C)
    sched_base    = pointer(smem_sched)

    full  = BarrierArray{GHH_N_STAGES}(pointer(mbar_full))
    empty = BarrierArray{GHH_N_STAGES}(pointer(mbar_empty))

    tid       = ptx"mov.u32"(sreg"tid.x")
    cta_rank  = ptx"mov.u32"(sreg"cluster_ctarank")
    clusterid = ptx"mov.u32"(sreg"clusterid.x")
    wg_id     = tid >> UInt32(7)
    lane128   = tid & UInt32(127)

    # ── Load this cluster's schedule slice into SMEM (warpgroup 0, lanes 0..SPACE_LEN-1) ──
    if tid < UInt32(GHH_SPACE_LEN)
        global_off = (clusterid * UInt32(GHH_SPACE_LEN) + tid) * UInt32(4)
        g_ptr = pointer(sched) + Int(global_off)
        v = ptx"ld.global.u32"(g_ptr)
        smem_off = tid * UInt32(4)
        ptx"st.shared.u32"(sched_base + Int(smem_off), v)
    end

    # ── Init mbarriers + pipeline warm-up (thread 0) ────────────────
    if tid == UInt32(0)
        pipeline_init!(full, empty, Val(GHH_EMPTY_COUNT), Val(true))
    end
    ptx"bar.sync"(Val(0))

    cluster_arrive()
    cluster_wait()

    num_blocks_k = K ÷ Int32(GHH_BK_TILE)

    if wg_id == UInt32(0)
        # ─────────────────────── PRODUCER ────────────────────────────
        ptx"setmaxnreg.dec.sync.aligned.u32"(Val(24))

        if lane128 == UInt32(0)
            # Persistent tile loop: read tile from SMEM-resident schedule,
            # bail on sentinel, otherwise run K-loop with TMA loads.
            tile_iter = UInt32(0)
            global_k_iter = Int32(0)   # monotonically increases across tiles → drives stage/phase
            while true
                packed = ptx"ld.shared.u32"(sched_base + Int(tile_iter * UInt32(4)))
                packed == GHH_SENTINEL && break

                tile_m = packed >> UInt32(16)
                tile_n = packed & UInt32(0xffff)
                m_off  = (tile_m * UInt32(GHH_CLUSTERS) + cta_rank) * UInt32(GHH_BM_CTA)
                n_off  = tile_n * UInt32(GHH_BN)

                @inbounds for kt in Int32(0):(num_blocks_k - Int32(1))
                    stage, phase = pipeline_cursor(Pipeline{GHH_N_STAGES}, global_k_iter)
                    barrier_wait(empty[stage], phase)

                    ptx"fence.proxy.async.shared::cta"()
                    barrier_arrive_expect_tx(full[stage], GHH_LOAD_BYTES)
                    a_stage_ptr = a_base_ptr + Int(stage) * GHH_A_STAGE_BYTES
                    b_stage_ptr = b_base_ptr + Int(stage) * GHH_B_STAGE_BYTES
                    k_off = kt * Int32(GHH_BK_TILE)
                    ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                        a_stage_ptr, tma_A, k_off,
                        reinterpret(Int32, m_off), full[stage])
                    if cta_rank == UInt32(0)
                        ptx"cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"(
                            b_stage_ptr, tma_B, k_off,
                            reinterpret(Int32, n_off), full[stage], UInt16(0x3))
                    end

                    global_k_iter += Int32(1)
                end

                tile_iter += UInt32(1)
                tile_iter < UInt32(GHH_SPACE_LEN) || break
            end
        end
    else
        # ─────────────────── CONSUMER (WG 1 or WG 2) ─────────────────
        ptx"setmaxnreg.inc.sync.aligned.u32"(Val(240))
        consumer_id = wg_id - UInt32(1)

        a_slice_ptr  = a_base_ptr + Int(consumer_id) * GHH_A_CONSUMER_BYTES
        a_slice_addr = smem_addr_u32(a_slice_ptr)
        b_addr       = smem_addr_u32(b_base_ptr)

        la = layout_for_a(dtype = :bf16, m = GHH_B_WG_M, k = GHH_BK_TILE)
        lb = layout_for_a(dtype = :bf16, m = GHH_BN,     k = GHH_BK_TILE)
        a_desc_base = wgmma_descriptor(a_slice_addr;
            leading_byte_offset = la.leading_byte_offset,
            stride_byte_offset  = la.stride_byte_offset,
            swizzle             = la.layout_type)
        b_desc_base = wgmma_descriptor(b_addr;
            leading_byte_offset = lb.leading_byte_offset,
            stride_byte_offset  = lb.stride_byte_offset,
            swizzle             = lb.layout_type)

        # Persistent tile loop.
        tile_iter = UInt32(0)
        global_k_iter = Int32(0)
        while true
            packed = ptx"ld.shared.u32"(sched_base + Int(tile_iter * UInt32(4)))
            packed == GHH_SENTINEL && break

            tile_m = packed >> UInt32(16)
            tile_n = packed & UInt32(0xffff)
            m_off  = (tile_m * UInt32(GHH_CLUSTERS) + cta_rank) * UInt32(GHH_BM_CTA)
            n_off  = tile_n * UInt32(GHH_BN)

            # Reset accumulator per tile.
            d = ntuple(_ -> 0f0, Val(64))

            @inbounds for kt in Int32(0):(num_blocks_k - Int32(1))
                stage, phase = pipeline_cursor(Pipeline{GHH_N_STAGES}, global_k_iter)
                barrier_wait(full[stage], phase)

                a_desc_stage = step_desc(a_desc_base, Int(stage) * GHH_A_STAGE_BYTES)
                b_desc_stage = step_desc(b_desc_base, Int(stage) * GHH_B_STAGE_BYTES)
                ptx"fence.proxy.async.shared::cta"()
                ptx"wgmma.fence.sync.aligned"()
                # Tile-K inner loop: WGMMA_KITERS wgmma sub-calls per K-tile.
                # Each sub-call advances the descriptor's SMEM address by
                # WGMMA_K × elem_bytes = 32 bytes.
                @inbounds for kk in 0:(GHH_WGMMA_KITERS - 1)
                    k_off = kk * GHH_K_STEP_BYTES
                    d = ptx"wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16"(
                        d, step_desc(a_desc_stage, k_off),
                           step_desc(b_desc_stage, k_off), true)
                end
                ptx"wgmma.commit_group.sync.aligned"()
                ptx"wgmma.wait_group.sync.aligned"(Val(0))

                if lane128 < UInt32(GHH_CLUSTERS)
                    barrier_arrive_cluster(empty[stage], lane128)
                end

                global_k_iter += Int32(1)
            end

            # ── Epilogue: f32 → bf16 → SMEM C-tile → TMA store ──────
            # m64n128k16 frag layout per lane: 16 col-groups of 4 frags.
            # Group g (0..15) → cols (col_a + 8g, col_a + 8g + 1).
            wid   = lane128 >> UInt32(5)
            lane  = lane128 & UInt32(31)
            row_a = (wid << UInt32(4)) + (lane >> UInt32(2))
            row_b = row_a + UInt32(8)
            col_a = (lane & UInt32(3)) << UInt32(1)

            c_consumer_ptr = c_base_ptr + Int(consumer_id) * Int(GHH_B_WG_M) * GHH_BN * 2

            # Unrolled epilogue stores. 16 groups × 2 packed stores (row_a, row_b).
            # Generated by Base.Cartesian.@nexprs for literal-index tuple access.
            Base.Cartesian.@nexprs 16 g -> begin
                fidx_a1 = 4 * (g - 1) + 1                       # d[4g-3]
                fidx_a2 = 4 * (g - 1) + 2                       # d[4g-2]
                fidx_b1 = 4 * (g - 1) + 3                       # d[4g-1]
                fidx_b2 = 4 * (g - 1) + 4                       # d[4g]
                col_g   = col_a + UInt32(8 * (g - 1))           # col base for this group
                pack_a  = bf16x2_pack(d[fidx_a1], d[fidx_a2])
                pack_b  = bf16x2_pack(d[fidx_b1], d[fidx_b2])
                off_a   = (row_a * UInt32(GHH_BN) + col_g) * UInt32(2)
                off_b   = (row_b * UInt32(GHH_BN) + col_g) * UInt32(2)
                ptx"st.shared.b32"(c_consumer_ptr + Int(off_a), pack_a)
                ptx"st.shared.b32"(c_consumer_ptr + Int(off_b), pack_b)
            end

            # Named-bar sync within just this consumer warpgroup.
            if consumer_id == UInt32(0)
                ptx"bar.sync"(Val(2), Val(128))
            else
                ptx"bar.sync"(Val(3), Val(128))
            end

            # Lane 0 of each consumer issues TMA store of its B_WG_M × BN slice.
            if lane128 == UInt32(0)
                ptx"fence.proxy.async.shared::cta"()
                store_m = reinterpret(Int32, m_off + consumer_id * UInt32(GHH_B_WG_M))
                store_n = reinterpret(Int32, n_off)
                ptx"cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"(
                    tma_C, store_n, store_m, c_consumer_ptr)
                ptx"cp.async.bulk.commit_group"()
            end

            tile_iter += UInt32(1)
            tile_iter < UInt32(GHH_SPACE_LEN) || break
        end
    end

    ptx"cp.async.bulk.wait_group"(Val(0))
    return nothing
end

# ── Cross-arch ptxas validation ────────────────────────────────────────

@testset "gemm_highperf_hopper compiles at sm_90a" begin
    types = Tuple{CuDeviceVector{UInt16, 1},
                  PTX.TMADescriptorPtr, PTX.TMADescriptorPtr, PTX.TMADescriptorPtr,
                  CuDeviceVector{UInt32, 1},
                  Int32, Int32, Int32}
    @test ptxas_compiles(_ghh_gemm_kernel!, types;
                         cap = v"9.0", feature_set = :arch)

    ptx = emit_ptx(_ghh_gemm_kernel!, types; cap = v"9.0", feature_set = :arch)
    @test occursin("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16", ptx)
    @test occursin("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster", ptx)
    @test occursin("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group", ptx)
    @test occursin("mapa.shared::cluster.u32",       ptx)
    @test occursin("mbarrier.arrive.shared::cluster.b64", ptx)
    @test occursin("setmaxnreg.inc.sync.aligned.u32", ptx)
    @test occursin("setmaxnreg.dec.sync.aligned.u32", ptx)
    @test occursin("ld.shared.u32",                  ptx)
    # 16 col-groups × 2 packs = 32 fused `cvt.rn.bf16x2.f32` per lane.
    n_pack = length(collect(eachmatch(r"cvt\.rn\.bf16x2\.f32", ptx)))
    @test n_pack >= 32
end

# ── Hilbert host helper unit test ─────────────────────────────────────

@testset "Hilbert schedule covers every tile exactly once" begin
    for (m_t, n_t, nc) in [(1, 1, 1), (2, 2, 2), (4, 4, 4), (2, 4, 2), (3, 5, 2)]
        sched = create_hilbert_schedule(m_t, n_t, nc; space_len = GHH_SPACE_LEN)
        @test length(sched) == nc * GHH_SPACE_LEN
        # Collect non-sentinel entries; assert covers all (m_tile, n_tile).
        seen = Set{Tuple{Int, Int}}()
        for v in sched
            v == GHH_SENTINEL && continue
            m = Int(v >> 16); n = Int(v & 0xffff)
            push!(seen, (m, n))
        end
        @test length(seen) == m_t * n_t
        for m in 0:(m_t-1), n in 0:(n_t-1)
            @test (m, n) in seen
        end
    end
end

# ── Runtime — clusters-capable arch only ───────────────────────────────

if v"9.0" <= DEV_CAP < v"12.0"
    function _run_ghh_case(rng, M_total, N_total, K_test, num_clusters)
        @assert M_total % GHH_BM_CLUSTER == 0
        @assert N_total % GHH_BN         == 0
        @assert K_test  % GHH_BK_TILE    == 0

        m_tiles = M_total ÷ GHH_BM_CLUSTER
        n_tiles = N_total ÷ GHH_BN

        A_f32 = randn(rng, Float32, M_total, K_test) .* 0.05f0
        B_f32 = randn(rng, Float32, K_test, N_total) .* 0.05f0

        A_packed = Array{UInt16}(undef, K_test, M_total)
        B_packed = Array{UInt16}(undef, K_test, N_total)
        for m in 1:M_total, k in 1:K_test
            A_packed[k, m] = bf16_bits(A_f32[m, k])
        end
        for k in 1:K_test, n in 1:N_total
            B_packed[k, n] = bf16_bits(B_f32[k, n])
        end
        A_d = CuArray(A_packed)
        B_d = CuArray(B_packed)
        C_d = CUDACore.zeros(UInt16, M_total * N_total)

        tmap_A = tensor_map_tile_2d(:bf16, pointer(A_d),
            M_total, K_test, GHH_BM_CTA, GHH_BK_TILE; swizzle = :B128)
        tmap_B = tensor_map_tile_2d(:bf16, pointer(B_d),
            N_total, K_test, GHH_BN, GHH_BK_TILE; swizzle = :B128)
        tmap_C = tensor_map_tile_2d(:bf16, pointer(C_d),
            M_total, N_total, GHH_B_WG_M, GHH_BN; swizzle = :NONE)
        A = upload_tma_descriptor(tmap_A)
        B = upload_tma_descriptor(tmap_B)
        Cd = upload_tma_descriptor(tmap_C)

        sched = create_hilbert_schedule(m_tiles, n_tiles, num_clusters)
        sched_d = CuArray(sched)

        grid = (GHH_CLUSTERS * num_clusters, 1, 1)
        # Compile without launching so we can opt the kernel into the
        # >48 KB dynamic-SMEM regime. Per CUDA, kernels requesting more
        # than the default 48 KB dynamic SMEM must explicitly set
        # MAX_DYNAMIC_SHARED_SIZE_BYTES on the function before launch
        # (Hopper allows up to ~228 KB per CTA when opted in).
        kern = @cuda launch = false feature_set = :arch _ghh_gemm_kernel!(
            C_d, A.ptr, B.ptr, Cd.ptr, sched_d,
            Int32(M_total), Int32(N_total), Int32(K_test))
        CUDACore.attributes(kern.fun)[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = GHH_TOTAL_SMEM
        kern(C_d, A.ptr, B.ptr, Cd.ptr, sched_d,
             Int32(M_total), Int32(N_total), Int32(K_test);
             blocks = grid, threads = GHH_THREADS,
             clustersize = (GHH_CLUSTERS, 1, 1),
             shmem = GHH_TOTAL_SMEM)
        CUDACore.synchronize()

        # C_d is N-innermost: Julia (N, M) col-major.
        C_packed = reshape(Array(C_d), N_total, M_total)
        C_got    = Array{Float32}(undef, M_total, N_total)
        for m in 1:M_total, n in 1:N_total
            C_got[m, n] = bf16_to_f32(C_packed[n, m])
        end
        C_ref = bf16_gemm_ref(A_f32, B_f32)
        return C_got, C_ref
    end

    @testset "gemm_highperf_hopper (M=256, N=128, K=64, single tile, single cluster)" begin
        # Degenerate: 1 cluster handles 1 tile with 1 K-iter. Smoke test
        # of the persistent loop's exit path on the sentinel.
        C_got, C_ref = _run_ghh_case(MersenneTwister(0x901), 256, 128, 64, 1)
        @test isapprox(C_got, C_ref; rtol = 5e-3, atol = 5e-3)
    end

    @testset "gemm_highperf_hopper (M=512, N=512, K=256, multi-tile + multi-K)" begin
        # 2 m-clusters × 4 n-tiles = 8 tiles. With 2 clusters launched,
        # each cluster does 4 tiles. K=256 → num_blocks_k=4 per tile.
        C_got, C_ref = _run_ghh_case(MersenneTwister(0x902), 512, 512, 256, 2)
        @test isapprox(C_got, C_ref; rtol = 5e-3, atol = 5e-3)
    end
end
