# TEST_TARGET: requires=toolkit evidence=mixed target=sm_90a
# FP8 GEMM with blockwise scaling — ported from
# cutlass/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling
# (NVIDIA CUTLASS, BSD-3-Clause).
#
# The CUTLASS example extends the warp-specialized FP8 GEMM (example 54)
# with *blockwise* scale factors: instead of one dequant scale per tensor,
# each (tile_M × tile_K) block of A and (tile_K × tile_N) block of B
# carries its own f32 scale, and the mainloop promotes each block's
# partial product into the f32 accumulator as
#
#     acc += scale_A[blk] * scale_B[blk] * (A_blk × B_blk)
#
# — the recipe DeepSeek-V3-style fp8 training popularized. The MMA result
# for a block therefore needs to land in a FRESH accumulator each K-block
# so the scale multiplies only that block's contribution.
#
# What's ported (on top of gemm_fp8_warpspec.jl, same producer/consumer
# 2-stage TMA ring, e4m3 operands, m64n8k32 wgmma):
#
#   - Per-K-block scale vectors for A and B (this kernel has a single
#     M/N tile, so CUTLASS's (M-blocks × K-blocks) scale grid degenerates
#     to one scale per K-tile per operand — the mainloop promotion logic
#     is identical, the grid is just 1×num_k_tiles).
#   - Fresh-accumulator wgmma via the `Val(false)` scale_d immediate:
#     the variant bakes `scale-d = 0` and drops the tied d input, so the
#     hardware ignores the accumulator and LLVM DCEs the zero init —
#     exactly the "clear the register block, then MMA" step CUTLASS's
#     promotion loop performs.
#   - In-register promotion: acc = fma(s, d_blk, acc) per fragment, with
#     s = scale_A[blk] · scale_B[blk] loaded by every consumer lane.
#
# Not ported: the abs-max(D) epilogue fusion (covered by
# gemm_fp8_warpspec.jl — dropped here to keep the diff against that file
# exactly the blockwise-promotion machinery), CTA rasterization tuning,
# and the C operand (beta = 0).

using PTX: layout_for_a, wgmma_descriptor, smem_addr_u32, tensor_map_tile_2d,
           step_desc
using PTX.MBarriers: BarrierArray,
                     barrier_wait, barrier_arrive, barrier_arrive_expect_tx
using PTX.Pipelines: Pipeline, pipeline_init!, pipeline_cursor
using CUDACore
using Microfloats
using Random

const FP8BS_BM       = 64
const FP8BS_BN       = 8
const FP8BS_BK       = 32                        # e4m3 wgmma K-step = block size
const FP8BS_N_STAGES = 2
const FP8BS_THREADS  = 256                       # 2 warpgroups

const FP8BS_A_STAGE_BYTES = FP8BS_BM * FP8BS_BK  # 2048
const FP8BS_B_STAGE_BYTES = FP8BS_BK * FP8BS_BN  # 256
const FP8BS_LOAD_BYTES    = FP8BS_A_STAGE_BYTES + FP8BS_B_STAGE_BYTES

function _fp8_bs_gemm_kernel!(
        D::CuDeviceVector{Float32, 1},
        scale_A::CuDeviceVector{Float32, 1},   # one f32 per K-block of A
        scale_B::CuDeviceVector{Float32, 1},   # one f32 per K-block of B
        tma_A::PTX.TMADescriptorPtr,
        tma_B::PTX.TMADescriptorPtr,
        K::Int32)

    smem_A = CuStaticSharedArray(UInt8, FP8BS_N_STAGES * FP8BS_BM * FP8BS_BK)
    smem_B = CuStaticSharedArray(UInt8, FP8BS_N_STAGES * FP8BS_BK * FP8BS_BN)
    mbar_full  = CuStaticSharedArray(UInt64, FP8BS_N_STAGES)
    mbar_empty = CuStaticSharedArray(UInt64, FP8BS_N_STAGES)

    a_base_ptr  = pointer(smem_A)
    b_base_ptr  = pointer(smem_B)
    a_base_addr = smem_addr_u32(a_base_ptr)
    b_base_addr = smem_addr_u32(b_base_ptr)

    full  = BarrierArray{FP8BS_N_STAGES}(pointer(mbar_full))
    empty = BarrierArray{FP8BS_N_STAGES}(pointer(mbar_empty))

    tid     = ptx"mov.u32"(sreg"tid.x")
    wg_id   = tid >> UInt32(7)
    lane128 = tid & UInt32(127)

    if tid == 0
        pipeline_init!(full, empty, Val(1), Val(false))
    end
    ptx"bar.sync"(Val(0))

    num_k_tiles = K >> Int32(5)                  # K / BK

    if wg_id == UInt32(0)
        # ── PRODUCER (WG 0) — identical to gemm_fp8_warpspec.jl ────────
        ptx"setmaxnreg.dec.sync.aligned.u32"(Val(24))

        @inbounds for k_iter in Int32(0):(num_k_tiles - Int32(1))
            stage, phase = pipeline_cursor(Pipeline{FP8BS_N_STAGES}, k_iter)
            barrier_wait(empty[stage], phase)

            if lane128 == UInt32(0)
                ptx"fence.proxy.async.shared::cta"()
                barrier_arrive_expect_tx(full[stage], FP8BS_LOAD_BYTES)
                a_stage_ptr = a_base_ptr + Int(stage) * FP8BS_A_STAGE_BYTES
                b_stage_ptr = b_base_ptr + Int(stage) * FP8BS_B_STAGE_BYTES
                k_off = k_iter * Int32(FP8BS_BK)
                ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                    a_stage_ptr, tma_A, k_off, Int32(0), full[stage])
                ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                    b_stage_ptr, tma_B, k_off, Int32(0), full[stage])
            end
        end
    else
        # ── CONSUMER (WG 1): fresh-acc wgmma per block + promotion ─────
        ptx"setmaxnreg.inc.sync.aligned.u32"(Val(240))

        la = layout_for_a(dtype = :e4m3, m = FP8BS_BM, k = FP8BS_BK)
        lb = layout_for_a(dtype = :e4m3, m = FP8BS_BN, k = FP8BS_BK)
        a_desc_base = wgmma_descriptor(a_base_addr;
            leading_byte_offset = la.leading_byte_offset,
            stride_byte_offset  = la.stride_byte_offset,
            swizzle             = la.layout_type)
        b_desc_base = wgmma_descriptor(b_base_addr;
            leading_byte_offset = lb.leading_byte_offset,
            stride_byte_offset  = lb.stride_byte_offset,
            swizzle             = lb.layout_type)

        acc   = ntuple(_ -> 0f0, Val(4))         # promoted f32 accumulator
        dzero = ntuple(_ -> 0f0, Val(4))         # dummy for Val(false) (DCE'd)

        @inbounds for k_iter in Int32(0):(num_k_tiles - Int32(1))
            stage, phase = pipeline_cursor(Pipeline{FP8BS_N_STAGES}, k_iter)
            barrier_wait(full[stage], phase)

            a_desc = step_desc(a_desc_base, Int(stage) * FP8BS_A_STAGE_BYTES)
            b_desc = step_desc(b_desc_base, Int(stage) * FP8BS_B_STAGE_BYTES)
            ptx"fence.proxy.async.shared::cta"()
            ptx"wgmma.fence.sync.aligned"()
            # scale_d = 0 (Val(false)): the block's product lands in fresh
            # registers, NOT accumulated — the scale below must multiply
            # only this block's contribution.
            d_blk = ptx"wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3"(
                dzero, a_desc, b_desc, Val(false))
            ptx"wgmma.commit_group.sync.aligned"()
            ptx"wgmma.wait_group.sync.aligned"(Val(0))

            if lane128 == UInt32(0)
                barrier_arrive(empty[stage])
            end

            # Promotion: acc += (sA·sB) · d_blk. Reading d_blk is only legal
            # after wait_group(0) drained the wgmma pipe (PTX 9.2 §9.7.14.5.2).
            s = scale_A[Int(k_iter) + 1] * scale_B[Int(k_iter) + 1]
            acc = (fma(s, d_blk[1], acc[1]), fma(s, d_blk[2], acc[2]),
                   fma(s, d_blk[3], acc[3]), fma(s, d_blk[4], acc[4]))
        end

        # ── Epilogue: plain per-lane v2 stores (frag layout as in
        # gemm_pc_pipeline.jl; scales already applied in the mainloop).
        wid      = lane128 >> UInt32(5)
        lane     = lane128 & UInt32(31)
        frag_row = (wid << UInt32(4)) + (lane >> UInt32(2))
        frag_col = (lane & UInt32(3)) << UInt32(1)
        pd       = pointer(D)
        off_a    = (frag_row * UInt32(FP8BS_BN) + frag_col) * UInt32(4)
        off_b    = ((frag_row + UInt32(8)) * UInt32(FP8BS_BN) + frag_col) * UInt32(4)
        ptx"st.global.v2.f32"(pd + Int(off_a), (acc[1], acc[2]))
        ptx"st.global.v2.f32"(pd + Int(off_b), (acc[3], acc[4]))
    end

    return nothing
end

# ── Cross-arch ptxas validation ────────────────────────────────────────

@testset "FP8 blockwise-scaling GEMM compiles at sm_90a" begin
    types = Tuple{CuDeviceVector{Float32, 1},
                  CuDeviceVector{Float32, 1}, CuDeviceVector{Float32, 1},
                  PTX.TMADescriptorPtr, PTX.TMADescriptorPtr, Int32}
    @test ptxas_compiles(_fp8_bs_gemm_kernel!, types;
                         cap = v"9.0", feature_set = :arch)

    ptx = emit_ptx(_fp8_bs_gemm_kernel!, types; cap = v"9.0", feature_set = :arch)
    @test occursin("wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3", ptx)
    # The Val(false) variant bakes scale-d as the "0" immediate (fp8 wgmma
    # has no trans bits, so the imm tail is `0, 1, 1`).
    @test occursin(r"e4m3\.e4m3[^;]*, 0, 1, 1;", ptx)
    # Promotion lowers to fused multiply-add on the block product.
    @test occursin("fma.rn.f32", ptx)
end

# ── Runtime — Hopper hardware ──────────────────────────────────────────

if test_runtime_supported(@__FILE__)
    @testset "FP8 blockwise-scaling GEMM (K=128, non-uniform block scales)" begin
        rng = MersenneTwister(0x67b5)
        K_test = 128
        num_blocks = K_test ÷ FP8BS_BK           # 4
        A_f32 = randn(rng, Float32, FP8BS_BM, K_test) .* 0.25f0
        B_f32 = randn(rng, Float32, K_test, FP8BS_BN) .* 0.25f0

        A_q = Float8_E4M3FN.(A_f32)
        B_q = Float8_E4M3FN.(B_f32)
        A_deq = Float32.(A_q)
        B_deq = Float32.(B_q)

        A_packed = Array{UInt8}(undef, K_test, FP8BS_BM)
        B_packed = Array{UInt8}(undef, K_test, FP8BS_BN)
        for m in 1:FP8BS_BM, k in 1:K_test
            A_packed[k, m] = reinterpret(UInt8, A_q[m, k])
        end
        for k in 1:K_test, n in 1:FP8BS_BN
            B_packed[k, n] = reinterpret(UInt8, B_q[k, n])
        end
        A_d = CuArray(A_packed)
        B_d = CuArray(B_packed)

        tmap_A = tensor_map_tile_2d(:u8, pointer(A_d),
            FP8BS_BM, K_test, FP8BS_BM, FP8BS_BK; swizzle = :B32)
        tmap_B = tensor_map_tile_2d(:u8, pointer(B_d),
            FP8BS_BN, K_test, FP8BS_BN, FP8BS_BK; swizzle = :B32)
        A = upload_tma_descriptor(tmap_A)
        B = upload_tma_descriptor(tmap_B)

        # Deliberately non-uniform, per-block-distinct scales (powers of two
        # times a random jitter) so a block/scale misalignment can't cancel.
        sA = Float32[2.0f0^(b % 3) * (0.5f0 + rand(rng, Float32)) for b in 0:num_blocks-1]
        sB = Float32[0.5f0^(b % 2) * (0.5f0 + rand(rng, Float32)) for b in 0:num_blocks-1]
        sA_d = CuArray(sA)
        sB_d = CuArray(sB)

        D_dev = CUDACore.zeros(Float32, FP8BS_BM * FP8BS_BN)
        @cuda threads = FP8BS_THREADS _fp8_bs_gemm_kernel!(
            D_dev, sA_d, sB_d, A.ptr, B.ptr, Int32(K_test))
        CUDACore.synchronize()

        D_packed = reshape(Array(D_dev), FP8BS_BN, FP8BS_BM)
        D_got    = Array{Float32}(undef, FP8BS_BM, FP8BS_BN)
        for m in 1:FP8BS_BM, n in 1:FP8BS_BN
            D_got[m, n] = D_packed[n, m]
        end

        # Reference: blockwise-promoted sum, matching the kernel's math
        # (each block's product scaled before accumulation).
        D_ref = zeros(Float32, FP8BS_BM, FP8BS_BN)
        for b in 1:num_blocks
            ks = (b-1)*FP8BS_BK + 1 : b*FP8BS_BK
            D_ref .+= (sA[b] * sB[b]) .* (A_deq[:, ks] * B_deq[ks, :])
        end
        @test isapprox(D_got, D_ref; rtol = 1e-3, atol = 1e-3)
    end
end
