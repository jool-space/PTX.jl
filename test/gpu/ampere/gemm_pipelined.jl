# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.0
#
# Ported from pyptx/examples/ampere/gemm_pipelined.py
# (https://github.com/patrick-toulme/pyptx).
# Copyright 2026 Patrick Toulmé. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Translated to Julia and adapted.
#
# Ampere (sm_80) bf16 GEMM with cp.async + SMEM ring buffer + mma.sync.
#
# Layered on top of test/gpu/ampere_gemm.jl (the no-SMEM straight-line port).
# Adds every Ampere primitive needed before the production kernel:
#   - cp.async.cg.shared.global   (16-byte async global → SMEM prefetch)
#   - cp.async.commit_group       (close in-flight prefetches into a group)
#   - cp.async.wait_group N       (wait until ≤ N groups remain)
#   - 2-stage SMEM ring buffer    (dynamic shared memory, shmem= at launch)
#   - 4-warp CTA                  (warp w handles M[w*16 : (w+1)*16])
#
# Per-thread `ld.shared.b32` (no ldmatrix) for the SMEM → register hand-off —
# the fragment-layout math is identical to ampere_gemm.jl. ldmatrix is
# deferred to the next stop on the way to the high-perf kernel.
#
# Tile shape:
#   BM × BN = 64 × 64,   BK = 16
#   4 warps/CTA, warp w owns M[w*16 : (w+1)*16]   →   WM = 16
#   N_FRAG_N = BN / 8 = 8 mma.sync per warp per K-iter
#
# Inputs (flat row-major flattened):
#   A    :: (M, K) bf16  stored as UInt16
#   B_T  :: (N, K) bf16  stored as UInt16  (B^T, K-contiguous for mma row.col)
#   D    :: (M, N) f32
#
# Grid:  (N/BN, M/BM)
# Block: (THREADS, 1, 1)   THREADS = 32 * 4 = 128
# Shmem: STAGES * (A_STAGE_BYTES + B_STAGE_BYTES) = 8 KiB

using Random
using Base: @nexprs

bf16_bits_p(x::Float32)  = UInt16(reinterpret(UInt32, x) >> 16)
bf16_to_f32_p(b::UInt16) = reinterpret(Float32, UInt32(b) << 16)

function pack_bf16_rowmajor_p(A::AbstractMatrix{Float32})
    rows, cols = size(A)
    out = Vector{UInt16}(undef, rows * cols)
    @inbounds for i in 1:rows, j in 1:cols
        out[(i-1)*cols + j] = bf16_bits_p(A[i, j])
    end
    out
end

quantize_bf16_p(A) = bf16_to_f32_p.(bf16_bits_p.(A))

const PIPE_BM, PIPE_BN, PIPE_BK = 64, 64, 16
const PIPE_NUM_WARPS = 4
const PIPE_THREADS   = 32 * PIPE_NUM_WARPS
const PIPE_N_FRAG_N  = PIPE_BN ÷ 8
const PIPE_STAGES    = 2

const PIPE_A_STAGE_BYTES = PIPE_BM * PIPE_BK * 2
const PIPE_B_STAGE_BYTES = PIPE_BN * PIPE_BK * 2
const PIPE_A_SMEM_BASE   = 0
const PIPE_B_SMEM_BASE   = PIPE_STAGES * PIPE_A_STAGE_BYTES
const PIPE_SMEM_BYTES    = PIPE_STAGES * (PIPE_A_STAGE_BYTES + PIPE_B_STAGE_BYTES)

function gemm_pipelined_kernel!(
        D::CuDeviceVector{Float32},
        A::CuDeviceVector{UInt16},
        B_T::CuDeviceVector{UInt16},
        ::Val{M}, ::Val{N}, ::Val{K}) where {M, N, K}

    smem      = CuDynamicSharedArray(UInt8, PIPE_SMEM_BYTES)
    smem_base = pointer(smem)

    pa = pointer(A); pb = pointer(B_T); pd = pointer(D)
    n_iters = K ÷ PIPE_BK

    m_base = Int(ptx"mov.u32"(sreg"ctaid.y")) * PIPE_BM
    n_base = Int(ptx"mov.u32"(sreg"ctaid.x")) * PIPE_BN

    tid     = ptx"mov.u32"(sreg"tid.x")
    warp_id = Int(tid >> UInt32(5))             # 0..3
    lane    = Int(tid & UInt32(31))             # 0..31

    # Per-thread cp.async layout: each thread loads exactly 16 bytes (= 8 bf16)
    # per stage per matrix. 128 threads × 16 B = 2 KB/stage = BM*BK*2 = BN*BK*2.
    # Map: thread t → row t/2, cols (t%2)*8..(t%2)*8+7.
    load_row      = Int(tid >> UInt32(1))       # 0..63
    col_chunk     = Int(tid & UInt32(1))
    col_start     = col_chunk << 3              # 0 or 8 (bf16 elements)
    load_smem_off = (load_row * PIPE_BK + col_start) * 2  # bytes within a stage

    # m16n8k16 per-lane fragment indices, warp's M slice.
    gid    = lane >> 2                          # 0..7
    tig    = lane & 0x3                         # 0..3
    col_lo = tig << 1                           # 0,2,4,6 (bf16 elements)
    warp_a_row_base = warp_id << 4              # warp_id * 16 (SMEM rows)

    @inline function issue_cp_async(s::Int, k_base::Int)
        a_smem_dst = smem_base + (PIPE_A_SMEM_BASE + s * PIPE_A_STAGE_BYTES) + load_smem_off
        b_smem_dst = smem_base + (PIPE_B_SMEM_BASE + s * PIPE_B_STAGE_BYTES) + load_smem_off
        a_off = ((m_base + load_row) * K + k_base + col_start) * 2
        b_off = ((n_base + load_row) * K + k_base + col_start) * 2
        ptx"cp.async.cg.shared.global"(a_smem_dst, pa + a_off, Val(16))
        ptx"cp.async.cg.shared.global"(b_smem_dst, pb + b_off, Val(16))
        return nothing
    end

    # Prologue: prime up to STAGES groups in flight.
    issue_cp_async(0, 0)
    ptx"cp.async.commit_group"()
    if n_iters > 1
        issue_cp_async(1, PIPE_BK)
        ptx"cp.async.commit_group"()
    end

    # Accumulator: PIPE_N_FRAG_N tiles × 4 f32 = 32 f32/lane.
    @nexprs 8 nf -> (acc_nf = (0f0, 0f0, 0f0, 0f0))

    @inbounds for ki in 0:n_iters-1
        stage = ki & 1
        # Steady iters: keep one prefetch in flight (STAGES-1 groups remain).
        # Last iter: drain everything (wait_all == wait_group 0).
        if ki < n_iters - 1
            ptx"cp.async.wait_group"(Val(PIPE_STAGES - 1))
        else
            ptx"cp.async.wait_all"()
        end
        ptx"bar.sync"(Val(0))

        a_stage_base = smem_base + PIPE_A_SMEM_BASE + stage * PIPE_A_STAGE_BYTES
        b_stage_base = smem_base + PIPE_B_SMEM_BASE + stage * PIPE_B_STAGE_BYTES

        # A frag: 4 b32 regs/lane at (drow, dcol) ∈ {(0,0),(8,0),(0,8),(8,8)}.
        a1 = ptx"ld.shared.b32"(a_stage_base + ((warp_a_row_base + gid + 0) * PIPE_BK + (col_lo + 0)) * 2)
        a2 = ptx"ld.shared.b32"(a_stage_base + ((warp_a_row_base + gid + 8) * PIPE_BK + (col_lo + 0)) * 2)
        a3 = ptx"ld.shared.b32"(a_stage_base + ((warp_a_row_base + gid + 0) * PIPE_BK + (col_lo + 8)) * 2)
        a4 = ptx"ld.shared.b32"(a_stage_base + ((warp_a_row_base + gid + 8) * PIPE_BK + (col_lo + 8)) * 2)
        a_frag = (a1, a2, a3, a4)

        # 8 N-frags: load 2 b32 regs/lane each, issue 8 mma.sync.
        @nexprs 8 nf -> begin
            b_row_base = ((nf - 1) * 8 + gid) * PIPE_BK
            b_lo = ptx"ld.shared.b32"(b_stage_base + (b_row_base + col_lo + 0) * 2)
            b_hi = ptx"ld.shared.b32"(b_stage_base + (b_row_base + col_lo + 8) * 2)
            acc_nf = ptx"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"(
                a_frag, (b_lo, b_hi), acc_nf)
        end

        # Sync before any warp issues the next prefetch into this stage —
        # the prefetch overwrites the SMEM we just read, so a fast warp
        # could clobber a slow warp's in-flight ld.shared without this.
        ptx"bar.sync"(Val(0))

        if ki + PIPE_STAGES < n_iters
            next_stage = (ki + PIPE_STAGES) & 1
            next_k     = (ki + PIPE_STAGES) * PIPE_BK
            issue_cp_async(next_stage, next_k)
            ptx"cp.async.commit_group"()
        end
    end

    # Epilogue: each lane writes its 8 m16n8 acc tiles to global D.
    warp_m_global = m_base + (warp_id << 4)
    row_lo = warp_m_global + gid
    row_hi = row_lo + 8
    @nexprs 8 nf -> begin
        n_global   = n_base + ((nf - 1) << 3)
        d_col_base = n_global + col_lo
        ptx"st.global.f32"(pd + (row_lo * N + (d_col_base + 0)) * 4, acc_nf[1])
        ptx"st.global.f32"(pd + (row_lo * N + (d_col_base + 1)) * 4, acc_nf[2])
        ptx"st.global.f32"(pd + (row_hi * N + (d_col_base + 0)) * 4, acc_nf[3])
        ptx"st.global.f32"(pd + (row_hi * N + (d_col_base + 1)) * 4, acc_nf[4])
    end

    return nothing
end

function run_pipelined_gemm(M::Int, N::Int, K::Int)
    @assert M % PIPE_BM == 0 "M=$M must be divisible by $PIPE_BM"
    @assert N % PIPE_BN == 0 "N=$N must be divisible by $PIPE_BN"
    @assert K % PIPE_BK == 0 "K=$K must be divisible by $PIPE_BK"

    rng    = MersenneTwister(M * 7919 + N * 31 + K)
    A_f32  = quantize_bf16_p(0.1f0 .* randn(rng, Float32, M, K))
    BT_f32 = quantize_bf16_p(0.1f0 .* randn(rng, Float32, N, K))
    A_d   = CuArray(pack_bf16_rowmajor_p(A_f32))
    BT_d  = CuArray(pack_bf16_rowmajor_p(BT_f32))
    D_d   = CUDACore.zeros(Float32, M * N)

    @cuda blocks=(N ÷ PIPE_BN, M ÷ PIPE_BM) threads=PIPE_THREADS shmem=PIPE_SMEM_BYTES gemm_pipelined_kernel!(
        D_d, A_d, BT_d, Val(M), Val(N), Val(K))
    CUDACore.synchronize()

    D     = Matrix(reshape(Array(D_d), N, M)')
    D_ref = A_f32 * BT_f32'
    return D, D_ref
end

@testset "Ampere bf16 GEMM (pipelined: cp.async + 2-stage SMEM)" begin
    # numpy-allclose-style criterion: |d - r| ≤ atol + rtol·|r|.
    # A pure relative-error metric blows up wherever a D_ref element happens to
    # cancel near zero from random matmul accumulation — that's a metric
    # artifact, not a correctness failure. atol absorbs it.
    for (M, N, K) in [(64, 64, 16), (64, 64, 64), (64, 64, 256),
                      (128, 128, 128), (256, 256, 256), (512, 512, 512)]
        D, D_ref = run_pipelined_gemm(M, N, K)
        @test all(abs.(D .- D_ref) .<= 1f-2 .+ 1f-2 .* abs.(D_ref))
    end
end
