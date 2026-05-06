# REQUIRES CC 8.0
#
# Ported from pyptx/examples/ampere/gemm_highperf_ampere.py
# (https://github.com/patrick-toulme/pyptx).
# Copyright 2026 Patrick Toulmé. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Translated to Julia and adapted.
#
# Production-leaning A100 (sm_80) bf16 GEMM following the CUTLASS SM80 design:
#
#   - 128 × 128 × 32 CTA tile
#   - 4 warps/CTA arranged 2×2 in (M, N), each warp owns a 64×64 output sub-tile
#   - Per warp per K-iter: 4 M-frags × 8 N-frags × 2 K-blocks = 64 mma.sync
#   - ldmatrix.sync.aligned.m8n8.x4.shared.b16 for both A and B (no .trans;
#     SMEM holds B_T = (N, K) row-major so the no-trans output already matches
#     mma's row.col B layout — see pyptx note for the per-element bookkeeping)
#   - 4-stage cp.async ring buffer (CUTLASS kStages=3 in flight, 4 buffers)
#   - SMEM XOR swizzle: atom_eff = atom ^ ((row & 3))  (4-row period over 4
#     16-byte atoms per 64-byte K-row). Applied identically on cp.async writes
#     and ldmatrix reads.
#   - Cross-iter register double-buffering (ping-pong): bank 0 holds (ki, kb=0),
#     bank 1 holds (ki, kb=1); end of iter ki pre-loads (ki+1, kb=0) into bank 0
#     overlapped with bank-1 mma.
#
# Inputs (flat row-major flattened):
#   A    :: (M, K) bf16  stored as UInt16
#   B_T  :: (N, K) bf16  stored as UInt16  (B^T, K-contiguous for mma row.col)
#   D    :: (M, N) f32
#
# Grid:  (N/128, M/128)
# Block: (128, 1, 1)  — 4 warps per CTA
# Shmem: 64 KiB        — opt-in (above the default 48 KiB), so the launch site
#                        sets FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES.
#
# Constraint: K must be ≥ STAGES * BK = 128 (n_iters ≥ 4) so that the prologue
# can prime STAGES-1 = 3 stages without prefetching past the K boundary, and
# the cross-iter pipeline reaches steady state.

using Random
using Base: @nexprs

bf16_bits_h(x::Float32)  = UInt16(reinterpret(UInt32, x) >> 16)
bf16_to_f32_h(b::UInt16) = reinterpret(Float32, UInt32(b) << 16)

function pack_bf16_rowmajor_h(A::AbstractMatrix{Float32})
    rows, cols = size(A)
    out = Vector{UInt16}(undef, rows * cols)
    @inbounds for i in 1:rows, j in 1:cols
        out[(i-1)*cols + j] = bf16_bits_h(A[i, j])
    end
    out
end

quantize_bf16_h(A) = bf16_to_f32_h.(bf16_bits_h.(A))

const HP_BM, HP_BN, HP_BK = 128, 128, 32
const HP_NUM_WARPS = 4
const HP_THREADS   = 32 * HP_NUM_WARPS
const HP_WARPS_M   = 2
const HP_WARPS_N   = 2
const HP_WM        = HP_BM ÷ HP_WARPS_M     # 64
const HP_WN        = HP_BN ÷ HP_WARPS_N     # 64
const HP_N_FRAG_M  = HP_WM ÷ 16             # 4
const HP_N_FRAG_N  = HP_WN ÷ 8              # 8
const HP_N_FRAG_K  = HP_BK ÷ 16             # 2
const HP_STAGES    = 4
const HP_A_STAGE_BYTES = HP_BM * HP_BK * 2  # 8192
const HP_B_STAGE_BYTES = HP_BN * HP_BK * 2  # 8192
const HP_A_SMEM_BASE   = 0
const HP_B_SMEM_BASE   = HP_STAGES * HP_A_STAGE_BYTES         # 32768
const HP_SMEM_BYTES    = HP_STAGES * (HP_A_STAGE_BYTES + HP_B_STAGE_BYTES)  # 65536

function gemm_highperf_kernel!(
        D::CuDeviceVector{Float32},
        A::CuDeviceVector{UInt16},
        B_T::CuDeviceVector{UInt16},
        ::Val{M}, ::Val{N}, ::Val{K}) where {M, N, K}

    smem      = CuDynamicSharedArray(UInt8, HP_SMEM_BYTES)
    smem_base = pointer(smem)

    pa = pointer(A); pb = pointer(B_T); pd = pointer(D)
    n_iters = K ÷ HP_BK

    # CTA bases.
    m_base = Int(ptx"mov.u32"(sreg"ctaid.y")) * HP_BM
    n_base = Int(ptx"mov.u32"(sreg"ctaid.x")) * HP_BN

    # Thread + warp identity.
    tid     = ptx"mov.u32"(sreg"tid.x")
    warp_id = Int(tid >> UInt32(5))    # 0..3
    lane    = Int(tid & UInt32(31))    # 0..31
    warp_m  = warp_id >> 1             # 0..1 (warps 0,1 → M=0; 2,3 → M=1)
    warp_n  = warp_id & 1              # 0..1
    warp_m_smem = warp_m << 6          # 0 or 64 (warp's M base in SMEM rows)
    warp_n_smem = warp_n << 6          # 0 or 64

    # ----- cp.async layout: 4 passes × 32 rows × 4-thread row -----
    # Each thread writes one 16-byte atom (= 8 bf16) per pass per matrix.
    # 128 threads × 16 B = 2 KB/pass; 4 passes per stage per matrix.
    load_row_in_pass = Int(tid >> UInt32(2))   # 0..31 (1 row per 4 threads)
    load_col_chunk   = Int(tid & UInt32(3))    # 0..3  (which 8-bf16 chunk in K=32)
    load_col_off     = load_col_chunk << 3     # 0,8,16,24 (bf16 elements)

    # SMEM XOR swizzle (CUTLASS / MatmulTutorial v15 pattern).
    # 4 atoms per 64-byte K-row; 4-row period via 2 bits of the row index.
    # Same formula on writes (cp.async) and reads (ldmatrix).
    @inline _swiz(row::Int, atom::Int) = atom ⊻ (row & 3)

    # Precompute per-pass loop-invariants (per-thread). Hoists the row*64 +
    # atom*16 + (row & 3) XOR out of the K loop.
    cp_smem_offs = (
        let r = load_row_in_pass +  0; r * (HP_BK * 2) + _swiz(r, load_col_chunk) * 16; end,
        let r = load_row_in_pass + 32; r * (HP_BK * 2) + _swiz(r, load_col_chunk) * 16; end,
        let r = load_row_in_pass + 64; r * (HP_BK * 2) + _swiz(r, load_col_chunk) * 16; end,
        let r = load_row_in_pass + 96; r * (HP_BK * 2) + _swiz(r, load_col_chunk) * 16; end,
    )
    cp_a_row_x_K = (
        (m_base + load_row_in_pass +  0) * K,
        (m_base + load_row_in_pass + 32) * K,
        (m_base + load_row_in_pass + 64) * K,
        (m_base + load_row_in_pass + 96) * K,
    )
    cp_b_row_x_K = (
        (n_base + load_row_in_pass +  0) * K,
        (n_base + load_row_in_pass + 32) * K,
        (n_base + load_row_in_pass + 64) * K,
        (n_base + load_row_in_pass + 96) * K,
    )

    @inline function issue_cp_async(stage::Int, k_idx::Int)
        a_stage = smem_base + HP_A_SMEM_BASE + stage * HP_A_STAGE_BYTES
        b_stage = smem_base + HP_B_SMEM_BASE + stage * HP_B_STAGE_BYTES
        @nexprs 4 p -> begin
            a_dst = a_stage + cp_smem_offs[p]
            b_dst = b_stage + cp_smem_offs[p]
            a_off = (cp_a_row_x_K[p] + k_idx + load_col_off) * 2
            b_off = (cp_b_row_x_K[p] + k_idx + load_col_off) * 2
            ptx"cp.async.cg.shared.global"(a_dst, pa + a_off, Val(16))
            ptx"cp.async.cg.shared.global"(b_dst, pb + b_off, Val(16))
        end
        return nothing
    end

    # ----- ldmatrix lane-to-address math -----
    # A: per-lane row in [0..15], col-in-frag = (lane >> 4) << 3 ∈ {0, 8} bf16.
    # B (no trans): per-lane row-in-pair in [0..15], col-in-block = ((lane>>3)&1)*8.
    a_ldsm_row         = lane & 15
    a_atom_lane_part   = lane >> 4                          # 0 or 1
    b_ldsm_row_in_pair = ((lane >> 4) << 3) + (lane & 7)    # 0..15
    b_atom_lane_part   = (lane >> 3) & 1                    # 0 or 1

    @inline function a_off_b(mf::Int, kb::Int)
        row_for_mf = warp_m_smem + (mf << 4) + a_ldsm_row
        atom       = (kb << 1) + a_atom_lane_part
        atom_eff   = atom ⊻ (row_for_mf & 3)
        return row_for_mf * (HP_BK * 2) + atom_eff * 16
    end
    @inline function b_off_b(np::Int, kb::Int)
        row_for_np = warp_n_smem + (np << 4) + b_ldsm_row_in_pair
        atom       = (kb << 1) + b_atom_lane_part
        atom_eff   = atom ⊻ (row_for_np & 3)
        return row_for_np * (HP_BK * 2) + atom_eff * 16
    end
    @inline ldm_a(stage_base, mf::Int, kb::Int) =
        ptx"ldmatrix.sync.aligned.m8n8.x4.shared.b16"(stage_base + a_off_b(mf, kb))
    @inline ldm_b(stage_base, np::Int, kb::Int) =
        ptx"ldmatrix.sync.aligned.m8n8.x4.shared.b16"(stage_base + b_off_b(np, kb))

    # ----- Prologue: prime STAGES-1 = 3 stages -----
    @nexprs 3 s_p1 -> begin
        issue_cp_async(s_p1 - 1, (s_p1 - 1) * HP_BK)
        ptx"cp.async.commit_group"()
    end

    # ----- Accumulator: 4 mf × 8 nf NTuple{4, Float32} = 128 f32/lane -----
    @nexprs 4 mf -> @nexprs 8 nf -> (acc_mf_nf = (0f0, 0f0, 0f0, 0f0))

    # ----- Pre-load (ki=0, kb=0) into bank 0 -----
    ptx"cp.async.wait_group"(Val(HP_STAGES - 2))   # Val(2): wait until 2 groups remain
    ptx"bar.sync"(Val(0))

    a_stage0 = smem_base + HP_A_SMEM_BASE
    b_stage0 = smem_base + HP_B_SMEM_BASE
    a_bank_0 = (ldm_a(a_stage0, 0, 0), ldm_a(a_stage0, 1, 0),
                ldm_a(a_stage0, 2, 0), ldm_a(a_stage0, 3, 0))
    b_bank_0 = (ldm_b(b_stage0, 0, 0), ldm_b(b_stage0, 1, 0),
                ldm_b(b_stage0, 2, 0), ldm_b(b_stage0, 3, 0))

    # Bank 1 dummy init — type-stable placeholder, overwritten on first iter.
    a_bank_1 = a_bank_0
    b_bank_1 = b_bank_0

    # ----- Main K loop with cross-iter register double-buffering -----
    for ki in 0:n_iters-1
        stage = ki & (HP_STAGES - 1)
        a_stage_base = smem_base + HP_A_SMEM_BASE + stage * HP_A_STAGE_BYTES
        b_stage_base = smem_base + HP_B_SMEM_BASE + stage * HP_B_STAGE_BYTES

        # Bank 0 already holds (ki, kb=0). Pre-load (ki, kb=1) into bank 1.
        a_bank_1 = (ldm_a(a_stage_base, 0, 1), ldm_a(a_stage_base, 1, 1),
                    ldm_a(a_stage_base, 2, 1), ldm_a(a_stage_base, 3, 1))
        b_bank_1 = (ldm_b(b_stage_base, 0, 1), ldm_b(b_stage_base, 1, 1),
                    ldm_b(b_stage_base, 2, 1), ldm_b(b_stage_base, 3, 1))

        # mma using bank 0 (overlaps with bank-1 ldmatrix above).
        @nexprs 4 mf -> begin
            a_regs = a_bank_0[mf]
            @nexprs 8 nf -> begin
                _np_p1     = ((nf - 1) >> 1) + 1            # 1..4
                _nf_in_pr  = (nf - 1) & 1                   # 0 or 1
                _b_pair    = b_bank_0[_np_p1]
                _b_lo      = _b_pair[_nf_in_pr * 2 + 1]
                _b_hi      = _b_pair[_nf_in_pr * 2 + 2]
                acc_mf_nf  = ptx"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"(
                    a_regs, (_b_lo, _b_hi), acc_mf_nf)
            end
        end

        # Issue prefetch for ki + STAGES - 1 if there's K left.
        if ki + HP_STAGES - 1 < n_iters
            next_pf_stage = (ki + HP_STAGES - 1) & (HP_STAGES - 1)
            next_pf_k     = (ki + HP_STAGES - 1) * HP_BK
            issue_cp_async(next_pf_stage, next_pf_k)
            ptx"cp.async.commit_group"()
        end

        # Wait for next iter's stage + pre-load (ki+1, kb=0) into bank 0.
        if ki + 1 < n_iters
            # Three regions of wait_target depending on tail (= ki+1 - (n_iters-STAGES)).
            #   tail=0  → wait_group(STAGES-2)=Val(2)
            #   tail=1  → wait_group(STAGES-3)=Val(1)
            #   tail≥2 → wait_all
            if ki < n_iters - HP_STAGES
                ptx"cp.async.wait_group"(Val(HP_STAGES - 2))
            elseif ki < n_iters - HP_STAGES + 1
                ptx"cp.async.wait_group"(Val(HP_STAGES - 3))
            else
                ptx"cp.async.wait_all"()
            end
            ptx"bar.sync"(Val(0))

            next_stage_a = (ki + 1) & (HP_STAGES - 1)
            a_next = smem_base + HP_A_SMEM_BASE + next_stage_a * HP_A_STAGE_BYTES
            b_next = smem_base + HP_B_SMEM_BASE + next_stage_a * HP_B_STAGE_BYTES
            a_bank_0 = (ldm_a(a_next, 0, 0), ldm_a(a_next, 1, 0),
                        ldm_a(a_next, 2, 0), ldm_a(a_next, 3, 0))
            b_bank_0 = (ldm_b(b_next, 0, 0), ldm_b(b_next, 1, 0),
                        ldm_b(b_next, 2, 0), ldm_b(b_next, 3, 0))
        end

        # mma using bank 1 (overlaps with bank-0 cross-iter pre-load above).
        @nexprs 4 mf -> begin
            a_regs = a_bank_1[mf]
            @nexprs 8 nf -> begin
                _np_p1     = ((nf - 1) >> 1) + 1
                _nf_in_pr  = (nf - 1) & 1
                _b_pair    = b_bank_1[_np_p1]
                _b_lo      = _b_pair[_nf_in_pr * 2 + 1]
                _b_hi      = _b_pair[_nf_in_pr * 2 + 2]
                acc_mf_nf  = ptx"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"(
                    a_regs, (_b_lo, _b_hi), acc_mf_nf)
            end
        end
    end

    # ----- Epilogue: write 4 mf × 8 nf × 4 f32 acc tiles to global D -----
    # m16n8 fragment per-lane: rows {gid, gid+8} × cols {2*tig, 2*tig+1}.
    gid = lane >> 2
    tig = lane & 0x3
    col_lo = tig << 1
    warp_m_global = m_base + warp_m_smem
    warp_n_global = n_base + warp_n_smem
    @nexprs 4 mf -> begin
        m_frag_global = warp_m_global + ((mf - 1) << 4)   # mf=1..4 → 0,16,32,48
        row_lo = m_frag_global + gid
        row_hi = row_lo + 8
        @nexprs 8 nf -> begin
            n_frag_global = warp_n_global + ((nf - 1) << 3)
            d_col_base    = n_frag_global + col_lo
            ptx"st.global.f32"(pd + (row_lo * N + (d_col_base + 0)) * 4, acc_mf_nf[1])
            ptx"st.global.f32"(pd + (row_lo * N + (d_col_base + 1)) * 4, acc_mf_nf[2])
            ptx"st.global.f32"(pd + (row_hi * N + (d_col_base + 0)) * 4, acc_mf_nf[3])
            ptx"st.global.f32"(pd + (row_hi * N + (d_col_base + 1)) * 4, acc_mf_nf[4])
        end
    end

    return nothing
end

function run_highperf_gemm(M::Int, N::Int, K::Int)
    @assert M % HP_BM == 0 "M=$M must be divisible by $HP_BM"
    @assert N % HP_BN == 0 "N=$N must be divisible by $HP_BN"
    @assert K % HP_BK == 0 "K=$K must be divisible by $HP_BK"
    @assert K ÷ HP_BK >= HP_STAGES "K=$K too small: K÷BK=$(K÷HP_BK) < STAGES=$HP_STAGES"

    rng    = MersenneTwister(M * 7919 + N * 31 + K)
    A_f32  = quantize_bf16_h(0.1f0 .* randn(rng, Float32, M, K))
    BT_f32 = quantize_bf16_h(0.1f0 .* randn(rng, Float32, N, K))
    A_d   = CuArray(pack_bf16_rowmajor_h(A_f32))
    BT_d  = CuArray(pack_bf16_rowmajor_h(BT_f32))
    D_d   = CUDACore.zeros(Float32, M * N)

    # 64 KiB shmem exceeds the default 48 KiB cap; opt in via the function
    # attribute before launching.
    kernel = @cuda launch=false gemm_highperf_kernel!(
        D_d, A_d, BT_d, Val(M), Val(N), Val(K))
    attrs = CUDACore.attributes(kernel.fun)
    attrs[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = HP_SMEM_BYTES
    kernel(D_d, A_d, BT_d, Val(M), Val(N), Val(K);
           blocks=(N ÷ HP_BN, M ÷ HP_BM), threads=HP_THREADS, shmem=HP_SMEM_BYTES)
    CUDACore.synchronize()

    D     = Matrix(reshape(Array(D_d), N, M)')
    D_ref = A_f32 * BT_f32'
    return D, D_ref
end

@testset "Ampere bf16 GEMM (highperf: 4-stage cp.async + ldmatrix.x4 + swizzle)" begin
    # Test sizes — minimum K = STAGES * BK = 128 (n_iters ≥ 4).
    for (M, N, K) in [(128, 128, 128), (128, 128, 256), (128, 128, 512),
                      (256, 256, 256), (512, 512, 512)]
        D, D_ref = run_highperf_gemm(M, N, K)
        # numpy-allclose-style: |d - r| ≤ atol + rtol·|r|.
        @test all(abs.(D .- D_ref) .<= 1f-2 .+ 1f-2 .* abs.(D_ref))
    end
end
