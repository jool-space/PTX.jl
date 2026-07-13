# TEST_TARGET: requires=gpu evidence=runtime target=sm_80
#
# Variant of test/gpu/ampere_gemm_highperf.jl with a CUTLASS-style log2
# ThreadBlock swizzle (grid rasterization) applied at the top of the kernel.
# Same compute, same SMEM/register pipeline; only the (ctaid.x, ctaid.y)→
# (n_block, m_block) mapping changes.
#
# Goal: improve L2 reuse on memory-bound shapes by scheduling GROUP =
# 2^LOG_GROUP consecutive CTAs to share the same N tile of B. Default raster
# order spreads simultaneously-running CTAs across many N tiles → B's
# L2 footprint blows up for large M·N. Grouping packs them so that B[n_block]
# is loaded once per group rather than once per CTA.
#
# Launch grid:
#   blocks.x = (N/BN) * GROUP
#   blocks.y = (M/BM) / GROUP
# Kernel-side remap (matches CUTLASS GemmIdentityThreadblockSwizzle):
#   n_block = ctaid.x >> LOG_GROUP
#   m_block = (ctaid.y << LOG_GROUP) + (ctaid.x & (GROUP - 1))
#
# For square M=N grids, GROUP is picked at host-side as min(8, GY) and
# requires GY % GROUP == 0. Test sizes pick the largest valid power of 2.

using Random
using Base: @nexprs

bf16_bits_hs(x::Float32)  = UInt16(reinterpret(UInt32, x) >> 16)
bf16_to_f32_hs(b::UInt16) = reinterpret(Float32, UInt32(b) << 16)

function pack_bf16_rowmajor_hs(A::AbstractMatrix{Float32})
    rows, cols = size(A)
    out = Vector{UInt16}(undef, rows * cols)
    @inbounds for i in 1:rows, j in 1:cols
        out[(i-1)*cols + j] = bf16_bits_hs(A[i, j])
    end
    out
end

quantize_bf16_hs(A) = bf16_to_f32_hs.(bf16_bits_hs.(A))

const HPS_BM, HPS_BN, HPS_BK = 128, 128, 32
const HPS_THREADS = 128
const HPS_STAGES  = 4
const HPS_A_STAGE_BYTES = HPS_BM * HPS_BK * 2
const HPS_B_STAGE_BYTES = HPS_BN * HPS_BK * 2
const HPS_A_SMEM_BASE   = 0
const HPS_B_SMEM_BASE   = HPS_STAGES * HPS_A_STAGE_BYTES
const HPS_SMEM_BYTES    = HPS_STAGES * (HPS_A_STAGE_BYTES + HPS_B_STAGE_BYTES)

function gemm_highperf_swizzled_kernel!(
        D::CuDeviceVector{Float32},
        A::CuDeviceVector{UInt16},
        B_T::CuDeviceVector{UInt16},
        ::Val{M}, ::Val{N}, ::Val{K},
        ::Val{LOG_GROUP}) where {M, N, K, LOG_GROUP}

    smem      = CuDynamicSharedArray(UInt8, HPS_SMEM_BYTES)
    smem_base = pointer(smem)
    pa = pointer(A); pb = pointer(B_T); pd = pointer(D)
    n_iters = K ÷ HPS_BK

    # ----- CUTLASS-style log2 ThreadBlock swizzle -----
    # Launch grid is widened on X by 2^LOG_GROUP and narrowed on Y by the same
    # factor. The bottom LOG_GROUP bits of ctaid.x become the within-group M
    # offset; the rest is the N tile.
    ctaid_x = Int(ptx"mov.u32"(sreg"ctaid.x"))
    ctaid_y = Int(ptx"mov.u32"(sreg"ctaid.y"))
    n_block = ctaid_x >> LOG_GROUP
    m_block = (ctaid_y << LOG_GROUP) + (ctaid_x & ((1 << LOG_GROUP) - 1))

    m_base = m_block * HPS_BM
    n_base = n_block * HPS_BN

    # Thread + warp identity (unchanged from highperf).
    tid     = ptx"mov.u32"(sreg"tid.x")
    warp_id = Int(tid >> UInt32(5))
    lane    = Int(tid & UInt32(31))
    warp_m  = warp_id >> 1
    warp_n  = warp_id & 1
    warp_m_smem = warp_m << 6
    warp_n_smem = warp_n << 6

    load_row_in_pass = Int(tid >> UInt32(2))
    load_col_chunk   = Int(tid & UInt32(3))
    load_col_off     = load_col_chunk << 3

    @inline _swiz(row::Int, atom::Int) = atom ⊻ (row & 3)

    cp_smem_offs = (
        let r = load_row_in_pass +  0; r * (HPS_BK * 2) + _swiz(r, load_col_chunk) * 16; end,
        let r = load_row_in_pass + 32; r * (HPS_BK * 2) + _swiz(r, load_col_chunk) * 16; end,
        let r = load_row_in_pass + 64; r * (HPS_BK * 2) + _swiz(r, load_col_chunk) * 16; end,
        let r = load_row_in_pass + 96; r * (HPS_BK * 2) + _swiz(r, load_col_chunk) * 16; end,
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
        a_stage = smem_base + HPS_A_SMEM_BASE + stage * HPS_A_STAGE_BYTES
        b_stage = smem_base + HPS_B_SMEM_BASE + stage * HPS_B_STAGE_BYTES
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

    a_ldsm_row         = lane & 15
    a_atom_lane_part   = lane >> 4
    b_ldsm_row_in_pair = ((lane >> 4) << 3) + (lane & 7)
    b_atom_lane_part   = (lane >> 3) & 1

    @inline function a_off_b(mf::Int, kb::Int)
        row_for_mf = warp_m_smem + (mf << 4) + a_ldsm_row
        atom_eff   = ((kb << 1) + a_atom_lane_part) ⊻ (row_for_mf & 3)
        return row_for_mf * (HPS_BK * 2) + atom_eff * 16
    end
    @inline function b_off_b(np::Int, kb::Int)
        row_for_np = warp_n_smem + (np << 4) + b_ldsm_row_in_pair
        atom_eff   = ((kb << 1) + b_atom_lane_part) ⊻ (row_for_np & 3)
        return row_for_np * (HPS_BK * 2) + atom_eff * 16
    end
    @inline ldm_a(stage_base, mf::Int, kb::Int) =
        ptx"ldmatrix.sync.aligned.m8n8.x4.shared.b16"(stage_base + a_off_b(mf, kb))
    @inline ldm_b(stage_base, np::Int, kb::Int) =
        ptx"ldmatrix.sync.aligned.m8n8.x4.shared.b16"(stage_base + b_off_b(np, kb))

    @nexprs 3 s_p1 -> begin
        issue_cp_async(s_p1 - 1, (s_p1 - 1) * HPS_BK)
        ptx"cp.async.commit_group"()
    end

    @nexprs 4 mf -> @nexprs 8 nf -> (acc_mf_nf = (0f0, 0f0, 0f0, 0f0))

    ptx"cp.async.wait_group"(Val(HPS_STAGES - 2))
    ptx"bar.sync"(Val(0))

    a_stage0 = smem_base + HPS_A_SMEM_BASE
    b_stage0 = smem_base + HPS_B_SMEM_BASE
    a_bank_0 = (ldm_a(a_stage0, 0, 0), ldm_a(a_stage0, 1, 0),
                ldm_a(a_stage0, 2, 0), ldm_a(a_stage0, 3, 0))
    b_bank_0 = (ldm_b(b_stage0, 0, 0), ldm_b(b_stage0, 1, 0),
                ldm_b(b_stage0, 2, 0), ldm_b(b_stage0, 3, 0))
    a_bank_1 = a_bank_0
    b_bank_1 = b_bank_0

    for ki in 0:n_iters-1
        stage = ki & (HPS_STAGES - 1)
        a_stage_base = smem_base + HPS_A_SMEM_BASE + stage * HPS_A_STAGE_BYTES
        b_stage_base = smem_base + HPS_B_SMEM_BASE + stage * HPS_B_STAGE_BYTES

        a_bank_1 = (ldm_a(a_stage_base, 0, 1), ldm_a(a_stage_base, 1, 1),
                    ldm_a(a_stage_base, 2, 1), ldm_a(a_stage_base, 3, 1))
        b_bank_1 = (ldm_b(b_stage_base, 0, 1), ldm_b(b_stage_base, 1, 1),
                    ldm_b(b_stage_base, 2, 1), ldm_b(b_stage_base, 3, 1))

        @nexprs 4 mf -> begin
            a_regs = a_bank_0[mf]
            @nexprs 8 nf -> begin
                _np_p1     = ((nf - 1) >> 1) + 1
                _nf_in_pr  = (nf - 1) & 1
                _b_pair    = b_bank_0[_np_p1]
                _b_lo      = _b_pair[_nf_in_pr * 2 + 1]
                _b_hi      = _b_pair[_nf_in_pr * 2 + 2]
                acc_mf_nf  = ptx"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"(
                    a_regs, (_b_lo, _b_hi), acc_mf_nf)
            end
        end

        if ki + HPS_STAGES - 1 < n_iters
            next_pf_stage = (ki + HPS_STAGES - 1) & (HPS_STAGES - 1)
            next_pf_k     = (ki + HPS_STAGES - 1) * HPS_BK
            issue_cp_async(next_pf_stage, next_pf_k)
            ptx"cp.async.commit_group"()
        end

        if ki + 1 < n_iters
            if ki < n_iters - HPS_STAGES
                ptx"cp.async.wait_group"(Val(HPS_STAGES - 2))
            elseif ki < n_iters - HPS_STAGES + 1
                ptx"cp.async.wait_group"(Val(HPS_STAGES - 3))
            else
                ptx"cp.async.wait_all"()
            end
            ptx"bar.sync"(Val(0))

            next_stage_a = (ki + 1) & (HPS_STAGES - 1)
            a_next = smem_base + HPS_A_SMEM_BASE + next_stage_a * HPS_A_STAGE_BYTES
            b_next = smem_base + HPS_B_SMEM_BASE + next_stage_a * HPS_B_STAGE_BYTES
            a_bank_0 = (ldm_a(a_next, 0, 0), ldm_a(a_next, 1, 0),
                        ldm_a(a_next, 2, 0), ldm_a(a_next, 3, 0))
            b_bank_0 = (ldm_b(b_next, 0, 0), ldm_b(b_next, 1, 0),
                        ldm_b(b_next, 2, 0), ldm_b(b_next, 3, 0))
        end

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

    gid = lane >> 2
    tig = lane & 0x3
    col_lo = tig << 1
    warp_m_global = m_base + warp_m_smem
    warp_n_global = n_base + warp_n_smem
    @nexprs 4 mf -> begin
        m_frag_global = warp_m_global + ((mf - 1) << 4)
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

# Largest LOG_GROUP in 0..3 such that GY (= M/BM) is divisible by 2^LOG_GROUP.
function pick_log_group(GY::Int)
    log = 0
    for L in 1:3
        GY % (1 << L) == 0 || break
        log = L
    end
    log
end

function run_highperf_swizzled_gemm(M::Int, N::Int, K::Int)
    @assert M % HPS_BM == 0 "M=$M must be divisible by $HPS_BM"
    @assert N % HPS_BN == 0 "N=$N must be divisible by $HPS_BN"
    @assert K % HPS_BK == 0 "K=$K must be divisible by $HPS_BK"
    @assert K ÷ HPS_BK >= HPS_STAGES "K=$K too small: K÷BK=$(K÷HPS_BK) < STAGES=$HPS_STAGES"

    GX = N ÷ HPS_BN
    GY = M ÷ HPS_BM
    LOG_GROUP = pick_log_group(GY)
    GROUP = 1 << LOG_GROUP

    rng    = MersenneTwister(M * 7919 + N * 31 + K)
    A_f32  = quantize_bf16_hs(0.1f0 .* randn(rng, Float32, M, K))
    BT_f32 = quantize_bf16_hs(0.1f0 .* randn(rng, Float32, N, K))
    A_d   = CuArray(pack_bf16_rowmajor_hs(A_f32))
    BT_d  = CuArray(pack_bf16_rowmajor_hs(BT_f32))
    D_d   = CUDACore.zeros(Float32, M * N)

    kernel = @cuda launch=false gemm_highperf_swizzled_kernel!(
        D_d, A_d, BT_d, Val(M), Val(N), Val(K), Val(LOG_GROUP))
    attrs = CUDACore.attributes(kernel.fun)
    attrs[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = HPS_SMEM_BYTES
    kernel(D_d, A_d, BT_d, Val(M), Val(N), Val(K), Val(LOG_GROUP);
           blocks=(GX * GROUP, GY ÷ GROUP), threads=HPS_THREADS, shmem=HPS_SMEM_BYTES)
    CUDACore.synchronize()

    D     = Matrix(reshape(Array(D_d), N, M)')
    D_ref = A_f32 * BT_f32'
    return D, D_ref, LOG_GROUP
end

@testset "Ampere bf16 GEMM (highperf + log2 ThreadBlock swizzle)" begin
    for (M, N, K) in [(128, 128, 128), (128, 128, 256), (128, 128, 512),
                      (256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]
        D, D_ref, log_group = run_highperf_swizzled_gemm(M, N, K)
        @test all(abs.(D .- D_ref) .<= 1f-2 .+ 1f-2 .* abs.(D_ref))
    end
end
