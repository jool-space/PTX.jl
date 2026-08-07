# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.9
#
# Pipelined fp8 GEMM on the CUTLASS SM80 design, retiled for the 1-byte
# dtype from gpu/ampere/gemm_highperf.jl (same 4-stage cp.async ring, XOR
# swizzle, ldmatrix.x4, cross-iter register double-buffering — see that
# file for the pipeline commentary). Every instruction is kind-less
# sm_89-floor PTX with no `a`/`f` suffix, so one compilation covers Ada
# and everything after it.
#
#   - 128 × 128 × 64 CTA tile: BK=64 bytes keeps the 64-byte SMEM K-row
#     of the bf16 kernel (4 × 16-byte atoms, same swizzle constants), and
#     one K-iter feeds two m16n8k32 K-blocks exactly as bf16 fed two
#     m16n8k16 blocks.
#   - ldmatrix.sync.aligned.m8n8.x4.shared.b16 on byte data: a b16 lane
#     pair is 4 consecutive bytes, so the loaded register is precisely the
#     m16n8k32 8-bit fragment (PTX 9.3 §9.7.15.5.10 — 4 elements along K
#     per .b32, row = lane>>2, byte-col = 4*(lane&3)). No `.b8` ldmatrix
#     needed (that token is sm_100a+).
#   - Two knobs, both compile-time: A/B storage formats (e4m3/e5m2 in any
#     combination — the kind-less grammar types the operands
#     independently) and accumulator precision CT ∈ {Float32, Float16}.
#     f32 accumulate is the arch-consistent path (half tensor rate on
#     Ada-class silicon); f16 accumulate is Ada's full-rate path with the
#     matching precision loss. D is f32 either way.
#
# Inputs (flat row-major flattened):
#   A    :: (M, K) fp8 bytes, row-major
#   B_T  :: (N, K) fp8 bytes, row-major  (B^T, K-contiguous for mma row.col)
#   D    :: (M, N) f32
#
# Grid:  (N/128, M/128); Block: 128 threads (4 warps, 2×2)
# Shmem: 64 KiB dynamic — opt-in above the 48 KiB default.
# Constraint: K ≥ STAGES * BK = 256.

using Random
using Microfloats
using Base: @nexprs

const FP8P_BM, FP8P_BN, FP8P_BK = 128, 128, 64
const FP8P_NUM_WARPS = 4
const FP8P_THREADS   = 32 * FP8P_NUM_WARPS
const FP8P_STAGES    = 4
const FP8P_A_STAGE_BYTES = FP8P_BM * FP8P_BK        # 8192
const FP8P_B_STAGE_BYTES = FP8P_BN * FP8P_BK        # 8192
const FP8P_A_SMEM_BASE   = 0
const FP8P_B_SMEM_BASE   = FP8P_STAGES * FP8P_A_STAGE_BYTES
const FP8P_SMEM_BYTES    = FP8P_STAGES * (FP8P_A_STAGE_BYTES + FP8P_B_STAGE_BYTES)

# The two compile-time knobs, closed over 8 opcode literals: accumulator
# carrier (NTuple{4, Float32} vs packed-f16x2 NTuple{2, UInt32}) × A/B
# format pair.
@inline _fp8p_zero(::Type{Float32}) = (0f0, 0f0, 0f0, 0f0)
@inline _fp8p_zero(::Type{Float16}) = (UInt32(0), UInt32(0))

@inline _fp8p_mma(::Type{Float32}, ::Val{:e4m3}, ::Val{:e4m3}, a, b, c) =
    ptx"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"(a, b, c)
@inline _fp8p_mma(::Type{Float32}, ::Val{:e4m3}, ::Val{:e5m2}, a, b, c) =
    ptx"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32"(a, b, c)
@inline _fp8p_mma(::Type{Float32}, ::Val{:e5m2}, ::Val{:e4m3}, a, b, c) =
    ptx"mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32"(a, b, c)
@inline _fp8p_mma(::Type{Float32}, ::Val{:e5m2}, ::Val{:e5m2}, a, b, c) =
    ptx"mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32"(a, b, c)
@inline _fp8p_mma(::Type{Float16}, ::Val{:e4m3}, ::Val{:e4m3}, a, b, c) =
    ptx"mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16"(a, b, c)
@inline _fp8p_mma(::Type{Float16}, ::Val{:e4m3}, ::Val{:e5m2}, a, b, c) =
    ptx"mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e5m2.f16"(a, b, c)
@inline _fp8p_mma(::Type{Float16}, ::Val{:e5m2}, ::Val{:e4m3}, a, b, c) =
    ptx"mma.sync.aligned.m16n8k32.row.col.f16.e5m2.e4m3.f16"(a, b, c)
@inline _fp8p_mma(::Type{Float16}, ::Val{:e5m2}, ::Val{:e5m2}, a, b, c) =
    ptx"mma.sync.aligned.m16n8k32.row.col.f16.e5m2.e5m2.f16"(a, b, c)

# Epilogue store of one m16n8 fragment; D is f32 for both accumulators.
@inline function _fp8p_store_frag(pd, n::Int, row_lo::Int, row_hi::Int,
                                  col::Int, acc::NTuple{4, Float32})
    ptx"st.global.f32"(pd + (row_lo * n + col    ) * 4, acc[1])
    ptx"st.global.f32"(pd + (row_lo * n + col + 1) * 4, acc[2])
    ptx"st.global.f32"(pd + (row_hi * n + col    ) * 4, acc[3])
    ptx"st.global.f32"(pd + (row_hi * n + col + 1) * 4, acc[4])
    return nothing
end
@inline _f16_lo(u::UInt32) = Float32(reinterpret(Float16, u % UInt16))
@inline _f16_hi(u::UInt32) = Float32(reinterpret(Float16, (u >> 16) % UInt16))
@inline function _fp8p_store_frag(pd, n::Int, row_lo::Int, row_hi::Int,
                                  col::Int, acc::NTuple{2, UInt32})
    ptx"st.global.f32"(pd + (row_lo * n + col    ) * 4, _f16_lo(acc[1]))
    ptx"st.global.f32"(pd + (row_lo * n + col + 1) * 4, _f16_hi(acc[1]))
    ptx"st.global.f32"(pd + (row_hi * n + col    ) * 4, _f16_lo(acc[2]))
    ptx"st.global.f32"(pd + (row_hi * n + col + 1) * 4, _f16_hi(acc[2]))
    return nothing
end

function gemm_fp8_pipelined_kernel!(
        D::CuDeviceVector{Float32},
        A::CuDeviceVector{UInt8},
        B_T::CuDeviceVector{UInt8},
        ::Val{M}, ::Val{N}, ::Val{K},
        ::Type{CT}, af::Val, bf::Val) where {M, N, K, CT}

    smem      = CuDynamicSharedArray(UInt8, FP8P_SMEM_BYTES)
    smem_base = pointer(smem)

    pa = pointer(A); pb = pointer(B_T); pd = pointer(D)
    n_iters = K ÷ FP8P_BK

    m_base = Int(ptx"mov.u32"(sreg"ctaid.y")) * FP8P_BM
    n_base = Int(ptx"mov.u32"(sreg"ctaid.x")) * FP8P_BN

    tid     = ptx"mov.u32"(sreg"tid.x")
    warp_id = Int(tid >> UInt32(5))
    lane    = Int(tid & UInt32(31))
    warp_m_smem = (warp_id >> 1) << 6           # 0 or 64
    warp_n_smem = (warp_id & 1) << 6            # 0 or 64

    # cp.async: 4 passes × 32 rows × 4-thread row; one 16-byte atom
    # (= 16 fp8) per thread per pass per matrix.
    load_row_in_pass = Int(tid >> UInt32(2))    # 0..31
    load_col_chunk   = Int(tid & UInt32(3))     # 0..3
    load_col_off     = load_col_chunk << 4      # 0,16,32,48 (elements)

    # Same XOR swizzle as the bf16 kernel: the K-row is 64 bytes = 4 atoms
    # for BK=64 fp8 exactly as for BK=32 bf16.
    @inline _swiz(row::Int, atom::Int) = atom ⊻ (row & 3)

    cp_smem_offs = (
        let r = load_row_in_pass +  0; r * FP8P_BK + _swiz(r, load_col_chunk) * 16; end,
        let r = load_row_in_pass + 32; r * FP8P_BK + _swiz(r, load_col_chunk) * 16; end,
        let r = load_row_in_pass + 64; r * FP8P_BK + _swiz(r, load_col_chunk) * 16; end,
        let r = load_row_in_pass + 96; r * FP8P_BK + _swiz(r, load_col_chunk) * 16; end,
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
        a_stage = smem_base + FP8P_A_SMEM_BASE + stage * FP8P_A_STAGE_BYTES
        b_stage = smem_base + FP8P_B_SMEM_BASE + stage * FP8P_B_STAGE_BYTES
        @nexprs 4 p -> begin
            a_dst = a_stage + cp_smem_offs[p]
            b_dst = b_stage + cp_smem_offs[p]
            a_off = cp_a_row_x_K[p] + k_idx + load_col_off
            b_off = cp_b_row_x_K[p] + k_idx + load_col_off
            ptx"cp.async.cg.shared.global"(a_dst, pa + a_off, Val(16))
            ptx"cp.async.cg.shared.global"(b_dst, pb + b_off, Val(16))
        end
        return nothing
    end

    # ldmatrix lane math — byte-identical to the bf16 kernel (16-byte
    # atoms, 64-byte rows); the loaded registers are fp8 k32 fragments.
    a_ldsm_row         = lane & 15
    a_atom_lane_part   = lane >> 4
    b_ldsm_row_in_pair = ((lane >> 4) << 3) + (lane & 7)
    b_atom_lane_part   = (lane >> 3) & 1

    @inline function a_off_b(mf::Int, kb::Int)
        row_for_mf = warp_m_smem + (mf << 4) + a_ldsm_row
        atom_eff   = _swiz(row_for_mf, (kb << 1) + a_atom_lane_part)
        return row_for_mf * FP8P_BK + atom_eff * 16
    end
    @inline function b_off_b(np::Int, kb::Int)
        row_for_np = warp_n_smem + (np << 4) + b_ldsm_row_in_pair
        atom_eff   = _swiz(row_for_np, (kb << 1) + b_atom_lane_part)
        return row_for_np * FP8P_BK + atom_eff * 16
    end
    @inline ldm_a(stage_base, mf::Int, kb::Int) =
        ptx"ldmatrix.sync.aligned.m8n8.x4.shared.b16"(stage_base + a_off_b(mf, kb))
    @inline ldm_b(stage_base, np::Int, kb::Int) =
        ptx"ldmatrix.sync.aligned.m8n8.x4.shared.b16"(stage_base + b_off_b(np, kb))

    # Prologue: prime STAGES-1 stages.
    @nexprs 3 s_p1 -> begin
        issue_cp_async(s_p1 - 1, (s_p1 - 1) * FP8P_BK)
        ptx"cp.async.commit_group"()
    end

    @nexprs 4 mf -> @nexprs 8 nf -> (acc_mf_nf = _fp8p_zero(CT))

    ptx"cp.async.wait_group"(Val(FP8P_STAGES - 2))
    ptx"bar.sync"(Val(0))

    a_stage0 = smem_base + FP8P_A_SMEM_BASE
    b_stage0 = smem_base + FP8P_B_SMEM_BASE
    a_bank_0 = (ldm_a(a_stage0, 0, 0), ldm_a(a_stage0, 1, 0),
                ldm_a(a_stage0, 2, 0), ldm_a(a_stage0, 3, 0))
    b_bank_0 = (ldm_b(b_stage0, 0, 0), ldm_b(b_stage0, 1, 0),
                ldm_b(b_stage0, 2, 0), ldm_b(b_stage0, 3, 0))
    a_bank_1 = a_bank_0
    b_bank_1 = b_bank_0

    for ki in 0:n_iters-1
        stage = ki & (FP8P_STAGES - 1)
        a_stage_base = smem_base + FP8P_A_SMEM_BASE + stage * FP8P_A_STAGE_BYTES
        b_stage_base = smem_base + FP8P_B_SMEM_BASE + stage * FP8P_B_STAGE_BYTES

        a_bank_1 = (ldm_a(a_stage_base, 0, 1), ldm_a(a_stage_base, 1, 1),
                    ldm_a(a_stage_base, 2, 1), ldm_a(a_stage_base, 3, 1))
        b_bank_1 = (ldm_b(b_stage_base, 0, 1), ldm_b(b_stage_base, 1, 1),
                    ldm_b(b_stage_base, 2, 1), ldm_b(b_stage_base, 3, 1))

        @nexprs 4 mf -> begin
            a_regs = a_bank_0[mf]
            @nexprs 8 nf -> begin
                _np_p1    = ((nf - 1) >> 1) + 1
                _nf_in_pr = (nf - 1) & 1
                _b_pair   = b_bank_0[_np_p1]
                acc_mf_nf = _fp8p_mma(CT, af, bf, a_regs,
                    (_b_pair[_nf_in_pr * 2 + 1], _b_pair[_nf_in_pr * 2 + 2]),
                    acc_mf_nf)
            end
        end

        if ki + FP8P_STAGES - 1 < n_iters
            next_pf_stage = (ki + FP8P_STAGES - 1) & (FP8P_STAGES - 1)
            issue_cp_async(next_pf_stage, (ki + FP8P_STAGES - 1) * FP8P_BK)
            ptx"cp.async.commit_group"()
        end

        if ki + 1 < n_iters
            if ki < n_iters - FP8P_STAGES
                ptx"cp.async.wait_group"(Val(FP8P_STAGES - 2))
            elseif ki < n_iters - FP8P_STAGES + 1
                ptx"cp.async.wait_group"(Val(FP8P_STAGES - 3))
            else
                ptx"cp.async.wait_all"()
            end
            ptx"bar.sync"(Val(0))

            next_stage = (ki + 1) & (FP8P_STAGES - 1)
            a_next = smem_base + FP8P_A_SMEM_BASE + next_stage * FP8P_A_STAGE_BYTES
            b_next = smem_base + FP8P_B_SMEM_BASE + next_stage * FP8P_B_STAGE_BYTES
            a_bank_0 = (ldm_a(a_next, 0, 0), ldm_a(a_next, 1, 0),
                        ldm_a(a_next, 2, 0), ldm_a(a_next, 3, 0))
            b_bank_0 = (ldm_b(b_next, 0, 0), ldm_b(b_next, 1, 0),
                        ldm_b(b_next, 2, 0), ldm_b(b_next, 3, 0))
        end

        @nexprs 4 mf -> begin
            a_regs = a_bank_1[mf]
            @nexprs 8 nf -> begin
                _np_p1    = ((nf - 1) >> 1) + 1
                _nf_in_pr = (nf - 1) & 1
                _b_pair   = b_bank_1[_np_p1]
                acc_mf_nf = _fp8p_mma(CT, af, bf, a_regs,
                    (_b_pair[_nf_in_pr * 2 + 1], _b_pair[_nf_in_pr * 2 + 2]),
                    acc_mf_nf)
            end
        end
    end

    # Epilogue — m16n8 fragment rows {gid, gid+8}, cols {2 tig, 2 tig + 1}.
    gid = lane >> 2
    tig = lane & 0x3
    col_lo = tig << 1
    warp_m_global = m_base + warp_m_smem
    warp_n_global = n_base + warp_n_smem
    @nexprs 4 mf -> begin
        row_lo = warp_m_global + ((mf - 1) << 4) + gid
        row_hi = row_lo + 8
        @nexprs 8 nf -> begin
            d_col = warp_n_global + ((nf - 1) << 3) + col_lo
            _fp8p_store_frag(pd, N, row_lo, row_hi, d_col, acc_mf_nf)
        end
    end

    return nothing
end

# --- host side --------------------------------------------------------------

_fp8p_type(::Val{:e4m3}) = Float8_E4M3FN
_fp8p_type(::Val{:e5m2}) = Float8_E5M2

_flat_rowmajor_bytes(M8::AbstractMatrix) =
    reinterpret(UInt8, vec(permutedims(M8)))

function run_fp8_pipelined_gemm(M::Int, N::Int, K::Int, CT::Type,
                                af::Val, bf::Val;
                                A_f32::Matrix{Float32}, BT_f32::Matrix{Float32})
    @assert M % FP8P_BM == 0 && N % FP8P_BN == 0 && K % FP8P_BK == 0
    @assert K ÷ FP8P_BK >= FP8P_STAGES

    A8   = _fp8p_type(af).(A_f32)
    BT8  = _fp8p_type(bf).(BT_f32)
    A_d  = CuArray(_flat_rowmajor_bytes(A8))
    BT_d = CuArray(_flat_rowmajor_bytes(BT8))
    D_d  = CUDACore.zeros(Float32, M * N)

    kernel = @cuda launch=false gemm_fp8_pipelined_kernel!(
        D_d, A_d, BT_d, Val(M), Val(N), Val(K), CT, af, bf)
    attrs = CUDACore.attributes(kernel.fun)
    attrs[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = FP8P_SMEM_BYTES
    kernel(D_d, A_d, BT_d, Val(M), Val(N), Val(K), CT, af, bf;
           blocks=(N ÷ FP8P_BN, M ÷ FP8P_BM), threads=FP8P_THREADS,
           shmem=FP8P_SMEM_BYTES)
    CUDACore.synchronize()

    D = Matrix(reshape(Array(D_d), N, M)')
    # Float64 reference over the quantized operands: the exact product the
    # device should approach; only accumulator precision separates them.
    D_ref = Float64.(Float32.(A8)) * Float64.(Float32.(BT8))'
    return D, D_ref
end

# Small integers are exact in both formats (e4m3: 3 mantissa bits; e5m2: 2
# — 1..4 are all representable); with them every product and partial sum
# is an integer, so a correct kernel is bit-exact against the reference in
# f32, and in f16 too while sums stay ≤ 2048.

function _fp8p_int_inputs(rng, M, N, K, hi)
    A  = Float32.(rand(rng, 1:hi, M, K))
    BT = Float32.(rand(rng, 1:hi, N, K))
    A, BT
end

@testset "fp8 pipelined GEMM: exact small-integer product, f32 acc" begin
    rng = MersenneTwister(0x8905)
    for (af, bf) in ((Val(:e4m3), Val(:e4m3)), (Val(:e4m3), Val(:e5m2)),
                     (Val(:e5m2), Val(:e4m3)), (Val(:e5m2), Val(:e5m2)))
        M, N, K = 128, 128, 512
        A_f32, BT_f32 = _fp8p_int_inputs(rng, M, N, K, 4)
        D, D_ref = run_fp8_pipelined_gemm(M, N, K, Float32, af, bf;
                                          A_f32, BT_f32)
        @test D == Float32.(D_ref)
    end
end

@testset "fp8 pipelined GEMM: exact small-integer product, f16 acc" begin
    # K = 256 with values ≤ 2 bounds every partial sum by 1024 < 2048, so
    # even the f16 accumulator is exact.
    rng = MersenneTwister(0x8906)
    for (af, bf) in ((Val(:e4m3), Val(:e4m3)), (Val(:e5m2), Val(:e4m3)))
        M, N, K = 128, 128, 256
        A_f32, BT_f32 = _fp8p_int_inputs(rng, M, N, K, 2)
        D, D_ref = run_fp8_pipelined_gemm(M, N, K, Float16, af, bf;
                                          A_f32, BT_f32)
        @test D == Float32.(D_ref)
    end
end

@testset "fp8 pipelined GEMM vs Float64 reference: gaussian inputs" begin
    rng = MersenneTwister(0x8907)
    for (M, N, K) in [(128, 128, 256), (128, 256, 512), (256, 256, 512)]
        A_f32  = 0.5f0 .* randn(rng, Float32, M, K)
        BT_f32 = 0.5f0 .* randn(rng, Float32, N, K)

        # f32 accumulate: same reduced-alignment accumulation as
        # gpu/ada/gemm_fp8.jl documents on sm_89 — atol scales with K.
        D, D_ref = run_fp8_pipelined_gemm(M, N, K, Float32,
                                          Val(:e4m3), Val(:e4m3);
                                          A_f32, BT_f32)
        @test all(abs.(D .- D_ref) .<= K * 8f-5 .+ 1f-3 .* abs.(D_ref))

        # f16 accumulate: measured max|Δ|/K ≈ 1.4e-4 on sm_89 (the f16
        # significand's 2^-11 against running sums); ~2× margin.
        Dh, _ = run_fp8_pipelined_gemm(M, N, K, Float16,
                                       Val(:e4m3), Val(:e4m3);
                                       A_f32, BT_f32)
        @test all(abs.(Dh .- D_ref) .<= K * 3f-4 .+ 1f-2 .* abs.(D_ref))

        # Both paths are deterministic: a repeated launch is bit-identical.
        D2, _ = run_fp8_pipelined_gemm(M, N, K, Float32,
                                       Val(:e4m3), Val(:e4m3);
                                       A_f32, BT_f32)
        @test D == D2
    end
end
