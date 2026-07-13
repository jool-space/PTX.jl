# TEST_TARGET: requires=toolkit evidence=mixed target=sm_100f|sm_110f
# Blackwell tcgen05 B128 *non-uniform* isolation probe.
#
# This is the probe BLACKWELL_INTERFACE_NOTES.md friction #1 and memory
# `gemm_experimental_blackwell_parked` call for: "isolate B128 LBO/SBO with
# a minimal non-uniform-fill B128 single-commit MMA probe (smaller than a
# full GEMM) before trusting any B128-swizzled GEMM."
#
# Unlike smoke / mma_probe / accum_probe (uniform fills → swizzle-invariant,
# B32), this stages A and B through the *real* 128B K-major swizzle with a
# NON-UNIFORM fill, so every part of the B128 path is load-bearing: a wrong
# leading/stride byte offset, a wrong swizzle permutation, or a wrong TMEM
# lane map all corrupt the result. One init + one commit + one drain (no
# phase reinit) → parity 0 is the verified single-commit convention
# (memory `blackwell-parity-faithful`: pyptx's parity-1 was for its
# *multi-phase reinit* loop; this is the single-init+commit pattern).
#
# Descriptors come from the new `tcgen05_layout_kmajor` picker — for K=64
# it is bit-identical to the maintained pyptx production constant
# MMA_DESC_B128 = 0x4000404000010000 (proven in test/host/tcgen05_layout.jl).
# So a green B300 run here PROVES the production B128 descriptor and clears
# friction #1; a red run isolates the bug to NOT being the descriptor.
#
# The swizzled copy loop is a faithful port of pyptx
# gemm_experimental_blackwell.py copy_a_loop/copy_b_loop, but using the
# host-tested PTX helpers instead of inlined magic.
#
# SMEM = A 16384 + B 16384 + bar 8 + slot 4 = 32796 B < 48 KiB → static.
# Always ptxas-validates cross-arch at sm_100a; runtime admits the
# PTX-defined sm_100f and sm_110f tcgen05 families.

using PTX: smem_addr_u32, tcgen05_descriptor, tcgen05_instr_desc_f16bf16_f32,
           tcgen05_layout_kmajor, kmajor_swizzled_logical_bytes,
           apply_blackwell_swizzle, tmem_lane_addr, BlackwellLayout
using PTX.MBarriers: barrier_try_wait
using CUDACore

const TBP_BM = 128
const TBP_BN = 128
const TBP_BK = 64                               # 128 B K-major row → B128
const TBP_A_BYTES = TBP_BM * TBP_BK * 2         # 16384
const TBP_B_BYTES = TBP_BN * TBP_BK * 2         # 16384
const TBP_AB_WORDS = (TBP_A_BYTES + TBP_B_BYTES) ÷ 4
const TBP_A_WORDS  = TBP_A_BYTES ÷ 4            # 4096
const TBP_B_WORDS  = TBP_B_BYTES ÷ 4            # 4096
const TBP_ROW_WORDS = TBP_BK ÷ 2               # 32 b32 words per K-major row
const TBP_EPI_ROWS  = 32                        # warp-0 readback subtile

# Non-uniform, bf16-exact (small integers): swizzle is load-bearing.
_tbp_A(m, k) = Float32((m + 3k) % 5)            # 0..4
_tbp_B(n, k) = Float32((2n + k) % 4)            # 0..3

function _tcgen05_b128_probe_kernel!(O::CuDeviceVector{Float32, 1},
                                     A_w::CuDeviceVector{UInt32, 1},
                                     B_w::CuDeviceVector{UInt32, 1})
    smem      = CuStaticSharedArray(UInt32, TBP_AB_WORDS)
    mbar      = CuStaticSharedArray(UInt64, 1)
    tmem_slot = CuStaticSharedArray(UInt32, 1)

    a_addr    = smem_addr_u32(pointer(smem))
    b_addr    = a_addr + UInt32(TBP_A_BYTES)
    mb_ptr    = pointer(mbar)
    bar_addr  = smem_addr_u32(mb_ptr)
    slot_addr = smem_addr_u32(pointer(tmem_slot))

    tid  = ptx"mov.u32"(sreg"tid.x")
    lane = tid & UInt32(31)

    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mb_ptr, UInt32(1))
    end
    if tid < UInt32(32)
        ptx"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"(
            slot_addr, UInt32(512))
    end

    # Swizzled SMEM staging — faithful port of pyptx copy_a/b_loop, via the
    # host-tested helpers. a_idx = row*32 + k_word; word packs (2kw, 2kw+1).
    sw = BlackwellLayout.B128
    i = tid
    while i < UInt32(TBP_A_WORDS)
        row = i >> UInt32(5)
        kw  = i & UInt32(31)
        logical  = UInt32(kmajor_swizzled_logical_bytes(row, kw << 1, TBP_BK))
        physical = apply_blackwell_swizzle(logical, sw)
        @inbounds smem[(physical >> 2) + UInt32(1)] = A_w[i + UInt32(1)]
        i += UInt32(128)
    end
    i = tid
    while i < UInt32(TBP_B_WORDS)
        row = i >> UInt32(5)
        kw  = i & UInt32(31)
        logical  = UInt32(kmajor_swizzled_logical_bytes(row, kw << 1, TBP_BK))
        physical = apply_blackwell_swizzle(logical, sw)
        @inbounds smem[UInt32(TBP_A_WORDS) + (physical >> 2) + UInt32(1)] =
            B_w[i + UInt32(1)]
        i += UInt32(128)
    end
    ptx"bar.sync"(Val(0))

    tmem  = @inbounds tmem_slot[1]
    idesc = tcgen05_instr_desc_f16bf16_f32(; m = TBP_BM, n = TBP_BN,
                                           ab_dtype = :bf16)
    l = tcgen05_layout_kmajor(k = TBP_BK)        # B128, lead=16, stride=1024
    a_desc0 = tcgen05_descriptor(a_addr; leading_bytes = l.leading_bytes,
                                 stride_bytes = l.stride_bytes,
                                 swizzle = l.swizzle)
    b_desc0 = tcgen05_descriptor(b_addr; leading_bytes = l.leading_bytes,
                                 stride_bytes = l.stride_bytes,
                                 swizzle = l.swizzle)

    if tid == UInt32(0)
        # Single ktile (K = BK = 64) → 4 accumulating k16 MMAs; p=false
        # only on the first (D = A·B, then D += A·B).
        for kk in 0:3
            da = a_desc0 + UInt64(2 * kk)
            db = b_desc0 + UInt64(2 * kk)
            ptx"tcgen05.mma.cta_group::1.kind::f16"(tmem, da, db, idesc,
                                                    kk != 0)
        end
        ptx"tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"(
            bar_addr)
    end

    # Single init + single commit → parity 0 (see header / memory).
    barrier_try_wait(mb_ptr, UInt32(0))

    tmem_addr = tmem_lane_addr(tmem, lane)
    if tid < UInt32(TBP_EPI_ROWS)
        dst = ptx"tcgen05.ld.sync.aligned.32x32b.x128.b32"(tmem_addr)
        ptx"tcgen05.wait::ld.sync.aligned"()
        base = Int(lane) * TBP_BN
        for c in 1:TBP_BN
            @inbounds O[base + c] = reinterpret(Float32, dst[c])
        end
    end

    if tid < UInt32(32)
        ptx"tcgen05.dealloc.cta_group::1.sync.aligned.b32"(tmem, UInt32(512))
        ptx"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"()
    end
    return nothing
end

@testset "tcgen05 B128 non-uniform probe compiles at sm_100a" begin
    types = Tuple{CuDeviceVector{Float32, 1},
                  CuDeviceVector{UInt32, 1}, CuDeviceVector{UInt32, 1}}
    @test ptxas_compiles(_tcgen05_b128_probe_kernel!, types;
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_tcgen05_b128_probe_kernel!, types;
                   cap = v"10.0", feature_set = :arch)
    @test occursin("tcgen05.mma.cta_group::1.kind::f16", ptx)
    @test occursin("tcgen05.ld.sync.aligned.32x32b.x128.b32", ptx)
    @test occursin("tcgen05.commit.cta_group::1", ptx)
end

# tcgen05 family targets only; see tcgen05_smoke.jl rationale.
if test_runtime_supported(@__FILE__)
    @testset "tcgen05 B128 non-uniform probe (B300 runtime, A·Bᵀ)" begin
        # Pack two bf16 per b32 word, K-major: word(row,kw) holds
        # elements 2kw, 2kw+1 of that row; index = row*32 + kw.
        function pack(f)
            w = Vector{UInt32}(undef, TBP_BM * TBP_ROW_WORDS)
            for r in 0:(TBP_BM - 1), kw in 0:(TBP_ROW_WORDS - 1)
                lo = bf16_bits(f(r, 2 * kw))
                hi = bf16_bits(f(r, 2 * kw + 1))
                w[r * TBP_ROW_WORDS + kw + 1] = UInt32(lo) | (UInt32(hi) << 16)
            end
            w
        end
        A_d = CuArray(pack(_tbp_A))
        B_d = CuArray(pack(_tbp_B))
        O_d = CUDACore.zeros(Float32, TBP_EPI_ROWS * TBP_BN)

        @cuda blocks=1 threads=128 _tcgen05_b128_probe_kernel!(
            O_d, A_d, B_d)
        CUDACore.synchronize()

        # CPU reference: bf16-rounded inputs, f32 accumulate (exact here —
        # all factors are small integers). got is (col, row) row-major.
        got = reshape(Array(O_d), TBP_BN, TBP_EPI_ROWS)
        ref = Array{Float32}(undef, TBP_BN, TBP_EPI_ROWS)
        for m in 0:(TBP_EPI_ROWS - 1), n in 0:(TBP_BN - 1)
            acc = 0.0f0
            for k in 0:(TBP_BK - 1)
                acc += bf16_to_f32(bf16_bits(_tbp_A(m, k))) *
                       bf16_to_f32(bf16_bits(_tbp_B(n, k)))
            end
            ref[n + 1, m + 1] = acc
        end
        @test got == ref
    end
end
