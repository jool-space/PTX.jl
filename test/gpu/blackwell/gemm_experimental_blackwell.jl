# Blackwell no-TMA debug GEMM. Ported from pyptx
# examples/blackwell/gemm_experimental_blackwell.py:build_gemm_no_tma_debug.
#
# The first *real* Blackwell GEMM on the blackwell-1 wrapper surface — no
# TMA, no clusters, no new wrappers. 128×256×64 bf16 tile per CTA:
#
#   1. cooperative global → shared copy, K-major 128B-swizzled
#   2. 4× accumulating tcgen05.mma over K (BK=64 = 4 × k16 chunks):
#      kk=0 → enable_input_d=false (D = A·B); kk>0 → true (D += A·B)
#   3. per-thread TMEM ld (32x32b.x1 × BN) + v4.b32 global store epilogue
#
# Unlike the probes the inputs are non-uniform, so the K-major swizzle
# is load-bearing and ported faithfully (apply_swizzle 128B =
# logical ⊻ ((logical & 0x380) >> 3); see pyptx smem.py).
#
# B300 access is gone, so this is validated by always-on cross-arch
# ptxas at sm_100a + a faithful pyptx port + a bf16 CPU reference behind
# the runtime gate. The numeric @test is ready for whenever datacenter
# Blackwell hardware returns. SMEM 49184 B > 48 KiB → dynamic + opt-in.

using PTX: smem_addr_u32, tcgen05_descriptor, tcgen05_instr_desc_f16bf16_f32,
           BlackwellLayout
using CUDACore
using Random

const GXB_BM = 128
const GXB_BN = 256
const GXB_BK = 64                       # == K (single K-tile, asserted)
const GXB_A_BYTES = GXB_BM * GXB_BK * 2 # 16384
const GXB_B_BYTES = GXB_BN * GXB_BK * 2 # 32768
const GXB_BAR_OFF  = GXB_A_BYTES + GXB_B_BYTES   # 49152
const GXB_SLOT_OFF = GXB_BAR_OFF + 16
const GXB_SMEM     = GXB_SLOT_OFF + 16            # 49184
const GXB_STRIDE   = GXB_BK * 16                  # 1024 (descriptor stride)

# K-major logical byte offset for a 128B-swizzled operand tile
# (contig_elems = 64, row_group_bytes = 64*8*2 = 1024). k_elem = k_word*2.
@inline function _gxb_logical(row::UInt32, k_word::UInt32)
    rg  = row >> UInt32(3)
    rig = row & UInt32(7)
    rg * UInt32(1024) + ((rig * UInt32(64)) + (k_word << UInt32(1))) * UInt32(2)
end

# apply_swizzle 128B (pyptx smem.py: mask 0x380, shift 3).
@inline _gxb_swizzle(logical::UInt32) =
    logical ⊻ ((logical & UInt32(0x380)) >> UInt32(3))

# A, B_T: bf16 (M,K)/(N,K) row-major, packed as UInt32 word pairs.
# D: f32 (M,N) row-major.
function _gxb_gemm_kernel!(D::CuDeviceVector{Float32, 1},
                           A::CuDeviceVector{UInt32, 1},
                           B::CuDeviceVector{UInt32, 1},
                           N::Int32)
    smem32    = CuDynamicSharedArray(UInt32, (GXB_A_BYTES + GXB_B_BYTES) ÷ 4, 0)
    mbar      = CuDynamicSharedArray(UInt64, 1, GXB_BAR_OFF)
    tmem_slot = CuDynamicSharedArray(UInt32, 1, GXB_SLOT_OFF)

    a_addr    = smem_addr_u32(pointer(smem32))
    b_addr    = a_addr + UInt32(GXB_A_BYTES)
    mb_ptr    = pointer(mbar)
    bar_addr  = smem_addr_u32(mb_ptr)
    slot_addr = smem_addr_u32(pointer(tmem_slot))

    tid   = ptx"mov.u32"(sreg"tid.x")
    nthr  = ptx"mov.u32"(sreg"ntid.x")
    lane  = tid & UInt32(31)
    cta_m = ptx"mov.u32"(sreg"ctaid.x")
    cta_n = ptx"mov.u32"(sreg"ctaid.y")

    m_base = cta_m << UInt32(7)     # * BM
    n_base = cta_n << UInt32(8)     # * BN

    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mb_ptr, UInt32(1))
    end
    if tid < UInt32(32)
        ptx"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"(
            slot_addr, UInt32(512))
    end
    ptx"bar.sync"(Val(0))

    tmem  = @inbounds tmem_slot[1]
    idesc = tcgen05_instr_desc_f16bf16_f32(; m = GXB_BM, n = GXB_BN,
                                           ab_dtype = :bf16)

    # --- cooperative gmem → smem copy, K-major 128B-swizzled ---
    a_base_w = UInt64(m_base) * UInt64(GXB_BK ÷ 2)
    i = tid
    while i < UInt32((GXB_A_BYTES) ÷ 4)
        word = @inbounds A[Int(a_base_w + UInt64(i)) + 1]
        row  = i >> UInt32(5)
        kw   = i & UInt32((GXB_BK ÷ 2) - 1)
        phys = _gxb_swizzle(_gxb_logical(row, kw))
        @inbounds smem32[Int(phys >> UInt32(2)) + 1] = word
        i += UInt32(128)
    end

    b_base_w = UInt64(n_base) * UInt64(GXB_BK ÷ 2)
    j = tid
    while j < UInt32((GXB_B_BYTES) ÷ 4)
        word = @inbounds B[Int(b_base_w + UInt64(j)) + 1]
        row  = j >> UInt32(5)
        kw   = j & UInt32((GXB_BK ÷ 2) - 1)
        phys = _gxb_swizzle(_gxb_logical(row, kw))
        @inbounds smem32[(GXB_A_BYTES ÷ 4) + Int(phys >> UInt32(2)) + 1] = word
        j += UInt32(128)
    end
    ptx"bar.sync"(Val(0))

    a_desc0 = tcgen05_descriptor(a_addr; leading_bytes = 16,
                                 stride_bytes = GXB_STRIDE,
                                 swizzle = BlackwellLayout.B128)
    b_desc0 = tcgen05_descriptor(b_addr; leading_bytes = 16,
                                 stride_bytes = GXB_STRIDE,
                                 swizzle = BlackwellLayout.B128)

    # 2 phases × 2 k-chunks = 4 accumulating UMMAs over K=64.
    for phase in 0:1
        if tid == UInt32(0)
            for kk in (2 * phase):(2 * phase + 1)
                da = a_desc0 + UInt64(2 * kk)
                db = b_desc0 + UInt64(2 * kk)
                ptx"tcgen05.mma.cta_group::1.kind::f16"(tmem, da, db, idesc,
                                                        kk != 0)
            end
            ptx"tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"(
                bar_addr)
        end
        # parity 0 — see tcgen05_mma_probe.jl (blackwell-1 B300-verified
        # the init(1)+single-commit pattern with 0; pyptx uses 1).
        while !ptx"mbarrier.try_wait.parity.shared.b64"(mb_ptr, UInt32(0))
        end
        if phase == 0
            if tid == UInt32(0)
                ptx"mbarrier.init.shared.b64"(mb_ptr, UInt32(1))
            end
            ptx"bar.sync"(Val(0))
        end
    end

    # --- epilogue: per-thread TMEM read → v4.b32 global store ---
    row_base  = m_base + tid                              # global D row
    d_elem    = UInt64(row_base) * UInt64(N) + UInt64(n_base)
    d_ptr     = pointer(D) + Int(d_elem) * 4              # f32 → bytes
    tmem_addr = tmem + ((tid << UInt32(16)) & UInt32(0x03E00000))

    out = ntuple(c -> ptx"tcgen05.ld.sync.aligned.32x32b.x1.b32"(
                          tmem_addr + UInt32(c - 1)), Val(GXB_BN))
    ptx"tcgen05.wait::ld.sync.aligned"()

    for vec in 0:(GXB_BN ÷ 4 - 1)
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

# Row-major bf16-packed UInt32 words from an (R, C) Float32 matrix.
function _gxb_pack(M::Array{Float32, 2})
    bits = bf16_bits.(M)                       # (R, C) UInt16
    flat = vec(permutedims(bits))              # row-major, col-fastest
    return collect(reinterpret(UInt32, flat))  # pairs (lo = lower col)
end

@testset "gemm_experimental_blackwell compiles at sm_100a" begin
    types = Tuple{CuDeviceVector{Float32, 1}, CuDeviceVector{UInt32, 1},
                  CuDeviceVector{UInt32, 1}, Int32}
    @test ptxas_compiles(_gxb_gemm_kernel!, types;
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_gxb_gemm_kernel!, types; cap = v"10.0", feature_set = :arch)
    @test occursin("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32", ptx)
    @test occursin("tcgen05.mma.cta_group::1.kind::f16", ptx)
    @test occursin("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64", ptx)
    @test occursin("tcgen05.ld.sync.aligned.32x32b.x1.b32", ptx)
    @test occursin("tcgen05.wait::ld.sync.aligned", ptx)
    @test occursin("st.global.v4.b32", ptx)
end

# Datacenter-Blackwell only [10.0, 11.0); see tcgen05_smoke.jl rationale.
# (No hardware currently available — kept ready + faithful for the
# eventual datacenter-Blackwell run.)
if v"10.0" <= DEV_CAP < v"11.0"
    @testset "gemm_experimental_blackwell (bf16 A·Bᵀ, CPU ref)" begin
        function _run(rng, M, Ntot)
            @assert M % GXB_BM == 0 && Ntot % GXB_BN == 0
            A_f32 = Float32.(randn(rng, M, GXB_BK)) .* 0.1f0
            B_f32 = Float32.(randn(rng, Ntot, GXB_BK)) .* 0.1f0   # B_T: (N,K)
            A_d = CuArray(_gxb_pack(A_f32))
            B_d = CuArray(_gxb_pack(B_f32))
            D_d = CUDACore.zeros(Float32, M * Ntot)
            @cuda blocks=(M ÷ GXB_BM, Ntot ÷ GXB_BN) threads=128 feature_set=:arch _gxb_gemm_kernel!(
                D_d, A_d, B_d, Int32(Ntot))
            CUDACore.synchronize()

            Dflat = Array(D_d)
            got = Array{Float32}(undef, M, Ntot)
            for m in 1:M, n in 1:Ntot
                got[m, n] = Dflat[(m - 1) * Ntot + n]
            end
            Ab = bf16_to_f32.(bf16_bits.(A_f32))
            Bb = bf16_to_f32.(bf16_bits.(B_f32))
            ref = Ab * permutedims(Bb)            # D[m,n] = Σ_k A[m,k]·B_T[n,k]
            return got, ref
        end

        got, ref = _run(MersenneTwister(0xB1A), GXB_BM, GXB_BN)
        @test isapprox(got, ref; rtol = 5e-2, atol = 5e-2)

        got2, ref2 = _run(MersenneTwister(0xB1B), 2 * GXB_BM, 2 * GXB_BN)
        @test isapprox(got2, ref2; rtol = 5e-2, atol = 5e-2)
    end
end
