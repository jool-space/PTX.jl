# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.0
#
# Ported from cutlass/examples/47_ampere_gemm_universal_streamk
# (NVIDIA CUTLASS, BSD-3-Clause). The original demonstrates CUTLASS's
# stream-K threadblock scheduler; what's ported is the decomposition
# itself, hand-rolled:
#
# Data-parallel GEMM assigns one CTA per output tile — quantization leaves
# SMs idle whenever #tiles doesn't divide the wave size. Stream-K instead
# launches a FIXED grid of G CTAs and divides the total mainloop-iteration
# space (tiles × k-iters) evenly: CTA j owns global iterations
# [j·total/G, (j+1)·total/G), crossing tile boundaries wherever they fall.
#
# A tile touched by more than one CTA needs a cross-CTA reduction:
#   - the CTA covering the tile's FIRST k-iter is the owner; it computes its
#     partial, spin-waits on a per-tile arrival counter, then accumulates
#     every contributor's partial from workspace and writes D;
#   - every other CTA touching the tile stores its partial accumulator to
#     workspace and bumps the counter with atom.add.release (one add per
#     lane — the release orders that lane's partial stores ahead of the
#     arrival becoming visible; the owner's spin uses ld.acquire).
#
# CTA j covering iteration i:   j(i) = ((i+1)·G − 1) ÷ total
# so the owner expects 32 · (j(last_iter_of_tile) − j(first_iter_of_tile))
# arrivals. All G CTAs must be co-resident for the spin to make progress —
# G is tiny here (≤ 8 warps); a real stream-K sizes G to one wave.
#
# Tiling per CTA-segment is the single-warp 16×8 bf16 m16n8k16 of gemm.jl,
# typed BFloat16 storage (bit-level host packing — f32→bf16 broadcast
# crashes LLVM on Julia 1.12.6/x86). Fragment layout: see gemm.jl.

using Random
using Microfloats: BFloat16

rne_bf16_bits(x::Float32) =
    (b = reinterpret(UInt32, x);
     UInt16((b + 0x7fff + ((b >> 16) & 0x1)) >> 16))
to_bf16(x::Array{Float32}) = collect(reinterpret(BFloat16, rne_bf16_bits.(x)))
quantize_bf16(x) = reinterpret.(Float32, UInt32.(rne_bf16_bits.(x)) .<< 16)

const SK_BM, SK_BN, SK_BK = 16, 8, 16

function gemm_streamk_kernel!(
        D::CuDeviceVector{Float32},
        A::CuDeviceVector{BFloat16},
        B_T::CuDeviceVector{BFloat16},
        partials::CuDeviceVector{Float32},   # (T, G, 32 lanes, 4) f32
        counters::CuDeviceVector{UInt32},    # (T,) arrival counters, zeroed
        ::Val{M}, ::Val{N}, ::Val{K}, ::Val{G}) where {M, N, K, G}
    L     = K ÷ SK_BK                  # k-iters per tile
    TN    = N ÷ SK_BN
    total = (M ÷ SK_BM) * TN * L       # global iteration count

    cta      = Int(ptx"mov.u32"(sreg"ctaid.x"))
    it       = (cta * total) ÷ G
    it_end   = ((cta + 1) * total) ÷ G

    tid    = ptx"mov.u32"(sreg"tid.x")
    lane   = Int(tid)
    gid    = lane >> 2
    tig    = lane & 0x3
    col_lo = tig << 1

    pa = pointer(A)
    pb = pointer(B_T)
    pc = pointer(counters)

    while it < it_end
        t  = it ÷ L                    # tile this segment lands in
        k0 = it - t * L                # first k-iter of the segment
        k1 = min(L, k0 + (it_end - it))

        tm, tn = divrem(t, TN)
        m_base = tm * SK_BM
        n_base = tn * SK_BN
        row_lo = m_base + gid
        row_hi = row_lo + 8
        n_col  = n_base + gid

        acc = (0f0, 0f0, 0f0, 0f0)
        @inbounds for ki in k0:k1-1
            k_base = ki * SK_BK
            a1 = ptx"ld.global.b32"(pa + (row_lo * K + k_base + col_lo + 0) * 2)
            a2 = ptx"ld.global.b32"(pa + (row_hi * K + k_base + col_lo + 0) * 2)
            a3 = ptx"ld.global.b32"(pa + (row_lo * K + k_base + col_lo + 8) * 2)
            a4 = ptx"ld.global.b32"(pa + (row_hi * K + k_base + col_lo + 8) * 2)
            b1 = ptx"ld.global.b32"(pb + (n_col * K + k_base + col_lo + 0) * 2)
            b2 = ptx"ld.global.b32"(pb + (n_col * K + k_base + col_lo + 8) * 2)
            acc = ptx"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"(
                (a1, a2, a3, a4), (b1, b2), acc)
        end

        if k0 > 0
            # Contributor: park the partial, announce arrival.
            base = ((t * G + cta) * 32 + lane) * 4
            @inbounds begin
                partials[base + 1] = acc[1]
                partials[base + 2] = acc[2]
                partials[base + 3] = acc[3]
                partials[base + 4] = acc[4]
            end
            ptx"atom.add.release.gpu.global.u32"(pc + t * 4, UInt32(1))
        else
            if k1 < L
                # Owner of a split tile: wait for every contributor, fold
                # their partials in. j(i) = ((i+1)G − 1) ÷ total.
                j_first = ((t * L + 1) * G - 1) ÷ total
                j_last  = (((t + 1) * L) * G - 1) ÷ total
                expected = UInt32(32 * (j_last - j_first))
                while ptx"ld.acquire.gpu.global.u32"(pc + t * 4) < expected
                end
                for j in j_first+1:j_last
                    base = ((t * G + j) * 32 + lane) * 4
                    @inbounds acc = (acc[1] + partials[base + 1],
                                     acc[2] + partials[base + 2],
                                     acc[3] + partials[base + 3],
                                     acc[4] + partials[base + 4])
                end
            end
            d_col = n_base + col_lo
            @inbounds begin
                D[row_lo * N + d_col + 1] = acc[1]
                D[row_lo * N + d_col + 2] = acc[2]
                D[row_hi * N + d_col + 1] = acc[3]
                D[row_hi * N + d_col + 2] = acc[4]
            end
        end

        it += k1 - k0
    end
    return nothing
end

flatten_rowmajor(A::AbstractMatrix{Float32}) = vec(permutedims(A))

function run_streamk_gemm(M::Int, N::Int, K::Int, G::Int)
    T = (M ÷ SK_BM) * (N ÷ SK_BN)
    total = T * (K ÷ SK_BK)
    @assert G <= total "every CTA needs at least one iteration"

    rng    = MersenneTwister(M * 7919 + N * 31 + K + G)
    A_f32  = 0.1f0 .* randn(rng, Float32, M, K)
    BT_f32 = 0.1f0 .* randn(rng, Float32, N, K)

    A_d  = CuArray(to_bf16(flatten_rowmajor(A_f32)))
    BT_d = CuArray(to_bf16(flatten_rowmajor(BT_f32)))
    D_d  = CUDACore.zeros(Float32, M * N)
    partials = CUDACore.zeros(Float32, T * G * 32 * 4)
    counters = CUDACore.zeros(UInt32, T)

    @cuda blocks=G threads=32 gemm_streamk_kernel!(
        D_d, A_d, BT_d, partials, counters, Val(M), Val(N), Val(K), Val(G))
    CUDACore.synchronize()

    D     = Matrix(reshape(Array(D_d), N, M)')
    D_ref = quantize_bf16(A_f32) * quantize_bf16(BT_f32)'
    return D, D_ref
end

@testset "Ampere stream-K GEMM (fixed grid, cross-CTA fixup)" begin
    # (M, N, K, G) sweeps chosen so segments split tiles mid-K in several
    # patterns: G=1 degenerates to a serial loop over all tiles; G that
    # doesn't divide `total` puts tile boundaries inside CTA ranges; the
    # K=64 case gives short tiles so single CTAs span several whole tiles
    # plus fractional ends.
    for (M, N, K, G) in [(32, 16, 256, 1), (32, 16, 256, 3),
                         (32, 16, 256, 6), (32, 16, 256, 8),
                         (64, 32, 64, 6), (64, 32, 64, 16)]
        D, D_ref = run_streamk_gemm(M, N, K, G)
        @test all(abs.(D .- D_ref) .<= 1f-2 .+ 1f-2 .* abs.(D_ref))
    end
end
