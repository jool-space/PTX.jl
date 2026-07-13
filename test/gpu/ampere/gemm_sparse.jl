# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.0
#
# Ported from cutlass/examples/15_ampere_sparse_tensorop_gemm
# (NVIDIA CUTLASS, BSD-3-Clause). The original runs CUTLASS's 2:4
# structured-sparse GEMM (with cuSPARSELt-style pruning/compression done by
# CUTLASS utilities); what's ported is the whole mechanism, hand-rolled:
# host-side 2:4 pruning + compression + metadata packing, and a kernel on
# mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32.
#
# 2:4 structured sparsity: every 4-wide chunk of a row of A keeps exactly
# two non-zeros. The kernel never sees the dense A — it loads a compressed
# (M, K/2) matrix plus a .b32 metadata word of 2-bit in-chunk positions,
# and the tensor core scatters the products back to the right K lanes.
# Same K-coverage per instruction as dense m16n8k16 at half the A traffic.
#
# Layouts (PTX ISA 9.3 §9.7.15.6.2.2, m16n8k32 f16/bf16, selector = 0):
#   gid = tid >> 2, tig = tid & 3
#
#   A (compressed (M, K/2) bf16 row-major): dense-m16n8k16-shaped fragment
#   over compressed columns — reg pairs are the two kept elements of one
#   original 4-wide chunk:
#     a[1] = Ac[g,   2tig + {0,1}]   ↔ chunk at orig cols 4tig      (row g)
#     a[2] = Ac[g+8, 2tig + {0,1}]   ↔ same chunk cols, row g+8
#     a[3] = Ac[g,   2tig + 8 + {0,1}] ↔ chunk at orig cols 4tig+16 (row g)
#     a[4] = Ac[g+8, 2tig + 8 + {0,1}]
#   B (32×8, .row.col ⇒ index B_T[n, k]): 4 regs at k-offsets 0/8/16/24:
#     b[r+1] = B_T[g, 8r + 2tig + {0,1}]
#   Metadata word (per contributing thread; selector 0 → T0/T1 per group):
#     T4g+h covers orig cols 16h..16h+15 of rows g (bits 15:0) and g+8
#     (bits 31:16); nibble j = chunk at cols 16h+4j, low 2 bits = position
#     of the first kept element, high 2 bits = the second (ascending).
#   D: d[i] = D[g + 8*(i>>1), 2tig + (i&1)] — as every m16n8kX f32 shape.
#
# Storage is typed CuArray{BFloat16}; host packing is bit-level (f32→bf16
# broadcast crashes LLVM on Julia 1.12.6/x86 — see conv2d_fprop.jl).

using Random
using Microfloats: BFloat16

rne_bf16_bits(x::Float32) =
    (b = reinterpret(UInt32, x);
     UInt16((b + 0x7fff + ((b >> 16) & 0x1)) >> 16))
to_bf16(x::Array{Float32}) = collect(reinterpret(BFloat16, rne_bf16_bits.(x)))
quantize_bf16(x) = reinterpret.(Float32, UInt32.(rne_bf16_bits.(x)) .<< 16)

const SP_BM, SP_BN, SP_BK = 16, 8, 32

function gemm_sparse_kernel!(
        D::CuDeviceVector{Float32},
        Ac::CuDeviceVector{BFloat16},      # (M, K/2) compressed, row-major
        B_T::CuDeviceVector{BFloat16},     # (N, K) row-major
        meta::CuDeviceVector{UInt32},      # (M/16, K/32, 8, 2) flat
        ::Val{M}, ::Val{N}, ::Val{K}) where {M, N, K}
    n_iters = K ÷ SP_BK
    Kc      = K ÷ 2

    tm     = Int(ptx"mov.u32"(sreg"ctaid.y"))
    m_base = tm * SP_BM
    n_base = Int(ptx"mov.u32"(sreg"ctaid.x")) * SP_BN

    tid = ptx"mov.u32"(sreg"tid.x")
    gid = Int(tid >> UInt32(2))
    tig = Int(tid & UInt32(0x3))
    cl  = 2 * tig

    row_lo = m_base + gid
    row_hi = row_lo + 8
    n_col  = n_base + gid

    pa = pointer(Ac)
    pb = pointer(B_T)

    acc = (0f0, 0f0, 0f0, 0f0)

    @inbounds for kt in 0:n_iters-1
        kc = kt * 16                       # compressed-column tile base

        a1 = ptx"ld.global.b32"(pa + (row_lo * Kc + kc + cl + 0) * 2)
        a2 = ptx"ld.global.b32"(pa + (row_hi * Kc + kc + cl + 0) * 2)
        a3 = ptx"ld.global.b32"(pa + (row_lo * Kc + kc + cl + 8) * 2)
        a4 = ptx"ld.global.b32"(pa + (row_hi * Kc + kc + cl + 8) * 2)

        k = kt * SP_BK
        b1 = ptx"ld.global.b32"(pb + (n_col * K + k + cl +  0) * 2)
        b2 = ptx"ld.global.b32"(pb + (n_col * K + k + cl +  8) * 2)
        b3 = ptx"ld.global.b32"(pb + (n_col * K + k + cl + 16) * 2)
        b4 = ptx"ld.global.b32"(pb + (n_col * K + k + cl + 24) * 2)

        # Selector 0: T0/T1 of each group contribute; T2/T3's word is
        # ignored, so tig&1 gives them a valid (unused) load.
        e = meta[((tm * n_iters + kt) * 8 + gid) * 2 + (tig & 1) + 1]

        acc = ptx"mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32"(
            (a1, a2, a3, a4), (b1, b2, b3, b4), acc, e, Val(0))
    end

    d_col = n_base + cl
    @inbounds begin
        D[row_lo * N + d_col + 1] = acc[1]
        D[row_lo * N + d_col + 2] = acc[2]
        D[row_hi * N + d_col + 1] = acc[3]
        D[row_hi * N + d_col + 2] = acc[4]
    end
    return nothing
end

# 2:4 prune (keep the two largest magnitudes per 4-chunk, ascending
# positions) and compress. Returns:
#   A_pruned :: (M, K)   f32 — dense with zeros, for the reference matmul
#   A_comp   :: (M, K/2) f32 — the kept elements
#   nib      :: (M, K/4) — per-chunk metadata nibble (idx0 | idx1 << 2)
function prune_2_4(A::Matrix{Float32})
    M, K = size(A)
    A_pruned = zeros(Float32, M, K)
    A_comp   = zeros(Float32, M, K ÷ 2)
    nib      = zeros(UInt8, M, K ÷ 4)
    for m in 1:M, c in 0:(K ÷ 4 - 1)
        chunk = abs.(A[m, 4c+1:4c+4])
        keep  = sort(sortperm(chunk; rev = true)[1:2])   # ascending positions
        p0, p1 = keep[1] - 1, keep[2] - 1
        A_pruned[m, 4c + p0 + 1] = A[m, 4c + p0 + 1]
        A_pruned[m, 4c + p1 + 1] = A[m, 4c + p1 + 1]
        A_comp[m, 2c + 1] = A[m, 4c + p0 + 1]
        A_comp[m, 2c + 2] = A[m, 4c + p1 + 1]
        nib[m, c + 1] = UInt8(p0) | (UInt8(p1) << 2)
    end
    return A_pruned, A_comp, nib
end

# One metadata word covering `span` chunks of row pair (r_lo, r_hi) starting
# at chunk `c0`: row r_lo in bits 15:0, row r_hi in bits 31:16, one nibble
# per chunk low-to-high. span = 4 for both shapes (16 original columns).
function meta_word(nib::Matrix{UInt8}, r_lo::Int, r_hi::Int, c0::Int)
    w = UInt32(0)
    for j in 0:3
        w |= UInt32(nib[r_lo + 1, c0 + j + 1]) << (4j)
        w |= UInt32(nib[r_hi + 1, c0 + j + 1]) << (16 + 4j)
    end
    return w
end

# m16n8k32 words, indexed [tm, kt, g, h]: thread-pair T0/T1 (selector 0),
# half h covers original columns 32kt + 16h .. +15.
function pack_meta_k32(nib::Matrix{UInt8}, M::Int, K::Int)
    meta = zeros(UInt32, (M ÷ 16) * (K ÷ 32) * 8 * 2)
    for tm in 0:(M ÷ 16 - 1), kt in 0:(K ÷ 32 - 1), g in 0:7, h in 0:1
        meta[((tm * (K ÷ 32) + kt) * 8 + g) * 2 + h + 1] =
            meta_word(nib, 16tm + g, 16tm + g + 8, 8kt + 4h)
    end
    return meta
end

# m16n8k16 words, indexed [tm, kt, g]: ONE thread per group (the one the
# selector names) carries the whole 16-column k-tile of its row pair.
function pack_meta_k16(nib::Matrix{UInt8}, M::Int, K::Int)
    meta = zeros(UInt32, (M ÷ 16) * (K ÷ 16) * 8)
    for tm in 0:(M ÷ 16 - 1), kt in 0:(K ÷ 16 - 1), g in 0:7
        meta[(tm * (K ÷ 16) + kt) * 8 + g + 1] =
            meta_word(nib, 16tm + g, 16tm + g + 8, 4kt)
    end
    return meta
end

# --- m16n8k16 variant: single-thread metadata + selector sweep --------------
#
# The k16 shape flips the metadata convention: ONE thread per group of four
# carries the word for the whole row pair × 16-column tile, and the selector
# names which thread (0..3). To make the selector observable, only the named
# lane loads the true word — every other lane passes a valid-but-wrong
# constant (all-chunks-keep-{0,1}, 0x44444444). The result is correct iff
# the hardware reads exactly the lane the selector says.
#
# Fragments (PTX ISA 9.3 §9.7.15.6.2.1): A is 2 regs — the kept pair of the
# chunk at orig cols 4tig, rows g / g+8 (compressed cols 2tig + {0,1});
# B and C/D are identical to dense m16n8k16.

function gemm_sparse_k16_kernel!(
        D::CuDeviceVector{Float32},
        Ac::CuDeviceVector{BFloat16},      # (M, K/2) compressed, row-major
        B_T::CuDeviceVector{BFloat16},     # (N, K) row-major
        meta::CuDeviceVector{UInt32},      # (M/16, K/16, 8) flat
        ::Val{M}, ::Val{N}, ::Val{K}, ::Val{SEL}) where {M, N, K, SEL}
    n_iters = K ÷ 16
    Kc      = K ÷ 2

    tm     = Int(ptx"mov.u32"(sreg"ctaid.y"))
    m_base = tm * SP_BM
    n_base = Int(ptx"mov.u32"(sreg"ctaid.x")) * SP_BN

    tid = ptx"mov.u32"(sreg"tid.x")
    gid = Int(tid >> UInt32(2))
    tig = Int(tid & UInt32(0x3))
    cl  = 2 * tig

    row_lo = m_base + gid
    row_hi = row_lo + 8
    n_col  = n_base + gid

    pa = pointer(Ac)
    pb = pointer(B_T)

    acc = (0f0, 0f0, 0f0, 0f0)

    @inbounds for kt in 0:n_iters-1
        kc = kt * 8                        # compressed-column tile base

        a1 = ptx"ld.global.b32"(pa + (row_lo * Kc + kc + cl) * 2)
        a2 = ptx"ld.global.b32"(pa + (row_hi * Kc + kc + cl) * 2)

        k = kt * 16
        b1 = ptx"ld.global.b32"(pb + (n_col * K + k + cl + 0) * 2)
        b2 = ptx"ld.global.b32"(pb + (n_col * K + k + cl + 8) * 2)

        e = tig == SEL ? meta[(tm * n_iters + kt) * 8 + gid + 1] :
                         UInt32(0x44444444)

        acc = ptx"mma.sp.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"(
            (a1, a2), (b1, b2), acc, e, Val(SEL))
    end

    d_col = n_base + cl
    @inbounds begin
        D[row_lo * N + d_col + 1] = acc[1]
        D[row_lo * N + d_col + 2] = acc[2]
        D[row_hi * N + d_col + 1] = acc[3]
        D[row_hi * N + d_col + 2] = acc[4]
    end
    return nothing
end

flatten_rowmajor(A::AbstractMatrix{Float32}) = vec(permutedims(A))

function sparse_inputs(M::Int, N::Int, K::Int)
    rng    = MersenneTwister(M * 7919 + N * 31 + K)
    A_f32  = quantize_bf16(0.1f0 .* randn(rng, Float32, M, K))
    BT_f32 = quantize_bf16(0.1f0 .* randn(rng, Float32, N, K))
    A_pruned, A_comp, nib = prune_2_4(A_f32)
    return (Ac_d  = CuArray(to_bf16(flatten_rowmajor(A_comp))),
            BT_d  = CuArray(to_bf16(flatten_rowmajor(BT_f32))),
            nib   = nib,
            D_ref = A_pruned * BT_f32')   # dense reference over the pruned A
end

function run_sparse_gemm(M::Int, N::Int, K::Int)
    @assert M % SP_BM == 0 && N % SP_BN == 0 && K % SP_BK == 0
    inp    = sparse_inputs(M, N, K)
    meta_d = CuArray(pack_meta_k32(inp.nib, M, K))
    D_d    = CUDACore.zeros(Float32, M * N)
    @cuda blocks=(N ÷ SP_BN, M ÷ SP_BM) threads=32 gemm_sparse_kernel!(
        D_d, inp.Ac_d, inp.BT_d, meta_d, Val(M), Val(N), Val(K))
    CUDACore.synchronize()
    return Matrix(reshape(Array(D_d), N, M)'), inp.D_ref
end

function run_sparse_gemm_k16(M::Int, N::Int, K::Int, sel::Int)
    @assert M % SP_BM == 0 && N % SP_BN == 0 && K % 16 == 0
    inp    = sparse_inputs(M, N, K)
    meta_d = CuArray(pack_meta_k16(inp.nib, M, K))
    D_d    = CUDACore.zeros(Float32, M * N)
    @cuda blocks=(N ÷ SP_BN, M ÷ SP_BM) threads=32 gemm_sparse_k16_kernel!(
        D_d, inp.Ac_d, inp.BT_d, meta_d, Val(M), Val(N), Val(K), Val(sel))
    CUDACore.synchronize()
    return Matrix(reshape(Array(D_d), N, M)'), inp.D_ref
end

@testset "Ampere 2:4 sparse GEMM (mma.sp m16n8k32 bf16)" begin
    for (M, N, K) in [(16, 8, 32), (16, 8, 64), (32, 16, 64), (64, 32, 128)]
        D, D_ref = run_sparse_gemm(M, N, K)
        @test all(abs.(D .- D_ref) .<= 1f-2 .+ 1f-2 .* abs.(D_ref))
    end
end

@testset "Ampere 2:4 sparse GEMM (mma.sp m16n8k16 bf16, selector sweep)" begin
    for sel in 0:3
        D, D_ref = run_sparse_gemm_k16(32, 16, 64, sel)
        @test all(abs.(D .- D_ref) .<= 1f-2 .+ 1f-2 .* abs.(D_ref))
    end
end
