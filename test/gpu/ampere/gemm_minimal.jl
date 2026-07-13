# TEST_TARGET: requires=gpu evidence=runtime target=sm_80
#
# Reference 16×8×16 bf16 GEMM tile — single warp, single block.
#
# Pipeline exercises (in order):
#   1. cp.async.ca.shared.global (cooperative load, 16-byte chunks)
#   2. cp.async.commit_group + wait_all (drain in-flight copies)
#   3. bar.sync (cross-warp barrier in case the block grows)
#   4. Per-lane fragment construction from shared memory
#   5. mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
#   6. Per-lane store to global D
#
# This is the "look what PTX.jl does" demo. It composes ~6 wrapper families
# the unit tests exercise individually but never together. ldmatrix is
# deferred to Phase 2 — the manual fragment construction here is what
# ldmatrix automates, and writing it out makes the lane-to-element layout
# explicit.

# --- bf16 packing -----------------------------------------------------------
#
# bf16 = top 16 bits of an IEEE Float32 (sign + 8-bit exp + 7-bit mantissa).
# Round-to-nearest-truncate is what Float32→bf16 reduces to; on bf16-friendly
# inputs the lost mantissa bits are negligible.

bf16_bits(x::Float32) = UInt16(reinterpret(UInt32, x) >> 16)
bf16_to_f32(b::UInt16) = reinterpret(Float32, UInt32(b) << 16)
bf16_pack(lo::UInt16, hi::UInt16) = UInt32(lo) | (UInt32(hi) << 16)

# --- mma m16n8k16 fragment layout (PTX ISA 9.2 §9.7.13.4) ------------------
#
# Per lane T (0..31):
#   group_id        = T >> 2          (0..7)
#   thread_in_group = T & 0x3         (0..3)
#
# A (16×16 bf16, row-major, NTuple{4,UInt32} per lane):
#   a[1] = pack(A[group,        2*in_group + 0], A[group,        2*in_group + 1])
#   a[2] = pack(A[group + 8,    2*in_group + 0], A[group + 8,    2*in_group + 1])
#   a[3] = pack(A[group,        2*in_group + 8], A[group,        2*in_group + 9])
#   a[4] = pack(A[group + 8,    2*in_group + 8], A[group + 8,    2*in_group + 9])
#
# B (16×8 bf16, col-major, NTuple{2,UInt32} per lane):
#   b[1] = pack(B[2*in_group + 0, group], B[2*in_group + 1, group])
#   b[2] = pack(B[2*in_group + 8, group], B[2*in_group + 9, group])
#
# D (16×8 f32, NTuple{4,Float32} per lane):
#   d[1] = D[group,     2*in_group + 0]
#   d[2] = D[group,     2*in_group + 1]
#   d[3] = D[group + 8, 2*in_group + 0]
#   d[4] = D[group + 8, 2*in_group + 1]

# --- the kernel -------------------------------------------------------------

function gemm_tile_kernel!(D, A_bits, B_bits)
    smem_A = CuStaticSharedArray(UInt16, 256)   # 16×16 bf16
    smem_B = CuStaticSharedArray(UInt16, 128)   # 16×8 bf16

    tid = ptx"mov.u32"(sreg"tid.x")

    # --- Stage 1: cp.async A (256 bf16 = 512B) — 32 lanes × 16B each.
    src_A = pointer(A_bits) + Int(tid) * 16
    dst_A = pointer(smem_A) + Int(tid) * 16
    ptx"cp.async.ca.shared.global"(dst_A, src_A, Val(16))

    # --- Stage 1: cp.async B (128 bf16 = 256B) — 16 lanes × 16B each.
    if tid < UInt32(16)
        src_B = pointer(B_bits) + Int(tid) * 16
        dst_B = pointer(smem_B) + Int(tid) * 16
        ptx"cp.async.ca.shared.global"(dst_B, src_B, Val(16))
    end

    ptx"cp.async.commit_group"()
    ptx"cp.async.wait_all"()
    ptx"bar.sync"(Val(0))

    # --- Stage 2: build fragments from shared memory.
    group     = tid >> UInt32(2)                  # 0..7
    in_group  = tid & UInt32(0x3)                 # 0..3
    g  = Int(group)
    ig = Int(in_group)

    # A row-major linear index: row*16 + col
    a1 = bf16_pack(@inbounds(smem_A[g*16        + 2*ig + 1]),
                   @inbounds(smem_A[g*16        + 2*ig + 2]))
    a2 = bf16_pack(@inbounds(smem_A[(g+8)*16    + 2*ig + 1]),
                   @inbounds(smem_A[(g+8)*16    + 2*ig + 2]))
    a3 = bf16_pack(@inbounds(smem_A[g*16        + 2*ig + 9]),
                   @inbounds(smem_A[g*16        + 2*ig + 10]))
    a4 = bf16_pack(@inbounds(smem_A[(g+8)*16    + 2*ig + 9]),
                   @inbounds(smem_A[(g+8)*16    + 2*ig + 10]))

    # B col-major linear index: col*16 + row  (col=group, rows=2*ig+{0,1,8,9})
    b1 = bf16_pack(@inbounds(smem_B[g*16 + 2*ig + 1]),
                   @inbounds(smem_B[g*16 + 2*ig + 2]))
    b2 = bf16_pack(@inbounds(smem_B[g*16 + 2*ig + 9]),
                   @inbounds(smem_B[g*16 + 2*ig + 10]))

    a = (a1, a2, a3, a4)
    b = (b1, b2)
    c = (0f0, 0f0, 0f0, 0f0)

    d = ptx"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"(a, b, c)

    # --- Stage 3: store fragments to D.
    # D row-major linear index: row*8 + col
    @inbounds D[g*8     + 2*ig + 1] = d[1]
    @inbounds D[g*8     + 2*ig + 2] = d[2]
    @inbounds D[(g+8)*8 + 2*ig + 1] = d[3]
    @inbounds D[(g+8)*8 + 2*ig + 2] = d[4]
    return nothing
end

# --- correctness checks -----------------------------------------------------

# Pack a Float32 matrix (row-major Julia layout) to bf16 UInt16 storage,
# row-major flattened. Each output is bf16 of the input value.
function pack_bf16_row(A::AbstractMatrix{Float32})
    m, n = size(A)
    out = Vector{UInt16}(undef, m * n)
    @inbounds for i in 1:m, j in 1:n
        out[(i-1)*n + j] = bf16_bits(A[i, j])
    end
    out
end

# Pack a Float32 matrix to bf16 UInt16 storage, column-major flattened.
function pack_bf16_col(B::AbstractMatrix{Float32})
    m, n = size(B)
    out = Vector{UInt16}(undef, m * n)
    @inbounds for j in 1:n, i in 1:m
        out[(j-1)*m + i] = bf16_bits(B[i, j])
    end
    out
end

# Quantize a Float32 matrix to bf16 precision (round-to-truncate by passing
# through bf16 storage and back). Used for the reference matmul so we
# compare apples-to-apples.
quantize_bf16(A) = bf16_to_f32.(bf16_bits.(A))

@testset "GEMM tile (manual frag): A=B=1.0, C=0 → D == 16.0" begin
    A_f32 = ones(Float32, 16, 16)
    B_f32 = ones(Float32, 16, 8)
    A_d = CuArray(pack_bf16_row(A_f32))
    B_d = CuArray(pack_bf16_col(B_f32))
    D_d = CUDACore.zeros(Float32, 16 * 8)
    @cuda threads=32 gemm_tile_kernel!(D_d, A_d, B_d)
    CUDACore.synchronize()
    D = reshape(Array(D_d), 8, 16)'    # row-major flat → 16×8 view
    @test all(D .== 16f0)
end

@testset "GEMM tile (manual frag): non-trivial A, B" begin
    # Small magnitudes so 16-term accumulation stays inside bf16 dynamic range.
    A_f32 = quantize_bf16(Float32[(i + j) / 100 for i in 1:16, j in 1:16])
    B_f32 = quantize_bf16(Float32[(i - j) / 100 for i in 1:16, j in 1:8])
    A_d = CuArray(pack_bf16_row(A_f32))
    B_d = CuArray(pack_bf16_col(B_f32))
    D_d = CUDACore.zeros(Float32, 16 * 8)
    @cuda threads=32 gemm_tile_kernel!(D_d, A_d, B_d)
    CUDACore.synchronize()
    D = reshape(Array(D_d), 8, 16)'
    D_ref = A_f32 * B_f32
    # bf16 mantissa is 8 bits; per-element relative error after a 16-term
    # accumulation is well under 1%.
    @test maximum(abs.(D .- D_ref) ./ max.(abs.(D_ref), 1f-6)) < 1f-2
end

# ============================================================================
# Phase 2: replace A's manual fragment construction with ldmatrix.x4.b16.
# ============================================================================
#
# `ldmatrix.sync.aligned.m8n8.x4.shared.b16` loads four 8×8 b16 fragment
# matrices from shared memory in one warp-cooperative instruction. Each lane
# provides ONE row-pointer; the 32 lanes are split into 4 octets, one octet
# per fragment matrix:
#
#     octet 0 (lanes 0..7)   → frag 0 row pointers (8 rows of a 8×8 b16 matrix)
#     octet 1 (lanes 8..15)  → frag 1 row pointers
#     octet 2 (lanes 16..23) → frag 2 row pointers
#     octet 3 (lanes 24..31) → frag 3 row pointers
#
# Each lane receives 8 b16 = 4 UInt32 (NTuple{4, UInt32}). The element ordering
# `l[1..4]` corresponds to fragments [0, 1, 2, 3]. For the m16n8k16.row.col
# mma the four submatrices we need to load are:
#
#     frag 0 = A[rows 0..7,  cols 0..7]    (top-left)
#     frag 1 = A[rows 8..15, cols 0..7]    (bot-left)
#     frag 2 = A[rows 0..7,  cols 8..15]   (top-right)
#     frag 3 = A[rows 8..15, cols 8..15]   (bot-right)
#
# This matches mma's expected `a[1..4]` layout exactly, so the ldmatrix
# return tuple feeds into mma without re-shuffling.

function gemm_tile_ldmatrix_kernel!(D, A_bits, B_bits)
    smem_A = CuStaticSharedArray(UInt16, 256)
    smem_B = CuStaticSharedArray(UInt16, 128)

    tid = ptx"mov.u32"(sreg"tid.x")

    src_A = pointer(A_bits) + Int(tid) * 16
    dst_A = pointer(smem_A) + Int(tid) * 16
    ptx"cp.async.ca.shared.global"(dst_A, src_A, Val(16))

    if tid < UInt32(16)
        src_B = pointer(B_bits) + Int(tid) * 16
        dst_B = pointer(smem_B) + Int(tid) * 16
        ptx"cp.async.ca.shared.global"(dst_B, src_B, Val(16))
    end

    ptx"cp.async.commit_group"()
    ptx"cp.async.wait_all"()
    ptx"bar.sync"(Val(0))

    # Per-lane row pointer for ldmatrix.x4.
    octet      = Int(tid >> UInt32(3))      # 0..3
    in_octet   = Int(tid & UInt32(0x7))     # 0..7
    row_block  = octet & 1                  # 0 → top half, 1 → bot half
    col_block  = octet >> 1                 # 0 → left half, 1 → right half
    row_in_A   = row_block * 8 + in_octet
    col_in_A   = col_block * 8
    a_addr     = pointer(smem_A) + (row_in_A * 16 + col_in_A) * sizeof(UInt16)
    a          = ptx"ldmatrix.sync.aligned.m8n8.x4.shared.b16"(a_addr)

    # B still uses manual fragment construction. Phase 3 would swap in
    # ldmatrix.x2 with `.trans` (or restructure shared layout).
    group    = tid >> UInt32(2)
    in_group = tid & UInt32(0x3)
    g        = Int(group)
    ig       = Int(in_group)

    b1 = bf16_pack(@inbounds(smem_B[g*16 + 2*ig + 1]),
                   @inbounds(smem_B[g*16 + 2*ig + 2]))
    b2 = bf16_pack(@inbounds(smem_B[g*16 + 2*ig + 9]),
                   @inbounds(smem_B[g*16 + 2*ig + 10]))
    b = (b1, b2)
    c = (0f0, 0f0, 0f0, 0f0)

    d = ptx"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"(a, b, c)

    @inbounds D[g*8     + 2*ig + 1] = d[1]
    @inbounds D[g*8     + 2*ig + 2] = d[2]
    @inbounds D[(g+8)*8 + 2*ig + 1] = d[3]
    @inbounds D[(g+8)*8 + 2*ig + 2] = d[4]
    return nothing
end

@testset "GEMM tile (ldmatrix.x4): A=B=1.0 → D == 16.0" begin
    A_f32 = ones(Float32, 16, 16)
    B_f32 = ones(Float32, 16, 8)
    A_d = CuArray(pack_bf16_row(A_f32))
    B_d = CuArray(pack_bf16_col(B_f32))
    D_d = CUDACore.zeros(Float32, 16 * 8)
    @cuda threads=32 gemm_tile_ldmatrix_kernel!(D_d, A_d, B_d)
    CUDACore.synchronize()
    D = reshape(Array(D_d), 8, 16)'
    @test all(D .== 16f0)
end

@testset "GEMM tile (ldmatrix.x4): non-trivial A, B" begin
    A_f32 = quantize_bf16(Float32[(i + j) / 100 for i in 1:16, j in 1:16])
    B_f32 = quantize_bf16(Float32[(i - j) / 100 for i in 1:16, j in 1:8])
    A_d = CuArray(pack_bf16_row(A_f32))
    B_d = CuArray(pack_bf16_col(B_f32))
    D_d = CUDACore.zeros(Float32, 16 * 8)
    @cuda threads=32 gemm_tile_ldmatrix_kernel!(D_d, A_d, B_d)
    CUDACore.synchronize()
    D = reshape(Array(D_d), 8, 16)'
    D_ref = A_f32 * B_f32
    @test maximum(abs.(D .- D_ref) ./ max.(abs.(D_ref), 1f-6)) < 1f-2
end

# ============================================================================
# Phase 3: ldmatrix.x2.b16 for B (no .trans needed for col-major B).
# ============================================================================
#
# B is 16×8 bf16 col-major in global. cp.async preserves the layout in shared,
# so storage has 8 contiguous columns of 16 b16 each (32 bytes per column,
# 256 bytes total).
#
# Plain ldmatrix (no .trans) loads each storage row as one row of an 8×8
# fragment. Per-lane delivery: lane T = (groupID = T/4, tIG = T%4) receives
# 2 b16 from row `groupID` of the fragment, cols `2*tIG..2*tIG+1`. Re-writing
# those indices back to the storage layout: "row groupID" of the fragment is
# "the first 8 bf16 of storage row groupID", i.e. "rows 0..7 of B column
# groupID"; "cols 2*tIG..+1" of the fragment is "rows 2*tIG..+1 of B column
# groupID". Which is *exactly* mma's expected b[1] = (B[2*tIG, groupID],
# B[2*tIG+1, groupID]).
#
# So plain ldmatrix.x2 on col-major shared B delivers the right b[1]/b[2]
# fragment to mma without a `.trans`. The two source 8×8 fragments map to:
#
#     frag 0 = B[rows 0..7,  cols 0..7]   →  l[1] feeds mma's b[1]
#     frag 1 = B[rows 8..15, cols 0..7]   →  l[2] feeds mma's b[2]
#
# Octet 0 lanes (0..7) provide row pointers for frag 0; octet 1 lanes (8..15)
# for frag 1. Lanes 16..31 are inactive but still need a valid shared
# address (any element of smem_B works).

function gemm_tile_ldmatrix_full_kernel!(D, A_bits, B_bits)
    smem_A = CuStaticSharedArray(UInt16, 256)
    smem_B = CuStaticSharedArray(UInt16, 128)

    tid = ptx"mov.u32"(sreg"tid.x")

    src_A = pointer(A_bits) + Int(tid) * 16
    dst_A = pointer(smem_A) + Int(tid) * 16
    ptx"cp.async.ca.shared.global"(dst_A, src_A, Val(16))

    if tid < UInt32(16)
        src_B = pointer(B_bits) + Int(tid) * 16
        dst_B = pointer(smem_B) + Int(tid) * 16
        ptx"cp.async.ca.shared.global"(dst_B, src_B, Val(16))
    end

    ptx"cp.async.commit_group"()
    ptx"cp.async.wait_all"()
    ptx"bar.sync"(Val(0))

    # ldmatrix.x4 for A.
    octet      = Int(tid >> UInt32(3))
    in_octet   = Int(tid & UInt32(0x7))
    row_block  = octet & 1
    col_block  = octet >> 1
    row_in_A   = row_block * 8 + in_octet
    col_in_A   = col_block * 8
    a_addr     = pointer(smem_A) + (row_in_A * 16 + col_in_A) * sizeof(UInt16)
    a          = ptx"ldmatrix.sync.aligned.m8n8.x4.shared.b16"(a_addr)

    # ldmatrix.x2 for B. octet 0 → frag 0 (B rows 0..7), octet 1 → frag 1
    # (B rows 8..15). Lanes 16..31 are inactive but need a valid address;
    # we compute one anyway via clamping the byte offset within smem_B.
    half_offset = (octet & 1) * 8
    b_byte_off  = (in_octet * 16 + half_offset) * sizeof(UInt16)
    b_addr      = pointer(smem_B) + b_byte_off
    b           = ptx"ldmatrix.sync.aligned.m8n8.x2.shared.b16"(b_addr)

    c = (0f0, 0f0, 0f0, 0f0)
    d = ptx"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"(a, b, c)

    group    = tid >> UInt32(2)
    in_group = tid & UInt32(0x3)
    g  = Int(group)
    ig = Int(in_group)

    @inbounds D[g*8     + 2*ig + 1] = d[1]
    @inbounds D[g*8     + 2*ig + 2] = d[2]
    @inbounds D[(g+8)*8 + 2*ig + 1] = d[3]
    @inbounds D[(g+8)*8 + 2*ig + 2] = d[4]
    return nothing
end

@testset "GEMM tile (ldmatrix.x4 + .x2): A=B=1.0 → D == 16.0" begin
    A_f32 = ones(Float32, 16, 16)
    B_f32 = ones(Float32, 16, 8)
    A_d = CuArray(pack_bf16_row(A_f32))
    B_d = CuArray(pack_bf16_col(B_f32))
    D_d = CUDACore.zeros(Float32, 16 * 8)
    @cuda threads=32 gemm_tile_ldmatrix_full_kernel!(D_d, A_d, B_d)
    CUDACore.synchronize()
    D = reshape(Array(D_d), 8, 16)'
    @test all(D .== 16f0)
end

@testset "GEMM tile (ldmatrix.x4 + .x2): non-trivial A, B" begin
    A_f32 = quantize_bf16(Float32[(i + j) / 100 for i in 1:16, j in 1:16])
    B_f32 = quantize_bf16(Float32[(i - j) / 100 for i in 1:16, j in 1:8])
    A_d = CuArray(pack_bf16_row(A_f32))
    B_d = CuArray(pack_bf16_col(B_f32))
    D_d = CUDACore.zeros(Float32, 16 * 8)
    @cuda threads=32 gemm_tile_ldmatrix_full_kernel!(D_d, A_d, B_d)
    CUDACore.synchronize()
    D = reshape(Array(D_d), 8, 16)'
    D_ref = A_f32 * B_f32
    @test maximum(abs.(D .- D_ref) ./ max.(abs.(D_ref), 1f-6)) < 1f-2
end
