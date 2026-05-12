using PTX: GmmaLayout, pick_gmma_layout, layout_for_a, layout_for_mn_major,
           swizzle_code, WgmmaSwizzle

# Mirrors pyptx/tests/test_wgmma_layout.py. Same shapes, same expected
# byte offsets — the two pickers should agree exactly so a Julia kernel
# can be cross-checked against pyptx's reference.

@testset "K-major (row-major A)" begin
    # K=8 bf16 → row 16 B = 1 u128 → INTERLEAVE.
    la = pick_gmma_layout(elem_bytes=2, m_or_n=64, k=8, major=:K)
    @test la.layout_type == WgmmaSwizzle.NONE
    @test swizzle_code(la) == 0
    @test la.tma_swizzle == :NONE
    @test la.smem_swizzle === nothing
    @test la.leading_byte_offset == 16
    @test la.stride_byte_offset == 8 * 16

    # K=16 bf16 → row 32 B → B32 (the canonical bf16 m64n*k16 case).
    la = pick_gmma_layout(elem_bytes=2, m_or_n=64, k=16, major=:K)
    @test la.layout_type == WgmmaSwizzle.B32
    @test swizzle_code(la) == 3
    @test la.tma_swizzle == :B32
    @test la.smem_swizzle == :B32
    @test la.leading_byte_offset == 16
    @test la.stride_byte_offset == 8 * 32

    # K=32 bf16 → row 64 B → B64.
    la = pick_gmma_layout(elem_bytes=2, m_or_n=64, k=32, major=:K)
    @test la.layout_type == WgmmaSwizzle.B64
    @test swizzle_code(la) == 2
    @test la.tma_swizzle == :B64
    @test la.stride_byte_offset == 8 * 64

    # K=64 bf16 → row 128 B → B128.
    la = pick_gmma_layout(elem_bytes=2, m_or_n=64, k=64, major=:K)
    @test la.layout_type == WgmmaSwizzle.B128
    @test swizzle_code(la) == 1
    @test la.tma_swizzle == :B128
    @test la.stride_byte_offset == 8 * 128

    # K=128 bf16 → row 256 B → multi-stripe B128. LBO carries the
    # inter-stripe distance (M * 128); SBO is the within-stripe stride.
    la = pick_gmma_layout(elem_bytes=2, m_or_n=64, k=128, major=:K)
    @test la.layout_type == WgmmaSwizzle.B128
    @test la.tma_swizzle == :B128
    @test la.smem_swizzle == :B128
    @test la.leading_byte_offset == 64 * 128
    @test la.stride_byte_offset == 8 * 128

    # K=4 f32 → row 16 B → INTERLEAVE (the f32 degenerate row).
    la = pick_gmma_layout(elem_bytes=4, m_or_n=64, k=4, major=:K)
    @test la.layout_type == WgmmaSwizzle.NONE

    # Bad row-width: K=12 bf16 = 24 B → not in {16,32,64,128}.
    @test_throws ArgumentError pick_gmma_layout(
        elem_bytes=2, m_or_n=64, k=12, major=:K)

    # M not divisible by 8 → reject.
    @test_throws ArgumentError pick_gmma_layout(
        elem_bytes=2, m_or_n=63, k=16, major=:K)
end

@testset "MN-major (row-major B)" begin
    # N=8 bf16 → row 16 B → INTERLEAVE. Degenerate N = 1 u128: LBO carries
    # the K-slab stride (8 K-rows × 16 B/row = 128 B).
    lb = pick_gmma_layout(elem_bytes=2, m_or_n=8, k=16, major=:MN)
    @test lb.layout_type == WgmmaSwizzle.NONE
    @test lb.leading_byte_offset == 8 * 16

    lb = pick_gmma_layout(elem_bytes=2, m_or_n=16, k=16, major=:MN)
    @test lb.layout_type == WgmmaSwizzle.B32
    @test lb.leading_byte_offset == 16
    @test lb.stride_byte_offset == 8 * 32

    # K not divisible by 8 → reject.
    @test_throws ArgumentError pick_gmma_layout(
        elem_bytes=2, m_or_n=8, k=15, major=:MN)
end

@testset "layout_for_a / layout_for_mn_major helpers" begin
    # bf16 m=64 k=16 — the canonical Hopper GEMM A tile.
    la = layout_for_a(dtype=:bf16, m=64, k=16)
    @test la.layout_type == WgmmaSwizzle.B32
    @test la.stride_byte_offset == 256

    # bf16 k=16 n=8 — degenerate single-warpgroup B tile (MN-major variant).
    lb = layout_for_mn_major(dtype=:bf16, k=16, n=8)
    @test lb.layout_type == WgmmaSwizzle.NONE
    @test lb.leading_byte_offset == 128

    # f16 same width as bf16 → same layout.
    @test layout_for_a(dtype=:f16, m=64, k=16).layout_type == WgmmaSwizzle.B32

    # f32 k=8 → row 32 B → B32.
    @test layout_for_a(dtype=:f32, m=64, k=8).layout_type == WgmmaSwizzle.B32

    # bf16 k=128 multi-stripe A.
    la = layout_for_a(dtype=:bf16, m=64, k=128)
    @test la.layout_type == WgmmaSwizzle.B128
    @test la.leading_byte_offset == 64 * 128
    @test la.stride_byte_offset == 8 * 128

    # Type-based dtype works too: sizeof(Float32) == 4.
    @test layout_for_a(dtype=Float32, m=64, k=8).layout_type == WgmmaSwizzle.B32
end

@testset "distinct row widths → distinct layouts" begin
    pairs = [(2, 8), (2, 16), (2, 32), (2, 64)]
    families = Set{UInt8}()
    for (eb, k) in pairs
        la = pick_gmma_layout(elem_bytes=eb, m_or_n=64, k=k, major=:K)
        push!(families, la.layout_type)
    end
    @test families == Set([WgmmaSwizzle.NONE, WgmmaSwizzle.B32,
                           WgmmaSwizzle.B64,  WgmmaSwizzle.B128])
end

@testset "feeds wgmma_descriptor" begin
    # End-to-end shape: layout result plugs into wgmma_descriptor with the
    # right field types (swizzle::Integer, leading/stride_byte_offset::Integer).
    la = layout_for_a(dtype=:bf16, m=64, k=16)
    desc = PTX.wgmma_descriptor(
        UInt32(0);
        leading_byte_offset = la.leading_byte_offset,
        stride_byte_offset  = la.stride_byte_offset,
        swizzle             = swizzle_code(la))
    # B32 → swizzle field (bits 63:62) == 3.
    @test (desc >> 62) & 0x3 == 3
    # stride field (bits 45:32) = (256 & 0x3FFFF) >> 4 = 16.
    @test (desc >> 32) & 0x3FFF == 16
    # leading field (bits 29:16) = (16 & 0x3FFFF) >> 4 = 1.
    @test (desc >> 16) & 0x3FFF == 1
end

@testset "step_desc advances start_addr by byte_offset/16" begin
    # Build a real-shaped descriptor at smem_addr=0, swizzle=B32. The
    # start_addr field lives in bits [13:0]. After step_desc(d, n * 16)
    # the field should read `n` (mod the field width).
    la = layout_for_a(dtype=:bf16, m=64, k=16)
    base = PTX.wgmma_descriptor(
        UInt32(0);
        leading_byte_offset = la.leading_byte_offset,
        stride_byte_offset  = la.stride_byte_offset,
        swizzle             = swizzle_code(la))
    @test base & 0x3FFF == 0                                # baseline start_addr
    @test PTX.step_desc(base, 16)   & 0x3FFF == 1
    @test PTX.step_desc(base, 32)   & 0x3FFF == 2
    @test PTX.step_desc(base, 2048) & 0x3FFF == 128         # typical A-stage step
    @test PTX.step_desc(base, 0)            == base         # identity

    # Higher fields (leading/stride/swizzle) are unchanged — step_desc only
    # touches the low 14 bits in practice.
    @test PTX.step_desc(base, 2048) >> 16 == base >> 16
end
