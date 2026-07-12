# Mixed-dtype Hopper GEMM (int8 weights → bf16 in-mainloop upconvert) —
# ported from cutlass/examples/55_hopper_mixed_dtype_gemm (NVIDIA CUTLASS,
# BSD-3-Clause).
#
# The CUTLASS example runs a warp-specialized GEMM where one operand is a
# narrow integer type (int8/int4 weights) and the other is bf16: the
# narrow operand is dequantized to bf16 *inside the mainloop* with a
# group-wise scale, then fed to the regular bf16 GMMA. The narrow type
# always passes through the register file (converted by threads, not by
# TMA), which is the whole point — TMA moves raw quantized bytes, the
# math runs in bf16.
#
# What's ported:
#
#   - A stays bf16 (TMA, B32-swizzled, exactly as gemm_warpgroup.jl).
#   - B is int8, TMA-loaded as raw `:u8` bytes into a small staging tile
#     (16-byte rows → swizzle :NONE — wgmma never touches this buffer).
#   - In-mainloop dequant: each of the 128 threads owns one B element per
#     K-tile, converts s8 → f32 · scale → bf16 through the register file,
#     and stores it into a second SMEM tile shaped exactly like the bf16
#     B tile wgmma wants: K-fast, B32 swizzle family. The B32 physical
#     layout is cute's Swizzle<1,4,3> — byte address bit 7 XORs into
#     bit 4 — replicated by hand at the st.shared address computation
#     (that formula is what TMA's :B32 mode applies when it writes A —
#     verified empirically on H100 with a TMA dump probe; here the
#     conversion loop plays the role of TMA, so it must swizzle itself).
#     The converted tile base is 256-byte aligned in-kernel: the swizzle
#     XOR acts on ABSOLUTE SMEM address bits with a 256-B pattern period,
#     so a merely-128-B-aligned base can flit-swap every row
#     (CuStaticSharedArray only guarantees 32 B).
#   - Group-wise dequant scale with group_size = K, i.e. one scale per B
#     column — the README's "per-column scales via group_size = problem_k"
#     degenerate case. Scales are applied during conversion, before the
#     bf16 round, matching CUTLASS's convert-then-MMA order.
#
# Not ported: int4 packing / ValueShuffle register-friendly reordering
# (sub-byte dtypes need the u4 tensor-map path — separate corpus entry),
# the A↔B operand swap for TMA epilogues (per-lane store epilogue here),
# and warp specialization (single warpgroup + sequential K-loop keeps the
# new brick — the in-mainloop upconversion — isolated; the pipelined
# producer/consumer brick is gemm_pc_pipeline.jl's).
#
# Proxy choreography per K-tile (the subtle part):
#   TMA write (async proxy) → mbarrier wait → generic-proxy LOADS of raw
#   B are ordered by the mbarrier itself; the conversion's generic-proxy
#   STORES need `fence.proxy.async.shared::cta` + `bar.sync` before wgmma
#   (async proxy) may read them — same generic→async direction every
#   hand-written-SMEM + wgmma kernel in this directory fences.

using PTX: layout_for_a, wgmma_descriptor, smem_addr_u32, tensor_map_tile_2d
using CUDACore
using Random

const MD_BM      = 64
const MD_BN      = 8
const MD_BK      = 16
const MD_THREADS = 128                       # one warpgroup

const MD_A_BYTES    = MD_BM * MD_BK * 2      # bf16 A tile      (2048)
const MD_BRAW_BYTES = MD_BN * MD_BK          # raw s8 B tile    (128)
const MD_LOAD_BYTES = MD_A_BYTES + MD_BRAW_BYTES

# f32 → bf16 bits, round-to-nearest (same formula as setup.jl's bf16_bits,
# inlined here so the device kernel doesn't depend on a test-harness helper).
@inline _md_bf16_bits(x::Float32) =
    UInt16((reinterpret(UInt32, x) + UInt32(0x8000)) >> 16)

function _md_gemm_kernel!(
        D::CuDeviceVector{Float32, 1},
        scale_B::CuDeviceVector{Float32, 1},   # one dequant scale per B column
        tma_A::PTX.TMADescriptorPtr,
        tma_Braw::PTX.TMADescriptorPtr,
        K::Int32)

    smem_A    = CuStaticSharedArray(UInt16, MD_BM * MD_BK)
    smem_Braw = CuStaticSharedArray(UInt8,  MD_BN * MD_BK)
    # Converted-B tile (256 B) + 256 B slack for in-kernel 256-B alignment.
    smem_Bcvt = CuStaticSharedArray(UInt16, MD_BN * MD_BK + 128)
    mbar      = CuStaticSharedArray(UInt64, 1)

    a_ptr    = pointer(smem_A)
    braw_ptr = pointer(smem_Braw)
    mb_ptr   = pointer(mbar)
    a_addr   = smem_addr_u32(a_ptr)

    # 256-B-align the converted tile: the B32 swizzle XOR reads ABSOLUTE
    # SMEM address bit 7 into bit 4 (pattern period = 256 B), so the tile
    # must start on a 256-B boundary for the tile-relative formula below to
    # match what wgmma dereferences. 128-B alignment is NOT enough: a base
    # ≡ 128 (mod 256) flips bit 7 for every row and the whole tile reads
    # flit-swapped (observed on H100; debugged via an identity-A probe).
    bcvt_addr_raw = smem_addr_u32(pointer(smem_Bcvt))
    bcvt_addr     = (bcvt_addr_raw + UInt32(255)) & ~UInt32(255)
    pad_elems     = Int(bcvt_addr - bcvt_addr_raw) >> 1

    tid = ptx"mov.u32"(sreg"tid.x")

    # Thread → B element ownership: 8 N-rows × 16 K = 128 elements, one each.
    n_row = tid >> UInt32(4)                 # 0..7
    k_col = tid & UInt32(15)                 # 0..15
    my_scale = @inbounds scale_B[Int(n_row) + 1]

    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mb_ptr, UInt32(1))
        ptx"fence.proxy.async.shared::cta"()
    end
    ptx"bar.sync"(Val(0))

    # Descriptors built once; SMEM tiles are reused every K-iter (no ring —
    # sequential loop, bar.sync-separated).
    la = layout_for_a(dtype = :bf16, m = MD_BM, k = MD_BK)
    lb = layout_for_a(dtype = :bf16, m = MD_BN, k = MD_BK)
    a_desc = wgmma_descriptor(a_addr;
        leading_byte_offset = la.leading_byte_offset,
        stride_byte_offset  = la.stride_byte_offset,
        swizzle             = la.layout_type)
    b_desc = wgmma_descriptor(bcvt_addr;
        leading_byte_offset = lb.leading_byte_offset,
        stride_byte_offset  = lb.stride_byte_offset,
        swizzle             = lb.layout_type)

    d = ntuple(_ -> 0f0, Val(4))
    num_k_tiles = K >> Int32(4)

    @inbounds for k_iter in Int32(0):(num_k_tiles - Int32(1))
        # ── TMA-load bf16 A + raw s8 B for this K-tile ──────────────────
        if tid == UInt32(0)
            ptx"fence.proxy.async.shared::cta"()
            ptx"mbarrier.arrive.expect_tx.shared.b64"(mb_ptr, UInt32(MD_LOAD_BYTES))
            k_off = k_iter * Int32(MD_BK)    # elements for A; bytes for :u8 B
            ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                a_ptr, tma_A, k_off, Int32(0), mb_ptr)
            ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
                braw_ptr, tma_Braw, k_off, Int32(0), mb_ptr)
        end
        # Single mbarrier, one expect_tx arrive per iter → parity flips
        # every iteration.
        while !ptx"mbarrier.test_wait.parity.shared.b64"(mb_ptr, UInt32(k_iter) & UInt32(1))
        end

        # ── In-mainloop dequant: s8 → f32·scale → bf16, swizzled store ──
        # Raw tile is K-fast (16 B rows): element (n, k) at byte n*16 + k;
        # thread tid owns exactly index tid.
        raw  = smem_Braw[Int(tid) + 1]
        f    = Float32(reinterpret(Int8, raw)) * my_scale
        bits = _md_bf16_bits(f)
        # Converted tile is the canonical K-major B32 layout: logical byte
        # offset n*32 + 2k, physical = logical ⊻ (bit7 → bit4) (cute
        # Swizzle<1,4,3> — the same pattern TMA :B32 writes).
        logical  = (n_row << UInt32(5)) + (k_col << UInt32(1))
        physical = logical ⊻ (((logical >> UInt32(7)) & UInt32(1)) << UInt32(4))
        smem_Bcvt[pad_elems + (Int(physical) >> 1) + 1] = bits

        # Conversions (generic proxy) must be CTA-visible and ordered
        # before wgmma's async-proxy reads.
        ptx"bar.sync"(Val(0))
        ptx"fence.proxy.async.shared::cta"()

        ptx"wgmma.fence.sync.aligned"()
        d = ptx"wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16"(
            d, a_desc, b_desc, true)
        ptx"wgmma.commit_group.sync.aligned"()
        ptx"wgmma.wait_group.sync.aligned"(Val(0))

        # All threads past the wgmma before lane 0 overwrites the tiles.
        ptx"bar.sync"(Val(0))
    end

    # ── Epilogue: per-lane v2.f32 stores (m64n8 frag layout, see
    # gemm_warpgroup.jl).
    wid      = tid >> UInt32(5)
    lane     = tid & UInt32(31)
    frag_row = (wid << UInt32(4)) + (lane >> UInt32(2))
    frag_col = (lane & UInt32(3)) << UInt32(1)
    pd       = pointer(D)
    off_a    = (frag_row * UInt32(MD_BN) + frag_col) * UInt32(4)
    off_b    = ((frag_row + UInt32(8)) * UInt32(MD_BN) + frag_col) * UInt32(4)
    ptx"st.global.v2.f32"(pd + Int(off_a), (d[1], d[2]))
    ptx"st.global.v2.f32"(pd + Int(off_b), (d[3], d[4]))
    return nothing
end

# ── Cross-arch ptxas validation ────────────────────────────────────────

@testset "mixed-dtype Hopper GEMM compiles at sm_90a" begin
    types = Tuple{CuDeviceVector{Float32, 1}, CuDeviceVector{Float32, 1},
                  PTX.TMADescriptorPtr, PTX.TMADescriptorPtr, Int32}
    @test ptxas_compiles(_md_gemm_kernel!, types;
                         cap = v"9.0", feature_set = :arch)

    ptx = emit_ptx(_md_gemm_kernel!, types; cap = v"9.0", feature_set = :arch)
    @test occursin("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16", ptx)
    @test occursin("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes", ptx)
    # The dequant path: s8 → f32 convert and the swizzled u16 store into
    # the converted tile.
    @test occursin("cvt.rn.f32.s", ptx)
    # the swizzled converted-tile store (NVPTX spells the 16-bit store .b16)
    @test occursin("st.shared.b16", ptx)
end

# ── Runtime — Hopper hardware ──────────────────────────────────────────

if v"9.0" <= DEV_CAP < v"10.0"
    @testset "mixed-dtype GEMM (bf16 A × int8 B, per-column scales, K=64)" begin
        rng = MersenneTwister(0x55d7)
        K_test = 64                          # 4 K-iters through one SMEM tile
        A_f32 = randn(rng, Float32, MD_BM, K_test) .* 0.1f0
        B_i8  = rand(rng, Int8(-8):Int8(8), K_test, MD_BN)
        # Per-column dequant scales — non-uniform so a column/scale
        # misalignment can't hide.
        sB = Float32[0.25f0 * n * (1.0f0 + 0.1f0 * rand(rng, Float32)) for n in 1:MD_BN]

        A_packed = Array{UInt16}(undef, K_test, MD_BM)
        for m in 1:MD_BM, k in 1:K_test
            A_packed[k, m] = bf16_bits(A_f32[m, k])
        end
        B_packed = Array{UInt8}(undef, K_test, MD_BN)
        for k in 1:K_test, n in 1:MD_BN
            B_packed[k, n] = reinterpret(UInt8, B_i8[k, n])
        end
        A_d  = CuArray(A_packed)
        B_d  = CuArray(B_packed)
        sB_d = CuArray(sB)

        tmap_A = tensor_map_tile_2d(:bf16, pointer(A_d),
            MD_BM, K_test, MD_BM, MD_BK; swizzle = :B32)
        # Raw B tile: 16-byte rows → no swizzle; threads (not wgmma) read it.
        tmap_B = tensor_map_tile_2d(:u8, pointer(B_d),
            MD_BN, K_test, MD_BN, MD_BK; swizzle = :NONE)
        A = upload_tma_descriptor(tmap_A)
        B = upload_tma_descriptor(tmap_B)

        D_dev = CUDACore.zeros(Float32, MD_BM * MD_BN)
        @cuda threads = MD_THREADS _md_gemm_kernel!(
            D_dev, sB_d, A.ptr, B.ptr, Int32(K_test))
        CUDACore.synchronize()

        D_packed = reshape(Array(D_dev), MD_BN, MD_BM)
        D_got    = Array{Float32}(undef, MD_BM, MD_BN)
        for m in 1:MD_BM, n in 1:MD_BN
            D_got[m, n] = D_packed[n, m]
        end

        # Reference replicates the kernel's dequant exactly: scale in f32,
        # round to bf16, then f32 accumulate over bf16-rounded A.
        D_ref = zeros(Float32, MD_BM, MD_BN)
        for m in 1:MD_BM, n in 1:MD_BN
            acc = 0f0
            for k in 1:K_test
                a = bf16_to_f32(bf16_bits(A_f32[m, k]))
                b = bf16_to_f32(bf16_bits(Float32(B_i8[k, n]) * sB[n]))
                acc += a * b
            end
            D_ref[m, n] = acc
        end
        @test isapprox(D_got, D_ref; rtol = 1e-3, atol = 1e-3)
    end
end
