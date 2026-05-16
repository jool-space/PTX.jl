using PTX: tmem_lane_addr, tmem_warp_band_lane,
           kmajor_swizzle, kmajor_swizzled_logical_bytes,
           apply_blackwell_swizzle, tcgen05_layout_kmajor,
           tcgen05_descriptor, BlackwellLayout

# Companion to test/host/wgmma_layout.jl. Two things are proved here:
#   1. tmem_lane_addr / tmem_warp_band_lane are a *bit-exact* no-op
#      refactor of the two magic masks the blackwell probes inlined.
#   2. The swizzle/layout helpers match the pyptx reference value-for-value
#      (inline Python-formula oracles), and the K-major picker reproduces
#      the maintained production constant MMA_DESC_B128.

@testset "tmem_lane_addr ≡ the old magic masks (bit-exact, tid 0:127)" begin
    tmem = UInt32(0x00070000)                       # arbitrary alloc base
    for tid in UInt32(0):UInt32(127)
        lane = tid & UInt32(31)
        # per-lane warp-0 pattern (roundtrip / mma_probe / accum_probe):
        old_lane = tmem + ((lane << UInt32(16)) & UInt32(0x1F0000))
        @test tmem_lane_addr(tmem, lane) == old_lane

        # 128-thread warp-band epilogue pattern (grouped_gemm):
        old_band = tmem + ((tid << UInt32(16)) & UInt32(0x03E00000))
        @test tmem_lane_addr(tmem, tmem_warp_band_lane(tid)) == old_band
    end
    # lane index lands in bits[31:16]; column bits[15:0] untouched.
    @test tmem_lane_addr(UInt32(0), UInt32(5)) == UInt32(5) << 16
    @test tmem_warp_band_lane(UInt32(0))  == UInt32(0)
    @test tmem_warp_band_lane(UInt32(31)) == UInt32(0)
    @test tmem_warp_band_lane(UInt32(32)) == UInt32(32)
    @test tmem_warp_band_lane(UInt32(96)) == UInt32(96)
end

@testset "kmajor_swizzle vs pyptx (row_bytes thresholds)" begin
    # pyptx: row_bytes = elems*2; ≥128→128B, ≥64→64B, ≥32→32B, else raise.
    @test kmajor_swizzle(16) == BlackwellLayout.B32     # 32 B
    @test kmajor_swizzle(31) == BlackwellLayout.B32     # 62 B
    @test kmajor_swizzle(32) == BlackwellLayout.B64     # 64 B
    @test kmajor_swizzle(63) == BlackwellLayout.B64     # 126 B
    @test kmajor_swizzle(64) == BlackwellLayout.B128    # 128 B
    @test kmajor_swizzle(128) == BlackwellLayout.B128
    @test_throws ArgumentError kmajor_swizzle(8)        # 16 B — unsupported
end

@testset "kmajor_swizzled_logical_bytes vs pyptx formula" begin
    # Inline oracle = the exact pyptx function (mn_extent dropped upstream).
    contig_of(elems) = (elems*2 >= 128 ? 64 : elems*2 >= 64 ? 32 : 16)
    function oracle(row, k_elem, rse)
        contig = contig_of(rse)
        rg, rig = row >> 3, row & 7
        rg * (contig*8*2) + ((rig*contig) + k_elem)*2
    end
    for rse in (16, 32, 64), row in 0:23, kw in 0:(rse÷2 - 1)
        ke = kw << 1
        @test kmajor_swizzled_logical_bytes(row, ke, rse) == oracle(row, ke, rse)
    end
end

@testset "apply_blackwell_swizzle vs CUTLASS Swizzle<B,4,3>" begin
    # pyptx smem.py: physical = logical ⊻ ((logical & mask) >> 3).
    for (sw, mask) in ((BlackwellLayout.B32,  UInt32(0x080)),
                       (BlackwellLayout.B64,  UInt32(0x180)),
                       (BlackwellLayout.B128, UInt32(0x380)))
        for logical in UInt32(0):UInt32(4):UInt32(0x800)
            @test apply_blackwell_swizzle(logical, sw) ==
                  logical ⊻ ((logical & mask) >> UInt32(3))
        end
    end
    # NONE = identity.
    for logical in UInt32(0):UInt32(7):UInt32(0x400)
        @test apply_blackwell_swizzle(logical, BlackwellLayout.NONE) == logical
    end
    # 4-aligned in → 4-aligned out (the copy loop relies on this for b32).
    for logical in UInt32(0):UInt32(4):UInt32(0x400)
        @test apply_blackwell_swizzle(logical, BlackwellLayout.B128) % 4 == 0
    end
end

@testset "tcgen05_layout_kmajor feeds tcgen05_descriptor (== MMA_DESC_B128)" begin
    # K=64 bf16 → 128 B row → B128, leading=16, stride=1024. The maintained
    # pyptx gemm_highperf_blackwell uses MMA_DESC_B128 = 0x4000404000010000;
    # masked_descriptor(const_bits=MMA_DESC_B128) ≡ descriptor(leading=16,
    # stride=1024, swizzle=128B). Reproduce it bit-for-bit from the picker.
    l = tcgen05_layout_kmajor(k = 64)
    @test l.swizzle == BlackwellLayout.B128
    @test l.leading_bytes == 16
    @test l.stride_bytes  == 1024
    desc = tcgen05_descriptor(UInt32(0);
                              leading_bytes = l.leading_bytes,
                              stride_bytes  = l.stride_bytes,
                              swizzle       = l.swizzle)
    @test desc == 0x4000404000010000          # == pyptx production MMA_DESC_B128

    # Smaller K-tiles pick the smaller families (matches the 32B probes).
    @test tcgen05_layout_kmajor(k = 16).swizzle == BlackwellLayout.B32
    @test tcgen05_layout_kmajor(k = 16).stride_bytes == 256
    @test tcgen05_layout_kmajor(k = 32).swizzle == BlackwellLayout.B64
end
