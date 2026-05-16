# Blackwell tcgen05 TMEM addressing + K-major swizzle / layout helpers.
#
# Companion to src/wgmma_layout.jl (the sm_90a analogue). tcgen05_descriptor.jl
# is the *bit-packer*; this is the *layout-picking* + index-math layer that
# friction #1 of BLACKWELL_INTERFACE_NOTES.md flagged as missing — every
# kernel was hand-passing `leading_bytes=16, stride_bytes=K*16` and inlining
# the swizzle math (the latter died with gemm_experimental_blackwell).
#
# Faithful, value-for-value ports of:
#   * pyptx examples/blackwell/gemm_experimental_blackwell.py
#       kmajor_swizzle / kmajor_swizzled_logical_bytes
#   * pyptx pyptx/smem.py apply_swizzle  (CUTLASS Swizzle<B,4,3>: shift 3,
#       mask {32B:0x080, 64B:0x180, 128B:0x380})
#   * pyptx pyptx/ptx.py _Tcgen05.descriptor const_bits — the *production*
#       gemm_highperf_blackwell MMA_DESC_B128 path (leading=16,
#       stride=K*16, swizzle by K-major row width), NOT a guess.
#
# Every function here is pure integer arithmetic (no Dict / no throw on the
# hot path) so it compiles unchanged inside a device kernel and is exercised
# host-side in test/host/tcgen05_layout.jl.

# ---------------------------------------------------------------------------
# TMEM addressing — PTX 9.2 §9.7.16.1 / PTX_ISA_DIGEST.md §11.1
# ---------------------------------------------------------------------------
#
# 32-bit TMEM address = bits[31:16] lane index | bits[15:0] column.
# §11.3 lane-band: warp i (0..3 in a warpgroup) accesses lanes 32i..32i+31.
#
# This replaces the two hand-rolled masks copied verbatim across the probes:
#   per-lane  (warp 0, 32 rows): tmem + ((lane<<16) & 0x1F0000), lane=tid&31
#   warp-band (128-thread epi):  tmem + ((tid <<16) & 0x03E00000)
# Both are exactly `tmem_lane_addr(tmem, <lane-index>)`; the call-site
# difference is now a readable lane expression, not two unrelated masks.
# (Bit-exact equivalence over tid∈0:127 is asserted in the host test.)

@inline tmem_lane_addr(tmem::UInt32, lane::UInt32) = tmem + (lane << UInt32(16))

# §11.3 warp-band base lane for a 128-thread epilogue: thread `tid` is in
# warp tid>>5, whose TMEM lane band starts at 32*(tid>>5). Bit-for-bit
# equal (tid∈0:127) to the old `(tid<<16) & 0x03E00000`.
@inline tmem_warp_band_lane(tid::UInt32) = (tid >> UInt32(5)) << UInt32(5)

# ---------------------------------------------------------------------------
# K-major swizzle family + index math
# ---------------------------------------------------------------------------

# Row width (in 2-byte elements) → Blackwell K-major swizzle family code
# (a `BlackwellLayout` value, usable directly as the descriptor swizzle).
# pyptx kmajor_swizzle: row_bytes = elems*2; ≥128→128B, ≥64→64B, ≥32→32B.
@inline function kmajor_swizzle(row_stride_elems::Integer)
    row_bytes = Int(row_stride_elems) * 2
    row_bytes >= 128 && return BlackwellLayout.B128
    row_bytes >=  64 && return BlackwellLayout.B64
    row_bytes >=  32 && return BlackwellLayout.B32
    throw(ArgumentError(
        "unsupported Blackwell K-major row width: $row_stride_elems elems " *
        "($row_bytes B); need ≥16 elements (≥32 B)"))
end

# Contiguous elements per swizzle atom: 32B→16, 64B→32, 128B→64.
@inline function _kmajor_contig(sw::Integer)
    s = UInt8(sw)
    s == BlackwellLayout.B128 && return 64
    s == BlackwellLayout.B64  && return 32
    s == BlackwellLayout.B32  && return 16
    throw(ArgumentError("no contiguous-element width for swizzle code $s"))
end

# pyptx kmajor_swizzled_logical_bytes(row, k_elem, row_stride_elems,
# mn_extent) — mn_extent is unused upstream (`del mn_extent`) and dropped.
# Returns the *logical* (pre-swizzle) byte offset of element (row, k_elem)
# within a K-major operand tile.
@inline function kmajor_swizzled_logical_bytes(row::Integer, k_elem::Integer,
                                               row_stride_elems::Integer)
    contig          = _kmajor_contig(kmajor_swizzle(row_stride_elems))
    row_group       = Int(row) >> 3
    row_in_group    = Int(row) & 7
    row_group_bytes = contig * 8 * 2
    row_group * row_group_bytes + ((row_in_group * contig) + Int(k_elem)) * 2
end

# CUTLASS Swizzle<B,4,3>: physical = logical ⊻ ((logical & mask) >> 3).
# mask = {NONE:0, B32:0x080, B64:0x180, B128:0x380}; NONE = identity.
# pyptx pyptx/smem.py apply_swizzle.
@inline function _blackwell_swizzle_mask(sw::Integer)
    s = UInt8(sw)
    s == BlackwellLayout.B128 && return UInt32(0x380)
    s == BlackwellLayout.B64  && return UInt32(0x180)
    s == BlackwellLayout.B32  && return UInt32(0x080)
    UInt32(0)                                   # NONE / unknown → identity
end

@inline function apply_blackwell_swizzle(logical::UInt32, sw::Integer)
    m = _blackwell_swizzle_mask(sw)
    m == UInt32(0) ? logical : logical ⊻ ((logical & m) >> UInt32(3))
end

# ---------------------------------------------------------------------------
# Layout picker — the Blackwell analogue of wgmma_layout's `layout_for_a`
# ---------------------------------------------------------------------------

struct Tcgen05Layout
    swizzle::UInt8           # BlackwellLayout.{B32,B64,B128}
    leading_bytes::Int
    stride_bytes::Int
end

# K-major bf16/f16 operand layout, exactly the *maintained* pyptx
# gemm_highperf_blackwell path: leading_bytes=16, stride_bytes=K*16,
# swizzle = kmajor_swizzle(K). For K=64 this reproduces the production
# constant MMA_DESC_B128 = 0x4000404000010000 bit-for-bit (asserted in
# the host test via tcgen05_descriptor) — so this is the canonical
# picker, not the conjectured-suspect constant from the deleted debug
# kernel. Plug straight into `tcgen05_descriptor`:
#   l = tcgen05_layout_kmajor(k=BK)
#   tcgen05_descriptor(addr; leading_bytes=l.leading_bytes,
#                       stride_bytes=l.stride_bytes, swizzle=l.swizzle)
@inline function tcgen05_layout_kmajor(; k::Integer)
    Tcgen05Layout(kmajor_swizzle(k), 16, Int(k) * 16)
end
