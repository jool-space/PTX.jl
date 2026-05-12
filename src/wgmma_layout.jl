# Canonical GMMA shared-memory layouts for `wgmma.mma_async` (sm_90a).
#
# wgmma takes its A/B operands as 64-bit SMEM descriptors. The descriptor's
# `layout_type` field (bits [63:62]) plus `leading_byte_offset` / `stride_byte_offset`
# must match the SMEM tile's actual geometry — exactly one of four canonical
# layout families (INTERLEAVE / B32 / B64 / B128) is natural for a given
# (dtype, M/N, K, major) tile, and the TMA swizzle that wrote the tile must
# pair with the same family. Mismatch = silently wrong results.
#
# Pure port of pyptx/wgmma_layout.py — itself a port of the layout-picking
# branch of `cute/atom/mma_traits_sm90_gmma.hpp::make_gmma_desc`.
#
# Output is plugged into the existing `wgmma_descriptor(...)` builder
# (see src/wgmma_descriptor.jl): `swizzle = layout.swizzle_code`,
# `leading_byte_offset = layout.leading_byte_offset`,
# `stride_byte_offset = layout.stride_byte_offset`. The `tma_swizzle` /
# `smem_swizzle` symbols feed the host-side TMA encoder + smem allocator
# (items 4 and 12 in ROADMAP).

# `major` axis:
#   :K  — K dimension is fast (row-major A of shape (M,K), col-major B of (K,N))
#   :MN — the M or N dimension is fast (col-major A, row-major B)

struct GmmaLayout
    layout_type::UInt8           # WgmmaSwizzle.{NONE,B32,B64,B128}
    leading_byte_offset::Int
    stride_byte_offset::Int
    tma_swizzle::Symbol          # :NONE / :B32 / :B64 / :B128
    smem_swizzle::Union{Symbol,Nothing}
end

# layout_type also doubles as the swizzle field of the wgmma descriptor.
swizzle_code(l::GmmaLayout) = l.layout_type

# Row-width-in-bytes → layout family (single-stripe case).
#   16 B  = 1 u128 wide → INTERLEAVE
#   32 B  = 2 u128 wide → B32
#   64 B  = 4 u128 wide → B64
#   128 B = 8 u128 wide → B128
function _layout_for_row_bytes(row_bytes::Int)
    row_bytes == 16  && return WgmmaSwizzle.NONE
    row_bytes == 32  && return WgmmaSwizzle.B32
    row_bytes == 64  && return WgmmaSwizzle.B64
    row_bytes == 128 && return WgmmaSwizzle.B128
    return nothing
end

function _tma_swizzle_for(layout_type::UInt8)
    layout_type == WgmmaSwizzle.NONE && return (:NONE, nothing)
    layout_type == WgmmaSwizzle.B32  && return (:B32, :B32)
    layout_type == WgmmaSwizzle.B64  && return (:B64, :B64)
    layout_type == WgmmaSwizzle.B128 && return (:B128, :B128)
    error("wgmma_layout: unknown layout_type $layout_type")
end

"""
    pick_gmma_layout(; elem_bytes, m_or_n, k, major) -> GmmaLayout

Canonical GMMA layout for a (dtype, M-or-N, K, major) SMEM tile. `major`
is `:K` or `:MN`. See module-top comment for the full mapping.
"""
function pick_gmma_layout(; elem_bytes::Integer, m_or_n::Integer, k::Integer,
                          major::Symbol)
    elem_bytes > 0 || throw(ArgumentError("elem_bytes must be positive, got $elem_bytes"))
    (m_or_n > 0 && k > 0) ||
        throw(ArgumentError("shape must be positive, got ($m_or_n, $k)"))

    if major === :K
        # Row-major A: K is fast. Row width = stride between consecutive M-rows.
        row_bytes = Int(k) * Int(elem_bytes)
        return _build_k_major(Int(elem_bytes), Int(m_or_n), Int(k), row_bytes)
    elseif major === :MN
        # MN-major B: N is fast. Row width = stride between consecutive K-rows.
        row_bytes = Int(m_or_n) * Int(elem_bytes)
        return _build_mn_major(Int(elem_bytes), Int(m_or_n), Int(k), row_bytes)
    else
        throw(ArgumentError("major must be :K or :MN, got $(repr(major))"))
    end
end

function _build_k_major(elem_bytes::Int, m::Int, k::Int, row_bytes::Int)
    # Multi-stripe: K wider than one B128 stripe (128 bytes) → clamp to B128,
    # LBO carries the inter-stripe distance instead of the trivial 16.
    if row_bytes > 128
        layout_type = WgmmaSwizzle.B128
    else
        lt = _layout_for_row_bytes(row_bytes)
        if lt === nothing
            throw(ArgumentError(
                "wgmma_layout: K-major tile with K=$k, elem_bytes=$elem_bytes " *
                "has row width $row_bytes bytes, which is not one of the " *
                "canonical GMMA row widths (16, 32, 64, 128). Try padding K " *
                "or using a different dtype."))
        end
        layout_type = lt
    end
    m % 8 == 0 || throw(ArgumentError(
        "wgmma_layout: K-major tile requires M divisible by 8 " *
        "(core-matrix row group). Got M=$m."))

    if row_bytes > 128
        # K-major multi-stripe: stripes are 128 B wide, each stores all M rows
        # for its K-subrange contiguously. LBO = stride between stripes.
        stripe_bytes = m * 128
        leading_byte_offset = stripe_bytes
        stride_byte_offset = 8 * 128
    else
        # Single-stripe K-major: LBO is trivial (1 u128 = 16 B), SBO is the
        # stride between 8-M-row core-matrix groups.
        leading_byte_offset = 16
        stride_byte_offset = 8 * row_bytes
    end

    tma_sw, smem_sw = _tma_swizzle_for(layout_type)
    return GmmaLayout(layout_type, leading_byte_offset, stride_byte_offset,
                      tma_sw, smem_sw)
end

function _build_mn_major(elem_bytes::Int, n::Int, k::Int, row_bytes::Int)
    if row_bytes > 128
        layout_type = WgmmaSwizzle.B128
    else
        lt = _layout_for_row_bytes(row_bytes)
        if lt === nothing
            throw(ArgumentError(
                "wgmma_layout: MN-major tile with N=$n, elem_bytes=$elem_bytes " *
                "has row width $row_bytes bytes, which is not one of the " *
                "canonical GMMA row widths (16, 32, 64, 128)."))
        end
        layout_type = lt
    end
    k % 8 == 0 || throw(ArgumentError(
        "wgmma_layout: MN-major tile requires K divisible by 8 " *
        "(core-matrix K group). Got K=$k."))

    if layout_type == WgmmaSwizzle.NONE
        # Degenerate N = 1 u128 wide (e.g. N=8 bf16 / N=4 f32). The canonical
        # form collapses: outer N is size 1, SBO is trivial, LBO carries the
        # stride between 8-K-row slabs.
        leading_byte_offset = 8 * row_bytes
        stride_byte_offset = 16
    elseif row_bytes > 128
        # MN-major multi-stripe: stripes are 128 B wide, each stores all K rows
        # for its N-subrange contiguously. LBO = stride between stripes,
        # SBO = stride between 8-K-row groups within a stripe.
        stripe_bytes = k * 128
        leading_byte_offset = stripe_bytes
        stride_byte_offset = 8 * 128
    else
        # Single-stripe non-INTERLEAVE (B32/B64/B128 with row_bytes ≤ 128).
        leading_byte_offset = 16
        stride_byte_offset = 8 * row_bytes
    end

    tma_sw, smem_sw = _tma_swizzle_for(layout_type)
    return GmmaLayout(layout_type, leading_byte_offset, stride_byte_offset,
                      tma_sw, smem_sw)
end

# dtype → element bytes. Accepts integer (bytes), Julia type, or `:bf16` etc.
_elem_bytes(x::Integer) = Int(x)
_elem_bytes(::Type{T}) where {T} = sizeof(T)
function _elem_bytes(s::Symbol)
    s === :bf16 && return 2
    s === :f16  && return 2
    s === :f32  && return 4
    s === :tf32 && return 4
    s === :e4m3 && return 1
    s === :e5m2 && return 1
    s === :e3m2 && return 1
    s === :e2m3 && return 1
    s === :e2m1 && return 1  # 4-bit, but row-width math wants byte count;
                             # FP4 tiles are packed two-per-byte by the caller.
    throw(ArgumentError("wgmma_layout: cannot determine elem_bytes from $(repr(s))"))
end

"""
    layout_for_a(; dtype, m, k) -> GmmaLayout

K-major GMMA layout — natural for row-major A (`MxK` with K-fast),
and also the right pick for operand B when the kernel lays B as
row-major `KxN` in SMEM with K-fast (the common case across the
Hopper kernels in this repo). Use this for either operand whose
K-dimension is fastest-varying in SMEM.
"""
layout_for_a(; dtype, m::Integer, k::Integer) =
    pick_gmma_layout(elem_bytes=_elem_bytes(dtype), m_or_n=m, k=k, major=:K)

"""
    layout_for_mn_major(; dtype, k, n) -> GmmaLayout

MN-major GMMA layout — operand B laid as col-major `KxN` in SMEM
with N-fast (i.e. the N-or-M dimension is fastest-varying). Required
only when the SMEM tile genuinely needs MN-fastness and a transposed
wgmma `trans_b=1` flag is used. Most Hopper kernels in this repo
use `layout_for_a` for both operands and let wgmma run with `trans_b=0`
against a K-fast B tile; reach for `layout_for_mn_major` only when
the data layout forces the issue.
"""
layout_for_mn_major(; dtype, k::Integer, n::Integer) =
    pick_gmma_layout(elem_bytes=_elem_bytes(dtype), m_or_n=n, k=k, major=:MN)
