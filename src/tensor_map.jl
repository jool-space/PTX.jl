# Host-side `cuTensorMapEncodeTiled` wrapper.
#
# The CUDA driver packs (tensor shape, strides, dtype, swizzle, OOB
# behaviour) into a 128-byte opaque blob (`CUtensorMap`). The kernel reads
# this blob via `cp.async.bulk.tensor.*` to issue TMA loads/stores. The
# blob is built once on the host with `cuTensorMapEncodeTiled` and either
# uploaded to device memory or passed as a kernel parameter.
#
# This file defines:
#   - `CuTensorMap` — Julia type holding the 128-byte blob
#   - Symbol → driver-enum codes for swizzle / OOB-fill / L2-promotion / etc.
#   - Type/Symbol → tensor-map dtype code
#   - `tensor_map_encode_tiled(...)` stub
#
# The actual `cuTensorMapEncodeTiled` ccall lives in `ext/PTXCUDACoreExt.jl`
# so PTX.jl doesn't pick up CUDACore as a hard dependency. Reference for
# the driver call: pyptx/jax_support.py:synthesize_tma_descriptor + the
# CUDACore binding at CUDACore/lib/cudadrv/libcuda.jl:6160.

# 128-byte opaque blob. Mutable so we get a stable pointer for the driver
# call (the driver writes into the blob). Compatible with
# CUDACore.CUtensorMap_st via a Ref-typed ccall in the extension.
mutable struct CuTensorMap
    data::NTuple{128, UInt8}
    CuTensorMap() = new(ntuple(_ -> 0x00, 128))
end

Base.sizeof(::CuTensorMap) = 128
Base.sizeof(::Type{CuTensorMap}) = 128

# --- Symbol → driver enum values --------------------------------------------
# Mirrors CUtensorMap{DataType,Interleave,Swizzle,L2promotion,FloatOOBfill}
# in CUDACore/lib/cudadrv/libcuda.jl. Keep values in sync with the driver.

# CUtensorMapInterleave (CU_TENSOR_MAP_INTERLEAVE_*).
const TENSOR_MAP_INTERLEAVE_CODES = Dict{Symbol, UInt32}(
    :NONE => 0, :B16 => 1, :B32 => 2,
)

# CUtensorMapSwizzle. The four canonical wgmma-compatible families plus the
# Blackwell sm_100a 128B-atom variants (driver enums 4–6). Aliases match
# WgmmaSwizzle naming for cross-reference with wgmma_layout.jl.
const TENSOR_MAP_SWIZZLE_CODES = Dict{Symbol, UInt32}(
    :NONE             => 0,
    :B32              => 1,
    :B64              => 2,
    :B128             => 3,
    :B128_ATOM_32B    => 4,
    :B128_ATOM_32B_FLIP_8B => 5,
    :B128_ATOM_64B    => 6,
)

# CUtensorMapL2promotion.
const TENSOR_MAP_L2_PROMOTION_CODES = Dict{Symbol, UInt32}(
    :NONE => 0, :B64 => 1, :B128 => 2, :B256 => 3,
)

# CUtensorMapFloatOOBfill.
const TENSOR_MAP_OOB_FILL_CODES = Dict{Symbol, UInt32}(
    :NONE => 0,
    :NAN_REQUEST_ZERO_FMA => 1,
)

# CUtensorMapDataType.
const TENSOR_MAP_DATA_TYPE_CODES = Dict{Symbol, UInt32}(
    :u8 => 0, :u16 => 1, :u32 => 2, :s32 => 3,
    :u64 => 4, :s64 => 5,
    :f16 => 6, :f32 => 7, :f64 => 8, :bf16 => 9,
    :f32_ftz => 10, :tf32 => 11, :tf32_ftz => 12,
    :u4_align8b  => 13,  # 16 packed u4 in 8 bytes
    :u4_align16b => 14,  # 16 packed u4 in 16 bytes
    :u6_align16b => 15,  # 16 packed u6 in 16 bytes
)

# Julia type → dtype symbol. Sub-byte FP types stay symbol-only (they have
# no Julia counterpart in Base).
_tensormap_dtype_symbol(::Type{UInt8})   = :u8
_tensormap_dtype_symbol(::Type{UInt16})  = :u16
_tensormap_dtype_symbol(::Type{UInt32})  = :u32
_tensormap_dtype_symbol(::Type{UInt64})  = :u64
_tensormap_dtype_symbol(::Type{Int32})   = :s32
_tensormap_dtype_symbol(::Type{Int64})   = :s64
_tensormap_dtype_symbol(::Type{Float16}) = :f16
_tensormap_dtype_symbol(::Type{Float32}) = :f32
_tensormap_dtype_symbol(::Type{Float64}) = :f64
_tensormap_dtype_symbol(s::Symbol)       = s

function _lookup_code(table::Dict{Symbol, UInt32}, key::Symbol, what::AbstractString)
    code = get(table, key, nothing)
    code === nothing && throw(ArgumentError(
        "tensor_map: unknown $what $(repr(key)). Valid: $(sort(collect(keys(table))))"))
    return code
end

tensor_map_dtype_code(dt) =
    _lookup_code(TENSOR_MAP_DATA_TYPE_CODES, _tensormap_dtype_symbol(dt), "dtype")
tensor_map_swizzle_code(s::Symbol) =
    _lookup_code(TENSOR_MAP_SWIZZLE_CODES, s, "swizzle")
tensor_map_interleave_code(s::Symbol) =
    _lookup_code(TENSOR_MAP_INTERLEAVE_CODES, s, "interleave")
tensor_map_l2_promotion_code(s::Symbol) =
    _lookup_code(TENSOR_MAP_L2_PROMOTION_CODES, s, "L2 promotion")
tensor_map_oob_fill_code(s::Symbol) =
    _lookup_code(TENSOR_MAP_OOB_FILL_CODES, s, "OOB fill")

"""
    tensor_map_encode_tiled(dtype, global_addr, global_dim, global_strides,
                            box_dim; kwargs...) -> CuTensorMap

Build a `CUtensorMap` for a tiled TMA descriptor. Calls `cuTensorMapEncodeTiled`.

Arguments (innermost-first convention, matching the driver):
- `dtype` — Julia type (`Float32`, `UInt16`, …) or symbol (`:bf16`, `:tf32`, …)
- `global_addr` — `Ptr{T}`, `CuPtr`, or `UInt`. Caller-owned; the driver
  stores it in the descriptor (use `cuTensorMapReplaceAddress` to swap later).
- `global_dim::NTuple{N, <:Integer}` — tensor shape, innermost first.
- `global_strides::NTuple{N-1, <:Integer}` — outer-dim strides in **bytes**.
  Length must be `N-1` (the innermost stride is `elem_bytes * 1`, implicit).
- `box_dim::NTuple{N, <:Integer}` — per-launch tile shape, innermost first.

Keyword arguments:
- `elem_strides::NTuple{N, <:Integer} = (1, 1, …)` — sub-tile stride.
- `interleave::Symbol = :NONE` — `:NONE` / `:B16` / `:B32`.
- `swizzle::Symbol = :NONE` — `:NONE` / `:B32` / `:B64` / `:B128`
  (plus Blackwell atom variants).
- `l2_promotion::Symbol = :NONE` — `:NONE` / `:B64` / `:B128` / `:B256`.
- `oob_fill::Symbol = :NONE` — `:NONE` / `:NAN_REQUEST_ZERO_FMA`.

Requires `using CUDACore`. Without it the package extension is inert and
this function raises a clear error.
"""
function tensor_map_encode_tiled end

"""
    tensor_map_tile_2d(dtype, global_addr, rows, cols, box_rows, box_cols;
                       swizzle=:B128, oob_fill=:NONE) -> CuTensorMap

Convenience for a 2D row-major `(rows, cols)` tensor + `(box_rows, box_cols)`
tile. Innermost dim is `cols`. Stride is `cols * elem_bytes`.

Mirrors pyptx `synthesize_tma_descriptor` (2D path). Caller is responsible
for keeping `box_cols * elem_bytes` consistent with the swizzle (e.g. 128B
for `:B128`).
"""
function tensor_map_tile_2d end

# --- internal helpers (no CUDACore needed) ----------------------------------

# Pick a default 2D box that respects the swizzle byte-width.
function _default_box_2d(elem_bytes::Int, rows::Int, cols::Int, swizzle::Symbol)
    swizzle_bytes = swizzle === :B128 ? 128 :
                    swizzle === :B64  ? 64  :
                    swizzle === :B32  ? 32  :
                    128
    box_cols = max(swizzle_bytes ÷ elem_bytes, 1)
    box_rows = 128
    return (min(box_rows, rows), min(box_cols, cols))
end

# Per-dtype element byte size. Pure host helper used by the convenience
# wrappers to set up strides; mirrors PtxType.bits // 8 logic in pyptx.
_tensor_map_elem_bytes(::Type{T}) where {T} = sizeof(T)
function _tensor_map_elem_bytes(s::Symbol)
    s === :u8 && return 1
    s in (:u16, :f16, :bf16) && return 2
    s in (:u32, :s32, :f32, :tf32, :f32_ftz, :tf32_ftz) && return 4
    s in (:u64, :s64, :f64) && return 8
    # Sub-byte FP: caller packs externally — return 1 so the bytes math is
    # correct when the caller has already converted element counts to bytes.
    s in (:u4_align8b, :u4_align16b, :u6_align16b) && return 1
    throw(ArgumentError("tensor_map: unknown dtype $(repr(s))"))
end
