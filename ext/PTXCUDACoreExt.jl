module PTXCUDACoreExt

# Provides the `cuTensorMapEncodeTiled` implementation for PTX.jl. Loaded
# automatically when CUDACore is in the user's environment alongside PTX.
# See src/tensor_map.jl for the public surface — this module only implements
# the methods that actually invoke the CUDA driver.

using PTX
using PTX: CuTensorMap, tensor_map_dtype_code, tensor_map_swizzle_code,
           tensor_map_interleave_code, tensor_map_l2_promotion_code,
           tensor_map_oob_fill_code, _tensor_map_elem_bytes
using CUDACore: CUtensorMap, CUtensorMapDataType, CUtensorMapInterleave,
                CUtensorMapSwizzle, CUtensorMapL2promotion,
                CUtensorMapFloatOOBfill, cuTensorMapEncodeTiled,
                cuuint32_t, cuuint64_t, CuPtr

# Inner-mode driver call. `tmap` is mutable so its data field is at a stable
# pointer for the lifetime of the GC.@preserve block.
function _encode_tiled_inner(
        tmap::CuTensorMap,
        dtype_code::UInt32, rank::Int, global_addr::Ptr{Cvoid},
        global_dim::Vector{cuuint64_t}, global_strides::Vector{cuuint64_t},
        box_dim::Vector{cuuint32_t}, elem_strides::Vector{cuuint32_t},
        interleave_code::UInt32, swizzle_code::UInt32,
        l2_promotion_code::UInt32, oob_fill_code::UInt32)
    GC.@preserve tmap global_dim global_strides box_dim elem_strides begin
        # `tmap.data::NTuple{128,UInt8}` lives at offset 0 of the mutable
        # struct; the driver writes the 128-byte blob there.
        tmap_ptr = Ptr{CUtensorMap}(pointer_from_objref(tmap))
        cuTensorMapEncodeTiled(
            tmap_ptr,
            CUtensorMapDataType(dtype_code),
            cuuint32_t(rank),
            global_addr,
            pointer(global_dim),
            pointer(global_strides),
            pointer(box_dim),
            pointer(elem_strides),
            CUtensorMapInterleave(interleave_code),
            CUtensorMapSwizzle(swizzle_code),
            CUtensorMapL2promotion(l2_promotion_code),
            CUtensorMapFloatOOBfill(oob_fill_code),
        )
    end
    return tmap
end

# Coerce a caller-supplied pointer-ish value to Ptr{Cvoid}. The driver
# stores it raw — CuPtr is the common case (device address from a CuArray).
_as_void_ptr(p::Ptr) = convert(Ptr{Cvoid}, p)
_as_void_ptr(p::CuPtr) = Ptr{Cvoid}(UInt(p))
_as_void_ptr(p::Integer) = Ptr{Cvoid}(UInt(p))

function PTX.tensor_map_encode_tiled(
        dtype, global_addr,
        global_dim::NTuple{N, <:Integer},
        global_strides::Tuple,
        box_dim::NTuple{N, <:Integer};
        elem_strides::NTuple{N, <:Integer} = ntuple(_ -> 1, Val(N)),
        interleave::Symbol = :NONE,
        swizzle::Symbol = :NONE,
        l2_promotion::Symbol = :NONE,
        oob_fill::Symbol = :NONE) where {N}
    1 <= N <= 5 || throw(ArgumentError(
        "tensor_map: rank must be 1..5 (TMA driver limit), got N=$N"))
    length(global_strides) == N - 1 || throw(ArgumentError(
        "tensor_map: global_strides must have length N-1 = $(N-1), " *
        "got $(length(global_strides)) (innermost stride is implicit elem_bytes×1)"))

    gd = collect(cuuint64_t, global_dim)
    gs = collect(cuuint64_t, global_strides)
    bd = collect(cuuint32_t, box_dim)
    es = collect(cuuint32_t, elem_strides)

    tmap = CuTensorMap()
    _encode_tiled_inner(
        tmap,
        tensor_map_dtype_code(dtype),
        N, _as_void_ptr(global_addr),
        gd, gs, bd, es,
        tensor_map_interleave_code(interleave),
        tensor_map_swizzle_code(swizzle),
        tensor_map_l2_promotion_code(l2_promotion),
        tensor_map_oob_fill_code(oob_fill),
    )
    return tmap
end

function PTX.tensor_map_tile_2d(
        dtype, global_addr,
        rows::Integer, cols::Integer,
        box_rows::Integer, box_cols::Integer;
        swizzle::Symbol = :B128,
        oob_fill::Symbol = :NONE)
    elem_bytes = _tensor_map_elem_bytes(PTX._tensormap_dtype_symbol(dtype))
    # Innermost-first: dim = (cols, rows). Outer stride = cols * elem_bytes
    # (the byte step between adjacent rows for row-major).
    return PTX.tensor_map_encode_tiled(
        dtype, global_addr,
        (Int(cols), Int(rows)),
        (Int(cols) * elem_bytes,),
        (Int(box_cols), Int(box_rows));
        swizzle = swizzle,
        oob_fill = oob_fill,
    )
end

end # module
