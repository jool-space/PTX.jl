# `spcompress` consumes a 32-bit sparsity descriptor (PTX 9.4 §9.7.10.30,
# Table 38) that names the selection rule for the two kept elements of each
# group of four and the exact element type of the dense input. The
# instruction's `.elemsize` qualifier must agree with the type's width; a
# mismatch is a runtime error on the device, so `spcompress_elemsize` lets
# callers derive the qualifier from the same type symbol.
#
#   bits 0–1  selection: 0 = MAX, 1 = MAXABS, 2 = MIN, 3 = MINABS
#   bits 2–4  element type: 0 = f16/u8, 1 = bf16/s8, 2 = e5m2, 3 = e4m3,
#             4 = e3m2, 5 = e2m3 (codes 0 and 1 are disambiguated by
#             `.elemsize`)
#   bits 5–31 reserved, zero

const _SPCOMPRESS_OPS = (max = 0x0, maxabs = 0x1, min = 0x2, minabs = 0x3)
const _SPCOMPRESS_DTYPES = (
    f16 = (0x0, 16), bf16 = (0x1, 16), u8 = (0x0, 8), s8 = (0x1, 8),
    e5m2 = (0x2, 8), e4m3 = (0x3, 8), e3m2 = (0x4, 8), e2m3 = (0x5, 8),
)

"""
    spcompress_desc(; op, dtype) -> UInt32

Pack the `spcompress` sparsity descriptor (PTX 9.4 §9.7.10.30, Table 38).
`op` is `:max`, `:maxabs`, `:min`, or `:minabs`; `dtype` is `:f16`, `:bf16`,
`:u8`, `:s8`, `:e5m2`, `:e4m3`, `:e3m2`, or `:e2m3`. The instruction's
`.elemsize` qualifier must match `spcompress_elemsize(dtype)`.
"""
function spcompress_desc(; op::Symbol, dtype::Symbol)
    haskey(_SPCOMPRESS_OPS, op) ||
        throw(ArgumentError("spcompress_desc: op must be one of " *
                            "$(keys(_SPCOMPRESS_OPS)), got $(repr(op))"))
    haskey(_SPCOMPRESS_DTYPES, dtype) ||
        throw(ArgumentError("spcompress_desc: dtype must be one of " *
                            "$(keys(_SPCOMPRESS_DTYPES)), got $(repr(dtype))"))
    UInt32(_SPCOMPRESS_OPS[op]) | UInt32(_SPCOMPRESS_DTYPES[dtype][1]) << 2
end

"""
    spcompress_elemsize(dtype) -> Symbol

The `.elemsize` qualifier (`:b8` or `:b16`) that `spcompress` requires for
the element type named in its descriptor.
"""
function spcompress_elemsize(dtype::Symbol)
    haskey(_SPCOMPRESS_DTYPES, dtype) ||
        throw(ArgumentError("spcompress_elemsize: dtype must be one of " *
                            "$(keys(_SPCOMPRESS_DTYPES)), got $(repr(dtype))"))
    _SPCOMPRESS_DTYPES[dtype][2] == 8 ? :b8 : :b16
end
