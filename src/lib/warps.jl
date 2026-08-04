"""
Warp-level collective idioms on top of the `shfl.sync` wrappers.

Sister module to `PTX.MBarriers`/`PTX.Pipelines`: no new hardware
surface — every call lowers to the reviewed `shfl.sync.bfly.b32`
wrapper — but it names the butterfly-reduce ladder that every
row-reduction kernel (softmax, the norm family, attention softmax
streams) hand-rolls. Not exported; access as
`using PTX.Warps: warp_reduce`.
"""
module Warps

using ..PTX: @ptx_str
using Republic: @public

@public warp_reduce

"""
    warp_reduce(op, v::T[, ::Val{W} = Val(32)]) -> T

Butterfly (XOR-shuffle) all-reduce: after `log₂(W)` rounds every lane
of each `W`-lane segment holds `op` folded over the segment's values.
`W` must be a power of two in 2:32 (32 = the full warp); `T` must be a
4-byte isbits type (`Float32`, `Int32`, `UInt32` — the shuffle moves
raw b32 lanes and `op` sees `T`).

`op` is any two-argument function, applied between shuffle rounds
exactly as written — pass the combining op the kernel *means*:
`(a, b) -> ptx"max.f32"(a, b)` and Julia's `max` (which lowers to the
NaN-propagating `max.NaN.f32`) are different instructions.

Convergent: all 32 lanes of the warp must reach the call with the full
membermask's lanes active, including lanes whose segment result goes
unused.

The ladder is pinned: XOR offsets descend `W/2, …, 1`, and the shuffle
c-operand is `((32 - W) << 8) | 0x1F` (CUDA's `__shfl_*_sync` width
encoding; segments never mix because XOR offsets below `W` cannot
cross a `W`-aligned boundary). At `W = 32` this is instruction-
identical to the hand-rolled descending ladders it replaces.
"""
@inline warp_reduce(op::F, v) where {F} = warp_reduce(op, v, Val(32))
@generated function warp_reduce(op::F, v::T, ::Val{W}) where {F, T, W}
    W isa Int && ispow2(W) && 2 <= W <= 32 ||
        error("warp_reduce: W must be a power of two in 2:32, got $W")
    isbitstype(T) && sizeof(T) == 4 ||
        error("warp_reduce: v must be a 4-byte isbits type, got $T")
    c = UInt32(((32 - W) << 8) | 0x1F)
    stmts = Expr[]
    offset = W >> 1
    while offset >= 1
        push!(stmts, quote
            u = reinterpret(UInt32, v)
            u_par = ptx"shfl.sync.bfly.b32"(u, $(UInt32(offset)), $c,
                                            0xFFFFFFFF % UInt32)
            v = op(v, reinterpret(T, u_par))
        end)
        offset >>= 1
    end
    quote
        $(Expr(:meta, :inline))
        $(stmts...)
        v
    end
end

end # module Warps
