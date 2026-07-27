# Hand-written cvt for the FP4 (e2m1x2) carrier shim. PTX e2m1x2 storage type
# is `.b8`, but NVPTX has no i8 constraint letter — bridge through a UInt16
# carrier with a `.reg .b8` plus `mov.b16` brace pair on either side (mirrors
# NVIDIA's `__nv_cvt_*` shims in <cuda_fp4.hpp>). All other cvt forms flow
# through the chain default in src/dsl/entries.jl.

@inline (::Operation{:cvt, (:rn, :satfinite, :e2m1x2, :f32)})(a::Float32, b::Float32) =
    @asmcall("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, \$1, \$2; mov.b16 \$0, {t, 0}; }",
             "=h,f,f", false, UInt16, Tuple{Float32, Float32}, a, b)

@inline (::Operation{:cvt, (:rn, :satfinite, :e2m1x2, :f16x2)})(a::UInt32) =
    @asmcall("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f16x2 t, \$1; mov.b16 \$0, {t, 0}; }",
             "=h,r", false, UInt16, Tuple{UInt32}, a)

@inline (::Operation{:cvt, (:rn, :satfinite, :e2m1x2, :bf16x2)})(a::UInt32) =
    @asmcall("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.bf16x2 t, \$1; mov.b16 \$0, {t, 0}; }",
             "=h,r", false, UInt16, Tuple{UInt32}, a)

@inline (::Operation{:cvt, (:rn, :f16x2, :e2m1x2)})(a::UInt16) =
    @asmcall("{ .reg .b8 t, hi; mov.b16 {t, hi}, \$1; cvt.rn.f16x2.e2m1x2 \$0, t; }",
             "=r,h", false, UInt32, Tuple{UInt16}, a)

@inline (::Operation{:cvt, (:rn, :bf16x2, :e2m1x2)})(a::UInt16) =
    @asmcall("{ .reg .b8 t, hi; mov.b16 {t, hi}, \$1; cvt.rn.bf16x2.e2m1x2 \$0, t; }",
             "=r,h", false, UInt32, Tuple{UInt16}, a)

# Ergonomic pack helper for `cvt.rn.bf16x2.f32`. PTX 9.2 §9.7.9.21:
# `cvt.rn.bf16x2.f32 d, a, b` puts input `a` in the UPPER half of `d`
# and input `b` in the LOWER half. That convention is opposite of
# `mov.b32 d, {x, y}` (which puts `x` in the LOW half), so kernels that
# want "low first, high second" tuple-style ordering need to swap.
# This helper hides the swap: caller passes `(lo, hi)` in natural order,
# we feed PTX `(a=hi, b=lo)`. Convention confirmed empirically on H100
# and GB10 across all four packed-cvt-from-f32 ops (bf16x2, f16x2,
# e4m3x2, e5m2x2) — uniform with the spec.
@inline bf16x2_pack(lo::Float32, hi::Float32)::UInt32 =
    ptx"cvt.rn.bf16x2.f32"(hi, lo)
