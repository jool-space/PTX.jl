# Hand-written cvt for the FP4 (e2m1x2) carrier shim. PTX e2m1x2 storage type
# is `.b8`, but NVPTX has no i8 constraint letter — bridge through a UInt16
# carrier with a `.reg .b8` plus `mov.b16` brace pair on either side (mirrors
# NVIDIA's `__nv_cvt_*` shims in <cuda_fp4.hpp>). All other cvt forms flow
# through the chain default in src/inst.jl.

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
