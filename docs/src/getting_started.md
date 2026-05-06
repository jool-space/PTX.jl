```@meta
CurrentModule = PTX
```

# Getting started

## Install

```julia
pkg> add PTX, CUDA
```

PTX.jl depends on `LLVM.jl` for the inline-asm machinery (`@asmcall`).
Kernel launch and memory go through `CUDA.jl` (or `CUDACore`).

## First kernel

Every PTX instruction is one [`@ptx_str`](@ref) call. Special registers
(`%tid.x`, `%ctaid.y`, …) are read via [`@sreg_str`](@ref). Inputs flow in
as Julia arguments, the result is the call's return value:

```julia
using PTX, CUDA

function add_kernel!(c, a, b)
    tid = ptx"mov.u32"(sreg"%tid.x")
    i = Int(tid) + 1
    c[i] = ptx"add.f32"(a[i], b[i])
    return
end

n = 128
a = cu(randn(Float32, n))
b = cu(randn(Float32, n))
c = similar(a)
@cuda threads = n add_kernel!(c, a, b)
@assert Array(c) ≈ Array(a) + Array(b)
```

`ptx"add.f32"` constructs an `Operation` singleton; the call site is
`@generated`, picks constraint letters from the argument types, infers
the return type from the trailing `.f32` modifier, and lowers to a single
`@asmcall("add.f32 $0, $1, $2;", "=f,f,f", false, Float32, Tuple{Float32,
Float32}, a, b)`.

## Composing with CUDA.jl

PTX.jl is **strictly additive**. CUDA.jl owns everything around the
instruction; PTX.jl owns the instruction itself.

| Layer | CUDA.jl owns | PTX.jl owns |
|---|---|---|
| Launch & dispatch | `@cuda`, `cudacall`, `cufunction` | — |
| Memory | `CuArray`, `CuDeviceVector`, `CuStaticSharedArray`, `pointer(...) :: Core.LLVMPtr` | — |
| Control flow | regular Julia loops, `if`, recursion | — (transpiler emits `@goto`/`@label` for legacy CFG) |
| Math intrinsics, basic warp prims | `sin`, `cos`, `sqrt`, atomics, `threadIdx()`, `shfl_sync`, `sync_threads()` | same ops at lower level — `ptx"ex2.approx.f32"(x)`, `ptx"shfl.sync.bfly.b32"(...)`, `ptx"bar.sync"(Val(0))` |
| Tensor-core / specialty ops | very limited (CUDA.WMMA only, partial coverage) | `mma.sync`, `wgmma`, `ldmatrix`, `cp.async`, vector ld/st, FP8 cvt, mbarrier, TMA, … |

Rule of thumb: write what you'd write in normal CUDA C++ in Julia + CUDA.jl;
drop into PTX.jl for what you'd write as a `<header>` library call that
nvcc lowers to `asm volatile`-equivalent inline PTX. Real kernels mix both.

## Address spaces

`ptx"ld.global.f32"` / `ptx"cp.async.ca.shared.global"` and friends take
typed pointers. PTX.jl re-exports a small `AS` module of LLVM
address-space numbers that match `CUDACore.AS.*`:

```julia
PTX.AS.Generic   # 0
PTX.AS.Global    # 1
PTX.AS.Shared    # 3
PTX.AS.Const     # 4
PTX.AS.Local     # 5
PTX.AS.Param     # 101
```

A pointer reaches a wrapper as `Core.LLVMPtr{T, AS.X}`. NVPTX lowers
non-`Generic` address-space pointers as i64 at the LLVM IR level even
when the underlying PTX address is 32-bit (shared, param, local). The
chain emits `l` (i64) for any `LLVMPtr`; hand-written shared-memory
wrappers (`cp.async`, `ldmatrix`, `stmatrix`, `mbarrier`) override to
`r` (i32) where ptxas requires the 32-bit form.

## Where to next

- [Chain DSL](dsl.md) — the full set of conventions the chain encodes.
- [Wrappers](wrappers.md) — when to reach for a hand-written wrapper.
- [Transpiler](transpiler.md) — turn existing `.ptx` files into Julia.
