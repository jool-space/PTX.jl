# PTX.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://docs.jool.space/PTX.jl/dev/)
[![Build Status](https://github.com/jool-space/PTX.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jool-space/PTX.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jool-space/PTX.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jool-space/PTX.jl)

The [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) exposed
directly in Julia through `ptx"..."(...)` syntax, composing naturally with [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).

```julia
using PTX, CUDA

function add_kernel!(c, a, b)
    tid = ptx"mov.u32"(sreg"%tid.x") # access instructions and special registers,
    i = tid + one(tid)               # while also writing familiar julia code.
    c[i] = ptx"add.f32"(a[i], b[i])  # same as a[i] + b[i], assuming Float32
    return
end

n = 128;
a = cu(randn(n));
b = cu(randn(n));
c = similar(a);
@cuda threads=n add_kernel!(c, a, b);
@assert c == a + b

# or simply use an instruction in a broadcast:
@assert c == ptx"add.f32".(a, b)
```

## Motivation

Julia code reaches the GPU through LLVM. CUDA.jl (via GPUCompiler.jl) compiles
ordinary generic Julia functions to LLVM IR, LLVM's NVPTX backend selects that
into PTX, and NVIDIA's `ptxas` compiles PTX to machine code. This pipeline is
why generic Julia code works on GPUs at all: the optimizer sees your whole
kernel, so abstractions, closures, and multiple dispatch cost nothing by the
time instructions are emitted.

The missing piece is the instruction surface: LLVM emits only what its
backend knows how to select, and CUDA.jl wraps only what it chooses to
support; it does not — and should not — expose every hardware instruction.
When you need one it doesn't (tensor core MMA shapes, `ldmatrix`, TMA,
mbarriers, cluster ops), the traditional escape hatches are:

- `Base.llvmcall` with handwritten LLVM IR, calling an `llvm.nvvm.*`
  intrinsic when one exists;
- inline assembly (`@asmcall` from LLVM.jl, or `llvmcall` with an `asm`
  snippet) when one doesn't;

These both work, but require hand-written IR, constraints, and declaration
of effects.

PTX.jl replaces them with one surface: you write the PTX instruction,
and the package picks the lowering. Instructions LLVM recognizes lower to the
corresponding `llvm.nvvm.*` intrinsic with explicit effect and convergence
attributes, so the optimizer can still reason around them. Everything else is
emitted as inline assembly that is opaque to LLVM. That cost is inherent to
inline assembly, not to PTX.jl.

The division of labor is: CUDA.jl owns arrays, kernel launch, and
the runtime; LLVM owns optimization of the surrounding Julia code; PTX.jl
owns the instruction boundary; `ptxas` owns final scheduling and register
allocation. PTX.jl adds a layer without replacing any of them.

## Credits

Primary design inspiration: [pyptx](https://github.com/patrick-toulme/pyptx)
by Patrick Toulmé. The parser, IR, and several wrappers and example kernels
are ported from pyptx (Apache 2.0); see per-file headers and `LICENSE`.
