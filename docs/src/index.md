```@meta
CurrentModule = PTX
```

# PTX.jl

A Julia interface for [NVIDIA PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/).
Composes additively with [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) — CUDA.jl
owns launch, memory, and control flow; PTX.jl owns instruction emission.

```julia
using PTX, CUDACore

function add_kernel!(c, a, b)
    tid = ptx"mov.u32"(sreg"%tid.x")
    i = Int(tid) + 1
    c[i] = ptx"add.f32"(a[i], b[i])
    return
end

let
    n = 128
    a = cu(randn(n))
    b = cu(randn(n))
    c = similar(a)
    @cuda threads=n add_kernel!(c, a, b)
    c == a + b
end
```

## Why

PTX is the lowest authoring tier in the NVIDIA GPU stack. PTX.jl fills the
gap CUDA.jl leaves uncovered: full TensorCore shape coverage (incl. TF32,
FP8, sub-byte), TMA descriptors, cluster APIs, mbarriers, FP8 conversions,
`setmaxnreg`, `match.sync`, `prmt`, and the rest of what `<mma.h>`,
`<cuda_pipeline.h>`, and `<cuda/barrier>` ultimately compile down to.

When porting modern CUTLASS / Triton / cuDNN-style kernels, **PTX.jl is on
the critical path**. CUDA.jl alone leaves a Julia user stuck for any
kernel that uses tensor cores beyond CUDA.WMMA's limited coverage, async
pipelines, cluster ops, or TMA. PTX.jl + CUDA.jl together close that gap.

## Two surfaces

- **Authoring.** Write PTX directly in Julia via the [`@ptx_str`](@ref)
  string macro. See the [Chain DSL](dsl.md) page.
- **Transpiling.** Turn an existing `.ptx` file into idiomatic Julia via
  [`ptx_to_julia`](@ref). See the [Transpiler](transpiler.md) page.

## Pages

- [Getting started](getting_started.md) — install, first kernel, how
  PTX.jl composes with CUDA.jl.
- [Chain DSL](dsl.md) — `@ptx_str` / `@sreg_str` semantics: return-type
  inference, side-effect classification, operand conventions.
- [Wrappers](wrappers.md) — hand-written wrapper families for ops whose
  operand shape breaks the chain default (mma, wgmma, ldmatrix, TMA,
  mbarrier, …).
- [Transpiler](transpiler.md) — parse PTX, transform IR, emit Julia.
- [Reference](reference.md) — public API.

## Credits

Primary design inspiration: [pyptx](https://github.com/patrick-toulme/pyptx)
by Patrick Toulmé. The parser, IR, and several wrappers and example kernels
are ported from pyptx (Apache 2.0); see per-file headers and `LICENSE`.
