# PTX.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://docs.jool.space/PTX.jl/dev/)
[![Build Status](https://github.com/jool-space/PTX.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jool-space/PTX.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jool-space/PTX.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jool-space/PTX.jl)

A Julia interface for [NVIDIA PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/),
composing additively with [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).

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

## Credits

Primary design inspiration: [pyptx](https://github.com/patrick-toulme/pyptx)
by Patrick Toulmé. The parser, IR, and several wrappers and example kernels
are ported from pyptx (Apache 2.0); see per-file headers and `LICENSE`.
