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
...
```

PTX.jl fills the gap CUDA.jl leaves uncovered: full TensorCore shape coverage
(incl. TF32, FP8, sub-byte), TMA descriptors, cluster APIs, mbarriers, FP8
conversions, `setmaxnreg`, `match.sync`, `prmt`, and the rest of what
`<mma.h>`, `<cuda_pipeline.h>`, and `<cuda/barrier>` ultimately compile down
to. Composition with CUDA.jl is strictly additive — CUDA.jl owns launch,
memory, and control flow; PTX.jl owns specialty op emission.

`ptx_to_julia(source)` turns a `.ptx` file into idiomatic Julia where each
register is a variable and each instruction is a `ptx"..."(...)` call. See
the [Transpiler](https://jool-space.github.io/PTX.jl/dev/transpiler/) docs.

## Credits

Primary design inspiration: [pyptx](https://github.com/patrick-toulme/pyptx)
 The parser, IR, and several wrappers and example kernels
are ported from pyptx (Apache 2.0); see per-file headers and `LICENSE`.
