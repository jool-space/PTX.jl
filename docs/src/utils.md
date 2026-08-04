```@meta
CurrentModule = PTX
```

# Kernel metaprogramming utilities

`PTX.Utils` collects the metaprogramming idioms register-resident
kernels repeat: loops whose induction value must be a *compile-time*
constant (`Val` arguments, per-iteration variable names, constant tuple
indexing), and wide tuple folds that must not become closures (an
`ntuple ... do` body past a handful of registers escapes the inliner
and turns into a real device call). Not exported; access as
`using PTX.Utils: @unrolled, strided_reduce`.

Note the division of labor with LLVM's own unroller: a loopinfo hint
(KernelAbstractions' `@unroll`, `llvm.loop.unroll.*` metadata) asks the
optimizer to unroll and cannot make the induction variable a constant —
`@unrolled` expands at macro time and can, which is what barrier-slot
`Val`s and distinct loop-carried phase registers require.

```@docs
PTX.Utils
PTX.Utils.@unrolled
PTX.Utils.strided_reduce
```
