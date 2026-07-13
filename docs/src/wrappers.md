```@meta
CurrentModule = PTX
```

# Wrappers

For ops whose operand semantics break the chain default, PTX.jl ships
hand-written wrapper methods on the matching `Operation{...}` singleton.
The user-facing call site is identical to the chain default — the
wrapper just provides a typed method that takes priority over the
`@generated` chain dispatch when its argument types match.

There are four patterns that force a wrapper:

1. **Mixed address-space pointer constraints.** Shared-AS pointers need
   the `r` (32-bit) constraint; the chain emits `l` (64-bit) for any
   `LLVMPtr`. `cp.async`, `ldmatrix`, `stmatrix`, `mbarrier` use this
   override to satisfy ptxas.
2. **Multi-output return.** The chain returns one value; ldmatrix x2/x4,
   mma fragments, and shfl-with-pred-output all return tuples.
3. **Special operand layout.** Braced register-vector operands (`{$N,
   $N+1, ...}`), tensor-coord forms (`[ptr, {c0, c1, c2}]`), tied
   accumulators for wgmma — none have a chain-default rendering.
4. **Compile-time-constant operand types.** TMA's `<N>d` rank, mma
   shape/dtype, fragment counts — these need to be pinned at the
   registration boundary so each variant gets its own typed method.

Each wrapper file follows the same shape: a small declarative table
maps `(shape, dtype, …)` to register counts and constraints, a
`_register` function builds the asm string + LLVM call, and a `for`
loop emits a typed method per valid combination.

Adding a new dtype/shape combination is one entry in the table plus one
line in the loop.

## Family overview

| Family | File | Surface |
|---|---|---|
| `cp.async` (scalar) | `wrappers/cp_async.jl` | `cp.async.{ca,cg}.shared.global [smem], [global], Val(N)` — needs shared `r` constraint and `N`-baked size |
| `cp.async.bulk.tensor` (TMA) | `wrappers/tma.jl` | `cp.async.bulk.tensor.{1..5}d.{shared::cluster\|shared::cta}.global.tile.mbarrier::complete_tx::bytes` (load) and `.global.shared::cta.tile.bulk_group` (store) — coord vector becomes a positional `Int32` argument list |
| `cvt` sub-byte FP packing | `wrappers/cvt.jl` | `cvt.rn.satfinite.e2m1x2.{f32,f16x2,bf16x2}` pack and `cvt.rn.{f16x2,bf16x2}.e2m1x2` unpack — `.b8` carrier through a `mov.b16` brace-pair shim because NVPTX has no i8 constraint |
| extended precision | `wrappers/extended_precision.jl` | All 48 `add.cc` / `addc` / `sub.cc` / `subc` / `mad.cc` / `madc` scalar forms with explicit `Bool` carry/borrow, plus fused arbitrary-limb add/sub and unsigned 2×2-limb multiply |
| `ldmatrix` | `wrappers/ldmatrix.jl` | `ldmatrix.sync.aligned.{m8n8,m16n16}.{x1,x2,x4}[.trans].{shared,shared::cta}.{b16,b8}` returning `UInt32` (x1) or `NTuple{N, UInt32}` (x2/x4) |
| `mbarrier` | `wrappers/mbarrier.jl` | init / inval / arrive[.noComplete\|.expect_tx] / expect_tx / test_wait[.parity] / try_wait[.parity] — three return shapes (`Nothing` / `UInt64` state / `Bool` pred); shared-AS `r` constraint |
| `mma.sync.aligned` | `wrappers/mma.jl` | `mma.sync.aligned.<shape>.<layA>.<layB>.<d>.<a>.<b>.<c>` for bf16/f16/tf32/FP8 (Ada) and `kind::f8f6f4` 5×5 sub-byte FP A/B (Blackwell sm_100a+); takes/returns `NTuple{N, UInt32}` for A/B and `NTuple{M, Float32\|UInt32}` for C/D |
| `mma.sync.aligned.kind::mxf*` (block-scaled) | `wrappers/mma_scaled.jl` | Three Blackwell-introduced kinds: `mxf4`, `mxf4nvf4`, `mxf8f6f4`. Operand layout `(scale_data::UInt32, byte_id::UInt16, thread_id::UInt16)` per side per PTX 9.2 §9.7.14.3 |
| `setp.dual` | `wrappers/setp.jl` | `setp.<cmp>.<dtype>` with `%p\|%q` dual-pred output — 6 cmps × 12 dtypes = 72 generated methods returning `Tuple{Bool, Bool}` |
| `shfl.sync` | `wrappers/shfl.jl` | `up` / `down` / `bfly` / `idx` × `b32` × {data-only, data+pred} |
| `stmatrix` | `wrappers/stmatrix.jl` | mirror of `ldmatrix` — `m8n8.b16` (sm_70+) and `m16n8.b8` (Hopper) |
| `vec_ldst` | `wrappers/vec_ldst.jl` | `ld.global.v{2,4}.{f32,b32,b16}` / `st.global.v{2,4}.{f32,b32,b16}` — braced register-vector I/O for HBM-saturating bandwidth |
| `wgmma.mma_async` (Hopper sm_90a) | `wrappers/wgmma.jl` | `wgmma.mma_async.sync.aligned.m64nNk{8,16,32}.<d>.<a>.<b>` — accumulator passed by value (tied operands), N stepped by 8 from 8 to 256, 12 dtype tuples × 32 N-values = 384 methods |
| `tcgen05` (Blackwell sm_100a/sm_110a) | `wrappers/tcgen05.jl` | Exact lifecycle, fence/wait, TMEM address, load/store, and dense-MMA forms. `shift` / `dealloc` / `cp` / `ld` / `st` take a 32-bit TMEM address returned by `tcgen05.alloc`, while alloc/commit use reviewed shared-memory carriers. |

The simple WGMMA synchronization ops (`wgmma.fence`,
`wgmma.commit_group`, `wgmma.wait_group`) still flow through the reviewed
generic chain contract. Every `tcgen05` form is typed-wrapper-only, including
the ptxas-covered pointer forms of `alloc` and non-multicast `commit` and the
before/after thread-sync fences. An exact method is authoritative and a
dispatch miss fails before generic asm rendering. This is necessary because
the shared opcode spans address destinations, register vectors, sinks,
descriptors, fences, and MMA operands with incompatible schemas.

## Extended-precision arithmetic

The extended-precision wrapper is a correctness boundary, not only an
ergonomic overload. PTX `CC.CF` is implicit state that LLVM cannot name as an
input/output constraint. Scalar typed wrappers reify it as `Bool`; aggregate
helpers keep the entire chain in one asm unit. Generic, raw, and transpiled
instruction-at-a-time fallbacks are rejected instead of emitting optimizer-
unsafe calls.

The public helpers are [`PTX.add_with_carry`](@ref),
[`PTX.sub_with_borrow`](@ref), and [`PTX.mul_wide`](@ref); their full API
documentation is in the [reference](reference.md#Extended-precision-arithmetic).

The blocks are per-thread (`convergent=false`), carry `sideeffect` and `~{cc}`
but no false memory clobber, and use early-clobber word outputs so a low result
cannot alias a high input that a later instruction still needs. Subtraction
seeds and materializes the flag through `sub.cc`/`subc`, directly mirroring PTX
borrow semantics instead of relying on a cross-family flag representation.

## Host-side descriptor builders

Two opcodes consume packed 64-bit shared-memory descriptors plus
(Blackwell only) a 32-bit instruction descriptor. PTX.jl ships pure
bit-packing helpers for both:

| Helper | Used by |
|---|---|
| `wgmma_descriptor` | Hopper `wgmma.mma_async` SMEM operand encoding (14-bit field windows + swizzle + base offset) |
| `tcgen05_descriptor` | Blackwell `tcgen05.mma` SMEM operand encoding (3-bit layout vs wgmma's 2-bit; adds `version` and `lbo_mode`) |
| `tcgen05_instr_desc_f16bf16_f32` | Blackwell `tcgen05.mma` 32-bit instruction descriptor for the dense F16/BF16/TF32 → F32 path. Mirrors CUTLASS/CuTe's `UMMA::make_instr_desc`. |
| `smem_addr_u32` | Convert a `Core.LLVMPtr{T, AS.Shared}` to its 32-bit in-CTA SMEM offset (used as the `smem_addr_u32` argument to the descriptor builders). |

These are not exported but are part of the documented API. Access them
as `PTX.wgmma_descriptor`, `PTX.tcgen05_descriptor`, etc.

## GMMA layout helpers

For `wgmma.mma_async` SMEM operands, the descriptor's
`leading_byte_offset` / `stride_byte_offset` / `swizzle` triple is
fully determined by the tile geometry (dtype, M-or-N, K, major axis).
The four canonical GMMA layout families (`INTERLEAVE` / `B32` / `B64`
/ `B128`) cover all wgmma-compatible SMEM tile widths.

```@docs
PTX.pick_gmma_layout
PTX.layout_for_a
PTX.layout_for_mn_major
```

## Host-side TMA descriptor encoder

Hopper TMA (`cp.async.bulk.tensor.*`) consumes a 128-byte
`CUtensorMap` blob built host-side by the CUDA driver's
`cuTensorMapEncodeTiled`. PTX.jl wraps the driver call so descriptors
take Julia types / symbols instead of raw `CUtensorMapDataType` enums,
with a thin convenience helper for the common 2D row-major case.

These methods live in `ext/PTXCUDACoreExt.jl` and load automatically
when `CUDACore` is in the environment.

```@docs
PTX.tensor_map_encode_tiled
PTX.tensor_map_tile_2d
```

## When to extend

Most chain-default coverage is sufficient. Reach for a wrapper when:

- ptxas rejects the chain output ("Arguments mismatch", wrong constraint
  letter on a shared-AS pointer, missing brackets on a memory operand);
- the op returns multiple values;
- the operand layout has a brace group, tensor-coord vector, or other
  shape with no `$N` rendering;
- the dispatch needs to key on a fragment shape or count that the chain
  can't see.

The pattern: copy the closest existing wrapper file, adjust the table
and asm template, add a `for` loop that emits one typed method per
combination. ~80 LOC for a new family in most cases.
