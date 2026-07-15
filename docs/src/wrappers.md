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
| `cp.async.bulk.tensor` (TMA) | `wrappers/tma.jl` | `cp.async.bulk.tensor.{1..5}d.{shared::cluster\|shared::cta}.global.tile.mbarrier::complete_tx::bytes` (load), `.global.shared::cta.tile.bulk_group` (store), and `cp.async.bulk.prefetch.tensor.{1..5}d.L2.global.tile[.L2::cache_hint]`. Coordinate vectors become positional `Int32` arguments; a cache-hinted prefetch adds one `UInt64` policy operand. Prefetch is a weak, fire-and-forget performance hint, so no runtime result is exposed. |
| `cvt` sub-byte FP packing | `wrappers/cvt.jl` | `cvt.rn.satfinite.e2m1x2.{f32,f16x2,bf16x2}` pack and `cvt.rn.{f16x2,bf16x2}.e2m1x2` unpack — `.b8` carrier through a `mov.b16` brace-pair shim because NVPTX has no i8 constraint |
| extended precision | `wrappers/extended_precision.jl` | All 48 `add.cc` / `addc` / `sub.cc` / `subc` / `mad.cc` / `madc` scalar forms with explicit `Bool` carry/borrow, plus fused arbitrary-limb add/sub and unsigned 2×2-limb multiply |
| `fence.proxy.tensormap::generic` | `wrappers/fence.jl` | Exact PTX 8.3 / sm_90+ acquire and release forms at `{cta,cluster,gpu,sys}` scope. Acquire takes a generic address that resolves to global memory plus the ISA-fixed `Val(128)` range; release takes no operands. All eight lower through NVVM as per-thread, non-convergent ordering operations. |
| `ldmatrix` | `wrappers/ldmatrix.jl` | `m8n8.{x1,x2,x4}[.trans].b16`; `m16n16.{x1,x2}.trans.b8`; and Blackwell optional decompression from `{b6x16_p32,b4x16_p64}` to `b8x16` for `m8n16.{x1,x2,x4}` and `m16n16.{x1,x2}.trans`, in `{shared,shared::cta}`. Returns one `UInt32` per `m8n16` matrix and two per `m16n16` matrix. Generic state-space addressing is not part of the typed wrapper surface. |
| `mbarrier` | `wrappers/mbarrier.jl`, `mbarrier_forms.jl` | Closed PTX 9.3 form schema for lifecycle, tx-count, arrive/drop, waits, pending-count, layout, sem/scope, and CTA/cluster spaces; exact wrappers accelerate common forms while delegating to the same schema emitter. Results are explicit sink (`.sink` → PTX `_`), `UInt64` state, `Bool`, `UInt32` pending count, `(waitComplete, reportPredicate)`, or `(waitComplete, reportPredicate, reportValueCarrier::UInt16)`. The ISA's two noncanonical `arrive_drop` example heads are provenance-marked aliases and normalize on emission. PTX's 1-byte report value occupies the carrier's low byte because NVPTX has no i8 inline-asm constraint. Explicitly shared addresses use the NVPTX `r` constraint; every route carries `convergent nomerge`. |
| `mma.sync.aligned` | `wrappers/mma.jl` | `mma.sync.aligned.<shape>.<layA>.<layB>.<d>.<a>.<b>.<c>` for bf16/f16/tf32/FP8, modern `m16n8` u8/s8 and u4/s4 integer forms, and `kind::f8f6f4` 5×5 sub-byte FP A/B (consumer Blackwell); floating A/B fragments use `NTuple{N, UInt32}` and C/D use `Float32\|Float64\|UInt32`; integer A/B fragments use `UInt32`, with `Int32` C/D and optional `.satfinite` |
| `mma.sp[::ordered_metadata].sync.aligned` | `wrappers/mma.jl` | The same closed 12 sparse floating ABIs for base and ordered metadata: `m16n8k{16,32}` f16/bf16, `m16n8k{8,16}` tf32, and all four e4m3/e5m2 A/B pairs at `m16n8k64`. A/B fragments and metadata use `UInt32` carriers; C/D use `Float32` or packed-f16 `UInt32`. Ordered metadata requires increasing retained indices where a chunk encodes a pair, plus exact shape-dependent selectors (`0:3`, `0:1`, or `0`). |
| `mma.sync.aligned.kind::mxf*` (block-scaled) | `wrappers/mma_scaled.jl` | Three Blackwell-introduced kinds: `mxf4`, `mxf4nvf4`, `mxf8f6f4`. Operand layout `(scale_data::UInt32, byte_id::UInt16, thread_id::UInt16)` per side per PTX 9.2 §9.7.14.3 |
| `shfl.sync` | `wrappers/shfl.jl` | `up` / `down` / `bfly` / `idx` × `b32` × {data-only, data+pred} |
| `stmatrix` | `wrappers/stmatrix.jl` | mirror of `ldmatrix` — `m8n8.b16` (sm_70+) and `m16n8.b8` (Hopper) |
| `vec_ldst` | `wrappers/vec_ldst.jl` | `ld.global.v{2,4}.{f32,b32,b16}` / `st.global.v{2,4}.{f32,b32,b16}` — braced register-vector I/O for HBM-saturating bandwidth |
| `wgmma.mma_async` (Hopper sm_90a) | `wrappers/wgmma.jl` | `wgmma.mma_async.sync.aligned.m64nNk{8,16,32}.<d>.<a>.<b>` — accumulator passed by value (tied operands). Floating forms use all 32 N values stepped by 8 through 256; integer forms use the ISA's 16-value grid `8,16,24,32,48:16:224`. The closed surface is 256 floating + 64 integer shape/type forms, each with SS runtime/constant `scale_d` and RF-A runtime variants. |
| `tcgen05` (Blackwell sm_100a/sm_110a and family targets) | `wrappers/tcgen05.jl` | Exact lifecycle, fence/wait, TMEM address, load/store, dense-MMA, and MX block-scale forms. MX uses the complete seven-operand schema `(d, a_desc_or_tmem, b_desc, idesc, scale_a_tmem, scale_b_tmem, enable_input_d)` and all eight legal `kind × {scale_vec,block}` spellings; the former five-argument MX surface is rejected. `shift` / `dealloc` / `cp` / `ld` / `st` take a 32-bit TMEM address returned by `tcgen05.alloc`, while alloc/commit use reviewed shared-memory carriers. Dense MMA covers both A sources (`UInt64` SMEM descriptor vs `UInt32` TMEM address, dispatched like the ISA's `a-desc` vs `[a-tmem]`), the `collector::a::{fill,use,lastuse}` qualifiers (discard is the spelled-nothing default), `.ashift` (TMEM A; discard/lastuse only), an optional positional disable-output-lane mask (`NTuple{4|8, UInt32}` before `enable_input_d`), and an optional `Val(scale)` scale-input-d immediate (f16/tf32, 0:15). Sparse MMA (`mma.sp`) mirrors the dense grid with the mandatory sparsity-metadata TMEM address between the B descriptor and `idesc`; sp block-scale (MX) records stay outside the surface. Weight-stationary MMA (`mma.ws{.sp}`, `cta_group::1` only) exposes the addressed B-side collector (`collector::bN::{fill,use,lastuse,discard}`, spelled-nothing = `b0::discard`) and the optional trailing `UInt64` zero-column-mask descriptor. |

## Schema-driven structured results

`setp`, `lop3`, `match.sync`, and `elect.sync` no longer rely on a small set of
hand-written dual-result methods. `src/structured_results.jl` expands the
complete PTX 9.3 grammar into 1,114 reviewed schemas and the generic chain
emitter consumes those schemas directly. This keeps scalar and grouped
siblings in the same closed boundary: modifier legality, source carriers,
result tuple shape, sink positions, and target metadata cannot drift between a
wrapper and the fallback.

General `setp` uses the Julia-only leading `.dual` selector for the optional
compare/complement result. Packed `.f16x2` and `.bf16x2` always return the two
lane predicates. BoolOp `lop3` returns `(UInt32, Bool)` and requires a
`Val{N}` LUT with `N` in `0:255`; `match.all.sync.*.pred` selects the optional predicate, and
`elect.sync` always returns `(UInt32, Bool)`. The two warp-collective families
use the same `convergent nomerge` LLVM call-site path as the existing MMA and
mbarrier wrappers.

For the mbarrier full-report form, `reportValue` is a PTX `.b8`
destination. This agrees with the CUDA Runtime API's description of
`cudaFabricOpStatusSourceMbarrierV1` as 1-byte aligned and 1-byte wide, and is
enforced by ptxas evidence at `sm_90` and `sm_121`. The Julia result is a
zero-extended `UInt16` carrier because LLVM's NVPTX inline-asm interface has no
i8 register constraint; only its low byte is the opaque status value.

Arrival calls that spell destination `_` use the synthetic `.sink` selector,
for example `ptx"mbarrier.arrive.sink.release.cluster.b64"(remote_addr)`.
This distinction cannot be inferred from argument types: the otherwise
identical local/generic form returns a `UInt64` state token. It is especially
important for a generic address produced for a remote cluster mbarrier, where
the ISA requires the sink destination. Explicit `shared::cluster` forms are
always sinks and need no selector. Base-plus-constant-offset addresses remain
valid; tensor-coordinate address lists are rejected rather than truncated.

The simple WGMMA synchronization ops (`wgmma.fence`,
`wgmma.commit_group`, `wgmma.wait_group`) still flow through the reviewed
generic chain contract. Every `tcgen05` form is typed-wrapper-only, including
the ptxas-covered pointer forms of `alloc` and non-multicast `commit` and the
before/after thread-sync fences. An exact method is authoritative and a
dispatch miss fails before generic asm rendering. This is necessary because
the shared opcode spans address destinations, register vectors, sinks,
descriptors, fences, and MMA operands with incompatible schemas.

MX target evidence is deliberately split by PTX target class.  The explicit
`.scale_vec::*` spellings are compiled offline through ptxas at `sm_100a` and
`sm_110a`; the equivalent `.block16`/`.block32` aliases are compiled at
`sm_100f` and `sm_110f`.  Those checks validate source schema, inline-asm
constraints, target gating, and assembler acceptance without launching a
kernel.  CUDACore 6.2.1 cannot emit an `sm_110*` module, so those cases emit
the identical body at the matching `sm_100a`/`sm_100f` feature level, assert
that only the `.target` directive changes, and invoke CUDA 13.3 ptxas directly
on the `sm_110a`/`sm_110f` text.  No tcgen05 runtime claim is made for the
available CC 12.1 GB10, which does not implement tcgen05.

## Ordered sparse MMA metadata

`mma.sp::ordered_metadata` is not a looser spelling of ordinary sparse MMA.
For 2:4 f16/bf16/FP8 storage, each metadata nibble must list its two retained
element indices in increasing order; other orderings have undefined behavior.
(tf32 uses one retained index per 1:2 chunk.) PTX.jl cannot infer or repair
that property from an opaque `UInt32` metadata word, so callers must establish
it while pruning and packing. The runtime evidence in
`test/gpu/ampere/gemm_sparse.jl` does this by sorting each retained pair before
encoding the nibble.

The typed surface deliberately mirrors only the 12 sparse MMA ABIs already
supported by `mma.sp`. The ordered qualifier was introduced in PTX 8.5 and the
classic f16/bf16/tf32 forms target `sm_80+`; the FP8 forms require `sm_89+`.
Selector legality is enforced by exact `Val` dispatch: k16 16-bit and k8 tf32
accept `Val(0)` through `Val(3)`, k32 16-bit and k16 tf32 accept `Val(0)` or
`Val(1)`, and k64 FP8 accepts only `Val(0)`. A wrong selector, fragment width,
or unreviewed shape/type combination fails before the generic scalar emitter.
Like all `mma.sync` operations, these calls are warp-collective and carry
`convergent nomerge`; they remain memory-free arithmetic operations.

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
| `tcgen05_descriptor` | Blackwell `tcgen05.mma` SMEM operand encoding (3-bit layout vs wgmma's 2-bit; legal swizzle values only). The ISA-fixed `0b001` field is inserted internally; `lbo_mode=1` selects the restricted `sm_103a` absolute leading-address mode. |
| `tcgen05_instr_desc_f16bf16_f32` | Blackwell `tcgen05.mma` 32-bit instruction descriptor for F16/BF16/TF32 → F32 paths. The integer-only saturation bit and reserved bits are fixed at zero. Mirrors CUTLASS/CuTe's `UMMA::make_instr_desc`. |
| `smem_addr_u32` | Convert a `Core.LLVMPtr{T, AS.Shared}` to its 32-bit in-CTA SMEM offset (used as the `smem_addr_u32` argument to the descriptor builders). |

These are not exported but are part of the documented API. Access them
as `PTX.wgmma_descriptor`, `PTX.tcgen05_descriptor`, etc.

The `tcgen05` builders deliberately expose semantic fields rather than every
bit in the packed values. In particular, `tcgen05_descriptor` has no `version`
keyword: PTX 9.3 §9.7.17.4.1 defines bits 46–48 as the fixed constant
`0b001`, not a version selector. Likewise,
`tcgen05_instr_desc_f16bf16_f32` has no `saturate` keyword because Table 45
defines that bit only for integer MMA kinds. Matrix addresses and byte offsets
must be 16-byte aligned and fit the 18-bit descriptor input window; invalid
swizzle encodings and values that would spill into reserved fields fail with
`ArgumentError` instead of being truncated.

Absolute leading-address mode (`lbo_mode=1`) is the PTX 9.3
§9.7.17.3.1.2 escape hatch for a 48-byte K dimension. The builder enforces
its shared-descriptor restrictions: 128-byte swizzling with 16-byte atomicity
(`BlackwellLayout.B128`, code 2) and `base_offset=0`. Because K and major axes
live outside the shared descriptor, the caller must still pair it with K-major
A and B (instruction-descriptor transpose bits 15 and 16 both zero), K=48B,
and the architecture-specific `sm_103a` target.

```@docs
PTX.tcgen05_descriptor
PTX.tcgen05_instr_desc_f16bf16_f32
```

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
