```@meta
CurrentModule = PTX
```

# Chain DSL

The user-facing surface for writing PTX is one string macro:

```julia
ptx"opcode.mod1.mod2..."(args...)
```

`@ptx_str` splits the literal on `.`, builds an `Operation{parts}`
singleton, and the call site is `@generated`. From the chain parts plus
the argument types, PTX.jl derives:

- the asm template string,
- LLVM constraint letters,
- side-effect classification + `~{memory}` clobber,
- return type,

and emits a single `@asmcall`.

```julia
ptx"add.f32"(a, b)                                 # Float32 → Float32
ptx"fma.rn.f32"(a, b, c)
ptx"atom.add.gpu.u32"(p, v)                        # *p += v atomically
ptx"bar.sync"(Val(0))                              # immediate baked in
ptx"mov.u32"(sreg"%tid.x")                         # special-register read
ptx"cp.async.commit_group"()
ptx"cp.async.wait_group"(Val(0))
ptx"fence.acq_rel.gpu"()
```

This handles the vast majority of PTX with no per-op declaration. For
ops whose operand layout breaks the chain default — multi-output, mixed
address-space pointers, fragment-shape dispatch, descriptor packing —
hand-written wrappers register typed methods on the same
`Operation{...}` singleton. Same call site, no user-visible change. See
[Wrappers](wrappers.md).

## Modifier syntax

The macro splits on `.`; each segment becomes one Symbol verbatim.
`::` (PTX sub-namespace separator), digit-leading tokens (`3d`,
`m16n8k32`), and underscores in modifier names all flow through cleanly:

```julia
ptx"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"(a, b, c)
ptx"cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(...)
```

Empty parts (consecutive `.`, leading/trailing `.`, or empty string)
error at expansion.

## Return type inference

The terminal modifier of the chain, if it's a recognized PTX dtype
suffix, gives the return type:

| Modifier | Julia type |
|---|---|
| `.f64` / `.f32` / `.f16` | `Float64` / `Float32` / `Float16` |
| `.u64` / `.u32` / `.u16` / `.u8` | `UInt64` / `UInt32` / `UInt16` / `UInt8` |
| `.s64` / `.s32` / `.s16` / `.s8` | `Int64` / `Int32` / `Int16` / `Int8` |
| `.b64` / `.b32` / `.b16` / `.b8` | `UInt64` / `UInt32` / `UInt16` / `UInt8` |
| `.bf16` / `.tf32` | `UInt16` / `UInt32` (bit-pattern carrier) |
| `.pred` | `Bool` |
| `.f16x2` / `.bf16x2` | `UInt32` (packed FP carrier) |
| `.e4m3x2` / `.e5m2x2` / `.e2m1x2` / `.e3m2x2` / `.ue8m0x2` / … | `UInt16` (packed FP carrier) |
| `.e4m3x4` / `.e5m2x4` / `.e2m3x4` / `.e3m2x4` / `.e2m1x4` | `UInt32` (packed FP carrier) |
| `.f32x2` | `UInt64` |

If the trailing modifier isn't a recognized dtype, the chain emits a void
asm and returns `Nothing`.

Three families break this rule and are special-cased:

- **`cvt`** — grammar is `cvt.<modifiers...>.<dst>.<src>`, so the
  destination is `parts[end-1]`. `cvt.rn.f16.f32` returns `Float16`.
- **`setp`** — `setp.<cmp>.<dtype>` always returns `Bool`. The trailing
  modifier describes the *input* compare type, not the output.
- **No-return families** — `setmaxnreg.{inc,dec}.sync.aligned.u32`,
  `tensormap.replace.tile.<field>...b{32,64}`, and the `tcgen05` sinks
  (`alloc`, `commit`, `relinquish_alloc_permit`) all carry a trailing
  width modifier that describes an *input* operand. The chain treats
  them as void; otherwise ptxas would reject with "Arguments mismatch".

### Packed FP carriers

PTX FP8 / FP6 / FP4 lanes have no native Julia primitive, so the chain
returns integer carriers (`UInt8` / `UInt16` / `UInt32`) of matching
width. NVPTX register classes (`h`/`r`/`l`) match these, and downstream
packages can layer a primitive type with `getindex` for lane extraction
on top:

```julia
primitive type Float8_E4M3x2 16 end          # 2 lanes × 8 bits
function Base.getindex(x::Float8_E4M3x2, i::Int)
    bits = reinterpret(UInt16, x)
    byte = i == 1 ? UInt8(bits & 0xFF) : UInt8(bits >> 8)
    reinterpret(Float8_E4M3FN, byte)
end
```

A 16-bit primitive lowers to LLVM `i16`, fitting the same `h` constraint
as `UInt16`. `reinterpret` between the carrier and the semantic type is
zero-cost.

## Side-effect classification

Inline asm is opaque to LLVM; without explicit annotation, LLVM may
DCE/CSE/fold or reorder it. The chain marks an op nonpure (`side_effects
= true` + `~{memory}` clobber) when:

- the opcode prefix is one of: `bar`, `mbarrier`, `fence`, `wgmma`,
  `tcgen05`, `cluster`, `cp`, `setmaxnreg`, `elect`, `prefetch`,
  `tensormap`, `ld`, `st`, `atom`, `red`, `ldmatrix`, `stmatrix`,
  `vote`, `shfl`, `match`, `redux`, `activemask`, `membar`, `mapa`,
  `getctarank`, `griddepcontrol`, `clusterlaunchcontrol`, `exit`; or
- any argument is a `SpecialReg` (sreg reads are observable).

Warp-collective ops (`vote`, `shfl`, `match`, `redux`, `activemask`)
need `~{memory}` even though they touch no memory: each lane's result
depends on every other lane's input, and without the clobber LLVM would
hoist or constant-fold them as if they were per-thread pure functions
and silently lose the cross-lane semantics.

## Constraint letters

Per-arg from a small mapping:

| Julia type | Letter | NVPTX register class |
|---|---|---|
| `Float64` | `d` | `f64` |
| `Float32` | `f` | `f32` |
| `Float16`, `Int16`/`UInt16`, `Int8`/`UInt8` | `h` | `i16` (NVPTX has no native i8 register) |
| `Int32`/`UInt32` | `r` | `i32` |
| `Int64`/`UInt64` | `l` | `i64` |
| `Bool` | `b` | `i1` (predicate) |
| `Core.LLVMPtr{T, AS}` | `l` | `i64` for any address space |

Pointer arguments always get `l` regardless of address space — NVPTX
represents non-zero address-space pointers as 64-bit at the LLVM IR
level even when the underlying PTX address is 32-bit (shared, param,
local). Hand-written wrappers for shared-memory ops override this to
`r` where ptxas wants the 32-bit form.

## Special argument shapes

The chain recognizes three argument shapes that don't render as plain
`$N` operands:

### `Val{N::Integer}` — compile-time immediate

Bakes `N` as a decimal literal into the asm string at the operand
position. Consumes no LLVM input slot:

```julia
ptx"bar.sync"(Val(0))                      # → "bar.sync 0;"
ptx"cp.async.cg.shared.global"(dst, src, Val(16))   # 16-byte size baked
ptx"shl.b32"(x, Val(2))                    # shl.b32 r, x, 2
```

### `SpecialReg{name}` — verbatim PTX token

Renders the name verbatim (always with `%` prefix). Constructed via
[`@sreg_str`](@ref):

```julia
sreg"%tid.x"                               # → "%tid.x"
sreg"tid.x"                                # ≡ sreg"%tid.x" (auto-prefixed)
sreg"cluster_ctarank"                      # → "%cluster_ctarank"
ptx"mov.u32"(sreg"%tid.x")                 # → "mov.u32 r, %tid.x;"
```

Underscore-bearing names (`%cluster_ctarank`, `%lanemask_eq`,
`%total_smem_size`) round-trip losslessly because the macro bakes the
exact PTX token.

PTX 9.3 spells the warp-size operand as the immediate `WARP_SZ`, not a
special register. For compatibility, standalone `sreg"%warpsize"` lowers
to `Val(32)`; when the transpiler encounters either spelling inside a parsed
PTX constant expression, it substitutes the literal `32`.

### Homogeneous tuple → braced register-vector

Each tuple element becomes its own LLVM input slot; the asm emits a
braced group:

```julia
# `add.f32x2 d, {a0, a1}, {b0, b1};`
ptx"add.f32x2"((a0, a1), (b0, b1))
```

Used for any op whose operand layout takes `{$N, $N+1, ...}`. Many
multi-output families (`ldmatrix`, `mma`, `stmatrix`) emit braced
operands and are covered by hand-written wrappers — see
[Wrappers](wrappers.md).

## Pointer bracketing

Memory-op opcodes render pointer arguments as `[$N]`; non-memory ops
(`cvta`, `mov`, …) emit unbracketed `$N`. The bracketing set:

- `ld`, `st`, `atom`, `red`, `cp`, `mbarrier`, `ldmatrix`, `stmatrix`,
  `prefetch`, `tcgen05`, `tensormap`, `fence`.

`fence` only takes a pointer in the
`fence.proxy.tensormap::generic.<acq|rel>.gpu [addr], size` form;
argument-less `fence.sc.gpu` forms emit no bracketed operand either way.

See the [Reference](reference.md) page for full docstrings of
[`@ptx_str`](@ref) and [`@sreg_str`](@ref).
