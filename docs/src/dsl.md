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

Some structured surfaces are **typed-wrapper-only**. All `mma.*`,
`wgmma.mma_async*`, and all `tcgen05.*` forms, plus the internal
multi-result selector `shfl.*.pred`,
reject a dispatch miss before generic asm is rendered. Result groups, tuple
widths, tied accumulators, address roles, or synthetic selector tokens cannot
be recovered by the scalar trailing-type rule. The established `tcgen05`
pointer forms of `alloc` and `commit` and the no-argument thread-sync fences
also have exact methods, so wrong carriers or arity cannot reopen the generic
path. A rejection usually means the modifier spelling, arity, tuple width, or
Julia carrier type does not match a reviewed wrapper.

`mbarrier` uses a separate closed form schema rather than a family-wide result
guess. Every generic or raw chain must match an exact reviewed modifier form
and operand role. The audited synthetic selector `.sink` preserves a PTX `_`
arrival destination; this is mandatory when a generic address names a remote
cluster mbarrier. `.report_pred` and `.report` choose
`waitComplete|reportPredicate` without or with the optional `reportValue`.
Only the emitted instruction head drops these selectors. The two
space-before-sem/scope `arrive_drop` heads printed by the ISA are accepted as
provenance-marked aliases and normalized to canonical emitted PTX. Ordinary
mbarrier addresses accept base-plus-constant-offset syntax, but reject TMA
coordinate lists. PTX's opaque report value is one byte. Because NVPTX has no
i8 inline-asm constraint, the full-result call carries it in the low byte of a
zero-extended `UInt16`. Every mbarrier lowering route also carries a call-site
`convergent nomerge` barrier, matching the complete NVVM mbarrier surface.

`setp`, `lop3`, `match.sync`, and `elect.sync` use a second closed schema
boundary for structured results. The PTX 9.3 ledger contains 1,114 exact API
forms: 768 general `setp`, 336 half/bfloat `setp`, three `lop3`, six
`match.sync`, and one `elect.sync`. It records emitted modifier order, result
grouping, source carriers, legal `_` positions, PTX version, target floor, and
ISA section. A spelling anywhere in one of these opcode islands must match the
ledger even on the raw tier; a misspelled grouped form cannot fall back to a
plausible scalar ABI.

Synthetic Julia-only modifiers select optional PTX destinations without
changing emitted text. A leading `.dual` requests general `setp`'s
compare/complement pair, and a trailing `.pred` requests `match.all`'s
optional predicate. Packed `.f16x2`/`.bf16x2` comparisons, BoolOp `lop3`, and
`elect.sync` are intrinsically grouped and need no selector. Direct calls
materialize every result; transpilation preserves a legal PTX `_` as `_` in
tuple destructuring. CUDA 13 ptxas rejects `_ |_`, so at most one result may
be discarded. The collective `match.sync` and `elect.sync` calls carry
call-site `convergent nomerge` in addition to their conservative memory
clobber.

```julia
p, q = ptx"setp.dual.lt.and.f32"(a, b, gate)
lane0, lane1 = ptx"setp.eq.f16x2"(packed_a, packed_b)
d, p = ptx"lop3.or.b32"(a, b, c, Val(0x96), gate)
matching = ptx"match.any.sync.b64"(value, UInt32(0xffffffff))
all_mask, all_p = ptx"match.all.sync.b64.pred"(value, UInt32(0xffffffff))
leader, elected = ptx"elect.sync"(UInt32(0xffffffff))
```

`ptx"..."raw` is an explicit opt-out from the reviewed form registry and most
typed-wrapper boundaries, not from audited result ABIs. It emits the requested
text under the maximally conservative contract, but generally does not
validate modifier grammar, operand grouping, PTX version, or target support.
The audited fixed-scalar and structured-result ledgers remain
enforced on raw calls: an exact form has the same result ABI and carrier checks
as the registered chain, while an in-island miss is rejected because raw has
no syntax for declaring an alternative result ABI. Raw does not turn a
malformed scalar rendering into a valid matrix or grouped-result instruction.
The implicit-`CC.CF` family is stricter still: its hidden dependency makes even a
standalone raw call unsafe, so raw extended-precision forms remain forbidden.
Likewise, a reviewed pure value operation whose spelling would infer `Nothing`,
or a noncanonical ordinary `cvt`, fails on both normal and raw paths instead of
emitting destination-less asm with an unknown result ABI.

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

For generic scalar forms, the terminal modifier of the chain, if it's a
recognized PTX dtype suffix, gives the return type. A closed audited ledger
handles scalar grammar islands where that convention is false; a spelling in
one of those islands must match the ledger exactly and must use the reviewed
operand carriers.

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

For reviewed pure value opcodes, an unrecognized trailing modifier is an error:
a pure PTX instruction cannot silently become destination-less asm. Reviewed
sink and side-effect families whose contract is genuinely void still return
`Nothing`.

The fixed-result ledger contains 126 exact spellings. Of these, 121 are
canonical ISA grammar (`provenance = :isa`). Five are the exact contradictory
spellings printed by PTX ISA examples—three mixed-float examples,
`add.s8x4.sat`, and `min.s16x2.relu`—and are marked
`provenance = :ptxas_compat`. This compatibility layer is deliberately not a
license to accept every modifier permutation ptxas happens to assemble;
undocumented postfix or duplicated variants fail loud. Each schema also records
its PTX version and target feature set, including `sm_120f` family-specific
packed operations.

The exceptions are:

- **Ordinary `cvt`** — grammar is `cvt.<modifiers...>.<dst>.<src>`, so the
  destination is `parts[end-1]`. Both terminal tokens must be recognized dtypes
  in that canonical order; `cvt.rn.f16.f32` returns `Float16`. A reviewed
  schema closes 193 destination/source/operand shapes over 21 source formats.
  It validates the result/source carriers and the extra operand positions of
  stochastic and scaled forms; ptxas remains authoritative for the complete
  rounding/saturation prefix cross-product. Contradictory reversed/postfix
  examples remain unsupported rather than being assigned a plausible but wrong
  ABI.
- **`cvt.pack`** — every reviewed pack form returns `UInt32`; its conversion,
  source, and optional carry-in width tokens all describe inputs.
- **`setp`** — general `setp.<cmp>[.<boolop>][.ftz].<dtype>` returns `Bool`;
  the leading Julia-only `.dual` selector returns `(compare, complement)` as
  `Tuple{Bool,Bool}`. Scalar `.f16`/`.bf16` also return `Bool`, while packed
  `.f16x2`/`.bf16x2` always return the independent low/high lane predicates as
  `Tuple{Bool,Bool}`. The trailing modifier describes the *input* compare
  type, not the output.
- **`lop3`** — `lop3.b32` returns `UInt32`; `.or.b32` and `.and.b32` return
  `Tuple{UInt32,Bool}` and take the predicate source last. The LUT is a
  compile-time `Val{N}` with `N` in `0:255`, so a runtime or out-of-range LUT
  cannot leak into inline assembly.
- **`match.sync` / `elect.sync`** — every match mask and elected lane is
  `UInt32`, including `match.*.b64`; trailing `.pred` adds `match.all`'s
  predicate. `elect.sync` always returns `Tuple{UInt32,Bool}`.
- **Mixed-precision `add` / `sub` / `fma`** — the reviewed
  `.f32.{f16,bf16}` forms return `Float32`; the final token describes their
  narrow multiplicand carrier.
- **`popc` / `clz`** — both `.b32` and `.b64` source forms return `UInt32`.
- **`dp2a` / `dp4a`** — the accumulator and result are `UInt32` only when
  both multiplicand modifiers are `.u32`; otherwise both are `Int32`.
- **`mul.wide` / `mad.wide`** — `.u16`/`.s16` inputs produce a 32-bit result,
  while `.u32`/`.s32` inputs produce a 64-bit result; `mad.wide` takes its
  accumulator at that widened width too.
- **`prmt.b32.<mode>`** — `.f4e`, `.b4e`, `.rc8`, `.ecl`, `.ecr`, and `.rc16`
  are control modes, not result types. All six return `UInt32` from three
  `.b32`-compatible sources. Base `prmt.b32` remains terminal-inference-safe.
- **Packed integer arithmetic** — reviewed `add`/`sub`/`neg`/`min`/`max`
  `.u16x2`/`.s16x2`/`.u8x4`/`.s8x4` forms use `UInt32` bit carriers for packed
  operands and results. The `.relu.s32` min/max forms instead return `Int32`.
- **No-return families** — `setmaxnreg.{inc,dec}.sync.aligned.u32` and
  `tensormap.replace.tile.<field>...b{32,64}` carry a trailing width modifier
  that describes an *input* operand. Their contracts treat them as void;
  otherwise ptxas would reject with "Arguments mismatch". The analogous
  `tcgen05` sinks (`alloc`, `commit`, `relinquish_alloc_permit`) are handled by
  exact typed wrappers instead of terminal-type inference.

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

### Ordinary `cvt` constants

PTX integer and floating-point constants acquire their effective type from the
instruction operand position. Ordinary `cvt` lowering therefore uses the
reviewed source schema instead of applying the destination type to every source:
integer immediates are accepted for `.u8` through `.s64`, and floating
literals for `.f32` and `.f64`. Exact `0f...` and `0d...` literals keep
their bit patterns and are converted to the declared source width, even when
their spelling uses the other exact-literal width.

Integer immediates use PTX's use-site conversion rule: the 64-bit integer
constant is reduced modulo the operand width before the typed Julia call. A
non-eval evaluator first applies PTX's fixed `.s64`/`.u64` expression rules,
so `(0xffffffff << 32)` remains `0xffffffff00000000` rather than inheriting
Julia's 32-bit hexadecimal-literal width. Thus a `.u8` source of `256`
carries `UInt8(0)`, while a `.s8` source of `255` carries `Int8(-1)`;
integer literals exceeding 64 bits fail loud.

The evaluator covers the immediate tokens and expressions the current frontend
already structures, including decimal, hexadecimal, C-style octal,
`WARP_SZ`, unary operators, casts, arithmetic, shifts, comparisons, bitwise
AND/OR, and logical AND/OR. PTX `U` suffixes, binary literals, XOR, ternary
expressions, and other frontend gaps remain tracked under `FRONT-LEXER-001`
and are not claimed here.

PTX constants cannot directly carry `.f16`, `.bf16`, or packed alternate
floating-point source formats, so those positions require registers rather than
silently retyping a Julia number. Stochastic x4 forms additionally require a
four-element vector of declared `.f32` or `.b32` registers and a declared
32-bit random-bits register; `.u32`/`.s32` declarations do not satisfy the
floating source role. Scaled forms use a separate `.b16` scale position.

CUDA 13.3 ptxas currently reports internal error C7907 when a live
`.s2f6x2` conversion result reaches code generation, including for minimal
PTX without debug metadata. The offline evidence suite therefore pins live
`sm_121a` PTX emission and separately assembles a dead-result syntax/target
probe; it makes no runtime claim for the `.s2f6x2` forms.

## Extended precision and `CC.CF`

PTX extended-precision arithmetic is not an ordinary scalar chain. The
`add.cc` / `sub.cc` / `mad.*.cc` producers and `addc` / `subc` / `madc`
consumers communicate through the implicit per-thread flag `CC.CF`. LLVM
inline-assembly operands cannot express that dependency between two calls, and
the PTX ISA does not preserve the flag across function calls.

PTX.jl therefore makes carry and borrow explicit at the Julia boundary:

```julia
lo, carry = ptx"add.cc.u32"(a0, b0)
hi, carry = ptx"addc.cc.u32"(a1, b1, carry)
top        = ptx"addc.u32"(a2, b2, carry)

lo, borrow = ptx"sub.cc.u32"(a0, b0)
hi         = ptx"subc.u32"(a1, b1, borrow)
```

Each typed call seeds and/or materializes `CC.CF` as needed inside one opaque
asm block; the extra `Bool` argument/result is an SSA dependency visible to
LLVM. For a whole multi-limb operation, prefer the fused helpers:

```julia
sum, carry = PTX.add_with_carry(a_words, b_words)
sum, carry = PTX.add_with_carry(a_words, b_words, carry_in)

difference, borrow = PTX.sub_with_borrow(a_words, b_words)
product = PTX.mul_wide(a_2words, b_2words)
```

Limb tuples are little-endian. Add/sub accept equally sized nonempty tuples of
`UInt32`, `Int32`, `UInt64`, or `Int64`; `mul_wide` accepts two two-limb
unsigned operands with 32- or 64-bit limbs and returns four limbs. Each fused
helper is exactly one side-effecting, non-convergent asm call with early-clobber
outputs and a `~{cc}` clobber. It does not claim a memory effect.

The scalar wrappers are correctness adapters for isolated composition, not the
efficient way to spell a long chain. For an `N`-limb add/sub that returns the
final flag, separate scalar wrappers emit `5N - 2` PTX instructions without
an incoming flag (`5N` with one), while the fused helper emits `N + 2`
(`N + 4` with one). At four limbs without carry/borrow-in, that is 18 versus
6 instructions. All of these asm blocks are marked `sideeffect`, so LLVM must
not CSE or freely hoist identical calls. Prefer the fused helpers in hot loops.
If optimizer visibility matters more than preserving a specific PTX chain,
compare ordinary Julia integer arithmetic on the target toolchain; that is a
separate lowering path and does not guarantee the emitted instruction sequence.

All 48 legal PTX 9.3 §9.7.2 spellings have typed wrappers. A generic or
`ptx"..."raw` standalone spelling is deliberately rejected: `raw` cannot make
the hidden dependency visible. The instruction-at-a-time PTX transpiler also
rejects these forms until it can fuse a complete straight-line chain.

Compatibility is independent by family: 32-bit add/sub forms date to PTX 1.2
and all targets; 32-bit `mad`/`madc` date to PTX 3.0 and require `sm_20`; every
64-bit extended-precision form dates to PTX 4.3 and requires `sm_20`.

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
