```@meta
CurrentModule = PTX
```

# Transpiler

The transpiler turns existing PTX source — from `nvcc -ptx`, Triton,
CUTLASS, NVIDIA samples — into idiomatic Julia where each register is a
variable and each instruction is a `ptx"..."(...)` call.

```julia
using PTX

source = read("kernel.ptx", String)
julia_src = ptx_to_julia(source)
println(julia_src)
```

For the supported instruction-at-a-time subset, output is `Meta.parse`-valid:
paste it into a Julia file (or `eval` it), add a `@cuda` launch, and the kernel
runs. Unsupported semantic shapes throw instead of emitting plausible but
unsafe Julia. In particular, `add.cc` / `addc`, `sub.cc` / `subc`, and
`mad.cc` / `madc` communicate through implicit `CC.CF`; splitting them into
independent calls would hide the dependency from LLVM. Write those operations
with the explicit-Bool wrappers or fused helpers described in
[Extended precision and `CC.CF`](dsl.md#Extended-precision-and-CC.CF).

## Pipeline

```
PTX text  ─tokenize→  Vector{Token}  ─parse→  IR.Module  ─ir_to_julia→  Julia source
                                              │
                                              └─format(::Module)→ PTX text  (round-trip)
```

Three independent stages, each usable on its own:

- [`PTX.Parser.tokenize(source)`](@ref) — text → `Vector{Token}` with
  newline / comment tokens preserved for round-trip fidelity.
- [`PTX.Parser.parse(source)`](@ref) — text → `IR.Module`. Opcode-agnostic;
  unrecognized lines round-trip as `RawLine`.
- [`PTX.IR.format(mod)`](@ref) — `IR.Module` → text. Returns
  `raw_source` verbatim when set (the lossless fast path); otherwise
  falls back to structural emission, consulting per-statement
  `raw_line` snapshots first.
- [`ptx_to_julia(source)`](@ref) ≡ `ir_to_julia(parse(source))`.

## Round-trip fidelity

The parser captures three layers of source text:

- **`Module.raw_source`** — the entire input file. `format(mod)`
  returns it verbatim when set (the lossless escape valve).
- **`Module.raw_header`** — the `.version` / `.target` /
  `.address_size` block.
- **`FormattingInfo.raw_line`** per statement — the captured source
  text for that single statement. Used when the structural emitter
  reaches a node that hasn't been reconstructed.

Programmatically constructed IR (e.g. by transformations) falls through
to structural emission. `format(parse(source))` is byte-identical for
all 10 corpus kernels under `test/corpus/` (covering minimal /
vector_add / predicates / branches / shared_memory / function_call /
mbarrier_full / wgmma_simple / cluster_ops + a 579-line
`less_slow_sm90a.ptx`).

## Transpiler output

Each Julia function carries a `# @ptx_kernel` metadata header that
preserves the original `.param` ABI:

```julia
# @ptx_kernel arch=sm_89 version=8.5
#   raw_params  = [("u64.ptr.global.palign16", "param_0"), ("u64.ptr.global.palign16", "param_1"), ("u64.ptr.global.palign16", "param_2")]
#   directives  = []
function vector_add(param_0, param_1, param_2)
    # ... body
end
```

The `raw_params` strings carry the lossless `.param` declarations in a
dot-separated form — `.param .u64 .ptr .global .align 16 param_0` ↔
`"u64.ptr.global.palign16"`. A future v2.1 sugar pass will use these to
emit typed pointer parameters (`param_0::Core.LLVMPtr{Float32,
AS.Global}`), but v2.0 keeps them untyped — Julia kernel arguments ARE
the values, no `ld.param` rebinding is needed.

Mechanical mapping rules (v2.0):

| PTX | Julia |
|---|---|
| `add.s32 %r1, %r0, %r0;` | `r1 = ptx"add.s32"(r0, r0)` |
| `ret;` | `return nothing` |
| `bra DONE;` | `@goto DONE` |
| `@p bra DONE;` | `if p; @goto DONE; end` |
| `@p mov.b32 %r1, 1;` | `if p; r1 = ptx"mov.b32"(UInt32(1)); end` |
| `cvt.rn.f32.u8 %f1, 17;` | `f1 = ptx"cvt.rn.f32.u8"(UInt8(17))` |
| `cvt.rs.satfinite.e4m3x2.f32 %h1, {%f0, ...}, %r0;` | `h1 = ptx"cvt.rs.satfinite.e4m3x2.f32"(...)` |
| `ld.param.u64 %rd0, [param0];` | `rd0 = param0` |
| `setp.lt.f32 %p0\|%p1, %f0, %f1;` | `(p0, p1) = ptx"setp.dual.lt.f32"(f0, f1)` |
| `setp.eq.f16x2 %p0\|_, %r0, %r1;` | `(p0, _) = ptx"setp.eq.f16x2"(r0, r1)` |
| `lop3.or.b32 _\|%p0, %r0, %r1, %r2, ((1 << 4) \| 3), %p1;` | `(_, p0) = ptx"lop3.or.b32"(r0, r1, r2, Val(19), p1)` |
| `match.all.sync.b64 %r0\|%p0, %rd0, %r1;` | `(r0, p0) = ptx"match.all.sync.b64.pred"(rd0, r1)` |
| `elect.sync _\|%p0, 0xffffffff;` | `(_, p0) = ptx"elect.sync"(UInt32(0xffffffff))` |
| `shfl.sync.bfly.b32 %r\|%p, ...;` | `(r, p) = ptx"shfl.sync.bfly.b32.pred"(...)` |
| `{ ... }` register-lifetime block | `let ... end` |
| `LBL:` label | `@label LBL` |

Predicated assignments hoist a `local` declaration so the variable is
visible after the `if`-block. Special registers are emitted as
`sreg"%tid.x"` calls, not `threadIdx().x` — chain-faithful, and avoids
the 0-vs-1-based off-by-one trap. The scalar PTX 9.3 inventory includes
vector-component aliases such as `%tid.w` and `%tid.r`; whole `.v4`
special-register values such as `%tid` are rejected until vector-valued
IR/lowering is implemented. A standalone PTX predefined immediate `WARP_SZ`
lowers to `Val(32)`; its token becomes `32` inside a parsed PTX constant
expression.

Structured destinations are validated against the same closed PTX 9.3 schema
used by direct calls. The transpiler proves each named destination's declared
register type before emitting Julia, preserves one legal PTX sink `_` as tuple
destructuring, and rejects `_ |_`, an undeclared destination, or an
incompatible carrier before alias propagation can erase the bad definition.
General `setp`'s optional complement uses the synthetic leading `.dual` token;
`match.all`'s optional predicate uses trailing `.pred`. Neither selector is
printed in the reconstructed PTX instruction head.

`lop3` LUTs are normalized to `Val(N)` by a non-evaluating PTX integer-constant
interpreter. It covers every constant-expression shape the current parser can
represent: decimal/hex/octal integers, `WARP_SZ`, unary operators,
`.s64`/`.u64` casts, arithmetic, shifts, comparisons, bitwise AND/OR,
logical operators, and parentheses, with PTX-style 64-bit range checks. The
remaining legal lexical forms—including binary literals, unsigned-suffix
literals, XOR, and the ternary operator—remain parser work under
`FRONT-LEXER-001`; they are not
silently treated as runtime LUTs. The wider `IMMEDIATE-001` finding remains
open for `setmaxnreg` and `pmevent`, while its `lop3` range/constness slice is
closed by this schema.

Ordinary `cvt` sources are position-aware. The transpiler closes the reviewed
destination/source carrier pairs and the structural operand roles of stochastic
and scaled forms, while leaving the full rounding/saturation prefix
cross-product to ptxas. In particular, narrow and packed floating sources are
not fabricated from Julia numeric literals, stochastic random bits must be a
declared 32-bit register, and scaled forms type their scale operand separately.
See [Ordinary `cvt` constants](dsl.md#Ordinary-cvt-constants).

## Diff against the original PTX

`PTX.IR.diff` compares two `IR.Module`s and returns a list of
human-readable difference lines (cosmetic content like comments and
blank lines is filtered on the fly):

```julia
m1 = PTX.Parser.parse(read("a.ptx", String))
m2 = PTX.Parser.parse(read("b.ptx", String))
diffs = PTX.IR.diff(m1, m2)
isempty(diffs) || foreach(println, diffs)
```

Pass `entry_only = true` to ignore non-entry `.func` helpers. Module-level
directives still compare, since globals, pragmas, and opaque fallback lines
can affect an `.entry` kernel.

See the [Reference](reference.md) page for full docstrings of
[`ptx_to_julia`](@ref) and [`ir_to_julia`](@ref).
