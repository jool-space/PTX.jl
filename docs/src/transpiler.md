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

## Lexical conformance

The lexer enforces PTX's ASCII source boundary before tokenization. Integer
literals accept decimal, hexadecimal, octal, and binary notation with PTX's
immediate uppercase `U` suffix; a leading sign remains a separate unary token.
Decimal floating-point literals accept a decimal point and signed exponent
independently (`.5`, `1.`, and `1e-2`), while exact `0f` and `0d` literals must
carry exactly 8 and 16 hexadecimal digits. Malformed prefixes, suffixes,
exponents, exact encodings, strings, and block comments raise `LexError` at
the construct's starting line and column instead of splitting into plausible
tokens or reaching `RawLine` fallback.

Lines beginning in column one with `#` are retained as one opaque
`PREPROCESSOR` token, including backslash-newline continuations. Parsing
preserves those lines as `RawLine` nodes before the module header and in module
or function bodies. PTX.jl does not execute cpp or choose conditional branches;
source whose structure depends on macro expansion must be preprocessed before
semantic parsing or transpilation.

## Round-trip fidelity

The parser captures three layers of source text:

- **`Module.raw_source`** — the entire input file. `format(mod)`
  returns it verbatim when set (the lossless escape valve).
- **`Module.raw_header`** — the required `.version` / initial `.target` and,
  when present, optional `.address_size` block. Omission retains the ISA-defined
  32-bit semantic default through `Module.address_size` while
  `Module.address_size_explicit == false` prevents structural formatting from
  manufacturing a directive.
- **`FormattingInfo.raw_line`** per statement — the captured source
  text for that single statement. Used when the structural emitter
  reaches a node that hasn't been reconstructed.

Subsequent module-scope `.target` directives are represented as ordered
`TargetDirective` statements. The parser validates the PTX 9.3 target ledger,
target and option introduction versions, duplicate options, and the
module-wide texturing-mode invariant. For a declared PTX version newer than
9.3 the parser retains the grammar invariants but accepts syntactically valid
future architectures and options rather than claiming that the bundled ledger
describes a future ISA. CUDA 12.9 and 13.3 `ptxas` independently confirm the
version/target boundary by rejecting `.version 8.5` with `.target sm_100a`.

Programmatically constructed IR (e.g. by transformations) falls through
to structural emission. `format(parse(source))` is byte-identical for
all 10 corpus kernels under `test/corpus/` (covering minimal /
vector_add / predicates / branches / shared_memory / function_call /
mbarrier_full / wgmma_simple / cluster_ops + a 579-line
`less_slow_sm90a.ptx`).

### Vector declarations

Declaration vector shape is modeled independently from scalar element type on
`IR.RegDecl` and `IR.VarDecl`. The accepted matrix is the closed PTX 9.3
§5.4.2 grammar, rather than a storage-width heuristic:

| shape | fundamental element types |
|---|---|
| `.v2` | `.b8/.b16/.b32/.b64`, `.u8/.u16/.u32/.u64`, `.s8/.s16/.s32/.s64`, `.f16/.f16x2/.f32/.f64` |
| `.v4` | `.b8/.b16/.b32`, `.u8/.u16/.u32`, `.s8/.s16/.s32`, `.f16/.f16x2/.f32` |

These shapes are accepted for `.reg`, `.global`, `.const`, `.local`, and
`.shared` declarations. Predicate and `.b128` elements, `.v4` 64-bit
elements, vector `.param` declarations, and declaration spellings using
`.shared::cta` or `.shared::cluster` are rejected structurally. Alternate
floating-point formats such as `.bf16` and `.tf32` are also excluded: PTX
§5.2.3 does not classify them as fundamental types, even when their storage
width would fit the 128-bit vector limit.

Explicit `.align` and array dimensions remain in declaration order through
parse/format and structural comparison. An omitted alignment stays omitted;
the formatter does not manufacture PTX's default alignment to the vector's
overall size. Invalid vector declaration lines enter the normal `RawLine`
recovery path, so one unsupported declaration does not abort parsing the rest
of a function or module. Aggregate-layout lowering in the PTX-to-Julia
transpiler is a separate support boundary from this lossless IR model.

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
| `ld.global.v2.u32 {%r0, %r1}, [%rd];` | `(r0, r1) = ptx"ld.global.v2.u32"(rd)` |
| wide `ld` with `_` destination lanes | live tuple assignment from `vector_load(..., Val(mask))` |
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
lexer recognizes binary literals, uppercase unsigned suffixes, XOR, and
ternary punctuation, but this deliberately smaller evaluator rejects those
shapes rather than silently treating them as runtime LUTs. The wider
`IMMEDIATE-001` finding remains
open for `setmaxnreg` and `pmevent`, while its `lop3` range/constness slice is
closed by this schema.

Ordinary `cvt` sources are position-aware. The transpiler closes the reviewed
destination/source carrier pairs and the structural operand roles of stochastic
and scaled forms, while leaving the full rounding/saturation prefix
cross-product to ptxas. In particular, narrow and packed floating sources are
not fabricated from Julia numeric literals, stochastic random bits must be a
declared 32-bit register, stochastic x4 source elements must be declared
`.f32`/`.b32` registers (not `.u32`/`.s32`), and scaled forms type their scale
operand separately.
See [Ordinary `cvt` constants](dsl.md#Ordinary-cvt-constants).

Brace-enclosed destinations for the audited vector `ld`, `atom`, and
`multimem.ld_reduce` forms remain structured through lowering. Destination
width, homogeneous lane carriers, atom source-vector width, address placement,
and optional cache-policy position are checked before register-alias
propagation can erase a malformed definition. Partial sink destinations on the
ISA's wide-load forms become `vector_load` masks and destructure only the live
lanes. All-sink loads fail loud because current ptxas rejects them and deleting
a possibly synchronizing load is not justified by the per-lane sink wording.
Likewise, `.L2::cache_hint` requires the 64-bit cache-policy operand that
CUDA 12.9/13.3 ptxas enforce. Register declarations must have an exact-size,
PTX-compatible carrier; the ISA permits widened integer/bit `ld` destinations,
but the transpiler rejects them until its tuple ABI can retain each extended
lane's declared width. Atom source constants are format-aware: CUDA 12.9/13.3
ptxas reject immediate lanes for `.f16`, `.bf16`, and packed-half forms, while
`.f32` accepts floating constants but not integer constants. Accepted `.f32`
constants are explicitly converted at their PTX use-site width; cache-policy
integer expressions use PTX's fixed 64-bit semantics before reduction. A
whole-result atom `_` becomes an unused, side-effecting tuple call rather than
an assignment. A vector spelling inside an audited grammar island that misses
the reviewed schema is rejected instead of falling back to a scalar result.

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
