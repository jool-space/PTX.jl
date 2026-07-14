```@meta
CurrentModule = PTX
```

# Transpiler

The transpiler turns a deliberately closed subset of existing PTX source into
Julia where each register is a variable and each instruction is a
`ptx"..."(...)` call. The parser and formatter accept much broader compiler
output from `nvcc -ptx`, Triton, CUTLASS, and NVIDIA samples; successful
parsing alone is not a claim that the source is semantically transpilable.

```julia
using PTX

source = read("kernel.ptx", String)
julia_src = ptx_to_julia(source)
println(julia_src)
```

For source that passes the closed transpiler contract, the output is Julia
function definitions containing `ptx"..."(...)` calls. Unsupported semantic
shapes throw before any Julia is emitted instead of producing a plausible but
unsafe partial program. Acceptance means that every emitted lowering has an
explicit contract; it is not a claim that every accepted PTX program has been
proved semantically equivalent end to end. Host, offline-compiler, and GPU
runtime evidence are kept as separate test tiers.

In particular, `add.cc` / `addc`, `sub.cc` / `subc`, and `mad.cc` / `madc`
communicate through implicit `CC.CF`; splitting them into independent calls
would hide the dependency from LLVM. Write those operations with the
explicit-Bool wrappers or fused helpers described in
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
- [`PTX.Parser.parse(source)`](@ref) — text → `IR.Module`. Instructions are
  parsed without an opcode inventory. Many post-header lines that the
  statement parser cannot model are retained as opaque `RawLine` nodes;
  lexical errors, invalid initial headers, malformed or misplaced
  `.version`/`.target`/`.address_size` directives, and target invariants still
  throw.
- [`PTX.IR.format(mod)`](@ref) — `IR.Module` → text. Returns
  `raw_source` verbatim when set (the lossless fast path); otherwise
  falls back to structural emission, consulting per-statement
  `raw_line` snapshots first.
- [`ptx_to_julia(source)`](@ref) ≡ `ir_to_julia(parse(source))`.

## Semantic boundary

`ir_to_julia` validates the complete module before writing the first output
line. Every accepted module, function, declaration, body node, control-flow
edge, and instruction operand role is explicit. Unsupported input raises
[`PTX.Codegen.TranspilerError`](@ref), whose `path`, `category`, and `detail`
identify the rejected IR node. It does not emit a plausible partial program or
turn an unknown executable line into a comment.

The current module subset requires 64-bit addressing, exactly one initial
target token with no target options, and at least one function definition. The
declaration subset is scalar `.reg`, scalar `.param`, and
unaligned/uninitialized/unlinked scalar `.shared` storage. Module storage,
pointer metadata, array parameters, vector declarations, function return
parameters, later `.target` directives, pragmas, opaque `RawLine` nodes, and
construction-only scopes reject. Comments and blank lines are nonsemantic.
Programmatically constructed `.file`/`.loc` instruction nodes are also
ignored as debug metadata, but source parsing currently represents those
directives as `RawLine`; textual PTX containing them therefore rejects until
[issue #48](https://github.com/jool-space/PTX.jl/issues/48) gives them
structural IR nodes.

Instructions fall into two groups. Reviewed ABI islands (structured, vector,
scalar, b128, `cvt`, mbarrier, and constant-only forms) use their exact schema.
The remaining scalar subset uses a finite table keyed by the complete opcode
and modifier sequence; that same table drives both preflight and emission,
including result-versus-sink classification and every source carrier. A
general `FormContract` is not enough because optimizer attributes do not prove
operand arity, address positions, or result ABI. Extending the transpiler
therefore requires an explicit form/role entry and evidence, not a fallback
based on the last modifier.

Labels and branches are accepted only when every target exists in the same
PTX brace scope. Shared-symbol pointer alias absorption is deliberately
straight-line: a function containing labels rejects an alias-producing
`mov`/`add`/`sub` instead of guessing which alias reaches a later load or
store.

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

`RawLine` is recovery of source text, not recovery of statement semantics. It
is re-emitted verbatim, retained by normalization, compared by exact text in
`PTX.IR.diff`, and rejected by the transpiler. A parse that contains one can
still be lossless without providing structural or semantic coverage for that
line.

## Round-trip fidelity

The parser captures three layers of source text:

- **`Module.raw_source`** — the entire input file. `format(mod)`
  returns it verbatim when set (the lossless escape valve).
- **`Module.raw_header`** — the required `.version` / initial `.target` and,
  when present, optional `.address_size` block. Omission retains the ISA-defined
  32-bit semantic default through `Module.address_size` while
  `Module.address_size_explicit == false` prevents structural formatting from
  manufacturing a directive.
- **`FormattingInfo.raw_line`** per statement — a captured source line for a
  modeled statement, when that statement exclusively owns the physical line.
  Statement formatters consult this snapshot before reconstructing text from
  fields. It is distinct from an opaque `RawLine` statement.

Subsequent module-scope `.target` directives are represented as ordered
`TargetDirective` statements. The parser validates the PTX 9.3 target ledger,
target and option introduction versions, duplicate options, and the
module-wide texturing-mode invariant. For a declared PTX version newer than
9.3 the parser retains the grammar invariants but accepts syntactically valid
future architectures and options rather than claiming that the bundled ledger
describes a future ISA. CUDA 12.9 and 13.3 `ptxas` independently confirm the
version/target boundary by rejecting `.version 8.5` with `.target sm_100a`.

Programmatically constructed IR (for example, after a transformation) uses
field-driven formatting for the PTX statement kinds that have a structural
spelling. Construction-only `IntrinsicScope` nodes throw instead of being
silently flattened into brace blocks.

The corpus tests distinguish three different claims rather than assigning all
of them to “round trip”:

1. **Lossless acceptance.** For a successfully parsed nonempty source,
   `format(parse(source))` returns `Module.raw_source` byte for byte. Every
   curated input and every parseable (or minimally header-repaired) external
   input exercises this path. This proves parsing completed; it does not prove
   that statements were reconstructed from fields.
2. **Deep structural reconstruction.** The test projection removes
   `raw_source`, `raw_header`, and every nested `FormattingInfo.raw_line`, then
   requires zero `RawLine` nodes and a parse/format fixed point. Every curated
   input is in this tier. External inputs are included only when they are
   fallback-free; the intentional fallback cases are pinned by a manifest
   rather than presented as structural coverage.
3. **Normalized module comparison.** The curated deep tier additionally
   requires `PTX.IR.diff` to find no difference between the first and reparsed
   structural trees. The wider external tier omits this more expensive check.

A deep structural fixed point proves that the modeled IR reconstructs its own
tree consistently. It does not promise byte identity with the original source,
PTX legality, or runtime semantic equivalence; ptxas and device tests provide
separate evidence for those properties.

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
#   raw_params  = [("u64", "param_0"), ("u64", "param_1"), ("u32", "count")]
#   directives  = []
function vector_add(param_0, param_1, count)
    # ... body
end
```

The `raw_params` strings carry the accepted scalar `.param` declarations in a
dot-separated form. Pointer state-space/alignment metadata is not merely
decorative ABI information, so the transpiler rejects it until emitted Julia
signatures can preserve it. Julia kernel arguments are already parameter
values; accepted scalar `ld.param` instructions become local rebinding.

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
| `shl.b64 %rd1, %rd0, 2;` | `rd1 = ptx"shl.b64"(rd0, UInt32(2))` |
| `ld.global.v2.u32 {%r0, %r1}, [%rd];` | `(r0, r1) = ptx"ld.global.v2.u32"(address(rd))` |
| wide `ld` with `_` destination lanes | live tuple assignment from `vector_load(..., Val(mask))` |
| `{ ... }` register-lifetime block | `let ... end` |
| `LBL:` label | `@label LBL` |

Predicated assignments hoist a `local` declaration so the variable is
visible after the `if`-block. Special registers are emitted as
`sreg"%tid.x"` calls, not `threadIdx().x` — chain-faithful, and avoids
the 0-vs-1-based off-by-one trap. The scalar PTX 9.3 inventory includes
vector-component aliases such as `%tid.w` and `%tid.r`; whole `.v4`
special-register values such as `%tid` are rejected until vector-valued
IR/lowering is implemented. The generic ledger admits only thread-index
components with an explicit `.u32` carrier; inventory membership does not
guess the type of other special registers. A standalone PTX predefined
immediate `WARP_SZ` lowers to `Val(32)`. Generic opaque constant expressions
reject; only reviewed constant-expression consumers use the PTX integer
interpreter.

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
immediate contract also validates destinationless `setmaxnreg` and `pmevent`
before ordinary destination inference or pointer-alias propagation.
`setmaxnreg` requires a multiple of eight in `24:256`; `pmevent` requires an
index in `0:15`, and `pmevent.mask` requires a 16-bit mask. Their reconstructed
Julia calls always carry `Val(N)` and never fabricate a destination.

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

`PTX.IR.diff` normalizes two `IR.Module`s and returns human-readable difference
lines. Normalization removes source/header snapshots, the leading prelude,
comments, blank lines, and formatting metadata. It retains module headers,
ordered module directives, function signatures and directives, lexical brace
scopes, instructions, declarations, pragmas, and opaque `RawLine` text:

```julia
m1 = PTX.Parser.parse(read("a.ptx", String))
m2 = PTX.Parser.parse(read("b.ptx", String))
diffs = PTX.IR.diff(m1, m2)
isempty(diffs) || foreach(println, diffs)
```

Pass `entry_only = true` to remove each non-entry `.func` directive in full,
including its declaration, signature, linkage, directives, and body.
Module-level directives still compare, since globals, pragmas, and opaque
fallback lines can affect an `.entry` kernel. `diff` does not canonicalize
register or label names, validate PTX against the ISA, or prove two programs
semantically equivalent.

See the [Reference](reference.md) page for full docstrings of
[`ptx_to_julia`](@ref) and [`ir_to_julia`](@ref).
