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

Output is `Meta.parse`-valid: paste it into a Julia file (or `eval` it),
add a `@cuda` launch, and the kernel runs. Each emitted line lowers to
the same `@asmcall` site the chain DSL produces, so the resulting PTX
is byte-for-byte close to the original.

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
| `ld.param.u64 %rd0, [param0];` | `rd0 = param0` |
| `setp.lt.f32 %p0\|%p1, %f0, %f1;` | `(p0, p1) = ptx"setp.dual.lt.f32"(f0, f1)` |
| `shfl.sync.bfly.b32 %r\|%p, ...;` | `(r, p) = ptx"shfl.sync.bfly.b32.pred"(...)` |
| `{ ... }` register-lifetime block | `let ... end` |
| `LBL:` label | `@label LBL` |

Predicated assignments hoist a `local` declaration so the variable is
visible after the `if`-block. Special registers are emitted as
`sreg"%tid.x"` calls, not `threadIdx().x` — chain-faithful, and avoids
the 0-vs-1-based off-by-one trap.

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

Pass `entry_only = true` to ignore `.func` helpers and only compare
`.entry` kernels.

See the [Reference](reference.md) page for full docstrings of
[`ptx_to_julia`](@ref) and [`ir_to_julia`](@ref).
