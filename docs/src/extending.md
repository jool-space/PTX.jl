```@meta
CurrentModule = PTX
```

# Extending the instruction surface

Maintainer documentation: the recipes for adding coverage, and the
obligations each one carries. Like [Internals](internals.md), nothing here
is public API. Working-environment traps (test invocation, hardware
matrix, rebase hazards) live in the repository-root `CLAUDE.md`.

## First: which tier?

Decide where the new spelling belongs before touching anything:

1. **A recognized NVVM intrinsic exists** (`NVVM.isintrinsic` on the name;
   the registry in `src/nvvm/table.jl` is the authority) — write a tier-2
   wrapper method that emits `NVVM.IntrinsicCall`. Recipe B.
2. **No intrinsic, but the generic chain grammar fits** — the opcode's
   result is the terminal dtype modifier (or void), operands are scalars
   or addresses in PTX order. Bless it with a `FORMS` entry. Recipe A.
3. **No intrinsic and the result ABI is *not* generic** — the tail names
   an operand, the destination is grouped/vector/predicate, or operands
   have immediate/carrier restrictions. Add entries to the matching result
   ledger. Recipe C.
4. **The form cannot be a chain at all** (shape-dependent register groups,
   descriptor operands, one opcode spanning many ABIs — the mma/wgmma/
   tcgen05 class) — typed wrapper methods only, plus a
   `TYPED_WRAPPER_ONLY_RULES` entry so a dispatch miss fails before LLVM.
   Recipe D.

Every recipe ends the same way: run the affected suites with
`julia --project=test test/runtests.jl <names...>` and update the
test-side oracles (see [Count pins and oracles](#count-pins-and-oracles)).

## Recipe A: blessing a chain opcode (`FORMS` entry)

Adding an entry to `FORMS` in `src/forms.jl` is a review act, not a
mechanical step. For the opcode (and any mods-prefix overrides) decide,
against the PTX ISA section:

- `pure` — may the optimizer delete/CSE/reorder it? Any memory access,
  architectural state, or observable effect ⇒ `false`. A false `pure` is
  a miscompile.
- `convergent` — is it warp- or warpgroup-collective? Duplicating it
  across divergent branches is the activemask miscompile class.
- `brackets` — do pointer operands render as `[%addr]`?
- `returns` — does the dtype tail name a *result* (`_MEM`, `_PURE`) or an
  *operand* (`st`, `red`, `nanosleep` ⇒ sink forms, `returns = false`)?

Then:

- a `@testset` in `test/host/inst.jl` or the family's host file pinning
  the rendered asm/constraints via `PTX.format_call` or `lowering`;
- ptxas evidence in `test/ptxas/` (`ptxas_compiles` at the op's floor
  capability) if the op is arch-gated;
- if the op's floor is above sm_90, a `_touch_target` special case in
  `test/setup.jl` so the compile-touch sweep compiles it at the right
  target.

## Recipe B: NVVM-intrinsic-backed wrapper

Files touched, in order:

1. `src/wrappers/<family>.jl` — a typed method on the operation
   singleton. Prefer the `optype"opcode.mods"` definition spelling over a
   hand-transcribed `(::Operation{:op, (...)})` tuple. Spell every
   `nvvm"..."` literal out (no name-building loops) — the conformance
   scan greps for them. Guard registration with `NVVM.isintrinsic` when
   the method should fall back to asm on older registries.
2. `src/forms.jl` — only if the opcode itself is new (the wrapper still
   needs the family's contract for the transpiler and reflection).
3. `test/host/conformance.jl` — **mandatory**: a selection probe per
   intrinsic (`(name, argtypes, mcpu, mattr, expected instruction
   regex)`). The "every wrapper intrinsic has a selection probe" testset
   scans `src/` for `nvvm"..."` literals and fails listing any you
   missed. Generated method families additionally maintain name-list
   constants with count pins.
4. `test/host/<family>.jl` — dispatch, argument validation, and
   `lowering` classification.
5. `test/ptxas/` — instruction-text assertions on `emit_ptx` output plus
   `ptxas_compiles` at the floor target. The four
   `test/ptxas/compile_touch_*.jl` shards must cover every wrapper opcode
   exactly once — a new opcode joins its shard (or `others`).
6. `test/gpu/<family>.jl` — a runtime semantic probe behind the correct
   `# TEST_TARGET:` banner, when hardware semantics are checkable.
7. `docs/src/wrappers.md` — the family's user-facing documentation.

`src/nvvm/table.jl` is never hand-edited. If the intrinsic is missing
from the registry, the wrapper is an asm-tier wrapper (same shape, but
building `convergent_asm_ir`/`@asmcall` bodies), and the registry gains
it on the next backend bump — see `gen/README.md`.

## Recipe C: result-ledger forms

The ledgers own every chain form whose result/operand ABI the generic
dtype rule cannot recover:

| Ledger | Owns |
|---|---|
| `src/scalar_results.jl` | scalar results not named by the tail |
| `src/structured_results.jl` | grouped/multi-destination (`setp`, `lop3`, ...) |
| `src/vector_results.jl` | homogeneous tuple results (`ld.vN`, ...) |
| `src/b128_forms.jl` | the 128-bit register carrier grammar |
| `src/mbarrier_forms.jl` | the closed mbarrier grammar |
| `src/cvt_forms.jl` | ordinary `cvt` source carriers |
| `src/immediate_forms.jl` | instruction-specific immediate domains |
| `src/address_operands.jl` | address-role markers and deny rules |

**Adding entries to an existing ledger** touches the ledger file, its
load-time count assertion (where present), and the test-side oracle +
pins in the matching `test/host/<ledger>.jl` file. The oracles are
deliberately independent reconstructions of the grammar — update the
oracle's generator to *derive* the new expectation; never make it read
the source ledger.

**Adding a new ledger** is a structural change. The consultation sequence
currently lives in four places that must all agree — `build_call` and
`lowering` in `src/inst.jl`, `infer_rettype`/`_result_abi_error` in
`src/types.jl`, and the per-ledger adapters in
`src/codegen/instruction.jl` — plus the include order in `src/PTX.jl`.
Budget for all of them, and add the mirrored `lowering` guard so
reflection and reality cannot drift.

## Recipe D: typed-wrapper-only families

For forms the chain cannot express, add the wrapper methods (Recipe B
shape) and close the boundary:

- a rule in `TYPED_WRAPPER_ONLY_RULES` (`src/forms.jl`) so an uncovered
  spelling errors at compile time instead of receiving a guessed
  contract — its count and content are pinned in
  `test/host/fallback_boundary.jl`;
- if operands accept integer addresses, forwarding adapters and their
  pins (see `test/host/address_roles.jl` for the tcgen05 pattern);
- combinatorial modifier grids should be generated (`@eval` over a
  declarative spec, like the tcgen05 ld/st grid), with the spec — not
  the expansion — as the reviewed artifact.

## Count pins and oracles

Policy: **double-entry, not triple-entry.** Each form inventory exists
exactly twice — once in `src/` (the ledger or wrapper registration) and
once in `test/` (an independent oracle, usually a small generator that
replays the grammar). Count pins make drift loud; when one fails, the
message prints expected/got. When adding forms:

1. update the test-side oracle generator first, from the ISA — not from
   what `src/` now produces;
2. run the suite; every failing pin is a place the change is visible;
3. update pins to the oracle-derived numbers only after confirming the
   set-equality assertions (the stronger form) pass.

## Golden tests

Anything that changes emitted PTX for the pinned kernels shows up in
`test/ptxas/golden.jl`. Regenerate with
`PTX_UPDATE_GOLDEN=1 julia --project=test test/runtests.jl ptxas/golden`
and review the `test/golden/*.ptx` git diff — the diff **is** the review
artifact. A missing golden is a failure by design; never make absence
regenerate silently.
