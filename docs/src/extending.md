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

Whatever the recipe, adding coverage also means updating `SURFACE.toml`
(repo root) — the machine-checked inventory that assigns every ISA
§9.7.x family an explicit disposition (`strict`/`generic`/`raw-only`/
`out-of-scope`). A new `FORMS` opcode, typed-wrapper-only rule, or
wrapper family that the inventory doesn't own fails
`test/host/surface.jl` by name; the [Coverage](coverage.md) page is
generated from the same file at docs-build time, so the inventory and
the documentation cannot drift apart.

## Recipe A: blessing a chain opcode (`FORMS` entry)

Adding an entry to `FORMS` in `src/ledgers/forms.jl` is a review act, not a
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
   singleton. A compile-time literal spelling uses the
   `optype"opcode.mods"` definition macro (never a hand-transcribed
   `(::Operation{:op, (...)})` tuple — the string form is dispatchable by
   the `ptx""` call spelling by construction); only generator loops that
   build the mods tuple programmatically define methods on the
   `Operation{op, mods}` singleton directly. Spell every `nvvm"..."`
   literal out (no name-building loops) — the conformance scan greps for
   them. A *generated* family instead routes its bookkeeping through the
   wrapper registry (`src/wrappers/registry.jl`): tier-2 generators call
   `wrapper_intrinsic_call(family, op, mods, name)` (which validates the
   built name against the pinned NVVM registry, records a `WrapperRecord`,
   and returns the `IntrinsicCall`; guard with `NVVM.isintrinsic` first
   when the method should fall back to asm on older registries), and
   asm/core-IR generators call `register_wrapper!(family, op, mods,
   tier)`. Registration is idempotent behind one shared key set — never
   add a per-family accumulator list.
2. `src/ledgers/forms.jl` — only if the opcode itself is new (the wrapper still
   needs the family's contract for the transpiler and reflection).
3. `test/host/conformance.jl` — **mandatory**: a selection probe per
   intrinsic (`(name, argtypes, mcpu, mattr, expected instruction
   regex)`). The "every wrapper intrinsic has a selection probe" testset
   scans `src/` for `nvvm"..."` literals and fails listing any you
   missed. Generated method families are instead pinned through the
   wrapper registry: the sweep replays the registration loops and asserts
   set equality plus a count pin against
   `PTX.wrapper_intrinsic_names(:family)`.
4. `test/host/<family>.jl` — dispatch, argument validation, and
   `lowering` classification.
5. `test/ptxas/` — instruction-text assertions on `emit_ptx` output plus
   `ptxas_compiles` at the floor target. The four
   `test/ptxas/compile_touch_*.jl` shards must cover every wrapper opcode
   exactly once — a new opcode joins its shard (or `others`).
6. `test/gpu/<family>.jl` — a runtime semantic probe behind the correct
   `# TEST_TARGET:` banner, when hardware semantics are checkable.
7. `docs/src/wrappers.md` — the family's user-facing documentation.

`src/nvvm/table.jl` is never hand-edited — a standing test
(`test/host/registry_generation.jl`) byte-compares it against
regeneration from the committed JSON snapshot in `gen/`. If the
intrinsic is missing from the registry, the wrapper is an asm-tier
wrapper (same shape, but building `convergent_asm_ir`/`@asmcall`
bodies), and the registry gains it on the next backend bump — see
`gen/README.md`.

## Recipe C: result-ledger forms

The ledgers own every chain form whose result/operand ABI the generic
dtype rule cannot recover:

Every ledger implements the shared protocol in
`src/ledgers/protocol.jl` (`claims`/`schema`/`miss`/
`validate_ledger_args` on a singleton handle), and every consumer walks
the single `CALL_LEDGERS` consultation order defined there.

| Ledger | Owns |
|---|---|
| `src/ledgers/scalar_results.jl` | scalar results not named by the tail |
| `src/ledgers/structured_results.jl` | grouped/multi-destination and predicate-result queries (`setp`, `lop3`, `testp`, `isspacep`, ...) |
| `src/ledgers/vector_results.jl` | homogeneous tuple results (`ld.vN`, ...) |
| `src/ledgers/b128_forms.jl` | the 128-bit register carrier grammar |
| `src/ledgers/mbarrier_forms.jl` | the closed mbarrier grammar |
| `src/ledgers/cvt_forms.jl` | ordinary `cvt` source carriers |
| `src/ledgers/immediate_forms.jl` | instruction-specific immediate domains |
| `src/ledgers/address_operands.jl` | address-role markers and deny rules |

**Adding entries to an existing ledger** touches the ledger file, its
load-time count assertion (where present), and the test-side oracle +
pins in the matching `test/host/<ledger>.jl` file. The oracles are
deliberately independent reconstructions of the grammar — update the
oracle's generator to *derive* the new expectation; never make it read
the source ledger.

**Adding a new ledger** is a structural change, but the consultation
sequence lives in exactly ONE place: `CALL_LEDGERS` in
`src/ledgers/protocol.jl`. A new ledger is:

1. a singleton handle `struct MyLedger <: FormLedger end` in
   `protocol.jl`, added to `CALL_LEDGERS` at its reviewed position in
   the consultation order (this single edit orders it for `build_call`,
   `lowering`, result-ABI inference, and the transpiler simultaneously);
2. the protocol methods in the new `src/ledgers/<my_ledger>.jl` file —
   `claims(::MyLedger, op, mods)`, `schema(::MyLedger, op, mods)`,
   `miss(::MyLedger, op, mods)`, and
   `validate_ledger_args(::MyLedger, s, argtypes)` (plus
   `prevalidates_call(::MyLedger) = true` if arguments must be checked
   at consult time rather than by a builder);
3. the four consumer dispatch methods:
   `build_ledger_call(::MyLedger, s, argtypes, contract)` in
   `src/dsl/render.jl` (omit it if the form lowers through `build_call`'s
   generic tail), `ledger_rettype`/`ledger_result_abi_error` methods in
   `src/ledgers/types.jl` (or the transparent `missing` stubs if the
   ledger never feeds generic scalar inference), and
   `transpile_ledger!(::MyLedger, cg, inst)` with its
   `_instruction_*`-style adapter in `src/codegen/adapters/`;
4. the include order in `src/PTX.jl` (ledger files load after
   `protocol.jl`).

`lowering` needs no per-ledger work: it walks `CALL_LEDGERS` through the
shared `lowering_entry`, so reflection and reality cannot drift. The
transpiler consults the same order twice: the module *preflight*
(`_validate_exact_schema!` in `src/codegen/contract.jl`) walks it with
the scalar boundary check deferred to the tail (`mov.b128`-class forms
must reach the b128 consult first), and emission (`emit_instruction!`)
walks it shielded by that preflight.

## Recipe D: typed-wrapper-only families

For forms the chain cannot express, add the wrapper methods (Recipe B
shape) and close the boundary:

- a rule in `TYPED_WRAPPER_ONLY_RULES` (`src/ledgers/forms.jl`) so an uncovered
  spelling errors at compile time instead of receiving a guessed
  contract — its count and content are pinned in
  `test/host/fallback_boundary.jl`;
- if operands accept integer addresses, forwarding adapters emitted by
  the *same* enumeration that emits (or, for literal methods, sits next
  to) the primary methods — the tcgen05 registration calls
  `_tcgen05_adapter!(mods, argtypes...)` with the exact reviewed
  signature, and the sealed `TCGEN05_INTEGER_ADDRESS_ADAPTERS` inventory
  is pinned by the independent oracle in `test/host/address_roles.jl`;
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
