# Design direction

This is a working document, not a contract. It records where the package is
headed and why, so decisions don't have to be re-derived from scratch. Anything
here can be revised when reality disagrees with it. Sections are ordered by how
settled they are: verified facts first, direction second, open questions last.

## What changed upstream

CUDA.jl v6.2 decoupled GPU machine-code generation from Julia's in-process
LLVM. GPUCompiler now writes the optimized module to bitcode and shells out to
an external `llc` shipped by `NVPTX_LLVM_Backend_jll` (LLVM 22 at the time of
writing). Julia's own LLVM only runs the middle end.

Consequences, verified against the shipped artifact (`llc` 22.1.7):

- Native targets through `sm_121a` and PTX ISA up to 9.0, regardless of Julia
  version. No more sm_90-plus-header-rewrite on Blackwell. Arch-specific
  (`a`-suffix) instructions assemble against a genuine target.
- Device code can call any intrinsic the *external* backend knows, even ones
  the in-process LLVM has never heard of: unknown `llvm.*` declarations pass
  through bitcode untouched and are lowered by `llc`. Confirmed end-to-end
  (one intrinsic call → exactly one PTX instruction).
- The external backend carries ~2,600 `llvm.nvvm.*` intrinsics, including the
  families this package wraps by hand today: scoped mbarrier, TMA
  (`cp.async.bulk.tensor.*`), the full tcgen05 verb set, ldmatrix/stmatrix
  (incl. m16n16 b8 shapes), FP8/FP6/FP4 cvt — and notably the block-scaled
  `mma.sync` variants (mxf8f6f4 / mxf4 / mxf4nvf4), i.e. the consumer-Blackwell
  microscaling path.
- `wgmma.mma_async` was never upstreamed and likely never will be (only
  fence/commit_group/wait_group exist). Hopper warpgroup MMA stays asm-only.

## What this package is, restated

The pitch used to be "inline asm access to PTX" — access to the ISA *around*
the NVPTX backend, whose in-process version was too old to know the modern
instruction families. With the backend now current and externally versioned,
routing around it is no longer the point; routing *through* it is. The pitch
now:

> PTX-vocabulary notation for GPU instruction atoms, lowered through whichever
> path produces the best code — plus the layer above instructions (descriptors,
> layouts, barriers, pipelines) that no intrinsic set provides.

Two observations anchor this:

- **PTX ISA names are more stable than `llvm.nvvm` names.** NVIDIA deprecates
  ISA vocabulary over many years; LLVM renames intrinsics between majors
  (`barrier0`, `ldg`, and `rotate` are all gone from LLVM 22). So the
  PTX-shaped surface (`ptx"..."`) is the public contract, and intrinsic names
  are an implementation detail we regenerate per backend major.
- **Inline asm is a string tunnel; intrinsics are a compiler.** Asm dictates
  exact text and blinds the optimizer (and carries hand-asserted correctness
  attributes — the `active_mask` class of miscompile). Intrinsics let the
  backend select forms, allocate registers freely, and carry convergence and
  memory attributes by construction. Where an intrinsic exists, it should win.

## Lowering tiers

Every operation lowers through one of three paths, chosen per form, recorded
in the registry (below):

1. **Core LLVM IR** where the backend pattern-matches plain IR: atomics
   (`atomicrmw`), fences with syncscopes, vector loads/stores, standard
   rounding `cvt`. Preferred when available — it's the only tier the
   in-process middle end can actually optimize through. Caution: this is a
   *semantic* translation, not a renaming. Mapping PTX memory orderings and
   scopes onto LLVM orderings/syncscopes is the most correctness-sensitive
   piece of the whole design, and each such mapping deserves an explicit
   entry (and a golden-output test) rather than an assumed equivalence.
2. **NVVM intrinsics** for the instruction-mirror families. Emitted as
   `Base.llvmcall` IR with explicit attribute groups (`convergent`, memory
   effects) on the declaration, since the in-process LLVM doesn't know these
   intrinsics and would otherwise treat them as unattributed unknown calls.
   The explicit attributes also buy uniformity, not just correctness: when a
   given Julia version's in-process LLVM *does* know an intrinsic (1.12 knows
   names 1.10 has never heard of), it supplies attributes from its own table —
   registry-attached attributes make optimization behavior identical across
   Julia versions instead of varying with each version's intrinsic knowledge.
   Aggregate returns unpack to tuples here, replacing long asm constraint
   strings.
3. **Inline asm** for what's left: `wgmma.mma_async`, forms newer than the
   pinned backend (PTX > 9.0 until an LLVM 23 JLL), and anything not yet
   upstreamed. The existing `ptx"..."` machinery, unchanged in mechanism,
   shrunk in scope.

The mapping between PTX modifier chains and intrinsics is *not* a generic
string rewrite. Within a family it's regular (tablegen multiclass products),
but across families the conventions differ: modifiers may become name segments
(in a renamed vocabulary), immediate flag operands (one intrinsic covering a
modifier sub-lattice, e.g. TMA multicast/cache-hint), pointer address spaces,
or return types. Hence: one hand-authored descriptor per family, forms within
a family enumerated programmatically.

## The registry

The single source of truth for what's valid, what it requires, and how it
lowers. It is the answer to three problems at once — discoverability, coverage,
and the silent-fallthrough hazard — so it's a first-class queryable API, not a
build artifact.

- **Machine-generated, not hand-maintained.** A dev-time script extracts the
  intrinsic table (names, signatures, attributes) from `IntrinsicsNVVM.td` at
  the llvm-project tag matching the pinned JLL — via `llvm-tblgen
  --dump-json`, not by parsing TableGen text (multiclass expansion makes raw
  parsing a trap). Hand-written content is limited to the per-family
  descriptors encoding the PTX↔NVVM translation and operand conventions.
- **Verified by trial compilation.** CI compiles a probe for every registry
  entry through the artifact `llc` and asserts the *expected instruction*
  appears in the emitted PTX — acceptance alone is not verification, since a
  signature-valid probe can still legalize to an unintended expansion on a
  given target. The verifier's error messages are precise enough to pinpoint
  signature drift; a JLL major bump produces a mechanical diff of
  renames/removals instead of user-reported breakage.
- **Queryable.** Enumerate valid forms of a family, look up a chain's
  requirements (sm, PTX ISA, lowering tier), validate before compile.
  Reference documentation for the instruction surface is generated from the
  registry, so the package documents itself rather than deferring to the ISA
  PDF.
- **A blessing boundary** (landed 2026-07-06, src/forms.jl). Registered
  chains get the verified lowering with correct attributes — including
  `convergent nomerge` for the collective families, which the old chain
  default could not attach. Unregistered chains error at chain build time;
  raw synthesis is the explicit `ptx"..."raw` opt-in, whose contract is
  maximally conservative (sideeffect + memory clobber + convergent, pointer
  operands bracketed). A deliberate breaking change to the old `ptx"..."`
  (which accepted any chain via the generic fallback), shipped pre-1.0; the
  error message carries the exact migration path. The registry also powers
  the property notation: `ptx"cvt".rn.f32.f16` composes in the type domain,
  and `propertynames` suggests valid continuations (REPL tab completion as
  ISA explorer). Suggestion sources are ISA-spelled by construction:
  registry override prefixes plus the wrapped surface enumerated from the
  method table (every wrapper form is an `Operation{op, mods}` method whose
  mods tuple is the ISA chain). NVVM intrinsic names were rejected as a
  source (2026-07-06): their grammar diverges from the ISA chain exactly
  where a family is irregular — `llvm.nvvm.mma.*` drops `.sync.aligned`
  and leads with the shape, so name-derived completion suggested segments
  invalid at that position while omitting the only valid one. Unwrapped
  pure chains (`add`, plain-rounding `cvt`) deliberately suggest nothing —
  see the rejected full-grammar note below.
- **Full-grammar enumeration: considered and rejected** (2026-07-06). The
  tempting completion of the story — per-family modifier grammars in the
  registry, closed validation at chain build, full-ISA tab completion —
  was piloted on cvt with ptxas as the grammar oracle (enumerate a
  candidate superset, trial-compile each across an arch ladder, commit
  the accepted set as generated source). The pilot's first run killed the
  idea honestly: the
  oracle only answers correctly if the per-form operand synthesis
  (register classes, arity, sub-word carriers) is right, and THAT
  knowledge is hand-written and rot-prone — the oracle doesn't eliminate
  transcription, it relocates it. Enforcement would convert an
  already-loud ptxas error into an earlier loud error at the cost of a
  second registry with its own maintenance treadmill (mandatory-vs-
  optional flags, per-family modifier order, arch-specific surfaces, ISA
  releases). The line that stands instead: enumeration is sound exactly
  where wrapping forces it anyway (wrapped families get exact completion
  as a side effect of being load-bearing, cross-checked by conformance
  and exercised by tests); everything else stays open on the
  trailing-modifier rule with ptxas as the late-but-loud gate. The
  registry stays a *semantics* table — the part no oracle can answer.

The method table is *not* the registry. Dispatch is already fully resolved by
the `Operation{op, mods}` singleton; enumerating the product space as eager
method definitions would add load time and pkgimage weight without changing
the compiled code or saving kernel-compile latency (device code is cached by
GPUCompiler, not pkgimages). One generated entry point consulting the registry
produces identical kernels with better errors.

## Layers, from the user's point of view

- **Curated wrappers** (mbarrier verbs, TMA, mma/tcgen05 families) and the
  structures above them (descriptors, swizzle/scale layouts, `TensorMap`,
  `MBarriers`/`Pipelines`): the human API. Finite, docstring-ed, browsable by
  normal Julia reflection. This is the durable core of the package — the part
  no intrinsic set replaces.
- **`ptx"..."`**: the notation. Stable vocabulary, compile-time validated
  against the registry, lowered through whichever tier the registry says. Note
  this is deliberately *not* WYSIWYG: the backend may emit an equivalent
  renamed/implicit-modifier form of what was written.
- **`ptx"..."raw`** (suffix flag, not a second macro): the escape hatch for
  day-0 ISA features and unregistered chains — exact instruction text, naive
  constraint synthesis, *maximally pessimistic* effect defaults (sideeffect,
  treated as convergent): slow before wrong. Raw means raw in instruction
  choice only — the string still rides through the backend, the ISA-version
  header, and ptxas. Greppable by design. Distinct from the curated asm tier
  (`wgmma`), which is registered, hand-constrained, and convergent-attributed
  where the instruction is warp-collective (CONCERNS.md, "Convergence on the
  asm tier").
- **`nvvm"..."`** (internal, exposed for power users with a pinned-to-backend
  caveat): direct intrinsic calls in the `llvm.nvvm.*` vocabulary, with
  registry-checked names and signatures, attribute-correct llvmcall synthesis,
  aggregate unpacking. The plumbing tier 2 stands on, not a user interface;
  its one user-facing niche is intrinsics present in the backend table whose
  family has no PTX-vocabulary descriptor yet. Stable only within a backend
  JLL major, by explicit policy.

Vocabularies never mix: `ptx""` speaks PTX ISA, `nvvm""` speaks the intrinsic
namespace, and no macro is named after the backend itself — the around/through
distinction is carried by the `raw` flag, not by a naming scheme that needs a
history lesson.

## Compatibility posture

Requires CUDA.jl ≥ 6.2 (external backend + intrinsic passthrough). The package
tracks one `NVPTX_LLVM_Backend_jll` major at a time — as a *direct* dependency
whose compat bound names the major the registry was generated from. The
resolver then enforces registry/backend consistency: environments that
instantiate are environments where the intrinsic names match. A backend bump
is one PR (regenerate from the `.td` at the new tag, let the trial-compile
harness report the churn, move the bound); until that lands, a CUDA.jl release
requiring a newer backend conflicts loudly at resolve time rather than failing
at kernel-compile time. Single-major pinning makes that PR a serialization
point for every dependent when CUDA.jl moves to a new backend major, so the
regen must stay a rehearsed, sub-day, mechanical operation — periodically
dry-run the regeneration against the *current* JLL (expected result: empty
diff) to prove the pipeline is still turnkey before a real bump demands it. Note the failure modes are benign either way: a
renamed intrinsic either gets AutoUpgraded by `llc` or fails instruction
selection with a clear error — never a silent miscompile.

Supporting multiple backend majors simultaneously is explicitly avoided — that
was the old in-process world's complexity, and the JLL exists so nobody has to
do that again. (A transitional `"22, 23"`-style bound with per-major registries
is possible if a migration ever demands it, but it's a tool of last resort.)

Julia's in-process LLVM remains a separate, independently-moving version: it
governs the middle end — which core-IR lowerings tier 1 can use and which
attributes are understood — and is tracked the way Julia compat always is.
Intrinsic-name churn is exclusively a backend-JLL concern.

## Approach

Decided posture for getting there, at the altitude of strategy rather than
schedule. Concerns gating each step live in `CONCERNS.md`.

**Spikes before structure.** Two short experiments run before any registry
code exists, as throwaway work whose only deliverable is a resolved entry in
`CONCERNS.md`: the convergence/attribute-survival test (the one result that
can change the architecture), and the aggregate-return llvmcall check. The
tblgen extraction toolchain gets scripted at the same time, since the
generator is blocked without it.

**In-place migration, not a parallel rewrite.** The package keeps working
throughout. Registry and `nvvm""` infrastructure land alongside the existing
wrappers; families migrate one at a time, each behind the same public
surface. Before the first migration, a golden-PTX harness locks the current
emitted code for the test kernels, so every migration is reviewed as a diff
of generated PTX rather than an act of faith. Comparison is structural, not
byte-exact — the backend renumbers registers the moment lowering goes through
instruction selection — so the harness parses both sides with the package's
own parser/IR stack and compares instruction sequences modulo register
naming. This, plus probe assertions in the conformance harness, is the
parser's load-bearing role in the rework; it stops being justified by the
transpiler.

**Family coverage comes in two flavors.** Most families are spelled as
hand-written `nvvm"..."` literals — one method per form, or a small
script-expanded set of literal methods where a grid is regular (tcgen05's
ld/st). The conformance harness scans `src/` for those literals and requires
a selection probe for each, so the names are greppable by construction and an
unprobed used intrinsic fails the suite. A few families have a dtype
cross-product too large to spell literally — `mma`'s `kind::f8f6f4` alone is
a hundred structurally-identical forms — and stay table-driven generators
that emit `IntrinsicCall` directly, invisible to the literal scan. Those are
covered instead by a registry-completeness assertion over the generator's
emitted name list plus one selection probe per structural class: strictly
stronger than per-form probing for a mechanical family, since it also catches
a name the generator *could* emit that the registry no longer has. Both
flavors converge on one guarantee — a JLL bump that renames or drops an
intrinsic a wrapper stands on surfaces as a red test naming the family, never
a silent miscompile. (The per-registry-entry trial-compile matrix in "The
registry" above is the broader, still-pending verifier; this is the
per-family standing defense that ships with each migration.) This settled the
"do families need a descriptor mechanism?" open question in the negative: the
generators carry the PTX↔NVVM translation in code, and conformance carries
the coverage guarantee — no separate descriptor object earned its keep.

**Family order by information per effort.** `shfl` first — the simplest
convergent family, so it doubles as the convergence guinea pig in real use.
`mbarrier` second — the largest wrapper count, one form already verified
end-to-end. Then TMA, tcgen05, and the mma families (where the payoff is —
block-scaled `mma.sync` replaces the largest asm surface). Then the fences:
this split on contact — the *proxy/init* fences (`fence.proxy.*`,
`fence.mbarrier_init.*`) name a memory proxy a core-IR `fence` cannot
express, so they are tier-2 intrinsics and migrated cleanly; the generic
memory fences (`fence.sc.*`, `fence.acq_rel.*`) are the true tier-1 case
(core-IR `fence` with an ordering and a syncscope — a semantic translation),
migrated 2026-07-02 with the mapping pinned in CONCERNS.md. The tier-1
candidates migrate as their semantic mapping decisions get made, not
before — vector ld/st traded a `~{memory}` barrier for an optimizable load,
a behavior change to working code that was verified explicitly before
landing (migrated 2026-06-13; see the CONCERNS.md ledger entry); `cvt`
resolved by dissolution 2026-07-06 (see Open questions — the collision set
is nearly empty once expressibility decides per form). The CTA barriers
(`bar.*`, `barrier.*`) migrated 2026-07-06, closing the last family that
had intrinsics available while sitting on asm — the asm tier is now
residue-only. `wgmma` never
migrates; it gets registered as asm-tier and stays there. Beyond that opening sequence, priority comes from
data, not completionism: parse a corpus of real kernels (own packages,
CUTLASS dumps) and rank unregistered forms by frequency.

**Registry as committed generated source.** The generator's output is Julia
source checked into the repo, not a build-time artifact: a JLL bump becomes a
reviewable PR diff, which is exactly the churn-inspection mechanism the design
wants anyway.

**The blessing flip landed 2026-07-06** — sequenced as ratified: after the
family migrations (shfl, mbarrier, TMA, tcgen05, mma, vec_ldst, fences,
ldmatrix/stmatrix) and with the registry seeded from a harvest of every
chain the package and suite surface use. The harness proved what it breaks:
nothing — full suite green on 1.10/1.11/1.12 across the flip, goldens
byte-identical.

## Non-goals

- Wrapping the entire PTX ISA. Coverage follows need; the registry makes
  gaps visible and cheap to fill, which is different from filling them.
- Competing with CUDA.jl's general-purpose programming model. This package is
  the instruction-atom and kernel-structure layer underneath things like
  microscaled GEMMs.
- Growing the transpiler. `ptx_to_julia` is frozen: it emits through the
  notation layer and inherits whatever the notation lowers to, so it stays
  correct for free but gets no investment from the rework. Kept because it's
  tested, decoupled, and occasionally the right tool for porting a reference
  kernel; if a real porting campaign ever needs it modernized, that campaign
  pays for it.
- API stability for `nvvm"..."` names across backend majors. That instability
  is upstream's; we surface it honestly instead of absorbing it.

## Open questions

- Provenance of per-form requirements (minimum sm, PTX ISA): the `.td` yields
  names, signatures, and attributes, but target predicates live on instruction-
  selection patterns, not intrinsic definitions. Likely answer: derive
  empirically by trial-compiling each entry across a matrix of `-mcpu`/`-mattr`
  targets (the harness exists anyway); hand-curate only where that's ambiguous.
- ~~Whether the family descriptors can be partially derived from the ISA
  grammar rather than written by hand.~~ Resolved, in the negative, by the
  migrations themselves (see "Family coverage comes in two flavors"): there
  are no descriptor *objects* to derive. The PTX↔NVVM translation lives in the
  per-family generator code, and the coverage guarantee lives in conformance;
  no queryable descriptor earned its keep across the migrated families.
- ~~Convergence attributes in the middle end.~~ Resolved — the spike ran on
  the GB10 and validated it (explicit attribute groups survive the pipeline
  and bind; the unattributed variant miscompiles exactly as predicted). See
  `CONCERNS.md`, "Convergence attributes through the middle end."
- ~~Whether `cvt`-style ops that have both core-IR and intrinsic lowerings
  should prefer optimizability (core IR) or exactness of form (intrinsic).~~
  Resolved (2026-07-06), largely by dissolution: decide per-form by
  *expressibility*, and the set where both lowerings genuinely exist is
  nearly empty. The cvt forms where PTX.jl adds value — satfinite, packed,
  fp8/fp6/fp4 narrow types — have no core-IR spelling at all (there is no
  `fptrunc` to fp8), so they are tier-2 by the same rule that decided proxy
  fences. Non-default rounding modes (`.rz`/`.rm`/`.rp`) also lack a plain
  core-IR spelling (`fptrunc` is round-to-nearest-even only), leaving only
  the default-rounding standard-type forms as a true collision set — and
  those are conversions Julia already emits natively without PTX.jl in the
  loop, so registering them is completeness, not value. Prior data points
  that established the rule: proxy fences took the intrinsic (no core-IR
  form), vector ld/st (2026-06-13) and generic memory fences (2026-07-02)
  took core IR (verified before landing; see the CONCERNS.md ledger
  entries). Standing caution for any cvt form that does take core IR:
  value-semantic identity must be pinned from the consumer side
  (double-rounding through widening chains, FTZ) — an exhaustive-sweep
  parity harness in the consumer package is the right instrument.
