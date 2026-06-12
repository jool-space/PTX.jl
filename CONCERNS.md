# Concerns to address

Companion to `DESIGN.md`. Each entry says what the concern is, why it matters,
and what resolves it. Status lines get updated as things land; resolved
entries stay (with their resolution) rather than being deleted, so the record
shows what was checked and how.

Grouped by weight: gating concerns can change the architecture; verification
concerns are expected to pass but must be demonstrated; diligence concerns are
per-op care that can't be globally discharged; process concerns are ecosystem
plumbing.

## Gating — resolve before building on tier 2

### Convergence attributes through the middle end

**Status: RESOLVED 2026-06-12 — validated. `spikes/convergence.jl`, run on the
GB10 (Julia 1.12.6 / in-process LLVM 18 / backend llc 22.1.7).**

Subject: `llvm.nvvm.activemask` (unknown in-process, known-convergent to the
backend), identical calls leading both arms of a divergent branch. With
`convergent` on the llvmcall declaration: attribute present in the optimized
module, both call sites preserved, two `activemask.b32` in the PTX, hardware
masks correct (`0000ffff` / `ffff0000`). Without it: the optimizer hoisted
the calls into one pre-branch site and both arms read `ffffffff` on hardware —
the predicted miscompile, confirming the test has teeth. En passant this also
validated the full tier-2 passthrough on the real pipeline: an intrinsic the
in-process LLVM has never heard of, declared via llvmcall with attributes,
optimized, shipped through bitcode, lowered by the external llc, and executed
correctly on the device.

Original concern, for the record:

Tier-2 calls target intrinsics the in-process LLVM doesn't know, so they get
no attributes from its intrinsic table. Absence of `convergent` is permission,
not caution: block duplication (jump threading, tail duplication) is legal for
unknown side-effecting calls but splits the call site, which breaks
warp-synchronous semantics. The planned defense is `Base.llvmcall` IR with
explicit attribute groups on the declaration. Unverified links: do the
attributes survive Julia's llvmcall splicing; do they survive GPUCompiler's
linking into the final optimized module; does Julia's own SSA-IR optimizer
(which has no convergence concept) stay out of trouble.

Resolved by: an IR-level test (attributed unknown intrinsic in
duplication-tempting control flow; assert post-`-O2` attribute presence,
single call site, no motion across the divergent branch — and confirm the
*unattributed* variant actually misbehaves, so the test has teeth) plus a
hardware test (the `activemask` divergence shape from CUDA.jl `2e134a314`,
routed through an unknown-intrinsic declaration, masks checked on the GB10).

If it fails: attribute responsibility moves from per-call-site IR strings to a
module pass over `llvm.nvvm.*` declarations, which means a GPUCompiler hook or
upstream change — that conversation would then precede the registry work.

### Attribute-group merging across call sites

**Status: mostly resolved; one corner deliberately untested.**

The convergence spike exercised the common case: two llvmcall expansions of
the same snippet, one merged declaration, attributes intact. The untested
corner is *conflicting* attribute groups for the same intrinsic from
different snippets — but the generator emits identical attributes for a given
intrinsic by construction, so this can only arise from hand-written llvmcall
alongside generated code. Worth one assertion in the test suite eventually;
not load-bearing.

## Verification — expected to pass, must be shown

### Aggregate returns through llvmcall

**Status: RESOLVED 2026-06-12 — validated. `spikes/aggregate_return.jl`, GB10.**

`ldmatrix.sync.aligned.m8n8.x4.b16` ({i32,i32,i32,i32} → `[4 x i32]` →
`NTuple{4,UInt32}` via extractvalue/insertvalue): the instruction selected
with four destination registers, and the fragment values matched the
documented m8n8 layout for all 32 threads × 4 matrices on hardware. The wide
(`x32`) `tcgen05.ld` shapes are the same pattern scaled up and need sm_100a,
which this machine lacks; trial-compile coverage in CI handles those.

### Pessimistic attributes on raw-asm call sites

**Status: RESOLVED 2026-06-12 — validated. `spikes/raw_asm_attrs.jl`, GB10.**

Raw emission goes through llvmcall IR sharing tier 2's attribute machinery:
`call i32 asm sideeffect "...", "=r"() #0` with `#0 = { convergent nounwind }`
parses, survives the optimized module with both call sites and the attribute
group intact, and produces correct divergent masks on hardware. `@asmcall` is
not needed for the raw tier. (The macro mechanics are not a concern: Julia
passes suffix flags as a second argument to the string macro. Note `sideeffect`
alone already blocks the both-arms hoist shape — convergent's marginal value
on raw asm is the duplication direction, per the convergence spike's logic.)

### Mangling of overloaded intrinsics

**Status: RESOLVED 2026-06-12 — scheme pinned against the 22.1.7 `llc`.**

306 of 2569 records are overloaded (signatures containing `llvm_any*` types):
228 legacy `wmma` (not a migration target), 22 `atomic` (tier-1 core IR), and
relevantly `ldmatrix` (18), `stmatrix` (9), `tensormap` (11) — overloaded on
pointer address space. The canonical scheme, confirmed via `-print-after`
(parse-time IR, after the auto-upgrader has normalized names): one suffix per
overload slot, in slot order — AnyTok appearance across (ret..., params...),
the same numbering the registry uses — spelled `pN` for a pointer in
address space N and by VT name otherwise (`f32`, `i64`, `v4i32`).
`ldmatrix...b16.p3` and `atomic.add.gen.f.cta.f32.p0` (ret slot first) both
select the expected instruction.

A finding with teeth: `llc` *accepts and silently remangles* wrong spellings —
the swapped `.p0.f32` parsed fine and was rewritten to `.f32.p0`, and the
aggregate spike's fully *unmangled* ldmatrix also worked. So acceptance
testing cannot catch mangling bugs on our side; the synthesizer emits
canonical names by construction and the conformance harness should compare
post-parse IR, not rely on llc rejecting mistakes. (The leniency itself is
upgrader behavior that may tighten at a major bump — another reason to be
canonical now.)

### Weakdep compat as the backend pin

**Status: open, low risk.**

The plan is `NVPTX_LLVM_Backend_jll` as a *weak* dependency with compat naming
the registry's generation major — host-side use of PTX.jl (parsing, layouts,
registry queries) shouldn't force a ~100 MB artifact download, and weakdep
compat still constrains any environment that contains the JLL (i.e. every
CUDA 6.2 environment). Verify Pkg actually enforces weakdep compat in the
resolution paths that matter; if it doesn't, fall back to a hard dependency.

## Diligence — per-op care, never globally discharged

### Tier-1 semantic mappings

PTX orderings/scopes → LLVM orderings/syncscopes is a semantic translation
with miscompile potential (see DESIGN.md, lowering tiers). Current exposure is
small — the package wraps no atomics; fences and vector ld/st are the tier-1
surface today — but every tier-1 entry needs an explicit mapping decision and
a golden-output test. No batch sign-off.

### Performance parity per migrated family

Intrinsic lowering frees the backend to pick instruction forms and registers —
usually a win, occasionally a regression vs. hand-tuned asm (register
pressure, form selection). Each family migration gets a golden-PTX diff, and
the MXFP8 GEMM numbers serve as the end-to-end benchmark baseline. Regressions
aren't necessarily blockers, but they must be *seen*, not discovered later.

Per-family ledger:
- **shfl** (2026-06-12, `test/golden/shfl@sm75.ptx` diff): strict win. Inline
  asm's `r` constraints forced every immediate into a register; ISel folds
  them into the instruction (`shfl.sync.idx.b32 %r1, %r0, %r0, 31, -1` —
  four `mov.b32` eliminated in the 8-form golden kernel). Same shfl
  sequence otherwise; pred forms still lower the i1 via `selp`.
- **mbarrier** (2026-06-12, `test/golden/mbarrier@sm{80,90}.ptx` diffs):
  strict win plus one renamed spelling. The asm tier's `r` pointer
  constraint forced `mov.b64` + `cvt.u32.u64` address materialization;
  ISel addresses the shared variable symbolically (`[var0]`) — both
  instructions gone. Standalone expect_tx now emits
  `mbarrier.expect_tx.relaxed.cta.shared.b64`: `.relaxed` is its only
  legal sem and `.cta` the default scope, i.e. the explicit spelling of
  the identical operation (notation is non-WYSIWYG by design). Cap-floor
  note: the scoped count-form `arrive.scope.cta.space.cta` cannot ISel at
  sm_80 — requirements live in ISel predicates, so the sm_80 forms ride
  the legacy `*.shared` intrinsics. Cluster-space sink forms stayed
  asm-tier pending AS-7 (`shared::cluster`) pointer modeling.

## Process — plumbing and ecosystem

### Obtaining llvm-tblgen at the backend's version

**Status: RESOLVED 2026-06-12 — `gen/extract_intrinsics.sh`, run end-to-end.
Conformance strengthened the same day: all 2569 *derived names* (default
rule + the 353 LLVMName overrides) now diff exactly against the name table
in the 22.1.7 llc binary, not just the count. The extraction also resolves
tblgen's anonymous records — per-argument ImmArg positions, value ranges
(`Range`), .td operand names (`ArgInfo`), and pointer address spaces — so
the registry knows which operands must be immediates and what their legal
values are.**

No source build needed: tblgen need only understand the tag's TableGen
*language*, not match its version. LLVM 18's tblgen chokes on 22's
`!listflatten` (added in 19), but LLVM_full_jll 21.1.8's parses the 22.1.7
tree cleanly. The extraction (sparse clone → tblgen `--dump-json` → jq filter)
yields 2569 `int_nvvm_*` records — exact agreement with the intrinsic name
table embedded in the 22.1.7 `llc` binary, which doubles as the standing
conformance check for tblgen-version skew on every backend bump. Output
committed as `gen/nvvm_intrinsics_22.1.7.json` (names, signatures, properties
incl. `IntrConvergent`); spot-checked against the hand-verified mbarrier
signature from the original llc experiments. Generator requirement recorded in
the script: 353 records carry explicit `LLVMName` overrides that must take
precedence over the default `int_nvvm_foo_bar → llvm.nvvm.foo.bar` rule.

### Requirements provenance

Per-form minimum sm / PTX ISA isn't in the intrinsic definitions (predicates
live on ISel patterns). Derive empirically: trial-compile each registry entry
across a target matrix with the artifact `llc`; hand-curate only ambiguous
cases. Same harness as CI verification, different axis.

### PTX.jl currently tracks CUDA.jl's main branch

Package sources point at the development branch (pre-6.2-release artifact of
needing unreleased fixes). Now that 6.2 is tagged, repoint to the release and
let compat do its job, or confirm what unreleased functionality is still
needed and note it here.

### Downstream migration

The blessing-boundary flip (unregistered chains error instead of hitting the
naive synthesizer) breaks any chain users relied on implicitly — including
this author's own kernels in dependent packages. Pre-1.0 makes this cheap, but
the golden-diff harness should run dependents' kernels too, and the flip lands
last, after the families those kernels use are registered.
