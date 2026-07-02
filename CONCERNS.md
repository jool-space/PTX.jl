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

**Status: RESOLVED 2026-06-12 — implemented and enforcement verified
empirically.**

`NVPTX_LLVM_Backend_jll` is a weak dependency with compat `"22"` (the
registry's generation major); host-side use of PTX.jl never downloads the
~60 MB artifact. Pkg enforcement was verified in both directions on a
fresh environment: a `=22.1.5` pin *steered* resolution away from 22.1.7,
and an unsatisfiable `"23"` produced a resolver conflict naming both PTX
and the JLL. Extension-less weakdep compat is honored.

The pin is backed by a standing conformance harness
(`test/host/conformance.jl`): (1) the resolved artifact's version must
equal `NVVM.BACKEND_LLVM_VERSION`; (2) the committed table's 2569 names
must diff empty against the `llvm.nvvm.*` name table embedded in the
resolved `llc` binary, both directions; (3) every intrinsic the wrappers
stand on has a selection probe — synthesized IR through the artifact llc,
asserting the *expected instruction* (acceptance alone is meaningless;
llc remangles and upgrades silently). The probe list is self-policing: a
scan of `src/` for `nvvm"..."` literals fails the suite if any used
intrinsic lacks a probe — which is also why wrappers spell intrinsic
names literally instead of building them in loops. This is the mechanism
that turns the legacy-intrinsic exposure (see the mbarrier ledger entry)
from a silent break into a red test at bump time.

Two further layers were added 2026-06-20 after auditing what the first three
*don't* cover: (4) **attribute-extraction faithfulness** — a pin asserting the
exact `props` tuple of a representative per attribute archetype (the
`convergent` set plus the memory-effect archetypes). This exists because
attributes can't be checked the way names are: the artifact llc **ignores the
attributes we attach and re-derives its own** — verified directly by declaring
`read.ptx.sreg.tid.x` as `memory(write)` and watching llc still CSE two reads.
So a trial-compile-through-llc matrix is *blind* to attributes; they are
load-bearing only for the in-process LLVM (the convergence-spike miscompile
path), and the realistic rot vector is an extraction-map regression in `gen/`
on a re-gen, which this pin trips. Anchor tuples were spot-checked against the
22.1.7 `IntrinsicsNVVM.td` source (`bar.warp.sync`, the `barrier.cluster.arrive`
family, `add.rn.f`, `read.ptx.sreg.tid.x` via `NVVMPureIntrinsic`,
`atomic.add.gen.f.cta`, `fence.proxy.async`). (5) **type-token coverage beyond
the used surface** — auditing the table, the only type tokens unique to
never-used intrinsics are `i128` (4 `clusterlaunchcontrol.query_cancel.*`) and
`Metadata` (1 legacy `texsurf.handle`); the wide-vector tokens are already
exercised by the tcgen05 ld/st probes. The `i128` token is now compile-pinned
through llc; `Metadata` has no normal SSA callsite and is the registry's sole
known-uncompiled token, logged (not silently skipped). Net: this closes the
"attributes/unused-surface unverified" gap noted earlier — see also the
"Requirements provenance" item, which a full trial-compile matrix would
*not* in fact discharge for attributes.

## Diligence — per-op care, never globally discharged

### Tier-1 semantic mappings

PTX orderings/scopes → LLVM orderings/syncscopes is a semantic translation
with miscompile potential (see DESIGN.md, lowering tiers). Every tier-1
entry needs an explicit mapping decision and a golden-output test. No batch
sign-off. Current exposure: the package wraps no atomics; both standing
tier-1 candidates are now migrated with their mappings pinned — vector
ld/st 2026-06-13, the generic memory fences 2026-07-02 (see the dedicated
sections below). The next tier-1 surface to arrive (atomics, `cvt`) re-opens
this concern; it is never globally discharged.

Split clarified 2026-06-13: the *proxy/init* fences are NOT tier-1. A
core-IR `fence` orders generic memory with an ordering+syncscope; it cannot
name a memory *proxy*. So `fence.proxy.async`, `fence.proxy.async.shared::cta`,
and `fence.mbarrier_init.release.cluster` are tier-2 — migrated to
`llvm.nvvm.fence.*` intrinsics (`test/golden/fences@sm90a.ptx`,
byte-identical: the intrinsics emit exactly the asm spellings). They are not
`convergent` (the registry marks them `nocallback` only), which is correct —
duplicating an idempotent ordering fence is harmless, unlike the
warp-collective ops the convergence spike was about. What remains genuinely
tier-1 (core-IR `fence <ordering> syncscope(...)`): the generic
`fence.sc.*` / `fence.acq_rel.*` — migrated 2026-07-02, see the next
section. `fence.proxy.tensormap::generic.*` takes operands and rides with
the unmigrated `tensormap.replace` path.

### Generic memory fences as tier-1 core IR — MIGRATED 2026-07-02

`fence.{sc,acq_rel}.{cta,cluster,gpu,sys}` (wrappers/fence.jl) now lower to
core-IR `fence <ordering> syncscope(...)`, off the asm chain default. The
mapping — the explicit decision this family was deferred behind:

    PTX sem    LLVM ordering      PTX scope   LLVM syncscope
    .sc        seq_cst            .cta        "block"
    .acq_rel   acq_rel            .cluster    "cluster"
                                  .gpu        "device"
                                  .sys        (default — system scope)

A clean bijection on the notation surface: PTX has exactly two generic
fence sems and LLVM's remaining orderings (acquire, release) have no PTX
generic-fence spelling, so nothing is lossy in either direction. Verified
by trial compilation through the artifact llc 22.1.7 (expected instruction
asserted, per the mangling lesson — llc acceptance alone proves nothing):

- All eight forms emit exactly the written PTX spelling at sm_70/sm_90
  (`fence.sc.cta` … `fence.acq_rel.cluster`). Goldens byte-identical
  across the migration (`test/golden/fences_generic@sm{70,90}.ptx`),
  matching the proxy-fence precedent.
- Cluster scope below its sm_90 floor fails ISel *loudly* ("Requires
  SM >= 90 and PTX >= 78") — never a silent scope downgrade.
- Below sm_70/PTX 6.0 the backend legalizes to the pre-Volta
  `membar.{cta,gl,sys}` equivalents (sem distinction collapses,
  conservatively) — where the asm tier handed ptxas an instruction it
  rejects. Non-WYSIWYG by design; strictly wider availability.
- The in-process LLVM 18 parses and preserves the target syncscopes
  through Julia codegen (asserted in test/host/wrappers.jl), so the
  ordering constraint binds in the middle end too — which is the point:
  unlike the asm tier's opaque sideeffect call, the optimizer now *knows*
  these order memory.

Like the proxy fences, not `convergent` (idempotent ordering ops). The
behavior change mirrors vec ld/st: the asm tier's sideeffect call pinned
all surrounding code motion; a core-IR fence pins memory accesses across
it (its actual semantics) and nothing else. Golden/baseline kernels
interleave stores between fences so each fence's position stays observable
regardless.

### Vector ld/st as tier-1 core IR — MIGRATED 2026-06-13

`ld.global.v{2,4}` / `st.global.v{2,4}` (wrappers/vec_ldst.jl) now lower to
core-IR `load`/`store <N x T>` on `ptr addrspace(1)` with `align N*sizeof(T)`,
dropping the asm `~{memory}` clobber. The body repacks between the LLVM vector
type `<N x T>` (what load/store want) and the array type `[N x T]` (how Julia
represents the homogeneous `NTuple` surface) via extractelement/insertvalue
pairs that ISel coalesces into the instruction's register group.

The decision the deferral was waiting on — *is dropping the barrier safe for
the working GEMM load paths?* — resolved **yes**, with the behavior verified
explicitly (not assumed):

- **The v{2,4} alignment was already a hard hardware requirement** of the asm
  instruction; asserting `align` adds no new caller obligation. Callers that
  fed misaligned pointers would already have faulted.
- **`.f32` → `.b32` is a cosmetic respelling**, not a semantic change. NVPTX
  canonicalizes float vector ld/st to the bit spelling because registers are
  typeless — verified that even with `mul.f32` consumers the load stays
  `ld.global.v4.b32` and ptxas emits identical SASS. (Tests assert `.b32`.)
- **Narrow b16 vectors are preserved when lanes are individually used** (the
  realistic case) — verified `ld.global.v4.b16` survives lane-wise arithmetic.
  An *opaque* b16 passthrough legally coalesces into a wider `.b32` access
  (same bytes, same transaction count), which is why the golden/baseline
  kernels touch each lane to lock the per-lane contract rather than the
  optimizer's packing choice.
- **Dead-store elimination is now live** and is the point: a same-address
  load→store round-trip is correctly eliminated to nothing (the asm barrier
  used to pin it). Test kernels store at a *distinct* offset to keep the
  access observable. Real GEMM paths transform the loaded values before
  storing, so nothing of value is eliminated — but this is the optimization
  the migration unlocks (offset folding into the addressing mode, reorder,
  CSE). Golden `test/golden/vec_ldst@sm70.ptx` pins the diff: the asm tier's
  per-offset `add.s64` scaffolding folds into `[%rd+imm]`.

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
- **tma / cp.async.bulk.tensor** (2026-06-12, `test/golden/tma@sm{90,100a}.ptx`
  diffs): win, with one visible cost and one renamed spelling. The asm
  tier's `r` constraints forced `mov.b64` + `cvt.u32.u64` materialization
  of the shared dst/mbar addresses; ISel addresses both symbolically
  (`[var0]`). New cost: the intrinsics carry every optional operand
  (multicast mask, cache hint), so unused ones materialize as zero
  registers — CSEd across all TMA calls in the kernel and trivially dead
  for ptxas, but visible in the PTX. Spelling: ISel renders `.cta_group::2`
  after the completion mechanism where the asm tier (pyptx order) put it
  after `.<N>d`; ptxas accepts both (validated sm_100a). Address spaces:
  the cluster-destination forms go through `g2s.tile.<N>d` whose dst is
  `ptr addrspace(7)` and all forms take the tensormap as a *generic*
  pointer — the wrapper raw-retypes both (`reinterpret_addrspace`,
  ptrtoint/inttoptr), never `addrspacecast`: NVPTX lowers the cast to
  `cvta` translation, a wasted round-trip for shared→cluster and
  *corrupting* for the descriptor (a global address typed AS.Const by
  convention). `shared::cta` loads ride `g2s.cta.tile.<N>d`, ISel floor
  PTX 8.6 — the same floor ptxas already imposed on that spelling.
  Residue: `shared::cta` × `cta_group::2` has no intrinsic (g2s.cta lacks
  cta_group; g2s renders shared::cluster) — stays asm-tier. En passant the
  migration exposed a golden-harness soundness bug: the parser dumped the
  TMA coordinate bracket into a raw offset string, so canonical renaming
  never reached the coordinate registers (raw names leaked, collisions
  possible) — fixed structurally before the baseline capture.
- **tcgen05** (2026-06-12, `test/golden/tcgen05@sm100a.ptx` diff): win,
  with two explicit-default spellings and one widening cost. Dense mma
  (kinds f16/tf32/f8f6f4/i8) lowers through `mma.shared` with kind /
  cta_group / collector_usage immargs — the MASKLESS dense form, so the
  asm tier's all-zero disable-output-lane mask (4/8 zero words per call)
  disappears; the output spells the ISA-default `.collector::a::discard`
  explicitly. commit lowers through `commit[.mc].shared.cgN` and emits
  the `.shared::cluster` spelling for both state-space notations (a
  shared::cta mbar address is valid in the cluster window) with multicast
  ordered after the space (pyptx put it first; ptxas accepts both).
  Address spaces: TMEM is addrspace(6), 32-bit in the backend datalayout,
  so the surface's raw `UInt32` taddr stays a 32-bit register —
  `reinterpret_addrspace(Val(AS.Tmem), ::UInt32)` is a free inttoptr; the
  p3 operands (alloc destination slot, commit mbar) widen via one CSEd
  `cvt.u64.u32` per kernel (asm used 32-bit `r`). ld/st move data as LLVM
  vectors (`v<N>i32`); wrappers repack to the surface's plain tuples
  (folds to the same registers). ISel floors: family ops PTX 8.6, mma
  PTX 8.8, datacenter-Blackwell targets only (sm_100/103; consumer sm_12x
  has no tensor memory — ISel now enforces what ptxas did). Residue: mx
  kinds (mxf8f6f4/mxf4/mxf4nvf4) stay asm-tier — their intrinsics are
  `.block_scale` forms requiring TMEM scale_a/scale_b operands the
  notation surface does not carry; revisit with the block-scale surface
  design.
- **mma.sync (dense + block-scaled)** (2026-06-13,
  `test/golden/mma@sm90a.ptx`, `mma_fp8@sm121a.ptx`,
  `mma_scaled@sm121a.ptx`): NEUTRAL — the migration's value is uniformity,
  the convergent-attribute correctness story, and a dead-form cleanup, not
  perf. mma operands are register-bound either way and the family has no
  immargs, so the emitted instructions are byte-identical; golden diffs are
  only (a) f32 zero-accumulator spelling `0`→`0f00000000`, (b) register
  renumbering, (c) the ISel qualifier reorder shape-before-kind for the
  kind/block_scale forms (`kind::f8f6f4.m16n8k32`→`m16n8k32.row.col.kind::
  f8f6f4`; ptxas accepts both — the *notation input* surface stays
  kind-first, only the emitted spelling reorders). The packed-UInt32 ⇄
  `<2 x half>` repack for f16-input/f16-acc forms is a free bitcast.
  Cleanup: the asm tier registered all four layA×layB combos, but modern
  m16n8k* shapes only support `.row.col` (ptxas rejects the rest, and the
  registry has no intrinsic) — three of every four were dead methods that
  emitted ptxas-rejected asm; the migration registers `.row.col` only, so
  a bogus-layout call is now a clean MethodError. Residue (asm-tier
  fallback, automatic where LLVM 22.1.7 lacks an intrinsic): the 50
  `kind::f8f6f4` forms at m16n8k16, and `mxf4nvf4` `scale_vec::4X` with a
  `ue8m0` scale type. First GENERATED family migrated (vs hand-literal):
  the dtype cross-product (`kind::f8f6f4` alone is 100 forms) makes literal
  `nvvm"..."` methods absurd, so the generator emits `IntrinsicCall`
  directly and conformance covers it by a registry-completeness assertion
  over the generated name list plus one selection probe per structural
  class — the src/ literal-scan only governs the hand-literal families.
  Gating note: `kind::f8f6f4` and the fp8 paths are consumer-Blackwell
  (sm_120a+, the GB10 sub-byte FP accelerator) — the *opposite* of
  tcgen05's datacenter-only floor.
- **vec ld/st.global** (2026-06-13, `test/golden/vec_ldst@sm70.ptx` diff):
  WIN. Tier-1 core IR (`load`/`store <N x T>`), no NVVM intrinsic. Dropping
  the `~{memory}` barrier lets the backend fold each constant store offset
  into the addressing mode — the asm tier's per-offset `add.s64` scaffolding
  (`add.s64 %rd3,%rd0,32; ... [%rd3]`) collapses into `[%rd0+32]`. Same
  vector instructions otherwise; `.f32` canonicalizes to the `.b32` bit
  spelling (cosmetic — identical SASS). No perf regression; the freed
  reorder/CSE/DSE is the upside. See the dedicated section above for the
  safety verification (alignment was already required, narrow b16 preserved
  under lane use, DSE is intended).
- **generic memory fences** (2026-07-02,
  `test/golden/fences_generic@sm{70,90}.ptx`): NEUTRAL on emission —
  byte-identical goldens, all eight forms emit exactly the written
  spelling. The migration's value is semantic honesty: the optimizer now
  sees a real ordering op (memory pinned across it) instead of an opaque
  sideeffect asm call (everything pinned). Mapping decision and
  verification in the dedicated tier-1 section above.

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

Scope clarified 2026-06-20: a full trial-compile matrix validates *names*
(already covered by the binary diff), *signatures*, and the *sm/PTX floor* —
it does **not** validate attributes (llc ignores the attributes we attach; see
the conformance harness layer 4 note). And the signature/token dimension is
nearly closed already: layers 4–5 established that the only type tokens outside
the used+probed surface are `i128` (now compile-pinned) and `Metadata` (no SSA
callsite). So the remaining *unique* value of this matrix is the per-form
sm/PTX floor for intrinsics we don't yet use — worth it when more of the
registry gets wrapped, low-value before then. Not a prerequisite for the
attribute-confidence question, which layer 4 answers directly.

### PTX.jl currently tracks CUDA.jl's main branch

**Status: RESOLVED 2026-07-02 — on the tagged release.**

Package sources pointed at the development branch (pre-6.2-release artifact
of needing unreleased fixes). Compat now bounds `CUDACore = "6.2"`, a fresh
resolve pulls the registered 6.2.0 release (no git source), and the full
suite (26984 tests) passes against it — nothing still depends on unreleased
functionality.

### Downstream migration

The blessing-boundary flip (unregistered chains error instead of hitting the
naive synthesizer) breaks any chain users relied on implicitly — including
this author's own kernels in dependent packages. Pre-1.0 makes this cheap, but
the golden-diff harness should run dependents' kernels too, and the flip lands
last, after the families those kernels use are registered.
