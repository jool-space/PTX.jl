# PTX.jl — working notes for agents and contributors

Facts here are the non-obvious ones that cost hours to rediscover. Code
structure is documented in `docs/src/internals.md`; this file is about how to
*work* on the repo without stepping on the known landmines.

## Running tests

- The documented invocation is `julia --project=test test/runtests.jl
  [names...]` (e.g. `host/inst ptxas/golden`). Plain `Pkg.test("PTX")` forces
  `--check-bounds=yes`, which injects bounds-check branches into device code
  and invalidates the golden-PTX comparison — the goldens are then *silently
  skipped* (a manifest `SKIP` line, not a failure). Use
  `Pkg.test("PTX"; julia_args=["--check-bounds=auto"])` if you must go
  through Pkg.
- Naming a test explicitly bypasses default routing/skips and hits loud
  refusals instead. `PTX_UPDATE_GOLDEN=1` regenerates `test/golden/*.ptx`;
  the review artifact is the git diff.
- Tiers: `host/` needs nothing, `ptxas/` needs only the offline CUDA compiler
  artifacts (no GPU — cubins are never loaded), `gpu/` files carry a
  `# TEST_TARGET:` banner (`cc>=X.Y` floor, `cc==X.Y` exact, integer `cc==X`
  hardware family). Never spell PTX `sm_*` targets in runtime policy.

## Hardware reality

- Local dev box: 2× RTX 6000 Ada (sm_89, CC 8.9), x86_64; ptxas comes from
  CUDACore artifacts, not PATH.
- CI: self-hosted GB10 runner (aarch64, CC 12.1 / sm_121a — *consumer*
  Blackwell, no tcgen05 runtime) plus hosted x64 host/ptxas lanes.
- The `gpu/hopper` (cc==9.0) and `gpu/blackwell` (cc==10|cc==11) runtime
  sections run in **no CI lane** — they are exercised in manual cloud
  sessions (H100, B200/B300). Their ptxas legs do run everywhere. Don't
  interpret green CI as runtime evidence for those tiers: `EVIDENCE.toml`
  records when each tier last executed on hardware and on which tree (the
  test manifest prints it), and a hardware session ends by updating it.

## The form registry is a review boundary

- An opcode absent from `FORMS` (src/ledgers/forms.jl) errors at compile time — by
  design. Adding an entry is a review act: check the ISA for memory effects,
  cross-lane (convergent) semantics, and whether the chain tail names a
  result or an operand. A wrong promise in the permissive direction is a
  miscompile, not a lost optimization.
- `src/nvvm/table.jl` is machine-generated (see `gen/`); never hand-edit.
  Every `nvvm"..."` literal used in `src/` must have a selection probe in
  `test/host/conformance.jl` — a scan enforces this.
- Tests deliberately double-enter expected form inventories (independent
  oracles + count pins) rather than deriving them from `src/`. Adding forms
  means updating the test-side oracle generators *and* their pinned counts
  (`test/host/{conformance,address_roles,tcgen05_*,mbarrier_forms,...}.jl`).
  Do not "fix" this by deriving oracles from the source ledgers.

## Julia-version traps

- 1.10 under GPUCompiler's overlay method table does not concretely evaluate
  ccall-based type tests (`T in (...)`), so kernel-path validation must be
  dispatch-based (methods), not runtime type membership. 1.11+ folds these.
- Test-support namespace is shared across all files in a worker: annotating
  a harness argument as `::Function` would rebind the name against
  `PTX.IR.Function` importers (see the comment on `compile_touch_sweep`).

## Comment discipline

- A comment states a present-tense constraint of the code. Provenance —
  dates, PR/issue numbers, plan or session names — goes in commit messages
  and PR bodies, where git keeps it attached to the change; `git blame`
  answers "when and why", and a comment that answers it instead goes stale
  silently. Same rule for prose fields in data files (SURFACE notes): no
  claims a test can't falsify.
- Deleted evidence scripts (spikes/) are cited by bare name only; the
  archaeology pointer lives in docs/src/internals.md "Evidence archaeology".

## Git and PR conventions

- Squash-merge; working branches use the `agent/` prefix; commits/PRs go
  through CI on both lanes.
- Before force-pushing a train branch, check nothing landed on it since your
  base: `git log origin/<branch> --not <local-rebase-base>`. A stale local
  checkout plus `--force-with-lease` silently discards a child PR squashed
  into the branch mid-rebase (this happened; recovery is fetching the squash
  SHA and cherry-picking).
- Squash-merge shared-tail conflicts: when both sides append functions with
  identical closing lines, git folds the tail into shared context; keeping
  "both sides" naively drops a closing block. Resolve as HEAD-side + tail +
  branch-side + tail, and read every hunk.

## Codegen traps for warp-specialized kernels

- `setmaxnreg` is ignored by ptxas unless the kernel declares `.reqntid` —
  pass `minthreads=N` through `@cuda`/compiler_config. Check
  EIATTR_REG_COUNT in the cubin.
- `@inbounds CuDynamicSharedArray(...)`: the @boundscheck emits a
  gpu_report_exception CALL in the entry region.
- NTuple getindex with a runtime index demotes the tuple to local memory;
  make the index a `Val` type parameter. Large `ntuple(...) do` closures
  become real device CALLs — write explicit tuples or check `code_llvm` for
  `call .*julia_`.
- `UInt32(x)` on loop-carried Ints leaves live InexactError branches; use
  `% UInt32` when exact by construction.
- Debug loop: `code_llvm` → grep `call .*julia_|report_exception|alloca`;
  `nvdisasm -c` → grep LDL/STL/CALL; `ptxas -v` for spill counts.
