# gen/ — the NVVM registry generator

`src/nvvm/table.jl` is machine-generated from the backend LLVM's
`IntrinsicsNVVM.td` and committed as source, so a backend bump becomes a
reviewable diff. It is never edited by hand. This directory holds the two
generator halves and the extracted JSON snapshot they communicate through.

- `extract_intrinsics.sh` — front half. Sparse-clones llvm-project at a
  tag, evaluates the intrinsic `.td` tree with `llvm-tblgen`, filters to
  `int_nvvm_*` records with `jq`, writes `nvvm_intrinsics_<version>.json`.
  The output schema and the `LLVMName` override rule are documented in the
  script header.
- `generate_registry.jl` — back half. Consumes the JSON and emits
  `src/nvvm/table.jl`. Closed-world by design: an unknown tblgen property,
  type token, or argument-property kind is an **error**, never a
  fallthrough — new vocabulary in a future `.td` must be reviewed, not
  absorbed.

## Prerequisites

- `bash`, `jq`, and a `git` new enough for partial clone
  (`--filter=blob:none --sparse`), plus network access to
  `github.com/llvm/llvm-project`.
- `julia --project=gen` resolves `LLVM_full_jll` to supply `llvm-tblgen`.
  The tblgen need only understand the tag's TableGen *language*, not match
  its version (LLVM 18's chokes on 22's `!listflatten`; 21's parses 22.1.7
  cleanly). If a future tag outgrows the pinned tblgen, bump
  `LLVM_full_jll` here — this pin is independent of the backend version.

## Backend bump runbook

When `NVPTX_LLVM_Backend_jll` moves to a new version `X.Y.Z` (the compat
entry in the root `Project.toml` is the trigger — see the comment there):

1. **Extract**: `gen/extract_intrinsics.sh llvmorg-X.Y.Z` (the tag is the
   JLL version prefixed with `llvmorg-`). This writes
   `gen/nvvm_intrinsics_X.Y.Z.json`; delete the previous snapshot —
   `generate_registry.jl` with no argument expects exactly one.
2. **Check the record count** the script prints against the intrinsic
   name table embedded in the new backend's `llc` binary (the standing
   "registry names" testset in `test/host/conformance.jl` does this set
   equality; a count mismatch at this stage means tblgen-version skew).
3. **Generate**: `julia --project=gen gen/generate_registry.jl`. If it
   errors on unknown vocabulary, review the new tblgen construct against
   its LLVM docs and extend the closed maps (`PROPS`, `ARGATTRS`, `VTS`,
   ...) deliberately.
4. **Review the `src/nvvm/table.jl` diff.** Every removal or rename is an
   API-facing event for the wrappers standing on that intrinsic; every
   property change (a gained/lost `IntrConvergent`, a memory-effect
   change) is an optimizer-contract change.
5. **Update compat + pins**: the `NVPTX_LLVM_Backend_jll` compat entries
   (root and `test/Project.toml`), `NVVM.BACKEND_LLVM_VERSION` consumers,
   and the registry census pins in `test/host/nvvm.jl` (table length,
   return-range/noundef inventories, side-effecting-nomem set) and
   `test/host/conformance.jl` (per-namespace counts, the mma/wmma
   convergence-overlay boundary — designed to go red when upstream gains
   `IntrConvergent`, at which point the overlay in `src/nvvm/emit.jl`
   shrinks instead).
6. **Run the conformance suites**:
   `julia --project=test test/runtests.jl host/nvvm host/conformance`.
