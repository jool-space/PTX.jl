# PTX robustness findings

This directory is the tracked, merge-safe ledger for the PTX.jl robustness
audit. Each finding has exactly one record in `findings/<ID>.toml`. Keeping
findings in separate files lets independent pull requests update different
records without contending on a shared generated index.

## Lifecycle

Every record has one of these statuses:

- `confirmed`: evidence establishes a defect, coverage gap, or validation
  weakness, but the finding has not been completely handled.
- `needs_experiment`: the concern is plausible, but the stated experiment is
  still needed before it can be classified or fixed confidently.
- `deferred`: the scope is intentionally postponed and its disposition must
  remain explicit.
- `resolved`: the stated scope has been handled and the record contains a
  resolving pull request, an implementation (or historical merge/squash)
  commit, and nonempty validation evidence.

A resolving pull request updates its finding to `status = "resolved"` and
records its evidence in the same PR. Normally the implementation is committed
first so that `resolution.commit` can name that stable commit; historical
backfills may name the merge or squash commit. The branch record describes the
PR's intended result, but the ledger on `main` is canonical only after that PR
merges. A partial fix must not mark an umbrella finding resolved: narrow the
finding's scope or add child findings for residual work.

## Record schema

Required top-level fields are:

- `id`: exactly the filename stem.
- `status`: one of the lifecycle values above.
- `severity`: `critical`, `high`, `medium`, or `low`.
- `category`: the finding class used by the audit.
- `summary`: the concrete problem or missing guarantee.
- `source_handoffs`: audit reports that established or refined the finding.
- `validation_tier`: current evidence or the evidence still required.

Resolved records also require a `[resolution]` table with:

- `pr`: the resolving pull-request number.
- `commit`: a full 40-character Git commit ID containing the implementation,
  or the historical merge/squash commit for a backfilled resolution.
- `scope`: what the resolution covers; use `full` only when no stated scope
  remains.
- `evidence`: nonempty, concrete validation facts. Keep host construction,
  emitted PTX/LLVM, `ptxas`, and hardware runtime evidence distinct rather
  than implying one tier proves another.

Historical/backfilled records also carry `merged_at`, the merge date. A new
resolving PR cannot know that date in advance and may add it in a post-merge
bookkeeping change.

`test/host/audit_findings.jl` discovers every record and validates the exact
top-level and resolution schemas, filename/ID agreement, unique IDs,
enumerated status/severity values, and the extra provenance required for
resolved findings. Its independent closed-world ID inventory catches deleted
or unreviewed records, and its historical resolution floor prevents an already
merged resolution from silently regressing while allowing additional findings
to become resolved.
