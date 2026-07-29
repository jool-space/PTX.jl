# H100 standing-queue experiments (issues #52, #49)

The two `[[queue]]` items in `EVIDENCE.toml` for the H100/H200 device class,
packaged turnkey. Run from the repo root on the box:

```bash
julia --project=test experiments/h100-queue/mapa_u64_probe.jl
julia --project=test experiments/h100-queue/matrix_ambig_ptxas.jl
```

Paste each script's full stdout into its issue (#52 / #49). Both scripts are
also runnable off-hardware — they degrade to their host/ptxas legs.

## Issue #52 — MAPA-U64-EXP (`mapa_u64_probe.jl`)

Audit concern: `optype"mapa.shared::cluster.u64"` passes an AS(3) LLVMPtr
under the 64-bit `l` constraint; never established whether that selects a
legal operand or works semantically. `mapa.shared::cluster.u64` itself is
documented (PTX ISA 9.4 §9.7.10.26, example `mapa.shared::cluster.u64 d1,
%reg1, cta;`).

Evidence already recorded off-hardware (2026-07-29, this tree):

- host leg: wrapper emits `mapa.shared::cluster.u64 %rd, %rd, %r` — the
  pointer operand materializes as a full 64-bit register (the NVPTX
  datalayout here keeps AS(3) pointers 64-bit; only p6 is 32-bit).
- ptxas 13.3 accepts both probe kernels at sm_90a, including
  `ld.shared::cluster.u32 %r, [%rd]` (64-bit address form).

What the box adds: the semantic leg. A 2-CTA cluster where each CTA
publishes `0xA0A0000r` in its own SMEM slot; thread 0 maps its slot into the
peer CTA with both carriers and reads back through `ld.shared::cluster`.
PASS = every `v32`/`v64` equals the *peer* marker, and `a64 & 0xffffffff ==
a32` (high bits are reported for the width question). Then flip the audit
disposition to evidence-backed.

## Issue #49 — MATRIX-AMBIG-EXP (`matrix_ambig_ptxas.jl`)

Pure assembly matrix (no GPU); the box's value is *more toolkits* — the
script discovers every ptxas in the artifact, `$PATH`, and
`/usr/local/cuda*`, dedups by release, and reruns the matrix per toolkit.

Results from ptxas 13.3 (artifact) and 13.0, 2026-07-29 — both fully agree:

- **f8f6f4 K shape** (`mma.sync .kind::f8f6f4`, ISA §9.7.16.5 documents
  dense k32 only): k64 dense rejected everywhere ("Incorrect instruction
  type ... '.m16n8k64'"), matching the text. **k16 dense is ACCEPTED though
  undocumented**, and emits real SASS (`QMMA.16816.F32.E4M3.E4M3` at
  sm_120a). Permissive-ptxas divergence; keep k16 out of canonical schemas.
- **integer WGMMA upper-N** (ISA §9.7.17.5 grids integer N as
  8,16,24,32,48..224 step 16): the holes n40/n56 and the 8-mod-16 values
  n232/n248 are rejected as documented, but **n240 and n256 are ACCEPTED**
  and emit real SASS (`IGMMA.64x240x32.S8.S8`, `IGMMA.64x256x32.S8.S8`).
  ptxas's real ceiling is 256; the documented 224 ceiling is understated.
  Keep the ledger on the documented grid (canonical strict), record the
  divergence as permissive-ptxas with SASS backing.
- **modifier ordering**: ptxas accepts *every* qualifier order probed —
  tcgen05.commit in the syntax-block order, this repo's order, and the
  spec's own example order (§9.7.18.12.1 example contradicts its syntax
  block); mbarrier.arrive sem/scope/space in both orders; ldmatrix
  .trans/.shared in both orders. Ordering is a pure canonicalization choice;
  closed-world single-spelling schemas stay justified.
- The 9.4 `multicast::cluster::32b` order pair is version-blocked on every
  ptxas ≤ 13.3 (`VERSION-UNSUPPORTED`) — re-run when a 13.4 artifact lands
  (gate G1 in #107).

Disposition suggestion for the audit record: `resolved` for the ordering
axis (order-insensitive, canonical spelling is ours), `confirmed` (spec
defect, permissive toolchain) for k16 f8f6f4 and n240/n256 integer WGMMA,
with this output attached. An optional future hardening: run an n256 s8
WGMMA numerically on sm_90a hardware — SASS existence makes this low-value.
