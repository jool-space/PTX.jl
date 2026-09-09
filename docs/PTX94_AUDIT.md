# PTX ISA 9.4 GA audit

Audit of issue [#107](https://github.com/jool-space/PTX.jl/issues/107), against
`main` at `4cc5b07` and the corrections accompanying this report.

## Specification baseline

The installed ptx-isa skill contains 501 sections and 272 PNG figures from
[NVIDIA's public PTX 9.4 specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).
Compared with preview commit `313c208e0a94ed3e1683c2b62ad1c860404ae814`,
70 Markdown files changed and one sparsity figure was added; every existing
PNG is byte-identical. After ignoring whitespace and image-source URL changes,
41 sections differ, including spelling corrections. The mixed packed arithmetic,
alternate packed arithmetic, and both `set` instruction sections are byte-identical.

The SHA-256 of the installed `ptx/` snapshot is
`cd2c7ff0940c7f92c0753c4300f0ad3a166864ed59c8b86444d5b87157d7f913`.
To reproduce, sort all file paths under `ptx/`; hash each relative path including
`ptx/`, a NUL byte, and the file contents, in that order.

## Corrections and evidence

- **Readonly loads (§9.7.10.8):** canonical GA syntax is
  `ld{.global}.type.proxy::readonly`. The previous result inference inspected
  the final token and returned `Nothing`. Result inference now reads the scalar
  type before the proxy. The assembler still accepts the preview prefix spelling.
  The new tests exercise all 14 scalar types, generic/global addressing, and both
  normal/raw calls with stored results. The finite transpiler grammar does not
  yet include readonly loads.
- **Stochastic conversion (§9.7.10.24):** GA removes `.pzo` from all seven
  `.rs` destination classes. Direct/raw calls and transpilation now reject that
  combination. Other conversion prefix legality remains delegated to ptxas.
- **Proxy fences (§9.7.15.4):** existing generic chains correctly express
  `fence.proxy.alias.{acquire,release}.sys` and
  `fence.proxy.async::generic.release.sync_restrict::shared::cluster::read.cluster`.
  New tests retain the memory clobber and assemble all three at `sm_90`, with
  rejection at `sm_80`. The latter is non-cumulative and orders shared-cluster
  reads only. Assembly does not prove a concurrent algorithm's ordering.
- **Sparse FP4 (§9.7.18.10.9.3):** GA explicitly distinguishes pair-wise 4:8
  metadata for `v0` on `sm_100a/sm_103a/sm_110a` from element-wise 2:4 for
  `v1` on `sm_107a`. The existing descriptor builders already expose and test
  bit 12. Their documentation now states the metadata distinction and the
  explicit override required from the default `sparsity_version=0`.
- **TF32 32-byte swizzle (§9.7.17.5.1.2.1 and §9.7.18.3):** the corrected
  `((8,2),(4,2)):((8,64),(1,4))` example agrees with the byte-based layout
  picker. A regression pins the 16-by-8 tile and 256-byte stride.
- **Fabric sync (§9.7.11.5.4):** `try_pullred.sync` requires the literal
  full-warp mask `0xffffffff`. No wrapper currently implements this operation;
  the deferred implementation must enforce the mask and convergence rules.

Validation uses the `host/ptx94_ga`, `ptxas/ptx94_ga`, and `gpu/ptx94_ga`
tests plus existing conversion, wrapper, descriptor, layout, barrier,
transpiler-contract, surface, and Aqua tests. CUDA 13.3 explicitly skips
the new assembly/runtime tests. CUDA 13.4.46 provides assembly evidence;
GB10 provides readonly-load numeric evidence. No fabric or CC 10.7 runtime
evidence was obtained.

## Feature coverage

“Binding” below means the reviewed API represents the form; it does not
imply assembly or numeric validation. Existing checked boxes in #107 must
retain that distinction.

| #107 group / GA feature | Binding and remaining work |
|---|---|
| `sm_107`, `sm_107f`, `sm_107a`; directives and special registers | Parser/IR and target metadata covered by #108. |
| Mixed packed arithmetic | All 36 forms; complete target-positive/negative assembly tests; CC 10.7 runtime pending. |
| Alternate FP8/FP6/FP4 x4 arithmetic | 896 canonical plus seven documented compatibility forms; assembly on `sm_100a/sm_103a`; runtime pending. |
| Packed integer `set` | All 32 forms plus 3,112 scalar/half forms; assembly covered by #158; packed runtime pending. |
| `cvt.pzo`, x2 `.rz`, `ue5m3x2`, n1 scaling | Carrier/schema coverage exists. The complete 9.4 legality and assembly matrix, b8 scale bridges, and numeric validation remain open. Stochastic `.pzo` is rejected. |
| Mbarrier multicast `::32b` | 26 schemas and exact wrappers exist; dedicated 9.4 target assembly/runtime evidence remains to be added. |
| Mbarrier layout, phase types, report operands, check-layout | Already implemented and tested. Although §1.3 lists them under 9.4, individual instruction notes identify 9.3; the existing 9.3 metadata is correct. |
| Bulk/TMA multicast `::16b/::32b` | `tcgen05.commit` bindings exist; copy-family additions remain open. |
| `tcgen05.alloc/dealloc.exclusive` and early A-read commit | Bindings exist; dedicated 9.4 assembly and runtime tests remain open. |
| `tcgen05.kind::ti16`, non-ws B collectors, LUT decompression | New instruction families remain unimplemented; unresolved specification defects below still apply. |
| UE5M3 / UE4M3 block32 / 128-lane scales | `mxf4nvf4` already exposes UE5M3 bits; complete target-aware shape/layout support and execution tests remain open. |
| Instruction and SMEM descriptors | Base packers and explicit sparsity-version bit exist. Widened address fields, extended K, scale-layout bit, and LUT offset support remain open. |
| `tcgen05.ld{.red}.spcompress`, register `spcompress/spdecompress` | Unimplemented; grouped operands, non-overlap constraints, descriptors, target tests, and numeric references needed. |
| `applypriority.async.bulk{.tensor}` | Unimplemented. |
| TMA report mechanism, overrides, im2col-no-offset W, eviction priority | New forms remain open. Overrides and the W load mode also apply to `cp.reduce.async.bulk.tensor`; do not restrict the inventory to loads. |
| TMA sub-byte/swizzle/scatter restrictions | Constraint audit and negative coverage remain open; existing tensor-map upload support does not establish these restrictions. |
| Atom/red `.noftz.f32`, bulk f32 semantics | Scalar chain and vector forms exist; #150 enables the two vector assembly forms. Bulk default preserves subnormals; numeric conformance remains open. |
| Readonly loads | Canonical scalar binding and numeric regression fixed here; transpiler promotion remains open. |
| Prefetch `.valid_addr/.L1::32B`, ldmatrix `.s8.s4` | Bindings exist; dedicated new-form target assembly/runtime evidence remains open. |
| GA alias fences / cluster-read proxy fence | Existing chain bindings, now explicit host and target assembly tests; concurrent runtime ordering evidence pending. |
| GA fabric tensor get/put/red | Unimplemented; `sm_107f` family. Requires composite tensor/handle operands, overrides, and completion accounting. |
| GA `fabric.try_atom` | Unimplemented; baseline `sm_100+`, not Rubin-only. Includes shared-memory result/source operands and special CAS alignment. |
| GA `createpolicy.range.fabric` | Unimplemented composite handle binding; baseline `sm_100+`. The ordinary raw scalar policy path is not this API. |
| GA fabric get/put/red cache hints | Unimplemented modifier plus 64-bit policy operand; baseline `sm_100+`. |
| Fabric full-warp mask / TF32 layout correction | Reviewed above; mask is a requirement on the still-deferred pull-reduction wrapper. |

## GA recheck does not resolve the four preview defects

All four remain in the installed GA specification:

1. Table 62 still spells `decpmpress::lut::b` twice.
2. The ti16 instruction-descriptor transpose restriction still conflicts with
   Table 62's transpose support.
3. The ti16 sparse MMA B-collector syntax still omits `[sp-meta-tmem]`.
4. The `.exclusive` target list still omits `sm_107f`, despite the explicitly
   described 576-column range.

The GA recheck is complete. These are now unresolved GA documentation
contradictions, not a reason to wait for the GA release again. Apply local
restrictions or obtain assembler/hardware evidence before resolving each one.
The new tcgen05 pipelining list also spells `tcgen05.mma.cp.4x256b` in one
pairing, and its different-thread example retains inconsistent remnants;
do not copy those literally.

## tcgen05 synchronization review

GA §9.7.18.6 limits implicit pipelines to particular tensor-memory accesses
issued within the same warp. Different warps are non-pipelined. Shared-memory
reads are explicitly outside those tensor-memory pipeline guarantees.

This needs a separate kernel-level correctness pass. In
`test/gpu/blackwell/gemm_highperf_blackwell.jl`, both dispatch loops return
a source slot with `barrier_arrive(bc[slot])` immediately after issuing
asynchronous MMAs, while `tcgen05.commit` occurs after the loop. That sequence
does not establish that the MMA has finished reading the source slot before
the producer may overwrite it. Completion-backed slot release and a fresh
B200/B300 numeric/stress run are required. Prior successful runs are not a
proof that this race is impossible.

The TMEM roundtrip also consumes `tcgen05.ld` results without an explicit
`wait::ld`; include it in that pass. This audit does not claim a reproduced
wrong-result failure or fresh hardware evidence for these kernels.

## PR #150

PR #150 is limited to compiler-ceiling tracking in tests and normalization
of the golden `.version` header. It applies cleanly to `4cc5b07`; all four
affected suites pass with CUDA 13.4.46 (504 assertions). All six GitHub checks
on its head passed. No merge blocker was found. It is now merged as `484ef77`.
