# B200 session runbook

Turnkey suite for the next B200/B300 (CC 10.x) session, covering every
`[[queue]]` item that actually needs the hardware. Two items were retired
off-hardware while preparing this (see "Already resolved" below).

Branch layout: `agent/b200-queue` = `tcgen05-flash` + `origin/main` merged
(clean merge; 97,400/97,400 green on the GB10) + this directory. Baseline
runs here double as the **first Blackwell runtime execution of #109–#116**
(current gpu/blackwell evidence stops at c6c3ba5, pre-9.4).

```bash
cd ~/PTX.jl && git fetch origin agent/b200-queue && git switch agent/b200-queue
```

## Phase 0 — baseline suite (≈3 min)

```bash
julia --project=test test/runtests.jl 2>&1 | tee ~/b200-suite.log
```

Green here validates on Blackwell hardware: the mbarrier all-asm demotion
(#110), memory-widen overlay (#109), effect ceiling (#112), the 9.4 feature
set (#113–#116), **and** the BarrierSet-arena FA kernel (queue item
"revalidate FA runtime after the BarrierSet arena adoption" — the FA
runtime testset plus Phase 1's fab_check are that revalidation).

## Phase 1 — FA perf: the exp2-emu sweep (the headline item)

The `emu` knob (commit 6e56298) routes softmax exp2 *pairs* through the
FMA-pipe f32x2 polynomial instead of SFU `ex2.approx`. The 2026-07-28 SASS
diagnosis showed softmax MUFU.EX2 pressure is the remaining bottleneck;
pyptx's headline 1240 TF config uses `emu` — our 1150 TF default doesn't.
`fab_check` gates every candidate (the poly changes numerics; the recorded
default maxdiff 1.55e-2 sits only ~3× under the 5e-2 gate).

```julia
# julia --project=test, from the repo root
include("bench/flash_attention.jl")

fab_check()                                   # gate the default; expect ~1.6e-2 OK
fab_time(; B = 1, H = 8, S = 8192)            # baseline; expect ≈1150 TF

emu4 = (; FAB_CFG_DEFAULT..., emu = (1, 3, 5, 7))      # pyptx headline set
emu8 = (; FAB_CFG_DEFAULT..., emu = (0, 2, 4, 6, 8, 10, 12, 14))
full = (; FAB_CFG_DEFAULT..., emu = Tuple(0:15))       # density endpoint

for (name, cfg) in ("emu4" => emu4, "emu8" => emu8, "full" => full)
    fab_check(cfg) || (println(name, " FAILS the gate — drop it"); continue)
    r = fab_time(cfg; B = 1, H = 8, S = 8192)
    println(name, ": ", round(r.tflops, digits = 1), " TF")
end

# Full table for the winner (each new cfg = ~15 s compile on first call):
fab_sweep(cfgs = ["default" => FAB_CFG_DEFAULT, "winner" => emu4])
```

If an emu set wins at S=8192, re-check the nreg split — emu moves work from
the SFU to the FMA pipe and changes softmax register pressure. Three
candidates around the current winner (all pass `fab_check_cfg`):

```julia
for nreg in ((136, 80, 152), (152, 80, 136), (136, 88, 144))
    cfg = (; FAB_CFG_DEFAULT..., emu = (1, 3, 5, 7), nreg)
    fab_check(cfg) && println(nreg, ": ",
        round(fab_time(cfg; B = 1, H = 8, S = 8192).tflops, digits = 1), " TF")
end
```

**2-CTA decision rule** (the queue's "then decide the 2-CTA port"): pyptx's
2-CTA variant was correct but perf-neutral. If emu lands ≥ ~1200 TF, the
1-CTA port is within a few percent of pyptx's 1240 headline and the 2-CTA
port stays skipped; if emu is neutral or fails the gate, the remaining gap
is the half-granular split-P work, not clusters — record and move on.

## Phase 2 — idesc probes (≈2 min)

```bash
julia --project=test experiments/b200-queue/idesc_probe.jl
```

Hand-encoded Table-51 descriptors for `.kind::i8` (s8×s8→s32, idesc
`0x084004A0`) and `.kind::f8f6f4` (e4m3×e4m3→f32, `0x08400010`), exact
uniform-ones references (32 / 32.0). Emit + ptxas legs already green
locally; the runtime PASS is the evidence the real `tcgen05_instr_desc_i8`
/ `_f8f6f4` builders will cite — it unblocks the tcgen05 coverage-gap PR
split. FAIL output prints the unique values seen (a wrong dtype encoding
shows up as a recognizable wrong constant, e.g. e4m3 bits read as e5m2).

## Already resolved — do NOT spend box time on these

- **argmem-widen A/B** (`agent/argmem-widen-b200`): widening every tcgen05
  management verb (alloc/dealloc/relinquish/commit/cp) and the
  cp.async.bulk/TMA families to unknown memory — fn-level props AND per-arg
  readonly/writeonly — produces **byte-identical instruction streams** for
  the FA kernel, gemm_highperf_blackwell, and tcgen05_b128_probe at sm_100a
  (gensym/decl-order noise only). The table's location precision is not
  load-bearing for any real kernel in the suite; demoting these families to
  plain clobbering asm is codegen-neutral and can proceed as ordinary
  reviewed PRs with goldens as evidence. Optional 5-minute paranoia check:
  `git switch agent/argmem-widen-b200 && julia --project=test test/runtests.jl`
  (expect green with identical counts).
- **issue #50 fabric**: the ptxas leg ran 2026-07-29 — all six fabric
  spellings assemble with ptxas 13.3 at sm_100a/100f/103a/110a and reject
  on sm_90a with per-feature diagnostics; toolkit floor is ISA 9.3
  (ptxas 13.0 rejects on `.version`). Runtime semantics remain
  **infrastructure-blocked** (configured CFT endpoints — not available on a
  spot instance); nothing to run here.

## Session close

Update `EVIDENCE.toml` in the same PR as any fixes:
- `[[tier]] gpu/blackwell`: date, device, tree = this branch's merge commit
  (or main if the merge lands first), result = suite count + "first
  Blackwell runtime execution of #109–#116" + FA sweep numbers.
- `[[queue]] B200/B300`: retire completed items; carry forward whatever the
  emu sweep decides about the 2-CTA port and split-P.
- Paste the idesc probe verdict into the queue item / future builder PR,
  and the fab_sweep table into the FA session comment block.
