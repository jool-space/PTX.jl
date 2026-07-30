```@meta
CurrentModule = PTX
```

# Internals

Machinery documented for contributors and the curious — **not** part of
the public API. Everything here may change without notice; the public
surface is the [Reference](reference.md) page.

## Form registry

The blessing boundary's data: every promise the chain default makes to
the optimizer (purity, memory effects, convergence) lives in one
auditable table in `src/ledgers/forms.jl`.

```@docs
PTX.form_contract
```

## Wrapper registry

Every typed wrapper method registers a record (family, op, mods, tier,
intrinsic) through one entry point; the accessors below are what the
conformance and coverage tests query instead of per-family bookkeeping
constants.

```@docs
PTX.register_wrapper!
PTX.wrapper_records
PTX.wrapper_intrinsic_names
PTX.wrapper_asm_forms
PTX.wrapper_missing_intrinsics
PTX.wrapper_intrinsic_call
PTX.ceiled
```

## Lowering reflection

Ask any operation what it will actually become — which lowering tier a
call binds to, and to which intrinsics — without compiling for a device.

```@docs
PTX.lowering
```

## NVVM intrinsic registry

Generated from the backend LLVM's `IntrinsicsNVVM.td` (see `gen/`) and
committed as source, so a backend bump is a reviewable diff. The
`nvvm"..."` tier emits calls through this registry with explicit
attribute groups — the in-process LLVM doesn't know these names, so the
attributes here are the only effect/convergence information the
optimizer gets.

The attributes are bounded by the package's own review: the FORMS/ledger
contract for the PTX form being emitted is the **ceiling** on optimizer
permissiveness for *both* lowering tiers. Table attributes may refine
below it (`speculatable`, return ranges, param attrs, narrower memory
location classes); a permissive-direction divergence — the table claiming
purity, reorderability, or non-convergence the reviewed contract does not
grant — fails at generation time (`check_ceiling`), and is resolved only
by a reviewed overlay (`CONVERGENT_OVERLAY_PREFIXES`,
`MEMORY_WIDEN_OVERLAY`) or a reviewed contract change. Every package call
site carries its ceiling (`ceiled(nvvm"...", ptx"...")`,
`wrapper_intrinsic_call`, the sreg fast path); the bare `nvvm""` macro is
unceiled plumbing, and `test/host/effect_ceiling.jl` keeps it out of
`src/`.

One asm-tier honesty note: the `:observable` effects class (never
deletable, touches no tracked memory) currently *renders* with the same
conservative barrier as `:clobbers` — LLVM treats `sideeffect` inline asm
without call-site `memory(...)` attributes as unknown memory, so the
class documents reviewed semantics without yet granting the optimizer
freedom. Attaching call-site memory attributes through the
handwritten-IR path is the known lever; measurement showed motion
windows in real pipelined kernels are bounded by their barrier waits
regardless, so the lever stays unpulled until a profile demands it.

```@autodocs
Modules = [PTX.NVVM]
```

## Golden-harness IR

The structural PTX comparison behind the golden tests: parsed modules are
normalized and canonicalized (registers, labels, and symbols renamed to a
stable scheme) so goldens pin modeled instruction structure, not formatting or
register-allocation accidents. Opaque `RawLine` nodes are not treated as
structural coverage; strict goldens reject them recursively before comparison.
Neither canonicalization nor an empty module diff is PTX ISA validation.

```@docs
PTX.IR.normalize
PTX.IR.canonicalize
PTX.IR.diff
PTX.IR._sym
```

## Transpiler contract

The transpiler validates the complete IR module against its deliberately
narrow semantic boundary before emitting any Julia source.

```@docs
PTX.Codegen.validate_transpilable
```

## Evidence archaeology

One-off validation scripts lived in `spikes/` until their findings were
baked into code, tests, and goldens; the directory was then deleted.
Comments citing a `spikes/*.jl` script refer to these — view any of them
with `git show ccdfb8a~1:spikes/`. The load-bearing ones:

- `spikes/convergence.jl` — reproduces the divergent-duplication
  miscompile class that motivates `convergent` on collective ops.
- `spikes/raw_asm_attrs.jl` — proves a `convergent` attribute group on an
  inline-asm call site parses through `Base.llvmcall` and survives the
  optimized module (the `convergent_asm_ir` mechanism).
- `spikes/aggregate_return.jl` — hardware validation of the ldmatrix
  aggregate-return repack and mangled overloaded-callsite names.
