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
