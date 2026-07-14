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
auditable table in `src/forms.jl`.

```@docs
PTX.form_contract
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
canonicalized (registers, labels, symbols renamed to a stable scheme) so
goldens pin instruction structure, not register allocation accidents.

```@docs
PTX.IR.canonicalize
PTX.IR._sym
```

## Transpiler contract

The transpiler validates the complete IR module against its deliberately
narrow semantic boundary before emitting any Julia source.

```@docs
PTX.Codegen.validate_transpilable
```
