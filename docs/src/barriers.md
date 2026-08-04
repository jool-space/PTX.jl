```@meta
CurrentModule = PTX
```

# Barriers and pipelines

Three small verb-style modules mirror the CUDA C++ convenience headers
on top of PTX.jl's raw wrappers: `PTX.MBarriers` mirrors
`<cuda/barrier>`, `PTX.Pipelines` mirrors `<cuda/pipeline>`, and
`PTX.Warps` names the warp-collective idioms (`__shfl_*_sync`-style
reductions). None are exported; access them as
`PTX.MBarriers.barrier_init` etc. or via `using PTX.MBarriers`. They
add no new hardware surface — every verb lowers to the same
`mbarrier.*` / `mapa` / `fence` / `shfl.sync` wrappers you could call
directly — but they name the idioms every Hopper/Blackwell
producer-consumer kernel repeats.

## MBarriers

```@docs
PTX.MBarriers
PTX.MBarriers.BarrierArray
PTX.MBarriers.barrier_init
PTX.MBarriers.barrier_arrive
PTX.MBarriers.barrier_arrive_expect_tx
PTX.MBarriers.barrier_wait
PTX.MBarriers.barrier_try_wait
PTX.MBarriers.barrier_arrive_cluster
```

### Barrier arenas

A warp-specialized kernel's synchronization plan is rarely one
homogeneous ring — it is a dozen named barrier groups of different
shapes and arrival counts packed into one SMEM region, historically
maintained as a hand-computed byte-offset table plus a thread-0 init
block that must agree with it. [`BarrierSet`](@ref
MBarriers.BarrierSet) makes that plan a single declaration: shapes and
counts in one NamedTuple, offsets/total size/init all derived, every
access a compile-time-constant pointer offset (pinned byte-identical
to the hand-rolled arithmetic by the ptxas-tier tests).

```@docs
PTX.MBarriers.BarrierSet
PTX.MBarriers.BarrierGroup
PTX.MBarriers.barrier_bytes
PTX.MBarriers.barrier_offset
PTX.MBarriers.barrierset_init!
```

## Pipelines

```@docs
PTX.Pipelines
PTX.Pipelines.Pipeline
PTX.Pipelines.pipeline_stage
PTX.Pipelines.pipeline_phase
PTX.Pipelines.pipeline_cursor
PTX.Pipelines.pipeline_init!
```

## Warp collectives

```@docs
PTX.Warps
PTX.Warps.warp_reduce
```
