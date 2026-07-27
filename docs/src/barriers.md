```@meta
CurrentModule = PTX
```

# Barriers and pipelines

Two small verb-style modules mirror the CUDA C++ convenience headers on
top of PTX.jl's raw wrappers: `PTX.MBarriers` mirrors `<cuda/barrier>`,
and `PTX.Pipelines` mirrors `<cuda/pipeline>`. Neither is exported;
access them as `PTX.MBarriers.barrier_init` etc. or via
`using PTX.MBarriers`. They add no new hardware surface — every verb
lowers to the same `mbarrier.*` / `mapa` / `fence` wrappers you could
call directly — but they name the idioms every Hopper/Blackwell
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

## Pipelines

```@docs
PTX.Pipelines
PTX.Pipelines.Pipeline
PTX.Pipelines.pipeline_stage
PTX.Pipelines.pipeline_phase
PTX.Pipelines.pipeline_cursor
PTX.Pipelines.pipeline_init!
```
