# TMA epilogue — the trailing half of a Hopper GEMM. Threads cooperatively
# write a per-CTA SMEM tile, then thread 0 issues a single
# `cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group` to land it in
# global memory.
#
# This is the wrapper-chain stress test for the TMA *store* path:
#   - `tensor_map_tile_2d` builds the destination descriptor host-side
#   - `cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group` (device)
#   - `cp.async.bulk.commit_group` + `cp.async.bulk.wait_group.read`
#
# Always validates cross-arch (sm_90a ptxas). Runtime launch deferred behind
# the same H100-only gate as gemm_warpgroup.jl — `cp.async.bulk.tensor.*`
# accepts global-space tensormaps per PTX 9.2 §9.7.8.24.6, but wiring an
# AS.Global tmap through the AS.Const-typed wrapper needs a wrapper overload
# (deferred work).

using PTX: smem_addr_u32
using CUDACore

const EPI_BM = 8
const EPI_BN = 8

function _tma_epilogue_kernel!(tma_D::PTX.TMADescriptorPtr)
    smem = CuStaticSharedArray(UInt16, EPI_BM * EPI_BN)

    tid = ptx"mov.u32"(sreg"tid.x")

    # Every thread fills one cell; bf16(tid) ≈ tid for small values.
    @inbounds smem[Int(tid) + 1] = UInt16(0x3f80) + UInt16(tid)

    ptx"bar.sync"(Val(0))
    # SMEM is in generic proxy here; TMA reads via async proxy. Fence converts.
    ptx"fence.proxy.async.shared::cta"()

    if tid == UInt32(0)
        ptx"cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"(
            tma_D, Int32(0), Int32(0), pointer(smem))
        ptx"cp.async.bulk.commit_group"()
        ptx"cp.async.bulk.wait_group.read"(Val(0))
    end
    return nothing
end

@testset "TMA store epilogue compiles at sm_90a" begin
    types = Tuple{PTX.TMADescriptorPtr}
    @test ptxas_compiles(_tma_epilogue_kernel!, types;
                         cap = v"9.0", feature_set = :arch)
    ptx = emit_ptx(_tma_epilogue_kernel!, types;
                   cap = v"9.0", feature_set = :arch)
    @test occursin("fence.proxy.async.shared::cta", ptx)
    @test occursin("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group", ptx)
    @test occursin("cp.async.bulk.commit_group", ptx)
    @test occursin("cp.async.bulk.wait_group.read 0", ptx)
end

@testset "TMA store epilogue compiles at sm_100a (Blackwell)" begin
    # Same wrapper chain has to keep emitting for Blackwell — TMA store is
    # not deprecated in PTX 9.0. Cap-bumps are mechanical (sm_100a is the
    # current Blackwell datacenter target).
    types = Tuple{PTX.TMADescriptorPtr}
    @test ptxas_compiles(_tma_epilogue_kernel!, types;
                         cap = v"10.0", feature_set = :arch)
end
