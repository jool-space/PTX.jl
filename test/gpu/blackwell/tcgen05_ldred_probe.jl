# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc==10.3
# tcgen05.ld.red numeric probe — seed one TMEM row per lane with a known
# signed pattern via tcgen05.st, then read it back through the ld.red
# reduction forms and check both outputs: the per-lane data registers
# (must equal what was stored) and the redval destination (min / max /
# max.abs over the row, plus the u32 bit-pattern max).
#
# The ISA supports ld.red on sm_110a and the sm_103f/sm_110f families
# only — no sm_100 — so the runtime gate is cc==10.3 (B300; a B200 can
# never run this) and the compile leg targets sm_103f. The 16x32bx2 split
# shape's rendering is pinned in ptxas/tcgen05_ldst.jl; this probe keeps
# to 32x32b, where every lane owns one full row.

using PTX: smem_addr_u32, tmem_lane_addr
using CUDACore

const TLR_COLS = 64

# Signed pattern where max, min, and max.abs are three distinct values:
# odd columns are large-magnitude negatives.
@inline _tlr_val(c) = ifelse(isodd(c), -Float32(c + 100), Float32(c))

function _tcgen05_ldred_kernel!(O::CuDeviceVector{Float32, 1},
                                U::CuDeviceVector{UInt32, 1})
    tmem_slot = CuStaticSharedArray(UInt32, 1)
    slot_addr = smem_addr_u32(pointer(tmem_slot))

    tid  = ptx"mov.u32"(sreg"tid.x")
    lane = tid & UInt32(31)

    if tid < UInt32(32)
        ptx"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"(
            slot_addr, UInt32(512))
    end
    ptx"bar.sync"(Val(0))

    tmem_base = @inbounds tmem_slot[1]
    tmem_addr = tmem_lane_addr(tmem_base, lane)

    src = ntuple(c -> reinterpret(UInt32, _tlr_val(c)), Val(TLR_COLS))

    if tid < UInt32(32)
        ptx"tcgen05.st.sync.aligned.32x32b.x64.b32"(tmem_addr, src)
        ptx"tcgen05.wait::st.sync.aligned"()

        rmax = ptx"tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"(tmem_addr)
        rmin = ptx"tcgen05.ld.red.sync.aligned.32x32b.x64.min.f32"(tmem_addr)
        rabs = ptx"tcgen05.ld.red.sync.aligned.32x32b.x64.max.abs.f32"(tmem_addr)
        rmxu = ptx"tcgen05.ld.red.sync.aligned.32x32b.x64.max.u32"(tmem_addr)
        ptx"tcgen05.wait::ld.sync.aligned"()

        base = Int(lane) * 3
        @inbounds O[base + 1] = rmax[TLR_COLS + 1]
        @inbounds O[base + 2] = rmin[TLR_COLS + 1]
        @inbounds O[base + 3] = rabs[TLR_COLS + 1]
        @inbounds U[Int(lane) + 1] = rmxu[TLR_COLS + 1]

        # Data destinations must be exactly the stored row (statically
        # unrolled: runtime indexing a heterogeneous tuple goes dynamic).
        bad = reduce(|, ntuple(c -> rmax[c] ⊻ src[c], Val(TLR_COLS)))
        @inbounds U[32 + Int(lane) + 1] = bad
    end

    if tid < UInt32(32)
        ptx"tcgen05.dealloc.cta_group::1.sync.aligned.b32"(tmem_base,
                                                           UInt32(512))
        ptx"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"()
    end
    return nothing
end

@testset "tcgen05.ld.red probe compiles at sm_103f" begin
    types = Tuple{CuDeviceVector{Float32, 1}, CuDeviceVector{UInt32, 1}}
    @test ptxas_compiles(_tcgen05_ldred_kernel!, types;
                         cap = v"10.3", feature_set = :family)
    ptx = emit_ptx(_tcgen05_ldred_kernel!, types;
                   cap = v"10.3", feature_set = :family)
    @test occursin("tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32", ptx)
    @test occursin("tcgen05.ld.red.sync.aligned.32x32b.x64.min.f32", ptx)
    @test occursin("tcgen05.ld.red.sync.aligned.32x32b.x64.max.abs.f32", ptx)
    @test occursin("tcgen05.ld.red.sync.aligned.32x32b.x64.max.u32", ptx)
end

if test_runtime_supported(@__FILE__)
    @testset "tcgen05.ld.red reductions (B300 runtime)" begin
        O = CUDACore.zeros(Float32, 32 * 3)
        U = CUDACore.zeros(UInt32, 64)
        @cuda blocks=1 threads=128 _tcgen05_ldred_kernel!(O, U)
        CUDACore.synchronize()

        vals = Float32[_tlr_val(c) for c in 1:TLR_COLS]
        o = reshape(Array(O), 3, 32)
        u = Array(U)
        @test all(o[1, :] .== maximum(vals))            # max   =  64.0
        @test all(o[2, :] .== minimum(vals))            # min   = -163.0
        @test all(o[3, :] .== maximum(abs.(vals)))      # |max| =  163.0
        @test all(u[1:32] .== maximum(reinterpret.(UInt32, vals)))
        @test all(u[33:64] .== 0)   # data outputs equal the stored row
    end
end
