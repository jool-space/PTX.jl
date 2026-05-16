# Blackwell tcgen05 TMEM roundtrip — write a known per-lane pattern into
# TMEM with tcgen05.st, drain with wait::st, read it back with tcgen05.ld,
# and store to global memory. Ported from pyptx
# examples/blackwell/tcgen05_roundtrip.py.
#
# This isolates TMEM addressing + the st/ld register-vector mapping from
# UMMA correctness: after the roundtrip every active lane's column `c`
# must read back the value it wrote (c + 1). It is the numeric counterpart
# to the tcgen05_smoke kernels (which only check "did it launch").
#
# SMEM is just the 16-byte tcgen05.alloc slot (static, no >48 KiB opt-in).
# Always validates cross-arch (sm_100a ptxas); runtime gated on Blackwell
# datacenter hw (CC >= 10.0) — B300 (sm_103a) is the cloud target.

using PTX: smem_addr_u32
using CUDACore

const TCR_ROWS = 32
const TCR_COLS = 64

function _tcgen05_roundtrip_kernel!(O::CuDeviceVector{Float32, 1})
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
    # Per-lane TMEM row: lane index in bits [20:16] (pyptx tmem_roundtrip).
    tmem_addr = tmem_base + ((lane << UInt32(16)) & UInt32(0x1F0000))

    # src[c] = bitcast(Float32(c)) for c = 1..COLS  (pyptx: col+1, col 0-based).
    src = ntuple(c -> reinterpret(UInt32, Float32(c)), Val(TCR_COLS))

    if tid < UInt32(TCR_ROWS)
        ptx"tcgen05.st.sync.aligned.32x32b.x64.b32"(tmem_addr, src)
        ptx"tcgen05.wait::st.sync.aligned"()
        dst = ptx"tcgen05.ld.sync.aligned.32x32b.x64.b32"(tmem_addr)

        base = Int(tid) * TCR_COLS
        for c in 1:TCR_COLS
            @inbounds O[base + c] = reinterpret(Float32, dst[c])
        end
    end

    if tid < UInt32(32)
        ptx"tcgen05.dealloc.cta_group::1.sync.aligned.b32"(tmem_base, UInt32(512))
        ptx"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"()
    end
    return nothing
end

@testset "tcgen05 TMEM roundtrip compiles at sm_100a" begin
    types = Tuple{CuDeviceVector{Float32, 1}}
    @test ptxas_compiles(_tcgen05_roundtrip_kernel!, types;
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_tcgen05_roundtrip_kernel!, types;
                   cap = v"10.0", feature_set = :arch)
    @test occursin("tcgen05.st.sync.aligned.32x32b.x64.b32", ptx)
    @test occursin("tcgen05.wait::st.sync.aligned", ptx)
    @test occursin("tcgen05.ld.sync.aligned.32x32b.x64.b32", ptx)
end

# Datacenter-Blackwell only [10.0, 11.0); see tcgen05_smoke.jl for the
# rationale (excludes consumer sm_120/sm_121).
if v"10.0" <= DEV_CAP < v"11.0"
    @testset "tcgen05 TMEM roundtrip (B300 runtime)" begin
        O = CUDACore.zeros(Float32, TCR_ROWS * TCR_COLS)
        @cuda blocks=1 threads=128 feature_set=:arch _tcgen05_roundtrip_kernel!(O)
        CUDACore.synchronize()

        got = reshape(Array(O), TCR_COLS, TCR_ROWS)   # row-major (row, col)
        # Every active lane reads back exactly what it wrote: col c → c.
        ref = Float32[c for c in 1:TCR_COLS, _ in 1:TCR_ROWS]
        @test got == ref
    end
end
