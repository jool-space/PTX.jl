# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=7.5
#
# nvvm"" semantics on hardware. The ptxas/ tier proves the right instruction
# is selected; this tier proves the values are right — warp data movement
# (shfl), predicate plumbing through the Bool/i1 glue (vote, shfl pred), the
# convergence shape from spikes/convergence.jl (spikes/ removed in ccdfb8a;
# `git show ccdfb8a~1:spikes/`), and the m8n8 fragment layout
# through the aggregate-return repack (ldmatrix).

using PTX.NVVM


@testset "shfl.sync.idx: rotate by one lane" begin
    function rotate!(out)
        tid = ptx"mov.u32"(sreg"tid.x")
        src = (tid + UInt32(1)) % UInt32(32)
        @inbounds out[tid + 1] =
            nvvm"shfl.sync.idx.i32"(0xffffffff, tid * tid, src, UInt32(0x1f))
        return nothing
    end
    out = CUDACore.zeros(UInt32, 32)
    @cuda threads=32 rotate!(out)
    @test Array(out) == UInt32[((t + 1) % 32)^2 for t in 0:31]
end

@testset "shfl.sync.idx.i32p: heterogeneous {i32,i1} return" begin
    function rotp!(vals, preds)
        tid = ptx"mov.u32"(sreg"tid.x")
        src = (tid + UInt32(1)) % UInt32(32)
        v, p = nvvm"shfl.sync.idx.i32p"(0xffffffff, tid, src, UInt32(0x1f))
        @inbounds vals[tid + 1] = v
        @inbounds preds[tid + 1] = p
        return nothing
    end
    vals = CUDACore.zeros(UInt32, 32)
    preds = CUDACore.zeros(Bool, 32)
    @cuda threads=32 rotp!(vals, preds)
    @test Array(vals) == UInt32[(t + 1) % 32 for t in 0:31]
    @test all(Array(preds))    # every source lane is in-bounds and active
end

@testset "vote.all.sync through the Bool/i1 glue" begin
    function votes!(out)
        tid = ptx"mov.u32"(sreg"tid.x")
        @inbounds out[tid + 1, 1] = nvvm"vote.all.sync"(0xffffffff, true)
        @inbounds out[tid + 1, 2] =
            nvvm"vote.all.sync"(0xffffffff, tid < UInt32(31))
        return nothing
    end
    out = CUDACore.zeros(Bool, 32, 2)
    @cuda threads=32 votes!(out)
    @test all(Array(out)[:, 1])
    @test !any(Array(out)[:, 2])
end

@testset "activemask under divergence (the convergence-spike shape)" begin
    function masks!(out)
        i = ptx"mov.u32"(sreg"tid.x")
        if i < UInt32(16)
            @inbounds out[i + 1] = nvvm"activemask"()
        else
            @inbounds out[i + 1] = nvvm"activemask"() ⊻ UInt32(1)
        end
        return nothing
    end
    out = CUDACore.zeros(UInt32, 32)
    @cuda threads=32 masks!(out)
    h = Array(out)
    @test h[1] == 0x0000ffff             # lower half-warp's own mask
    @test (h[17] ⊻ UInt32(1)) == 0xffff0000   # upper half-warp's own mask
end

@testset "ldmatrix x4: m8n8 fragment layout via aggregate repack" begin
    function ldm!(out)
        smem = CuStaticSharedArray(UInt16, 256)   # four 8x8 b16 matrices
        t = threadIdx().x                          # 1-based lane
        for j in 1:8                               # smem[i] = i - 1, row-major
            @inbounds smem[(t - 1) * 8 + j] = UInt16((t - 1) * 8 + j - 1)
        end
        sync_threads()
        # lane l supplies the address of 16-byte row l (lanes 0-7 matrix 0, …)
        r = nvvm"ldmatrix.sync.aligned.m8n8.x4.b16"(pointer(smem) + (t - 1) * 16)
        @inbounds for m in 1:4
            out[t, m] = r[m]
        end
        return nothing
    end
    out = CUDACore.zeros(UInt32, 32, 4)
    @cuda threads=32 ldm!(out)
    h = Array(out)
    # thread t (0-based), matrix m holds the b16 pair at row t÷4,
    # cols 2(t%4)..2(t%4)+1: elements 64m + 8(t÷4) + 2(t%4), +1 in the
    # high half (same oracle as spikes/aggregate_return.jl)
    expected(t, m) = let e = UInt32(64m + 8(t ÷ 4) + 2(t % 4))
        (e + UInt32(1)) << 16 | e
    end
    @test all(h[t + 1, m + 1] == expected(t, m) for t in 0:31, m in 0:3)
end
