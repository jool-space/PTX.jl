# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=9.0
#
# mbarrier semantics on hardware after the tier-2 migration: a full
# init → arrive → wait cycle through the notation surface, exercising the
# state-token form (test_wait on the arrive token), the suspend form
# (try_wait), and the parity form. The big mbarrier consumers (TMA/tcgen05
# pipeline kernels) cover the production shapes; this pins the primitive
# contract in isolation.

@testset "mbarrier: init -> arrive -> wait roundtrip" begin
    function roundtrip!(out)
        bar = CuStaticSharedArray(Int64, 1)
        mbar = pointer(bar)
        t = threadIdx().x
        if t == 1
            ptx"mbarrier.init.shared.b64"(mbar, UInt32(32))
        end
        sync_threads()
        state = ptx"mbarrier.arrive.shared.b64"(mbar)
        # all 32 arrived -> the phase the token belongs to completes
        done = false
        for _ in 1:1_000_000
            done = ptx"mbarrier.test_wait.shared.b64"(mbar, state)
            done && break
        end
        @inbounds out[t, 1] = done
        # the completed phase has parity 0; try_wait on it returns true
        ok = false
        for _ in 1:1_000_000
            ok = ptx"mbarrier.try_wait.parity.shared.b64"(mbar, UInt32(0))
            ok && break
        end
        @inbounds out[t, 2] = ok
        sync_threads()
        if t == 1
            ptx"mbarrier.inval.shared.b64"(mbar)
        end
        return nothing
    end
    out = CUDACore.zeros(Bool, 32, 2)
    @cuda threads=32 roundtrip!(out)
    h = Array(out)
    @test all(h[:, 1])    # every token-form wait completed
    @test all(h[:, 2])    # every parity-form wait completed
end

@testset "mbarrier: noComplete arrive keeps the phase open" begin
    function nocomplete!(out)
        bar = CuStaticSharedArray(Int64, 1)
        mbar = pointer(bar)
        t = threadIdx().x
        if t == 1
            ptx"mbarrier.init.shared.b64"(mbar, UInt32(63))
            # 31 pending arrivals consumed without closing the phase
            s = ptx"mbarrier.arrive.noComplete.shared.b64"(mbar, UInt32(31))
            @inbounds out[33] = !ptx"mbarrier.test_wait.shared.b64"(mbar, s)
        end
        sync_threads()
        state = ptx"mbarrier.arrive.shared.b64"(mbar)   # 32 more: 31+32 = 63
        done = false
        for _ in 1:1_000_000
            done = ptx"mbarrier.test_wait.shared.b64"(mbar, state)
            done && break
        end
        @inbounds out[t] = done
        return nothing
    end
    out = CUDACore.zeros(Bool, 33)
    @cuda threads=32 nocomplete!(out)
    h = Array(out)
    @test all(h[1:32])    # phase closed once the full count arrived
    @test h[33]           # ...and was still open right after noComplete
end

@testset "mbarrier: layout v1 primary report ABI" begin
    function report!(flags, status)
        bar = CuStaticSharedArray(Int64, 1)
        mbar = pointer(bar)
        t = threadIdx().x
        if t == 1
            ptx"mbarrier.init.layout::v1.shared.b64"(mbar, UInt32(32))
        end
        sync_threads()
        state = ptx"mbarrier.arrive.shared.b64"(mbar)
        complete = false
        report_pred = true
        # Public carrier is UInt16; PTX writes the opaque reportValue as b8
        # and the wrapper zero-extends that byte because NVPTX has no i8
        # inline-asm constraint.
        report_value = typemax(UInt16)
        for _ in 1:1_000_000
            complete, report_pred, report_value =
                ptx"mbarrier.test_wait.report.phase_type::primary.shared.b64"(
                    mbar, state)
            complete && break
        end
        @inbounds flags[t, 1] = complete
        @inbounds flags[t, 2] = report_pred
        @inbounds status[t] = report_value
        sync_threads()
        if t == 1
            ptx"mbarrier.inval.shared.b64"(mbar)
        end
        return nothing
    end
    flags = CUDACore.zeros(Bool, 32, 2)
    status = CUDACore.zeros(UInt16, 32)
    @cuda threads=32 report!(flags, status)
    hflags, hstatus = Array(flags), Array(status)
    @test all(hflags[:, 1])
    @test !any(hflags[:, 2])
    @test all(iszero, hstatus)
end
