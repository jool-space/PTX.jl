# Exact-target compiler evidence for the schema-only mbarrier surface. These
# kernels stop after ptxas and never require a live matching GPU.

function _mbarrier_callsite_is_convergent(llvm::AbstractString,
                                           needle::AbstractString)
    groups = Dict{String,String}()
    for line in eachline(IOBuffer(llvm))
        m = match(r"^attributes #([0-9]+) = \{([^}]*)\}", strip(line))
        m === nothing || (groups[m.captures[1]] = m.captures[2])
    end
    calls = [String(line) for line in eachline(IOBuffer(llvm))
             if occursin(" call ", line) && occursin(needle, line)]
    length(calls) == 1 || return false
    m = match(r" #([0-9]+)(?:,|$)", strip(only(calls)))
    m === nothing && return false
    attrs = get(groups, m.captures[1], "")
    occursin(r"\bconvergent\b", attrs) && occursin(r"\bnomerge\b", attrs)
end

# Julia 1.10/1.11 use LLVM's typed-pointer IR, while newer LLVM prints opaque
# pointers. Pin the address space without making the evidence dialect-specific.
_mbarrier_has_as3_pointer(llvm::AbstractString) =
    occursin(r"(?:ptr addrspace\(3\)|i8 addrspace\(3\)\*)", llvm)

@testset "mbarrier LLVM AS3 pointer dialects" begin
    @test _mbarrier_has_as3_pointer("ptr addrspace(3)")
    @test _mbarrier_has_as3_pointer("i8 addrspace(3)*")
    @test !_mbarrier_has_as3_pointer("ptr addrspace(1)")
end

function _mbarrier_schema_sm80!(out64::CuDeviceVector{UInt64, 1},
                                 out32::CuDeviceVector{UInt32, 1},
                                 mbar::Core.LLVMPtr{UInt64, PTX.AS.Shared})
    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        # Both noComplete arrivals leave the phase open; the final ordinary
        # arrival is the only operation that completes it.
        ptx"mbarrier.init.shared::cta.b64"(mbar, UInt32(3))
        ptx"mbarrier.arrive.sink.noComplete.shared::cta.b64"(
            mbar, UInt32(1))
        state = ptx"mbarrier.arrive.noComplete.release.cta.shared::cta.b64"(
            mbar, UInt32(1))
        pending = ptx"mbarrier.pending_count.b64"(state)
        final_state = ptx"mbarrier.arrive.release.cta.shared::cta.b64"(mbar)
        complete = ptx"mbarrier.test_wait.acquire.cta.shared::cta.b64"(
            mbar, final_state)
        @inbounds out64[1] = state
        @inbounds out64[2] = final_state
        @inbounds out32[1] = pending
        @inbounds out32[2] = complete ? UInt32(1) : UInt32(0)
        ptx"mbarrier.inval.shared::cta.b64"(mbar)
    end
    return nothing
end

@testset "mbarrier exact schema at sm_80" begin
    types = Tuple{CuDeviceVector{UInt64, 1}, CuDeviceVector{UInt32, 1},
                  Core.LLVMPtr{UInt64, PTX.AS.Shared}}
    @test ptxas_compiles(_mbarrier_schema_sm80!, types; cap = v"8.0")
    ptx = emit_ptx(_mbarrier_schema_sm80!, types; cap = v"8.0")
    @test occursin(".target sm_80", ptx)
    @test occursin("mbarrier.init.shared::cta.b64", ptx)
    @test occursin(r"mbarrier\.arrive\.noComplete\.shared::cta\.b64\s+_,",
                   ptx)
    @test occursin("mbarrier.arrive.noComplete.release.cta.shared::cta.b64", ptx)
    @test occursin("mbarrier.pending_count.b64", ptx)
    @test occursin("mbarrier.arrive.release.cta.shared::cta.b64", ptx)
    @test occursin("mbarrier.test_wait.acquire.cta.shared::cta.b64", ptx)
    @test occursin("mbarrier.inval.shared::cta.b64", ptx)

    # The LLVM value remains addrspace(3), while the inline-asm constraint
    # deliberately selects the 32-bit shared-address PTX register class.
    llvm = emit_llvm(_mbarrier_schema_sm80!, types; cap = v"8.0")
    @test _mbarrier_has_as3_pointer(llvm)
    @test occursin("mbarrier.init.shared::cta.b64", llvm)
    @test occursin("r,r,~{memory}", llvm)
    @test _mbarrier_callsite_is_convergent(
        llvm, "mbarrier.init.shared::cta.b64")
end

@noinline function _mbarrier_intrinsic_convergence_probe(
        mbar::Core.LLVMPtr{UInt64, PTX.AS.Shared})
    ptx"mbarrier.init.shared.b64"(mbar, UInt32(1))
    return nothing
end

@testset "mbarrier intrinsic call-site convergence at sm_80" begin
    types = Tuple{Core.LLVMPtr{UInt64, PTX.AS.Shared}}
    llvm = emit_llvm(_mbarrier_intrinsic_convergence_probe, types;
                     cap = v"8.0")
    @test _mbarrier_callsite_is_convergent(
        llvm, "llvm.nvvm.mbarrier.init.shared")
    @test ptxas_compiles(_mbarrier_intrinsic_convergence_probe, types;
                         cap = v"8.0")
end

@noinline function _mbarrier_generic_remote_sink_probe(remote::UInt64)
    ptx"mbarrier.arrive.sink.release.cluster.b64"(remote)
    return nothing
end

@testset "mbarrier generic-address remote sink at sm_90" begin
    types = Tuple{UInt64}
    @test ptxas_compiles(_mbarrier_generic_remote_sink_probe, types;
                         cap = v"9.0")
    ptx = emit_ptx(_mbarrier_generic_remote_sink_probe, types; cap = v"9.0")
    @test occursin(r"mbarrier\.arrive\.release\.cluster\.b64\s+_,\s*\[%rd",
                   ptx)
end

function _mbarrier_schema_sm90!(out16::CuDeviceVector{UInt16, 1},
                                 out32::CuDeviceVector{UInt32, 1},
                                 mbar::Core.LLVMPtr{UInt64, PTX.AS.Shared},
                                 state::UInt64, phase::UInt32,
                                 hint::UInt32)
    ptx"mbarrier.init.layout::v1.shared::cta.b64"(mbar, UInt32(1))
    layout = ptx"mbarrier.check_layout.layout::v1.shared::cta.b64"(mbar)
    ptx"mbarrier.expect_tx.relaxed.cta.shared::cta.b64"(mbar, UInt32(16))
    ptx"mbarrier.complete_tx.relaxed.cta.shared::cta.b64"(mbar, UInt32(16))
    # PTX 9.3 prints these two space-before-sem/scope heads as examples.
    # The Julia aliases ingest them but emit the syntax-block canonical order.
    ptx"mbarrier.arrive_drop.sink.shared::cta.release.cluster.b64"(
        mbar, UInt32(1))
    ptx"mbarrier.arrive_drop.sink.expect_tx.shared::cta.relaxed.cluster.b64"(
        mbar, UInt32(0))
    wait_complete, report_pred =
        ptx"mbarrier.test_wait.report_pred.phase_type::primary.shared::cta.b64"(
            mbar, state)
    try_complete, try_report, report_value =
        ptx"mbarrier.try_wait.report.parity.phase_type::primary.relaxed.cta.shared::cta.b64"(
            mbar, phase, hint)
    conditional =
        ptx"mbarrier.try_wait.parity.phase_type::conditional.shared::cta.b64"(
            mbar, phase, hint)
    @inbounds out16[1] = report_value
    @inbounds out32[1] = layout ? UInt32(1) : UInt32(0)
    @inbounds out32[2] = wait_complete ? UInt32(1) : UInt32(0)
    @inbounds out32[3] = report_pred ? UInt32(1) : UInt32(0)
    @inbounds out32[4] = try_complete ? UInt32(1) : UInt32(0)
    @inbounds out32[5] = try_report ? UInt32(1) : UInt32(0)
    @inbounds out32[6] = conditional ? UInt32(1) : UInt32(0)
    return nothing
end

const _MBARRIER_SCHEMA_SM90_TYPES =
    Tuple{CuDeviceVector{UInt16, 1}, CuDeviceVector{UInt32, 1},
          Core.LLVMPtr{UInt64, PTX.AS.Shared}, UInt64, UInt32, UInt32}

@noinline function _mbarrier_raw_report!(out16::CuDeviceVector{UInt16, 1},
                                          out32::CuDeviceVector{UInt32, 1},
                                          mbar::Core.LLVMPtr{UInt64, PTX.AS.Shared},
                                          state::UInt64)
    complete, report_pred, report_value =
        ptx"mbarrier.test_wait.report.phase_type::primary.shared::cta.b64"raw(
            mbar, state)
    @inbounds out16[1] = report_value
    @inbounds out32[1] = complete ? UInt32(1) : UInt32(0)
    @inbounds out32[2] = report_pred ? UInt32(1) : UInt32(0)
    return nothing
end

const _MBARRIER_RAW_REPORT_TYPES =
    Tuple{CuDeviceVector{UInt16, 1}, CuDeviceVector{UInt32, 1},
          Core.LLVMPtr{UInt64, PTX.AS.Shared}, UInt64}

@testset "mbarrier layout/report schema at sm_90 and sm_121" begin
    for cap in (v"9.0", v"12.1")
        @test ptxas_compiles(_mbarrier_schema_sm90!,
                             _MBARRIER_SCHEMA_SM90_TYPES; cap)
        ptx = emit_ptx(_mbarrier_schema_sm90!,
                       _MBARRIER_SCHEMA_SM90_TYPES; cap)
        @test occursin(".target sm_$(cap.major)$(cap.minor)", ptx)
        @test occursin("mbarrier.init.layout::v1.shared::cta.b64", ptx)
        @test occursin("mbarrier.check_layout.layout::v1.shared::cta.b64", ptx)
        @test occursin("mbarrier.expect_tx.relaxed.cta.shared::cta.b64", ptx)
        @test occursin("mbarrier.complete_tx.relaxed.cta.shared::cta.b64", ptx)
        @test occursin(r"mbarrier\.arrive_drop\.release\.cluster\.shared::cta\.b64\s+_,",
                       ptx)
        @test occursin(r"mbarrier\.arrive_drop\.expect_tx\.relaxed\.cluster\.shared::cta\.b64\s+_,",
                       ptx)
        @test !occursin("mbarrier.arrive_drop.shared::cta.release.cluster", ptx)
        @test !occursin("mbarrier.arrive_drop.expect_tx.shared::cta.relaxed.cluster", ptx)
        @test occursin(r"mbarrier\.test_wait\.phase_type::primary\.shared::cta\.b64\s+%p\d+\|%p\d+",
                       ptx)
        @test occursin(r"mbarrier\.try_wait\.parity\.phase_type::primary\.relaxed\.cta\.shared::cta\.b64\s+%p\d+\|%p\d+,\s*report_value",
                       ptx)
        @test occursin(r"mov\.b16\s+%rs\d+,\s*\{report_value,\s*0\}", ptx)
        @test occursin("mbarrier.try_wait.parity.phase_type::conditional.shared::cta.b64",
                       ptx)
    end

    # Exercise exact raw through the real optimized GPUCompiler.code_llvm path,
    # rather than merely inspecting the string used to build Base.llvmcall.
    llvm = emit_llvm(_mbarrier_raw_report!, _MBARRIER_RAW_REPORT_TYPES;
                     cap = v"9.0")
    raw_sites = filter(split(llvm, '\n')) do line
        occursin("call { i8, i8, i16 } asm sideeffect", line) &&
            occursin(".reg .b8 report_value; mbarrier.test_wait", line)
    end
    @test length(raw_sites) == 1
    @test _mbarrier_has_as3_pointer(only(raw_sites))
    @test occursin("=b,=b,=h,r,l,~{memory}", only(raw_sites))
    @test _mbarrier_callsite_is_convergent(
        llvm, ".reg .b8 report_value; mbarrier.test_wait")
    @test ptxas_compiles(_mbarrier_raw_report!, _MBARRIER_RAW_REPORT_TYPES;
                         cap = v"9.0")
end
