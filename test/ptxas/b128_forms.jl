# Offline LLVM/PTX/ptxas evidence for each target partition of the audited
# scalar-b128 ABI. No kernel is loaded onto a device.

function _b128_sm75!(ptr::Core.LLVMPtr{UInt64,PTX.AS.Global},
                     lo::UInt64, hi::UInt64)
    value = ptx"mov.b128"((lo, hi))
    ptx"st.global.b128"(ptr, value)
    ptx"ld.global.b128"(ptr)
    ptx"ldu.global.b128"(ptr)
    return nothing
end

function _b128_sm80!(ptr::Core.LLVMPtr{UInt64,PTX.AS.Global},
                     policy::UInt64)
    ptx"ld.global.L2::cache_hint.b128"(ptr, policy)
    ptx"ld.global.L2::256B.b128"(ptr)
    return nothing
end

function _b128_sm90!(ptr::Core.LLVMPtr{UInt64,PTX.AS.Global},
                     lo::UInt64, hi::UInt64)
    value = (lo, hi)
    ptx"atom.global.exch.b128"(ptr, value)
    ptx"atom.acq_rel.sys.global.cas.b128"(ptr, value, value)
    return nothing
end

function _b128_sm100!(lo::UInt64, hi::UInt64)
    value = (lo, hi)
    ptx"clusterlaunchcontrol.query_cancel.is_canceled.pred.b128"(value)
    ptx"clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128"(value)
    ptx"clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128"(value)
    return nothing
end

@testset "scalar-b128 LLVM/PTX/ptxas target partitions" begin
    cases = (
        # The ISA floor is sm_70. CUDA 13.3 ptxas has retired sm_70, so the
        # earliest retained executable oracle is sm_75.
        (_b128_sm75!, Tuple{Core.LLVMPtr{UInt64,PTX.AS.Global},UInt64,UInt64},
         v"7.5", ("mov.b128", "ld.global.b128", "st.global.b128", "ldu.global.b128")),
        (_b128_sm80!, Tuple{Core.LLVMPtr{UInt64,PTX.AS.Global},UInt64},
         v"8.0", ("L2::cache_hint.b128", "L2::256B.b128")),
        (_b128_sm90!, Tuple{Core.LLVMPtr{UInt64,PTX.AS.Global},UInt64,UInt64},
         v"9.0", ("atom.global.exch.b128", "atom.acq_rel.sys.global.cas.b128")),
        (_b128_sm100!, Tuple{UInt64,UInt64}, v"10.0",
         ("query_cancel.is_canceled.pred.b128",
          "query_cancel.get_first_ctaid.v4.b32.b128",
          "query_cancel.get_first_ctaid::x.b32.b128")),
    )
    for (kernel, types, cap, heads) in cases
        @test ptxas_compiles(kernel, types; cap, feature_set = :baseline)
        llvm = emit_llvm(kernel, types; cap, feature_set = :baseline)
        ptx = emit_ptx(kernel, types; cap, feature_set = :baseline)
        @test occursin(".reg .b128", llvm)
        @test occursin("mov.b128", llvm)
        for head in heads
            @test occursin(head, ptx)
        end
    end
end

function _b128_raw_ptxas(source::String, target::String)
    mktempdir() do dir
        ptx_path = joinpath(dir, "b128_raw.ptx")
        cubin_path = joinpath(dir, "b128_raw.cubin")
        write(ptx_path, source)
        cmd = `$(CUDACore.CUDA_Compiler.ptxas()) --gpu-name $target --output-file $cubin_path $ptx_path`
        log = IOBuffer()
        proc = run(pipeline(ignorestatus(cmd), stdout = log, stderr = log))
        (; accepted = success(proc), log = String(take!(log)))
    end
end

@testset "PTX 9.3 b128 mmio qualifier cells" begin
    # PTX 9.3 §9.7.9.8/.11 added acquire/release with mmio. The b128 sys
    # combination itself dates to PTX 8.4; compile the exact syntax directly
    # because an sm_75 backend job need not choose a 9.3 module directive.
    source = """.version 9.3
    .target sm_75
    .address_size 64
    .visible .entry b128_mmio(.param .u64 p) {
      .reg .u64 a;
      .reg .b128 x;
      ld.param.u64 a, [p];
      ld.mmio.acquire.sys.global.b128 x, [a];
      st.mmio.release.sys.global.b128 [a], x;
      ret;
    }
    """
    result = _b128_raw_ptxas(source, "sm_75")
    @test result.accepted
end
