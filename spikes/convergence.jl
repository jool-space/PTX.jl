# Convergence spike — resolves the gating concerns in CONCERNS.md.
#
# Question: do hand-written attribute groups on unknown-intrinsic declarations
# (the tier-2 emission strategy) survive Julia's llvmcall splicing and
# GPUCompiler's optimization pipeline, and does `convergent` actually bind?
#
# Subject: llvm.nvvm.activemask — unknown to the in-process LLVM 18 (added
# upstream in 20), known to the external llc 22, and semantically observable:
# threads in a divergent branch arm must see their arm's mask, not the warp's.
#
# Shape: the identical call leads BOTH arms of a divergent branch. SimplifyCFG
# hoists identical leading instructions regardless of side effects (executing
# once before the branch is trace-equivalent per thread), so `convergent` is
# the only thing that can block the merge. The unattributed variant is the
# test's teeth: if it doesn't get hoisted, the test proves nothing.
#
# The two variants are deliberately used by SEPARATE kernels: both declare
# @llvm.nvvm.activemask, and linking them into one module would merge the
# declarations and confound the attribute comparison. (Mixing them on purpose
# is the follow-up attribute-merging experiment, not this one.)
#
# Run: julia --project=/tmp/convspike spikes/convergence.jl

using CUDACore
using GPUCompiler: CompilerJob, methodinstance
import GPUCompiler

# reflection lives in CUDATools in the 6.x split; build the jobs directly
function cuda_job(kernel, tt)
    source = methodinstance(typeof(kernel), Base.to_tuple_type(tt))
    config = CUDACore.compiler_config(CUDACore.device(); kernel=true)
    CompilerJob(source, config)
end

@inline activemask_conv() = Base.llvmcall(("""
    declare i32 @llvm.nvvm.activemask() #0
    define i32 @entry() #1 {
        %m = call i32 @llvm.nvvm.activemask()
        ret i32 %m
    }
    attributes #0 = { convergent nounwind memory(none) }
    attributes #1 = { alwaysinline }
    """, "entry"), UInt32, Tuple{})

@inline activemask_noconv() = Base.llvmcall(("""
    declare i32 @llvm.nvvm.activemask() #0
    define i32 @entry() #1 {
        %m = call i32 @llvm.nvvm.activemask()
        ret i32 %m
    }
    attributes #0 = { nounwind memory(none) }
    attributes #1 = { alwaysinline }
    """, "entry"), UInt32, Tuple{})

# Both arms lead with the identical call; the follow-up differs (xor tag) so
# the branch itself can't be folded away entirely.
function kernel_conv!(out)
    i = threadIdx().x
    if i <= 16
        @inbounds out[i] = activemask_conv()
    else
        @inbounds out[i] = activemask_conv() ⊻ 0x00000001
    end
    return
end

function kernel_noconv!(out)
    i = threadIdx().x
    if i <= 16
        @inbounds out[i] = activemask_noconv()
    else
        @inbounds out[i] = activemask_noconv() ⊻ 0x00000001
    end
    return
end

function report(name, kernel)
    println("="^70)
    println(name)
    println("="^70)

    job = cuda_job(kernel, Tuple{CuDeviceVector{UInt32,1}})
    llvm = sprint() do io
        GPUCompiler.code_llvm(io, job; optimize=true, dump_module=true)
    end
    calls = count("call i32 @llvm.nvvm.activemask", llvm)
    hasattr = occursin(r"declare .*@llvm\.nvvm\.activemask\(\) #", llvm) &&
              occursin("convergent", llvm)
    println("optimized IR: $calls call site(s); convergent in module: $hasattr")
    for l in eachline(IOBuffer(llvm))
        if occursin("activemask", l) || occursin("attributes", l)
            println("    ", strip(l))
        end
    end

    ptx = sprint() do io
        GPUCompiler.code_native(io, job)
    end
    ptxcalls = count("activemask.b32", ptx)
    println("emitted PTX: $ptxcalls activemask.b32 instruction(s)")

    out = CUDACore.zeros(UInt32, 32)
    @cuda threads=32 kernel(out)
    h = Array(out)
    println("hardware (warp of 32, split 16/16):")
    println("    lower arm mask: ", string(h[1], base=16), "  (expect 0000ffff if not merged)")
    println("    upper arm mask: ", string(h[17] ⊻ 0x1, base=16), "  (expect ffff0000 if not merged)")
    return (; calls, ptxcalls, lower=h[1], upper=h[17] ⊻ 0x1)
end

conv   = report("WITH convergent", kernel_conv!)
noconv = report("WITHOUT convergent", kernel_noconv!)

println("="^70)
println("VERDICT")
println("="^70)
teeth = noconv.calls < 2 || noconv.lower == 0xffffffff
held  = conv.calls == 2 && conv.lower == 0x0000ffff && conv.upper == 0xffff0000
println("unattributed variant misbehaves (test has teeth): ", teeth)
println("convergent variant survives and binds:            ", held)
println(held && teeth  ? "RESULT: tier-2 emission strategy VALIDATED" :
        held && !teeth ? "RESULT: attribute survives, but optimizer never attacked — INCONCLUSIVE" :
                         "RESULT: convergent did NOT protect — architecture rethink needed")
