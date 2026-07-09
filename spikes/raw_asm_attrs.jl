# Raw-asm attribute spike — hardware validation (GB10, 2026-06-12) that a
# convergent attribute on an inline-asm call site binds; the suite inherited
# the shape (test/ptxas/hopper.jl).
#
# `ptx"..."raw` promises sideeffect + convergent defaults. `@asmcall` can't
# attach callsite attributes, so raw emission must go through llvmcall IR like
# tier 2. Verify: a `convergent` attribute group on an inline-asm *call site*
# parses through Base.llvmcall, survives the optimized module, and the asm
# reaches the PTX intact.
#
# (Binding behavior was demonstrated by spikes/convergence.jl; LLVM's
# isConvergent() consults callsite attributes the same way. This spike checks
# the mechanism carries over to asm callsites. Note `sideeffect` alone already
# blocks the both-arms hoist shape — consistent with CUDA.jl 2e134a314 — so
# convergent's marginal value on raw asm is the duplication direction.)
#
# Run: julia --project=/tmp/convspike spikes/raw_asm_attrs.jl

using CUDACore
using GPUCompiler: CompilerJob, methodinstance
import GPUCompiler

@inline raw_activemask() = Base.llvmcall(("""
    define i32 @entry() #1 {
        %m = call i32 asm sideeffect "activemask.b32 \$0;", "=r"() #0
        ret i32 %m
    }
    attributes #0 = { convergent nounwind }
    attributes #1 = { alwaysinline }
    """, "entry"), UInt32, Tuple{})

function kernel!(out)
    i = threadIdx().x
    if i <= 16
        @inbounds out[i] = raw_activemask()
    else
        @inbounds out[i] = raw_activemask() ⊻ 0x00000001
    end
    return
end

source = methodinstance(typeof(kernel!), Tuple{CuDeviceVector{UInt32,1}})
config = CUDACore.compiler_config(CUDACore.device(); kernel=true)
job = CompilerJob(source, config)

llvm = sprint(io -> GPUCompiler.code_llvm(io, job; optimize=true, dump_module=true))
sites = count("call i32 asm sideeffect", llvm)
# the callsite attribute renders inline on the call in textual IR
attr_on_site = occursin(r"call i32 asm sideeffect [^\n]*#\d", llvm) &&
               occursin(r"attributes #\d+ = \{[^}]*convergent", llvm)
println("optimized IR: $sites asm call site(s); convergent attr group present: $attr_on_site")

ptx = sprint(io -> GPUCompiler.code_native(io, job))
println("emitted PTX: ", count("activemask.b32", ptx), " activemask.b32 instruction(s)")

out = CUDACore.zeros(UInt32, 32)
@cuda threads=32 kernel!(out)
h = Array(out)
println("hardware: lower=", string(h[1], base=16), " upper=", string(h[17] ⊻ 0x1, base=16))

ok = sites == 2 && attr_on_site && h[1] == 0x0000ffff && (h[17] ⊻ 0x1) == 0xffff0000
println(ok ? "RESULT: raw-asm attribute mechanism VALIDATED" : "RESULT: needs investigation")
