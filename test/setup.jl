# Shared helpers loaded by every test worker via runtests.jl's `init_code`.
#
# Three tiers of test live alongside each other:
#
#   host/   — pure host. No CUDA toolkit, no GPU.
#   ptxas/  — needs a functional CUDA install (toolkit + a device for
#             compiler_config(device(); ...)), but the `cap` we compile for
#             is independent of the device's actual capability. Validates
#             wrappers cross-arch (sm_50..sm_100a) without needing the
#             corresponding hardware.
#   gpu/    — real device execution via @cuda. Cap-gated by the
#             `# REQUIRES CC` banner (see runtests.jl).
#
# `emit_ptx` stops at the LLVM NVPTX backend (string-match only, no ptxas).
# `ptxas_compiles` runs LLVM → PTX → ptxas → cubin and stops before `link`,
# so the cubin is never loaded onto the device — meaning a sm_89 box can
# validate sm_90a or sm_100a wrapper output. ptxas's stderr surfaces in the
# thrown error when it rejects.

using CUDACore
using CUDATools
using CUDACore.GPUCompiler: methodinstance, CompilerJob

const DEV_CAP = CUDACore.functional() ?
                CUDACore.capability(CUDACore.device()) :
                v"0.0"

# LLVM NVPTX backend → PTX text. No ptxas, no driver. Compiled with
# kernel ABI so `kernel_state` intrinsics (e.g. ptx"mov.u32"(sreg"%tid.x"))
# resolve correctly.
function emit_ptx(f, tt::Type{<:Tuple};
                  cap::VersionNumber, feature_set::Symbol = :baseline)
    io = IOBuffer()
    CUDATools.code_ptx(io, f, tt; cap, feature_set, kernel = true)
    String(take!(io))
end

# Full LLVM → PTX → ptxas → cubin path; no `link`, so no device load.
# Throws on ptxas rejection (stderr is in the error message).
function ptxas_compiles(f, tt::Type{<:Tuple};
                        cap::VersionNumber, feature_set::Symbol = :baseline)
    source = methodinstance(typeof(f), Base.to_tuple_type(tt))
    config = CUDACore.compiler_config(CUDACore.device();
                                      kernel = true, cap, feature_set)
    job = CompilerJob(source, config)
    CUDACore.invoke_frozen(CUDACore.compile, job)
    true
end
