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

# --- Hopper kernel test helpers ---------------------------------------------
# Patterns repeated 3+ times across the test/gpu/hopper/*.jl kernels.

# `f32 → bf16` (round-to-nearest-even) and `bf16 → f32` reinterpretations
# used by every bf16 kernel's host-side reference + tile-pack code.
bf16_bits(x::Float32) = UInt16((reinterpret(UInt32, x) + UInt32(0x8000)) >> 16)
bf16_to_f32(b::UInt16) = reinterpret(Float32, UInt32(b) << 16)

# Host → device upload for a TMA descriptor. Allocates a 128-byte device
# blob, copies the encoded descriptor into it, and returns a
# `(ptr, blob)` NamedTuple. The caller MUST bind the NamedTuple to a
# named variable for the lifetime of the kernel launch — `blob` is the
# `CuArray` that owns the device memory, and the LLVMPtr does not keep
# it alive on its own. Pattern:
#
#     A = upload_tma_descriptor(tmap_A)
#     B = upload_tma_descriptor(tmap_B)
#     @cuda kernel!(D, A.ptr, B.ptr)        # A and B alive through @cuda
#
function upload_tma_descriptor(tmap::PTX.CuTensorMap)
    blob = CuArray{UInt8}(undef, 128)
    copyto!(blob, collect(tmap.data))
    ptr = reinterpret(PTX.TMADescriptorPtr, UInt64(pointer(blob)))
    return (; ptr, blob)
end

# bf16-round-tripping triple-loop matmul. Mirrors what a bf16-tile kernel
# actually computes: round both inputs to bf16, then accumulate in f32.
# Used 4× in the Hopper GEMM/FA reference paths.
function bf16_gemm_ref(A::Array{Float32, 2}, B::Array{Float32, 2})
    M, K = size(A); _, N = size(B)
    Ab = bf16_to_f32.(bf16_bits.(A))
    Bb = bf16_to_f32.(bf16_bits.(B))
    D = zeros(Float32, M, N)
    for m in 1:M, n in 1:N, k in 1:K
        @inbounds D[m, n] += Ab[m, k] * Bb[k, n]
    end
    return D
end
