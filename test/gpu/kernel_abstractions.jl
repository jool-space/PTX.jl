# REQUIRES CC 7.0
#
# Verifies PTX.jl's `ptx"..."` and `sreg"..."` macros survive
# KernelAbstractions.jl's `@kernel` lowering on the CUDA backend.
# KA wraps the body in its own launch shim, but on CUDABackend that shim
# still goes through GPUCompiler → NVPTX → ptxas — the same pipeline
# `@cuda` uses — so the inline-PTX wrappers should pass through unchanged.

using KernelAbstractions
using CUDACore.CUDAKernels: CUDABackend

const KA_BACKEND = CUDABackend()

# --- ptx"add.f32" inside @kernel, indexed via KA's portable @index ----------

@kernel function _ka_add_kernel!(out, a, b)
    i = @index(Global)
    @inbounds out[i] = ptx"add.f32"(a[i], b[i])
end

@testset "KA @kernel + ptx\"add.f32\" (global index via @index)" begin
    n = 64
    a = CuArray(Float32.(1:n))
    b = CuArray(Float32.(n:-1:1))
    out = CUDACore.zeros(Float32, n)

    _ka_add_kernel!(KA_BACKEND)(out, a, b; ndrange = n)
    KernelAbstractions.synchronize(KA_BACKEND)

    @test Array(out) == Float32.(1:n) .+ Float32.(n:-1:1)
end

# --- sreg"%tid.x" inside @kernel (bypassing @index) -------------------------
# Each thread reads its own hardware tid via the PTX special register and
# writes it into the output slot. Confirms the sreg macro lowers cleanly
# even when KA's @index machinery isn't in the picture.

@kernel function _ka_tid_kernel!(out)
    tid = ptx"mov.u32"(sreg"%tid.x")
    @inbounds out[tid + 1] = tid
end

@testset "KA @kernel + sreg\"%tid.x\" (raw hardware tid)" begin
    n = 32
    out = CUDACore.zeros(UInt32, n)

    _ka_tid_kernel!(KA_BACKEND, n)(out; ndrange = n)
    KernelAbstractions.synchronize(KA_BACKEND)

    @test Array(out) == UInt32.(0:n-1)
end
