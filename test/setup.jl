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
#
# Since CUDACore 6.2 the feature set is part of the target (`SMVersion`),
# not a separate compiler kwarg; the helpers keep the (cap, feature_set)
# signature so the ~100 call sites stay as they are.
function emit_ptx(f, tt::Type{<:Tuple};
                  cap::VersionNumber, feature_set::Symbol = :baseline)
    io = IOBuffer()
    arch = SMVersion(cap.major, cap.minor, feature_set)
    CUDATools.code_ptx(io, f, tt; arch, kernel = true)
    String(take!(io))
end

# Full LLVM → PTX → ptxas → cubin path; no `link`, so no device load.
# Throws on ptxas rejection (stderr is in the error message).
function ptxas_compiles(f, tt::Type{<:Tuple};
                        cap::VersionNumber, feature_set::Symbol = :baseline)
    source = methodinstance(typeof(f), Base.to_tuple_type(tt))
    arch = SMVersion(cap.major, cap.minor, feature_set)
    config = CUDACore.compiler_config(CUDACore.device();
                                      kernel = true, arch)
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

# --- Golden-PTX harness ------------------------------------------------------
#
# Locks emitted PTX for migration review (DESIGN.md, "Approach"): comparison
# is structural — parsed with the package's own parser, canonicalized modulo
# register/label/name numbering (IR.canonicalize) — so allocator churn never
# trips it, while any change to the instruction sequence does. Golden files
# live in test/golden/ and are committed; a deliberate lowering change
# regenerates them with PTX_UPDATE_GOLDEN=1 and the *git diff of the golden
# file* is the review artifact.

const GOLDEN_DIR = joinpath(@__DIR__, "golden")

# Pkg.test's default --check-bounds=yes overrides @inbounds in device code,
# injecting bounds branches the committed baselines don't have. Comparing in
# that state produces environmental mismatches; REGENERATING in that state
# would commit polluted goldens that then fail CI. Refuse both, loudly.
_forced_bounds_checks() = Base.JLOptions().check_bounds == 1

# A golden must be fully structural: a body RawLine keeps its original
# register numbers (escaping the modulo-renaming guarantee), and a top-level
# RawLine is dropped by normalize (escaping comparison entirely). Either way
# the harness silently weakens — a RawLine here means the parser needs
# extending, and that should be a red test, not a quiet degradation.
function _assert_structural(m::PTX.IR.Module, name::String)
    raws = String[]
    for d in m.directives
        d isa PTX.IR.RawLine && push!(raws, d.text)
        d isa PTX.IR.Function || continue
        for s in d.body
            s isa PTX.IR.RawLine && push!(raws, s.text)
        end
    end
    isempty(raws) && return nothing
    error("golden $name: emitted PTX contains $(length(raws)) line(s) the " *
          "parser could not parse structurally; raw lines bypass canonical " *
          "renaming. Extend the parser to cover them. First: " *
          repr(first(raws)))
end

canonical_ptx(f, tt::Type{<:Tuple}; cap::VersionNumber,
              feature_set::Symbol = :baseline) =
    PTX.IR.format(PTX.IR.canonicalize(PTX.Parser.parse(
        emit_ptx(f, tt; cap, feature_set))))

function golden_test(name::String, f, tt::Type{<:Tuple}; cap::VersionNumber,
                     feature_set::Symbol = :baseline)
    if _forced_bounds_checks()
        @error """golden_test($name): running under --check-bounds=yes (Pkg.test's default), \
                  which injects bounds branches into the golden kernels. Refusing to compare \
                  or regenerate — run `julia --project=test test/runtests.jl` or \
                  `Pkg.test("PTX"; julia_args=["--check-bounds=auto"])` instead \
                  (CI sets check_bounds: 'auto'). Default runs skip goldens in this mode; \
                  you selected this test explicitly."""
        return false
    end
    parsed = PTX.Parser.parse(emit_ptx(f, tt; cap, feature_set))
    _assert_structural(parsed, name)
    got = PTX.IR.format(PTX.IR.canonicalize(parsed))
    path = joinpath(GOLDEN_DIR, name * ".ptx")
    if get(ENV, "PTX_UPDATE_GOLDEN", "") == "1"
        mkpath(GOLDEN_DIR)
        write(path, got)
        @info "golden written — review the git diff" name path
        return true
    end
    if !isfile(path)
        # Goldens are committed review artifacts. A missing baseline must be
        # a failure — regenerate-on-absence would let a deleted golden pass
        # green and let a first golden land with zero review.
        @error "golden baseline missing — create it deliberately with PTX_UPDATE_GOLDEN=1 and review the git diff" name path
        return false
    end
    want = read(path, String)
    want == got && return true
    println("=== golden mismatch: $name ===")
    println("    (if this lowering change is INTENDED, regenerate with PTX_UPDATE_GOLDEN=1")
    println("     and review the git diff of the golden file)")
    wl, gl = split(want, "\n"), split(got, "\n")
    for i in 1:max(length(wl), length(gl))
        a = i <= length(wl) ? wl[i] : "<missing>"
        b = i <= length(gl) ? gl[i] : "<missing>"
        a == b || println("  golden: ", a, "\n  got:    ", b)
    end
    return false
end
