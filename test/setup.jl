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
# Locks emitted PTX for migration review: comparison
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

# --- Structural-IR oracle helpers ------------------------------------------
#
# `IR.unraw` deliberately clears only the module-level raw source. It is a
# production formatting operation, not a claim that every parsed statement can
# be reconstructed from its fields. The corpus and golden oracles need the
# stronger test-only projection below: remove every raw escape while retaining
# every node, including RawLine fallback nodes. That way a RawLine is visible to
# the test instead of being quietly preserved as source text.

function _without_raw_line(fi::Union{PTX.IR.FormattingInfo, Nothing})
    fi === nothing && return nothing
    PTX.IR.FormattingInfo(
        indent = fi.indent,
        trailing = fi.trailing,
        blank_lines_before = fi.blank_lines_before,
        preceding_comments = fi.preceding_comments,
        raw_line = nothing,
    )
end

_deep_unraw_statements(stmts::Tuple{Vararg{PTX.IR.Statement}}) =
    Tuple(_deep_unraw_stmt(stmt) for stmt in stmts)

# Leaf statements without formatting (including RawLine) are intentionally
# retained. A structural test must reject fallback nodes explicitly, rather
# than erase them while constructing its test input.
_deep_unraw_stmt(stmt::PTX.IR.Statement) = stmt

function _deep_unraw_stmt(stmt::PTX.IR.Instruction)
    PTX.IR.Instruction(
        opcode = stmt.opcode,
        modifiers = stmt.modifiers,
        operands = stmt.operands,
        predicate = stmt.predicate,
        formatting = _without_raw_line(stmt.formatting),
    )
end

function _deep_unraw_stmt(stmt::PTX.IR.Label)
    PTX.IR.Label(name = stmt.name,
                 formatting = _without_raw_line(stmt.formatting))
end

function _deep_unraw_stmt(stmt::PTX.IR.RegDecl)
    PTX.IR.RegDecl(
        type = stmt.type,
        name = stmt.name,
        count = stmt.count,
        formatting = _without_raw_line(stmt.formatting),
    )
end

function _deep_unraw_stmt(stmt::PTX.IR.VarDecl)
    PTX.IR.VarDecl(
        state_space = stmt.state_space,
        type = stmt.type,
        name = stmt.name,
        array_size = stmt.array_size,
        alignment = stmt.alignment,
        initializer = stmt.initializer,
        linking = stmt.linking,
        formatting = _without_raw_line(stmt.formatting),
    )
end

function _deep_unraw_stmt(stmt::PTX.IR.PragmaDirective)
    PTX.IR.PragmaDirective(value = stmt.value,
                            formatting = _without_raw_line(stmt.formatting))
end

function _deep_unraw_stmt(stmt::PTX.IR.Block)
    PTX.IR.Block(
        body = _deep_unraw_statements(stmt.body),
        formatting = _without_raw_line(stmt.formatting),
    )
end

function _deep_unraw_stmt(stmt::PTX.IR.IntrinsicScope)
    PTX.IR.IntrinsicScope(
        name = stmt.name,
        args_repr = stmt.args_repr,
        body = _deep_unraw_statements(stmt.body),
        formatting = _without_raw_line(stmt.formatting),
    )
end

function _deep_unraw_stmt(stmt::PTX.IR.Function)
    PTX.IR.Function(
        is_entry = stmt.is_entry,
        name = stmt.name,
        params = stmt.params,
        return_params = stmt.return_params,
        body = _deep_unraw_statements(stmt.body),
        linking = stmt.linking,
        directives = stmt.directives,
        formatting = _without_raw_line(stmt.formatting),
    )
end

function _deep_unraw(m::PTX.IR.Module)
    PTX.IR.Module(
        version = m.version,
        target = m.target,
        address_size = m.address_size,
        leading = _deep_unraw_statements(m.leading),
        directives = _deep_unraw_statements(m.directives),
        raw_header = nothing,
        raw_source = nothing,
    )
end

function _raw_snapshot_paths!(paths::Vector{Pair{String, String}},
                              stmts::Tuple{Vararg{PTX.IR.Statement}},
                              prefix::String)
    for (index, stmt) in enumerate(stmts)
        path = "$prefix[$index]"
        if hasfield(typeof(stmt), :formatting)
            formatting = getfield(stmt, :formatting)
            formatting !== nothing && formatting.raw_line !== nothing &&
                push!(paths, path => formatting.raw_line)
        end
        if stmt isa PTX.IR.Function || stmt isa PTX.IR.Block ||
           stmt isa PTX.IR.IntrinsicScope
            _raw_snapshot_paths!(paths, stmt.body, path * ".body")
        end
    end
    paths
end

"""Return every per-statement `FormattingInfo.raw_line`, recursively."""
function _raw_snapshot_paths(m::PTX.IR.Module)
    paths = Pair{String, String}[]
    _raw_snapshot_paths!(paths, m.leading, "module.leading")
    _raw_snapshot_paths!(paths, m.directives, "module.directives")
    paths
end

function _rawline_paths!(paths::Vector{Pair{String, String}},
                         stmts::Tuple{Vararg{PTX.IR.Statement}},
                         prefix::String)
    for (index, stmt) in enumerate(stmts)
        path = "$prefix[$index]"
        if stmt isa PTX.IR.RawLine
            push!(paths, path => stmt.text)
        elseif stmt isa PTX.IR.Function
            _rawline_paths!(paths, stmt.body, path * ".body")
        elseif stmt isa PTX.IR.Block
            _rawline_paths!(paths, stmt.body, path * ".body")
        elseif stmt isa PTX.IR.IntrinsicScope
            _rawline_paths!(paths, stmt.body, path * ".body")
        end
    end
    paths
end

"""Return every `RawLine` fallback as `path => text`, recursively."""
function _rawline_paths(m::PTX.IR.Module)
    paths = Pair{String, String}[]
    _rawline_paths!(paths, m.leading, "module.leading")
    _rawline_paths!(paths, m.directives, "module.directives")
    paths
end

_rawline_count(m::PTX.IR.Module) = length(_rawline_paths(m))

function _assert_no_rawlines(m::PTX.IR.Module, context::AbstractString)
    raws = _rawline_paths(m)
    isempty(raws) && return nothing
    nshow = min(length(raws), 3)
    examples = join(("$(first(raws[i])) = $(repr(last(raws[i])))"
                     for i in 1:nshow), "; ")
    error("$context: parser left $(length(raws)) RawLine fallback node(s); " *
          "they bypass structural reconstruction. First path(s): $examples")
end

# A golden must be fully structural: a RawLine retains uncanonicalized source
# text, while canonicalization can otherwise omit some container detail. Do not
# let any fallback node at any nesting depth weaken the comparison.
function _assert_structural(m::PTX.IR.Module, name::String)
    _assert_no_rawlines(m, "golden $name")
    nothing
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
                  or regenerate — run `Pkg.test("PTX"; julia_args=["--check-bounds=auto"])` instead \
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
