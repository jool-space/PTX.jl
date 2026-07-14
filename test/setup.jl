# Shared helpers loaded by every test worker via runtests.jl's `init_code`.
#
# Three tiers of test live alongside each other:
#
#   host/   — pure host. No CUDA toolkit, no GPU.
#   ptxas/  — needs the CUDA compiler/ptxas artifact, but no live GPU.
#             `compiler_config(nothing; arch=...)` validates wrappers across
#             baseline, family, and architecture-specific targets without
#             device discovery or corresponding hardware.
#   gpu/    — active-device compilation and/or real execution. Routed by the
#             structured `# TEST_TARGET:` banner (see runtests.jl).
#
# `emit_ptx` stops at the LLVM NVPTX backend (string-match only, no ptxas).
# `ptxas_compiles` runs LLVM → PTX → ptxas → cubin and stops before `link`,
# so the cubin is never loaded onto a device — meaning a host with no visible
# GPU can validate sm_90a or sm_100a wrapper output. ptxas's stderr surfaces
# in the thrown error when it rejects.

using CUDACore
using CUDATools
using CUDACore.GPUCompiler: methodinstance, CompilerJob

isdefined(@__MODULE__, :TestTargets) ||
    include(joinpath(@__DIR__, "target_requirements.jl"))

const _TEST_DEVICE_CAP = Ref{Union{Nothing,VersionNumber}}(nothing)

# Avoid a redundant `functional()` / device-capability query in workers that
# never reach a mixed file's optional runtime section. CUDACore itself is
# still imported by the shared harness and performs its normal initialization.
function _test_device_capability()
    cap = _TEST_DEVICE_CAP[]
    cap === nothing || return cap
    cap = CUDACore.functional() ?
          CUDACore.capability(CUDACore.device()) :
          v"0.0"
    _TEST_DEVICE_CAP[] = cap
    cap
end

# Mixed gpu/ files always run their cross-target ptxas testsets.  Their
# optional runtime testsets use the same parsed target policy as the root
# runner, eliminating ad-hoc capability ranges while retaining per-file scope.
test_runtime_supported(file::AbstractString) =
    TestTargets.runtime_supported(file, _test_device_capability())

# LLVM NVPTX backend → PTX text. No ptxas, no driver. Compiled with
# kernel ABI so `kernel_state` intrinsics (e.g. ptx"mov.u32"(sreg"%tid.x"))
# resolve correctly.
#
# Since CUDACore 6.2 the feature set is part of the target (`SMVersion`),
# not a separate compiler kwarg; the helpers keep the (cap, feature_set)
# signature so the ~100 call sites stay as they are.
function _explicit_target_job(f, tt::Type{<:Tuple};
                              cap::VersionNumber,
                              feature_set::Symbol = :baseline)
    source = methodinstance(typeof(f), Base.to_tuple_type(tt))
    arch = SMVersion(cap.major, cap.minor, feature_set)
    config = CUDACore.compiler_config(nothing; kernel = true, arch)
    CompilerJob(source, config)
end

function emit_ptx(f, tt::Type{<:Tuple};
                  cap::VersionNumber, feature_set::Symbol = :baseline)
    io = IOBuffer()
    job = _explicit_target_job(f, tt; cap, feature_set)
    CUDACore.invoke_frozen(CUDACore.GPUCompiler.code_native, io, job)
    String(take!(io))
end

# Optimized LLVM for attribute/code-motion assertions, using the same
# device-free explicit target as PTX emission. `dump_module=true` keeps
# call-site attribute groups visible to the test oracle.
function emit_llvm(f, tt::Type{<:Tuple};
                   cap::VersionNumber, feature_set::Symbol = :baseline)
    io = IOBuffer()
    job = _explicit_target_job(f, tt; cap, feature_set)
    CUDACore.invoke_frozen(CUDACore.GPUCompiler.code_llvm, io, job;
                           optimize = true, dump_module = true)
    String(take!(io))
end

# Full LLVM → PTX → ptxas → cubin path; no `link`, so no device load.
# Throws on ptxas rejection (stderr is in the error message).
function ptxas_compiles(f, tt::Type{<:Tuple};
                        cap::VersionNumber, feature_set::Symbol = :baseline)
    job = _explicit_target_job(f, tt; cap, feature_set)
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

# Rebuild through every declared field instead of hand-copying each node
# type. The projection consequently fails loudly for a node without a normal
# positional constructor, and automatically preserves a future IR field rather
# than quietly weakening a structural oracle by omitting it.
function _deep_unraw_field(name::Symbol, value)
    name === :formatting && return _without_raw_line(value)
    (name === :raw_header || name === :raw_source) && return nothing

    # RawLine is a Statement too, so it remains present in any projected tree.
    # Only actual statement containers are recursed into; operand/parameter
    # tuples and every other field retain their exact values.
    value isa PTX.IR.Statement && return _deep_unraw_stmt(value)
    if value isa Tuple && all(item -> item isa PTX.IR.Statement, value)
        return Tuple(_deep_unraw_stmt(item) for item in value)
    end
    value
end

function _deep_unraw_node(node::T) where {T}
    names = fieldnames(T)
    values = ntuple(i -> _deep_unraw_field(names[i], getfield(node, i)),
                    fieldcount(T))
    T(values...)
end

_deep_unraw_stmt(stmt::PTX.IR.Statement) = _deep_unraw_node(stmt)
_deep_unraw(m::PTX.IR.Module) = _deep_unraw_node(m)

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

# --- compile-touch sweep engine ---------------------------------------------
#
# Shared by the ptxas/compile_touch_* shards. Every wrapper-owned Operation
# method must compile through its own Julia surface: conformance pins intrinsic
# selection by synthesizing IR directly, so the Julia-side glue (operand
# marshaling, address-space casts, tuple flattening) of the long-tail wrappers
# is verified nowhere else. The sweep enumerates the wrapper method table,
# synthesizes the argument-TYPE tuple per method (emit_ptx needs types, not
# values), and backend-compiles each at its family's cap floor — no hardware,
# no goldens.
#
# The surface is sharded across files (wgmma alone is ~3/4 of it) so the
# parallel runner spreads the cost over workers, and methods within one
# (cap, feature_set) group are batched several calls per kernel: most of the
# per-method cost is GPUCompiler per-job overhead, not the wrapper itself.
# A failing batch is recompiled one method at a time, so attribution is only
# paid on failure.

# The kernel body: dispatch each operation, discard the results.
# record_coverage fires at cache-populate time (pre-optimization), so even a
# pure form that DCEs away still verifies inference and records coverage.
_touch(o, args...) = (o(args...); return nothing)

@inline _touch_all(::Tuple{}, ::Tuple{}) = nothing
@inline function _touch_all(ops::Tuple, args::Tuple)
    first(ops)(first(args)...)
    _touch_all(Base.tail(ops), Base.tail(args))
end
_touch_batch(ops, args) = (_touch_all(ops, args); return nothing)

const _WRAPPER_DIR = joinpath(dirname(@__DIR__), "src", "wrappers")

# Cap floor per family — refined by mods where one opcode spans arch
# generations. Everything here is a *compilation* target; no device.
function _touch_target(op::Symbol, mods)
    has(s) = any(m -> occursin(s, String(m)), mods)
    op === :tcgen05 && return (v"10.0", :arch)
    op === :wgmma   && return (v"9.0", :arch)
    op === :mma     && (has("kind::") || :block_scale in mods) &&
        return (v"12.1", :arch)
    op === :mma     && return (v"9.0", :arch)
    (op === :ldmatrix || op === :stmatrix) && :b8 in mods &&
        return (v"10.0", :arch)
    # cta_group is a Blackwell cluster-pair feature; g2s with a nonzero
    # cta_group operand cannot ISel below sm_100 (ledger: validated sm_100a).
    op === :cp && has("cta_group::2") && return (v"10.0", :arch)
    op === :cvt && (has("e2m") || has("e3m") || has("e4m") || has("e5m") ||
                    has("ue8m0")) && return (v"12.1", :arch)
    (v"9.0", :arch)
end

# Argument-type synthesis from the method signature. Substitution rules:
#   - free TypeVars instantiate to their `_touch_tv` pick (LLVMPtr element
#     types and the like — UInt64 unless a failure teaches us otherwise)
#   - Union parameters take their first concrete member
_touch_tv(tv::TypeVar) = UInt64

function _touch_argtypes(m::Method)
    sig = m.sig
    while sig isa UnionAll
        sig = sig{_touch_tv(sig.var)}
    end
    out = Any[]
    for T in sig.parameters[2:end]
        if T isa Union
            members = Base.uniontypes(T)
            i = findfirst(isconcretetype, members)
            i === nothing && return nothing
            push!(out, members[i])
        elseif T === Integer
            # duck-typed coordinate/count/mask operands — any Integer works
            push!(out, Int32)
        elseif T === Val
            # unconstrained immediate (mma.sp sparsity selector) — 0 is
            # inside every form's immarg range
            push!(out, Val{0})
        elseif T isa DataType || T isa Core.TypeofBottom
            isconcretetype(T) || return nothing
            push!(out, T)
        else
            return nothing
        end
    end
    out
end

# Sweep every wrapper method with `want(op) == true`. Returns
# (; touched, failures, unsynthesized); the caller owns the @test layer.
function compile_touch_sweep(want::Function; batch::Int = 16)
    entries = Tuple{Method, Symbol, Any, Vector{Any}, String}[]
    failures = String[]
    unsynthesized = String[]
    PTX._visit_operation_methods() do m, op, mods
        want(op) || return
        startswith(String(m.file), _WRAPPER_DIR) || return
        label = "ptx\"$(join((String(op), String.(mods)...), '.'))\"" *
                "($(m.sig isa UnionAll ? "…" : join(m.sig.parameters[2:end], ", ")))"
        argts = _touch_argtypes(m)
        if argts === nothing
            push!(unsynthesized, label)
            return
        end
        push!(entries, (m, op, mods, argts, label))
    end

    # Group by compilation target, then compile `batch` methods per kernel.
    groups = Dict{Tuple{VersionNumber, Symbol}, Vector{Int}}()
    for (i, (_, op, mods, _, _)) in enumerate(entries)
        push!(get!(Vector{Int}, groups, _touch_target(op, mods)), i)
    end

    touched = 0
    single_failure(i, cap) = begin
        m, _, _, argts, label = entries[i]
        opT = Base.unwrap_unionall(m.sig).parameters[1]
        try
            emit_ptx(_touch, Tuple{opT, argts...};
                     cap = cap[1], feature_set = cap[2])
            touched += 1
            nothing
        catch err
            push!(failures,
                  label * "  @sm_" * string(cap[1].major) * string(cap[1].minor) *
                  "\n      " * first(sprint(showerror, err), 200))
        end
    end
    for (cap, idxs) in groups
        for chunk in Iterators.partition(idxs, batch)
            opTs = Tuple{(Base.unwrap_unionall(entries[i][1].sig).parameters[1]
                          for i in chunk)...}
            argTs = Tuple{(Tuple{entries[i][4]...} for i in chunk)...}
            ok = try
                emit_ptx(_touch_batch, Tuple{opTs, argTs};
                         cap = cap[1], feature_set = cap[2])
                true
            catch
                false
            end
            if ok
                touched += length(chunk)
            else
                # Attribute the failure: recompile the batch one method at a
                # time so the report names exact spellings, not a chunk.
                foreach(i -> single_failure(i, cap), chunk)
            end
        end
    end
    (; touched, failures, unsynthesized)
end
