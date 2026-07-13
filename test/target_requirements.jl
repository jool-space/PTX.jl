module TestTargets

export CapabilityPredicate, TestEnvironment, TestRequirement, PlanEntry,
       format_manifest, parse_test_requirement, plan_entry,
       capability_matches, read_test_requirement, requirement_for_test,
       requires_gpu, requires_toolchain, runtime_supported

"""A direct live-device compute-capability predicate."""
struct CapabilityPredicate
    kind::Symbol
    cap::VersionNumber
end

"""Structured resource, evidence, and live-device capability policy for one test file."""
struct TestRequirement
    requires::Symbol
    evidence::Symbol
    runtime::Vector{CapabilityPredicate}
end

struct TestEnvironment
    toolchain_checked::Bool
    toolchain_available::Bool
    toolchain_version::Union{Nothing,VersionNumber}
    gpu_checked::Bool
    gpu_functional::Bool
    capability::VersionNumber
    device_name::String
end

struct PlanEntry
    test::String
    action::Symbol
    requirement::TestRequirement
    reason::String
end

const _BANNER_START_RE = r"^\s*#\s*TEST_TARGET\b"
const _BANNER_RE = r"^\s*#\s*TEST_TARGET\s*:\s*(.*?)\s*$"
const _FIELD_RE = r"^([a-z_]+)=([^\s]+)$"
const _MIN_CC_RE = r"^cc>=(\d+)(?:\.(\d+))?$"
const _EQUAL_CC_RE = r"^cc==(\d+)(?:\.(\d+))?$"
const _ALLOWED_FIELDS = Set(("requires", "evidence", "runtime"))

_cap_string(cap::VersionNumber) = "$(cap.major).$(cap.minor)"

function _parse_runtime(text::AbstractString)
    m = match(_MIN_CC_RE, text)
    if m !== nothing
        major = parse(Int, m.captures[1])
        minor = m.captures[2] === nothing ? 0 : parse(Int, m.captures[2])
        return CapabilityPredicate(:minimum, VersionNumber(major, minor))
    end
    m = match(_EQUAL_CC_RE, text)
    if m !== nothing
        major = parse(Int, m.captures[1])
        if m.captures[2] === nothing
            return CapabilityPredicate(:major, VersionNumber(major))
        end
        return CapabilityPredicate(:exact,
                                   VersionNumber(major, parse(Int, m.captures[2])))
    end
    throw(ArgumentError("invalid TEST_TARGET runtime predicate $(repr(text)); expected cc>=X[.Y], cc==X.Y, or cc==X"))
end

function _runtime_string(predicate::CapabilityPredicate)
    predicate.kind === :minimum && return "cc>=$(_cap_string(predicate.cap))"
    predicate.kind === :exact && return "cc==$(_cap_string(predicate.cap))"
    predicate.kind === :major && return "cc==$(predicate.cap.major)"
    error("unknown capability predicate kind $(repr(predicate.kind))")
end

function _validate_requirement(req::TestRequirement)
    if req.requires === :host
        req.evidence === :host && isempty(req.runtime) ||
            throw(ArgumentError("requires=host needs evidence=host and no runtime predicate"))
    elseif req.requires === :toolkit
        if req.evidence in (:compile, :ptxas)
            isempty(req.runtime) ||
                throw(ArgumentError("offline compile tests must not declare a live-device runtime predicate"))
        elseif req.evidence === :mixed
            isempty(req.runtime) &&
                throw(ArgumentError("mixed ptxas/runtime tests need a runtime predicate"))
        else
            throw(ArgumentError("requires=toolkit needs evidence=compile, evidence=ptxas, or evidence=mixed"))
        end
    elseif req.requires === :gpu
        req.evidence in (:runtime, :compile) ||
            throw(ArgumentError("requires=gpu needs evidence=runtime or evidence=compile"))
        isempty(req.runtime) &&
            throw(ArgumentError("requires=gpu needs a live-device runtime predicate"))
    else
        throw(ArgumentError("unknown TEST_TARGET resource $(repr(req.requires)); expected host, toolkit, or gpu"))
    end
    req
end

"""
    parse_test_requirement(line)

Parse one strict metadata banner. Runtime alternatives use `|`, for example:

    # TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.9
    # TEST_TARGET: requires=toolkit evidence=compile
    # TEST_TARGET: requires=toolkit evidence=mixed runtime=cc==10|cc==11
    # TEST_TARGET: requires=toolkit evidence=mixed runtime=cc==9.0
"""
function parse_test_requirement(line::AbstractString)
    m = match(_BANNER_RE, line)
    m === nothing &&
        throw(ArgumentError("malformed TEST_TARGET banner: $(repr(line))"))

    fields = Dict{String,String}()
    for token in split(m.captures[1])
        fm = match(_FIELD_RE, token)
        fm === nothing &&
            throw(ArgumentError("malformed TEST_TARGET field $(repr(token)); expected key=value"))
        key, value = fm.captures
        key in _ALLOWED_FIELDS ||
            throw(ArgumentError("unknown TEST_TARGET field $(repr(key))"))
        haskey(fields, key) &&
            throw(ArgumentError("duplicate TEST_TARGET field $(repr(key))"))
        fields[key] = value
    end

    for key in ("requires", "evidence")
        haskey(fields, key) ||
            throw(ArgumentError("TEST_TARGET banner is missing $key=..."))
    end
    runtime = CapabilityPredicate[]
    if haskey(fields, "runtime")
        raw_predicates = split(fields["runtime"], '|')
        any(isempty, raw_predicates) &&
            throw(ArgumentError("TEST_TARGET runtime alternatives must be nonempty"))
        append!(runtime, _parse_runtime.(raw_predicates))
        length(unique(_runtime_string.(runtime))) == length(runtime) ||
            throw(ArgumentError("TEST_TARGET contains a duplicate runtime alternative"))
    end

    _validate_requirement(TestRequirement(Symbol(fields["requires"]),
                                          Symbol(fields["evidence"]),
                                          runtime))
end

function read_test_requirement(file::AbstractString)
    isfile(file) || throw(ArgumentError("test file does not exist: $file"))
    banners = Pair{Int,String}[]
    for (line_number, line) in enumerate(eachline(file))
        occursin(_BANNER_START_RE, line) || continue
        # Validate every banner-looking line, even one far below the header.
        # A later malformed or conflicting policy must not hide outside the
        # 20-line discovery window.
        match(_BANNER_RE, line) === nothing &&
            throw(ArgumentError("malformed TEST_TARGET banner at $file:$line_number: $(repr(line))"))
        push!(banners, line_number => line)
    end
    isempty(banners) &&
        throw(ArgumentError("GPU test $file is missing a structured # TEST_TARGET: banner"))
    length(banners) == 1 ||
        throw(ArgumentError("GPU test $file has $(length(banners)) TEST_TARGET banners; expected exactly one"))
    line_number, line = only(banners)
    line_number <= 20 ||
        throw(ArgumentError("GPU test $file has its TEST_TARGET banner at line $line_number; it must be within the first 20 lines"))
    parse_test_requirement(line)
end

const _HOST_REQUIREMENT = TestRequirement(:host, :host, CapabilityPredicate[])
const _PTXAS_REQUIREMENT = TestRequirement(:toolkit, :ptxas, CapabilityPredicate[])

"""Return the explicit or path-derived policy for a discovered test."""
function requirement_for_test(test::AbstractString, test_dir::AbstractString)
    if startswith(test, "gpu/")
        return read_test_requirement(joinpath(test_dir, test * ".jl"))
    elseif startswith(test, "ptxas/")
        # These compile to a cubin and stop before device linking.  Their exact
        # cross-target is declared at each ptxas_compiles call.
        return _PTXAS_REQUIREMENT
    else
        return _HOST_REQUIREMENT
    end
end

capability_matches(predicate::CapabilityPredicate, cap::VersionNumber) =
    predicate.kind === :minimum ? cap >= predicate.cap :
    predicate.kind === :exact ? cap == predicate.cap :
    predicate.kind === :major ? cap.major == predicate.cap.major :
    error("unknown capability predicate kind $(repr(predicate.kind))")
capability_matches(req::TestRequirement, cap::VersionNumber) =
    any(predicate -> capability_matches(predicate, cap), req.runtime)

runtime_supported(req::TestRequirement, cap::VersionNumber) =
    !isempty(req.runtime) && capability_matches(req, cap)
runtime_supported(file::AbstractString, cap::VersionNumber) =
    runtime_supported(read_test_requirement(file), cap)

"""Whether a requirement needs the offline CUDA compiler/toolchain."""
requires_toolchain(req::TestRequirement) = req.requires !== :host

"""Whether a requirement contains active-device compile or runtime evidence."""
requires_gpu(req::TestRequirement) =
    req.requires === :gpu ||
    (req.requires === :toolkit && req.evidence === :mixed)

function _describe(req::TestRequirement)
    runtime = isempty(req.runtime) ? "" :
              " runtime=" * join(_runtime_string.(req.runtime), "|")
    "requires=$(req.requires) evidence=$(req.evidence)$runtime"
end

function plan_entry(test::AbstractString, req::TestRequirement,
                    env::TestEnvironment; forced::Bool = false)
    eligible = runtime_supported(req, env.capability)
    if forced
        if req.requires === :toolkit && req.evidence === :mixed
            runtime = if !env.gpu_checked
                "runtime routing check skipped"
            elseif !env.gpu_functional
                "runtime remains skipped (functional GPU unavailable)"
            elseif eligible
                "runtime eligible"
            else
                "runtime remains skipped (live device capability mismatch)"
            end
            return PlanEntry(String(test), :execute, req,
                             "explicit selection executes offline cross-target ptxas compile; $runtime")
        end
        detail = isempty(req.runtime) ? "" :
                 env.gpu_functional && eligible ? "; runtime capability eligible" :
                 env.gpu_functional ? "; runtime capability ineligible but gate bypassed" :
                                      "; functional GPU unavailable but gate bypassed"
        return PlanEntry(String(test), :execute, req,
                         "explicit selection bypasses default routing$detail")
    end

    if req.requires === :host
        return PlanEntry(String(test), :execute, req, "host-only")
    elseif !env.toolchain_checked
        return PlanEntry(String(test), :skip, req,
                         "offline compiler routing check skipped for host-only selection")
    elseif !env.toolchain_available
        return PlanEntry(String(test), :skip, req,
                         "offline CUDA compiler/ptxas unavailable")
    elseif req.requires === :toolkit
        if req.evidence in (:compile, :ptxas)
            kind = req.evidence === :compile ? "PTX emission" : "ptxas compile"
            return PlanEntry(String(test), :execute, req,
                             "offline explicit-target $kind; live GPU is not required")
        end
        runtime = if !env.gpu_checked
            "runtime routing check skipped"
        elseif !env.gpu_functional
            "runtime skipped (functional GPU unavailable)"
        elseif eligible
            "runtime eligible"
        else
            "runtime skipped (live device capability mismatch)"
        end
        return PlanEntry(String(test), :execute, req,
                         "offline cross-target ptxas compile; $runtime")
    elseif !env.gpu_checked
        return PlanEntry(String(test), :skip, req,
                         "GPU routing check skipped for non-GPU selection")
    elseif !env.gpu_functional
        return PlanEntry(String(test), :skip, req,
                         "functional GPU unavailable")
    elseif eligible
        kind = req.evidence === :compile ? "active-device compile capability satisfied" :
                                          "runtime capability satisfied"
        return PlanEntry(String(test), :execute, req, kind)
    else
        return PlanEntry(String(test), :skip, req,
                         "live device capability mismatch")
    end
end

"""Render a stable, sorted executed/skipped plan for CI and local audit logs."""
function format_manifest(entries::AbstractVector{PlanEntry}, env::TestEnvironment)
    ordered = sort(entries; by = entry -> entry.test)
    executed = count(entry -> entry.action === :execute, ordered)
    skipped = count(entry -> entry.action === :skip, ordered)
    toolchain = if !env.toolchain_checked
        "offline-compiler routing-check=skipped"
    elseif env.toolchain_available
        version = env.toolchain_version === nothing ? "unknown" :
                  string(env.toolchain_version)
        "offline-compiler=available version=$version"
    else
        "offline-compiler=unavailable"
    end
    gpu = if !env.gpu_checked
        "GPU routing-check=skipped"
    elseif env.gpu_functional
        # Capability is factual device evidence.  Do not synthesize an
        # `sm_NNa` target from it: architecture-specific targets exist only
        # for selected architectures, and the compiler may choose another
        # compatible target.
        "GPU=functional device=$(repr(env.device_name)) capability=$(_cap_string(env.capability))"
    else
        "GPU=unavailable"
    end
    environment = "$toolchain; $gpu"
    lines = String[
        "PTX test target manifest",
        "  environment: $environment",
        "  summary: execute=$executed skip=$skipped total=$(length(ordered))",
    ]
    for entry in ordered
        action = entry.action === :execute ? "EXEC" : "SKIP"
        push!(lines, "  $action $(entry.test) [$(_describe(entry.requirement))] — $(entry.reason)")
    end
    join(lines, '\n')
end

end # module TestTargets
