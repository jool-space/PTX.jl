module TestTargets

using CUDACore: SMVersion
import CUDACore

export ActiveArchFloor, TestEnvironment, TestRequirement, PlanEntry,
       format_manifest, parse_test_requirement, plan_entry,
       read_test_requirement, requirement_for_test, runtime_supported,
       suite_requires_cuda_routing, target_matches

"""
A source-level requirement that selects the live device's own architecture-
specific target at or above `cap`.  This is deliberately distinct from an
`sm_NNa` requirement: an `sm_100a` cubin is exact-CC and does not become
forward-compatible merely because the same Julia source can be recompiled for
the active device's `a` target.
"""
struct ActiveArchFloor
    cap::VersionNumber
end

const RuntimeTarget = Union{SMVersion, ActiveArchFloor}

"""Structured resource, evidence, and live-device target policy for one test file."""
struct TestRequirement
    requires::Symbol
    evidence::Symbol
    targets::Vector{RuntimeTarget}
end

struct TestEnvironment
    cuda_routing_checked::Bool
    cuda_functional::Bool
    capability::VersionNumber
    device_name::String
end

TestEnvironment(cuda_functional::Bool, capability::VersionNumber,
                device_name::String) =
    TestEnvironment(true, cuda_functional, capability, device_name)

struct PlanEntry
    test::String
    action::Symbol
    requirement::TestRequirement
    reason::String
end

const _BANNER_START_RE = r"^\s*#\s*TEST_TARGET\b"
const _BANNER_RE = r"^\s*#\s*TEST_TARGET\s*:\s*(.*?)\s*$"
const _FIELD_RE = r"^([a-z_]+)=([^\s]+)$"
const _ACTIVE_ARCH_RE = r"^active-arch>=(\d+)\.(\d+)$"
const _ALLOWED_FIELDS = Set(("requires", "evidence", "target"))

_cap_string(cap::VersionNumber) = "$(cap.major).$(cap.minor)"

function _sm_string(sm::SMVersion)
    suffix = sm.feature_set === :arch ? "a" :
             sm.feature_set === :family ? "f" : ""
    "sm_$(sm.major)$(sm.minor)$suffix"
end

_target_string(target::SMVersion) = _sm_string(target)
_target_string(target::ActiveArchFloor) =
    "active-arch>=$(_cap_string(target.cap))"

function _parse_target(text::AbstractString)::RuntimeTarget
    m = match(_ACTIVE_ARCH_RE, text)
    if m !== nothing
        major, minor = parse.(Int, m.captures)
        return ActiveArchFloor(VersionNumber(major, minor))
    end
    try
        return SMVersion(text)
    catch err
        throw(ArgumentError("invalid TEST_TARGET target $(repr(text)): $(sprint(showerror, err))"))
    end
end

function _validate_requirement(req::TestRequirement)
    if req.requires === :host
        req.evidence === :host && isempty(req.targets) ||
            throw(ArgumentError("requires=host needs evidence=host and no target"))
    elseif req.requires === :toolkit
        if req.evidence === :ptxas
            isempty(req.targets) ||
                throw(ArgumentError("ptxas-only tests must declare their explicit targets at the compile call, not as a live-device target"))
        elseif req.evidence === :mixed
            isempty(req.targets) &&
                throw(ArgumentError("mixed ptxas/runtime tests need a runtime target"))
        else
            throw(ArgumentError("requires=toolkit needs evidence=ptxas or evidence=mixed"))
        end
    elseif req.requires === :gpu
        req.evidence in (:runtime, :compile) ||
            throw(ArgumentError("requires=gpu needs evidence=runtime or evidence=compile"))
        isempty(req.targets) &&
            throw(ArgumentError("requires=gpu needs a live-device target"))
    else
        throw(ArgumentError("unknown TEST_TARGET resource $(repr(req.requires)); expected host, toolkit, or gpu"))
    end
    req
end

"""
    parse_test_requirement(line)

Parse one strict metadata banner.  Target alternatives use `|`, for example:

    # TEST_TARGET: requires=gpu evidence=runtime target=sm_89
    # TEST_TARGET: requires=toolkit evidence=mixed target=sm_100f|sm_110f
    # TEST_TARGET: requires=gpu evidence=runtime target=active-arch>=10.0
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
    targets = RuntimeTarget[]
    if haskey(fields, "target")
        raw_targets = split(fields["target"], '|')
        any(isempty, raw_targets) &&
            throw(ArgumentError("TEST_TARGET target alternatives must be nonempty"))
        append!(targets, _parse_target.(raw_targets))
        length(unique(_target_string.(targets))) == length(targets) ||
            throw(ArgumentError("TEST_TARGET contains a duplicate target alternative"))
    end

    _validate_requirement(TestRequirement(Symbol(fields["requires"]),
                                          Symbol(fields["evidence"]),
                                          targets))
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

const _HOST_REQUIREMENT = TestRequirement(:host, :host, RuntimeTarget[])
const _PTXAS_REQUIREMENT = TestRequirement(:toolkit, :ptxas, RuntimeTarget[])

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

target_matches(target::SMVersion, cap::VersionNumber) =
    CUDACore.runs_on(target, cap)
target_matches(target::ActiveArchFloor, cap::VersionNumber) =
    cap >= target.cap
target_matches(req::TestRequirement, cap::VersionNumber) =
    any(target -> target_matches(target, cap), req.targets)

runtime_supported(req::TestRequirement, cap::VersionNumber) =
    !isempty(req.targets) && target_matches(req, cap)
runtime_supported(file::AbstractString, cap::VersionNumber) =
    runtime_supported(read_test_requirement(file), cap)

"""Whether a selected suite needs a CUDA functionality/capability routing check."""
suite_requires_cuda_routing(tests) =
    any(test -> startswith(test, "gpu/") || startswith(test, "ptxas/"), tests)

function _describe(req::TestRequirement)
    target = isempty(req.targets) ? "" :
             " target=" * join(_target_string.(req.targets), "|")
    "requires=$(req.requires) evidence=$(req.evidence)$target"
end

function plan_entry(test::AbstractString, req::TestRequirement,
                    env::TestEnvironment; forced::Bool = false)
    eligible = runtime_supported(req, env.capability)
    if forced
        if req.requires === :toolkit && req.evidence === :mixed
            runtime = eligible ? "runtime eligible" :
                                 "runtime remains skipped (live device target mismatch)"
            return PlanEntry(String(test), :execute, req,
                             "explicit selection executes cross-target ptxas compile; $runtime")
        end
        detail = isempty(req.targets) ? "" :
                 eligible ? "; runtime target eligible" :
                            "; runtime target ineligible but gate bypassed"
        return PlanEntry(String(test), :execute, req,
                         "explicit selection bypasses default routing$detail")
    end

    if req.requires === :host
        return PlanEntry(String(test), :execute, req, "host-only")
    elseif !env.cuda_routing_checked
        return PlanEntry(String(test), :skip, req,
                         "CUDA routing check skipped for host-only selection")
    elseif !env.cuda_functional
        return PlanEntry(String(test), :skip, req,
                         "functional CUDA toolkit/device unavailable")
    elseif req.requires === :toolkit
        if req.evidence === :ptxas
            return PlanEntry(String(test), :execute, req,
                             "cross-target ptxas compile; live-device target is not a gate")
        end
        runtime = eligible ? "runtime eligible" :
                             "runtime skipped (live device target mismatch)"
        return PlanEntry(String(test), :execute, req,
                         "cross-target ptxas compile; $runtime")
    elseif eligible
        kind = req.evidence === :compile ? "active-device compile target satisfied" :
                                          "runtime target satisfied"
        return PlanEntry(String(test), :execute, req, kind)
    else
        return PlanEntry(String(test), :skip, req,
                         "live device target mismatch")
    end
end

"""Render a stable, sorted executed/skipped plan for CI and local audit logs."""
function format_manifest(entries::AbstractVector{PlanEntry}, env::TestEnvironment)
    ordered = sort(entries; by = entry -> entry.test)
    executed = count(entry -> entry.action === :execute, ordered)
    skipped = count(entry -> entry.action === :skip, ordered)
    environment = if !env.cuda_routing_checked
        # CUDACore is still imported by the test harness and performs its own
        # initialization.  This only says that the runner did not issue an
        # additional functionality/device-capability query for routing.
        "CUDA routing-check=skipped selection=host-only"
    elseif env.cuda_functional
        # Capability is factual device evidence.  Do not synthesize an
        # `sm_NNa` target from it: architecture-specific targets exist only
        # for selected architectures, and the compiler may choose another
        # compatible target.
        "CUDA=functional device=$(repr(env.device_name)) capability=$(_cap_string(env.capability))"
    else
        "CUDA=unavailable"
    end
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
