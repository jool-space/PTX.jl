import PTX, CUDACore, CUDATools
using ParallelTestRunner

include(joinpath(@__DIR__, "target_requirements.jl"))
using .TestTargets

const init_code = quote
    using PTX, CUDACore, Test
    include($(joinpath(@__DIR__, "setup.jl")))
end

# `gpu/` files carry a strict `# TEST_TARGET:` banner with resource, evidence,
# and live-device target policy.  PTX target suffixes retain their ISA meaning:
# baseline targets are forward-compatible, `f` targets stay within a family,
# and `a` targets are exact-CC.  `active-arch>=N.N` is a separate source-level
# policy for tests recompiled for the live device's own architecture target.
#
# `ptxas/` is intentionally different: it compiles to a cubin without linking
# it onto the device, so active capability never gates those cross-target tests.

testsuite = find_tests(@__DIR__)

args = parse_args(ARGS)
apply_default_routing = filter_tests!(testsuite, args)
if args.list === nothing
    check_cuda_routing = suite_requires_cuda_routing(keys(testsuite))
    cuda_functional = check_cuda_routing && CUDACore.functional()
    cap = cuda_functional ? CUDACore.capability(CUDACore.device()) : v"0.0"
    device_name = cuda_functional ? String(CUDACore.name(CUDACore.device())) : ""
    environment = TestEnvironment(check_cuda_routing, cuda_functional, cap, device_name)
    if check_cuda_routing && cuda_functional
        @info "Running GPU tests" device=CUDACore.name(CUDACore.device()) capability=cap
    elseif check_cuda_routing
        @warn "CUDACore not functional — skipping GPU tests"
    end
    manifest = PlanEntry[]
    filter!(testsuite) do (test, _)
        # These support files are loaded into every worker via init_code; don't
        # run them as standalone tests during default routing.
        if test in ("setup", "target_requirements") && apply_default_routing
            req = requirement_for_test(test, @__DIR__)
            push!(manifest, PlanEntry(test, :skip, req,
                                      "test support file loaded by each worker"))
            return false
        end
        if test == "ptxas/golden" && Base.JLOptions().check_bounds == 1 &&
           apply_default_routing
            # Golden comparison is byte-exact, and forced bounds checks
            # (Pkg.test's default) inject branches into the golden kernels —
            # the tests cannot meaningfully run in this mode, so skip them
            # rather than fail. CI enforces goldens with check_bounds: 'auto'.
            # Naming the test explicitly bypasses this filter and hits the
            # loud refusal in setup.jl instead.
            req = requirement_for_test(test, @__DIR__)
            push!(manifest, PlanEntry(test, :skip, req,
                                      "--check-bounds=yes invalidates golden emission"))
            return false
        end
        req = requirement_for_test(test, @__DIR__)
        entry = plan_entry(test, req, environment;
                           forced = !apply_default_routing)
        push!(manifest, entry)
        entry.action === :execute
    end
    println(format_manifest(manifest, environment))
end

runtests(PTX, ARGS; init_code, testsuite)
