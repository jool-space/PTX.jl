import PTX, CUDACore, CUDATools
using ParallelTestRunner

include(joinpath(@__DIR__, "target_requirements.jl"))
using .TestTargets

const init_code = quote
    using PTX, CUDACore, Test
    include($(joinpath(@__DIR__, "setup.jl")))
end

# `gpu/` files carry a strict `# TEST_TARGET:` banner with resource, evidence,
# and direct live-device compute-capability predicates. Runtime policy never
# embeds PTX target spellings: `cc>=X.Y` is a floor, `cc==X.Y` is exact, and
# integer `cc==X` selects one hardware family. PTX `sm_*`, `f`, and `a` targets
# remain local to explicit compilation calls where their ISA meaning applies.
#
# `ptxas/` is intentionally different: it compiles to a cubin without linking
# it onto the device. Explicit-target PTX and ptxas evidence use
# `compiler_config(nothing; arch=...)`, so a live GPU never gates them.

testsuite = find_tests(@__DIR__)

args = parse_args(ARGS)
apply_default_routing = filter_tests!(testsuite, args)
if args.list === nothing
    requirements = Dict(test => requirement_for_test(test, @__DIR__)
                        for test in keys(testsuite))

    check_toolchain = any(requires_toolchain, values(requirements))
    toolchain_available = check_toolchain &&
                          CUDACore.CUDA_Compiler.is_available()
    toolchain_version = toolchain_available ? CUDACore.compiler_version() : nothing

    check_gpu = any(requires_gpu, values(requirements))
    gpu_functional = check_gpu && CUDACore.functional()
    cap = gpu_functional ? CUDACore.capability(CUDACore.device()) : v"0.0"
    device_name = gpu_functional ? String(CUDACore.name(CUDACore.device())) : ""
    environment = TestEnvironment(check_toolchain, toolchain_available,
                                  toolchain_version, check_gpu, gpu_functional,
                                  cap, device_name)
    if check_toolchain && !toolchain_available
        @warn "Offline CUDA compiler/ptxas unavailable — skipping compile evidence"
    end
    if check_gpu && gpu_functional
        @info "Running GPU tests" device=CUDACore.name(CUDACore.device()) capability=cap
    elseif check_gpu
        @warn "Functional GPU unavailable — runtime evidence will be skipped"
    end
    manifest = PlanEntry[]
    filter!(testsuite) do (test, _)
        # These support files are loaded into every worker via init_code; don't
        # run them as standalone tests during default routing.
        if test in ("setup", "target_requirements") && apply_default_routing
            req = requirements[test]
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
            req = requirements[test]
            push!(manifest, PlanEntry(test, :skip, req,
                                      "--check-bounds=yes invalidates golden emission"))
            return false
        end
        req = requirements[test]
        entry = plan_entry(test, req, environment;
                           forced = !apply_default_routing)
        push!(manifest, entry)
        entry.action === :execute
    end
    println(format_manifest(manifest, environment))
end

runtests(PTX, ARGS; init_code, testsuite)
