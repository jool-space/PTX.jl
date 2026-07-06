import PTX, CUDACore, CUDATools
using ParallelTestRunner

const init_code = quote
    using PTX, CUDACore, Test
    include($(joinpath(@__DIR__, "setup.jl")))
end

# Tests under `gpu/` may declare a compute-capability requirement via a
# `# REQUIRES CC [op]<major>.<minor>` banner near the top of the file:
#
#   # REQUIRES CC 8.9        →  cap >= 8.9   (default operator)
#   # REQUIRES CC >=8.9      →  cap >= 8.9   (explicit)
#   # REQUIRES CC ==9.0      →  cap == 9.0   (exact — for arch-specific PTX
#                                             like wgmma on CC 9.0 / sm_90a,
#                                             tcgen05 on CC 10.0 / sm_100a)
const DEFAULT_GPU_MIN_CAP = v"7.0"
const REQUIRES_CC_RE = r"^\s*#\s*REQUIRES\s+CC\s+(>=|==)?\s*(\d+)\.(\d+)\b"i

struct CapReq
    cap::VersionNumber
    exact::Bool
end
Base.show(io::IO, r::CapReq) =
    print(io, "CC ", r.exact ? "==" : ">=", r.cap.major, ".", r.cap.minor)

const DEFAULT_CAP_REQ = CapReq(DEFAULT_GPU_MIN_CAP, false)

satisfies(r::CapReq, dev::VersionNumber) =
    r.exact ? dev == r.cap : dev >= r.cap

function read_cap_req(file::AbstractString)::CapReq
    isfile(file) || return DEFAULT_CAP_REQ
    open(file) do io
        for _ in 1:20
            eof(io) && break
            line = readline(io)
            m = match(REQUIRES_CC_RE, line)
            m === nothing && continue
            op, major, minor = m.captures
            return CapReq(VersionNumber(parse(Int, major), parse(Int, minor)),
                          op == "==")
        end
        return DEFAULT_CAP_REQ
    end
end

testsuite = find_tests(@__DIR__)

args = parse_args(ARGS)
if filter_tests!(testsuite, args)
    cuda_functional = CUDACore.functional()
    cap = cuda_functional ? CUDACore.capability(CUDACore.device()) : v"0.0"
    if cuda_functional
        @info "Running GPU tests" device=CUDACore.name(CUDACore.device()) capability=cap
    else
        @warn "CUDACore not functional — skipping GPU tests"
    end
    filter!(testsuite) do (test, _)
        # setup.jl is loaded into every worker via init_code; don't run it
        # as a standalone test.
        test == "setup" && return false
        if test == "ptxas/golden" && Base.JLOptions().check_bounds == 1
            # Golden comparison is byte-exact, and forced bounds checks
            # (Pkg.test's default) inject branches into the golden kernels —
            # the tests cannot meaningfully run in this mode, so skip them
            # rather than fail. CI enforces goldens with check_bounds: 'auto'.
            # Naming the test explicitly bypasses this filter and hits the
            # loud refusal in setup.jl instead.
            @warn """skipping ptxas/golden: --check-bounds=yes (Pkg.test's default) injects \
                     bounds branches into the golden kernels. To include goldens, run \
                     `julia --project=test test/runtests.jl` or \
                     `Pkg.test("PTX"; julia_args=["--check-bounds=auto"])`."""
            return false
        end
        if startswith(test, "gpu/")
            cuda_functional || return false
            req = read_cap_req(joinpath(@__DIR__, test * ".jl"))
            if !satisfies(req, cap)
                @info "skipping GPU test (compute capability mismatch)" test required=req got=cap
                return false
            end
            return true
        elseif startswith(test, "ptxas/")
            # ptxas/ tests compile through CUDACore.compile(job), which runs
            # ptxas but never links the cubin onto the device. They validate
            # wrappers cross-arch (sm_50..sm_100a) and only require a
            # functional CUDA install — no cap match needed.
            cuda_functional || return false
            return true
        end
        return true
    end
end

runtests(PTX, ARGS; init_code, testsuite)
