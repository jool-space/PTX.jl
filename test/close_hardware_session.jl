# Regenerates test/EVIDENCE.toml for the tier matching the attached device.
# Run at the end of a hardware session, after the full suite:
#
#   julia --project=test test/runtests.jl          # note the passed count
#   julia --project=test test/close_hardware_session.jl <passed-count>
#
# The entry is machine fields only (device, capability tier, HEAD tree,
# date, suite count); session narrative belongs in the session-close PR.
# Commit the rewritten file in the same PR as any fixes the session produced.

import CUDACore, TOML

const EVIDENCE_PATH = joinpath(@__DIR__, "EVIDENCE.toml")

function main(args)
    length(args) == 1 || error("usage: close_hardware_session.jl <passed-count>")
    suite = parse(Int, args[1])

    CUDACore.functional() || error("no functional GPU — nothing to record")
    dev = CUDACore.device()
    cap = CUDACore.capability(dev)
    tests = cap == v"9.0"        ? "gpu/hopper" :
            cap.major in (10, 11) ? "gpu/blackwell" :
            error("CC $cap has no manual-evidence tier (see EVIDENCE.toml)")
    device = "$(CUDACore.name(dev)) (CC $(cap.major).$(cap.minor))"

    root = normpath(joinpath(@__DIR__, ".."))
    tree = readchomp(`git -C $root rev-parse --short HEAD`)
    # The evidence file is this script's own output; a previous invocation
    # (e.g. correcting a mistyped count) must not look like unrecorded drift.
    dirty = filter(l -> !isempty(l) && !endswith(l, "test/EVIDENCE.toml"),
                   readlines(`git -C $root status --porcelain`))
    isempty(dirty) ||
        @warn "working tree is dirty — `tree = $tree` will not describe what ran"

    evidence = TOML.parsefile(EVIDENCE_PATH)
    tiers = evidence["tier"]
    i = findfirst(t -> t["tests"] == tests, tiers)
    i === nothing && error("no [[tier]] entry with tests = \"$tests\"")
    merge!(tiers[i], Dict("last_validated" => Libc.strftime("%Y-%m-%d", time()),
                          "device" => device, "tree" => tree,
                          "suite" => suite))

    header = join(Iterators.takewhile(l -> startswith(l, '#') || isempty(l),
                                      readlines(EVIDENCE_PATH)), '\n')
    open(EVIDENCE_PATH, "w") do io
        println(io, header)
        for t in tiers
            println(io, "[[tier]]")
            for key in ("tests", "runtime", "evidence", "last_validated",
                        "device", "tree", "suite")
                haskey(t, key) || continue
                v = t[key]
                println(io, key, " = ", v isa Integer ? string(v) :
                            repr(String(v)))
            end
            println(io)
        end
    end
    println("recorded: $tests on $device, tree $tree, suite $suite")
end

main(ARGS)
