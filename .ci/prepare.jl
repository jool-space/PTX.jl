# Isolated environments let every supported Julia test the same CUDA sources,
# including Julia 1.10, whose Pkg does not support Project.toml [sources].
using Pkg

lane = only(ARGS)
lane in ("test", "docs") || error("expected test or docs")
root = dirname(@__DIR__)
environment = mktempdir(; cleanup = false)
cp(joinpath(root, lane, "Project.toml"), joinpath(environment, "Project.toml"))
Pkg.activate(environment)
specs = [PackageSpec(path = root)]
if lane == "test"
    # Registered CUDACore releases require backend 22. Keep this checkout
    # reproducible until a release provides the library-based backend.
    revision = "fdd4f703acb8e829abe4de0cddf03c851799ed2e"
    checkout = joinpath(environment, "CUDA.jl")
    run(`git init --quiet $checkout`)
    run(`git -C $checkout remote add origin https://github.com/JuliaGPU/CUDA.jl`)
    run(`git -C $checkout fetch --quiet --depth 1 origin $revision`)
    run(`git -C $checkout checkout --quiet --detach FETCH_HEAD`)
    for directory in ("CUDACore", "CUDATools", "lib/cupti", "lib/nvml")
        push!(specs, PackageSpec(path = joinpath(checkout, directory)))
    end
end
# Pkg.develop does not refresh cached registries on persistent CI runners.
Pkg.Registry.update()
Pkg.develop(specs)
Pkg.instantiate()
if haskey(ENV, "GITHUB_ENV")
    open(ENV["GITHUB_ENV"], "a") do io
        println(io, "PTX_CI_PROJECT=", environment)
    end
end
println("Prepared ", lane, " environment: ", environment)
