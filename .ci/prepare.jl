# Resolve test and documentation dependencies in isolated environments on
# every supported Julia, including versions without Pkg workspace support.
using Pkg

lane = only(ARGS)
lane in ("test", "docs") || error("expected test or docs")
root = dirname(@__DIR__)
environment = mktempdir(; cleanup = false)
cp(joinpath(root, lane, "Project.toml"), joinpath(environment, "Project.toml"))
Pkg.activate(environment)
# Pkg.develop does not refresh cached registries on persistent CI runners.
Pkg.Registry.update()
Pkg.develop(PackageSpec(path = root))
Pkg.instantiate()
if haskey(ENV, "GITHUB_ENV")
    open(ENV["GITHUB_ENV"], "a") do io
        println(io, "PTX_CI_PROJECT=", environment)
    end
end
println("Prepared ", lane, " environment: ", environment)
