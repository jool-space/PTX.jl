# Standing conformance between the committed JSON snapshot and the committed
# machine-generated table: regenerate src/nvvm/table.jl in memory from
# gen/nvvm_intrinsics_<version>.json and byte-compare against the file on
# disk. This closes the hand-edit hole that name-level conformance
# (test/host/conformance.jl layer 2) cannot see: an edit that names a real
# intrinsic but lies about its types or properties keeps the name table
# identical, and only an intrinsic with a selection probe would catch it.
#
# Host-only. Deliberately exercises just the emission half of the generator
# (generate_table_source), which needs JSON and nothing from LLVM_full_jll —
# the tblgen-driven extraction half stays a gen/-environment concern.

module RegistryGenerator
include(joinpath(dirname(dirname(@__DIR__)), "gen", "generate_registry.jl"))
end

@testset "table.jl matches the committed JSON snapshot" begin
    root = dirname(dirname(@__DIR__))
    gen_dir = joinpath(root, "gen")

    # Mirror the generator's own no-argument discovery rule: exactly one
    # snapshot lives in gen/ (the backend-bump runbook deletes the old one).
    snapshot = joinpath(gen_dir,
        only(filter(f -> occursin(r"^nvvm_intrinsics_.*\.json$", f),
                    readdir(gen_dir))))

    generated = RegistryGenerator.generate_table_source(snapshot)
    committed = read(joinpath(root, "src", "nvvm", "table.jl"), String)

    # Compare a Bool, not the strings: a failing @test on two ~2500-line
    # sources would dump both into the log.
    matches = generated == committed
    matches || @error("table.jl does not match the committed JSON snapshot — " *
                      "regenerate via gen/ (see gen/README.md) or revert the " *
                      "hand-edit",
                      snapshot, generated_length = length(generated),
                      committed_length = length(committed))
    @test matches
end
