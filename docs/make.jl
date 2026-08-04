using PTX
using Documenter
using TOML

DocMeta.setdocmeta!(PTX, :DocTestSetup, :(using PTX); recursive=true)

# --- docs/src/coverage.md is GENERATED from docs/SURFACE.toml ----------------
# The inventory is the reviewed artifact and test/host/surface.jl enforces it
# against the code; this step only renders it. The output is gitignored.
function write_coverage_page()
    surface = TOML.parsefile(joinpath(@__DIR__, "SURFACE.toml"))
    entries = surface["family"]
    badge(status) = Dict(
        "strict"       => "🟢 strict",
        "generic"      => "🔵 generic",
        "raw-only"     => "🟠 raw-only",
        "out-of-scope" => "⚪ out-of-scope",
    )[status]
    io = IOBuffer()
    print(io, """
        # Instruction-surface coverage

        *This page is generated from `docs/SURFACE.toml` at docs-build time; edit
        that file, not this page. `test/host/surface.jl` machine-checks the
        inventory against the form registry, the typed-wrapper-only rules,
        and the wrapper registry.*

        Every PTX ISA 9.4 instruction family (§9.7.1–§9.7.21) has an explicit
        disposition, so textual expressibility is never mistaken for reviewed
        coverage. The four statuses:

        - **strict** — closed, reviewed schemas and/or typed wrappers;
          an unreviewed spelling fails loudly before reaching LLVM.
        - **generic** — the chain default renders the form under a reviewed
          `FORMS` contract (purity/convergence/brackets/return promises).
        - **raw-only** — no `FORMS` entry or wrapper exists; the explicit
          `ptx"..."raw` escape hatch (maximally conservative contract) is the
          only route, and the entry's note states its evidence boundary.
        - **out-of-scope** — deliberately unsupported: no contract, no
          wrapper, and raw use is unsanctioned (unsound or structurally
          inexpressible as a single inline-asm call).

        A status describes *binding*, not proof. Evidence climbs a ladder:
        **spelled** (the form renders) → **assembles** (ptxas accepts it, the
        `ptxas/` tier) → **executes** (a live device runs it, the `gpu/`
        tier) → **numerically validated** (its results are checked against a
        reference). Entries marked *verify* carry an unresolved question —
        see their notes.

        """)
    bysec = Dict{Int, Vector{Dict{String, Any}}}()
    for e in entries
        push!(get!(Vector{Dict{String, Any}}, bysec,
                   parse(Int, split(e["section"], '.')[3])), e)
    end
    section_title(group) = begin
        tops = [e["title"] for e in group if count('.', e["section"]) == 2]
        isempty(tops) ? split(first(group)["title"], ':')[1] : first(tops)
    end
    for n in sort!(collect(keys(bysec)))
        group = bysec[n]
        println(io, "## §9.7.$n ", section_title(group), "\n")
        println(io, "| Section | Opcodes | Status | Notes |")
        println(io, "|---|---|---|---|")
        for e in group
            ops = isempty(e["opcodes"]) ? "—" :
                  join(("`$op`" for op in e["opcodes"]), ", ")
            status = badge(e["status"]) *
                     (get(e, "verify", false) ? " ⚠ verify" : "")
            notes = replace(e["notes"], "|" => "\\|")
            println(io, "| §", e["section"], " | ", ops, " | ", status,
                    " | ", notes, " |")
        end
        println(io)
    end
    # The hardware-evidence ledger (test/EVIDENCE.toml): which runtime tiers
    # CI exercises continuously, and when the manual tiers last ran on
    # hardware.
    evidence = TOML.parsefile(joinpath(pkgdir(PTX), "test", "EVIDENCE.toml"))
    println(io, "## Hardware evidence\n")
    println(io, """
        The `gpu/hopper` and `gpu/blackwell` runtime tiers run in no CI
        lane; their evidence comes from manual cloud sessions recorded in
        `test/EVIDENCE.toml` (the test manifest prints these records — and
        what has changed since the recorded tree — whenever it skips a
        tier). Dates matter: a record older than the tree under test is
        *spelled/assembles* evidence only.
        """)
    println(io, "| Tier | Evidence | Last validated | Device | Tree | Suite |")
    println(io, "|---|---|---|---|---|---|")
    for t in evidence["tier"]
        println(io, "| `", t["tests"], "` (",
                replace(t["runtime"], "|" => "\\|"), ") | ",
                t["evidence"], " | ", get(t, "last_validated", "—"), " | ",
                t["device"], " | ",
                haskey(t, "tree") ? "`$(t["tree"])`" : "—", " | ",
                get(t, "suite", "—"), " |")
    end
    write(joinpath(@__DIR__, "src", "coverage.md"), take!(io))
end
write_coverage_page()

makedocs(;
    modules=[PTX, PTX.IR, PTX.Codegen, PTX.Parser, PTX.MBarriers,
             PTX.Pipelines, PTX.Warps, PTX.Utils],
             PTX.Pipelines, PTX.Warps],
    authors="Anton Oresten <antonoresten@proton.me>",
    sitename="PTX.jl",
    format=Documenter.HTML(;
        canonical="https://docs.jool.space/PTX.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Chain DSL" => "dsl.md",
        "Wrappers" => "wrappers.md",
        "Coverage" => "coverage.md",
        "Barriers & pipelines" => "barriers.md",
        "Kernel utilities" => "utils.md",
        "Transpiler" => "transpiler.md",
        "Reference" => "reference.md",
        "Internals" => "internals.md",
        "Extending" => "extending.md",
    ],
)

deploydocs(;
    repo="github.com/jool-space/PTX.jl",
    deploy_repo="github.com/jool-space/docs",
    devbranch="main",
    dirname="PTX.jl",
)
