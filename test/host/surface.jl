# SURFACE.toml enforcement — keeps the instruction-surface inventory honest
# (issue #55). The inventory is the reviewed artifact; this file asserts it
# against the live code in both directions so a FORMS entry, a typed-wrapper
# rule, or a wrapper family can never appear or disappear without the
# inventory (and therefore the generated coverage page) noticing.
#
# Deliberately NOT an oracle in the count-pin sense: SURFACE.toml is policy
# documentation, and this test checks *consistency* with src/, not grammar
# reconstruction. The double-entry oracles for form grammars stay in their
# per-ledger test files.

using TOML

const SURFACE_PATH = joinpath(pkgdir(PTX), "docs", "SURFACE.toml")
const SURFACE_STATUSES = ("strict", "generic", "raw-only", "out-of-scope")

@testset "SURFACE.toml: the instruction-surface inventory" begin
    surface = TOML.parsefile(SURFACE_PATH)
    entries = get(surface, "family", nothing)
    @test entries isa Vector
    @test !isempty(entries)

    @testset "entry shape and closed status vocabulary" begin
        for e in entries
            label = string(get(e, "section", "?"), " ", get(e, "title", "?"))
            for field in ("section", "title", "status", "notes")
                @test haskey(e, field) || error("entry $label missing `$field`")
            end
            @test haskey(e, "opcodes") ||
                  error("entry $label missing `opcodes`")
            @test e["status"] in SURFACE_STATUSES ||
                  error("entry $label has status `$(e["status"])` outside " *
                        "$(SURFACE_STATUSES)")
            @test occursin(r"^9\.7\.\d+(\.\d+)*$", e["section"]) ||
                  error("entry $label has malformed section `$(e["section"])`")
        end
    end

    # opcode => owning entry; duplicates are a structural failure (one opcode
    # belongs to exactly one entry).
    owner = Dict{String, Dict{String, Any}}()
    @testset "one opcode, one entry" begin
        duplicated = String[]
        for e in entries, op in e["opcodes"]
            haskey(owner, op) && push!(duplicated, op)
            owner[op] = e
        end
        @test isempty(duplicated) ||
              error("opcodes owned by more than one entry: $duplicated")
    end

    @testset "(a) every FORMS opcode is inventoried exactly once" begin
        unowned = sort!([String(k) for k in keys(PTX.FORMS)
                         if !haskey(owner, String(k))])
        @test isempty(unowned) ||
              error("FORMS opcodes missing from SURFACE.toml: $unowned")
    end

    @testset "status is consistent with FORMS membership" begin
        # generic = chain-default with a FORMS contract, so every opcode in a
        # generic entry must be a FORMS key; raw-only/out-of-scope mean *no*
        # FORMS entry exists.
        formkeys = Set(String(k) for k in keys(PTX.FORMS))
        for e in entries
            wrong = if e["status"] == "generic"
                [op for op in e["opcodes"] if !(op in formkeys)]
            elseif e["status"] in ("raw-only", "out-of-scope")
                [op for op in e["opcodes"] if op in formkeys]
            else
                String[]
            end
            @test isempty(wrong) ||
                  error("entry $(e["section"]) `$(e["title"])` ($(e["status"])): " *
                        "opcodes with contradictory FORMS membership: $wrong")
        end
    end

    @testset "(b) typed-wrapper-only ops live in strict entries" begin
        for rule in PTX.TYPED_WRAPPER_ONLY_RULES
            op = String(rule.op)
            @test haskey(owner, op) ||
                  error("TYPED_WRAPPER_ONLY op `$op` missing from SURFACE.toml")
            haskey(owner, op) || continue
            @test owner[op]["status"] == "strict" ||
                  error("TYPED_WRAPPER_ONLY op `$op` is in a " *
                        "`$(owner[op]["status"])` entry; must be `strict`")
        end
    end

    @testset "(c) every wrapper-registry family maps to an entry" begin
        families = unique!([r.family for r in PTX.WRAPPER_REGISTRY])
        @test !isempty(families)   # the registry populates at load time
        for fam in families
            ops = unique!([String(r.op) for r in PTX.wrapper_records(fam)])
            covered = [op for op in ops if haskey(owner, op)]
            @test !isempty(covered) ||
                  error("wrapper family `$fam` (ops: $ops) maps to no " *
                        "SURFACE.toml entry")
        end
    end

    @testset "(d) all 21 ISA §9.7.x families are covered" begin
        present = Set{Int}()
        for e in entries
            parts = split(e["section"], '.')
            length(parts) >= 3 && push!(present, parse(Int, parts[3]))
        end
        missing_sections = sort!([n for n in 1:21 if !(n in present)])
        @test isempty(missing_sections) ||
              error("ISA sections with no SURFACE.toml entry: " *
                    join(("9.7.$n" for n in missing_sections), ", "))
        stray = sort!(collect(setdiff(present, 1:21)))
        @test isempty(stray) ||
              error("SURFACE.toml entries for nonexistent §9.7.x sections: $stray")
    end
end
