using TOML

const FINDING_STATUSES = Set([
    "confirmed",
    "needs_experiment",
    "deferred",
    "resolved",
])
const FINDING_SEVERITIES = Set(["critical", "high", "medium", "low"])
const FINDINGS_DIR = normpath(joinpath(@__DIR__, "..", "..", "audit", "findings"))
const FINDING_KEYS = Set([
    "id",
    "status",
    "severity",
    "category",
    "summary",
    "source_handoffs",
    "validation_tier",
])
const RESOLUTION_KEYS = Set([
    "pr",
    "commit",
    "scope",
    "evidence",
    "merged_at",
])

# This inventory is deliberately independent of the records it protects. A
# deleted TOML must fail instead of silently shrinking the set the loop sees;
# adding or splitting a finding therefore requires an explicit audit review.
const EXPECTED_FINDING_IDS = Set([
    "ADDRESS-001",
    "ASYNC-COVERAGE-001",
    "B128-COVERAGE-001",
    "BF16-PTR-EXP",
    "CARRY-001",
    "CLC-TRY-001",
    "CVT-IMMEDIATE-001",
    "CVTPACK-001",
    "DOCS-001",
    "FABRIC-EXP",
    "FALLBACK-001",
    "FRONT-DECL-001",
    "FRONT-DIFF-001",
    "FRONT-HEADER-001",
    "FRONT-LEXER-001",
    "FRONT-OPERAND-001",
    "FRONT-STRUCTURAL-001",
    "FRONT-TRANSPILER-001",
    "FRONT-VECTOR-001",
    "IMMEDIATE-001",
    "LOWERING-REFLECT-EXP",
    "MAPA-U64-EXP",
    "MATRIX-AMBIG-EXP",
    "MATRIX-COVERAGE-001",
    "MBARRIER-001",
    "MBARRIER-DOC-001",
    "NVVM-RET-META-001",
    "NVVM-SIDEFX-001",
    "SCALAR-RESULT-001",
    "SREG-001",
    "STRUCTURED-RESULT-001",
    "TCGEN-DESC-001",
    "TCGEN-MX-001",
    "TEST-TARGET-001",
    "UNSUPPORTED-SURFACE-001",
    "VECTOR-RESULT-001",
    "WGMMA-SHAPE-001",
    "WMMA-001",
])

# These findings were resolved before the tracked ledger was introduced.
# Future resolutions extend `resolved_ids`; this floor prevents a backfilled
# resolution from being downgraded or losing its provenance unnoticed.
const HISTORICAL_RESOLVED_FINDING_IDS = Set([
    "CARRY-001",
    "FRONT-DIFF-001",
    "FRONT-STRUCTURAL-001",
    "SREG-001",
    "WMMA-001",
])

function _required_nonempty_string(record, key)
    @test haskey(record, key)
    haskey(record, key) || return nothing
    value = record[key]
    @test value isa String
    value isa String || return nothing
    @test !isempty(strip(value))
    value
end

function _required_nonempty_strings(record, key)
    @test haskey(record, key)
    haskey(record, key) || return nothing
    values = record[key]
    @test values isa AbstractVector
    values isa AbstractVector || return nothing
    @test !isempty(values)
    @test all(value -> value isa String && !isempty(strip(value)), values)
    values
end

@testset "tracked audit finding ledger" begin
    @test isdir(FINDINGS_DIR)
    files = sort(filter(path -> endswith(path, ".toml"),
                        readdir(FINDINGS_DIR; join = true)))
    @test !isempty(files)

    seen_ids = Set{String}()
    resolved_ids = Set{String}()
    for file in files
        filename_id = splitext(basename(file))[1]
        @testset "$filename_id" begin
            record = TOML.parsefile(file)
            id = _required_nonempty_string(record, "id")
            if id !== nothing
                @test id == filename_id
                @test id ∉ seen_ids
                push!(seen_ids, id)
            end

            status = _required_nonempty_string(record, "status")
            status === nothing || @test status in FINDING_STATUSES
            if id !== nothing && status == "resolved"
                push!(resolved_ids, id)
            end
            severity = _required_nonempty_string(record, "severity")
            severity === nothing || @test severity in FINDING_SEVERITIES
            _required_nonempty_string(record, "category")
            _required_nonempty_string(record, "summary")
            _required_nonempty_strings(record, "source_handoffs")
            _required_nonempty_string(record, "validation_tier")

            if status == "resolved"
                @test Set(keys(record)) == union(FINDING_KEYS, Set(["resolution"]))
                @test haskey(record, "resolution")
                if haskey(record, "resolution")
                    resolution = record["resolution"]
                    @test resolution isa AbstractDict
                    if resolution isa AbstractDict
                        @test Set(keys(resolution)) ⊆ RESOLUTION_KEYS
                        pr = get(resolution, "pr", nothing)
                        @test pr isa Integer && !(pr isa Bool)
                        if pr isa Integer && !(pr isa Bool)
                            @test pr > 0
                        end

                        commit = _required_nonempty_string(resolution, "commit")
                        commit === nothing ||
                            @test match(r"^[0-9a-f]{40}$", commit) !== nothing
                        _required_nonempty_string(resolution, "scope")
                        _required_nonempty_strings(resolution, "evidence")

                        if haskey(resolution, "merged_at")
                            merged_at = resolution["merged_at"]
                            @test merged_at isa String
                            if merged_at isa String
                                @test !isempty(strip(merged_at))
                            end
                        end
                    end
                end
            else
                @test Set(keys(record)) == FINDING_KEYS
            end
        end
    end

    @test length(seen_ids) == length(files)
    @test seen_ids == EXPECTED_FINDING_IDS
    @test HISTORICAL_RESOLVED_FINDING_IDS ⊆ resolved_ids
end
