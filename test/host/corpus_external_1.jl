# External-corpus shard 1 of 6 — lossless acceptance, RawLine-manifest
# inventory, and deep structural round-trip for this shard's slice of the
# real-world corpus. The slicing, manifest, and roundtrip helper live in
# test/setup.jl (external-corpus sweep support); host/corpus.jl pins that
# the six slices partition the corpus exactly.

using PTX: IR
using PTX.Parser: parse as parse_ptx
using PTX.IR: format

const _SHARD_FILES = external_corpus_shard(1)

@testset "external lossless: $(relpath(path, EXTERNAL_SWEEP_DIR))" for path in _SHARD_FILES
    src = read(path, String)
    if _external_header_requires_repair(src)
        error = try parse_ptx(src); nothing catch caught; caught end
        @test error isa PTX.Parser.ParseError
        @test occursin("requires PTX ISA", sprint(showerror, error))
        tokens = PTX.Parser.tokenize(src)
        @test join(t.leading_whitespace * t.text for t in tokens
                   if t.kind != PTX.Parser.TokenKind.EOF) == src
    end
    parser_src = _external_parser_source(src)
    local m::IR.Module
    @test (m = parse_ptx(parser_src); true)
    @test format(m) == parser_src
end

# The manifest makes exclusion from deep structural evidence auditable rather
# than allowing a broad external testset to quietly inherit raw snapshots.
# Each shard checks its slice; the slices partition the corpus.
@testset "external RawLine fallback inventory (shard 1)" begin
    observed = Dict{String, Int}()
    for path in _SHARD_FILES
        count = _rawline_count(parse_ptx(_external_parser_source(read(path, String))))
        count == 0 || (observed[relpath(path, EXTERNAL_SWEEP_DIR)] = count)
    end
    expected = Dict(rel => n for (rel, n) in EXTERNAL_SWEEP_RAWLINE_MANIFEST
                    if joinpath(EXTERNAL_SWEEP_DIR, split(rel, '/')...) in _SHARD_FILES)
    @test observed == expected
end

# Only fallback-free external inputs provide deep structural evidence. A new
# fallback in this shard fails `_assert_no_rawlines` instead of silently
# dropping to a weaker test tier.
@testset "external deep structural: $(relpath(path, EXTERNAL_SWEEP_DIR))" for path in filter(
        p -> !haskey(EXTERNAL_SWEEP_RAWLINE_MANIFEST, relpath(p, EXTERNAL_SWEEP_DIR)),
        _SHARD_FILES)
    _deep_structural_roundtrip(_external_parser_source(read(path, String)),
                               "external/$(relpath(path, EXTERNAL_SWEEP_DIR))";
                               semantic = false)
end
