using PTX: IR
using PTX.Parser: parse as parse_ptx
using PTX.IR: format

CORPUS_DIR = joinpath(@__DIR__, "..", "corpus")

# Top-level corpus = the original 10 hand-curated kernels. Used for the
# strict three-tier round-trip suite (lossless / structural fixed-point /
# size-delta < 3%).
CORPUS_FILES = sort(filter(p -> endswith(p, ".ptx"),
                           readdir(CORPUS_DIR; join = true)))

# External corpus = real-world kernels lifted from LLVM unit tests, Triton
# emission, and other compilers. Larger (~200 files, ~22K LOC). Stress-
# tests the parser on operand patterns the hand-curated set doesn't cover.
EXTERNAL_DIR = joinpath(CORPUS_DIR, "external")
function _gather_ptx_files(dir)
    files = String[]
    isdir(dir) || return files
    for entry in readdir(dir; join = true)
        if isdir(entry)
            append!(files, _gather_ptx_files(entry))
        elseif endswith(entry, ".ptx")
            push!(files, entry)
        end
    end
    sort(files)
end
EXTERNAL_FILES = _gather_ptx_files(EXTERNAL_DIR)

# Tier 1 — lossless byte-identical round-trip via raw_source.
# The parser always populates raw_source; format() returns it verbatim.
# Certifies that the parser accepts the input — the round-trip is trivial
# but the parsing isn't.
@testset "lossless round-trip: $(basename(path))" for path in CORPUS_FILES
    src = read(path, String)
    local m::IR.Module
    @test (m = parse_ptx(src); true)
    @test format(m) == src
end

# Tier 2 — structural fixed point. IR.unraw() disables the raw_source
# short-circuit so format() takes the structural walk through children.
# Output differs slightly from the input (trailing-newline / blank-line-
# with-whitespace cosmetics) but re-parsing then re-emitting must
# converge. Catches container-node bugs (Function/Block headers don't
# carry raw_line, so structural emission has to be self-consistent).
@testset "structural fixed-point: $(basename(path))" for path in CORPUS_FILES
    src  = read(path, String)
    m1   = parse_ptx(src)
    out1 = format(IR.unraw(m1))
    out2 = format(IR.unraw(parse_ptx(out1)))
    @test out1 == out2
end

# Tier 3 — close-to-source check: structural emission shouldn't lose
# more than ~3% of bytes vs. the input. This is a smoke test for
# regression — if format() suddenly drops a whole instruction class, the
# delta would balloon. The 3% slack covers known cosmetic gaps (trailing
# newline, multi-statement-per-line splitting).
@testset "structural size delta: $(basename(path))" for path in CORPUS_FILES
    src  = read(path, String)
    out1 = format(IR.unraw(parse_ptx(src)))
    @test abs(length(out1) - length(src)) / length(src) < 0.03
end

# --- external corpus (LLVM unit tests, Triton, compiler outputs) ---------
#
# Wider, real-world PTX. Three tiers, same as the curated set:
#   1. lossless round-trip via raw_source (parser accepts);
#   2. structural fixed-point with raw_source stripped (formatter is
#      self-consistent across re-emit);
#   3. size delta < 3% (formatter doesn't drop instruction classes).
@testset "external/$(relpath(path, EXTERNAL_DIR))" for path in EXTERNAL_FILES
    src = read(path, String)
    local m::IR.Module
    @test (m = parse_ptx(src); true)
    @test format(m) == src                                # tier 1

    out1 = format(IR.unraw(m))
    out2 = format(IR.unraw(parse_ptx(out1)))
    @test out1 == out2                                    # tier 2

    @test abs(length(out1) - length(src)) / length(src) < 0.03   # tier 3
end
