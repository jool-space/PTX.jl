using PTX: IR
using PTX.Parser: parse as parse_ptx
using PTX.IR: format

CORPUS_DIR = joinpath(@__DIR__, "..", "corpus")

# Top-level corpus = the hand-curated kernels. It is the strict oracle tier:
# lossless acceptance plus deep structural fixed-point and semantic checks.
CORPUS_FILES = sort(filter(p -> endswith(p, ".ptx"),
                           readdir(CORPUS_DIR; join = true)))

# A byte-loss smoke is intentionally weaker than the structural and semantic
# checks below. Most curated inputs must stay within the historical 3% limit.
# `less_slow_sm90a` currently needs 3.1% after raw snapshots are removed:
# header-inline comments, comma-packed register declarations, and multiple
# statements on one physical line are not all represented by the current IR.
# Keep that exception local and reviewable rather than weakening the corpus
# gate as a whole.
const DEFAULT_CURATED_BYTE_LOSS_LIMIT = 0.03
const CURATED_BYTE_LOSS_LIMITS = Dict(
    "less_slow_sm90a.ptx" => 0.031,
)

@testset "curated deep byte-loss exception manifest" begin
    curated_names = Set(basename.(CORPUS_FILES))
    @test Set(keys(CURATED_BYTE_LOSS_LIMITS)) ⊆ curated_names
    @test Set(keys(CURATED_BYTE_LOSS_LIMITS)) == Set(("less_slow_sm90a.ptx",))
end

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

# These external Triton kernels currently exercise the intentional RawLine
# fallback. Keep them in the acceptance/lossless corpus, but do not present
# their raw-text re-emission as structural coverage. The inventory is a
# deliberate review point: parser progress (or a regression) changes it only
# with an explicit manifest update.
const EXTERNAL_RAWLINE_MANIFEST = Dict(
    "triton/fa_ws_pingpong_sm90a.ptx" => 494,
    "triton/matmul_tma_sm120a.ptx" => 233,
    "triton/matmul_tma_v33_sm90a.ptx" => 240,
    "triton/matmul_tma_v34_sm90a.ptx" => 241,
    "triton/matmul_tma_v35_sm90a.ptx" => 244,
    "triton/matmul_wgmma_v32_sm90a.ptx" => 245,
)

const EXTERNAL_STRUCTURAL_FILES = filter(EXTERNAL_FILES) do path
    !haskey(EXTERNAL_RAWLINE_MANIFEST, relpath(path, EXTERNAL_DIR))
end

function _deep_structural_roundtrip(src::String, name::String;
                                    semantic::Bool = true,
                                    byte_loss_limit::Union{Nothing, Float64} = nothing)
    parsed = parse_ptx(src)
    _assert_no_rawlines(parsed, name)

    first_ir = _deep_unraw(parsed)
    @test first_ir.raw_source === nothing
    @test first_ir.raw_header === nothing
    @test isempty(_raw_snapshot_paths(first_ir))
    first_text = format(first_ir)

    reparsed = parse_ptx(first_text)
    _assert_no_rawlines(reparsed, "$name after deep structural re-emit")
    second_ir = _deep_unraw(reparsed)
    @test isempty(_raw_snapshot_paths(second_ir))
    second_text = format(second_ir)

    # Fixed-point checks formatting. The curated tier also runs the more
    # expensive semantic module diff; the broad external tier is a structural
    # stress suite and would turn that O(n) IR walk into an impractical
    # repeated-normalization cost for large compiler outputs.
    @test first_text == second_text
    if byte_loss_limit !== nothing
        byte_loss = abs(length(first_text) - length(src)) / length(src)
        @test byte_loss < byte_loss_limit
    end
    if semantic
        @test isempty(IR.diff(first_ir, second_ir))
    end
    first_text
end

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

# Tier 2 — deep structural round-trip. This removes module source/header
# snapshots and every nested statement snapshot, then requires field-driven
# reconstruction to parse, preserve structure, and reach a fixed point.
# Curated fixtures are deliberately held to zero RawLine fallback nodes.
@testset "deep structural round-trip: $(basename(path))" for path in CORPUS_FILES
    limit = get(CURATED_BYTE_LOSS_LIMITS, basename(path),
                DEFAULT_CURATED_BYTE_LOSS_LIMIT)
    _deep_structural_roundtrip(read(path, String), "curated/$(basename(path))";
                               byte_loss_limit = limit)
end

# A deep structural projection must not erase a fallback node. This is a
# regression for the old golden guard, which checked only direct function-body
# children and missed RawLine inside nested Block/IntrinsicScope containers.
@testset "structural oracle: recursive RawLine and raw snapshots" begin
    nested = PTX.IR.RawLine("opaque.nested;" )
    scope = PTX.IR.IntrinsicScope(
        name = "scope",
        args_repr = "",
        body = (nested,),
        formatting = PTX.IR.FormattingInfo(raw_line = "raw scope"),
    )
    block = PTX.IR.Block(
        body = (scope,),
        formatting = PTX.IR.FormattingInfo(raw_line = "raw block"),
    )
    fn = PTX.IR.Function(
        is_entry = true,
        name = "nested_rawline",
        body = (block,),
        formatting = PTX.IR.FormattingInfo(raw_line = "raw function"),
    )
    nested_module = PTX.IR.Module(
        version = PTX.IR.Version(8, 0),
        target = PTX.IR.Target(("sm_89",)),
        address_size = PTX.IR.AddressSize(64),
        directives = (fn,),
        raw_header = ".version 8.0\n.target sm_89\n.address_size 64",
        raw_source = "raw module",
    )

    nested_path = "module.directives[1].body[1].body[1].body[1]"
    @test _rawline_count(nested_module) == 1
    @test _rawline_paths(nested_module) == [nested_path => "opaque.nested;"]
    err = try
        _assert_structural(nested_module, "nested RawLine")
        nothing
    catch error
        error
    end
    @test err isa ErrorException
    @test occursin(nested_path, sprint(showerror, err))

    nested_deep = _deep_unraw(nested_module)
    @test nested_deep.raw_source === nothing
    @test nested_deep.raw_header === nothing
    @test _rawline_paths(nested_deep) == _rawline_paths(nested_module)
    @test isempty(_raw_snapshot_paths(nested_deep))

    raw_formatting = PTX.IR.FormattingInfo(raw_line = "\tmov.u32 %r0, 1;")
    changed = PTX.IR.Instruction(
        opcode = "add",
        modifiers = (".u32",),
        operands = (PTX.IR.RegisterOperand("%r0"),
                    PTX.IR.RegisterOperand("%r1"),
                    PTX.IR.RegisterOperand("%r2")),
        formatting = raw_formatting,
    )
    mutation_module = PTX.IR.Module(
        version = PTX.IR.Version(8, 0),
        target = PTX.IR.Target(("sm_89",)),
        address_size = PTX.IR.AddressSize(64),
        directives = (PTX.IR.Function(is_entry = true, name = "mutation",
                                      body = (changed,)),),
    )
    @test occursin("mov.u32 %r0, 1", format(IR.unraw(mutation_module)))
    mutation_deep = _deep_unraw(mutation_module)
    @test isempty(_raw_snapshot_paths(mutation_deep))
    @test occursin("add.u32 %r0, %r1, %r2", format(mutation_deep))
    @test !occursin("mov.u32 %r0, 1", format(mutation_deep))
end

# --- external corpus (LLVM unit tests, Triton, compiler outputs) ---------
#
# Wider, real-world PTX. Every file remains in the lossless acceptance tier.
@testset "external lossless: $(relpath(path, EXTERNAL_DIR))" for path in EXTERNAL_FILES
    src = read(path, String)
    local m::IR.Module
    @test (m = parse_ptx(src); true)
    @test format(m) == src
end

# The manifest makes exclusion from deep structural evidence auditable rather
# than allowing a broad external testset to quietly inherit raw snapshots.
@testset "external RawLine fallback inventory" begin
    observed = Dict{String, Int}()
    for path in EXTERNAL_FILES
        count = _rawline_count(parse_ptx(read(path, String)))
        count == 0 || (observed[relpath(path, EXTERNAL_DIR)] = count)
    end
    @test observed == EXTERNAL_RAWLINE_MANIFEST
end

# Only fallback-free external inputs provide deep structural evidence. A new
# fallback in this partition fails `_assert_no_rawlines` instead of silently
# dropping to a weaker test tier.
@testset "external deep structural: $(relpath(path, EXTERNAL_DIR))" for path in EXTERNAL_STRUCTURAL_FILES
    _deep_structural_roundtrip(read(path, String),
                               "external/$(relpath(path, EXTERNAL_DIR))";
                               semantic = false)
end
