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
# Tracked in issue #48: `less_slow_sm90a` currently needs 3.1%
# after raw snapshots are removed. Header-inline comments, comma-packed
# register declarations, and multiple statements on one physical line are not
# all represented by the current IR. Keep this exception local and reviewable
# rather than weakening the corpus gate as a whole.
const DEFAULT_CURATED_BYTE_LOSS_LIMIT = 0.03
const CURATED_BYTE_LOSS_LIMITS = Dict(
    "less_slow_sm90a.ptx" => 0.031,
)

@testset "curated deep byte-loss exception manifest" begin
    curated_names = Set(basename.(CORPUS_FILES))
    @test all(name -> name in curated_names, keys(CURATED_BYTE_LOSS_LIMITS))
end

# External corpus = real-world kernels lifted from LLVM unit tests, Triton
# emission, and other compilers. Larger (~200 files, ~22K LOC). Stress-
# tests the parser on operand patterns the hand-curated set doesn't cover.
EXTERNAL_DIR = joinpath(CORPUS_DIR, "external")
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
# Wider, real-world PTX. Invalid synthetic LLVM headers remain byte-lossless at
# the lexer tier and must reject for their exact version/target mismatch. Their
# bodies stay in parser coverage through an in-memory minimum-version repair;
# the committed provenance fixture is never rewritten.
# The external per-file evidence lives in host/corpus_external_* (see
# test/setup.jl, external-corpus sweep support). Pin here, once, that the
# sweep exists, that the shard slices partition it exactly, and that the
# RawLine manifest names real files — so a shard-count edit or a corpus
# rename cannot silently drop coverage.
@testset "external corpus shard partition is exact" begin
    @test !isempty(EXTERNAL_SWEEP_FILES)
    sharded = sort(reduce(vcat, [external_corpus_shard(i)
                                 for i in 1:EXTERNAL_SWEEP_SHARDS]))
    @test sharded == EXTERNAL_SWEEP_FILES
    rels = Set(relpath(path, EXTERNAL_SWEEP_DIR) for path in EXTERNAL_SWEEP_FILES)
    @test all(name -> name in rels, keys(EXTERNAL_SWEEP_RAWLINE_MANIFEST))
end
