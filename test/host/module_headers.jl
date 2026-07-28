using PTX: IR
using PTX.Parser: parse as parse_ptx, ParseError
using PTX.IR: Target, AddressSize, TargetDirective,
              format, normalize, canonicalize, diff

header(version, target; address = nothing, tail = "") =
    ".version $version\n.target $target\n" *
    (address === nothing ? "" : ".address_size $address\n") * tail

@testset "module header: optional address size has an explicit semantic default" begin
    source = header("8.0", "sm_89"; tail = "// after header\n")
    module_ir = parse_ptx(source)
    @test module_ir.address_size == AddressSize(32)
    @test !module_ir.address_size_explicit
    @test format(module_ir) == source

    structural = _deep_unraw(module_ir)
    emitted = format(structural)
    @test !occursin(".address_size", emitted)
    reparsed = parse_ptx(emitted)
    @test !reparsed.address_size_explicit
    @test isempty(diff(structural, _deep_unraw(reparsed)))
    @test format(_deep_unraw(reparsed)) == emitted

    explicit = parse_ptx(header("8.0", "sm_89"; address = 32))
    @test explicit.address_size == AddressSize(32)
    @test explicit.address_size_explicit
    @test occursin(".address_size 32", format(_deep_unraw(explicit)))
    @test !isempty(diff(normalize(module_ir), normalize(explicit)))
end

@testset "module header: complete PTX 9.4 architecture spelling ledger" begin
    observed = Set{String}()
    for (_, suffixes) in TEST_HEADER_TARGET_FLOORS, suffix in suffixes
        for prefix in ("sm_", "compute_")
            spelling = prefix * suffix
            push!(observed, spelling)
            module_ir = parse_ptx(header("9.4", spelling))
            @test module_ir.target == Target((spelling,))
        end
    end
    @test length(observed) == 92

    err = try
        parse_ptx(header("8.5", "sm_100a"))
        nothing
    catch caught
        caught
    end
    @test err isa ParseError
    @test occursin("requires PTX ISA 8.6", sprint(showerror, err))
end

@testset "module header: target grammar and explicit version constraints" begin
    positives = (
        ("3.0", "sm_20, debug"),
        ("1.5", "sm_13, texmode_independent"),
        ("2.0", "compute_20, map_f64_to_f32"),
        ("9.3", "sm_110f"),
        ("9.3", "sm_121a, debug"),
        ("9.4", "sm_107f"),
        ("9.4", "sm_107a, debug"),
    )
    for (version, target) in positives
        @test parse_ptx(header(version, target)).target.targets[1] ==
              split(target, ',')[1]
    end

    negatives = (
        ("9.4", "sm_130"),
        ("9.4", "gfx_90"),
        ("9.3", "sm_107"),
        ("9.3", "debug, sm_90"),
        ("9.3", "sm_90, sm_89"),
        ("9.3", "sm_90, debug, debug"),
        ("9.3", "sm_90, mystery"),
        ("9.3", "sm_90, texmode_unified, texmode_independent"),
        ("2.3", "sm_20, debug"),
        ("1.4", "sm_13, texmode_unified"),
        ("1.5", "sm_13, map_f64_to_f32"),
    )
    for (version, target) in negatives
        @test_throws ParseError parse_ptx(header(version, target))
    end
    @test_throws ParseError parse_ptx(".version 9.3\n.target sm_90 debug\n")

    # A bundled 9.4 ledger must not pretend to describe a future ISA.
    future = parse_ptx(header("9.5", "sm_130f, future_platform_option"))
    @test future.target.targets == ("sm_130f", "future_platform_option")
    @test_throws ParseError parse_ptx(header("9.5", "sm_130f, sm_131"))
end

@testset "module header: later target directives retain order and structure" begin
    source = """.version 9.3
.target sm_90, texmode_unified
.address_size 64
.global .u32 before;
.target sm_100f, texmode_unified
.global .u32 after;
.target compute_110f
"""
    module_ir = parse_ptx(source)
    target_nodes = filter(d -> d isa TargetDirective, module_ir.directives)
    @test getfield.(target_nodes, :target) ==
        (Target(("sm_100f", "texmode_unified")), Target(("compute_110f",)))
    @test findfirst(==(target_nodes[1]), module_ir.directives) <
          findfirst(d -> d isa IR.VarDecl && d.name == "after", module_ir.directives)
    @test format(module_ir) == source

    structural = _deep_unraw(module_ir)
    normalized = normalize(module_ir)
    canon = canonicalize(module_ir)
    @test count(d -> d isa TargetDirective, normalized.directives) == 2
    @test count(d -> d isa TargetDirective, canon.directives) == 2
    @test getfield.(filter(d -> d isa TargetDirective, canon.directives), :target) ==
          getfield.(target_nodes, :target)
    @test canon.address_size_explicit
    emitted = format(structural)
    reparsed = _deep_unraw(parse_ptx(emitted))
    @test isempty(diff(structural, reparsed))
    @test canonicalize(structural).directives == canonicalize(reparsed).directives
    @test format(reparsed) == emitted

    changed = replace(source, "sm_100f, texmode_unified" =>
                              "sm_100f, texmode_independent")
    @test_throws ParseError parse_ptx(changed)
end

@testset "module header: duplicate, placement, and scope rejections" begin
    invalid = (
        ".target sm_89\n",
        ".version 8.0\n.address_size 64\n.target sm_89\n",
        header("8.0", "sm_89"; tail = ".version 8.1\n"),
        header("8.0", "sm_89"; address = 64,
               tail = ".address_size 64\n"),
        header("8.0", "sm_89"; tail = ".global .u32 x;\n.address_size 64\n"),
        header("8.0", "sm_89"; tail =
               ".entry f()\n{\n  .target sm_90\n}\n"),
    )
    for source in invalid
        @test_throws ParseError parse_ptx(source)
    end

    for size in (0, 31, 33, 63, 65, 128)
        @test_throws ParseError parse_ptx(header("8.0", "sm_89"; address = size))
    end
    @test_throws ParseError parse_ptx(header("8.0", "sm_89"; address = "0x20"))
    @test_throws ParseError parse_ptx(".version 8.0\n.target sm_89\n.address_size 64 junk\n")
    @test_throws ParseError parse_ptx(header("2.2", "sm_20"; address = 32))
    @test_throws ArgumentError AddressSize(16)

    commented = ".version 8.0\n// language selected\n.target sm_89\n" *
                "// target selected\n.address_size 64\n"
    @test parse_ptx(commented).address_size == AddressSize(64)
end

@testset "module header: committed external corpus remains lossless" begin
    external = joinpath(@__DIR__, "..", "corpus", "external")
    files = sort(filter(path -> endswith(path, ".ptx"),
                        [joinpath(root, file)
                         for (root, _, names) in walkdir(external) for file in names]))
    @test !isempty(files)
    invalid_count = 0
    for path in files
        source = read(path, String)
        if _external_header_requires_repair(source)
            invalid_count += 1
            error = try parse_ptx(source); nothing catch caught; caught end
            @test error isa ParseError
            @test occursin("requires PTX ISA", sprint(showerror, error))
            tokens = PTX.Parser.tokenize(source)
            @test join(t.leading_whitespace * t.text for t in tokens
                       if t.kind != PTX.Parser.TokenKind.EOF) == source
        end
        repaired = _external_parser_source(source)
        @test format(parse_ptx(repaired)) == repaired
    end
    @test invalid_count == 89
end
