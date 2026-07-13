using .TestTargets

@testset "structured test target metadata" begin
    @testset "strict parser" begin
        floor = parse_test_requirement(
            "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.9")
        @test floor.requires === :gpu
        @test floor.evidence === :runtime
        @test only(floor.runtime) == CapabilityPredicate(:minimum, v"8.9")

        family = parse_test_requirement(
            "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc==10")
        @test only(family.runtime) == CapabilityPredicate(:major, v"10.0")

        exact = parse_test_requirement(
            "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc==12.1")
        @test only(exact.runtime) == CapabilityPredicate(:exact, v"12.1")

        tcgen = parse_test_requirement(
            "# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc==10|cc==11")
        @test tcgen.requires === :toolkit
        @test tcgen.evidence === :mixed
        @test length(tcgen.runtime) == 2

        compile = parse_test_requirement(
            "# TEST_TARGET: requires=toolkit evidence=compile")
        @test isempty(compile.runtime)

        ptxas = parse_test_requirement(
            "# TEST_TARGET: requires=toolkit evidence=ptxas")
        @test isempty(ptxas.runtime)

        bad = [
            "# TEST_TARGET requires=gpu evidence=runtime runtime=cc>=8.9",
            "# TEST_TARGET: evidence=runtime runtime=cc>=8.9",
            "# TEST_TARGET: requires=gpu runtime=cc>=8.9",
            "# TEST_TARGET: requires=gpu evidence=runtime",
            "# TEST_TARGET: requires=gpu evidence=ptxas runtime=cc>=8.9",
            "# TEST_TARGET: requires=gpu evidence=compile runtime=cc>=8.9",
            "# TEST_TARGET: requires=toolkit evidence=mixed",
            "# TEST_TARGET: requires=toolkit evidence=ptxas runtime=cc==9.0",
            "# TEST_TARGET: requires=toolkit evidence=compile runtime=cc>=8.9",
            "# TEST_TARGET: requires=host evidence=host runtime=cc>=7.0",
            "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.9|cc>=8.9",
            "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.9|",
            "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc=8.9",
            "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc==10.0.1",
            "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc-major==10",
            "# TEST_TARGET: requires=gpu evidence=runtime target=sm_89",
            "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.9 surprise=yes",
            "# TEST_TARGET: requires=gpu requires=host evidence=runtime runtime=cc>=8.9",
        ]
        for line in bad
            @test_throws ArgumentError parse_test_requirement(line)
        end
    end

    @testset "minimum, exact-major, and exact-minor capability predicates" begin
        floor = parse_test_requirement(
            "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.9")
        @test !capability_matches(floor, v"8.8")
        @test capability_matches(floor, v"8.9")
        @test capability_matches(floor, v"9.0")
        @test capability_matches(floor, v"12.1")

        family = parse_test_requirement(
            "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc==10")
        @test !capability_matches(family, v"9.0")
        @test capability_matches(family, v"10.0")
        @test capability_matches(family, v"10.3")
        @test !capability_matches(family, v"11.0")
        @test !capability_matches(family, v"12.1")

        exact = parse_test_requirement(
            "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc==10.0")
        @test capability_matches(exact, v"10.0")
        @test !capability_matches(exact, v"10.3")
        @test !capability_matches(exact, v"12.1")

        tcgen = parse_test_requirement(
            "# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc==10|cc==11")
        @test capability_matches(tcgen, v"10.0")
        @test capability_matches(tcgen, v"10.3")
        @test capability_matches(tcgen, v"11.0")
        @test capability_matches(tcgen, v"11.7")
        @test !capability_matches(tcgen, v"12.1") # GB10 is not tcgen05.

        gb10 = parse_test_requirement(
            "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc==12.1")
        @test capability_matches(gb10, v"12.1")
        @test !capability_matches(gb10, v"12.0")

        floor10 = parse_test_requirement(
            "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=10")
        @test !capability_matches(floor10, v"9.0")
        @test capability_matches(floor10, v"10.0")
        @test capability_matches(floor10, v"10.3")
        @test capability_matches(floor10, v"12.1")
    end

    @testset "all GPU files have one parseable policy" begin
        gpu_dir = joinpath(@__DIR__, "..", "gpu")
        files = sort([joinpath(root, name)
                      for (root, _, names) in walkdir(gpu_dir)
                      for name in names if endswith(name, ".jl")])

        # Closed-world without a hand-maintained mirror: every discovered GPU
        # file must carry exactly one parseable policy. Mixed files must also
        # contain a runtime gate so their offline compile tier cannot fall
        # through into hardware execution on a host runner.
        @test !isempty(files)
        requirements = read_test_requirement.(files)
        @test length(requirements) == length(files)
        for (file, req) in zip(files, requirements)
            if req.requires === :toolkit && req.evidence === :mixed
                @test occursin("if test_runtime_supported(@__FILE__)",
                               read(file, String))
            end
        end

        mktemp() do path, io
            write(io, "# ordinary comment\n")
            close(io)
            @test_throws ArgumentError read_test_requirement(path)
        end
        mktemp() do path, io
            write(io, "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.0\n")
            write(io, "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.9\n")
            close(io)
            @test_throws ArgumentError read_test_requirement(path)
        end
        mktemp() do path, io
            write(io, "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.0\n")
            for _ in 1:24
                write(io, "# padding\n")
            end
            write(io, "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.9\n")
            close(io)
            @test_throws ArgumentError read_test_requirement(path)
        end
        mktemp() do path, io
            write(io, "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.0\n")
            for _ in 1:24
                write(io, "# padding\n")
            end
            write(io, "# TEST_TARGET: malformed-late-policy\n")
            close(io)
            @test_throws ArgumentError read_test_requirement(path)
        end
        mktemp() do path, io
            write(io, "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.0\n")
            write(io, "# TEST_TARGET banners are parsed by runtests.jl\n")
            close(io)
            @test read_test_requirement(path).requires === :gpu
        end
        mktemp() do path, io
            for _ in 1:20
                write(io, "# padding\n")
            end
            write(io, "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.0\n")
            close(io)
            @test_throws ArgumentError read_test_requirement(path)
        end
    end

    @testset "deterministic routing and manifest" begin
        host = parse_test_requirement(
            "# TEST_TARGET: requires=host evidence=host")
        compile = parse_test_requirement(
            "# TEST_TARGET: requires=toolkit evidence=compile")
        ptxas = parse_test_requirement(
            "# TEST_TARGET: requires=toolkit evidence=ptxas")
        hopper = parse_test_requirement(
            "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc==9.0")
        mixed = parse_test_requirement(
            "# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc==9.0")
        tcgen = parse_test_requirement(
            "# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc==10|cc==11")

        @test !requires_toolchain(host)
        @test !requires_gpu(host)
        @test requires_toolchain(compile)
        @test !requires_gpu(compile)
        @test requires_toolchain(ptxas)
        @test !requires_gpu(ptxas)
        @test requires_toolchain(mixed)
        @test requires_gpu(mixed)
        @test requires_toolchain(hopper)
        @test requires_gpu(hopper)

        ada = TestEnvironment(true, true, v"13.3.33",
                              true, true, v"8.9", "Ada")
        entries = [
            plan_entry("z/hopper", hopper, ada),
            plan_entry("a/host", host, ada),
            plan_entry("p/ptxas", ptxas, ada),
            plan_entry("m/mixed", mixed, ada),
        ]
        @test entries[1].action === :skip
        @test entries[2].action === :execute
        @test entries[3].action === :execute
        @test entries[4].action === :execute

        manifest = format_manifest(entries, ada)
        @test manifest == format_manifest(reverse(entries), ada)
        @test occursin("summary: execute=3 skip=1 total=4", manifest)
        @test occursin("offline-compiler=available version=13.3.33", manifest)
        @test occursin("device=\"Ada\" capability=8.9", manifest)
        @test occursin("z/hopper", manifest)
        @test !occursin("a/host", manifest)
        verbose_manifest = format_manifest(entries, ada; verbose = true)
        @test verbose_manifest ==
              format_manifest(reverse(entries), ada; verbose = true)
        @test findfirst("a/host", verbose_manifest) <
              findfirst("z/hopper", verbose_manifest)

        offline = TestEnvironment(true, true, v"13.3.33",
                                  true, false, v"0.0", "")
        @test plan_entry("host", host, offline).action === :execute
        @test plan_entry("compile", compile, offline).action === :execute
        @test plan_entry("ptxas", ptxas, offline).action === :execute
        @test plan_entry("mixed", mixed, offline).action === :execute
        @test plan_entry("gpu", hopper, offline).action === :skip
        @test occursin("GPU=unavailable",
                       format_manifest([plan_entry("ptxas", ptxas, offline)], offline))

        no_toolchain = TestEnvironment(true, false, nothing,
                                       true, false, v"0.0", "")
        @test plan_entry("host", host, no_toolchain).action === :execute
        @test plan_entry("ptxas", ptxas, no_toolchain).action === :skip
        @test plan_entry("mixed", mixed, no_toolchain).action === :skip
        @test plan_entry("gpu", hopper, no_toolchain).action === :skip

        host_only = TestEnvironment(false, false, nothing,
                                    false, false, v"0.0", "")
        host_only_entry = plan_entry("host/parser", host, host_only)
        @test host_only_entry.action === :execute
        host_manifest = format_manifest([host_only_entry], host_only)
        @test occursin("offline-compiler routing-check=skipped", host_manifest)
        @test occursin("GPU routing-check=skipped", host_manifest)
        @test plan_entry("gpu", hopper, host_only).action === :skip

        forced = plan_entry("hopper", hopper, ada; forced = true)
        @test forced.action === :execute

        forced_mixed = plan_entry("mixed", mixed, ada; forced = true)
        @test forced_mixed.action === :execute

        gb10 = TestEnvironment(true, true, v"13.3.33",
                               true, true, v"12.1", "GB10")
        tcgen_entry = plan_entry("gpu/blackwell/tcgen05", tcgen, gb10)
        @test tcgen_entry.action === :execute # ptxas still runs
        @test occursin("device=\"GB10\" capability=12.1",
                       format_manifest([tcgen_entry], gb10))
    end

    @testset "path tiers do not conflate ptxas with runtime" begin
        ptxas = requirement_for_test("ptxas/hopper", joinpath(@__DIR__, ".."))
        @test ptxas.requires === :toolkit
        @test ptxas.evidence === :ptxas
        @test isempty(ptxas.runtime)

        host = requirement_for_test("host/parser", joinpath(@__DIR__, ".."))
        @test host.requires === :host
        @test host.evidence === :host
    end
end
