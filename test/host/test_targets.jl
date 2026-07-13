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

        # Independent closed-world inventory: registration and oracle do not
        # derive from the same banners, so changing an exact predicate to a
        # floor (or moving a file between evidence tiers) is review-visible.
        expected_by_banner = Dict(
            "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=7.0" => Set([
                "extended_precision.jl", "kernel_abstractions.jl",
                "layer_norm.jl", "rms_norm.jl", "softmax.jl", "swiglu.jl",
                "transpiler_roundtrip.jl",
            ]),
            "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=7.5" =>
                Set(["nvvm.jl"]),
            "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.0" => Set([
                "ampere/conv2d_fprop.jl", "ampere/gemm.jl",
                "ampere/gemm_3xtf32.jl", "ampere/gemm_fp64.jl",
                "ampere/gemm_highperf.jl", "ampere/gemm_highperf_swizzled.jl",
                "ampere/gemm_minimal.jl", "ampere/gemm_pipelined.jl",
                "ampere/gemm_sparse.jl", "ampere/gemm_streamk.jl",
                "ampere/gemm_tf32.jl",
            ]),
            "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.9" => Set([
                "ada/gemm_fp8.jl", "corpus_launch.jl", "cvt_fp8.jl",
                "exec.jl", "mma_fp8.jl",
            ]),
            "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=9.0" =>
                Set(["hopper/tma_copy.jl", "mbarrier_roundtrip.jl",
                     "tensor_map.jl"]),
            "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=10.0" =>
                Set(["add_f32x2.jl", "cvt_subbyte_fp.jl"]),
            "# TEST_TARGET: requires=gpu evidence=runtime runtime=cc==12.1" =>
                Set(["sm121a_smoke.jl"]),
            "# TEST_TARGET: requires=toolkit evidence=compile" =>
                Set(["text.jl"]),
            "# TEST_TARGET: requires=toolkit evidence=ptxas" => Set([
                "corpus_compile.jl", "hopper/tma_epilogue.jl",
                "hopper/wgmma_sweep.jl",
            ]),
            "# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc==9.0" => Set([
                "hopper/cluster_arrive_mapa.jl", "hopper/flash_attention.jl",
                "hopper/gemm_activation_fusion.jl",
                "hopper/gemm_epilogue_swizzle.jl",
                "hopper/gemm_fp8_blockwise_scaling.jl",
                "hopper/gemm_fp8_warpspec.jl",
                "hopper/gemm_gather_scatter.jl",
                "hopper/gemm_highperf_hopper.jl",
                "hopper/gemm_mixed_dtype.jl",
                "hopper/gemm_pc_pipeline.jl",
                "hopper/gemm_pc_pipeline_cluster.jl",
                "hopper/gemm_pc_pipeline_split.jl",
                "hopper/gemm_pc_tma_store.jl", "hopper/gemm_permute.jl",
                "hopper/gemm_topk_softmax.jl", "hopper/gemm_warpgroup.jl",
                "hopper/gemm_weight_prefetch.jl", "hopper/gett.jl",
                "hopper/grouped_gemm.jl", "hopper/grouped_gemm_multik.jl",
                "hopper/ptr_array_batched_gemm.jl", "hopper/sparse_gemm.jl",
                "hopper/tma_multicast_cluster.jl",
                "hopper/tma_swizzle_probe.jl",
            ]),
            # PTX 9.3 sections 9.7.17.7, .8, .10, and .12 support the
            # alloc/ld/st/wait/f16-mma/commit forms used by these files in
            # both CC 10.x and CC 11.x families, but not CC 12.x.
            "# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc==10|cc==11" => Set([
                "blackwell/gemm_highperf_blackwell.jl",
                "blackwell/grouped_gemm.jl",
                "blackwell/tcgen05_accum_probe.jl",
                "blackwell/tcgen05_b128_probe.jl",
                "blackwell/tcgen05_mma_probe.jl",
                "blackwell/tcgen05_roundtrip.jl",
                "blackwell/tcgen05_smoke.jl",
            ]),
        )
        expected = Dict{String,String}()
        for (banner, names) in expected_by_banner, name in names
            @test !haskey(expected, name)
            expected[name] = banner
        end
        actual = Dict{String,String}()
        guard_counts = Dict{String,Int}()
        for file in files
            name = replace(relpath(file, gpu_dir), '\\' => '/')
            lines = readlines(file)
            banners = filter(line -> occursin(r"^\s*#\s*TEST_TARGET\b", line),
                             lines)
            actual[name] = only(banners)
            guard_counts[name] = count(==("if test_runtime_supported(@__FILE__)"),
                                       strip.(lines))
        end
        @test actual == expected
        @test length(files) == length(expected) == 65

        mixed_names = union(
            expected_by_banner["# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc==9.0"],
            expected_by_banner["# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc==10|cc==11"],
        )
        two_runtime_sections = Set([
            "blackwell/gemm_highperf_blackwell.jl",
            "hopper/gemm_mixed_dtype.jl",
        ])
        for (name, count) in guard_counts
            expected_count = name in mixed_names ?
                             (name in two_runtime_sections ? 2 : 1) : 0
            @test count == expected_count
        end

        requirements = read_test_requirement.(files)
        @test length(requirements) == length(files)
        @test count(req -> req.requires === :toolkit &&
                           req.evidence === :mixed, requirements) == 31
        @test count(req -> req.requires === :toolkit &&
                           req.evidence === :ptxas, requirements) == 3
        @test count(req -> req.requires === :toolkit &&
                           req.evidence === :compile, requirements) == 1
        @test count(req -> req.requires === :gpu, requirements) == 30
        @test all(file -> !occursin("REQUIRES CC", read(file, String)), files)
        @test all(file -> !occursin(r"\bDEV_CAP\b", read(file, String)), files)
        @test all(file -> !occursin(r"TEST_TARGET.*\bsm_", read(file, String)), files)

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
            write(io, "# TEST_TARGET malformed-late-policy\n")
            close(io)
            @test_throws ArgumentError read_test_requirement(path)
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
        @test occursin("live GPU is not required", entries[3].reason)
        @test occursin("runtime skipped", entries[4].reason)

        manifest = format_manifest(entries, ada)
        @test manifest == format_manifest(reverse(entries), ada)
        @test occursin("summary: execute=3 skip=1 total=4", manifest)
        @test occursin("offline-compiler=available version=13.3.33", manifest)
        @test occursin("device=\"Ada\" capability=8.9", manifest)
        @test !occursin("active_target", manifest)
        @test !occursin("sm_89a", manifest)
        @test findfirst("a/host", manifest) < findfirst("z/hopper", manifest)

        offline = TestEnvironment(true, true, v"13.3.33",
                                  true, false, v"0.0", "")
        @test plan_entry("host", host, offline).action === :execute
        @test plan_entry("compile", compile, offline).action === :execute
        @test plan_entry("ptxas", ptxas, offline).action === :execute
        @test plan_entry("mixed", mixed, offline).action === :execute
        @test occursin("functional GPU unavailable",
                       plan_entry("mixed", mixed, offline).reason)
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
        @test occursin("offline compiler routing check skipped",
                       plan_entry("gpu", hopper, host_only).reason)

        forced = plan_entry("hopper", hopper, ada; forced = true)
        @test forced.action === :execute
        @test occursin("gate bypassed", forced.reason)

        forced_mixed = plan_entry("mixed", mixed, ada; forced = true)
        @test forced_mixed.action === :execute
        @test occursin("offline cross-target ptxas compile", forced_mixed.reason)
        @test occursin("runtime remains skipped", forced_mixed.reason)
        @test !occursin("gate bypassed", forced_mixed.reason)

        gb10 = TestEnvironment(true, true, v"13.3.33",
                               true, true, v"12.1", "GB10")
        tcgen_entry = plan_entry("gpu/blackwell/tcgen05", tcgen, gb10)
        @test tcgen_entry.action === :execute # ptxas still runs
        @test occursin("runtime skipped", tcgen_entry.reason)
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
