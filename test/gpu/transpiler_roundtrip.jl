using CUDATools

# Roundtrip test: capture PTX from each example kernel via CUDATools.code_ptx,
# run it through `ptx_to_julia`, verify the output `Meta.parse`s cleanly.
# This exercises the transpiler on real-world compiler-emitted PTX —
# wider operand patterns, mangled names, comments, inline-asm markers,
# long basic blocks — than the small hand-written corpus.
#
# Primarily a syntactic-validity sweep (Meta.parse with no `:error` Exprs).
# Per-kernel testsets add surface-level structural checks (e.g. shared-
# memory decls came through as `CuStaticSharedArray`).

# Kernel definitions live alongside their own testsets — pull them in so
# this file is self-contained under ParallelTestRunner's per-worker model.
include("swiglu.jl")
include("rms_norm.jl")
include("layer_norm.jl")

function _roundtrip_emit_and_transpile(kernel, types)
    io = IOBuffer()
    CUDATools.code_ptx(io, kernel, types)
    ptx = String(take!(io))
    julia = PTX.ptx_to_julia(ptx)
    (ptx, julia)
end

function _parses_cleanly(julia::AbstractString)
    expr = Meta.parseall(julia)
    expr isa Expr || return false
    expr.head == :toplevel || return false
    !any(a -> a isa Expr && a.head == :error, expr.args)
end

@testset "transpiler roundtrip: emitted-PTX → parseable Julia" begin
    @testset "swiglu" begin
        types = Tuple{CuDeviceVector{Float32, 1}, CuDeviceVector{Float32, 1},
                      CuDeviceVector{Float32, 1}, Val{512}, Val{128}}
        ptx, julia = _roundtrip_emit_and_transpile(_swiglu_v4_kernel!, types)
        @test occursin("ld.global.v4.f32", ptx)
        @test occursin("st.global.v4.f32", ptx)
        @test occursin("ex2.approx.f32", ptx)
        @test occursin("rcp.approx.f32", ptx)
        @test _parses_cleanly(julia)
        # Vector ld/st surface as tuple-destructure / tuple-construct.
        @test occursin(r"=\s*ptx\"ld\.global\.v4\.f32\"\(", julia)
        @test occursin(r"ptx\"st\.global\.v4\.f32\"\([^,]+,\s*\(", julia)
    end

    @testset "rms_norm" begin
        types = Tuple{CuDeviceVector{Float32, 1}, CuDeviceVector{Float32, 1},
                      CuDeviceVector{Float32, 1},
                      Val{1024}, Val{256}, Val{Float32(1f-6)}}
        ptx, julia = _roundtrip_emit_and_transpile(_rms_norm_v4_kernel!, types)
        @test occursin("shfl.sync.bfly.b32", ptx)
        @test occursin("bar.sync 0", ptx)
        @test occursin("rsqrt.approx.f32", ptx)
        @test _parses_cleanly(julia)
        # bar.sync should NOT come through as `0 = ptx"bar.sync"()` — that
        # was the no-dest-opcode bug fixed in v0.3-D.
        @test !occursin("0 = ptx\"bar.sync\"", julia)
        @test occursin("ptx\"bar.sync\"(0)", julia)
        # No `$` should leak into Julia identifiers (they're parse-error
        # outside string interpolation). Comments are fine.
        @test !any(eachsplit(julia, '\n')) do line
            !startswith(strip(line), "#") && occursin('$', line)
        end
    end

    @testset "layer_norm" begin
        types = Tuple{CuDeviceVector{Float32, 1}, CuDeviceVector{Float32, 1},
                      CuDeviceVector{Float32, 1}, CuDeviceVector{Float32, 1},
                      Val{1024}, Val{256}, Val{Float32(1f-5)}}
        ptx, julia = _roundtrip_emit_and_transpile(_layer_norm_v4_kernel!, types)
        @test occursin("shfl.sync.bfly.b32", ptx)
        @test count("bar.sync 0", ptx) == 2     # one per pass-1/2 boundary
        @test _parses_cleanly(julia)
        @test count("ptx\"bar.sync\"(0)", julia) == 2
    end
end
