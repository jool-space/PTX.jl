# REQUIRES CC 8.9
using PTX: IR
using PTX.Parser: parse as parse_ptx
using PTX.IR: format

# End-to-end launch + correctness on selected corpus PTX files. Loads the
# source AND the structurally re-emitted form, runs both, and compares to
# a CPU reference. Anything our formatter quietly broke would show up as a
# numerical mismatch (or a launch failure) — the strongest end-to-end test
# for the parser/formatter contract.
#
# Subset: ptoxide_add_simple (3-arg pointwise add, 1 block) and
# ptoxide_add (4-arg add with bounds-check, multi-block). Both target sm_89
# (the lowest-target add-style corpus we currently ship), hence the banner.

const _CORPUS_EXT_DIR = joinpath(@__DIR__, "..", "corpus", "external")

@testset "ptoxide_add_simple: launch + verify (source + re-emit)" begin
    n = 256
    a_h = rand(Float32, n)
    b_h = rand(Float32, n)
    src         = read(joinpath(_CORPUS_EXT_DIR, "ptoxide_add_simple.ptx"), String)
    reformatted = format(IR.unraw(parse_ptx(src)))
    expected = a_h .+ b_h
    for ptx_text in (src, reformatted)
        a_d   = CuArray(a_h)
        b_d   = CuArray(b_h)
        out_d = CUDACore.zeros(Float32, n)
        mod = CUDACore.CuModule(ptx_text)
        fn  = CUDACore.CuFunction(mod, "_Z10add_simplePfS_S_")
        CUDACore.cudacall(fn,
                      Tuple{CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}},
                      a_d, b_d, out_d; threads = n)
        CUDACore.synchronize()
        @test Array(out_d) ≈ expected
    end
end

@testset "ptoxide_add: launch + verify (source + re-emit)" begin
    n = 1024
    a_h = rand(Float32, n)
    b_h = rand(Float32, n)
    src         = read(joinpath(_CORPUS_EXT_DIR, "ptoxide_add.ptx"), String)
    reformatted = format(IR.unraw(parse_ptx(src)))
    expected = a_h .+ b_h
    for ptx_text in (src, reformatted)
        a_d   = CuArray(a_h)
        b_d   = CuArray(b_h)
        out_d = CUDACore.zeros(Float32, n)
        mod = CUDACore.CuModule(ptx_text)
        fn  = CUDACore.CuFunction(mod, "_Z3addPfS_S_m")
        threads = 256
        blocks  = cld(n, threads)
        CUDACore.cudacall(fn,
                      Tuple{CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, Csize_t},
                      a_d, b_d, out_d, Csize_t(n);
                      threads = threads, blocks = blocks)
        CUDACore.synchronize()
        @test Array(out_d) ≈ expected
    end
end
