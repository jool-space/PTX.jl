# TEST_TARGET: requires=toolkit evidence=ptxas
using PTX: IR
using PTX.Parser: parse as parse_ptx
using PTX.IR: format

# ptxas external-corpus shard 4 of 4 — ptxas-acceptance of this shard's slice
# of the real-world corpus, for both the source and its structural re-emit.
# The machinery and design rationale (malformed-fixture filter, version
# stamping, arch retargeting, corpus selection, size-balanced slicing) live
# in test/setup.jl (ptxas external-corpus acceptance support); host/corpus.jl
# pins that the four slices partition the selected corpus exactly.

@testset "ptxas accepts source + structural re-emit ($(basename(path)))" for (path, arch) in ptxas_corpus_shard(4)
    # Preserve provenance fixtures on disk. The host header tier asserts the
    # exact rejection of LLVM's PTX 8.5/sm_100a originals; this ptxas/body tier
    # uses the mechanically derived minimum-version repair before both source
    # assembly and structural parse/re-emission.
    src = _external_parser_source(read(path, String))
    ok, err = _ptxas_accepts(src, arch)
    @test ok || (println(err); false)
    reformatted = format(IR.unraw(parse_ptx(src)))
    ok2, err2 = _ptxas_accepts(reformatted, arch)
    @test ok2 || (println(err2); false)
end
