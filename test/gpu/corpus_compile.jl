using PTX: IR
using PTX.Parser: parse as parse_ptx
using PTX.IR: format

# ptxas-acceptance of real-world external PTX — AND of the PTX our
# parser/formatter re-emits from it. ptxas is the ground truth for "valid
# PTX", far stricter than the parser's syntactic check:
#
#   1. the source is real, valid PTX (sanity);
#   2. parser+formatter preserve enough that ptxas STILL accepts the
#      structurally re-emitted form (raw_source stripped).
#
# Pure parse + round-trip *fidelity* is already covered hardware-free by
# test/host/corpus.jl (lossless / fixed-point / size-delta tiers). This
# file's sole added value is the ptxas leg, so it stays scoped to that.
#
# Design (and why it changed):
#  * Uses the CUDACore-managed ptxas (`CUDA_Compiler.ptxas()`) invoked
#    STANDALONE (PTX → cubin, --compile-only, no device load) — not the
#    driver JIT. Fully hardware-independent: Ada, the GB10 dev box, a
#    B200 and a B300 all produce identical results.
#  * The old path (`CUDACore.CuModule`) JIT-compiled for the *live
#    device*, which forced a device-CC gate (`_target_runnable`,
#    arch-exact for `sm_NNa`). That silently made corpus coverage depend
#    on which GPU ran the suite: only a CC-exactly-10.0 box ever
#    exercised the sm_100a corpus, so the LLVM-fixture `.version 8.5` /
#    `.target sm_100a` defect was invisible until a B200 first ran it.
#    The whole CC gate is gone.
#  * `.version` is stamped to the managed assembler's max PTX ISA —
#    mirroring `CUDACore.rewrite_ptx_header` ("the ISA we want ptxas to
#    validate against"). LLVM-derived fixtures carry loose `.version`
#    directives (e.g. 8.5 with sm_100a+tcgen05, whose floor is 8.6) that
#    ptxas correctly rejects; but the corpus exists to exercise our
#    parser/formatter, not to assert LLVM's filecheck headers are
#    ptxas-legal. Stamping to the assembler's own ISA makes "are the
#    *instructions* valid for this target" a well-posed, single-rule
#    question (no per-target floor table). Safe: ptxas is
#    backward-compatible and the stamped ISA == the binary's own ceiling.
#  * Two deterministic, hardware-independent skip reasons for the ptxas
#    leg (these files are still covered by host/corpus.jl round-trip):
#      - `_is_malformed_llvm_fixture`: LLVM FileCheck snippets that aren't
#        complete modules (undefined `_param_N` / `func_retval0` /
#        32-bit `[%rN]` address ABI) — structurally un-assemblable.
#      - declared `.target` the managed ptxas dropped (CUDA 13 removed
#        sm_50 / sm_70): the binary cannot target it; not a corpus defect.

const _CORPUS_EXT_DIR = joinpath(@__DIR__, "..", "corpus", "external")
const _PTXAS = CUDACore.CUDA_Compiler.ptxas()
# Max PTX ISA the managed assembler accepts (CUDA 13.2 → 9.2). Tracks the
# shipped toolkit automatically; never below any feature floor, never
# above the binary's ceiling.
const _ISA = maximum(CUDACore.cuda_ptx_support(CUDACore.compiler_version()))

# Many LLVM-testsuite-derived corpus files are filecheck fixtures that
# generate PTX from IR without emitting a full prologue. They share one of
# three malformed patterns, all of which ptxas correctly rejects:
#
#   (a) `.visible .entry NAME()` with zero params but body references
#       `[NAME_param_N]` — undefined param symbols.
#   (b) `.visible .entry NAME()` with no return slot but body stores to
#       `func_retval0` — undefined retval symbol.
#   (c) Body uses 32-bit regs `[%rN]` as memory addresses; ptxas treats
#       this as 32-bit ABI which is rejected on sm_90+. Real PTX uses
#       `%rdN` (i64) for addresses there.
#
# Also catches the clusterlaunchcontrol__* family which adds undeclared
# 128-bit `%clc_handle` regs on top of (a). Easier to filter structurally
# than to maintain a per-file blacklist.
function _is_malformed_llvm_fixture(path::AbstractString)
    src = read(path, String)
    m = match(r"\.visible\s+\.entry\s+([A-Za-z_0-9]+)\(\)", src)
    m === nothing && return false
    entry_name = m.captures[1]
    # (a) `_param_N` body references against an empty `()` entry.
    occursin(Regex("\\b" * entry_name * "_param_\\d"), src) && return true
    # (b) `func_retval0` body references against an empty `()` entry.
    occursin("func_retval0", src) && return true
    # (c) 32-bit reg as memory address — `[%r<digits>]` or `[%r<digits>+...]`,
    # but NOT `[%rd...]` (the 64-bit form). ptxas rejects on sm_90+.
    occursin(r"\[%r\d", src) && return true
    return false
end

# Stamp `.version` to the managed assembler's ISA; leave `.target` alone.
_stamp_version(src::AbstractString) =
    replace(src, r"^(\.version[ \t]+)[0-9.]+"m =>
            SubstitutionString("\\g<1>$(_ISA.major).$(_ISA.minor)"))

# Run the managed ptxas standalone: PTX → cubin, no device. Returns
# (ok::Bool, stderr::String). `arch` = the file's declared `sm_*`.
function _ptxas_accepts(src::AbstractString, arch::AbstractString)
    ptx = tempname() * ".ptx"
    cub = tempname() * ".cubin"
    write(ptx, _stamp_version(src))
    errbuf = IOBuffer()
    proc = run(pipeline(ignorestatus(
                   `$_PTXAS --compile-only --gpu-name $arch --output-file $cub $ptx`);
               stdout = devnull, stderr = errbuf))
    rm(ptx; force = true)
    rm(cub; force = true)
    (proc.exitcode == 0, String(take!(errbuf)))
end

# `(path, declared_arch)` for every non-malformed external `.ptx`. No
# device, no CC gate — selection is a static property of the corpus.
function _ptxas_corpus_files()
    files = Tuple{String,String}[]
    for (root, _, names) in walkdir(_CORPUS_EXT_DIR), name in names
        endswith(name, ".ptx") || continue
        path = joinpath(root, name)
        _is_malformed_llvm_fixture(path) && continue
        target_line = nothing
        for line in eachline(path)
            startswith(line, ".target") && (target_line = line; break)
        end
        target_line === nothing && continue
        m = match(r"^\.target\s+(sm_\w+)", target_line)
        m === nothing && continue
        push!(files, (path, String(m.captures[1])))
    end
    sort!(files)
end

const _PTXAS_CORPUS = _ptxas_corpus_files()

@testset "ptxas accepts source + structural re-emit ($(basename(path)))" for (path, arch) in _PTXAS_CORPUS
    src = read(path, String)
    ok, err = _ptxas_accepts(src, arch)
    if !ok && occursin("not defined for option 'gpu-name'", err)
        # Managed ptxas (CUDA $(CUDACore.compiler_version())) dropped this
        # arch (CUDA 13 removed sm_50 / sm_70). Deterministic & identical
        # on every machine; round-trip still covered by host/corpus.jl.
        @test_skip _ptxas_accepts(src, arch)[1]
    else
        @test ok || (println(err); false)
        reformatted = format(IR.unraw(parse_ptx(src)))
        ok2, err2 = _ptxas_accepts(reformatted, arch)
        @test ok2 || (println(err2); false)
    end
end
