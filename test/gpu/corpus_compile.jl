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
#  * `_is_malformed_llvm_fixture` excludes LLVM FileCheck snippets that
#    aren't complete modules (undefined `_param_N` / `func_retval0` /
#    32-bit `[%rN]` address ABI) — structurally un-assemblable. Their
#    parse/round-trip is still covered by host/corpus.jl.
#  * The point of this file is PTX-*ISA* coverage, not which legacy GPU
#    still builds on a modern toolkit. ptxas's instruction-superset
#    (the PTX *language*) is distinct from its per-arch SASS backends
#    (which NVIDIA prunes — CUDA 13 dropped sm_50/sm_70). So if a file's
#    declared `.target` arch was pruned, we retarget UP to the lowest
#    arch the managed ptxas still backs (PTX is forward-portable) and
#    validate the instructions there — symmetric with stamping
#    `.version` up to the assembler's ISA. Verbatim-`.target` fidelity
#    stays covered by host/corpus.jl's round-trip tiers. Net: every
#    selected file is asserted; nothing is skipped.

const _CORPUS_EXT_DIR = joinpath(@__DIR__, "..", "corpus", "external")
const _PTXAS = CUDACore.CUDA_Compiler.ptxas()
# Max PTX ISA the managed assembler accepts (CUDA 13.2 → 9.2). Tracks the
# shipped toolkit automatically; never below any feature floor, never
# above the binary's ceiling.
const _ISA = maximum(CUDACore.ptxas_ptx_support(CUDACore.compiler_version()))

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

# Stamp `.version` to the managed assembler's ISA.
_stamp_version(src::AbstractString) =
    replace(src, r"^(\.version[ \t]+)[0-9.]+"m =>
            SubstitutionString("\\g<1>$(_ISA.major).$(_ISA.minor)"))

# Can the managed ptxas emit SASS for `arch`? A property of the shipped
# binary (its per-arch backends), not of any device — memoized.
const _ARCH_OK = Dict{String,Bool}()
function _arch_supported(arch::AbstractString)
    get!(_ARCH_OK, arch) do
        stub = ".version $(_ISA.major).$(_ISA.minor)\n.target $arch\n" *
               ".address_size 64\n.visible .entry _probe(){ ret; }\n"
        ptx = tempname() * ".ptx"; write(ptx, stub)
        cub = tempname() * ".cubin"
        proc = run(pipeline(ignorestatus(
                       `$_PTXAS --compile-only --gpu-name $arch --output-file $cub $ptx`);
                   stdout = devnull, stderr = devnull))
        rm(ptx; force = true); rm(cub; force = true)
        proc.exitcode == 0
    end
end

# Lowest real arch the managed ptxas still backs. Derived (not hardcoded)
# by scanning a fixed ascending ladder — self-adjusts when CUDACore bumps
# the toolkit (CUDA 13 → sm_75; a future CUDA pruning more just moves it).
const _ARCH_LADDER = ["sm_50", "sm_52", "sm_53", "sm_60", "sm_61", "sm_62",
                      "sm_70", "sm_72", "sm_75", "sm_80", "sm_86", "sm_87",
                      "sm_89", "sm_90", "sm_90a", "sm_100a", "sm_103a",
                      "sm_120a"]
const _MIN_ARCH = _ARCH_LADDER[findfirst(_arch_supported, _ARCH_LADDER)]

# Arch to actually assemble at: the file's declared one if the managed
# ptxas still backs it, else retarget UP to `_MIN_ARCH` (PTX is
# forward-portable; we want ISA coverage, not legacy-GPU coverage).
_target_arch(declared::AbstractString) =
    _arch_supported(declared) ? declared : _MIN_ARCH

# Run the managed ptxas standalone: PTX → cubin, no device. `.version`
# stamped to the managed ISA; `.target` retargeted up iff its declared
# arch was pruned from the binary. Returns (ok::Bool, stderr::String).
function _ptxas_accepts(src::AbstractString, declared::AbstractString)
    arch = _target_arch(declared)
    text = _stamp_version(src)
    arch == declared ||
        (text = replace(text, r"^\.target[ \t]+sm_\w+"m => ".target $arch"))
    ptx = tempname() * ".ptx"; write(ptx, text)
    cub = tempname() * ".cubin"
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
    @test ok || (println(err); false)
    reformatted = format(IR.unraw(parse_ptx(src)))
    ok2, err2 = _ptxas_accepts(reformatted, arch)
    @test ok2 || (println(err2); false)
end
