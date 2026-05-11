using PTX: IR
using PTX.Parser: parse as parse_ptx
using PTX.IR: format

# Load every corpus file runnable on the active device via CUDACore.CuModule.
# ptxas (the PTX assembler) is the ground truth for "valid PTX" — far
# stricter than the parser's syntactic check. Running the source AND the
# structurally re-emitted form proves:
#
#   1. the source is real, valid PTX (sanity);
#   2. our parser+formatter preserves enough semantics that ptxas STILL
#      accepts the result after a round-trip, even with raw_source stripped.
#
# Subset rule per file's `.target sm_<X><Y>[a]?`:
#   - bare `sm_<X><Y>`: runnable iff dev_cap >= X.Y
#   - `sm_<X><Y>a`:     runnable iff dev_cap == X.Y  (arch-specific, no
#                       forward compat across major arches)
# So on Ada (CC 8.9) we run sm_50..sm_89; on Hopper (CC 9.0) sm_90a is
# included; on Blackwell sm_90a drops out and sm_100a appears.

const _CORPUS_EXT_DIR = joinpath(@__DIR__, "..", "corpus", "external")
const _DEV_CAP = CUDACore.capability(CUDACore.device())

# Whether a `.target sm_<digits>[a|f]?` token is runnable on dev_cap.
function _target_runnable(target::AbstractString, dev_cap::VersionNumber)
    m = match(r"^sm_(\d+)([af])?$", target)
    m === nothing && return false
    digits, suffix = m.captures
    target_cc = VersionNumber(parse(Int, digits[1:end-1]),
                              parse(Int, digits[end:end]))
    suffix == "a" ? dev_cap == target_cc : dev_cap >= target_cc
end

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

function _runnable_corpus_files()
    files = String[]
    for (root, _, names) in walkdir(_CORPUS_EXT_DIR)
        for name in names
            endswith(name, ".ptx") || continue
            path = joinpath(root, name)
            _is_malformed_llvm_fixture(path) && continue
            target_line = nothing
            for line in eachline(path)
                if startswith(line, ".target")
                    target_line = line
                    break
                end
            end
            target_line === nothing && continue
            m = match(r"^\.target\s+([A-Za-z0-9_]+)", target_line)
            m === nothing && continue
            _target_runnable(String(m.captures[1]), _DEV_CAP) || continue
            push!(files, path)
        end
    end
    sort(files)
end

const _RUNNABLE_FILES = _runnable_corpus_files()

@testset "ptxas accepts source + structural re-emit ($(basename(path)))" for path in _RUNNABLE_FILES
    src = read(path, String)
    @test (CUDACore.CuModule(src); true)
    reformatted = format(IR.unraw(parse_ptx(src)))
    @test (CUDACore.CuModule(reformatted); true)
end
