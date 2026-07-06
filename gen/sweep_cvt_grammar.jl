# Grammar sweep for the `cvt` family — ptxas as the grammar oracle.
#
# The PTX ISA manual documents cvt's grammar as prose + tables with no
# machine-readable source, and hand transcription rots silently as the ISA
# moves. This script derives the grammar empirically instead: enumerate a
# loose superset of candidate modifier chains, synthesize a minimal .ptx
# kernel per candidate, and trial-compile through the artifact ptxas across
# an arch ladder. What ptxas accepts IS the grammar — the same authority
# that would reject the form at kernel-compile time. The accepted set (with
# per-form arch floors) is committed as generated source
# (src/grammar_cvt.jl), the same pattern as the intrinsic registry: a
# regeneration is a reviewable diff.
#
# Soundness notes:
#   - False negatives (valid chain, wrong synthesized operands) are the
#     real hazard. Two defenses: every candidate is tried with two register
#     mappings (typed float regs and raw bit regs), and a known-good anchor
#     list must be 100% accepted or the sweep ABORTS rather than commit a
#     hole. Operand arity: packed-destination-from-f32 forms take two
#     sources; both arities are tried where ambiguous.
#   - Acceptance is swept per arch; floor = first accepting rung. If a form
#     is accepted at some rung but rejected at a later one (arch-specific
#     surface), that is printed loudly and the form keeps its floor — the
#     later-arch rejection is ptxas's to enforce at compile time.
#
# Usage:  julia --project=test gen/sweep_cvt_grammar.jl [--jobs=N]
# Writes: src/grammar_cvt.jl
#
# Oracle provenance is stamped into the output header. Re-run on a ptxas
# bump; the diff shows exactly what the new ISA added.

using Base.Threads

const PTXAS = let
    cands = filter(isfile, [joinpath(a, "bin", "ptxas")
                            for a in readdir(joinpath(homedir(), ".julia", "artifacts");
                                             join = true) if isdir(a)])
    isempty(cands) && error("no artifact ptxas found")
    first(cands)
end
const PTXAS_VERSION = let
    v = read(`$PTXAS --version`, String)
    strip(split(v, '\n')[end-1])
end

const LADDER = ["sm_70", "sm_75", "sm_80", "sm_89", "sm_90", "sm_100a", "sm_120a"]
const PTX_VERSION = "8.8"

# ---------------------------------------------------------------------------
# Candidate space

const SCALAR_INT   = [:u8, :u16, :u32, :u64, :s8, :s16, :s32, :s64]
const SCALAR_FLOAT = [:f16, :bf16, :f32, :f64]
const EXOTIC       = [:tf32, :f16x2, :bf16x2,
                      :e4m3x2, :e5m2x2, :e2m3x2, :e3m2x2, :e2m1x2, :ue8m0x2]
const TYPES = vcat(SCALAR_INT, SCALAR_FLOAT, EXOTIC)

const ROUNDINGS = [Symbol[], [:rn], [:rz], [:rm], [:rp],
                   [:rni], [:rzi], [:rmi], [:rpi], [:rna]]

# Flag strings: empty, each single flag, and every ordered pair — modifier
# order differs between cvt sub-grammars (`{.ftz}{.sat}` vs
# `{.relu}{.satfinite}` vs `.satfinite{.relu}`), so both orders are
# candidates and ptxas keeps the legal spellings.
const FLAGS = [:ftz, :sat, :relu, :satfinite]
const FLAGSETS = let
    out = Vector{Vector{Symbol}}([Symbol[]])
    for f in FLAGS
        push!(out, [f])
    end
    for a in FLAGS, b in FLAGS
        a === b || push!(out, [a, b])
    end
    out
end

const PACKED_DST = Set([:f16x2, :bf16x2, :e4m3x2, :e5m2x2,
                        :e2m3x2, :e3m2x2, :e2m1x2, :ue8m0x2])

# dtype/atype → (register declaration type, operand-synth) — variant A uses
# typed float regs (what NVCC emits), variant B raw bit regs.
regtype(t::Symbol, variant::Symbol) = begin
    t in (:u8, :s8, :u16, :s16)                       ? ".b16" :
    t in (:f16, :bf16)                                ? ".b16" :
    t in (:e4m3x2, :e5m2x2, :e2m3x2, :e3m2x2, :ue8m0x2) ? ".b16" :
    t === :e2m1x2                                     ? ".b8"  :
    t in (:u32, :s32, :tf32, :f16x2, :bf16x2)         ? ".b32" :
    t in (:u64, :s64)                                 ? ".b64" :
    t === :f32 ? (variant === :A ? ".f32" : ".b32")   :
    t === :f64 ? (variant === :A ? ".f64" : ".b64")   :
    error("unmapped type $t")
end

mods_str(mods) = join(string.(mods), ".")

# One candidate = mods tuple + operand arity. Returns the .ptx text.
function kernel_ptx(arch::String, mods, two_src::Bool, variant::Symbol)
    dt, at = mods[end-1], mods[end]
    rd, ra = regtype(dt, variant), regtype(at, variant)
    srcs = two_src ? "a, b" : "a"
    decl_b = two_src ? "\n    .reg $ra b;" : ""
    """
    .version $PTX_VERSION
    .target $arch
    .address_size 64
    .visible .entry k()
    {
        .reg $rd d;
        .reg $ra a;$decl_b
        cvt.$(mods_str(mods)) d, $srcs;
        ret;
    }
    """
end

# ---------------------------------------------------------------------------
# Oracle

const TMP = mktempdir()

function accepts(arch::String, mods, two_src::Bool, variant::Symbol)
    path = joinpath(TMP, "c$(threadid())_$(hash((arch, mods, two_src, variant))).ptx")
    write(path, kernel_ptx(arch, mods, two_src, variant))
    ok = success(pipeline(`$PTXAS --gpu-name $arch -o /dev/null $path`;
                          stderr = devnull, stdout = devnull))
    rm(path; force = true)
    ok
end

# A form is accepted if ANY operand synthesis compiles (arity × reg variant).
# Variant B differs from A only for f32/f64 operands — skip it otherwise.
function form_accepts(arch::String, mods)
    dt, at = mods[end-1], mods[end]
    arities = (dt in PACKED_DST && at === :f32) ? (true, false) : (false,)
    variants = (dt in (:f32, :f64) || at in (:f32, :f64)) ? (:A, :B) : (:A,)
    for two_src in arities, variant in variants
        accepts(arch, mods, two_src, variant) && return true
    end
    false
end

# ---------------------------------------------------------------------------
# Sweep

function sweep()
    # Stage 1: rounding × dtype × atype, no flags, at the two frontier archs
    # (union covers datacenter- and consumer-specific surfaces) plus sm_70
    # (so pre-Ampere classic forms are live even if a frontier arch ever
    # dropped one).
    stage1 = Tuple{Vararg{Symbol}}[]
    for rnd in ROUNDINGS, dt in TYPES, at in TYPES
        push!(stage1, (rnd..., dt, at))
    end
    println("stage 1: $(length(stage1)) rounding×type candidates")
    live = Vector{Tuple{Vararg{Symbol}}}()
    lck = ReentrantLock()
    @threads for mods in stage1
        if form_accepts("sm_120a", mods) || form_accepts("sm_100a", mods) ||
           form_accepts("sm_70", mods)
            lock(lck) do; push!(live, mods); end
        end
    end
    println("stage 1 live: $(length(live))")

    # Stage 2: expand flags around each live rounding skeleton.
    stage2 = Tuple{Vararg{Symbol}}[]
    for mods in live, flags in FLAGSETS
        isempty(flags) && continue
        rnd = mods[1:end-2]; dt = mods[end-1]; at = mods[end]
        push!(stage2, (rnd..., flags..., dt, at))
    end
    println("stage 2: $(length(stage2)) flagged candidates")
    accepted = copy(live)
    @threads for mods in stage2
        if form_accepts("sm_120a", mods) || form_accepts("sm_100a", mods) ||
           form_accepts("sm_70", mods)
            lock(lck) do; push!(accepted, mods); end
        end
    end
    println("total accepted: $(length(accepted))")

    # Floors: first accepting rung of the ladder; report non-monotone
    # acceptance loudly (arch-specific surface).
    floors = Dict{Tuple{Vararg{Symbol}}, String}()
    nonmono = Tuple{Tuple{Vararg{Symbol}}, String}[]
    @threads for mods in accepted
        floor = nothing
        seen = Bool[]
        for arch in LADDER
            ok = form_accepts(arch, mods)
            push!(seen, ok)
            ok && floor === nothing && (floor = arch)
        end
        # non-monotone: accepted somewhere, rejected later
        first_ok = findfirst(seen)
        if first_ok !== nothing && !all(seen[first_ok:end])
            rej = LADDER[first_ok - 1 .+ findall(!, seen[first_ok:end])]
            lock(lck) do; push!(nonmono, (mods, join(rej, ","))); end
        end
        lock(lck) do; floors[mods] = something(floor, "REJECTED-EVERYWHERE"); end
    end
    for (mods, rej) in sort(nonmono)
        println("  NON-MONOTONE: cvt.$(mods_str(mods)) floor=$(floors[mods]) rejected at: $rej")
    end
    floors
end

# ---------------------------------------------------------------------------
# Anchors — known-good forms (from wrappers, goldens, and gpu kernels in the
# suite). If ANY fails the sweep harness itself is broken (wrong operand
# synthesis): abort, never commit a grammar with holes.

const ANCHORS = [
    (:f32, :f16), (:rn, :f32, :f16), (:rn, :f16, :f32), (:rzi, :u32, :f32),
    (:rni, :s32, :f32), (:sat, :u8, :s32), (:rn, :f64, :f32), (:rn, :f32, :f64),
    (:rn, :bf16, :f32), (:rn, :bf16x2, :f32), (:rn, :f16x2, :f32),
    (:rna, :tf32, :f32), (:rn, :satfinite, :e4m3x2, :f32),
    (:rn, :satfinite, :e5m2x2, :f16x2), (:rn, :f16x2, :e4m3x2),
    (:rn, :satfinite, :e2m1x2, :f32), (:rn, :f16x2, :e2m1x2),
    (:rz, :satfinite, :ue8m0x2, :bf16x2), (:rn, :bf16x2, :ue8m0x2),
]

function check_anchors(floors)
    bad = [a for a in ANCHORS if !haskey(floors, a) ||
                                 floors[a] == "REJECTED-EVERYWHERE"]
    if !isempty(bad)
        for a in bad
            println("ANCHOR MISSING: cvt.$(mods_str(a))")
        end
        error("$(length(bad)) known-good anchors missing — operand synthesis " *
              "is wrong for their type class; fix the sweep, do not commit.")
    end
end

# ---------------------------------------------------------------------------
# Emit

function emit(floors)
    forms = sort!([m for (m, f) in floors if f != "REJECTED-EVERYWHERE"])
    byfloor = Dict{String, Vector{Tuple{Vararg{Symbol}}}}()
    for m in forms
        push!(get!(Vector{Tuple{Vararg{Symbol}}}, byfloor, floors[m]), m)
    end
    out = joinpath(@__DIR__, "..", "src", "grammar_cvt.jl")
    open(out, "w") do io
        println(io, "# GENERATED by gen/sweep_cvt_grammar.jl — DO NOT EDIT.")
        println(io, "# Grammar oracle: ptxas ($PTXAS_VERSION), .version $PTX_VERSION,")
        println(io, "# ladder $(join(LADDER, ' ')). A form's floor is the first")
        println(io, "# accepting rung; ptxas enforces everything above it at compile")
        println(io, "# time. $(length(forms)) forms.")
        for arch in LADDER
            haskey(byfloor, arch) || continue
            ms = sort(byfloor[arch])
            println(io, "\ngrammar!(:cvt, :$arch, [")
            for m in ms
                println(io, "    (", join((":$s" for s in m), ", "),
                        length(m) == 1 ? "," : "", "),")
            end
            println(io, "])")
        end
        println(io, "\nclose_family!(:cvt)")
    end
    println("wrote $out ($(length(forms)) forms)")
end

floors = sweep()
check_anchors(floors)
emit(floors)
