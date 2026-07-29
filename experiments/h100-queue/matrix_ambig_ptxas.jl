# Issue #49 (MATRIX-AMBIG-EXP): reconcile disputed matrix forms with ptxas
# evidence across every toolkit on the machine. Pure assembly — no GPU.
#
# Run:   julia --project=test experiments/h100-queue/matrix_ambig_ptxas.jl
#
# Three dispute groups (see README.md for the spec citations):
#   D1  f8f6f4 K shape — mma.sync .kind::f8f6f4 documents only m16n8k32
#       dense / m16n8k64 sparse. Do toolkits accept k16 / k64 dense, and
#       does acceptance track the target or the .version stamp?
#   D2  integer WGMMA upper-N — PTX §9.7.17.5 grids integer N at
#       8,16,24,32,48..224 (step 16). Do toolkits enforce the documented
#       holes (n40, n56) and ceiling (n232..n256)?
#   D3  residual modifier ordering — tcgen05.commit's syntax block orders
#       qualifiers `.cta_group.completion_mechanism{.shared::cluster}
#       {.multicast}` while the spec's own example (§9.7.18.12.1) spells
#       `tcgen05.commit.shared::cluster.multicast::cluster::32b.cta_group::2
#       .mbarrier::arrive::one`, and this repo's validated wrapper uses a
#       third order (multicast before space, after the mechanism). Similar
#       order questions for mbarrier.arrive scope/space and ldmatrix
#       .trans/.shared. Which spellings does each ptxas accept?
#
# Every case assembles a minimal self-contained module. Errors are recorded
# with their first diagnostic line so shape rejections are distinguishable
# from operand-count mistakes.

using CUDACore

# --- assembler discovery -----------------------------------------------------

function _tool_version(tool)
    out = IOBuffer()
    ok = success(pipeline(ignorestatus(`$tool --version`); stdout = out, stderr = out))
    ok || return nothing
    m = match(r"release\s+([0-9.]+)", String(take!(out)))
    m === nothing ? "unknown" : m.captures[1]
end

function _discover_tools()
    tools = Tuple{String, Any}[]
    seen = Set{String}()
    art = try
        CUDACore.CUDA_Compiler.ptxas()
    catch
        nothing
    end
    art === nothing || push!(tools, ("artifact", art))
    candidates = String[]
    which = Sys.which("ptxas")
    which === nothing || push!(candidates, which)
    if isdir("/usr/local")
        for d in readdir("/usr/local"; join = true)
            startswith(basename(d), "cuda") || continue
            p = joinpath(d, "bin", "ptxas")
            isfile(p) && push!(candidates, p)
        end
    end
    labeled = Tuple{String, Any}[]
    for (label, tool) in tools
        v = _tool_version(tool)
        v === nothing && continue
        push!(seen, v)
        push!(labeled, ("$label($v)", tool))
    end
    for p in candidates
        v = _tool_version(p)
        (v === nothing || v in seen) && continue  # dedup by release
        push!(seen, v)
        push!(labeled, ("$(realpath(p))($v)", p))
    end
    labeled
end

# Highest .version the tool accepts, probed with a trivial module.
const _ISA_LADDER = ["9.4", "9.3", "9.2", "9.1", "9.0",
                     "8.8", "8.7", "8.6", "8.5", "8.4", "8.3", "8.0"]

function _assemble(tool, src::String, target::String)
    dir = mktempdir()
    ptx = joinpath(dir, "k.ptx")
    cub = joinpath(dir, "k.cubin")
    write(ptx, src)
    err = IOBuffer()
    ok = success(pipeline(ignorestatus(
        `$tool --compile-only --gpu-name $target --output-file $cub $ptx`);
        stdout = err, stderr = err))
    diag = ""
    if !ok
        lines = filter(l -> occursin("error", lowercase(l)), split(String(take!(err)), '\n'))
        diag = isempty(lines) ? "(no error line captured)" : strip(first(lines))
    end
    rm(dir; recursive = true, force = true)
    ok, diag
end

_module(version, body) = """
    .version $version
    .target placeholder
    .address_size 64
    .visible .entry k()
    {
    $body
        ret;
    }
    """
# .target is passed on the command line; the directive must still parse, so
# substitute the real target in.
_stamp(version, target, body) =
    replace(_module(version, body), ".target placeholder" => ".target $target")

function _isa_ceiling(tool)
    for v in _ISA_LADDER
        ok, _ = _assemble(tool, _stamp(v, "sm_90", "    "), "sm_90")
        ok && return v
    end
    nothing
end

# --- case definitions --------------------------------------------------------

_reglist(prefix, n) = "{" * join(("$prefix$i" for i in 0:(n - 1)), ", ") * "}"

# D2: integer WGMMA, SS form. N/2 s32 accumulators per thread.
function _wgmma_int_body(n)
    """
        .reg .b32 d<$(max(n ÷ 2, 4))>;
        .reg .b64 descA, descB;
        wgmma.fence.sync.aligned;
        wgmma.mma_async.sync.aligned.m64n$(n)k32.s32.s8.s8 $(_reglist("d", n ÷ 2)), descA, descB, 1;
        wgmma.commit_group.sync.aligned;
        wgmma.wait_group.sync.aligned 0;
    """
end

# fp control at the same N ceiling (e4m3, K32; d = N/2 f32 regs).
function _wgmma_fp_body(n)
    """
        .reg .f32 d<$(n ÷ 2)>;
        .reg .b64 descA, descB;
        wgmma.fence.sync.aligned;
        wgmma.mma_async.sync.aligned.m64n$(n)k32.f32.e4m3.e4m3 $(_reglist("d", n ÷ 2)), descA, descB, 1, 1, 1;
        wgmma.commit_group.sync.aligned;
        wgmma.wait_group.sync.aligned 0;
    """
end

# D1: mma.sync kind::f8f6f4 dense. a = k/8, b = k/16 b32 regs; f32 accum.
function _mma_f8f6f4(k)
    a, b = k ÷ 8, k ÷ 16
    alist = "{" * join(("a$i" for i in 0:(a - 1)), ", ") * "}"
    blist = "{" * join(("b$i" for i in 0:(b - 1)), ", ") * "}"
    """
        .reg .f32 f<8>;
        .reg .b32 a<$a>;
        .reg .b32 b<$b>;
        mma.sync.aligned.m16n8k$(k).row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 {f0, f1, f2, f3}, $alist, $blist, {f4, f5, f6, f7};
    """
end

# D1 sparse control: documented m16n8k64 sp::ordered_metadata (A halved).
const _MMA_SP_F8F6F4_K64 = """
    .reg .f32 f<8>;
    .reg .b32 a<4>;
    .reg .b32 b<4>;
    .reg .b32 e0;
    mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 {f0, f1, f2, f3}, {a0, a1, a2, a3}, {b0, b1, b2, b3}, {f4, f5, f6, f7}, e0, 0x0;
"""

# D3: tcgen05.commit qualifier orders (16b-default multicast, ISA 8.6 floor).
_tcgen05_commit(quals) = """
    .reg .b32 r0;
    .reg .b16 h0;
    tcgen05.commit.$quals.b64 [r0], h0;
"""
const _TCGEN05_COMMIT_PLAIN = """
    .reg .b32 r0;
    tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [r0];
"""

# D3: mbarrier.arrive sem/scope vs space order (sink dest, cluster form).
_mbarrier_arrive(quals) = """
    .reg .b32 a0;
    mbarrier.arrive.$quals.b64 _, [a0];
"""

# D3: 9.4 multicast::cluster::32b arrive orders (G1/G4-gated: needs a
# 13.4+ ptxas and sm_107f — expected UNSUPPORTED until then; kept so the
# matrix self-documents when the gate opens).
_mbarrier_arrive_mc(quals) = """
    .reg .b32 a0, m0;
    mbarrier.arrive.$quals.b64 _, [a0], m0;
"""

# D3: ldmatrix .trans/.shared order.
_ldmatrix(quals) = """
    .reg .b32 r0, a0;
    ldmatrix.sync.aligned.m8n8.x1.$quals.b16 {r0}, [a0];
"""

struct Case
    name::String
    floor::String            # .version stamp (also the documented intro)
    targets::Vector{String}
    body::String
    also_at_ceiling::Bool    # re-assemble at the tool's max ISA to separate
end                          # version-gating from target-gating

const CASES = Case[
    # --- D1: f8f6f4 K shape ---
    Case("mma.f8f6f4.k32.dense   [documented]", "8.7",
         ["sm_100a", "sm_103a", "sm_120a", "sm_121a"], _mma_f8f6f4(32), true),
    Case("mma.f8f6f4.k16.dense   [disputed]", "8.7",
         ["sm_100a", "sm_120a"], _mma_f8f6f4(16), true),
    Case("mma.f8f6f4.k64.dense   [disputed]", "8.7",
         ["sm_100a", "sm_103a", "sm_120a", "sm_121a"], _mma_f8f6f4(64), true),
    Case("mma.sp::om.f8f6f4.k64  [documented sparse]", "8.7",
         ["sm_100a", "sm_120a"], _MMA_SP_F8F6F4_K64, false),
    # --- D2: integer WGMMA upper-N ---
    Case("wgmma.s8.n224          [documented max]", "8.0", ["sm_90a"], _wgmma_int_body(224), false),
    Case("wgmma.s8.n232          [documented-illegal]", "8.0", ["sm_90a"], _wgmma_int_body(232), false),
    Case("wgmma.s8.n240          [documented-illegal]", "8.0", ["sm_90a"], _wgmma_int_body(240), false),
    Case("wgmma.s8.n248          [documented-illegal]", "8.0", ["sm_90a"], _wgmma_int_body(248), false),
    Case("wgmma.s8.n256          [documented-illegal]", "8.0", ["sm_90a"], _wgmma_int_body(256), false),
    Case("wgmma.s8.n40           [documented-hole]", "8.0", ["sm_90a"], _wgmma_int_body(40), false),
    Case("wgmma.s8.n56           [documented-hole]", "8.0", ["sm_90a"], _wgmma_int_body(56), false),
    Case("wgmma.e4m3.n256        [fp control]", "8.0", ["sm_90a"], _wgmma_fp_body(256), false),
    # --- D3: modifier ordering ---
    Case("tcgen05.commit plain   [control]", "8.6", ["sm_100a"], _TCGEN05_COMMIT_PLAIN, false),
    Case("tcgen05.commit order: mech.space.mc [syntax block]", "8.6", ["sm_100a"],
         _tcgen05_commit("cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster"), false),
    Case("tcgen05.commit order: mech.mc.space [this repo]", "8.6", ["sm_100a"],
         _tcgen05_commit("cta_group::1.mbarrier::arrive::one.multicast::cluster.shared::cluster"), false),
    Case("tcgen05.commit order: space.mc.mech [spec example]", "8.6", ["sm_100a"],
         _tcgen05_commit("shared::cluster.multicast::cluster.cta_group::1.mbarrier::arrive::one"), false),
    Case("mbarrier.arrive order: sem.scope.space [syntax block]", "8.0", ["sm_90"],
         _mbarrier_arrive("release.cluster.shared::cluster"), false),
    Case("mbarrier.arrive order: space.sem.scope [alt]", "8.0", ["sm_90"],
         _mbarrier_arrive("shared::cluster.release.cluster"), false),
    Case("mbarrier.arrive mc32b: space.mc [9.4 syntax block]", "9.4", ["sm_107f"],
         _mbarrier_arrive_mc("release.cluster.shared::cluster.multicast::cluster::32b"), false),
    Case("mbarrier.arrive mc32b: mc.space [alt]", "9.4", ["sm_107f"],
         _mbarrier_arrive_mc("release.cluster.multicast::cluster::32b.shared::cluster"), false),
    Case("ldmatrix order: trans.shared [syntax block]", "6.5", ["sm_75"],
         _ldmatrix("trans.shared"), false),
    Case("ldmatrix order: shared.trans [alt]", "6.5", ["sm_75"],
         _ldmatrix("shared.trans"), false),
]

# --- SASS leg ----------------------------------------------------------------
# For disputed forms ptxas ACCEPTS, check whether real SASS is emitted (the
# accumulators must be consumed or ptxas nulls the op to `HGMMA ... RZ`).

function _wgmma_int_consumed(n)
    regs = join(("d$i" for i in 0:(n ÷ 2 - 1)), ", ")
    stores = join(("    st.global.b32 [ptr+$(4i)], d$i;" for i in 0:16:(n ÷ 2 - 1)), '\n')
    """
    .version 8.0
    .target sm_90a
    .address_size 64
    .visible .entry k(.param .u64 p)
    {
        .reg .b64 %rd<2>;
        .reg .b32 d<$(n ÷ 2)>;
        .reg .b64 descA, descB, ptr;
        ld.param.u64 %rd0, [p];
        cvta.to.global.u64 ptr, %rd0;
        wgmma.fence.sync.aligned;
        wgmma.mma_async.sync.aligned.m64n$(n)k32.s32.s8.s8 {$regs}, descA, descB, 1;
        wgmma.commit_group.sync.aligned;
        wgmma.wait_group.sync.aligned 0;
    $stores
        ret;
    }
    """
end

function _mma_f8f6f4_consumed(k)
    a, b = k ÷ 8, k ÷ 16
    alist = join(("a$i" for i in 0:(a - 1)), ", ")
    blist = join(("b$i" for i in 0:(b - 1)), ", ")
    """
    .version 8.7
    .target sm_120a
    .address_size 64
    .visible .entry k(.param .u64 p)
    {
        .reg .b64 %rd<2>;
        .reg .f32 f<8>;
        .reg .b32 a<$a>;
        .reg .b32 b<$b>;
        .reg .b64 ptr;
        ld.param.u64 %rd0, [p];
        cvta.to.global.u64 ptr, %rd0;
        mma.sync.aligned.m16n8k$(k).row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 {f0, f1, f2, f3}, {$alist}, {$blist}, {f4, f5, f6, f7};
        st.global.v4.f32 [ptr], {f0, f1, f2, f3};
        ret;
    }
    """
end

const SASS_CASES = [
    ("wgmma.s8.n224 [documented max]", "sm_90a", _wgmma_int_consumed(224)),
    ("wgmma.s8.n240 [documented-illegal, accepted]", "sm_90a", _wgmma_int_consumed(240)),
    ("wgmma.s8.n256 [documented-illegal, accepted]", "sm_90a", _wgmma_int_consumed(256)),
    ("mma.f8f6f4.k32 [documented]", "sm_120a", _mma_f8f6f4_consumed(32)),
    ("mma.f8f6f4.k16 [disputed, accepted]", "sm_120a", _mma_f8f6f4_consumed(16)),
]

function _nvdisasm_for(label, tool)
    if occursin("artifact", label)
        return try
            CUDACore.CUDA_Compiler.nvdisasm()
        catch
            nothing
        end
    end
    p = joinpath(dirname(String(tool)), "nvdisasm")
    isfile(p) ? p : nothing
end

function _sass_mma_lines(tool, dis, src, target)
    dir = mktempdir()
    ptx = joinpath(dir, "k.ptx")
    cub = joinpath(dir, "k.cubin")
    write(ptx, src)
    ok = success(pipeline(ignorestatus(
        `$tool --compile-only --gpu-name $target --output-file $cub $ptx`);
        stdout = devnull, stderr = devnull))
    lines = nothing
    if ok
        out = IOBuffer()
        if success(pipeline(ignorestatus(`$dis -c $cub`); stdout = out, stderr = devnull))
            lines = [strip(l) for l in split(String(take!(out)), '\n') if occursin("MMA", l)]
        end
    end
    rm(dir; recursive = true, force = true)
    ok, lines
end

# --- run ---------------------------------------------------------------------

tools = _discover_tools()
isempty(tools) && error("no working ptxas found (artifact, PATH, /usr/local/cuda*)")

for (label, tool) in tools
    ceiling = _isa_ceiling(tool)
    println("=" ^ 78)
    println("ptxas: ", label, "   max .version accepted: ", something(ceiling, "?"))
    println("=" ^ 78)
    for case in CASES
        versions = [case.floor]
        if case.also_at_ceiling && ceiling !== nothing &&
           VersionNumber(ceiling) > VersionNumber(case.floor)
            push!(versions, ceiling)
        end
        for target in case.targets, v in versions
            if ceiling === nothing || VersionNumber(v) > VersionNumber(ceiling)
                println(rpad(case.name, 52), " ", rpad(target, 9), " @", v,
                        "  VERSION-UNSUPPORTED by this ptxas")
                continue
            end
            ok, diag = _assemble(tool, _stamp(v, target, case.body), target)
            println(rpad(case.name, 52), " ", rpad(target, 9), " @", v, "  ",
                    ok ? "ACCEPT" : "reject: $diag")
        end
    end
end
println()
println("=" ^ 78)
println("SASS leg — do accepted-but-undocumented forms emit real encodings?")
println("=" ^ 78)
for (label, tool) in tools
    dis = _nvdisasm_for(label, tool)
    if dis === nothing
        println(label, ": no nvdisasm alongside this ptxas — skipped")
        continue
    end
    println("-- ", label)
    for (name, target, src) in SASS_CASES
        ok, lines = _sass_mma_lines(tool, dis, src, target)
        if !ok
            println(rpad(name, 46), " ", target, "  ptxas rejected the consumed variant")
        elseif lines === nothing
            println(rpad(name, 46), " ", target, "  nvdisasm failed")
        else
            println(rpad(name, 46), " ", target, "  ",
                    isempty(lines) ? "NO MMA in SASS" : join(lines, "  ||  "))
        end
    end
end
println()
println("done — paste this output into issue #49")
