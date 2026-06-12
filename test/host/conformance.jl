# Registry <-> backend conformance (DESIGN.md, "The registry"; CONCERNS.md,
# "Weakdep compat as the backend pin" / "Obtaining llvm-tblgen"). Three
# layers, each catching a different way the committed table can rot:
#
#   1. Version: the artifact the environment resolved is the version the
#      table was generated from (compat pins the major; this catches
#      minor/patch skew worth knowing about at a glance).
#   2. Names: every table entry's name exists in the artifact llc binary's
#      intrinsic name table, and vice versa — the same diff the extraction
#      pipeline runs at generation time, re-checked continuously against
#      the binary that environments actually resolve.
#   3. Selection: for every intrinsic the package's wrappers stand on,
#      synthesized IR compiles through the artifact `llc` and selects the
#      *expected instruction*. Acceptance alone proves nothing — llc
#      remangles wrong names and upgrades sloppy callsites silently — so
#      each probe asserts instruction text.
#
# The probe list is self-policing: the testset scans src/ for `nvvm"..."`
# literals and fails if any used intrinsic lacks a probe. Wrappers
# therefore spell intrinsic names literally (greppable by design).

using PTX
using PTX.NVVM: NVVM, synthesize, TABLE, BACKEND_LLVM_VERSION
using NVPTX_LLVM_Backend_jll: NVPTX_LLVM_Backend_jll, llc

@testset "backend artifact matches the registry's generation version" begin
    jll = pkgversion(NVPTX_LLVM_Backend_jll)
    @test VersionNumber(jll.major, jll.minor, jll.patch) == BACKEND_LLVM_VERSION
end

@testset "table names == llc binary's intrinsic name table" begin
    blob = read(llc().exec[1], String)
    binary_names = Set{String}()
    for s in eachsplit(blob, '\0')
        startswith(s, "llvm.nvvm.") && push!(binary_names, String(s))
    end
    table_names = Set(keys(TABLE))
    @test isempty(setdiff(table_names, binary_names))   # stale entries
    @test isempty(setdiff(binary_names, table_names))   # missing entries
end

# --- Selection probes --------------------------------------------------------

const pS  = Core.LLVMPtr{UInt64, 3}
const pS8 = Core.LLVMPtr{UInt8, 3}
const p7  = Core.LLVMPtr{UInt8, 7}    # shared::cluster
const p0  = Core.LLVMPtr{UInt8, 0}    # generic (TMA descriptor)
const I4 = (UInt32, UInt32, UInt32, UInt32)

# (name, argtypes, mcpu, mattr, expected instruction regex) — a vector, not
# a dict, so one intrinsic can carry several probes (immarg flag combos
# select different instruction qualifiers).
const PROBES = [
    # shfl (wrappers/shfl.jl) — data and {i32,i1}-pred forms
    ("llvm.nvvm.shfl.sync.idx.i32",   I4, "sm_70", "+ptx60", r"shfl\.sync\.idx\.b32 \s*%r\d+,"),
    ("llvm.nvvm.shfl.sync.idx.i32p",  I4, "sm_70", "+ptx60", r"shfl\.sync\.idx\.b32 \s*%r\d+\|%p\d+,"),
    ("llvm.nvvm.shfl.sync.up.i32",    I4, "sm_70", "+ptx60", r"shfl\.sync\.up\.b32 \s*%r\d+,"),
    ("llvm.nvvm.shfl.sync.up.i32p",   I4, "sm_70", "+ptx60", r"shfl\.sync\.up\.b32 \s*%r\d+\|%p\d+,"),
    ("llvm.nvvm.shfl.sync.down.i32",  I4, "sm_70", "+ptx60", r"shfl\.sync\.down\.b32 \s*%r\d+,"),
    ("llvm.nvvm.shfl.sync.down.i32p", I4, "sm_70", "+ptx60", r"shfl\.sync\.down\.b32 \s*%r\d+\|%p\d+,"),
    ("llvm.nvvm.shfl.sync.bfly.i32",  I4, "sm_70", "+ptx60", r"shfl\.sync\.bfly\.b32 \s*%r\d+,"),
    ("llvm.nvvm.shfl.sync.bfly.i32p", I4, "sm_70", "+ptx60", r"shfl\.sync\.bfly\.b32 \s*%r\d+\|%p\d+,"),

    # mbarrier (wrappers/mbarrier.jl) — legacy intrinsics at the sm_80
    # floor, scoped ones for parity and the sm_90 forms; expected spellings
    # per the golden diffs (expect_tx emits its explicit default)
    ("llvm.nvvm.mbarrier.init.shared",
        (pS, UInt32), "sm_80", "+ptx71", r"mbarrier\.init\.shared\.b64"),
    ("llvm.nvvm.mbarrier.inval.shared",
        (pS,), "sm_80", "+ptx71", r"mbarrier\.inval\.shared\.b64"),
    ("llvm.nvvm.mbarrier.arrive.shared",
        (pS,), "sm_80", "+ptx71", r"mbarrier\.arrive\.shared\.b64"),
    ("llvm.nvvm.mbarrier.arrive.noComplete.shared",
        (pS, UInt32), "sm_80", "+ptx71", r"mbarrier\.arrive\.noComplete\.shared\.b64"),
    ("llvm.nvvm.mbarrier.test.wait.shared",
        (pS, UInt64), "sm_80", "+ptx71", r"mbarrier\.test_wait\.shared\.b64"),
    ("llvm.nvvm.mbarrier.test.wait.parity.scope.cta.space.cta",
        (pS, UInt32), "sm_80", "+ptx71", r"mbarrier\.test_wait\.parity\.shared\.b64"),
    ("llvm.nvvm.mbarrier.expect.tx.scope.cta.space.cta",
        (pS, UInt32), "sm_90", "+ptx80", r"mbarrier\.expect_tx\.relaxed\.cta\.shared\.b64"),
    ("llvm.nvvm.mbarrier.arrive.expect.tx.scope.cta.space.cta",
        (pS, UInt32), "sm_90", "+ptx80", r"mbarrier\.arrive\.expect_tx\.shared\.b64"),
    ("llvm.nvvm.mbarrier.try.wait.scope.cta.space.cta",
        (pS, UInt64), "sm_90", "+ptx80", r"mbarrier\.try_wait\.shared\.b64"),
    ("llvm.nvvm.mbarrier.try.wait.parity.scope.cta.space.cta",
        (pS, UInt32), "sm_90", "+ptx80", r"mbarrier\.try_wait\.parity\.shared\.b64"),

    # TMA (wrappers/tma.jl) — g2s takes (dst p7, mbar p3, tmap p0, coords,
    # mc i16, ch i64, flag_mc, flag_ch, cta_group); g2s.cta and s2g drop the
    # cluster-only operands. Flag immarg combos are probed separately below
    # the per-rank base forms.
    ("llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d",
        (p7, pS, p0, Int32, UInt16, UInt64, Val{false}, Val{false}, Val{0}),
        "sm_90", "+ptx80",
        r"cp\.async\.bulk\.tensor\.1d\.shared::cluster\.global\.tile\.mbarrier::complete_tx::bytes \["),
    ("llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d",
        (p7, pS, p0, Int32, Int32, UInt16, UInt64, Val{false}, Val{false}, Val{0}),
        "sm_90", "+ptx80",
        r"cp\.async\.bulk\.tensor\.2d\.shared::cluster\.global\.tile\.mbarrier::complete_tx::bytes \["),
    ("llvm.nvvm.cp.async.bulk.tensor.g2s.tile.3d",
        (p7, pS, p0, Int32, Int32, Int32, UInt16, UInt64, Val{false}, Val{false}, Val{0}),
        "sm_90", "+ptx80",
        r"cp\.async\.bulk\.tensor\.3d\.shared::cluster\.global\.tile\.mbarrier::complete_tx::bytes \["),
    ("llvm.nvvm.cp.async.bulk.tensor.g2s.tile.4d",
        (p7, pS, p0, Int32, Int32, Int32, Int32, UInt16, UInt64, Val{false}, Val{false}, Val{0}),
        "sm_90", "+ptx80",
        r"cp\.async\.bulk\.tensor\.4d\.shared::cluster\.global\.tile\.mbarrier::complete_tx::bytes \["),
    ("llvm.nvvm.cp.async.bulk.tensor.g2s.tile.5d",
        (p7, pS, p0, Int32, Int32, Int32, Int32, Int32, UInt16, UInt64, Val{false}, Val{false}, Val{0}),
        "sm_90", "+ptx80",
        r"cp\.async\.bulk\.tensor\.5d\.shared::cluster\.global\.tile\.mbarrier::complete_tx::bytes \["),
    # flag combos: multicast (sm_90), cta_group::2 and both (sm_100a; ISel
    # renders cta_group trailing — notation non-WYSIWYG, see wrappers/tma.jl)
    ("llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d",
        (p7, pS, p0, Int32, Int32, UInt16, UInt64, Val{true}, Val{false}, Val{0}),
        "sm_90", "+ptx80",
        r"cp\.async\.bulk\.tensor\.2d\.shared::cluster\.global\.tile\.mbarrier::complete_tx::bytes\.multicast::cluster \["),
    ("llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d",
        (p7, pS, p0, Int32, Int32, UInt16, UInt64, Val{false}, Val{false}, Val{2}),
        "sm_100a", "+ptx86",
        r"cp\.async\.bulk\.tensor\.2d\.shared::cluster\.global\.tile\.mbarrier::complete_tx::bytes\.cta_group::2 \["),
    ("llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d",
        (p7, pS, p0, Int32, Int32, UInt16, UInt64, Val{true}, Val{false}, Val{2}),
        "sm_100a", "+ptx86",
        r"cp\.async\.bulk\.tensor\.2d\.shared::cluster\.global\.tile\.mbarrier::complete_tx::bytes\.multicast::cluster\.cta_group::2 \["),
    # shared::cta destination — PTX 8.6 floor (cannot ISel at +ptx80)
    ("llvm.nvvm.cp.async.bulk.tensor.g2s.cta.tile.1d",
        (pS8, pS, p0, Int32, UInt64, Val{false}), "sm_90", "+ptx86",
        r"cp\.async\.bulk\.tensor\.1d\.shared::cta\.global\.tile\.mbarrier::complete_tx::bytes \["),
    ("llvm.nvvm.cp.async.bulk.tensor.g2s.cta.tile.2d",
        (pS8, pS, p0, Int32, Int32, UInt64, Val{false}), "sm_90", "+ptx86",
        r"cp\.async\.bulk\.tensor\.2d\.shared::cta\.global\.tile\.mbarrier::complete_tx::bytes \["),
    ("llvm.nvvm.cp.async.bulk.tensor.g2s.cta.tile.3d",
        (pS8, pS, p0, Int32, Int32, Int32, UInt64, Val{false}), "sm_90", "+ptx86",
        r"cp\.async\.bulk\.tensor\.3d\.shared::cta\.global\.tile\.mbarrier::complete_tx::bytes \["),
    ("llvm.nvvm.cp.async.bulk.tensor.g2s.cta.tile.4d",
        (pS8, pS, p0, Int32, Int32, Int32, Int32, UInt64, Val{false}), "sm_90", "+ptx86",
        r"cp\.async\.bulk\.tensor\.4d\.shared::cta\.global\.tile\.mbarrier::complete_tx::bytes \["),
    ("llvm.nvvm.cp.async.bulk.tensor.g2s.cta.tile.5d",
        (pS8, pS, p0, Int32, Int32, Int32, Int32, Int32, UInt64, Val{false}), "sm_90", "+ptx86",
        r"cp\.async\.bulk\.tensor\.5d\.shared::cta\.global\.tile\.mbarrier::complete_tx::bytes \["),
    # stores (s2g)
    ("llvm.nvvm.cp.async.bulk.tensor.s2g.tile.1d",
        (pS8, p0, Int32, UInt64, Val{false}), "sm_90", "+ptx80",
        r"cp\.async\.bulk\.tensor\.1d\.global\.shared::cta\.tile\.bulk_group \["),
    ("llvm.nvvm.cp.async.bulk.tensor.s2g.tile.2d",
        (pS8, p0, Int32, Int32, UInt64, Val{false}), "sm_90", "+ptx80",
        r"cp\.async\.bulk\.tensor\.2d\.global\.shared::cta\.tile\.bulk_group \["),
    ("llvm.nvvm.cp.async.bulk.tensor.s2g.tile.3d",
        (pS8, p0, Int32, Int32, Int32, UInt64, Val{false}), "sm_90", "+ptx80",
        r"cp\.async\.bulk\.tensor\.3d\.global\.shared::cta\.tile\.bulk_group \["),
    ("llvm.nvvm.cp.async.bulk.tensor.s2g.tile.4d",
        (pS8, p0, Int32, Int32, Int32, Int32, UInt64, Val{false}), "sm_90", "+ptx80",
        r"cp\.async\.bulk\.tensor\.4d\.global\.shared::cta\.tile\.bulk_group \["),
    ("llvm.nvvm.cp.async.bulk.tensor.s2g.tile.5d",
        (pS8, p0, Int32, Int32, Int32, Int32, Int32, UInt64, Val{false}), "sm_90", "+ptx80",
        r"cp\.async\.bulk\.tensor\.5d\.global\.shared::cta\.tile\.bulk_group \["),
]

"All intrinsic names spelled as nvvm\"...\" literals under src/."
function used_intrinsics()
    used = Set{String}()
    for (root, _, files) in walkdir(joinpath(pkgdir(PTX), "src"))
        for f in files
            endswith(f, ".jl") || continue
            for m in eachmatch(r"nvvm\"([^\"]+)\"", read(joinpath(root, f), String))
                n = m.captures[1]
                n = startswith(n, "llvm.nvvm.") ? n : "llvm.nvvm." * n
                # docstring placeholders (nvvm"name") aren't registered; a
                # misspelled *real* literal can't survive package load — the
                # macro errors at expansion — so this filter drops only docs
                NVVM.isintrinsic(n) && push!(used, n)
            end
        end
    end
    return used
end

@testset "every wrapper intrinsic has a selection probe" begin
    used = used_intrinsics()
    @test !isempty(used)
    probed = Set(p[1] for p in PROBES)
    unprobed = sort!(collect(setdiff(used, probed)))
    isempty(unprobed) ||
        @info "add probes to test/host/conformance.jl for:" unprobed
    @test isempty(unprobed)
end

@testset "selection probes through the artifact llc" begin
    exe = llc().exec[1]
    for (name, argtypes, mcpu, mattr, expect) in sort(PROBES; by=first)
        s = synthesize(name, argtypes)
        ll = "target triple = \"nvptx64-nvidia-cuda\"\n" * s.ir
        out = IOBuffer(); err = IOBuffer()
        ok = success(pipeline(`$exe -mcpu=$mcpu -mattr=$mattr -o -`;
                              stdin = IOBuffer(ll), stdout = out, stderr = err))
        ptx = String(take!(out))
        if !(ok && occursin(expect, ptx))
            @info "probe failed" name mcpu llc_error=String(take!(err)) ptx
        end
        @test ok
        @test occursin(expect, ptx)
    end
end
