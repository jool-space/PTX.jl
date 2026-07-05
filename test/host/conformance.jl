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
const pS16 = Core.LLVMPtr{UInt16, 3}
const p7  = Core.LLVMPtr{UInt8, 7}    # shared::cluster
const p0  = Core.LLVMPtr{UInt8, 0}    # generic (TMA descriptor)
const I4 = (UInt32, UInt32, UInt32, UInt32)

# (name, argtypes, mcpu, mattr, expected instruction regex) — a vector, not
# a dict, so one intrinsic can carry several probes (immarg flag combos
# select different instruction qualifiers). Families with regular
# shape×count grids (tcgen05 ld/st) append their probes in loops below the
# literal — the probe *data* may be generated; only wrapper sources must
# spell intrinsic names literally.
const PROBES = Tuple{String, Tuple, String, String, Regex}[
    # shfl (wrappers/shfl.jl) — data and {i32,i1}-pred forms
    ("llvm.nvvm.shfl.sync.idx.i32",   I4, "sm_70", "+ptx60", r"shfl\.sync\.idx\.b32 \s*%r\d+,"),
    ("llvm.nvvm.shfl.sync.idx.i32p",  I4, "sm_70", "+ptx60", r"shfl\.sync\.idx\.b32 \s*%r\d+\|%p\d+,"),
    ("llvm.nvvm.shfl.sync.up.i32",    I4, "sm_70", "+ptx60", r"shfl\.sync\.up\.b32 \s*%r\d+,"),
    ("llvm.nvvm.shfl.sync.up.i32p",   I4, "sm_70", "+ptx60", r"shfl\.sync\.up\.b32 \s*%r\d+\|%p\d+,"),
    ("llvm.nvvm.shfl.sync.down.i32",  I4, "sm_70", "+ptx60", r"shfl\.sync\.down\.b32 \s*%r\d+,"),
    ("llvm.nvvm.shfl.sync.down.i32p", I4, "sm_70", "+ptx60", r"shfl\.sync\.down\.b32 \s*%r\d+\|%p\d+,"),
    ("llvm.nvvm.shfl.sync.bfly.i32",  I4, "sm_70", "+ptx60", r"shfl\.sync\.bfly\.b32 \s*%r\d+,"),
    ("llvm.nvvm.shfl.sync.bfly.i32p", I4, "sm_70", "+ptx60", r"shfl\.sync\.bfly\.b32 \s*%r\d+\|%p\d+,"),

    # barrier.cluster (wrappers/barrier_cluster.jl) — sm_90 floor
    ("llvm.nvvm.barrier.cluster.arrive",
        (), "sm_90", "+ptx78", r"barrier\.cluster\.arrive;"),
    ("llvm.nvvm.barrier.cluster.arrive.relaxed",
        (), "sm_90", "+ptx80", r"barrier\.cluster\.arrive\.relaxed;"),
    ("llvm.nvvm.barrier.cluster.wait",
        (), "sm_90", "+ptx78", r"barrier\.cluster\.wait;"),
    ("llvm.nvvm.barrier.cluster.arrive.aligned",
        (), "sm_90", "+ptx78", r"barrier\.cluster\.arrive\.aligned;"),
    ("llvm.nvvm.barrier.cluster.arrive.relaxed.aligned",
        (), "sm_90", "+ptx80", r"barrier\.cluster\.arrive\.relaxed\.aligned;"),
    ("llvm.nvvm.barrier.cluster.wait.aligned",
        (), "sm_90", "+ptx78", r"barrier\.cluster\.wait\.aligned;"),

    # ldmatrix/stmatrix (wrappers/{ldmatrix,stmatrix}.jl) — b16 shapes at
    # their family floors, b8 shapes at sm_100a (family-specific). The
    # brace groups in the regexes pin destination/source arity — the b8
    # shapes are 2 regs per count step, the bug class the old asm
    # generator had.
    ("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.b16",
        (pS16,), "sm_75", "+ptx65",
        r"ldmatrix\.sync\.aligned\.m8n8\.x1\.shared\.b16 \s*\{%r\d+\},"),
    ("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.trans.b16",
        (pS16,), "sm_75", "+ptx65",
        r"ldmatrix\.sync\.aligned\.m8n8\.x1\.trans\.shared\.b16 \s*\{%r\d+\},"),
    ("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.b16",
        (pS16,), "sm_75", "+ptx65",
        r"ldmatrix\.sync\.aligned\.m8n8\.x2\.shared\.b16 \s*\{%r\d+, %r\d+\},"),
    ("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.trans.b16",
        (pS16,), "sm_75", "+ptx65",
        r"ldmatrix\.sync\.aligned\.m8n8\.x2\.trans\.shared\.b16 \s*\{%r\d+, %r\d+\},"),
    ("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16",
        (pS16,), "sm_75", "+ptx65",
        r"ldmatrix\.sync\.aligned\.m8n8\.x4\.shared\.b16 \s*\{%r\d+, %r\d+, %r\d+, %r\d+\},"),
    ("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.trans.b16",
        (pS16,), "sm_75", "+ptx65",
        r"ldmatrix\.sync\.aligned\.m8n8\.x4\.trans\.shared\.b16 \s*\{%r\d+, %r\d+, %r\d+, %r\d+\},"),
    ("llvm.nvvm.ldmatrix.sync.aligned.m16n16.x1.trans.b8",
        (pS8,), "sm_100a", "+ptx86",
        r"ldmatrix\.sync\.aligned\.m16n16\.x1\.trans\.shared\.b8 \s*\{%r\d+, %r\d+\},"),
    ("llvm.nvvm.ldmatrix.sync.aligned.m16n16.x2.trans.b8",
        (pS8,), "sm_100a", "+ptx86",
        r"ldmatrix\.sync\.aligned\.m16n16\.x2\.trans\.shared\.b8 \s*\{%r\d+, %r\d+, %r\d+, %r\d+\},"),
    ("llvm.nvvm.stmatrix.sync.aligned.m8n8.x1.b16",
        (pS16, UInt32), "sm_90", "+ptx78",
        r"stmatrix\.sync\.aligned\.m8n8\.x1\.shared\.b16 \s*\[%rd?\d+\], \{%r\d+\};"),
    ("llvm.nvvm.stmatrix.sync.aligned.m8n8.x1.trans.b16",
        (pS16, UInt32), "sm_90", "+ptx78",
        r"stmatrix\.sync\.aligned\.m8n8\.x1\.trans\.shared\.b16 \s*\[%rd?\d+\], \{%r\d+\};"),
    ("llvm.nvvm.stmatrix.sync.aligned.m8n8.x2.b16",
        (pS16, UInt32, UInt32), "sm_90", "+ptx78",
        r"stmatrix\.sync\.aligned\.m8n8\.x2\.shared\.b16 \s*\[%rd?\d+\], \{%r\d+, %r\d+\};"),
    ("llvm.nvvm.stmatrix.sync.aligned.m8n8.x2.trans.b16",
        (pS16, UInt32, UInt32), "sm_90", "+ptx78",
        r"stmatrix\.sync\.aligned\.m8n8\.x2\.trans\.shared\.b16 \s*\[%rd?\d+\], \{%r\d+, %r\d+\};"),
    ("llvm.nvvm.stmatrix.sync.aligned.m8n8.x4.b16",
        (pS16, UInt32, UInt32, UInt32, UInt32), "sm_90", "+ptx78",
        r"stmatrix\.sync\.aligned\.m8n8\.x4\.shared\.b16 \s*\[%rd?\d+\], \{%r\d+, %r\d+, %r\d+, %r\d+\};"),
    ("llvm.nvvm.stmatrix.sync.aligned.m8n8.x4.trans.b16",
        (pS16, UInt32, UInt32, UInt32, UInt32), "sm_90", "+ptx78",
        r"stmatrix\.sync\.aligned\.m8n8\.x4\.trans\.shared\.b16 \s*\[%rd?\d+\], \{%r\d+, %r\d+, %r\d+, %r\d+\};"),
    ("llvm.nvvm.stmatrix.sync.aligned.m16n8.x1.trans.b8",
        (pS8, UInt32), "sm_100a", "+ptx86",
        r"stmatrix\.sync\.aligned\.m16n8\.x1\.trans\.shared\.b8 \s*\[%rd?\d+\], \{%r\d+\};"),
    ("llvm.nvvm.stmatrix.sync.aligned.m16n8.x2.trans.b8",
        (pS8, UInt32, UInt32), "sm_100a", "+ptx86",
        r"stmatrix\.sync\.aligned\.m16n8\.x2\.trans\.shared\.b8 \s*\[%rd?\d+\], \{%r\d+, %r\d+\};"),
    ("llvm.nvvm.stmatrix.sync.aligned.m16n8.x4.trans.b8",
        (pS8, UInt32, UInt32, UInt32, UInt32), "sm_100a", "+ptx86",
        r"stmatrix\.sync\.aligned\.m16n8\.x4\.trans\.shared\.b8 \s*\[%rd?\d+\], \{%r\d+, %r\d+, %r\d+, %r\d+\};"),

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

# tcgen05 (wrappers/tcgen05.jl) — datacenter Blackwell only (consumer
# sm_12x has no tensor memory; ISel enforces what ptxas did). Expected
# spellings per the golden diff: commit emits `.shared::cluster` for both
# state-space notations and orders multicast after the space; dense mma is
# the maskless form with collector::a::discard explicit (PTX 8.8).
const p6 = Core.LLVMPtr{UInt8, 6}
for cg in (1, 2)
    append!(PROBES, [
        ("llvm.nvvm.tcgen05.shift.down.cg$cg", (p6,), "sm_100a", "+ptx86",
            Regex("tcgen05\\.shift\\.cta_group::$cg\\.down")),
        ("llvm.nvvm.tcgen05.dealloc.cg$cg", (p6, UInt32), "sm_100a", "+ptx86",
            Regex("tcgen05\\.dealloc\\.cta_group::$cg\\.sync\\.aligned\\.b32")),
        ("llvm.nvvm.tcgen05.alloc.shared.cg$cg", (pS8, UInt32), "sm_100a", "+ptx86",
            Regex("tcgen05\\.alloc\\.cta_group::$cg\\.sync\\.aligned\\.shared::cta\\.b32")),
        ("llvm.nvvm.tcgen05.relinq.alloc.permit.cg$cg", (), "sm_100a", "+ptx86",
            Regex("tcgen05\\.relinquish_alloc_permit\\.cta_group::$cg\\.sync\\.aligned")),
        ("llvm.nvvm.tcgen05.commit.shared.cg$cg", (pS8,), "sm_100a", "+ptx86",
            Regex("tcgen05\\.commit\\.cta_group::$cg\\.mbarrier::arrive::one\\.shared::cluster\\.b64")),
        ("llvm.nvvm.tcgen05.commit.mc.shared.cg$cg", (pS8, UInt16), "sm_100a", "+ptx86",
            Regex("tcgen05\\.commit\\.cta_group::$cg\\.mbarrier::arrive::one\\.shared::cluster\\.multicast::cluster\\.b64")),
    ])
    for shape in ("128x256b", "4x256b", "128x128b")
        push!(PROBES, ("llvm.nvvm.tcgen05.cp.$shape.cg$cg", (p6, UInt64),
                       "sm_100a", "+ptx86",
                       Regex("tcgen05\\.cp\\.cta_group::$cg\\.$shape")))
    end
end
for w in ("ld", "st")
    push!(PROBES, ("llvm.nvvm.tcgen05.wait.$w", (), "sm_100a", "+ptx86",
                   Regex("tcgen05\\.wait::$w\\.sync\\.aligned")))
end
for (shape, base) in (("16x64b", 1), ("32x32b", 1), ("16x128b", 2), ("16x256b", 4)),
    c in (1, 2, 4, 8, 16, 32, 64, 128)
    n = base * c
    n > 128 && continue
    push!(PROBES, ("llvm.nvvm.tcgen05.ld.$shape.x$c", (p6, Val{false}),
                   "sm_100a", "+ptx86",
                   Regex("tcgen05\\.ld\\.sync\\.aligned\\.$shape\\.x$c\\.b32")))
    data = n == 1 ? UInt32 : NTuple{n, VecElement{UInt32}}
    push!(PROBES, ("llvm.nvvm.tcgen05.st.$shape.x$c", (p6, data, Val{false}),
                   "sm_100a", "+ptx86",
                   Regex("tcgen05\\.st\\.sync\\.aligned\\.$shape\\.x$c\\.b32")))
end
for (kind, kv) in (("f16", 0), ("tf32", 1), ("f8f6f4", 2), ("i8", 3))
    push!(PROBES, ("llvm.nvvm.tcgen05.mma.shared",
                   (p6, UInt64, UInt64, UInt32, Bool, Val{kv}, Val{1}, Val{0}),
                   "sm_100a", "+ptx88",
                   Regex("tcgen05\\.mma\\.cta_group::1\\.kind::$kind\\.collector::a::discard")))
end
push!(PROBES, ("llvm.nvvm.tcgen05.mma.shared",
               (p6, UInt64, UInt64, UInt32, Bool, Val{0}, Val{2}, Val{0}),
               "sm_100a", "+ptx88",
               r"tcgen05\.mma\.cta_group::2\.kind::f16\.collector::a::discard"))

# mma.sync (wrappers/mma.jl, mma_scaled.jl) — a GENERATED family, not
# hand-literal, so the src/ literal-scan can't see it. One probe per
# structural class instead (the dtype cross-product within a class shares
# the i32/v2f16 convention and fragment counts); the full name list is
# guarded by the completeness assertion below. Expected spellings are the
# ISel order (kind/scale_vec after row.col).
const V2H = NTuple{2, VecElement{Float16}}
append!(PROBES, [
    # bf16 / tf32 inputs → i32 A/B (classic, Hopper floor)
    ("llvm.nvvm.mma.m16n8k16.row.col.bf16",
        (UInt32,UInt32,UInt32,UInt32,UInt32,UInt32,Float32,Float32,Float32,Float32),
        "sm_90a", "+ptx80", r"mma\.sync\.aligned\.m16n8k16\.row\.col\.f32\.bf16\.bf16\.f32"),
    ("llvm.nvvm.mma.m16n8k8.row.col.tf32",
        (UInt32,UInt32,UInt32,UInt32,UInt32,UInt32,Float32,Float32,Float32,Float32),
        "sm_90a", "+ptx80", r"mma\.sync\.aligned\.m16n8k8\.row\.col\.f32\.tf32\.tf32\.f32"),
    # pure-f16 inputs → v2f16 A/B; f32 vs f16 accumulator
    ("llvm.nvvm.mma.m16n8k16.row.col.f32.f32",
        (V2H,V2H,V2H,V2H,V2H,V2H,Float32,Float32,Float32,Float32),
        "sm_90a", "+ptx80", r"mma\.sync\.aligned\.m16n8k16\.row\.col\.f32\.f16\.f16\.f32"),
    ("llvm.nvvm.mma.m16n8k16.row.col.f16.f16",
        (V2H,V2H,V2H,V2H,V2H,V2H,V2H,V2H),
        "sm_90a", "+ptx80", r"mma\.sync\.aligned\.m16n8k16\.row\.col\.f16\.f16\.f16\.f16"),
    # fp8 e4m3 → i32 A/B; f32 and v2f16(C) accumulators (consumer-Blackwell)
    ("llvm.nvvm.mma.m16n8k16.row.col.f32.e4m3.e4m3.f32",
        (UInt32,UInt32,UInt32,Float32,Float32,Float32,Float32),
        "sm_121a", "+ptx88", r"mma\.sync\.aligned\.m16n8k16\.row\.col\.f32\.e4m3\.e4m3\.f32"),
    ("llvm.nvvm.mma.m16n8k16.row.col.f16.e4m3.e4m3.f16",
        (UInt32,UInt32,UInt32,V2H,V2H),
        "sm_121a", "+ptx88", r"mma\.sync\.aligned\.m16n8k16\.row\.col\.f16\.e4m3\.e4m3\.f16"),
    # kind::f8f6f4 mixed → i32 A/B (the GB10 sub-byte FP path); ISel order
    ("llvm.nvvm.mma.m16n8k32.row.col.kind.f8f6f4.f32.e2m1.e3m2.f32",
        (UInt32,UInt32,UInt32,UInt32,UInt32,UInt32,Float32,Float32,Float32,Float32),
        "sm_121a", "+ptx88", r"mma\.sync\.aligned\.m16n8k32\.row\.col\.kind::f8f6f4\.f32\.e2m1\.e3m2\.f32"),
    # block-scaled — one per kind
    ("llvm.nvvm.mma.block.scale.m16n8k32.row.col.mxf8f6f4.scale.1x.f32.e4m3.e4m3.f32.ue8m0",
        (UInt32,UInt32,UInt32,UInt32,UInt32,UInt32,Float32,Float32,Float32,Float32,
         UInt32,UInt16,UInt16,UInt32,UInt16,UInt16),
        "sm_121a", "+ptx88",
        r"mma\.sync\.aligned\.m16n8k32\.row\.col\.kind::mxf8f6f4\.block_scale\.scale_vec::1X\.f32\.e4m3\.e4m3\.f32\.ue8m0"),
    ("llvm.nvvm.mma.block.scale.m16n8k64.row.col.mxf4.scale.2x.f32.e2m1.e2m1.f32.ue8m0",
        (UInt32,UInt32,UInt32,UInt32,UInt32,UInt32,Float32,Float32,Float32,Float32,
         UInt32,UInt16,UInt16,UInt32,UInt16,UInt16),
        "sm_121a", "+ptx88",
        r"mma\.sync\.aligned\.m16n8k64\.row\.col\.kind::mxf4\.block_scale\.scale_vec::2X\.f32\.e2m1\.e2m1\.f32\.ue8m0"),
    ("llvm.nvvm.mma.block.scale.m16n8k64.row.col.mxf4nvf4.scale.4x.f32.e2m1.e2m1.f32.ue4m3",
        (UInt32,UInt32,UInt32,UInt32,UInt32,UInt32,Float32,Float32,Float32,Float32,
         UInt32,UInt16,UInt16,UInt32,UInt16,UInt16),
        "sm_121a", "+ptx88",
        r"mma\.sync\.aligned\.m16n8k64\.row\.col\.kind::mxf4nvf4\.block_scale\.scale_vec::4X\.f32\.e2m1\.e2m1\.f32\.ue4m3"),

    # proxy/init fences (wrappers/fence.jl)
    ("llvm.nvvm.fence.proxy.async", (), "sm_90a", "+ptx80",
        r"fence\.proxy\.async;"),
    ("llvm.nvvm.fence.proxy.async.shared_cta", (), "sm_90a", "+ptx80",
        r"fence\.proxy\.async\.shared::cta;"),
    ("llvm.nvvm.fence.mbarrier_init.release.cluster", (), "sm_90a", "+ptx80",
        r"fence\.mbarrier_init\.release\.cluster;"),
])

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

# The mma family is generated (not hand-literal), so the scan above can't
# see it. Its standing guarantee instead: every name the generators route
# to tier 2 must exist in the registry — re-checking the generators'
# load-time contract here means a name-table churn at an LLVM bump surfaces
# as a red test naming the family, not a silent miss. Per-class selection
# is covered by the mma probes above.
@testset "mma generated-family names exist in the registry" begin
    names = vcat(PTX.MMA_INTRINSIC_NAMES, PTX.MMA_SCALED_INTRINSIC_NAMES)
    @test length(PTX.MMA_INTRINSIC_NAMES) == 66    # dense tier-2 forms
    @test length(PTX.MMA_SCALED_INTRINSIC_NAMES) == 28
    @test all(NVVM.isintrinsic, names)
    # every per-class mma probe is one of the names the family stands on
    mma_probes = filter(p -> startswith(p[1], "llvm.nvvm.mma."), PROBES)
    @test !isempty(mma_probes)
    @test all(p -> p[1] in names, mma_probes)
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

# --- Layer 4: attribute-extraction faithfulness ------------------------------
#
# The three layers above cover names and the *instructions* the used surface
# selects. They say nothing about the per-intrinsic attribute tuple — and that
# tuple can't be checked the way names are. Two facts force a different test:
#
#   - The artifact llc IGNORES the attributes we attach. Verified directly:
#     declaring `read.ptx.sreg.tid.x` as `memory(write)` (a lie that should
#     suppress CSE) still lets llc fold two reads into one — it re-derives
#     attributes from its own compiled-in table. So no llc probe can validate
#     an attribute; a trial-compile matrix would be blind to this dimension.
#   - The attributes are therefore load-bearing ONLY for the in-process LLVM
#     (Julia's, which knows nothing of these names and is constrained solely by
#     what we attach). A wrong-in-the-permissive-direction attribute — claiming
#     `memory(none)`/`speculatable`, or dropping `convergent` — is exactly the
#     miscompile the convergence spike reproduced (CONCERNS.md).
#
# The realistic rot vector is an extraction-map regression in gen/ (PROPS / the
# jq filter) on a future JLL re-gen, silently changing a tuple while names stay
# clean. This pins a representative per attribute archetype so that turns red.
# Each anchor's expected tuple was checked against the 22.1.7 IntrinsicsNVVM.td
# source where marked [td]; the rest pin the current extracted value (still a
# regression tripwire, just not independently source-verified here).
const ATTR_ANCHORS = Pair{String, Tuple{Vararg{Symbol}}}[
    # convergent — the load-bearing one; absence = permission to duplicate
    "llvm.nvvm.bar.warp.sync"                  => (:convergent, :nocallback),                  # [td]
    "llvm.nvvm.barrier.cluster.arrive"         => (:convergent, :nocallback),                  # [td]
    "llvm.nvvm.barrier.cluster.arrive.aligned" => (:convergent, :nocallback),                  # [td]
    "llvm.nvvm.shfl.sync.idx.i32"              => (:inaccessiblememonly, :convergent, :nocallback),
    "llvm.nvvm.activemask"                     => (:inaccessiblememonly, :convergent, :nocallback, :sideeffects),
    # memory-effect archetypes — wrong-permissive here lets the optimizer
    # hoist/CSE/DCE a call that actually touches memory
    "llvm.nvvm.read.ptx.sreg.tid.x"            => (:nomem, :speculatable),                      # [td] NVVMPureIntrinsic
    "llvm.nvvm.add.rn.f"                        => (:nomem, :speculatable, :commutative),        # [td]
    "llvm.nvvm.atomic.add.gen.f.cta"           => (:argmemonly, :nocallback),                   # [td]
    # a fence proxy — deliberately NOT convergent (idempotent ordering op);
    # pins the distinction the fence migration turned on (CONCERNS.md)
    "llvm.nvvm.fence.proxy.async"              => (:nocallback,),                               # [td]
]

@testset "registry attribute extraction is faithful" begin
    for (name, expected) in ATTR_ANCHORS
        @test NVVM.intrinsic(name).props == expected
    end
end

# --- Layer 5: type-token coverage beyond the wrapper surface -----------------
#
# The selection probes compile only intrinsics the wrappers use, so a bug in a
# type-token → IR mapping (NVVM.llvmtype / the generator's VTS set) is caught
# only for tokens that surface there. Auditing the table, the wide-vector
# tokens (v16/32/64/128i32) are already exercised by the tcgen05 ld/st probes;
# the ONLY tokens unique to never-used intrinsics are `i128` (4
# clusterlaunchcontrol.query_cancel.* entries) and `Metadata` (1 legacy
# texsurf.handle). Compile-pin the i128 token here so its mapping isn't taken
# on faith. `Metadata` is left uncompiled — it has no normal SSA callsite
# (texsurf.handle takes a metadata operand) and no wrapper will ever use it;
# this is the registry's sole known-uncompiled token, logged rather than
# silently skipped.
@testset "type tokens outside the wrapper surface lower (i128)" begin
    exe = llc().exec[1]
    name = "llvm.nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.x"
    s = synthesize(name, (UInt128,))   # i128 param
    ll = "target triple = \"nvptx64-nvidia-cuda\"\n" * s.ir
    out = IOBuffer(); err = IOBuffer()
    ok = success(pipeline(`$exe -mcpu=sm_100a -mattr=+ptx86 -o -`;
                          stdin = IOBuffer(ll), stdout = out, stderr = err))
    ptx = String(take!(out))
    ok || @info "i128 token probe failed" llc_error=String(take!(err)) ptx
    @test ok
    @test occursin("clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128", ptx)

    @info "registry token `Metadata` is left uncompiled (no SSA callsite): " *
          "only llvm.nvvm.texsurf.handle uses it; no wrapper does"
end
