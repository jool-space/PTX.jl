# TMA shared-memory swizzle placement probe — documents and regression-
# guards what the hardware actually does with `:B32` / `:B64` / `:B128`
# tensor-map swizzle modes.
#
# Why this exists: the PTX ISA delegates tensor-map swizzle semantics to
# the driver/hardware ("Refer to the CUDA programming guide"), yet
# hand-written kernels in this corpus depend on the exact byte placement
# — gemm_mixed_dtype.jl stores a hand-swizzled B32 tile that wgmma then
# reads through a B32 descriptor, and gemm_gather_scatter.jl reproduces
# a full-tile B32 TMA load flit-by-flit. Those files bake in rules that
# were probed empirically on an H100; this test IS that probe, kept
# runnable, so a driver/hardware change (or a wrong transcription of the
# rule) fails loudly instead of silently corrupting wgmma operands.
#
# The rule, in cute's Swizzle<B,M,S> notation (M=4: 16-byte flits are
# the unit; S=3: the XOR key sits 3 bits above the flit field):
#
#   physical_addr = logical_addr ⊻ (key << 4),
#   key = (logical_addr >> 7) & ((1 << B) - 1)
#
#   :B32  = Swizzle<1,4,3> — bit    7   → bit  4      (256-B period)
#   :B64  = Swizzle<2,4,3> — bits  8:7  → bits 5:4    (512-B period)
#   :B128 = Swizzle<3,4,3> — bits  9:7  → bits 6:4   (1024-B period)
#
# Crucially the key is taken from the ABSOLUTE shared-memory address,
# not the offset within the tile: a B32 destination at 128 (mod 256)
# reads back with every row's flit pair swapped relative to a 256-B-
# aligned destination. The runtime testsets verify the formula at a
# 1024-B-aligned base for all three modes, then re-verify B32 at a
# +128 B destination to pin the absolute-address behavior. Kernels that
# hand-apply a swizzle must therefore align their tiles to the swizzle
# period (CuStaticSharedArray guarantees only 32 B — align in-kernel).
#
# Alignment contract (probed, not asserted — a faulting launch poisons
# the CUDA context and would take the test worker down with it): the
# TMA destination must be aligned to the swizzle SPAN (32/64/128 B).
# Sub-span alignment (e.g. a 1-row box landing at an odd 32-B offset
# under B32) faults with CUDA_ERROR_MISALIGNED_ADDRESS — that is why
# gemm_gather_scatter.jl gathers with plain loads instead of per-row
# swizzled TMA boxes.
#
# Each byte of the source tile encodes its own logical position:
#   byte(o) = (flit(o) << 2) | (o & 3),  flit(o) = o >> 4
# so any misplacement is visible in the dump, at flit granularity and
# below (the low 2 bits catch byte-level scrambling inside a flit,
# which must never happen — swizzling permutes whole 16-B flits).

using PTX: tensor_map_tile_2d, smem_addr_u32
using CUDACore
using Random

const SWP_ROWS = 8                       # one full swizzle atom of rows
const SWP_PAD  = 1024                    # in-kernel alignment headroom

# (mode, row bytes, tile bytes, XOR key width B)
const SWP_MODES = ((:B32, 32, 256, 1), (:B64, 64, 512, 2), (:B128, 128, 1024, 3))

swp_expected_byte(o::Int) = UInt8((((o >> 4) & 0x3f) << 2) | (o & 3))

# The Swizzle<B,4,3> placement function on absolute byte addresses.
swp_swizzle(addr::Integer, B::Int) =
    addr ⊻ (((addr >> 7) & ((1 << B) - 1)) << 4)

# TMA-load one tile at (1024-aligned base + dst_off), dump the raw SMEM
# bytes plus the destination's absolute SMEM address. TB is the tile
# size in bytes; the box geometry lives in the tensor map.
function _swp_dump_kernel!(
        out::CuDeviceVector{UInt8, 1},
        addr_out::CuDeviceVector{UInt32, 1},
        tma::PTX.TMADescriptorPtr,
        ::Val{TB},
        dst_off::UInt32) where {TB}

    smem = CuStaticSharedArray(UInt8, TB + SWP_PAD + 128)
    mbar = CuStaticSharedArray(UInt64, 1)
    base = pointer(smem)
    mb   = pointer(mbar)

    raw     = smem_addr_u32(base)
    aligned = (raw + UInt32(SWP_PAD - 1)) & ~UInt32(SWP_PAD - 1)
    dst     = base + Int(aligned - raw) + Int(dst_off)

    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mb, UInt32(1))
        ptx"fence.proxy.async.shared::cta"()
        ptx"mbarrier.arrive.expect_tx.shared.b64"(mb, UInt32(TB))
        ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
            dst, tma, Int32(0), Int32(0), mb)
        @inbounds addr_out[1] = aligned + dst_off
    end
    ptx"bar.sync"(Val(0))
    while !ptx"mbarrier.test_wait.parity.shared.b64"(mb, UInt32(0))
    end

    # Raw byte dump — no interpretation kernel-side.
    i = Int(tid)
    @inbounds while i < TB
        out[i + 1] = unsafe_load(dst, i + 1)
        i += 128
    end
    return nothing
end

# Host: pack the position-encoded tile as (K, ROWS) u16, K-fast, run the
# dump at `dst_off`, and return (bytes, absolute dst address).
function swp_run_probe(mode::Symbol, row_bytes::Int, tile_bytes::Int;
                       dst_off::UInt32 = UInt32(0))
    k_elems = row_bytes ÷ 2
    host = Array{UInt16}(undef, k_elems, SWP_ROWS)
    for r in 0:(SWP_ROWS - 1), k in 0:(k_elems - 1)
        o = r * row_bytes + 2k
        host[k + 1, r + 1] = UInt16(swp_expected_byte(o)) |
                             (UInt16(swp_expected_byte(o + 1)) << 8)
    end
    dev = CuArray(host)
    tmap = tensor_map_tile_2d(:u16, pointer(dev),
        SWP_ROWS, k_elems, SWP_ROWS, k_elems; swizzle = mode)
    d = upload_tma_descriptor(tmap)

    out  = CUDACore.zeros(UInt8, tile_bytes)
    addr = CUDACore.zeros(UInt32, 1)
    @cuda threads = 128 _swp_dump_kernel!(out, addr, d.ptr,
                                          Val(tile_bytes), dst_off)
    CUDACore.synchronize()
    return Array(out), Array(addr)[1]
end

# Check the dump against the absolute-address Swizzle<B,4,3> formula.
function swp_formula_holds(dump::Vector{UInt8}, dst_addr::UInt32, B::Int)
    ok = true
    for o in 0:(length(dump) - 1)
        a_log  = Int(dst_addr) + o
        a_phys = swp_swizzle(a_log, B)
        p = a_phys - Int(dst_addr)
        # The XOR only touches bits 6:4, so a span-aligned destination
        # keeps every flit inside the tile.
        ok &= 0 <= p < length(dump) && dump[p + 1] == swp_expected_byte(o)
    end
    return ok
end

# ── Cross-arch ptxas validation ────────────────────────────────────────

@testset "swizzle probe kernel compiles at sm_90a" begin
    types = Tuple{CuDeviceVector{UInt8, 1}, CuDeviceVector{UInt32, 1},
                  PTX.TMADescriptorPtr, Val{256}, UInt32}
    @test ptxas_compiles(_swp_dump_kernel!, types;
                         cap = v"9.0", feature_set = :arch)
    ptx = emit_ptx(_swp_dump_kernel!, types; cap = v"9.0", feature_set = :arch)
    @test occursin("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes", ptx)
    @test occursin("mbarrier.arrive.expect_tx.shared.b64", ptx)
end

# ── Runtime — Hopper hardware ──────────────────────────────────────────

if v"9.0" <= DEV_CAP < v"10.0"
    @testset "TMA swizzle placement = Swizzle<B,4,3> ($(mode))" for
            (mode, row_bytes, tile_bytes, B) in SWP_MODES
        dump, addr = swp_run_probe(mode, row_bytes, tile_bytes)
        # The in-kernel alignment must have landed on the 1024-B boundary,
        # otherwise the absolute-address math below tests nothing.
        @test addr % SWP_PAD == 0
        @test swp_formula_holds(dump, addr, B)
    end

    @testset "B32 swizzle keys on ABSOLUTE address (dst at +128)" begin
        # Same tile, same tensor map, destination 128 B past the aligned
        # base — still span-aligned (32 B), but the key bit (abs bit 7)
        # is now set for the FIRST half of the tile instead of the
        # second. The absolute-address formula must still hold; the
        # tile-relative version of the formula must NOT.
        dump, addr = swp_run_probe(:B32, 32, 256; dst_off = UInt32(128))
        @test addr % SWP_PAD == 128
        @test swp_formula_holds(dump, addr, 1)
        rel_ok = all(0:255) do o
            p = swp_swizzle(o, 1)   # relative-offset (wrong) formula
            dump[p + 1] == swp_expected_byte(o)
        end
        @test !rel_ok
    end
end
