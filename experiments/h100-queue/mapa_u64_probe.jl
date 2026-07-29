# Issue #52 (MAPA-U64-EXP): validate the shared-cluster u64 mapa carrier.
#
# The audit concern: `optype"mapa.shared::cluster.u64"` passes an AS(3)
# LLVMPtr under the 64-bit `l` constraint, and it was never established
# whether that selects a legal 64-bit operand, whether ptxas accepts the
# result, or whether the mapped address is semantically usable.
#
# Three legs, each printed with a `[leg]` banner:
#   host    — optimized LLVM + emitted PTX for the wrapper path; asserts the
#             mapa.u64 operand renders as a 64-bit register pair (%rd).
#             Runs anywhere.
#   ptxas   — assembles both probe kernels at sm_90a. Needs the compiler
#             artifact (no GPU).
#   runtime — 2-CTA cluster semantic probe on CC 9.x. Each CTA publishes a
#             marker in its own SMEM slot; after a cluster barrier, thread 0
#             maps its slot address into the peer CTA with BOTH carriers and
#             reads the peer marker back via ld.shared::cluster. Raw carrier
#             bit patterns are recorded for width inspection.
#
# Run on the box:   julia --project=test experiments/h100-queue/mapa_u64_probe.jl
#
# Verdict criteria (paste the output into issue #52):
#   - host leg shows `mapa.shared::cluster.u64 %rdX, %rdY, %rZ` (64-bit regs)
#   - ptxas accepts both kernels at sm_90a
#   - runtime: v32 and v64 both equal the PEER's marker per rank, and the
#     u64 carrier's low 32 bits equal the u32 carrier (high bits reported).

using Test, Random
using PTX
using CUDACore, CUDATools
using LLVM.Interop: @asmcall

include(joinpath(dirname(dirname(@__DIR__)), "test", "setup.jl"))

const _AS_SHARED = CUDACore.AS.Shared

# Raw-bits variants of the two carriers: same asm as the src wrappers, but
# returning the integer bit pattern instead of an LLVMPtr, so the kernel can
# record the mapped addresses for host-side inspection.
@inline _mapa_u32_bits(p::Core.LLVMPtr{UInt32, _AS_SHARED}, rank::UInt32) =
    @asmcall("mapa.shared::cluster.u32 \$0, \$1, \$2;", "=r,r,r", true,
             UInt32, Tuple{Core.LLVMPtr{UInt32, _AS_SHARED}, UInt32}, p, rank)
@inline _mapa_u64_bits(p::Core.LLVMPtr{UInt32, _AS_SHARED}, rank::UInt32) =
    @asmcall("mapa.shared::cluster.u64 \$0, \$1, \$2;", "=l,l,r", true,
             UInt64, Tuple{Core.LLVMPtr{UInt32, _AS_SHARED}, UInt32}, p, rank)

# Cluster-decode loads consuming the mapped pointers. No src wrapper exists
# for ld.shared::cluster yet — these are probe-local. The `l` (64-bit
# address) variant is itself part of the experiment.
@inline _ld_cluster_r(p::Core.LLVMPtr{UInt32, _AS_SHARED}) =
    @asmcall("ld.shared::cluster.u32 \$0, [\$1];", "=r,r", true,
             UInt32, Tuple{Core.LLVMPtr{UInt32, _AS_SHARED}}, p)
@inline _ld_cluster_l(p::Core.LLVMPtr{UInt32, _AS_SHARED}) =
    @asmcall("ld.shared::cluster.u32 \$0, [\$1];", "=r,l", true,
             UInt32, Tuple{Core.LLVMPtr{UInt32, _AS_SHARED}}, p)

# Marker: 0xA0A0000r for CTA rank r.
_marker(rank::UInt32) = UInt32(0xA0A0_0000) | rank

# Kernel A — documented u32 carrier baseline + raw bit patterns of BOTH
# carriers (u64 used only as data here, never as an address).
function _mapa_carrier_bits_kernel!(vals::CuDeviceVector{UInt32, 1},
                                    addrs::CuDeviceVector{UInt64, 1})
    slot = CuStaticSharedArray(UInt32, 1)
    p = pointer(slot)
    tid = ptx"mov.u32"(sreg"tid.x")
    rank = ptx"mov.u32"(sreg"cluster_ctarank")

    if tid == UInt32(0)
        @inbounds slot[1] = _marker(rank)
    end
    ptx"barrier.cluster.arrive"()
    ptx"barrier.cluster.wait"()

    if tid == UInt32(0)
        peer = rank ⊻ UInt32(1)
        p32 = ptx"mapa.shared::cluster.u32"(p, peer)
        v32 = _ld_cluster_r(p32)
        base = Int(rank)
        @inbounds vals[base + 1] = v32
        @inbounds addrs[base * 3 + 1] = UInt64(_mapa_u32_bits(p, peer))
        @inbounds addrs[base * 3 + 2] = _mapa_u64_bits(p, peer)
        @inbounds addrs[base * 3 + 3] = _mapa_u64_bits(p, rank)  # identity map
    end
    return nothing
end

# Kernel B — the disputed path: wrapper-produced u64 carrier consumed as a
# 64-bit ld.shared::cluster address.
function _mapa_u64_read_kernel!(vals::CuDeviceVector{UInt32, 1})
    slot = CuStaticSharedArray(UInt32, 1)
    p = pointer(slot)
    tid = ptx"mov.u32"(sreg"tid.x")
    rank = ptx"mov.u32"(sreg"cluster_ctarank")

    if tid == UInt32(0)
        @inbounds slot[1] = _marker(rank)
    end
    ptx"barrier.cluster.arrive"()
    ptx"barrier.cluster.wait"()

    if tid == UInt32(0)
        peer = rank ⊻ UInt32(1)
        p64 = ptx"mapa.shared::cluster.u64"(p, peer)
        v64 = _ld_cluster_l(p64)
        @inbounds vals[Int(rank) + 1] = v64
    end
    return nothing
end

const _A_TYPES = Tuple{CuDeviceVector{UInt32, 1}, CuDeviceVector{UInt64, 1}}
const _B_TYPES = Tuple{CuDeviceVector{UInt32, 1}}

println("=" ^ 72)
println("[host] mapa lines in emitted PTX (sm_90a)")
for (label, f, tt) in (("kernel A", _mapa_carrier_bits_kernel!, _A_TYPES),
                       ("kernel B", _mapa_u64_read_kernel!, _B_TYPES))
    ptx = emit_ptx(f, tt; cap = v"9.0", feature_set = :arch)
    for line in split(ptx, '\n')
        if occursin("mapa", line) || occursin("shared::cluster", line)
            println("  ", label, ": ", strip(line))
        end
    end
end
begin
    ptx = emit_ptx(_mapa_u64_read_kernel!, _B_TYPES; cap = v"9.0", feature_set = :arch)
    m = match(r"mapa\.shared::cluster\.u64\s+(%\w+),\s*(%\w+)", ptx)
    ok = m !== nothing && startswith(m.captures[1], "%rd") && startswith(m.captures[2], "%rd")
    println("[host] u64 operands are 64-bit (%rd): ", ok ? "YES" : "NO — got $(m)")
end

println("=" ^ 72)
println("[ptxas] sm_90a assembly")
for (label, f, tt) in (("kernel A (u32 carrier + raw bits)", _mapa_carrier_bits_kernel!, _A_TYPES),
                       ("kernel B (u64 carrier load)", _mapa_u64_read_kernel!, _B_TYPES))
    ok, err = try
        ptxas_compiles(f, tt; cap = v"9.0", feature_set = :arch), ""
    catch e
        false, sprint(showerror, e)
    end
    println("  ", label, ": ", ok ? "ACCEPTED" : "REJECTED")
    ok || println(err)
end

println("=" ^ 72)
cap = try
    CUDACore.functional() ? CUDACore.capability(CUDACore.device()) : nothing
catch
    nothing
end
if cap === nothing || !(v"9.0" <= cap < v"11.0")
    println("[runtime] SKIPPED — no functional CC 9.x/10.x device (found: $cap)")
else
    println("[runtime] device = ", CUDACore.name(CUDACore.device()), " CC ", cap)
    vals_a = CUDACore.zeros(UInt32, 2)
    addrs  = CUDACore.zeros(UInt64, 6)
    @cuda blocks = (2, 1, 1) threads = 128 clustersize = (2, 1, 1) _mapa_carrier_bits_kernel!(vals_a, addrs)
    CUDACore.synchronize()
    va, ad = Array(vals_a), Array(addrs)
    for r in 0:1
        expected = _marker(UInt32(r ⊻ 1))
        got = va[r + 1]
        println("  rank $r: v32 = 0x", string(got, base = 16, pad = 8),
                got == expected ? "  == peer marker OK" : "  MISMATCH (expected 0x$(string(expected, base = 16, pad = 8)))")
        a32, a64, a64self = ad[r * 3 + 1], ad[r * 3 + 2], ad[r * 3 + 3]
        println("    a32(peer)      = 0x", string(a32, base = 16, pad = 16))
        println("    a64(peer)      = 0x", string(a64, base = 16, pad = 16),
                (a64 & 0xffffffff) == a32 ? "  low32 == a32" : "  LOW32 DIFFERS FROM a32")
        println("    a64(identity)  = 0x", string(a64self, base = 16, pad = 16))
    end

    vals_b = CUDACore.zeros(UInt32, 2)
    b_ok = try
        @cuda blocks = (2, 1, 1) threads = 128 clustersize = (2, 1, 1) _mapa_u64_read_kernel!(vals_b)
        CUDACore.synchronize()
        true
    catch e
        println("  kernel B launch FAILED: ", sprint(showerror, e))
        false
    end
    if b_ok
        vb = Array(vals_b)
        for r in 0:1
            expected = _marker(UInt32(r ⊻ 1))
            println("  rank $r: v64 = 0x", string(vb[r + 1], base = 16, pad = 8),
                    vb[r + 1] == expected ? "  == peer marker OK" : "  MISMATCH (expected 0x$(string(expected, base = 16, pad = 8)))")
        end
    end
end
println("=" ^ 72)
println("done — paste this output into issue #52")
