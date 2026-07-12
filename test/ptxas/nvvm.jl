# nvvm"" tier-2 emission through the real pipeline: Julia → llvmcall splice →
# in-process LLVM 18 (which has never heard of these intrinsics — our IR and
# attribute spellings must parse there) → bitcode → external llc 22 (which
# selects them) → ptxas. Compile-only; semantics run in test/gpu/nvvm.jl.
#
# This is the seed of the registry conformance harness: each
# case asserts the *expected instruction*, not mere acceptance — llc happily
# remangles or upgrades sloppy callsites, so acceptance proves little.

using PTX.NVVM


# --- shfl + vote at sm_75 ---------------------------------------------------

function _nvvm_shfl_vote!(out::CuDeviceVector{UInt32, 1})
    tid = ptx"mov.u32"(sreg"tid.x")
    v = nvvm"shfl.sync.idx.i32"(0xffffffff, tid, UInt32(5), UInt32(0x1f))
    vp, ok = nvvm"shfl.sync.idx.i32p"(0xffffffff, tid, UInt32(5), UInt32(0x1f))
    all31 = nvvm"vote.all.sync"(0xffffffff, tid < UInt32(31))
    @inbounds out[tid + 1] = v + vp + UInt32(ok) + UInt32(all31)
    return nothing
end

@testset "nvvm shfl/vote at sm_75" begin
    types = Tuple{CuDeviceVector{UInt32, 1}}
    @test ptxas_compiles(_nvvm_shfl_vote!, types; cap = v"7.5")
    ptx = emit_ptx(_nvvm_shfl_vote!, types; cap = v"7.5")
    @test occursin("shfl.sync.idx.b32", ptx)
    @test occursin("vote.sync.all.pred", ptx)
end


# --- the convergence regression, compile-only ------------------------------
#
# The shape from spikes/convergence.jl (removed in ccdfb8a; view with
# `git show ccdfb8a~1:spikes/convergence.jl`): identical convergent calls
# leading both arms of a divergent branch. With the registry's `convergent`
# on the declaration, both call sites must survive -O2 into the PTX; the
# hoisted single-site version is the miscompile the spike demonstrated on
# hardware.

function _nvvm_divergent_mask!(out::CuDeviceVector{UInt32, 1})
    i = ptx"mov.u32"(sreg"tid.x")
    if i < UInt32(16)
        @inbounds out[i + 1] = nvvm"activemask"()
    else
        @inbounds out[i + 1] = nvvm"activemask"() ⊻ UInt32(1)
    end
    return nothing
end

@testset "convergent survives -O2 (two activemask sites)" begin
    types = Tuple{CuDeviceVector{UInt32, 1}}
    ptx = emit_ptx(_nvvm_divergent_mask!, types; cap = v"7.5")
    @test count("activemask.b32", ptx) == 2
end


# --- mbarrier at sm_90 ------------------------------------------------------

function _nvvm_mbarrier!(out::CuDeviceVector{UInt64, 1}, n::UInt32)
    bar = CuStaticSharedArray(Int64, 1)
    state = nvvm"mbarrier.arrive.expect.tx.scope.cta.space.cta"(pointer(bar), n)
    @inbounds out[1] = state
    return nothing
end

@testset "nvvm mbarrier.arrive.expect_tx at sm_90" begin
    types = Tuple{CuDeviceVector{UInt64, 1}, UInt32}
    @test ptxas_compiles(_nvvm_mbarrier!, types; cap = v"9.0")
    ptx = emit_ptx(_nvvm_mbarrier!, types; cap = v"9.0")
    @test occursin("mbarrier.arrive.expect_tx", ptx)
end


# --- ldmatrix at sm_75 (mangled overload) -----------------------------------

function _nvvm_ldmatrix!(out::CuDeviceMatrix{UInt32, 1})
    smem = CuStaticSharedArray(UInt16, 256)
    t = ptx"mov.u32"(sreg"tid.x")
    r = nvvm"ldmatrix.sync.aligned.m8n8.x4.b16"(pointer(smem) + t * UInt32(16))
    @inbounds for m in 1:4
        out[t + 1, m] = r[m]
    end
    return nothing
end

@testset "nvvm ldmatrix (.p3 mangle) at sm_75" begin
    types = Tuple{CuDeviceMatrix{UInt32, 1}}
    @test ptxas_compiles(_nvvm_ldmatrix!, types; cap = v"7.5")
    ptx = emit_ptx(_nvvm_ldmatrix!, types; cap = v"7.5")
    @test occursin("ldmatrix.sync.aligned.m8n8.x4.shared.b16", ptx)
end


# --- tcgen05.mma with immediate operands at sm_100a -------------------------
#
# Compile-only is exactly what this tier is for: the dev box (sm_121a)
# can't launch sm_100a code, and the Val-spliced immediates (`kind`,
# `cta_group`, `collector`) surface in the selected instruction text, which
# makes the splice visible in the assertion.

function _nvvm_tcgen05_mma!(tmem::Core.LLVMPtr{UInt32, 6},
                            adesc::UInt64, bdesc::UInt64, idesc::UInt32,
                            enable::Bool)
    nvvm"tcgen05.mma.shared"(tmem, adesc, bdesc, idesc, enable,
                             Val(2), Val(1), Val(0))
    return nothing
end

@testset "nvvm tcgen05.mma.shared (immargs) at sm_100a" begin
    types = Tuple{Core.LLVMPtr{UInt32, 6}, UInt64, UInt64, UInt32, Bool}
    @test ptxas_compiles(_nvvm_tcgen05_mma!, types;
                         cap = v"10.0", feature_set = :arch)
    ptx = emit_ptx(_nvvm_tcgen05_mma!, types;
                   cap = v"10.0", feature_set = :arch)
    @test occursin("tcgen05.mma.cta_group::1.kind::f8f6f4", ptx)
end

# --- sreg !range metadata reaches the optimizer ------------------------------
# The registry's position-0 `ranges` entries render as call-site `!range`
# metadata (emit.jl). Behavioral proof, not text-matching: tid.x ∈ [0,1024)
# makes `tid & 1023` an identity — the AND must fold out of the PTX. The
# control masks with 511, which the range does NOT subsume — that AND must
# survive (guards against the op vanishing for an unrelated reason).

function _nvvm_sreg_range_fold!(out::Core.LLVMPtr{UInt32, 1})
    tid = ptx"mov.u32"(sreg"tid.x")
    ptx"st.global.b32"(out, tid & UInt32(1023))
    return nothing
end

function _nvvm_sreg_range_ctrl!(out::Core.LLVMPtr{UInt32, 1})
    tid = ptx"mov.u32"(sreg"tid.x")
    ptx"st.global.b32"(out, tid & UInt32(511))
    return nothing
end

@testset "sreg !range folds subsumed masks" begin
    types = Tuple{Core.LLVMPtr{UInt32, 1}}
    ptx = emit_ptx(_nvvm_sreg_range_fold!, types; cap = v"7.5")
    @test !occursin("and.b32", ptx)
    ptx = emit_ptx(_nvvm_sreg_range_ctrl!, types; cap = v"7.5")
    @test occursin("and.b32", ptx)
end
