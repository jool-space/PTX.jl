# Consumer-Blackwell (sm_121a) wrapper validation. Same compile-through-ptxas
# pattern as ptxas/blackwell.jl — every kernel ptxas-validates without ever
# being launched. Separate file from blackwell.jl because that one targets
# sm_100a (datacenter) and the feature sets diverge — sm_100a has tcgen05,
# sm_121a does not.
#
# Coverage today: one chain-default `mma.sync.aligned.kind::f8f6f4.*` op as
# the foothold for the Phase-2 sub-byte FP mma generator. Block-scaled
# `kind::mxf4*` / `kind::mxf4nvf4` / `kind::mxf8f6f4` variants land alongside
# the generator (extra scale operand pair — separate frag-count table).


# --- mma.sync.aligned.kind::f8f6f4 (sub-byte FP tensor-core, sm_100a+) ----
#
# Same operand shape as the Ada e4m3.e4m3.f32 m16n8k32 mma — the
# `kind::f8f6f4` segment selects the unified sub-byte FP accelerator path.
# ptxas rejects this op without the `a` suffix even on Blackwell baseline.

function _sm121a_mma_kind_f8f6f4!(out::CuDeviceVector{Float32, 1},
                                   a1::UInt32, a2::UInt32, a3::UInt32, a4::UInt32,
                                   b1::UInt32, b2::UInt32)
    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        a = (a1, a2, a3, a4)
        b = (b1, b2)
        c = (0f0, 0f0, 0f0, 0f0)
        d = ptx"mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32"(a, b, c)
        @inbounds out[1] = d[1]
        @inbounds out[2] = d[2]
        @inbounds out[3] = d[3]
        @inbounds out[4] = d[4]
    end
    return nothing
end

@testset "mma.sync.aligned.kind::f8f6f4 m16n8k32 e4m3 at sm_121a" begin
    types = Tuple{CuDeviceVector{Float32, 1},
                  UInt32, UInt32, UInt32, UInt32, UInt32, UInt32}
    @test ptxas_compiles(_sm121a_mma_kind_f8f6f4!, types;
                         cap = v"12.1", feature_set = :arch)
    ptx = emit_ptx(_sm121a_mma_kind_f8f6f4!, types;
                   cap = v"12.1", feature_set = :arch)
    @test occursin(".target sm_121a", ptx)
    # tier-2: ISel renders the kind qualifier after row.col (shape-before-
    # kind), where the asm tier put it first. ptxas accepts both. The
    # m16n8k16 kind form below has no intrinsic and keeps the asm order.
    @test occursin("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32", ptx)
end

# Mixed sub-byte FP: kind::f8f6f4 admits any A,B ∈ {e4m3,e5m2,e3m2,e2m3,e2m1};
# all five share the 8-bit container so register counts are uniform. One
# canary combo (e2m1 × e3m2) confirms the mixed-type method dispatches.

function _sm121a_mma_kind_f8f6f4_mixed!(out::CuDeviceVector{Float32, 1},
                                         a1::UInt32, a2::UInt32, a3::UInt32, a4::UInt32,
                                         b1::UInt32, b2::UInt32)
    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        a = (a1, a2, a3, a4)
        b = (b1, b2)
        c = (0f0, 0f0, 0f0, 0f0)
        d = ptx"mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e3m2.f32"(a, b, c)
        @inbounds out[1] = d[1]
        @inbounds out[2] = d[2]
        @inbounds out[3] = d[3]
        @inbounds out[4] = d[4]
    end
    return nothing
end

@testset "mma.sync.aligned.kind::f8f6f4 m16n8k32 mixed e2m1×e3m2 at sm_121a" begin
    types = Tuple{CuDeviceVector{Float32, 1},
                  UInt32, UInt32, UInt32, UInt32, UInt32, UInt32}
    @test ptxas_compiles(_sm121a_mma_kind_f8f6f4_mixed!, types;
                         cap = v"12.1", feature_set = :arch)
    ptx = emit_ptx(_sm121a_mma_kind_f8f6f4_mixed!, types;
                   cap = v"12.1", feature_set = :arch)
    @test occursin("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e3m2.f32", ptx)
end

# --- mma.sync.aligned.kind::f8f6f4 m16n8k16 ------------------------------
#
# The smaller of the two valid kind::f8f6f4 shapes (PTX 9.2 §9.7.14.5
# line 1389). Per §9.7.14.5.9 register counts are A=2, B=1, D=C=4 for any
# 8-bit-container ab type — half what m16n8k32 needs.

function _sm121a_mma_kind_f8f6f4_k16!(out::CuDeviceVector{Float32, 1},
                                       a1::UInt32, a2::UInt32, b1::UInt32)
    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        a = (a1, a2)
        b = (b1,)
        c = (0f0, 0f0, 0f0, 0f0)
        d = ptx"mma.sync.aligned.kind::f8f6f4.m16n8k16.row.col.f32.e4m3.e4m3.f32"(a, b, c)
        @inbounds out[1] = d[1]
        @inbounds out[2] = d[2]
        @inbounds out[3] = d[3]
        @inbounds out[4] = d[4]
    end
    return nothing
end

@testset "mma.sync.aligned.kind::f8f6f4 m16n8k16 e4m3 at sm_121a" begin
    types = Tuple{CuDeviceVector{Float32, 1}, UInt32, UInt32, UInt32}
    @test ptxas_compiles(_sm121a_mma_kind_f8f6f4_k16!, types;
                         cap = v"12.1", feature_set = :arch)
    ptx = emit_ptx(_sm121a_mma_kind_f8f6f4_k16!, types;
                   cap = v"12.1", feature_set = :arch)
    @test occursin("mma.sync.aligned.kind::f8f6f4.m16n8k16.row.col.f32.e4m3.e4m3.f32", ptx)
end


# --- mma.sync.aligned.kind::mxf8f6f4 (block-scaled, m16n8k32) -----------
#
# Smallest representative of the block-scaled family. ue8m0 scale, scale_vec::1X.
# Per §9.7.14.3 Table 36, kind::mxf8f6f4 admits any A/B in the 5 sub-byte
# FP types and only ue8m0 scale.

function _sm121a_mma_kind_mxf8f6f4!(out::CuDeviceVector{Float32, 1},
                                     a1::UInt32, a2::UInt32, a3::UInt32, a4::UInt32,
                                     b1::UInt32, b2::UInt32,
                                     sa::UInt32, sb::UInt32)
    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        a = (a1, a2, a3, a4)
        b = (b1, b2)
        c = (0f0, 0f0, 0f0, 0f0)
        d = ptx"mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0"(
            a, b, c, sa, UInt16(0), UInt16(0), sb, UInt16(0), UInt16(0))
        @inbounds out[1] = d[1]
        @inbounds out[2] = d[2]
        @inbounds out[3] = d[3]
        @inbounds out[4] = d[4]
    end
    return nothing
end

@testset "mma.sync.aligned.kind::mxf8f6f4 m16n8k32 e4m3 at sm_121a" begin
    types = Tuple{CuDeviceVector{Float32, 1},
                  UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                  UInt32, UInt32}
    @test ptxas_compiles(_sm121a_mma_kind_mxf8f6f4!, types;
                         cap = v"12.1", feature_set = :arch)
    ptx = emit_ptx(_sm121a_mma_kind_mxf8f6f4!, types;
                   cap = v"12.1", feature_set = :arch)
    @test occursin(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0",
        ptx)
end


# --- mma.sync.aligned.kind::mxf4 (block-scaled, m16n8k64, e2m1) ---------
#
# Smallest e2m1-only path. ue8m0 scale, scale_vec::2X.

function _sm121a_mma_kind_mxf4!(out::CuDeviceVector{Float32, 1},
                                 a1::UInt32, a2::UInt32, a3::UInt32, a4::UInt32,
                                 b1::UInt32, b2::UInt32,
                                 sa::UInt32, sb::UInt32)
    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        a = (a1, a2, a3, a4)
        b = (b1, b2)
        c = (0f0, 0f0, 0f0, 0f0)
        d = ptx"mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0"(
            a, b, c, sa, UInt16(0), UInt16(0), sb, UInt16(0), UInt16(0))
        @inbounds out[1] = d[1]
        @inbounds out[2] = d[2]
        @inbounds out[3] = d[3]
        @inbounds out[4] = d[4]
    end
    return nothing
end

@testset "mma.sync.aligned.kind::mxf4 m16n8k64 e2m1 at sm_121a" begin
    types = Tuple{CuDeviceVector{Float32, 1},
                  UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                  UInt32, UInt32}
    @test ptxas_compiles(_sm121a_mma_kind_mxf4!, types;
                         cap = v"12.1", feature_set = :arch)
    ptx = emit_ptx(_sm121a_mma_kind_mxf4!, types;
                   cap = v"12.1", feature_set = :arch)
    @test occursin(
        "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0",
        ptx)
end


# --- mma.sync.aligned.kind::mxf4nvf4 (block-scaled, m16n8k64, e2m1, ue4m3) ---
#
# nvf4 variant: scale_vec::4X is mandatory with ue4m3 stype.

function _sm121a_mma_kind_mxf4nvf4!(out::CuDeviceVector{Float32, 1},
                                     a1::UInt32, a2::UInt32, a3::UInt32, a4::UInt32,
                                     b1::UInt32, b2::UInt32,
                                     sa::UInt32, sb::UInt32)
    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        a = (a1, a2, a3, a4)
        b = (b1, b2)
        c = (0f0, 0f0, 0f0, 0f0)
        d = ptx"mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3"(
            a, b, c, sa, UInt16(0), UInt16(0), sb, UInt16(0), UInt16(0))
        @inbounds out[1] = d[1]
        @inbounds out[2] = d[2]
        @inbounds out[3] = d[3]
        @inbounds out[4] = d[4]
    end
    return nothing
end

@testset "mma.sync.aligned.kind::mxf4nvf4 m16n8k64 e2m1 ue4m3 at sm_121a" begin
    types = Tuple{CuDeviceVector{Float32, 1},
                  UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                  UInt32, UInt32}
    @test ptxas_compiles(_sm121a_mma_kind_mxf4nvf4!, types;
                         cap = v"12.1", feature_set = :arch)
    ptx = emit_ptx(_sm121a_mma_kind_mxf4nvf4!, types;
                   cap = v"12.1", feature_set = :arch)
    @test occursin(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3",
        ptx)
end
