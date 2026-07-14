# Device-free exact-floor evidence for the 12 ordered-metadata sparse MMA
# ABIs.  Classic 16-bit/tf32 forms start at sm_80; FP8 sparse MMA starts at
# sm_89.  The qualifier itself requires PTX 8.5.

function _ordered_sp_classic!(outf::CuDeviceVector{Float32, 1},
        outu::CuDeviceVector{UInt32, 1},
        x1::UInt32, x2::UInt32, x3::UInt32, x4::UInt32, e::UInt32)
    a2 = (x1, x2)
    a4 = (x1, x2, x3, x4)
    c2 = (UInt32(0), UInt32(0))
    c4 = (0f0, 0f0, 0f0, 0f0)

    d1 = ptx"mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"(
        a2, a2, c4, e, Val(3))
    d2 = ptx"mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"(
        a2, a2, c2, e, Val(3))
    d3 = ptx"mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"(
        a2, a2, c4, e, Val(3))
    d4 = ptx"mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32"(
        a4, a4, c4, e, Val(1))
    d5 = ptx"mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16"(
        a4, a4, c2, e, Val(1))
    d6 = ptx"mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32"(
        a4, a4, c4, e, Val(1))
    d7 = ptx"mma.sp::ordered_metadata.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"(
        a2, a2, c4, e, Val(3))
    d8 = ptx"mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32"(
        a4, a4, c4, e, Val(1))

    @inbounds begin
        outf[1] = d1[1]; outu[1] = d2[1]; outf[2] = d3[1]
        outf[3] = d4[1]; outu[2] = d5[1]; outf[4] = d6[1]
        outf[5] = d7[1]; outf[6] = d8[1]
    end
    return nothing
end

function _ordered_sp_fp8!(out::CuDeviceVector{Float32, 1},
        x1::UInt32, x2::UInt32, x3::UInt32, x4::UInt32, e::UInt32)
    a = (x1, x2, x3, x4)
    c = (0f0, 0f0, 0f0, 0f0)
    d1 = ptx"mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32"(a, a, c, e, Val(0))
    d2 = ptx"mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32.e4m3.e5m2.f32"(a, a, c, e, Val(0))
    d3 = ptx"mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32.e5m2.e4m3.f32"(a, a, c, e, Val(0))
    d4 = ptx"mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32.e5m2.e5m2.f32"(a, a, c, e, Val(0))
    @inbounds begin
        out[1] = d1[1]; out[2] = d2[1]; out[3] = d3[1]; out[4] = d4[1]
    end
    return nothing
end

const _ORDERED_CLASSIC_TYPES = Tuple{
    CuDeviceVector{Float32, 1}, CuDeviceVector{UInt32, 1},
    UInt32, UInt32, UInt32, UInt32, UInt32}
const _ORDERED_FP8_TYPES = Tuple{
    CuDeviceVector{Float32, 1}, UInt32, UInt32, UInt32, UInt32, UInt32}

@testset "mma.sp::ordered_metadata classic forms at sm_80" begin
    @test ptxas_compiles(_ordered_sp_classic!, _ORDERED_CLASSIC_TYPES;
                         cap = v"8.0")
    ptx = emit_ptx(_ordered_sp_classic!, _ORDERED_CLASSIC_TYPES; cap = v"8.0")
    @test count("mma.sp::ordered_metadata.sync.aligned", ptx) == 8
    for suffix in (
            "m16n8k16.row.col.f32.f16.f16.f32",
            "m16n8k16.row.col.f16.f16.f16.f16",
            "m16n8k16.row.col.f32.bf16.bf16.f32",
            "m16n8k32.row.col.f32.f16.f16.f32",
            "m16n8k32.row.col.f16.f16.f16.f16",
            "m16n8k32.row.col.f32.bf16.bf16.f32",
            "m16n8k8.row.col.f32.tf32.tf32.f32",
            "m16n8k16.row.col.f32.tf32.tf32.f32")
        @test occursin("mma.sp::ordered_metadata.sync.aligned.$suffix", ptx)
    end

    llvm = emit_llvm(_ordered_sp_classic!, _ORDERED_CLASSIC_TYPES; cap = v"8.0")
    @test occursin("llvm.nvvm.mma.sp.ordered.metadata", llvm)
    @test occursin("convergent nomerge", llvm)
end

@testset "mma.sp::ordered_metadata FP8 forms at sm_89" begin
    @test ptxas_compiles(_ordered_sp_fp8!, _ORDERED_FP8_TYPES; cap = v"8.9")
    ptx = emit_ptx(_ordered_sp_fp8!, _ORDERED_FP8_TYPES; cap = v"8.9")
    @test count("mma.sp::ordered_metadata.sync.aligned", ptx) == 4
    for a in ("e4m3", "e5m2"), b in ("e4m3", "e5m2")
        @test occursin("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32.$a.$b.f32", ptx)
    end
end
