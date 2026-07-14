# Offline compiler evidence for the six ordinary-cvt source-role shapes closed
# by the host ledger. These functions use the exact Julia carriers emitted by
# the transpiler; ptxas therefore checks that position-aware wrapping reaches
# legal PTX register classes without requiring a live device.

function _cvt_immediate_fundamental!(ints, floats)
    @inbounds begin
        ints[1] = ptx"cvt.u16.u8"(UInt8(17))
        ints[2] = ptx"cvt.u32.u16"(UInt16(513))
        ints[3] = ptx"cvt.u64.u32"(UInt32(65539))
        ints[4] = ptx"cvt.u32.u64"(UInt64(7))
        ints[5] = UInt64(ptx"cvt.s16.s8"(Int8(11)))
        ints[6] = UInt64(ptx"cvt.s32.s16"(Int16(1025)))
        ints[7] = UInt64(ptx"cvt.s64.s32"(Int32(65537)))
        ints[8] = UInt64(ptx"cvt.s32.s64"(Int64(19)))
        ints[9] = ptx"cvt.u16.u8"(256 % UInt8)
        ints[10] = UInt64(ptx"cvt.s16.s8"(255 % Int8))
        ints[11] = ptx"cvt.u32.u32"(-1 % UInt32)
        ints[12] = ptx"cvt.u32.u32"(0x100000000 % UInt32)

        # PTX exact literals retain their spelling width, then convert to the
        # cvt source type at use. These are the two cross-width renderer cases.
        floats[1] = ptx"cvt.f64.f32"(
            Float32(reinterpret(Float64, 0x3ff0000000000000)))
        floats[2] = Float64(ptx"cvt.rn.f32.f64"(
            Float64(reinterpret(Float32, 0x3f800000))))
    end
    return nothing
end

function _cvt_immediate_pack2!(out)
    @inbounds begin
        out[1] = ptx"cvt.rn.f16x2.f32"(Float32(1.0), Float32(2.0))
        out[2] = ptx"cvt.rn.bf16x2.f32"(Float32(3.0), Float32(4.0))
    end
    return nothing
end

function _cvt_immediate_fp8!(out)
    @inbounds begin
        out[1] = ptx"cvt.rn.satfinite.e4m3x2.f32"(
            Float32(1.0), Float32(2.0))
        out[2] = ptx"cvt.rn.satfinite.e5m2x2.f32"(
            Float32(3.0), Float32(4.0))
    end
    return nothing
end

function _cvt_immediate_stochastic!(out32, rbits::UInt32)
    @inbounds begin
        # Two scalar f32 sources + one b32 register rbits role.
        out32[1] = ptx"cvt.rs.f16x2.f32"(
            Float32(1.0), Float32(2.0), rbits)
        # One four-register f32 vector + one b32 register rbits role.
        out32[2] = ptx"cvt.rs.satfinite.e4m3x4.f32"(
            (Float32(1.0), Float32(2.0), Float32(3.0), Float32(4.0)),
            rbits)
    end
    return nothing
end

function _cvt_immediate_scaled!(out32, packed::UInt16, scale::UInt16)
    @inbounds begin
        # Packed narrow source + a distinct b16 scale-factor role.
        out32[1] = ptx"cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2"(
            packed, scale)
    end
    return nothing
end

function _cvt_immediate_s2f6_down!(out16, scale::UInt16)
    @inbounds begin
        # Two f32 data sources + trailing b16 scale-factor.
        out16[1] = ptx"cvt.rn.satfinite.scaled::n2::ue8m0.s2f6x2.f32"(
            Float32(1.0), Float32(2.0), scale)
    end
    return nothing
end

function _cvt_immediate_s2f6_up!(out32, packed::UInt16, scale::UInt16)
    @inbounds out32[1] =
        ptx"cvt.rn.scaled::n2::ue8m0.bf16x2.s2f6x2"(packed, scale)
    return nothing
end

@noinline function _cvt_immediate_s2f6_syntax(scale::UInt16)
    # RAW_CONTRACT keeps the inline-asm call until PTX emission even though
    # the result is intentionally dead. CUDA 13.3 can therefore validate the
    # exact syntax and sm_121a target without reaching its live-result ICE.
    ptx"cvt.rn.satfinite.scaled::n2::ue8m0.s2f6x2.f32"raw(
        Float32(1.0), Float32(2.0), scale)
    return nothing
end

const _CVT_FUNDAMENTAL_TYPES =
    Tuple{CuDeviceVector{UInt64,1}, CuDeviceVector{Float64,1}}
const _CVT_PACKED_OUT_TYPES = Tuple{CuDeviceVector{UInt32,1}}
const _CVT_STOCHASTIC_TYPES = Tuple{CuDeviceVector{UInt32,1}, UInt32}
const _CVT_SCALED_TYPES =
    Tuple{CuDeviceVector{UInt32,1}, UInt16, UInt16}
const _CVT_S2F6_DOWN_TYPES = Tuple{CuDeviceVector{UInt16,1}, UInt16}
const _CVT_S2F6_UP_TYPES =
    Tuple{CuDeviceVector{UInt32,1}, UInt16, UInt16}
const _CVT_S2F6_SYNTAX_TYPES = Tuple{UInt16}

@testset "ordinary cvt source carriers assemble by target partition" begin
    # LLVM retains the Julia i8 carrier even though the NVPTX `h` constraint
    # allocates a 16-bit PTX register. Keep both layers visible: these exact
    # optimized-IR calls plus the sm_75 ptxas case below prove the backend's
    # narrow-value legalization rather than assuming an i16 Julia promotion.
    narrow_llvm = emit_llvm(_cvt_immediate_fundamental!,
                            _CVT_FUNDAMENTAL_TYPES; cap = v"7.5")
    @test occursin(
        r"call i16 asm \"cvt\.u16\.u8 \$0, \$1;\", \"=h,h\"\(i8 17\)",
        narrow_llvm)
    @test occursin(
        r"call i16 asm \"cvt\.s16\.s8 \$0, \$1;\", \"=h,h\"\(i8 11\)",
        narrow_llvm)
    @test occursin(
        r"call i16 asm \"cvt\.u16\.u8 \$0, \$1;\", \"=h,h\"\(i8 0\)",
        narrow_llvm)
    @test occursin(
        r"call i16 asm \"cvt\.s16\.s8 \$0, \$1;\", \"=h,h\"\(i8 -1\)",
        narrow_llvm)
    @test occursin(
        r"call i32 asm \"cvt\.u32\.u32 \$0, \$1;\", \"=r,r\"\(i32 -1\)",
        narrow_llvm)
    @test occursin(
        r"call i32 asm \"cvt\.u32\.u32 \$0, \$1;\", \"=r,r\"\(i32 0\)",
        narrow_llvm)

    narrow_ptx = emit_ptx(_cvt_immediate_fundamental!,
                          _CVT_FUNDAMENTAL_TYPES; cap = v"7.5")
    @test occursin(
        r"mov\.b16\s+%rs\d+, 0;\s+// begin inline asm\s+cvt\.u16\.u8",
        narrow_ptx)
    @test occursin(
        r"mov\.b16\s+%rs\d+, 255;\s+// begin inline asm\s+cvt\.s16\.s8",
        narrow_ptx)
    @test occursin(
        r"mov\.b32\s+%r\d+, -1;\s+// begin inline asm\s+cvt\.u32\.u32",
        narrow_ptx)
    @test occursin(
        r"mov\.b32\s+%r\d+, 0;\s+// begin inline asm\s+cvt\.u32\.u32",
        narrow_ptx)

    cases = (
        (_cvt_immediate_fundamental!, _CVT_FUNDAMENTAL_TYPES,
         v"7.5", :baseline, :normal,
         ("cvt.u16.u8", "cvt.u64.u32", "cvt.f64.f32", "cvt.rn.f32.f64")),
        (_cvt_immediate_pack2!, _CVT_PACKED_OUT_TYPES,
         v"8.0", :baseline, :normal,
         ("cvt.rn.f16x2.f32", "cvt.rn.bf16x2.f32")),
        (_cvt_immediate_fp8!, _CVT_PACKED_OUT_TYPES,
         v"8.9", :baseline, :normal,
         ("cvt.rn.satfinite.e4m3x2.f32",
          "cvt.rn.satfinite.e5m2x2.f32")),
        (_cvt_immediate_stochastic!, _CVT_STOCHASTIC_TYPES,
         v"10.0", :arch, :normal,
         ("cvt.rs.f16x2.f32", "cvt.rs.satfinite.e4m3x4.f32")),
        (_cvt_immediate_scaled!, _CVT_SCALED_TYPES,
         v"12.0", :family, :normal,
         ("cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2",
          ".target sm_120f")),
        # CUDA 13.3 ptxas ICEs (C7907) when these otherwise-valid s2f6 results
        # are live, including in minimal hand-written PTX without debug
        # metadata. Keep the compiler-emitted live-result PTX as evidence and
        # validate syntax/target separately with the dead-result probe below.
        (_cvt_immediate_s2f6_down!, _CVT_S2F6_DOWN_TYPES,
         v"12.1", :arch, :live_emission,
         ("cvt.rn.satfinite.scaled::n2::ue8m0.s2f6x2.f32",
          ".target sm_121a")),
        (_cvt_immediate_s2f6_up!, _CVT_S2F6_UP_TYPES,
         v"12.1", :arch, :live_emission,
         ("cvt.rn.scaled::n2::ue8m0.bf16x2.s2f6x2",
          ".target sm_121a")),
    )
    for (kernel, types, cap, feature_set, assembly, heads) in cases
        if assembly === :normal
            @test ptxas_compiles(kernel, types; cap, feature_set)
        elseif assembly === :live_emission
            # The exact CUDA 13.3 ICE boundary above is intentionally not
            # treated as a package failure.
        else
            error("unknown ordinary-cvt ptxas mode $assembly")
        end
        emitted = emit_ptx(kernel, types; cap, feature_set)
        for head in heads
            @test occursin(head, emitted)
        end
    end

    @test ptxas_compiles(
        _cvt_immediate_s2f6_syntax, _CVT_S2F6_SYNTAX_TYPES;
        cap = v"12.1", feature_set = :arch)
    syntax_ptx = emit_ptx(
        _cvt_immediate_s2f6_syntax, _CVT_S2F6_SYNTAX_TYPES;
        cap = v"12.1", feature_set = :arch)
    @test occursin(".target sm_121a", syntax_ptx)
    @test occursin(
        "cvt.rn.satfinite.scaled::n2::ue8m0.s2f6x2.f32", syntax_ptx)
end
