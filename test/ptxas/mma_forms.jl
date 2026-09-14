# Exact-floor offline evidence for the classic warp-level mma.sync form
# inventories: the six single-bit forms (m8n8k128.xor is PTX 7.0 / sm_75;
# AND raises the floor to PTX 7.1 / sm_80, and both m16 shapes require
# sm_80), the 32 modern dense integer forms (PTX 7.0, sm_80), the 64
# integer mma.sp forms (ordinary sparse PTX 7.1, ordered metadata PTX 8.5,
# sm_80 throughout), and the 12 ordered-metadata floating ABIs (classic
# 16-bit/tf32 at sm_80, FP8 at sm_89).

using PTX: Operation

# --- single-bit ------------------------------------------------------------------

const _PTXAS_B1_MMA_FORMS = (
    (; shape = :m8n8k128,  bitop = :xor, n_a = 1, n_b = 1, n_cd = 2),
    (; shape = :m8n8k128,  bitop = :and, n_a = 1, n_b = 1, n_cd = 2),
    (; shape = :m16n8k128, bitop = :xor, n_a = 2, n_b = 1, n_cd = 4),
    (; shape = :m16n8k128, bitop = :and, n_a = 2, n_b = 1, n_cd = 4),
    (; shape = :m16n8k256, bitop = :xor, n_a = 4, n_b = 2, n_cd = 4),
    (; shape = :m16n8k256, bitop = :and, n_a = 4, n_b = 2, n_cd = 4),
)

function _b1_callsite_is_convergent(llvm::AbstractString,
                                    needle::AbstractString)
    groups = Dict{String,String}()
    for line in eachline(IOBuffer(llvm))
        m = match(r"^attributes #([0-9]+) = \{([^}]*)\}", strip(line))
        m === nothing || (groups[m.captures[1]] = m.captures[2])
    end
    calls = [String(line) for line in eachline(IOBuffer(llvm))
             if occursin(" call ", line) && occursin(needle, line)]
    length(calls) == 1 || return false
    m = match(r" #([0-9]+)(?:,|$)", strip(only(calls)))
    m === nothing && return false
    attrs = get(groups, m.captures[1], "")
    occursin(r"\bconvergent\b", attrs) && occursin(r"\bnomerge\b", attrs)
end

function _b1_raw_floor_ptxas(version::AbstractString)
    source = """
    .version $version
    .target sm_75
    .address_size 64
    .visible .entry b1_floor() {
      .reg .b32 a, b;
      .reg .s32 c<2>, d<2>;
      mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc
        {d0, d1}, {a}, {b}, {c0, c1};
      ret;
    }
    """
    mktempdir() do dir
        ptx_path = joinpath(dir, "input.ptx")
        cubin_path = joinpath(dir, "output.cubin")
        write(ptx_path, source)
        cmd = `$(CUDACore.CUDA_Compiler.ptxas()) --gpu-name sm_75 --output-file $cubin_path $ptx_path`
        err = IOBuffer()
        ok = success(pipeline(cmd; stdout = devnull, stderr = err))
        (; accepted = ok, log = String(take!(err)))
    end
end

let calls = Expr(:block)
    for (i, row) in enumerate(_PTXAS_B1_MMA_FORMS)
        helper = Symbol("_ptxas_b1_mma_", i, "!")
        mods = (:sync, :aligned, row.shape, :row, :col,
                :s32, :b1, :b1, :s32, row.bitop, :popc)
        op = Operation{:mma, mods}()
        @eval @inline function $helper(
                out::Core.LLVMPtr{Int32, PTX.AS.Global})
            a = ntuple(j -> xor(UInt32(0x9e3779b9), UInt32(j)),
                       Val($(row.n_a)))
            b = ntuple(j -> xor(UInt32(0x7f4a7c15), UInt32(j)),
                       Val($(row.n_b)))
            c = ntuple(j -> Int32(j), Val($(row.n_cd)))
            d = $op(a, b, c)
            ptx"st.global.s32"(out + $(4 * (i - 1)), d[1])
            nothing
        end
        push!(calls.args, :($helper(out)))
    end
    @eval function _ptxas_b1_mma_all!(
            out::Core.LLVMPtr{Int32, PTX.AS.Global})
        $calls
        nothing
    end
end

function _ptxas_b1_mma_sm75!(
        out::Core.LLVMPtr{Int32, PTX.AS.Global})
    _ptxas_b1_mma_1!(out)
    nothing
end

@testset "single-bit mma exact target floors" begin
    types = Tuple{Core.LLVMPtr{Int32, PTX.AS.Global}}

    @test ptxas_compiles(_ptxas_b1_mma_sm75!, types; cap = v"7.5")
    ptx75 = emit_ptx(_ptxas_b1_mma_sm75!, types; cap = v"7.5")
    @test occursin(".target sm_75", ptx75)
    @test occursin("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc",
                   ptx75)
    @test _b1_raw_floor_ptxas("7.0").accepted
    below_version = _b1_raw_floor_ptxas("6.5")
    @test !below_version.accepted
    @test occursin("Feature '.m8n8k128' requires PTX ISA .version 7.0 or later",
                   below_version.log)

    @test ptxas_compiles(_ptxas_b1_mma_all!, types; cap = v"8.0")
    ptx80 = emit_ptx(_ptxas_b1_mma_all!, types; cap = v"8.0")
    @test occursin(".target sm_80", ptx80)
    for row in _PTXAS_B1_MMA_FORMS
        head = "mma.sync.aligned.$(row.shape).row.col." *
               "s32.b1.b1.s32.$(row.bitop).popc"
        @test count(head, ptx80) == 1
    end

    # Optimized LLVM retains all six warp collectives and attaches the
    # dedicated convergence barrier to every call site.
    llvm = emit_llvm(_ptxas_b1_mma_all!, types; cap = v"8.0")
    for row in _PTXAS_B1_MMA_FORMS
        if row.shape === :m8n8k128 && row.bitop === :xor
            asm = "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc"
            @test count(asm, llvm) == 1
            @test _b1_callsite_is_convergent(llvm, asm)
        else
            intr = "llvm.nvvm.mma.$(row.bitop).popc.$(row.shape).row.col.b1"
            @test count(intr, llvm) >= 2
            @test _b1_callsite_is_convergent(llvm, intr)
        end
    end
    @test occursin("convergent nomerge", llvm)
end

# --- dense integer ---------------------------------------------------------------

const _PTXAS_INTEGER_MMA_FORMS = let rows = NamedTuple[]
    for (shape, types, n_a, n_b) in (
            (:m16n8k16, (:u8, :s8), 2, 1),
            (:m16n8k32, (:u8, :s8), 4, 2),
            (:m16n8k32, (:u4, :s4), 2, 1),
            (:m16n8k64, (:u4, :s4), 4, 2))
        for a in types, b in types, satfinite in (false, true)
            sat = satfinite ? (:satfinite,) : ()
            mods = (:sync, :aligned, shape, :row, :col, sat...,
                    :s32, a, b, :s32)
            spelling = join(("mma", String.(mods)...), '.')
            push!(rows, (; mods, spelling, n_a, n_b))
        end
    end
    Tuple(rows)
end

# Generate one statically named helper per form, then one kernel that calls
# all helpers.  This keeps exact-floor evidence to a single GPUCompiler/ptxas
# job while ensuring each wrapper's Julia-side marshaling reaches ISel.
let calls = Expr(:block)
    for (i, row) in enumerate(_PTXAS_INTEGER_MMA_FORMS)
        helper = Symbol("_ptxas_integer_mma_", i, "!")
        op = Operation{:mma, row.mods}()
        @eval @inline function $helper(out)
            a = ntuple(_ -> UInt32(0x01010101), Val($(row.n_a)))
            b = ntuple(_ -> UInt32(0x01010101), Val($(row.n_b)))
            c = ntuple(_ -> Int32(0), Val(4))
            d = $op(a, b, c)
            @inbounds out[$i] = d[1]
            nothing
        end
        push!(calls.args, :($helper(out)))
    end
    @eval function _ptxas_integer_mma_all!(out)
        $calls
        nothing
    end
end

@testset "modern dense integer mma at sm_80" begin
    @test length(_PTXAS_INTEGER_MMA_FORMS) == 32
    types = Tuple{CuDeviceVector{Int32, 1}}
    @test ptxas_compiles(_ptxas_integer_mma_all!, types; cap = v"8.0")
    ptx = emit_ptx(_ptxas_integer_mma_all!, types; cap = v"8.0")
    @test occursin(".target sm_80", ptx)
    for row in _PTXAS_INTEGER_MMA_FORMS
        @test occursin(row.spelling, ptx)
    end
end

# --- sparse integer --------------------------------------------------------------

const _PTXAS_INTEGER_SP_FORMS = let rows = NamedTuple[]
    for (shape, types, n_a, n_b, selectors, packed_one) in (
            (:m16n8k32,  (:u8, :s8), 2, 2, 0:1, UInt32(0x01010101)),
            (:m16n8k64,  (:u8, :s8), 4, 4, 0:0, UInt32(0x01010101)),
            (:m16n8k64,  (:u4, :s4), 2, 2, 0:1, UInt32(0x11111111)),
            (:m16n8k128, (:u4, :s4), 4, 4, 0:0, UInt32(0x11111111)))
        for a in types, b in types, satfinite in (false, true),
                ordered in (false, true)
            variant = ordered ? Symbol("sp::ordered_metadata") : :sp
            sat = satfinite ? (:satfinite,) : ()
            mods = (variant, :sync, :aligned, shape, :row, :col, sat...,
                    :s32, a, b, :s32)
            spelling = join(("mma", String.(mods)...), '.')
            push!(rows, (; mods, spelling, n_a, n_b,
                         selector=last(selectors), packed_one))
        end
    end
    Tuple(rows)
end

let calls = Expr(:block)
    for (i, row) in enumerate(_PTXAS_INTEGER_SP_FORMS)
        helper = Symbol("_ptxas_integer_sp_", i, "!")
        op = Operation{:mma, row.mods}()
        @eval @inline function $helper(out)
            a = ntuple(_ -> $(row.packed_one), Val($(row.n_a)))
            b = ntuple(_ -> $(row.packed_one), Val($(row.n_b)))
            c = ntuple(_ -> Int32(0), Val(4))
            d = $op(a, b, c, UInt32(0x44444444), Val($(row.selector)))
            @inbounds out[$i] = d[1]
            nothing
        end
        push!(calls.args, :($helper(out)))
    end
    @eval function _ptxas_integer_sp_all!(out)
        $calls
        nothing
    end
end

const _PTXAS_INTEGER_SP_TYPES = Tuple{CuDeviceVector{Int32, 1}}

@testset "integer mma.sp: all 64 forms at sm_80" begin
    @test length(_PTXAS_INTEGER_SP_FORMS) == 64
    @test ptxas_compiles(_ptxas_integer_sp_all!, _PTXAS_INTEGER_SP_TYPES;
                         cap = v"8.0")
    ptx = emit_ptx(_ptxas_integer_sp_all!, _PTXAS_INTEGER_SP_TYPES;
                   cap = v"8.0")
    @test occursin(".target sm_80", ptx)
    @test count("mma.sp", ptx) == 64
    for row in _PTXAS_INTEGER_SP_FORMS
        @test occursin(row.spelling, ptx)
    end

    llvm = emit_llvm(_ptxas_integer_sp_all!, _PTXAS_INTEGER_SP_TYPES;
                     cap = v"8.0")
    @test occursin("llvm.nvvm.mma.sp.m16", llvm)
    @test occursin("llvm.nvvm.mma.sp.ordered.metadata.m16", llvm)
    @test occursin("convergent nomerge", llvm)
end

@testset "integer mma.sp: sm_80 floor rejects sm_75" begin
    rejected = try
        ptxas_compiles(_ptxas_integer_sp_all!, _PTXAS_INTEGER_SP_TYPES;
                       cap = v"7.5")
        false
    catch
        true
    end
    @test rejected
end

# --- sp::ordered_metadata (floating) ---------------------------------------------

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
