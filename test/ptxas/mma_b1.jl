# Exact-floor offline evidence for the six classic warp-level single-bit MMA
# forms. PTX 9.3 §9.7.15.5.14: m8n8k128.xor is PTX 7.0 / sm_75; AND
# raises the floor to PTX 7.1 / sm_80, and both m16 shapes require sm_80.

using PTX: Operation

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
