# Offline SASS comparison: both paths compile with the same Julia, LLVM,
# CUDA toolkit and sm_100a target, with debug information stripped before
# codegen. Binding the waits must cost nothing: the bound kernel executes the
# same instructions as the unbound one, and its memory, synchronization and
# control-flow instructions keep their order. Operands, register numbering and
# the placement of the remaining arithmetic are not compared: the PTX of the
# two kernels differs throughout in register numbering, and on Julia 1.10
# ptxas schedules that arithmetic differently depending on what the process
# compiled before.
module _WRBoundAttention
using PTX
include("../gpu/blackwell/flash_attention_defs.jl")
end

module _WRUnboundAttention
using PTX
const replaced_waits = Ref(0)
@inline _unbound_wait(op, values) = (op(); values)
function without_register_bindings(ex)
    ex isa Expr || return ex
    if ex.head === :call && ex.args[1] == :(PTX.wait_registers)
        replaced_waits[] += 1
        return Expr(:call, :_unbound_wait, ex.args[2:end]...)
    end
    Expr(ex.head, map(without_register_bindings, ex.args)...)
end
include(without_register_bindings, "../gpu/blackwell/flash_attention_defs.jl")
end

# Mnemonics, with modifiers and without predicate guards, of a kernel's SASS
# in program order.
function _wr_mnemonics(f, tt)
    job = _explicit_target_job(f, tt; cap=v"10.0", feature_set=:arch,
                                minthreads=512)
    # Debug metadata can perturb instruction scheduling even when only code
    # is compared. Strip it before the backend sees either kernel.
    job = CompilerJob(job.source, CompilerConfig(job.config; strip=true))
    image, _ = CUDACore.invoke_frozen(CUDACore.compile, job)
    mktemp() do path, io
        write(io, image)
        close(io)
        sass = read(`$(CUDACore.CUDA_Compiler.nvdisasm()) --print-code $path`, String)
        mnemonics = [String(m[1]) for m in eachmatch(
            r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)", sass)]
        @test !isempty(mnemonics)
        mnemonics
    end
end

# Instructions whose relative order is observable: memory, synchronization,
# tensor-core and control flow.
const _WR_ORDERED = Regex("^(" * join((
    "LD", "ST", "ATOM", "RED", "CCTL",                       # memory
    "SYNCS", "BAR", "MEMBAR", "FENCE", "DEPBAR", "ERRBAR", "CGAERRBAR",
    "WARPSYNC", "ELECT", "NANOSLEEP", "YIELD",               # synchronization
    "UTC", "UTMA",                                           # tensor core, TMA
    "BRA", "EXIT", "RET", "CALL", "BSSY", "BSYNC", "BREAK"), # control flow
    "|") * ")")
_wr_ordered(mnemonic) = occursin(_WR_ORDERED, mnemonic)

function _wr_counts(mnemonics)
    counts = Dict{String,Int}()
    for m in mnemonics
        counts[m] = get(counts, m, 0) + 1
    end
    counts
end

# Mnemonics whose instruction counts differ, as `mnemonic => (bound, unbound)`.
function _wr_count_differences(bound, unbound)
    b, u = _wr_counts(bound), _wr_counts(unbound)
    Dict(m => (get(b, m, 0), get(u, m, 0))
         for m in union(keys(b), keys(u)) if get(b, m, 0) != get(u, m, 0))
end

# First position where two ordered streams diverge, with a little context
# from each, or `nothing` when they are equal.
function _wr_first_divergence(bound, unbound)
    bound == unbound && return nothing
    n = min(length(bound), length(unbound))
    i = something(findfirst(k -> bound[k] != unbound[k], 1:n), n + 1)
    window(v) = v[max(1, i - 2):min(length(v), i + 2)]
    (; index = i, bound = window(bound), unbound = window(unbound))
end

@testset "register waits preserve six attention instruction streams" begin
    @test _WRUnboundAttention.replaced_waits[] == 6
    default = _WRBoundAttention.FAB_CFG_DEFAULT
    configs = (
        :default => default,
        :scoreboard => (; default..., scoreboard=true),
        :beacon => (; default..., beacon=true),
        :registers => (; default..., nreg=(144,88,128)),
        :emulation => (; default..., emu=(1,3,5,7)),
        :quarters => (; default..., splitp=false),
    )
    for (name, cfg) in configs
        @testset "$name" begin
            tt = Tuple{CuDeviceVector{BFloat16,1},
                       PTX.TMADescriptorPtr, PTX.TMADescriptorPtr, PTX.TMADescriptorPtr,
                       UInt32, UInt32, UInt32, UInt32, UInt32, Float32,
                       CuDeviceVector{UInt32,1}, typeof(Val(cfg))}
            unbound = _wr_mnemonics(_WRUnboundAttention.fab_kernel!, tt)
            bound = _wr_mnemonics(_WRBoundAttention.fab_kernel!, tt)
            @test isempty(_wr_count_differences(bound, unbound))
            @test _wr_first_divergence(filter(_wr_ordered, bound),
                                       filter(_wr_ordered, unbound)) === nothing
        end
    end
end
