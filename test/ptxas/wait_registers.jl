# Offline SASS comparison: both paths compile with the same Julia, LLVM,
# CUDA toolkit and sm_100a target. Kernel instruction encodings are compared,
# excluding cubin metadata and debug information.
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

function _wr_instruction_encodings(f, tt)
    job = _explicit_target_job(f, tt; cap=v"10.0", feature_set=:arch,
                                minthreads=512)
    image, _ = CUDACore.invoke_frozen(CUDACore.compile, job)
    mktemp() do path, io
        write(io, image)
        close(io)
        sass = read(`$(CUDACore.CUDA_Compiler.nvdisasm()) --print-code --print-instruction-encoding $path`, String)
        # Each Blackwell instruction has two 64-bit encoding comments.
        # Match comments specifically, excluding hexadecimal immediates.
        words = [parse(UInt64, m[1]; base=16)
                 for m in eachmatch(r"/\* 0x([0-9a-f]{16}) \*/", sass)]
        @test !isempty(words) && iseven(length(words))
        words
    end
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
            unbound = _wr_instruction_encodings(_WRUnboundAttention.fab_kernel!, tt)
            bound = _wr_instruction_encodings(_WRBoundAttention.fab_kernel!, tt)
            @test bound == unbound
        end
    end
end
