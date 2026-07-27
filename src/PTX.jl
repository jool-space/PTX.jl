module PTX

using LLVM
using LLVM.Interop: @asmcall

include("ir/nodes.jl")
using .IR

include("parser/Parser.jl")

include("implicit_state.jl")

include("structured_results.jl")
include("immediate_forms.jl")
include("scalar_results.jl")
include("cvt_forms.jl")
include("vector_results.jl")
include("b128_forms.jl")
include("mbarrier_forms.jl")

include("nvvm/NVVM.jl")
using .NVVM: NVVM, @nvvm_str

include("address_space.jl")
include("address_operands.jl")
export Address, address
export B128, b128
include("forms.jl")
include("types.jl")

# The transpiler shares the audited structured/scalar/cvt and mbarrier ABI
# validators above, so load it only after their definitions are available.
include("codegen/Codegen.jl")
using .Codegen: ptx_to_julia, ir_to_julia
export ptx_to_julia, ir_to_julia

include("inst.jl")
export @ptx_str, @optype_str, @sreg_str
export vector_load

include("wrappers/barrier.jl")
include("wrappers/barrier_cluster.jl")
include("wrappers/cp_async.jl")
include("wrappers/cvt.jl")
include("wrappers/extended_precision.jl")
include("wrappers/fabric.jl")
include("wrappers/fence.jl")
include("wrappers/ldmatrix.jl")
include("wrappers/mapa.jl")
include("wrappers/mbarrier.jl")
include("wrappers/mma.jl")
include("wrappers/mma_scaled.jl")
include("wrappers/shfl.jl")
include("wrappers/stmatrix.jl")
include("wrappers/tma.jl")
include("wrappers/vec_ldst.jl")
include("wrappers/wgmma.jl")
include("wrappers/wgmma_sp.jl")
include("wrappers/tcgen05.jl")
include("wgmma_descriptor.jl")
include("wgmma_layout.jl")
include("tcgen05_descriptor.jl")
include("tcgen05_layout.jl")
include("tensor_map.jl")
include("mbarriers.jl")
include("pipelines.jl")

function __init__()
    # The TMA tensor-map encoders are implemented by the CUDACore package
    # extension; without it they have no methods. Name the missing package
    # in the MethodError instead of leaving a bare zero-methods failure.
    Base.Experimental.register_error_hint(MethodError) do io, exc, _, _
        if exc.f === tensor_map_encode_tiled || exc.f === tensor_map_tile_2d
            print(io, "\nThe TMA tensor-map encoder is provided by PTX.jl's ",
                      "CUDACore extension — run `using CUDA` (or ",
                      "`using CUDACore`) to load it.")
        end
    end
end

end
