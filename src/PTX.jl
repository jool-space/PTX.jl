module PTX

using LLVM
using LLVM.Interop: @asmcall

include("ir/nodes.jl")
using .IR

include("parser/Parser.jl")

include("codegen/Codegen.jl")
using .Codegen: ptx_to_julia, ir_to_julia
export ptx_to_julia, ir_to_julia

include("address_space.jl")
include("types.jl")

include("inst.jl")
export @ptx_str, @sreg_str

include("wrappers/cp_async.jl")
include("wrappers/cvt.jl")
include("wrappers/fence.jl")
include("wrappers/ldmatrix.jl")
include("wrappers/mapa.jl")
include("wrappers/mbarrier.jl")
include("wrappers/mma.jl")
include("wrappers/mma_scaled.jl")
include("wrappers/setp.jl")
include("wrappers/shfl.jl")
include("wrappers/stmatrix.jl")
include("wrappers/tma.jl")
include("wrappers/vec_ldst.jl")
include("wrappers/wgmma.jl")
include("wrappers/tcgen05.jl")
include("wgmma_descriptor.jl")
include("wgmma_layout.jl")
include("tcgen05_descriptor.jl")
include("tensor_map.jl")
include("pipeline.jl")

end
