module AS

# Match CUDACore.AS.* so `LLVMPtr{T, PTX.AS.Global}` and
# `LLVMPtr{T, CUDACore.AS.Global}` are the same type.
const Generic = 0
const Global  = 1
const Shared  = 3
const Const   = 4
const Local   = 5
const Param   = 101

end # module AS
