# --- Host gate ----------------------------------------------------------------
#
# Inline PTX asm and `llvm.nvvm.*` calls are only meaningful to the NVPTX
# backend. Handed to the host backend they do not raise a Julia error: an
# unknown constraint letter, an unselectable intrinsic, or an asm string the
# host assembler rejects is an LLVM fatal error that takes the whole process
# down — and asm the host assembler happens to accept would run. So every
# emission site is guarded by `on_device()`: `false` under the native
# interpreter, which folds the branch away before LLVM sees the device code,
# and `true` under a device compiler's method table, where the guard folds
# away instead.

"""
    PTX.on_device() -> Bool

`false` in host code and `true` in device code. Every PTX.jl call that emits
inline PTX asm or an NVVM intrinsic is guarded by it, so calling one on the
host throws a [`DeviceOnlyError`](@ref) instead of handing NVPTX code to the
host backend.

The device answer is a method overlay, which PTX.jl's CUDACore extension
installs for CUDA.jl's method table. A GPUCompiler client that compiles for
NVPTX under a method table of its own opts in by overlaying it there:

```julia
Base.Experimental.@overlay MyMethodTable PTX.on_device() = true
```

The overlay must be a plain `@overlay`, not `@consistent_overlay` (which
`CUDACore.@device_override` expands to on Julia 1.11+): a consistent overlay
licenses the compiler to evaluate the host method in its place.
"""
on_device() = false

"""
    DeviceOnlyError(what)

Thrown when PTX device code is called from the host. `what` names the inline
asm or intrinsic the call would have emitted.
"""
struct DeviceOnlyError <: Exception
    what::String
end

function Base.showerror(io::IO, e::DeviceOnlyError)
    print(io, "DeviceOnlyError: `", e.what, "` is PTX device code and cannot ",
              "run on the host. Call it from a kernel (e.g. under `@cuda`), and ",
              "inspect it with `CUDATools.code_llvm`/`@device_code_llvm` rather ",
              "than host reflection. A GPUCompiler client other than CUDA.jl ",
              "must overlay `PTX.on_device() = true` in its method table.")
end

@noinline device_only_error(what::String) = throw(DeviceOnlyError(what))

# Guard the device-code expression `ex`; `what` names it in the host error.
device_only(ex, what::String) =
    :($(GlobalRef(PTX, :on_device))() ? $ex :
      $(GlobalRef(PTX, :device_only_error))($what))

# `LLVM.Interop.@asmcall` behind the host gate. Every caller passes a literal
# asm template (LLVM.Interop requires one), which names the host error.
macro asmcall(asm, args...)
    call = Expr(:macrocall, GlobalRef(LLVM.Interop, Symbol("@asmcall")),
                __source__, asm, args...)
    esc(device_only(call, string(asm)))
end

"""
    PTX.device_code_typed(f, argtypes; optimize = true) -> Vector{Pair{CodeInfo, Type}}

`Base.code_typed` for device code: `f` applied to `argtypes`, inferred under
a device compiler's method table, where the host gate resolves to the device
body. Needs no GPU. Provided by PTX.jl's CUDACore extension.
"""
function device_code_typed end
