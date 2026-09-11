# Direct backend probes consume synthesized textual IR without passing through
# Julia's older LLVM parser. The ABI mirrors the installed libnvptx.h.
using NVPTX_LLVM_Backend_jll: NVPTX_LLVM_Backend_jll, libnvptx

struct NVPTXProbeOptions
    cpu::Cstring
    ptx_major::Cuint
    ptx_minor::Cuint
    is_64bit::Cint
    opt_level::Cint
    fma_contraction::Cint
    verbose::Cint
end

function _nvptx_probe_diagnostic(::Cint, message::Cstring, context::Ptr{Cvoid})
    diagnostics = unsafe_pointer_to_objref(context)::Vector{String}
    push!(diagnostics, unsafe_string(message))
    nothing
end

function nvptx_backend_version()
    major, minor, patch = Ref{Cuint}(), Ref{Cuint}(), Ref{Cuint}()
    ccall((:NVPTXGetLLVMVersion, libnvptx), Cvoid,
          (Ref{Cuint}, Ref{Cuint}, Ref{Cuint}), major, minor, patch)
    VersionNumber(major[], minor[], patch[])
end

function compile_nvvm_ir(ir::String, cpu::String, feature::String;
                         fma_contraction::Bool = true)
    ptx = match(r"^\+ptx(\d+)(\d)$", feature)
    ptx === nothing && throw(ArgumentError("expected one PTX ISA feature, got $feature"))
    diagnostics = String[]
    buffer = Ref{Ptr{Cvoid}}(C_NULL)
    message = Ref{Cstring}(C_NULL)
    handler = @cfunction(_nvptx_probe_diagnostic, Cvoid, (Cint, Cstring, Ptr{Cvoid}))
    try
        status = GC.@preserve cpu diagnostics begin
            options = Ref(NVPTXProbeOptions(Base.unsafe_convert(Cstring, cpu),
                parse(UInt32, ptx.captures[1]), parse(UInt32, ptx.captures[2]),
                1, 2, fma_contraction, 1))
            ccall((:NVPTXCompile, libnvptx), Cint,
                  (Ptr{UInt8}, Csize_t, Ref{NVPTXProbeOptions}, Ptr{Cvoid}, Ptr{Cvoid},
                   Ref{Ptr{Cvoid}}, Ref{Cstring}),
                  codeunits(ir), sizeof(ir), options, handler,
                  pointer_from_objref(diagnostics), buffer, message)
        end
        message[] == C_NULL || push!(diagnostics, unsafe_string(message[]))
        text = if buffer[] == C_NULL
            ""
        else
            start = ccall((:NVPTXGetBufferStart, libnvptx), Ptr{UInt8},
                          (Ptr{Cvoid},), buffer[])
            size = ccall((:NVPTXGetBufferSize, libnvptx), Csize_t,
                         (Ptr{Cvoid},), buffer[])
            unsafe_string(start, size)
        end
        (ok = status == 0, ptx = text, diagnostics = join(diagnostics, '\n'))
    finally
        buffer[] == C_NULL || ccall((:NVPTXDisposeMemoryBuffer, libnvptx),
                                    Cvoid, (Ptr{Cvoid},), buffer[])
        message[] == C_NULL || ccall((:NVPTXDisposeMessage, libnvptx),
                                     Cvoid, (Cstring,), message[])
    end
end
