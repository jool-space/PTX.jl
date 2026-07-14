# Direct syntax evidence for every PTX ISA 9.3 §5.4.2 declaration cell in
# every structurally modeled state space. No cubin is loaded onto a device.

const _VECTOR_DECL_TYPES = (
    ".v2" => (".b8", ".b16", ".b32", ".b64",
              ".u8", ".u16", ".u32", ".u64",
              ".s8", ".s16", ".s32", ".s64",
              ".f16", ".f16x2", ".f32", ".f64"),
    ".v4" => (".b8", ".b16", ".b32",
              ".u8", ".u16", ".u32",
              ".s8", ".s16", ".s32",
              ".f16", ".f16x2", ".f32"),
)

function _vector_declaration_ptx()
    module_decls = String[]
    body_decls = String[]
    n = 0
    for (shape, types) in _VECTOR_DECL_TYPES, type in types
        for state in (".global", ".const", ".shared")
            push!(module_decls, "$state $shape $type m$n;")
            n += 1
        end
        push!(body_decls, ".reg $shape $type %r$n;")
        n += 1
        push!(body_decls, ".local $shape $type l$n;")
        n += 1
    end
    """.version 9.3
    .target sm_75
    .address_size 64
    $(join(module_decls, "\n"))
    .visible .entry vector_declarations() {
      $(join(body_decls, "\n  "))
      ret;
    }
    """
end

function _raw_vector_ptxas(source::String; target = "sm_75")
    mktempdir() do dir
        ptx_path = joinpath(dir, "vector_declarations.ptx")
        cubin_path = joinpath(dir, "vector_declarations.cubin")
        write(ptx_path, source)
        cmd = `$(CUDACore.CUDA_Compiler.ptxas()) --gpu-name $target --output-file $cubin_path $ptx_path`
        log = IOBuffer()
        proc = run(pipeline(ignorestatus(cmd), stdout = log, stderr = log))
        (; accepted = success(proc), log = String(take!(log)))
    end
end

@testset "ptxas accepts the complete vector declaration matrix" begin
    @test sum(length(last(cell)) for cell in _VECTOR_DECL_TYPES) == 28
    result = _raw_vector_ptxas(_vector_declaration_ptx())
    @test result.accepted

    aligned_array = """.version 9.3
    .target sm_75
    .address_size 64
    .global .align 16 .v4 .u16 aligned_vectors[3];
    .visible .entry aligned_vector_array() { ret; }
    """
    @test _raw_vector_ptxas(aligned_array).accepted
end

@testset "ptxas rejects nearby non-fundamental and illegal-space forms" begin
    invalid = (
        ".reg .v2 .pred %v;",
        ".reg .v2 .b128 %v;",
        ".reg .v4 .u64 %v;",
        ".reg .v2 .bf16 %v;",
        ".param .v2 .u32 v;",
        ".shared::cta .v2 .u32 v;",
    )
    for declaration in invalid
        source = """.version 9.3
        .target sm_90
        .address_size 64
        .visible .entry invalid_vector() {
          $declaration
          ret;
        }
        """
        @test !_raw_vector_ptxas(source; target = "sm_90").accepted
    end
end
