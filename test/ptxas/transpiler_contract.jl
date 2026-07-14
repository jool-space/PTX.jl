# End-to-end compiler evidence for the narrow scalar transpiler subset.  The
# generated Julia is evaluated, compiled through LLVM/NVPTX, and accepted by
# ptxas without requiring a visible GPU.

const _TRANSPILE_SHIFT_PTX = """
.version 8.5
.target sm_75
.address_size 64
.visible .entry transpile_shift(
    .param .u64 output,
    .param .u64 value
)
{
    .reg .b64 %rd<3>;
    ld.param.u64 %rd0, [output];
    ld.param.u64 %rd1, [value];
    shl.b64 %rd2, %rd1, 2;
    st.global.b64 [%rd0], %rd2;
    ret;
}
"""

const _TRANSPILE_SHIFT_JULIA = PTX.ptx_to_julia(_TRANSPILE_SHIFT_PTX)
Core.eval(@__MODULE__, Meta.parseall(_TRANSPILE_SHIFT_JULIA))

@testset "transpiled scalar kernel survives LLVM, NVPTX, and ptxas" begin
    types = Tuple{UInt64, UInt64}
    @test occursin("UInt32(2)", _TRANSPILE_SHIFT_JULIA)
    @test !occursin("UInt64(2)", _TRANSPILE_SHIFT_JULIA)

    llvm = emit_llvm(transpile_shift, types; cap = v"7.5")
    @test occursin("shl.b64", llvm)
    @test occursin("i32 2", llvm)

    ptx = emit_ptx(transpile_shift, types; cap = v"7.5")
    @test occursin("shl.b64", ptx)
    @test occursin("st.global.b64", ptx)
    @test ptxas_compiles(transpile_shift, types; cap = v"7.5")
end
