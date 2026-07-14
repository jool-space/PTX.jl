# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=7.0

# A parseable Julia string is not a semantic round trip.  Execute a function
# defined from the transpiler output so Ada and GB10 CI prove the complete
# PTX -> IR -> Julia -> LLVM -> PTX path and the heterogeneous shl.b64 ABI.

const _TRANSPILE_RUNTIME_PTX = """
.version 8.5
.target sm_75
.address_size 64
.visible .entry transpile_runtime_shift(
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

const _TRANSPILE_RUNTIME_JULIA = PTX.ptx_to_julia(_TRANSPILE_RUNTIME_PTX)
Core.eval(@__MODULE__, Meta.parseall(_TRANSPILE_RUNTIME_JULIA))

@testset "transpiled scalar kernel executes" begin
    output = CUDACore.zeros(UInt64, 1)
    input = UInt64(0x1234_5678_9abc_def0)
    @cuda threads=1 transpile_runtime_shift(UInt64(pointer(output)), input)
    CUDACore.synchronize()
    @test only(Array(output)) == input << 2
end
