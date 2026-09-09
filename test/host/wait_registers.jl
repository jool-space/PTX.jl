using PTX: wait_registers

const _WR_LD = ptx"tcgen05.wait::ld.sync.aligned"
const _WR_ST = ptx"tcgen05.wait::st.sync.aligned"

@testset "wait_registers preserves the value ABI" begin
    for op in (_WR_LD, _WR_ST),
            T in (UInt32, Int32, Float32, UInt64, Int64, Float64),
            V in (T, NTuple{1,T}, NTuple{3,T}, NTuple{32,T}, NTuple{64,T})
        ci, rt = only(Base.code_typed(wait_registers, (typeof(op), V)))
        @test rt === V
        ir = replace(string(ci), "\\\"" => "\"")
        @test occursin("asm sideeffect", ir)
        @test occursin("convergent nomerge nounwind", ir)
        @test occursin("~{memory}", ir)
    end
    for op in (_WR_LD, _WR_ST)
        ci, rt = only(Base.code_typed(wait_registers, (typeof(op), Tuple{})))
        @test rt === Tuple{}
        @test occursin("asm sideeffect", string(ci))
    end
end

function _wr_scalar_dependency!(out::CuDeviceVector{UInt64,1}, value::UInt64)
    result = wait_registers(_WR_LD, value)
    @inbounds out[1] = result ⊻ UInt64(0x123456789abcdef0)
    nothing
end

@testset "a register consumer depends on the wait's output" begin
    ir = emit_host_llvm(_wr_scalar_dependency!,
                   Tuple{CuDeviceVector{UInt64,1},UInt64};
                   cap=v"10.0", feature_set=:arch)
    wait = match(r"(%[\w.]+) = (?:tail )?call i64 asm sideeffect \"tcgen05\.wait::ld\.sync\.aligned;\", \"=l,0,~\{memory\}\"\(i64 [^\n]+\) #(\d+)", ir)
    @test wait !== nothing
    if wait !== nothing
        result, attr = wait.captures
        @test occursin("xor i64 $result, 1311768467463790320", ir)
        attributes = match(Regex("attributes #$attr = \\{([^}]+)\\}"), ir)
        @test attributes !== nothing
        @test all(flag -> occursin(flag, attributes[1]), ("convergent", "nomerge", "nounwind"))
    end
    @test !occursin("alloca", ir)
    @test !occursin("gpu_gc", ir)
end

@generated function _wr_tuple_copy!(out::CuDeviceVector{T,1},
                                    values::NTuple{N,T}, op::W) where {T,N,W}
    stores = [:(out[$i] = result[$i]) for i in 1:N]
    quote
        result = wait_registers(op, values)
        @inbounds begin
            $(stores...)
        end
        nothing
    end
end

@testset "32-bit pairs and 64-bit carriers survive optimized lowering" begin
    for (V, expected) in (
            (NTuple{2,UInt32}, "=l,0,~{memory}"),
            (NTuple{3,Float32}, "=l,=f,0,1,~{memory}"),
            (NTuple{3,Int32}, "=l,=r,0,1,~{memory}"),
            (NTuple{2,UInt64}, "=l,=l,0,1,~{memory}"),
            (NTuple{2,Float64}, "=d,=d,0,1,~{memory}"))
        T = eltype(V)
        ir = emit_host_llvm(_wr_tuple_copy!, Tuple{CuDeviceVector{T,1},V,typeof(_WR_LD)};
                       cap=v"10.0", feature_set=:arch)
        @test occursin(expected, ir)
        @test !occursin("alloca", ir)
        @test !occursin("gpu_gc", ir)
        @test !occursin("call fastcc", ir)
    end
end

@testset "unsupported waits and values have no method" begin
    for op in (ptx"bar.sync", ptx"tcgen05.fence::before_thread_sync",
               ptx"tcgen05.wait::ld.sync", ptx"tcgen05.wait::ld.sync.aligned"raw,
               ptx"wgmma.wait_group.sync.aligned")
        @test_throws MethodError wait_registers(op, UInt32(1))
    end
    for value in (true, UInt16(1), Int128(1), (UInt32(1), UInt64(2)),
                  (a=UInt32(1),), [UInt32(1)], ((UInt32(1), UInt32(2)),))
        @test_throws MethodError wait_registers(_WR_LD, value)
    end
end
