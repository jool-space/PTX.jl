# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.0
#
# Runtime evidence for the complete modern dense integer MMA slice from PTX
# 9.3 §9.7.15.5.14.  Every A/B element is +1, so every distributed result
# is K regardless of input signedness.  A separate overflowing s4 case
# distinguishes `.satfinite` from the wrapping form.

using PTX: Operation

const _RUNTIME_INTEGER_MMA_FORMS = let rows = NamedTuple[]
    for (shape, types, n_a, n_b, k, packed_one) in (
            (:m16n8k16, (:u8, :s8), 2, 1, 16, UInt32(0x01010101)),
            (:m16n8k32, (:u8, :s8), 4, 2, 32, UInt32(0x01010101)),
            (:m16n8k32, (:u4, :s4), 2, 1, 32, UInt32(0x11111111)),
            (:m16n8k64, (:u4, :s4), 4, 2, 64, UInt32(0x11111111)))
        for a in types, b in types, satfinite in (false, true)
            sat = satfinite ? (:satfinite,) : ()
            mods = (:sync, :aligned, shape, :row, :col, sat...,
                    :s32, a, b, :s32)
            push!(rows, (; mods, n_a, n_b, k, packed_one))
        end
    end
    Tuple(rows)
end

let calls = Expr(:block)
    for (form_index, row) in enumerate(_RUNTIME_INTEGER_MMA_FORMS)
        helper = Symbol("_runtime_integer_mma_", form_index, "!")
        op = Operation{:mma, row.mods}()
        @eval @inline function $helper(out, tid::UInt32)
            a = ntuple(_ -> $(row.packed_one), Val($(row.n_a)))
            b = ntuple(_ -> $(row.packed_one), Val($(row.n_b)))
            c = ntuple(_ -> Int32(0), Val(4))
            d = $op(a, b, c)
            base = ($(form_index - 1) * 32 + Int(tid)) * 4
            @inbounds for i in 1:4
                out[base + i] = d[i]
            end
            nothing
        end
        push!(calls.args, :($helper(out, tid)))
    end
    @eval function _runtime_integer_mma_all!(out)
        tid = ptx"mov.u32"(sreg"tid.x")
        $calls
        nothing
    end
end

function _runtime_integer_mma_overflow!(sat_out, wrap_out)
    packed_seven = UInt32(0x77777777)
    a = ntuple(_ -> packed_seven, Val(4))
    b = ntuple(_ -> packed_seven, Val(2))
    c = ntuple(_ -> typemax(Int32), Val(4))
    sat = ptx"mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32"(
        a, b, c)
    wrap = ptx"mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32"(
        a, b, c)
    tid = ptx"mov.u32"(sreg"tid.x")
    base = Int(tid) * 4
    @inbounds for i in 1:4
        sat_out[base + i] = sat[i]
        wrap_out[base + i] = wrap[i]
    end
    nothing
end

@testset "modern dense integer mma: all 32 forms execute" begin
    @test length(_RUNTIME_INTEGER_MMA_FORMS) == 32
    out = CUDACore.zeros(Int32, 32 * 32 * 4)
    @cuda threads=32 _runtime_integer_mma_all!(out)
    CUDACore.synchronize()
    got = reshape(Array(out), 4 * 32, 32)
    for (i, row) in enumerate(_RUNTIME_INTEGER_MMA_FORMS)
        @test all(@view(got[:, i]) .== Int32(row.k))
    end
end

@testset "integer mma: satfinite clamps while ordinary form wraps" begin
    sat_out = CUDACore.zeros(Int32, 32 * 4)
    wrap_out = CUDACore.zeros(Int32, 32 * 4)
    @cuda threads=32 _runtime_integer_mma_overflow!(sat_out, wrap_out)
    CUDACore.synchronize()

    expected_wrap = reinterpret(Int32,
        UInt32(typemax(Int32)) + UInt32(64 * 7 * 7))
    @test all(Array(sat_out) .== typemax(Int32))
    @test all(Array(wrap_out) .== expected_wrap)
end
