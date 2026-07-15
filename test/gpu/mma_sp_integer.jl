# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.0
#
# Semantic evidence for representative classic warp-level integer mma.sp
# shapes, signedness products, metadata variants, and saturation modes.

using PTX: Operation

const _RUNTIME_INTEGER_SP_FORMS = (
    (shape=:m16n8k32,  a=:u8, b=:u8, ordered=false, satfinite=false,
     n_a=2, n_b=2, k=32,  selector=1, packed=UInt32(0x01010101)),
    (shape=:m16n8k32,  a=:s8, b=:u8, ordered=true,  satfinite=true,
     n_a=2, n_b=2, k=32,  selector=1, packed=UInt32(0x01010101)),
    (shape=:m16n8k64,  a=:u8, b=:s8, ordered=false, satfinite=true,
     n_a=4, n_b=4, k=64,  selector=0, packed=UInt32(0x01010101)),
    (shape=:m16n8k64,  a=:s8, b=:s8, ordered=true,  satfinite=false,
     n_a=4, n_b=4, k=64,  selector=0, packed=UInt32(0x01010101)),
    (shape=:m16n8k64,  a=:u4, b=:s4, ordered=false, satfinite=false,
     n_a=2, n_b=2, k=64,  selector=1, packed=UInt32(0x11111111)),
    (shape=:m16n8k64,  a=:s4, b=:u4, ordered=true,  satfinite=true,
     n_a=2, n_b=2, k=64,  selector=1, packed=UInt32(0x11111111)),
    (shape=:m16n8k128, a=:u4, b=:u4, ordered=false, satfinite=true,
     n_a=4, n_b=4, k=128, selector=0, packed=UInt32(0x11111111)),
    (shape=:m16n8k128, a=:s4, b=:s4, ordered=true,  satfinite=false,
     n_a=4, n_b=4, k=128, selector=0, packed=UInt32(0x11111111)),
)

let calls = Expr(:block)
    for (form_index, row) in enumerate(_RUNTIME_INTEGER_SP_FORMS)
        helper = Symbol("_runtime_integer_sp_", form_index, "!")
        variant = row.ordered ? Symbol("sp::ordered_metadata") : :sp
        sat = row.satfinite ? (:satfinite,) : ()
        mods = (variant, :sync, :aligned, row.shape, :row, :col, sat...,
                :s32, row.a, row.b, :s32)
        op = Operation{:mma, mods}()
        @eval @inline function $helper(out, tid::UInt32)
            a = ntuple(_ -> $(row.packed), Val($(row.n_a)))
            b = ntuple(_ -> $(row.packed), Val($(row.n_b)))
            c = ntuple(_ -> Int32(0), Val(4))
            # 0x4 in every nibble selects indices 0 then 1. For u4/s4 those
            # indices select the first two all-nonzero two-element subchunks.
            d = $op(a, b, c, UInt32(0x44444444), Val($(row.selector)))
            base = ($(form_index - 1) * 32 + Int(tid)) * 4
            @inbounds for i in 1:4
                out[base + i] = d[i]
            end
            nothing
        end
        push!(calls.args, :($helper(out, tid)))
    end
    @eval function _runtime_integer_sp_all!(out)
        tid = ptx"mov.u32"(sreg"tid.x")
        $calls
        nothing
    end
end

function _runtime_integer_sp_overflow!(sat_out, wrap_out)
    a = ntuple(_ -> UInt32(0x77777777), Val(4))
    b = ntuple(_ -> UInt32(0x77777777), Val(4))
    c = ntuple(_ -> typemax(Int32), Val(4))
    sat = ptx"mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.satfinite.s32.s4.s4.s32"(
        a, b, c, UInt32(0x44444444), Val(0))
    wrap = ptx"mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.s32.s4.s4.s32"(
        a, b, c, UInt32(0x44444444), Val(0))
    tid = ptx"mov.u32"(sreg"tid.x")
    base = Int(tid) * 4
    @inbounds for i in 1:4
        sat_out[base + i] = sat[i]
        wrap_out[base + i] = wrap[i]
    end
    nothing
end

@testset "integer mma.sp representative semantic matrix" begin
    @test length(_RUNTIME_INTEGER_SP_FORMS) == 8
    out = CUDACore.zeros(Int32, length(_RUNTIME_INTEGER_SP_FORMS) * 32 * 4)
    @cuda threads=32 _runtime_integer_sp_all!(out)
    CUDACore.synchronize()
    got = reshape(Array(out), 4 * 32, length(_RUNTIME_INTEGER_SP_FORMS))
    for (i, row) in enumerate(_RUNTIME_INTEGER_SP_FORMS)
        # Each logical A row has K/2 retained +1 values, and every B value is
        # +1. This oracle is independent of fragment and metadata routing.
        @test all(@view(got[:, i]) .== Int32(row.k ÷ 2))
    end
end

@testset "ordered integer mma.sp satfinite versus wrap" begin
    sat_out = CUDACore.zeros(Int32, 32 * 4)
    wrap_out = CUDACore.zeros(Int32, 32 * 4)
    @cuda threads=32 _runtime_integer_sp_overflow!(sat_out, wrap_out)
    CUDACore.synchronize()

    # m16n8k128 u4/s4 sparse A retains 64 values. Both inputs are +7.
    expected_wrap = reinterpret(Int32,
        UInt32(typemax(Int32)) + UInt32(64 * 7 * 7))
    @test all(Array(sat_out) .== typemax(Int32))
    @test all(Array(wrap_out) .== expected_wrap)
end
