# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.0
#
# Runtime evidence for the classic warp-level mma.sync form inventories:
# all six single-bit forms against a host XOR/AND+popcount reference, the
# complete modern dense integer slice (§9.7.15.5.14), and representative
# integer mma.sp shapes, signedness products, metadata variants, and
# saturation modes. The cc>=8.0 gate is the common floor of the products.

using PTX: Operation
using Random: MersenneTwister

# --- single-bit ------------------------------------------------------------------
# Random per-lane fragments are decoded with the normative maps from
# Figures 62–64 and 97–103, then checked against the host reference.

const _RUNTIME_B1_MMA_FORMS = (
    (; shape = :m8n8k128,  bitop = :xor, n_a = 1, n_b = 1, n_cd = 2, k = 128),
    (; shape = :m8n8k128,  bitop = :and, n_a = 1, n_b = 1, n_cd = 2, k = 128),
    (; shape = :m16n8k128, bitop = :xor, n_a = 2, n_b = 1, n_cd = 4, k = 128),
    (; shape = :m16n8k128, bitop = :and, n_a = 2, n_b = 1, n_cd = 4, k = 128),
    (; shape = :m16n8k256, bitop = :xor, n_a = 4, n_b = 2, n_cd = 4, k = 256),
    (; shape = :m16n8k256, bitop = :and, n_a = 4, n_b = 2, n_cd = 4, k = 256),
)

let
    for (i, row) in enumerate(_RUNTIME_B1_MMA_FORMS)
        kernel = Symbol("_runtime_b1_mma_", i, "!")
        launcher = Symbol("_launch_runtime_b1_mma_", i, "!")
        mods = (:sync, :aligned, row.shape, :row, :col,
                :s32, :b1, :b1, :s32, row.bitop, :popc)
        op = Operation{:mma, mods}()
        @eval function $kernel(out, a_words, b_words, c_words)
            lane = Int(ptx"mov.u32"(sreg"tid.x"))
            a = ntuple(j -> @inbounds(a_words[lane * $(row.n_a) + j]),
                       Val($(row.n_a)))
            b = ntuple(j -> @inbounds(b_words[lane * $(row.n_b) + j]),
                       Val($(row.n_b)))
            c = ntuple(j -> @inbounds(c_words[lane * $(row.n_cd) + j]),
                       Val($(row.n_cd)))
            d = $op(a, b, c)
            @inbounds for j in 1:$(row.n_cd)
                out[lane * $(row.n_cd) + j] = d[j]
            end
            nothing
        end
        @eval function $launcher(out, a_words, b_words, c_words)
            @cuda threads=32 $kernel(out, a_words, b_words, c_words)
        end
    end
end

_b1_bit(words, lane, nwords, i) =
    !iszero((words[lane * nwords + (i ÷ 32) + 1] >> (i & 31)) & 0x1)

function _b1_reference(row, a_words, b_words, c_words)
    m = row.shape === :m8n8k128 ? 8 : 16
    A = falses(m, row.k)
    B = falses(row.k, 8)

    for lane in 0:31
        group = lane >> 2
        thread = lane & 3
        for i in 0:(32 * row.n_a - 1)
            if row.shape === :m8n8k128
                r, col = group, thread * 32 + i
            elseif row.shape === :m16n8k128
                r = group + (i >= 32 ? 8 : 0)
                col = thread * 32 + (i & 31)
            else
                r = group + ((32 <= i < 64 || i >= 96) ? 8 : 0)
                col = thread * 32 + (i & 31) + (i >= 64 ? 128 : 0)
            end
            A[r + 1, col + 1] = _b1_bit(a_words, lane, row.n_a, i)
        end
        for i in 0:(32 * row.n_b - 1)
            r = thread * 32 + (i & 31) + (i >= 32 ? 128 : 0)
            B[r + 1, group + 1] = _b1_bit(b_words, lane, row.n_b, i)
        end
    end

    expected = similar(c_words)
    for lane in 0:31
        group = lane >> 2
        thread = lane & 3
        for i in 0:(row.n_cd - 1)
            r = group + (row.n_cd == 4 && i >= 2 ? 8 : 0)
            col = thread * 2 + (i & 1)
            dots = count(0:(row.k - 1)) do kk
                row.bitop === :xor ? xor(A[r + 1, kk + 1], B[kk + 1, col + 1]) :
                                     (A[r + 1, kk + 1] & B[kk + 1, col + 1])
            end
            idx = lane * row.n_cd + i + 1
            expected[idx] = c_words[idx] + Int32(dots)
        end
    end
    expected
end

@testset "single-bit mma: all six forms match popcount reference" begin
    @test length(_RUNTIME_B1_MMA_FORMS) == 6
    rng = MersenneTwister(0xb1_2026)
    for (i, row) in enumerate(_RUNTIME_B1_MMA_FORMS)
        a_words = rand(rng, UInt32, 32 * row.n_a)
        b_words = rand(rng, UInt32, 32 * row.n_b)
        c_words = rand(rng, Int32(-31):Int32(31), 32 * row.n_cd)
        expected = _b1_reference(row, a_words, b_words, c_words)

        out = CUDACore.zeros(Int32, length(expected))
        launcher = getfield(@__MODULE__, Symbol("_launch_runtime_b1_mma_", i, "!"))
        launcher(out, CuArray(a_words), CuArray(b_words), CuArray(c_words))
        CUDACore.synchronize()
        @test Array(out) == expected
    end
end

# --- dense integer ---------------------------------------------------------------
# Every A/B element is +1, so every distributed result is K regardless of
# input signedness. A separate overflowing s4 case distinguishes
# `.satfinite` from the wrapping form.

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

# --- sparse integer --------------------------------------------------------------

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
