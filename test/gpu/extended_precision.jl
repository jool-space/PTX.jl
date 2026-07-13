# TEST_TARGET: requires=gpu evidence=runtime target=sm_70

using Random

function _extended_addsub_runtime!(sum_out, diff_out, sum_in_out, diff_in_out,
                                   scalar_sum_out, scalar_diff_out,
                                   a_words, b_words, n::Int)
    tid = Int(ptx"mov.u32"(sreg"tid.x"))
    bid = Int(ptx"mov.u32"(sreg"ctaid.x"))
    nt = Int(ptx"mov.u32"(sreg"ntid.x"))
    i = bid * nt + tid + 1
    i > n && return nothing
    off = 4 * (i - 1)
    a = @inbounds (a_words[off + 1], a_words[off + 2],
                   a_words[off + 3], a_words[off + 4])
    b = @inbounds (b_words[off + 1], b_words[off + 2],
                   b_words[off + 3], b_words[off + 4])

    sum, carry = PTX.add_with_carry(a, b)
    diff, borrow = PTX.sub_with_borrow(a, b)
    sum_in, carry_in = PTX.add_with_carry(a, b, true)
    diff_in, borrow_in = PTX.sub_with_borrow(a, b, true)

    s0, sc = ptx"add.cc.u32"(a[1], b[1])
    s1, sc = ptx"addc.cc.u32"(a[2], b[2], sc)
    s2, sc = ptx"addc.cc.u32"(a[3], b[3], sc)
    s3, sc = ptx"addc.cc.u32"(a[4], b[4], sc)

    d0, sb = ptx"sub.cc.u32"(a[1], b[1])
    d1, sb = ptx"subc.cc.u32"(a[2], b[2], sb)
    d2, sb = ptx"subc.cc.u32"(a[3], b[3], sb)
    d3, sb = ptx"subc.cc.u32"(a[4], b[4], sb)

    out = 5 * (i - 1)
    @inbounds begin
        for j in 1:4
            sum_out[out + j] = sum[j]
            diff_out[out + j] = diff[j]
            sum_in_out[out + j] = sum_in[j]
            diff_in_out[out + j] = diff_in[j]
        end
        sum_out[out + 5] = UInt32(carry)
        diff_out[out + 5] = UInt32(borrow)
        sum_in_out[out + 5] = UInt32(carry_in)
        diff_in_out[out + 5] = UInt32(borrow_in)
        scalar_sum_out[out + 1] = s0
        scalar_sum_out[out + 2] = s1
        scalar_sum_out[out + 3] = s2
        scalar_sum_out[out + 4] = s3
        scalar_sum_out[out + 5] = UInt32(sc)
        scalar_diff_out[out + 1] = d0
        scalar_diff_out[out + 2] = d1
        scalar_diff_out[out + 3] = d2
        scalar_diff_out[out + 4] = d3
        scalar_diff_out[out + 5] = UInt32(sb)
    end
    return nothing
end

function _extended_mul32_runtime!(fused_out, scalar_out, a_words, b_words, n::Int)
    tid = Int(ptx"mov.u32"(sreg"tid.x"))
    bid = Int(ptx"mov.u32"(sreg"ctaid.x"))
    nt = Int(ptx"mov.u32"(sreg"ntid.x"))
    i = bid * nt + tid + 1
    i > n && return nothing
    off = 2 * (i - 1)
    a = @inbounds (a_words[off + 1], a_words[off + 2])
    b = @inbounds (b_words[off + 1], b_words[off + 2])
    product = PTX.mul_wide(a, b)

    # Same ISA-documented u32 sequence through the explicit-Bool scalar
    # surface. Each call contains its own seed/materialization block, so no
    # implicit CC.CF state crosses the Julia/LLVM boundary.
    r0 = ptx"mul.lo.u32"(a[1], b[1])
    r1 = ptx"mul.hi.u32"(a[1], b[1])
    r1, carry = ptx"mad.lo.cc.u32"(a[2], b[1], r1)
    r2 = ptx"madc.hi.u32"(a[2], b[1], UInt32(0), carry)
    r1, carry = ptx"mad.lo.cc.u32"(a[1], b[2], r1)
    r2, carry = ptx"madc.hi.cc.u32"(a[1], b[2], r2, carry)
    r3 = ptx"addc.u32"(UInt32(0), UInt32(0), carry)
    r2, carry = ptx"mad.lo.cc.u32"(a[2], b[2], r2)
    r3 = ptx"madc.hi.u32"(a[2], b[2], r3, carry)

    out = 4 * (i - 1)
    @inbounds for j in 1:4
        fused_out[out + j] = product[j]
    end
    @inbounds begin
        scalar_out[out + 1] = r0
        scalar_out[out + 2] = r1
        scalar_out[out + 3] = r2
        scalar_out[out + 4] = r3
    end
    return nothing
end

function _extended_mul64_runtime!(out_words, a_words, b_words, n::Int)
    tid = Int(ptx"mov.u32"(sreg"tid.x"))
    bid = Int(ptx"mov.u32"(sreg"ctaid.x"))
    nt = Int(ptx"mov.u32"(sreg"ntid.x"))
    i = bid * nt + tid + 1
    i > n && return nothing
    off = 2 * (i - 1)
    a = @inbounds (a_words[off + 1], a_words[off + 2])
    b = @inbounds (b_words[off + 1], b_words[off + 2])
    product = PTX.mul_wide(a, b)
    out = 4 * (i - 1)
    @inbounds for j in 1:4
        out_words[out + j] = product[j]
    end
    return nothing
end

_words_value(words, bits) = sum(BigInt(w) << (bits * (i - 1))
                                 for (i, w) in enumerate(words))

function _value_words(value::BigInt, ::Type{T}, count::Int) where T<:Unsigned
    bits = 8sizeof(T)
    mask = (big(1) << bits) - 1
    T[(value >> (bits * i)) & mask for i in 0:count-1]
end

_flatten_words(cases) = reduce(vcat, (collect(x) for x in cases))

@testset "randomized 128-bit add/sub and explicit carry chains" begin
    rng = MersenneTwister(0xCCCF)
    cases_a = NTuple{4,UInt32}[
        (0, 0, 0, 0),
        (typemax(UInt32), typemax(UInt32), typemax(UInt32), typemax(UInt32)),
        (0, 0, 0, 0),
        (typemax(UInt32), 0, typemax(UInt32), 0),
    ]
    cases_b = NTuple{4,UInt32}[
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 0, 0),
        (1, typemax(UInt32), 1, typemax(UInt32)),
    ]
    for _ in 1:256
        push!(cases_a, ntuple(_ -> rand(rng, UInt32), 4))
        push!(cases_b, ntuple(_ -> rand(rng, UInt32), 4))
    end
    n = length(cases_a)
    a_d, b_d = CuArray(_flatten_words(cases_a)), CuArray(_flatten_words(cases_b))
    sum_d = CUDACore.zeros(UInt32, 5n)
    diff_d = CUDACore.zeros(UInt32, 5n)
    sum_in_d = CUDACore.zeros(UInt32, 5n)
    diff_in_d = CUDACore.zeros(UInt32, 5n)
    scalar_sum_d = CUDACore.zeros(UInt32, 5n)
    scalar_diff_d = CUDACore.zeros(UInt32, 5n)
    threads = min(n, 256)
    @cuda threads=threads blocks=cld(n, threads) _extended_addsub_runtime!(
        sum_d, diff_d, sum_in_d, diff_in_d,
        scalar_sum_d, scalar_diff_d, a_d, b_d, n)
    CUDACore.synchronize()

    modulus = big(1) << 128
    expected_sum = UInt32[]
    expected_diff = UInt32[]
    expected_sum_in = UInt32[]
    expected_diff_in = UInt32[]
    for (a, b) in zip(cases_a, cases_b)
        av, bv = _words_value(a, 32), _words_value(b, 32)
        total = av + bv
        append!(expected_sum, _value_words(mod(total, modulus), UInt32, 4))
        push!(expected_sum, UInt32(total >= modulus))
        append!(expected_diff, _value_words(mod(av - bv, modulus), UInt32, 4))
        push!(expected_diff, UInt32(av < bv))
        total_in = total + 1
        append!(expected_sum_in,
                _value_words(mod(total_in, modulus), UInt32, 4))
        push!(expected_sum_in, UInt32(total_in >= modulus))
        append!(expected_diff_in,
                _value_words(mod(av - bv - 1, modulus), UInt32, 4))
        push!(expected_diff_in, UInt32(av < bv + 1))
    end
    @test Array(sum_d) == expected_sum
    @test Array(diff_d) == expected_diff
    @test Array(sum_in_d) == expected_sum_in
    @test Array(diff_in_d) == expected_diff_in
    @test Array(scalar_sum_d) == expected_sum
    @test Array(scalar_diff_d) == expected_diff
end

@testset "randomized unsigned wide multiply" begin
    rng = MersenneTwister(0x9_7_2_6)
    a32 = NTuple{2,UInt32}[(0, 0), (typemax(UInt32), typemax(UInt32)),
                           (typemax(UInt32), 0)]
    b32 = NTuple{2,UInt32}[(0, 0), (typemax(UInt32), typemax(UInt32)),
                           (typemax(UInt32), typemax(UInt32))]
    for _ in 1:256
        push!(a32, ntuple(_ -> rand(rng, UInt32), 2))
        push!(b32, ntuple(_ -> rand(rng, UInt32), 2))
    end
    n32 = length(a32)
    fused32_d = CUDACore.zeros(UInt32, 4n32)
    scalar32_d = CUDACore.zeros(UInt32, 4n32)
    threads = min(n32, 256)
    @cuda threads=threads blocks=cld(n32, threads) _extended_mul32_runtime!(
        fused32_d, scalar32_d, CuArray(_flatten_words(a32)),
        CuArray(_flatten_words(b32)), n32)
    CUDACore.synchronize()
    expected32 = UInt32[]
    for (a, b) in zip(a32, b32)
        product = _words_value(a, 32) * _words_value(b, 32)
        append!(expected32, _value_words(product, UInt32, 4))
    end
    @test Array(fused32_d) == expected32
    @test Array(scalar32_d) == expected32

    a64 = NTuple{2,UInt64}[(0, 0), (typemax(UInt64), typemax(UInt64)),
                           (typemax(UInt64), 0)]
    b64 = NTuple{2,UInt64}[(0, 0), (typemax(UInt64), typemax(UInt64)),
                           (typemax(UInt64), typemax(UInt64))]
    for _ in 1:128
        push!(a64, ntuple(_ -> rand(rng, UInt64), 2))
        push!(b64, ntuple(_ -> rand(rng, UInt64), 2))
    end
    n64 = length(a64)
    out64_d = CUDACore.zeros(UInt64, 4n64)
    threads = min(n64, 256)
    @cuda threads=threads blocks=cld(n64, threads) _extended_mul64_runtime!(
        out64_d, CuArray(_flatten_words(a64)), CuArray(_flatten_words(b64)), n64)
    CUDACore.synchronize()
    expected64 = UInt64[]
    for (a, b) in zip(a64, b64)
        product = _words_value(a, 64) * _words_value(b, 64)
        append!(expected64, _value_words(product, UInt64, 4))
    end
    @test Array(out64_d) == expected64
end
