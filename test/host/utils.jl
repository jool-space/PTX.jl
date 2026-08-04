# PTX.Utils — parse-time unrolling and closure-free tuple reduction.
# The expected values below are hand-computed from the documented
# contracts (substitution rules, pairwise tree shape), not derived from
# the implementation.

using PTX.Utils: @unroll, strided_reduce
using Random: MersenneTwister

# Unwrap the LoadError layers macro-expansion errors arrive in.
function _expansion_error(ex)
    err = try
        macroexpand(@__MODULE__, ex)
        nothing
    catch e
        e
    end
    while err isa LoadError
        err = err.error
    end
    err
end

@testset "@unroll: literal ranges" begin
    acc = Int[]
    @unroll for i in 0:3
        push!(acc, i * i)
    end
    @test acc == [0, 1, 4, 9]

    stepped = Int[]
    @unroll for i in 1:2:7
        push!(stepped, i)
    end
    @test stepped == [1, 3, 5, 7]
end

@testset "@unroll: the value is a compile-time constant" begin
    valof(::Val{N}) where {N} = N
    seen = Int[]
    @unroll for i in 2:4
        push!(seen, valof(Val(i)))   # Val(i) requires a literal i
    end
    @test seen == [2, 3, 4]
end

@testset "@unroll: `name_x` suffix mints per-iteration names" begin
    @unroll for s in 0:2
        ph_s = s + 10
    end
    # assignments splice into the enclosing scope and persist
    @test (ph_0, ph_1, ph_2) == (10, 11, 12)

    # only the literal `_s` suffix rewrites; other names pass through
    bars = 7
    total = 0
    @unroll for s in 0:1
        total += bars + s
    end
    @test total == 15
end

@testset "@unroll: tuple iterators and destructuring" begin
    inc(x) = x + 1
    dbl(x) = 2x
    out = Int[]
    @unroll for (k, fn) in ((0, inc), (1, dbl))
        r_k = fn(k)
        push!(out, r_k)
    end
    @test out == [1, 2]
    @test (r_0, r_1) == (1, 2)
end

@testset "@unroll: keyword-argument names are not substituted" begin
    kwecho(; s = 0) = s
    got = Int[]
    @unroll for s in 1:2
        push!(got, kwecho(s = s))
    end
    @test got == [1, 2]
end

@testset "@unroll: capped form" begin
    n = 3                                # not a parse-time literal
    acc = Int[]
    @unroll 8 for j in 1:n
        push!(acc, j)
    end
    @test acc == [1, 2, 3]

    # the guard splices the stop EXPRESSION: 0:(n-1) means what it says
    zero_based = Int[]
    @unroll 8 for j in 0:(n - 1)
        push!(zero_based, j)
    end
    @test zero_based == [0, 1, 2]

    # per-iteration names mint exactly for the live copies
    @unroll 4 for s in 0:(n - 2)
        cap_s = 10 + s
    end
    @test (cap_0, cap_1) == (10, 11)
    @test !@isdefined(cap_2)

    # step ranges: literal start/step, dynamic stop
    m = 5
    stepped = Int[]
    @unroll 4 for j in 0:2:m
        push!(stepped, j)
    end
    @test stepped == [0, 2, 4]
    descending = Int[]
    @unroll 4 for j in 8:-2:m
        push!(descending, j)
    end
    @test descending == [8, 6]

    # the stop expression is evaluated once
    calls = Ref(0)
    stop_once() = (calls[] += 1; 2)
    hits = Int[]
    @unroll 8 for j in 1:stop_once()
        push!(hits, j)
    end
    @test hits == [1, 2] && calls[] == 1

    # the cap is a caller-asserted ceiling: iterations beyond it are
    # silently absent (assert the bound where the values are chosen)
    over = Int[]
    @unroll 2 for j in 1:100
        push!(over, j)
    end
    @test over == [1, 2]
end

@testset "@unroll: capped-form refusals" begin
    unrolled2(cap, loop) = Expr(:macrocall, Symbol("@unroll"),
                                LineNumberNode(@__LINE__, @__FILE__), cap, loop)

    err = _expansion_error(unrolled2(:n, :(for i in 1:m; i; end)))
    @test err isa ErrorException && occursin("literal positive integer", err.msg)

    err = _expansion_error(unrolled2(4, :(for i in a:m; i; end)))
    @test err isa ErrorException && occursin("start must be a literal", err.msg)

    err = _expansion_error(unrolled2(4, :(for i in 0:0:m; i; end)))
    @test err isa ErrorException && occursin("nonzero literal", err.msg)

    err = _expansion_error(unrolled2(4, :(for (a, b) in 1:m; a; end)))
    @test err isa ErrorException && occursin("plain symbol binding", err.msg)

    err = _expansion_error(unrolled2(4, :(for i in (1, 2); i; end)))
    @test err isa ErrorException && occursin("range iterator", err.msg)
end

@testset "@unroll: refusals" begin
    unrolled(loop) = Expr(:macrocall, Symbol("@unroll"),
                          LineNumberNode(@__LINE__, @__FILE__), loop)

    err = _expansion_error(unrolled(:(for i in 1:n; i; end)))
    @test err isa ErrorException && occursin("literal range", err.msg)

    err = _expansion_error(unrolled(:(for i in [1, 2]; i; end)))
    @test err isa ErrorException && occursin("literal range", err.msg)

    err = _expansion_error(unrolled(:(while true; end)))
    @test err isa ErrorException && occursin("needs a `for` loop", err.msg)

    err = _expansion_error(unrolled(:(for i in 1:2, j in 1:2; i; end)))
    @test err isa ErrorException && occursin("exactly one induction", err.msg)

    err = _expansion_error(unrolled(:(for (a, b) in ((1, 2), 3); a; end)))
    @test err isa ErrorException && occursin("destructuring", err.msg)
end

@testset "strided_reduce: values" begin
    @test strided_reduce(+, ntuple(identity, 8), Val(2)) == (16, 20)
    @test strided_reduce(max, (3, 1, 4, 1, 5, 9, 2, 6), Val(4)) == (5, 9, 4, 6)
    # S == N: identity
    @test strided_reduce(+, (1, 2, 3), Val(3)) == (1, 2, 3)
    # S == 1: full reduction
    @test strided_reduce(*, (2, 3, 4), Val(1)) == (24,)
end

@testset "strided_reduce: the pairwise tree shape is pinned" begin
    # Float32 addition is non-associative, so exact equality against the
    # hand-written pairwise trees pins the association order — the same
    # trees the blackwell flash-attention softmax carries (32 lanes into
    # 4 strided accumulators).
    rng = MersenneTwister(42)
    w = ntuple(_ -> rand(rng, Float32) * 1.0f3, 32)
    expected_sum =
        (((w[1] + w[5]) + (w[9] + w[13])) + ((w[17] + w[21]) + (w[25] + w[29])),
         ((w[2] + w[6]) + (w[10] + w[14])) + ((w[18] + w[22]) + (w[26] + w[30])),
         ((w[3] + w[7]) + (w[11] + w[15])) + ((w[19] + w[23]) + (w[27] + w[31])),
         ((w[4] + w[8]) + (w[12] + w[16])) + ((w[20] + w[24]) + (w[28] + w[32])))
    @test strided_reduce(+, w, Val(4)) === expected_sum
    expected_max =
        (max(max(w[1], w[5], w[9]),  max(w[13], w[17]), max(w[21], w[25], w[29])),
         max(max(w[2], w[6], w[10]), max(w[14], w[18]), max(w[22], w[26], w[30])),
         max(max(w[3], w[7], w[11]), max(w[15], w[19]), max(w[23], w[27], w[31])),
         max(max(w[4], w[8], w[12]), max(w[16], w[20]), max(w[24], w[28], w[32])))
    @test strided_reduce(max, w, Val(4)) === expected_max

    # odd leaf count: the unpaired leaf carries into the next round
    v = ntuple(_ -> rand(rng, Float32) * 1.0f3, 6)
    @test strided_reduce(+, v, Val(2)) ===
          ((v[1] + v[3]) + v[5], (v[2] + v[4]) + v[6])

    # fanout 3: up-to-ternary groups; Julia's 3-arg `op` associates left,
    # so 8 leaves fold as ((l1 l2 l3), (l4 l5 l6), (l7 l8)) then the
    # 3-node root — pinned exactly, since a fanout change IS an
    # association change for floating-point +
    u = ntuple(_ -> rand(rng, Float32) * 1.0f3, 8)
    @test strided_reduce(+, u, Val(1), Val(3)) ===
          (((((u[1] + u[2]) + u[3]) + ((u[4] + u[5]) + u[6])) + (u[7] + u[8])),)
    # value-neutral for an exactly associative op
    @test strided_reduce(max, u, Val(1), Val(3)) === (maximum(u),)

    @test @inferred(strided_reduce(+, w, Val(4))) isa NTuple{4, Float32}
end

@testset "strided_reduce: refusals" begin
    @test_throws ErrorException strided_reduce(+, (1, 2, 3), Val(2))
    @test_throws ErrorException strided_reduce(+, (1, 2, 3), Val(0))
    @test_throws ErrorException strided_reduce(+, (1, 2, 3, 4), Val(2), Val(1))
end
