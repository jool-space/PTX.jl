# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=8.0
#
# Ada/forward-compatible semantic evidence for all six PTX 9.3 single-bit
# MMA forms. Random per-lane fragments are decoded with the normative maps
# from Figures 62–64 and 97–103, then checked against a host XOR/AND+popcount
# reference. The cc>=8.0 gate is the common floor of the complete product.

using PTX: Operation
using Random: MersenneTwister

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
