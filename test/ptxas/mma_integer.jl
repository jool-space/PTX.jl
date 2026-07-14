# Exact-floor offline evidence for the 32 modern dense integer MMA forms.
# PTX 9.3 §9.7.15.5.14 introduced every m16n8 u8/s8 and u4/s4 form in
# PTX 7.0 with an sm_80 target floor.

using PTX: Operation

const _PTXAS_INTEGER_MMA_FORMS = let rows = NamedTuple[]
    for (shape, types, n_a, n_b) in (
            (:m16n8k16, (:u8, :s8), 2, 1),
            (:m16n8k32, (:u8, :s8), 4, 2),
            (:m16n8k32, (:u4, :s4), 2, 1),
            (:m16n8k64, (:u4, :s4), 4, 2))
        for a in types, b in types, satfinite in (false, true)
            sat = satfinite ? (:satfinite,) : ()
            mods = (:sync, :aligned, shape, :row, :col, sat...,
                    :s32, a, b, :s32)
            spelling = join(("mma", String.(mods)...), '.')
            push!(rows, (; mods, spelling, n_a, n_b))
        end
    end
    Tuple(rows)
end

# Generate one statically named helper per form, then one kernel that calls
# all helpers.  This keeps exact-floor evidence to a single GPUCompiler/ptxas
# job while ensuring each wrapper's Julia-side marshaling reaches ISel.
let calls = Expr(:block)
    for (i, row) in enumerate(_PTXAS_INTEGER_MMA_FORMS)
        helper = Symbol("_ptxas_integer_mma_", i, "!")
        op = Operation{:mma, row.mods}()
        @eval @inline function $helper(out)
            a = ntuple(_ -> UInt32(0x01010101), Val($(row.n_a)))
            b = ntuple(_ -> UInt32(0x01010101), Val($(row.n_b)))
            c = ntuple(_ -> Int32(0), Val(4))
            d = $op(a, b, c)
            @inbounds out[$i] = d[1]
            nothing
        end
        push!(calls.args, :($helper(out)))
    end
    @eval function _ptxas_integer_mma_all!(out)
        $calls
        nothing
    end
end

@testset "modern dense integer mma at sm_80" begin
    @test length(_PTXAS_INTEGER_MMA_FORMS) == 32
    types = Tuple{CuDeviceVector{Int32, 1}}
    @test ptxas_compiles(_ptxas_integer_mma_all!, types; cap = v"8.0")
    ptx = emit_ptx(_ptxas_integer_mma_all!, types; cap = v"8.0")
    @test occursin(".target sm_80", ptx)
    for row in _PTXAS_INTEGER_MMA_FORMS
        @test occursin(row.spelling, ptx)
    end
end
