# Device-free evidence for all 64 classic warp-level integer mma.sp forms.
# Ordinary sparse MMA was introduced in PTX 7.1; ordered metadata in PTX 8.5;
# every form has an sm_80 target floor.

using PTX: Operation

const _PTXAS_INTEGER_SP_FORMS = let rows = NamedTuple[]
    for (shape, types, n_a, n_b, selectors, packed_one) in (
            (:m16n8k32,  (:u8, :s8), 2, 2, 0:1, UInt32(0x01010101)),
            (:m16n8k64,  (:u8, :s8), 4, 4, 0:0, UInt32(0x01010101)),
            (:m16n8k64,  (:u4, :s4), 2, 2, 0:1, UInt32(0x11111111)),
            (:m16n8k128, (:u4, :s4), 4, 4, 0:0, UInt32(0x11111111)))
        for a in types, b in types, satfinite in (false, true),
                ordered in (false, true)
            variant = ordered ? Symbol("sp::ordered_metadata") : :sp
            sat = satfinite ? (:satfinite,) : ()
            mods = (variant, :sync, :aligned, shape, :row, :col, sat...,
                    :s32, a, b, :s32)
            spelling = join(("mma", String.(mods)...), '.')
            push!(rows, (; mods, spelling, n_a, n_b,
                         selector=last(selectors), packed_one))
        end
    end
    Tuple(rows)
end

let calls = Expr(:block)
    for (i, row) in enumerate(_PTXAS_INTEGER_SP_FORMS)
        helper = Symbol("_ptxas_integer_sp_", i, "!")
        op = Operation{:mma, row.mods}()
        @eval @inline function $helper(out)
            a = ntuple(_ -> $(row.packed_one), Val($(row.n_a)))
            b = ntuple(_ -> $(row.packed_one), Val($(row.n_b)))
            c = ntuple(_ -> Int32(0), Val(4))
            d = $op(a, b, c, UInt32(0x44444444), Val($(row.selector)))
            @inbounds out[$i] = d[1]
            nothing
        end
        push!(calls.args, :($helper(out)))
    end
    @eval function _ptxas_integer_sp_all!(out)
        $calls
        nothing
    end
end

const _PTXAS_INTEGER_SP_TYPES = Tuple{CuDeviceVector{Int32, 1}}

@testset "integer mma.sp: all 64 forms at sm_80" begin
    @test length(_PTXAS_INTEGER_SP_FORMS) == 64
    @test ptxas_compiles(_ptxas_integer_sp_all!, _PTXAS_INTEGER_SP_TYPES;
                         cap = v"8.0")
    ptx = emit_ptx(_ptxas_integer_sp_all!, _PTXAS_INTEGER_SP_TYPES;
                   cap = v"8.0")
    @test occursin(".target sm_80", ptx)
    @test count("mma.sp", ptx) == 64
    for row in _PTXAS_INTEGER_SP_FORMS
        @test occursin(row.spelling, ptx)
    end

    llvm = emit_llvm(_ptxas_integer_sp_all!, _PTXAS_INTEGER_SP_TYPES;
                     cap = v"8.0")
    @test occursin("llvm.nvvm.mma.sp.m16", llvm)
    @test occursin("llvm.nvvm.mma.sp.ordered.metadata.m16", llvm)
    @test occursin("convergent nomerge", llvm)
end

@testset "integer mma.sp: sm_80 floor rejects sm_75" begin
    rejected = try
        ptxas_compiles(_ptxas_integer_sp_all!, _PTXAS_INTEGER_SP_TYPES;
                       cap = v"7.5")
        false
    catch
        true
    end
    @test rejected
end
