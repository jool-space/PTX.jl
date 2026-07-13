# Cross-target assembly validation for the optimizer-safe CC.CF surface.  All
# forms used here are available by sm_20; sm_75 is the oldest target retained by
# CUDA 13 ptxas in this test environment.

# Independent 48-form scalar ledger. Compile all of it in one kernel so the
# signed variants and every hi/lo/.cc/no-.cc branch reach ptxas directly; the
# general compile-touch sweep is a second, method-table-derived oracle.
function _ptxas_cc_ledger()
    forms = Tuple{Symbol, Tuple{Vararg{Symbol}}, Type}[]
    for (dt, T) in ((:u32, UInt32), (:s32, Int32),
                    (:u64, UInt64), (:s64, Int64))
        push!(forms, (:add, (:cc, dt), T))
        push!(forms, (:addc, (dt,), T), (:addc, (:cc, dt), T))
        push!(forms, (:sub, (:cc, dt), T))
        push!(forms, (:subc, (dt,), T), (:subc, (:cc, dt), T))
        for half in (:lo, :hi)
            push!(forms, (:mad, (half, :cc, dt), T))
            push!(forms, (:madc, (half, dt), T),
                         (:madc, (half, :cc, dt), T))
        end
    end
    forms
end

const _PTXAS_CC_FORMS = _ptxas_cc_ledger()
let calls = Expr[]
    for (op, mods, T) in _PTXAS_CC_FORMS
        value_inputs = op in (:mad, :madc) ? 3 : 2
        args = Any[QuoteNode(T(i)) for i in 1:value_inputs]
        op in (:addc, :subc, :madc) && push!(args, false)
        callee = :(PTX.Operation{$(QuoteNode(op)), $mods}())
        push!(calls, Expr(:call, callee, args...))
    end
    @eval function _all_extended_scalar_forms!()
        $(calls...)
        return nothing
    end
end

function _extended_precision_compile!(out32::CuDeviceVector{UInt32,1},
                                      out64::CuDeviceVector{UInt64,1},
                                      a::UInt32, b::UInt32,
                                      c::UInt32, d::UInt32)
    x = (a, b, c, d)
    y = (d, c, b, a)
    sum, carry = PTX.add_with_carry(x, y)
    sum_in, carry_in = PTX.add_with_carry(x, y, true)
    diff, borrow = PTX.sub_with_borrow(x, y)
    diff_in, borrow_in = PTX.sub_with_borrow(x, y, true)

    s0, sc = ptx"add.cc.u32"(a, b)
    s1, sc = ptx"addc.cc.u32"(c, d, sc)
    s2 = ptx"addc.u32"(a, d, sc)
    t0, sb = ptx"sub.cc.u32"(a, b)
    t1, sb = ptx"subc.cc.u32"(c, d, sb)
    t2 = ptx"subc.u32"(a, d, sb)
    m0, mc = ptx"mad.lo.cc.u32"(a, b, c)
    m1, mc = ptx"madc.hi.cc.u32"(a, b, d, mc)
    m2 = ptx"madc.lo.u32"(c, d, m0, mc)
    prod32 = PTX.mul_wide((a, b), (c, d))

    a64, b64 = UInt64(a) << 32 | UInt64(b), UInt64(c) << 32 | UInt64(d)
    wide64, carry64 = PTX.add_with_carry((a64, b64), (b64, a64))
    prod64 = PTX.mul_wide((a64, b64), (b64, a64))

    @inbounds begin
        for i in 1:4
            out32[i] = sum[i]
            out32[4 + i] = sum_in[i]
            out32[8 + i] = diff[i]
            out32[12 + i] = diff_in[i]
            out32[22 + i] = prod32[i]
            out64[i] = prod64[i]
        end
        out32[17] = UInt32(carry)
        out32[18] = UInt32(carry_in)
        out32[19] = UInt32(borrow)
        out32[20] = UInt32(borrow_in)
        out32[21] = s0 ⊻ s1 ⊻ s2 ⊻ t0 ⊻ t1 ⊻ t2 ⊻ m0 ⊻ m1 ⊻ m2
        out64[5] = wide64[1]
        out64[6] = wide64[2]
        out64[7] = UInt64(carry64)
    end
    return nothing
end

@testset "extended precision assembles at sm_75" begin
    types = Tuple{CuDeviceVector{UInt32,1}, CuDeviceVector{UInt64,1},
                  UInt32, UInt32, UInt32, UInt32}
    @test ptxas_compiles(_extended_precision_compile!, types;
                         cap = v"7.5", feature_set = :baseline)
    ptx = emit_ptx(_extended_precision_compile!, types;
                   cap = v"7.5", feature_set = :baseline)
    @test occursin(".target sm_75", ptx)
    for mnemonic in ("add.cc.u32", "addc.cc.u32", "addc.u32",
                     "sub.cc.u32", "subc.cc.u32", "subc.u32",
                     "mad.lo.cc.u32", "madc.hi.cc.u32", "madc.lo.u32",
                     "add.cc.u64", "addc.cc.u64",
                     "mad.lo.cc.u64", "madc.hi.cc.u64")
        @test occursin(mnemonic, ptx)
    end
end

@testset "all 48 scalar CC.CF forms assemble" begin
    @test length(_PTXAS_CC_FORMS) == 48
    @test length(unique(_PTXAS_CC_FORMS)) == 48
    @test ptxas_compiles(_all_extended_scalar_forms!, Tuple{};
                         cap = v"7.5", feature_set = :baseline)
    ptx = emit_ptx(_all_extended_scalar_forms!, Tuple{};
                   cap = v"7.5", feature_set = :baseline)
    for (op, mods, _) in _PTXAS_CC_FORMS
        spelling = join((String(op), String.(mods)...), ".")
        @test occursin(spelling, ptx)
    end
end
