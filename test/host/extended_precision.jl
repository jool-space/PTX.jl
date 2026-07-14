using InteractiveUtils: code_llvm
using PTX: Operation

# Independent ISA ledger: do not derive this from the wrapper registration
# table, so either side changing forces an explicit review of all 48 forms.
const _CC_DTYPES = ((:u32, UInt32), (:s32, Int32),
                    (:u64, UInt64), (:s64, Int64))

function _implicit_cc_ledger()
    forms = Tuple{Symbol, Tuple{Vararg{Symbol}}, Type}[]
    for (dt, T) in _CC_DTYPES
        push!(forms, (:add,  (:cc, dt), T))
        push!(forms, (:addc, (dt,), T), (:addc, (:cc, dt), T))
        push!(forms, (:sub,  (:cc, dt), T))
        push!(forms, (:subc, (dt,), T), (:subc, (:cc, dt), T))
        for half in (:lo, :hi)
            push!(forms, (:mad,  (half, :cc, dt), T))
            push!(forms, (:madc, (half, dt), T),
                         (:madc, (half, :cc, dt), T))
        end
    end
    forms
end

const _IMPLICIT_CC_FORMS = _implicit_cc_ledger()

# Julia 1.10 cannot infer these @generated methods directly through
# `Base.return_types`, even with concrete tuple arguments. Route inference
# through concretely typed callers so the declared minimum Julia remains part
# of the oracle rather than a test-harness accident.
_add_with_carry_type_probe(a::NTuple{4,UInt32}, b::NTuple{4,UInt32}) =
    PTX.add_with_carry(a, b)
_sub_with_borrow_type_probe(a::NTuple{2,Int64}, b::NTuple{2,Int64},
                            borrow::Bool) =
    PTX.sub_with_borrow(a, b, borrow)
_mul_wide_type_probe(a::NTuple{2,UInt64}, b::NTuple{2,UInt64}) =
    PTX.mul_wide(a, b)

function _cc_signature(op::Symbol, mods::Tuple, T::Type)
    value_inputs = op in (:mad, :madc) ? 3 : 2
    carry_in = op in (:addc, :subc, :madc)
    carry_out = :cc in mods
    argtypes = Tuple{ntuple(_ -> T, value_inputs)...,
                     (carry_in ? (Bool,) : ())...}
    rettype = carry_out ? Tuple{T, Bool} : T
    (; argtypes, rettype)
end

@testset "implicit CC.CF inventory and fail-loud boundary" begin
    @test length(_IMPLICIT_CC_FORMS) == 48
    @test length(unique(_IMPLICIT_CC_FORMS)) == 48
    @test count(x -> x[1] == :add,  _IMPLICIT_CC_FORMS) == 4
    @test count(x -> x[1] == :addc, _IMPLICIT_CC_FORMS) == 8
    @test count(x -> x[1] == :sub,  _IMPLICIT_CC_FORMS) == 4
    @test count(x -> x[1] == :subc, _IMPLICIT_CC_FORMS) == 8
    @test count(x -> x[1] == :mad,  _IMPLICIT_CC_FORMS) == 8
    @test count(x -> x[1] == :madc, _IMPLICIT_CC_FORMS) == 16

    expected = Set((op, mods) for (op, mods, _) in _IMPLICIT_CC_FORMS)
    @test Set(PTX.EXTENDED_PRECISION_WRAPPER_FORMS) == expected
    @test length(PTX.EXTENDED_PRECISION_WRAPPER_FORMS) == 48

    for (op, mods, T) in _IMPLICIT_CC_FORMS
        @test PTX.uses_implicit_cc(op, mods)
        @test PTX.form_contract(op, mods) === nothing

        sig = _cc_signature(op, mods, T)
        typed = Operation{op, mods}()
        raw = PTX.RawOperation{op, mods}()

        # Every reviewed spelling has an exact typed wrapper. Generic/raw
        # fallback remains forbidden even when a caller supplies RAW_CONTRACT.
        @test which(typed, sig.argtypes).module == PTX
        info = PTX.lowering(typed, sig.argtypes)
        @test info.tier === :asm
        @test info.rettype === sig.rettype
        ci, _ = first(Base.code_typed(typed, sig.argtypes))
        lowering_text = string(ci)
        @test occursin("sideeffect", lowering_text)
        @test occursin("~{cc}", lowering_text)
        @test !occursin("~{memory}", lowering_text)
        @test !occursin("convergent", lowering_text)
        @test PTX.lowering(raw, sig.argtypes).tier === :forbidden
        @test_throws ArgumentError PTX.build_call(op, mods,
            Tuple(sig.argtypes.parameters); contract = PTX.RAW_CONTRACT)
        @test_throws ArgumentError PTX.format_call(typed, sig.argtypes)
    end

    err = try
        PTX.build_call(:add, (:cc, :u32), (UInt32, UInt32)); ""
    catch ex
        sprint(showerror, ex)
    end
    @test occursin("CC.CF", err)
    @test occursin("raw tier", err)
    @test occursin("add_with_carry", err)

    # Future/malformed CC spellings must not leak back through opcode-wide
    # pure defaults, while ordinary neighbors remain exactly as before.
    @test PTX.uses_implicit_cc(:add, (:sat, :cc, :s32))
    @test PTX.uses_implicit_cc(:madc, (:nonsense,))
    for (op, mods) in ((:add, (:f32,)), (:add, (:sat, :s32)),
                       (:sub, (:u32,)), (:sub, (:sat, :s32)),
                       (:mad, (:lo, :s32)), (:mad, (:hi, :sat, :s32)),
                       (:mad24, (:lo, :u32)), (:clmad, (:lo, :u64)))
        @test !PTX.uses_implicit_cc(op, mods)
        @test PTX.form_contract(op, mods).pure
    end
end

@testset "scalar carry wrappers expose CC.CF as Bool" begin
    addc = PTX._scalar_extended_spec("addc.cc.u32", UInt32, 2;
                                     carry_in = true, carry_out = true)
    @test addc.asm == "{ .reg .u32 cc_tmp; " *
        "selp.u32 cc_tmp, 1, 0, \$4; " *
        "add.cc.u32 cc_tmp, cc_tmp, -1; " *
        "addc.cc.u32 \$0, \$2, \$3; " *
        "addc.u32 cc_tmp, 0, 0; setp.ne.u32 \$1, cc_tmp, 0; }"
    @test addc.constraints == "=&r,=b,r,r,b,~{cc}"
    @test addc.rettype === Tuple{UInt32, Bool}
    @test addc.argtype === Tuple{UInt32, UInt32, Bool}
    @test !occursin("memory", addc.constraints)

    mad = PTX._scalar_extended_spec("mad.hi.cc.s64", Int64, 3;
                                    carry_in = false, carry_out = true)
    @test occursin("mad.hi.cc.s64 \$0, \$2, \$3, \$4", mad.asm)
    @test mad.constraints == "=&l,=b,l,l,l,~{cc}"

    # Missing the explicit carry input cannot silently select the generic/raw
    # path: dispatch reaches the forbidden generator and fails before LLVM.
    @test_throws Exception ptx"addc.u32"(UInt32(1), UInt32(2))
    @test_throws Exception ptx"madc.lo.u64"(UInt64(1), UInt64(2), UInt64(3))
end

@testset "fused add/sub specifications are one early-clobber block" begin
    add = PTX._addsub_aggregate_spec(:add, 3, UInt32, false)
    @test add.asm == "{ .reg .u32 cc_tmp; " *
        "add.cc.u32 \$0, \$4, \$7; " *
        "addc.cc.u32 \$1, \$5, \$8; " *
        "addc.cc.u32 \$2, \$6, \$9; " *
        "addc.u32 cc_tmp, 0, 0; setp.ne.u32 \$3, cc_tmp, 0; }"
    @test add.constraints == "=&r,=&r,=&r,=b,r,r,r,r,r,r,~{cc}"
    @test add.rettype === Tuple{UInt32, UInt32, UInt32, Bool}
    @test add.argtype === NTuple{6, UInt32}

    add_in = PTX._addsub_aggregate_spec(:add, 2, UInt64, true)
    @test startswith(add_in.asm, "{ .reg .u64 cc_tmp; " *
        "selp.u64 cc_tmp, 1, 0, \$7; add.cc.u64 cc_tmp, cc_tmp, -1; " *
        "addc.cc.u64 \$0, \$3, \$5;")
    @test add_in.constraints == "=&l,=&l,=b,l,l,l,l,b,~{cc}"
    @test add_in.argtype === Tuple{UInt64, UInt64, UInt64, UInt64, Bool}

    sub = PTX._addsub_aggregate_spec(:sub, 2, Int32, true)
    @test occursin("sub.cc.s32 cc_tmp, 0, cc_tmp", sub.asm)
    @test occursin("subc.cc.s32 \$0, \$3, \$5", sub.asm)
    @test occursin("subc.cc.s32 \$1, \$4, \$6", sub.asm)
    @test occursin("subc.s32 cc_tmp, 0, 0", sub.asm)
    @test_throws ArgumentError PTX._addsub_aggregate_spec(:add, 0, UInt32, false)

    _, add_rt = first(Base.code_typed(_add_with_carry_type_probe,
        (NTuple{4,UInt32}, NTuple{4,UInt32})))
    @test add_rt === Tuple{NTuple{4,UInt32}, Bool}
    _, sub_rt = first(Base.code_typed(_sub_with_borrow_type_probe,
        (NTuple{2,Int64}, NTuple{2,Int64}, Bool)))
    @test sub_rt === Tuple{NTuple{2,Int64}, Bool}
end

@testset "ISA-derived unsigned 2x2 mul_wide sequence" begin
    u32 = PTX._mul_wide_spec(UInt32)
    @test u32.asm == "{ " *
        "mul.lo.u32 \$0, \$4, \$6; mul.hi.u32 \$1, \$4, \$6; " *
        "mad.lo.cc.u32 \$1, \$5, \$6, \$1; " *
        "madc.hi.u32 \$2, \$5, \$6, 0; " *
        "mad.lo.cc.u32 \$1, \$4, \$7, \$1; " *
        "madc.hi.cc.u32 \$2, \$4, \$7, \$2; " *
        "addc.u32 \$3, 0, 0; " *
        "mad.lo.cc.u32 \$2, \$5, \$7, \$2; " *
        "madc.hi.u32 \$3, \$5, \$7, \$3; }"
    @test u32.constraints == "=&r,=&r,=&r,=&r,r,r,r,r,~{cc}"
    @test PTX._mul_wide_spec(UInt64).constraints ==
        "=&l,=&l,=&l,=&l,l,l,l,l,~{cc}"
    _, mul_rt = first(Base.code_typed(_mul_wide_type_probe,
        (NTuple{2,UInt64}, NTuple{2,UInt64})))
    @test mul_rt === NTuple{4,UInt64}
    @test !hasmethod(PTX.mul_wide, Tuple{NTuple{2,Int32}, NTuple{2,Int32}})
end

@noinline _fused_add_probe(a::NTuple{3,UInt32}, b::NTuple{3,UInt32}) =
    PTX.add_with_carry(a, b)

@noinline function _explicit_carry_probe(a::UInt32, b::UInt32,
                                         x::UInt32, y::UInt32)
    _, carry = ptx"add.cc.u32"(a, b)  # arithmetic result deliberately unused
    ptx"addc.u32"(x, y, carry)
end

@testset "optimized LLVM retains opaque, non-convergent carry units" begin
    ir = sprint(io -> code_llvm(io, _fused_add_probe,
        (NTuple{3,UInt32}, NTuple{3,UInt32}); debuginfo = :none))
    @test length(collect(eachmatch(r"asm sideeffect", ir))) == 1
    @test occursin("add.cc.u32", ir)
    @test length(collect(eachmatch(r"addc.cc.u32", ir))) == 2
    @test occursin("=&r,=&r,=&r,=b,r,r,r,r,r,r,~{cc}", ir)
    @test !occursin("convergent", ir)

    scalar_ir = sprint(io -> code_llvm(io, _explicit_carry_probe,
        (UInt32, UInt32, UInt32, UInt32); debuginfo = :none))
    @test length(collect(eachmatch(r"asm sideeffect", scalar_ir))) == 2
    @test findfirst("add.cc.u32", scalar_ir) < findlast("addc.u32", scalar_ir)
    @test !occursin("convergent", scalar_ir)

    mul_ir = sprint(io -> code_llvm(io, PTX.mul_wide,
        (NTuple{2,UInt32}, NTuple{2,UInt32}); debuginfo = :none))
    @test length(collect(eachmatch(r"asm sideeffect", mul_ir))) == 1
    @test occursin("madc.hi.cc.u32", mul_ir)
    @test !occursin("convergent", mul_ir)
end

function _cc_transpile_module(instruction::String)
    """.version 8.0
    .target sm_80
    .address_size 64
    .visible .entry carry_probe()
    {
        .reg .b32 %r<8>;
        .reg .b64 %rd<8>;
        $instruction
        ret;
    }
    """
end

@testset "transpiler rejects instruction-at-a-time CC.CF lowering" begin
    for instruction in (
        "add.cc.u32 %r0, %r1, %r2;",
        "addc.u32 %r0, %r1, %r2;",
        "sub.cc.u64 %rd0, %rd1, %rd2;",
        "subc.cc.u32 %r0, %r1, %r2;",
        "mad.lo.cc.u32 %r0, %r1, %r2, %r3;",
        "madc.hi.u64 %rd0, %rd1, %rd2, %rd3;",
    )
        err = try
            PTX.ptx_to_julia(_cc_transpile_module(instruction)); ""
        catch ex
            sprint(showerror, ex)
        end
        @test occursin("CC.CF", err)
        @test occursin("cannot cross", err)
    end

    # Keep the guard ahead of shared-pointer alias propagation. If it moves
    # below `_try_alias_def!`, this add.cc producer can be mistaken for pointer
    # arithmetic, recorded as an alias, and silently erased.
    alias_src = """
    .version 8.0
    .target sm_80
    .address_size 64
    .visible .entry carry_alias_probe()
    {
        .reg .b64 %rd<3>;
        .shared .b8 smem[16];
        mov.u64 %rd1, smem;
        add.cc.u64 %rd2, %rd1, 4;
        ret;
    }
    """
    alias_err = try
        PTX.ptx_to_julia(alias_src); ""
    catch ex
        sprint(showerror, ex)
    end
    @test occursin("CC.CF", alias_err)
    @test occursin("cannot cross", alias_err)
end
