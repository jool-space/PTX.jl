# spcompress / spdecompress (PTX 9.4 §9.7.10.30–31, sm_107a): independent
# enumeration of the admitted qualifier tuples from the ISA's sizing tables
# and validity conditions, checked against the wrapper registry, plus
# lowered-code assertions for the pure, early-clobber register contract and
# the grouped return ABI. Assembler evidence lives in ptxas/spcompress.jl;
# runtime evidence needs a CC 10.7 device.

using PTX: Operation, spcompress_desc, spcompress_elemsize

const _SP_NUMS = (1, 2, 4, 8, 16, 32, 64)
_sp_width(s) = Dict(:b8 => 8, :b16 => 16, :b2 => 2, :b4 => 4)[s]

# spcompress: every elemsize × idxsize × num (Table 37 sizes; the 253
# register cap is never reached).
function _spcompress_expected()
    forms = Dict{Tuple{Vararg{Symbol}}, NTuple{3, Int}}()
    for elem in (:b8, :b16), idx in (:b2, :b4), num in _SP_NUMS
        mdata = cld(num * _sp_width(idx), _sp_width(elem))
        forms[(elem, idx, Symbol("sp::2:4"), Symbol("x", num))] =
            (mdata, num, 2 * num)
    end
    forms
end

# spdecompress: the five validity conditions of §9.7.10.31 applied to every
# candidate tuple (Table 39 sizes).
function _spdecompress_expected()
    forms = Dict{Tuple{Vararg{Symbol}}, NTuple{3, Int}}()
    for elem in (:b8, :b16), idx in (:b2, :b4),
            (s, t) in ((1, 2), (1, 4), (1, 8), (1, 16), (2, 4), (2, 8),
                       (2, 16), (4, 8), (4, 16)),
            num in _SP_NUMS
        e = _sp_width(elem)
        i = _sp_width(idx)
        s * e <= 32 || continue                       # one input span per b32
        (idx === :b4 || t <= 4) || continue           # .b2 must index T
        32 <= t * e * num <= 4096 || continue         # output 1..128 registers
        mdata = cld(s * i * num, 32)
        cdata = cld(s * e * num, 32)
        data = cld(t * e * num, 32)
        mdata + cdata + data <= 253 || continue
        forms[(elem, idx, Symbol("sp::$s:$t"), Symbol("x", num))] =
            (mdata, cdata, data)
    end
    forms
end

@testset "spcompress registry inventory: 28 asm forms" begin
    expected = _spcompress_expected()
    @test length(expected) == 28
    @test Set(PTX.wrapper_asm_forms(:spcompress)) == Set(keys(expected))
    @test expected[(:b8, :b4, Symbol("sp::2:4"), :x64)] == (32, 64, 128)
    @test expected[(:b16, :b2, Symbol("sp::2:4"), :x1)] == (1, 1, 2)
end

@testset "spdecompress registry inventory: 143 asm forms" begin
    expected = _spdecompress_expected()
    @test length(expected) == 143
    @test Set(PTX.wrapper_asm_forms(:spdecompress)) == Set(keys(expected))
    per_width = Dict((elem, idx) => count(k -> k[1] === elem && k[2] === idx,
                                          keys(expected))
                     for elem in (:b8, :b16), idx in (:b2, :b4))
    @test per_width[(:b8, :b2)] == 20
    @test per_width[(:b8, :b4)] == 59
    @test per_width[(:b16, :b2)] == 21
    @test per_width[(:b16, :b4)] == 43
    # Assembler-confirmed sizes.
    @test expected[(:b16, :b4, Symbol("sp::1:4"), :x2)] == (1, 1, 4)
    @test expected[(:b8, :b2, Symbol("sp::2:4"), :x32)] == (4, 16, 32)
    @test expected[(:b16, :b4, Symbol("sp::1:16"), :x16)] == (2, 8, 128)
    # Rejected by the assembler: fewer than 32 output bits.
    @test !haskey(expected, (:b8, :b2, Symbol("sp::1:2"), :x1))
end

@testset "spcompress lowering: pure early-clobber register contract" begin
    mismatches = String[]
    for (mods, (nm, nc, nd)) in sort!(collect(_spcompress_expected()))
        op = Operation{:spcompress, mods}()
        argts = (NTuple{nd, UInt32}, UInt32)
        ci, rt = first(Base.code_typed(op, argts))
        code = string(ci)
        head = "spcompress." * join(String.(mods), ".") * " {"
        ok = rt === Tuple{NTuple{nm, UInt32}, NTuple{nc, UInt32}} &&
             occursin(head, code) && occursin("=&r", code) &&
             !occursin("sideeffect", code) && !occursin("~{memory}", code) &&
             !occursin("convergent", code) && !occursin("llvm.nvvm", code)
        ok || push!(mismatches, "$(mods) (rt = $rt)")
    end
    isempty(mismatches) ||
        foreach(x -> println("SPCOMPRESS LOWERING MISMATCH: ", x), mismatches)
    @test isempty(mismatches)
end

@testset "spdecompress lowering: pure early-clobber register contract" begin
    mismatches = String[]
    for (mods, (nm, nc, nd)) in sort!(collect(_spdecompress_expected()))
        op = Operation{:spdecompress, mods}()
        argts = (NTuple{nm, UInt32}, NTuple{nc, UInt32})
        ci, rt = first(Base.code_typed(op, argts))
        code = string(ci)
        head = "spdecompress." * join(String.(mods), ".") * " {"
        ok = rt === NTuple{nd, UInt32} &&
             occursin(head, code) && occursin("=&r", code) &&
             !occursin("sideeffect", code) && !occursin("~{memory}", code) &&
             !occursin("convergent", code) && !occursin("llvm.nvvm", code)
        ok || push!(mismatches, "$(mods) (rt = $rt)")
    end
    isempty(mismatches) ||
        foreach(x -> println("SPDECOMPRESS LOWERING MISMATCH: ", x), mismatches)
    @test isempty(mismatches)
end

@testset "spcompress asm: hand-pinned operand order" begin
    ci, _ = first(Base.code_typed(
        Operation{:spcompress, (:b8, :b2, Symbol("sp::2:4"), :x4)}(),
        (NTuple{8, UInt32}, UInt32)))
    # `string(ci)` shows the asm template with its `$` escaped.
    code = replace(string(ci), "\\\$" => "\$")
    @test occursin("spcompress.b8.b2.sp::2:4.x4 {\$0}, {\$1, \$2, \$3, \$4}, " *
                   "{\$5, \$6, \$7, \$8, \$9, \$10, \$11, \$12}, \$13;", code)
    @test occursin("=&r,=&r,=&r,=&r,=&r,r,r,r,r,r,r,r,r,r", code)
    ci, _ = first(Base.code_typed(
        Operation{:spdecompress, (:b16, :b4, Symbol("sp::1:4"), :x2)}(),
        (NTuple{1, UInt32}, NTuple{1, UInt32})))
    code = replace(string(ci), "\\\$" => "\$")
    @test occursin("spdecompress.b16.b4.sp::1:4.x2 {\$0, \$1, \$2, \$3}, " *
                   "{\$4}, {\$5};", code)
    @test occursin("=&r,=&r,=&r,=&r,r,r", code)
end

@testset "spcompress/spdecompress reject grammar and ABI misses" begin
    for (op, mods, argts) in (
            # sp::4:8 needs S·elemsize ≤ 32.
            (:spdecompress, (:b16, :b4, Symbol("sp::4:8"), :x4),
             (NTuple{2, UInt32}, NTuple{8, UInt32})),
            # .b2 cannot index eight targets.
            (:spdecompress, (:b8, :b2, Symbol("sp::1:8"), :x4),
             (NTuple{1, UInt32}, NTuple{1, UInt32})),
            # Fewer than 32 output bits.
            (:spdecompress, (:b8, :b2, Symbol("sp::1:2"), :x1),
             (NTuple{1, UInt32}, NTuple{1, UInt32})),
            # More than 4096 output bits.
            (:spdecompress, (:b16, :b4, Symbol("sp::1:16"), :x32),
             (NTuple{4, UInt32}, NTuple{16, UInt32})),
            # spcompress only compresses 2:4.
            (:spcompress, (:b8, :b2, Symbol("sp::1:4"), :x4),
             (NTuple{8, UInt32}, UInt32)),
            # Wrong dense-vector width for the repeat factor.
            (:spcompress, (:b8, :b2, Symbol("sp::2:4"), :x4),
             (NTuple{4, UInt32}, UInt32)),
            # Wrong index-vector width.
            (:spdecompress, (:b8, :b2, Symbol("sp::2:4"), :x32),
             (NTuple{2, UInt32}, NTuple{16, UInt32})))
        o = Operation{op, mods}()
        @test PTX.lowering(o, argts).tier === :forbidden
        args = Tuple(T <: Tuple ? ntuple(_ -> UInt32(0), length(T.parameters)) :
                     UInt32(0) for T in argts)
        @test_throws ArgumentError o(args...)
    end
end

@testset "spcompress descriptor" begin
    @test spcompress_desc(; op = :maxabs, dtype = :s8) == 0x5
    @test spcompress_desc(; op = :max, dtype = :f16) == 0x0
    @test spcompress_desc(; op = :minabs, dtype = :e2m3) == 0x3 | 0x5 << 2
    @test spcompress_desc(; op = :min, dtype = :bf16) == 0x2 | 0x1 << 2
    for dtype in (:f16, :bf16)
        @test spcompress_elemsize(dtype) === :b16
    end
    for dtype in (:u8, :s8, :e5m2, :e4m3, :e3m2, :e2m3)
        @test spcompress_elemsize(dtype) === :b8
    end
    @test_throws ArgumentError spcompress_desc(; op = :sum, dtype = :s8)
    @test_throws ArgumentError spcompress_desc(; op = :max, dtype = :e2m1)
    @test_throws ArgumentError spcompress_elemsize(:f32)
end
