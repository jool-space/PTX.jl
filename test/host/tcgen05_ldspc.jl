# tcgen05.ld{.red}.spcompress — single-route asm family (PTX 9.4 §9.7.18.8,
# sm_107a): independent grid oracle against the wrapper registry, plus
# lowered-code assertions for the collective-asm contract and the
# (mdata, cdata[, redval]) return ABI. Assembler evidence lives in
# ptxas/ptx94_ga.jl (sm_107a assembles; sm_107f and sm_100a refuse the
# `.sp::2:4` qualifier); runtime evidence needs a CC 10.7 device.

using PTX: Operation

const _LDSPC_GRID = (
    counts = (4, 8, 16, 32, 64, 128),
    rowops = (:min, :max),
    ld_variants = ((), (:abs,)),
    red_variants = ((), (:abs,), (Symbol("NaN"),), (:abs, Symbol("NaN"))),
)

function _ldspc_expected_forms()
    forms = Set{Tuple{Vararg{Symbol}}}()
    for count in _LDSPC_GRID.counts, rowop in _LDSPC_GRID.rowops
        for variant in _LDSPC_GRID.ld_variants
            push!(forms, (:ld, :spcompress, :sync, :aligned, Symbol("32x32b"),
                          Symbol("x", count), rowop, Symbol("sp::2:4"),
                          variant..., :f32, :b2))
        end
        for variant in _LDSPC_GRID.red_variants
            push!(forms, (:ld, :red, :spcompress, :sync, :aligned,
                          Symbol("32x32b"), Symbol("x", count), rowop,
                          Symbol("sp::2:4"), variant..., :f32, :b2))
        end
    end
    forms
end

@testset "ld.spcompress registry inventory: 72 asm forms" begin
    expected = _ldspc_expected_forms()
    @test length(expected) == 72
    @test count(m -> m[2] === :red, expected) == 48
    @test Set(PTX.wrapper_asm_forms(:tcgen05_ldspc)) == expected
end

@testset "ld.spcompress lowering: asm head, collective contract, return ABI" begin
    mismatches = String[]
    checked = 0
    for mods in sort!(collect(_ldspc_expected_forms()))
        op = Operation{:tcgen05, mods}()
        red = mods[2] === :red
        n = parse(Int, String(mods[red ? 7 : 6])[2:end])
        nm = cld(n, 32)
        nc = n ÷ 2
        ci, rt = first(Base.code_typed(op, (UInt32,)))
        code = string(ci)
        head = "tcgen05." * join(String.(mods), ".") * " {"
        want = Tuple{NTuple{nm, UInt32}, NTuple{nc, UInt32},
                     (red ? (Float32,) : ())...}
        checked += 1
        ok = rt === want &&
             occursin(head, code) &&
             occursin("sideeffect", code) &&
             occursin("~{memory}", code) &&
             occursin("convergent nomerge", code)
        ok && continue
        length(mismatches) < 8 &&
            push!(mismatches, "$(mods) (rt = $rt)")
    end
    isempty(mismatches) ||
        foreach(x -> println("LDSPC LOWERING MISMATCH: ", x), mismatches)
    @test isempty(mismatches)
    @test checked == 72
end

@testset "ld.spcompress IR: hand-pinned exact shape" begin
    # One index register, two kept-data registers, the f32 redval, then the
    # bracketed TMEM address; outputs are ordinary (non-early-clobber)
    # register classes like ld.red.
    mods = (:ld, :red, :spcompress, :sync, :aligned, Symbol("32x32b"), :x4,
            :max, Symbol("sp::2:4"), :abs, Symbol("NaN"), :f32, :b2)
    ir, flat, nm, nc = PTX._tcgen05_ldspc_ir(mods, true, 4)
    @test (nm, nc) == (1, 2)
    @test flat === Tuple{UInt32, UInt32, UInt32, Float32}
    @test occursin("call { i32, i32, i32, float } asm sideeffect " *
                   "\"tcgen05.ld.red.spcompress.sync.aligned.32x32b.x4.max" *
                   ".sp::2:4.abs.NaN.f32.b2 {\$0}, {\$1, \$2}, \$3, [\$4];\", " *
                   "\"=r,=r,=r,=f,r,~{memory}\"(i32 %a0) #0", ir)
    @test occursin("attributes #0 = { convergent nomerge nounwind }", ir)

    # Without .red the address follows the data group directly, and x128
    # spreads its 128 indices over four registers.
    mods = (:ld, :spcompress, :sync, :aligned, Symbol("32x32b"), :x128,
            :min, Symbol("sp::2:4"), :f32, :b2)
    ir, flat, nm, nc = PTX._tcgen05_ldspc_ir(mods, false, 128)
    @test (nm, nc) == (4, 64)
    @test flat === Tuple{fill(UInt32, 68)...}
    @test occursin("{\$0, \$1, \$2, \$3}, {\$4, ", ir)
    @test occursin("\$67}, [\$68];", ir)
end

@testset "ld.spcompress rejects grammar and ABI misses" begin
    for mods in (
            # .x2 is below the family floor (num ≥ x4).
            (:ld, :spcompress, :sync, :aligned, Symbol("32x32b"), :x2, :min,
             Symbol("sp::2:4"), :f32, :b2),
            # Only the 32x32b shape compresses.
            (:ld, :spcompress, :sync, :aligned, Symbol("16x32bx2"), :x4, :min,
             Symbol("sp::2:4"), :f32, :b2),
            # .NaN belongs to the .red grammar.
            (:ld, :spcompress, :sync, :aligned, Symbol("32x32b"), :x4, :min,
             Symbol("sp::2:4"), Symbol("NaN"), :f32, :b2),
            # Integer rows are not compressible.
            (:ld, :red, :spcompress, :sync, :aligned, Symbol("32x32b"), :x4,
             :min, Symbol("sp::2:4"), :u32, :b2),
            # .b4 indices are not in the tcgen05 grammar.
            (:ld, :spcompress, :sync, :aligned, Symbol("32x32b"), :x4, :min,
             Symbol("sp::2:4"), :f32, :b4))
        op = Operation{:tcgen05, mods}()
        @test PTX.lowering(op, (UInt32,)).tier === :forbidden
        @test_throws ArgumentError op(UInt32(0))
    end
end
