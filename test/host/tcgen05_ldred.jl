# tcgen05.ld.red — single-route asm family (PTX 8.8 §9.7.18.8): independent
# grid oracle against the wrapper registry, plus lowered-code assertions for
# the collective-asm contract and the (data..., redval) return ABI. Target
# evidence lives in ptxas/tcgen05_ldst.jl (sm_103f assembles; sm_100a
# refuses the instruction); runtime evidence needs a CC 10.3+ device.

using PTX: Operation

const _LDRED_GRID = (
    shapes = (Symbol("32x32b"), Symbol("16x32bx2")),
    counts = (2, 4, 8, 16, 32, 64, 128),
    redops = (:min, :max),
    types  = (((), :f32), ((:abs,), :f32), ((Symbol("NaN"),), :f32),
              ((:abs, Symbol("NaN")), :f32), ((), :u32), ((), :s32)),
)

function _ldred_expected_forms()
    forms = Set{Tuple{Vararg{Symbol}}}()
    for shape in _LDRED_GRID.shapes, count in _LDRED_GRID.counts,
            redop in _LDRED_GRID.redops, (variant, dtype) in _LDRED_GRID.types
        push!(forms, (:ld, :red, :sync, :aligned, shape, Symbol("x", count),
                      redop, variant..., dtype))
    end
    forms
end

const _LDRED_REDT = Dict(:f32 => Float32, :u32 => UInt32, :s32 => Int32)

@testset "ld.red registry inventory: 168 asm forms" begin
    expected = _ldred_expected_forms()
    @test length(expected) == 168
    @test Set(PTX.wrapper_asm_forms(:tcgen05_ldred)) == expected
end

@testset "ld.red lowering: asm head, collective contract, return ABI" begin
    mismatches = String[]
    checked = 0
    for mods in sort!(collect(_ldred_expected_forms()))
        op = Operation{:tcgen05, mods}()
        split = mods[5] === Symbol("16x32bx2")
        n = parse(Int, String(mods[6])[2:end])
        redT = _LDRED_REDT[mods[end]]
        argts = split ? (UInt32, Val{16}) : (UInt32,)
        ci, rt = first(Base.code_typed(op, argts))
        code = string(ci)
        head = "tcgen05." * join(String.(mods), ".") * " {"
        checked += 1
        ok = rt === Tuple{fill(UInt32, n)..., redT} &&
             occursin(head, code) &&
             occursin("sideeffect", code) &&
             occursin("~{memory}", code) &&
             occursin("convergent nomerge", code) &&
             (!split || occursin("], 16;", code))
        ok && continue
        length(mismatches) < 8 &&
            push!(mismatches, "$(mods) (rt = $rt)")
    end
    isempty(mismatches) ||
        foreach(x -> println("LDRED LOWERING MISMATCH: ", x), mismatches)
    @test isempty(mismatches)
    @test checked == 168
end

@testset "ld.red IR: hand-pinned exact shape" begin
    # Golden-style pin of one complete rendering: operand order (data
    # destinations, redval destination, bracketed taddr), the f32 redval's
    # float register class, and the callsite attribute group.
    ir, rt = PTX._tcgen05_ldred_ir(
        (:ld, :red, :sync, :aligned, Symbol("32x32b"), :x2, :min, :f32),
        2, Float32, nothing)
    @test rt === Tuple{UInt32, UInt32, Float32}
    @test occursin("call { i32, i32, float } asm sideeffect " *
                   "\"tcgen05.ld.red.sync.aligned.32x32b.x2.min.f32 " *
                   "{\$0, \$1}, \$2, [\$3];\", " *
                   "\"=r,=r,=f,r,~{memory}\"(i32 %a0) #0", ir)
    @test occursin("attributes #0 = { convergent nomerge nounwind }", ir)

    # Split shape: the immediate is part of the asm text, after the address.
    irs, rts = PTX._tcgen05_ldred_ir(
        (:ld, :red, :sync, :aligned, Symbol("16x32bx2"), :x2, :max, :u32),
        2, UInt32, 8)
    @test rts === Tuple{UInt32, UInt32, UInt32}
    @test occursin("{\$0, \$1}, \$2, [\$3], 8;", irs)
    @test occursin("\"=r,=r,=r,r,~{memory}\"", irs)
end

@testset "ld.red split offset: Val-typed and validated" begin
    op = Operation{:tcgen05, (:ld, :red, :sync, :aligned,
                              Symbol("16x32bx2"), :x2, :min, :f32)}()
    @test_throws ArgumentError op(UInt32(0), Val(-8))
    @test_throws ArgumentError op(UInt32(0), Val(:x))
    # A bare integer where the immediate Val belongs hits the
    # typed-wrapper-only refusal instead of dispatching.
    @test_throws ArgumentError op(UInt32(0), 8)
end
