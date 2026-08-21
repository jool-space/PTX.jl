# TEST_TARGET: requires=gpu evidence=runtime runtime=cc>=9.0
#
# End-to-end proof for the transpiler's register-default semantics: transpile
# the tileiras (cuTile `code_ptx`) vector-add fixture, evaluate the emitted
# Julia, and launch it. The ragged sizes are the load-bearing part — their
# bounds branches jump over the `ld.global` assignments, which in PTX leaves
# the registers undefined-but-harmless and in Julia (before the zero-init
# hoists) threw UndefVarError and trapped the kernel.

const _TILEIRAS_FIXTURE = joinpath(@__DIR__, "..", "corpus",
                                   "tileiras_vadd_sm121a.ptx")

module _TileirasVadd
    using PTX, CUDACore
end

@testset "transpiled tileiras vadd executes (full and ragged tiles)" begin
    julia_src = ptx_to_julia(read(_TILEIRAS_FIXTURE, String))
    @test occursin(r"local r11 = zero\(UInt32\)", julia_src)
    @test occursin(r"local r12 = zero\(UInt32\)", julia_src)
    Base.include_string(_TileirasVadd, julia_src)

    for n in (256, 250, 17)   # full tiles, ragged tail, single ragged tile
        a = CUDACore.rand(Float32, n)
        b = CUDACore.rand(Float32, n)
        c = CUDACore.zeros(Float32, n)
        pa = Int64(UInt64(pointer(a)))
        pb = Int64(UInt64(pointer(b)))
        pc = Int64(UInt64(pointer(c)))
        # ABI from the fixture: (ptr, len, stride) per array + one pad word;
        # .reqntid 128, one 16-element tile per block.
        CUDACore.@sync @cuda threads=128 blocks=cld(n, 16) _TileirasVadd.vadd(
            pa, UInt32(n), UInt32(1),
            pb, UInt32(n), UInt32(1),
            pc, UInt32(n), UInt32(1), UInt32(0))
        @test Array(c) ≈ Array(a) .+ Array(b)
    end
end
