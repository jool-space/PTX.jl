using PTX
using PTX: Operation, lowering

# Independent tcgen05 TMEM data-movement oracles: the PTX 9.3 §9.7.17.8
# ld/st grid (tier-2 intrinsics), the §9.7.17.9 cp shapes, and the
# single-route asm families ld.red (PTX 8.8 §9.7.18.8) and
# ld{.red}.spcompress (PTX 9.4 §9.7.18.8, sm_107a). The production methods
# are intentionally not the source of any inventory here. Assembler
# evidence lives in ptxas/tcgen05_ldst.jl.

# A grammar or ABI miss: no reviewed method, and the call builder refuses.
function _t5_forbidden(mods, args)
    @test lowering(Operation{:tcgen05, mods}(), args).tier === :forbidden
    @test_throws ArgumentError PTX.build_call(:tcgen05, mods, args)
    nothing
end

# --- ld/st (PTX 9.3 §9.7.17.8 Table 52) --------------------------------------

# Independent PTX 9.3 §9.7.17.8 Table 52 inventory (shape, per-count base
# registers, legal counts, needs-immHalfSplitoff). The production methods
# are intentionally not the source of this oracle.
const _T5_LDST_SHAPES = [
    (Symbol("16x64b"),   1, (1, 2, 4, 8, 16, 32, 64, 128), false),
    (Symbol("32x32b"),   1, (1, 2, 4, 8, 16, 32, 64, 128), false),
    (Symbol("16x128b"),  2, (1, 2, 4, 8, 16, 32, 64),      false),
    (Symbol("16x256b"),  4, (1, 2, 4, 8, 16, 32),          false),
    (Symbol("16x32bx2"), 1, (1, 2, 4, 8, 16, 32, 64, 128), true),
]

_t5_mods(op, shape, c; repack = false) =
    (op, :sync, :aligned, shape, Symbol("x", c),
     (repack ? (Symbol(op === :ld ? "pack::16b" : "unpack::16b"),) : ())...,
     :b32)

@testset "tcgen05 ld/st closed callable surface" begin
    reviewed = Set{String}()

    for (shape, base, counts, split) in _T5_LDST_SHAPES, c in counts,
            repack in (false, true)
        n = base * c
        splitargs = split ? (Val{8},) : ()
        for op in (:ld, :st)
            intrinsic = "llvm.nvvm.tcgen05.$op.$shape.x$c"
            push!(reviewed, intrinsic)
            mods = _t5_mods(op, shape, c; repack)
            args = op === :ld ? (UInt32, splitargs...) :
                                (UInt32, splitargs..., NTuple{n, UInt32})
            info = lowering(Operation{:tcgen05, mods}(), args)
            @test info.tier === :intrinsic
            @test intrinsic in info.intrinsics
            if op === :ld
                @test info.rettype === (n == 1 ? UInt32 : NTuple{n, UInt32})
            else
                @test info.rettype === Nothing
            end

            record = PTX.NVVM.intrinsic(intrinsic)
            @test :convergent in record.props
            if op === :ld
                @test record.ret ===
                      (Symbol("v$(n)i32"),)
                @test record.immargs == (split ? (2, 3) : (2,))
            else
                @test record.ret === ()
                @test record.immargs == (split ? (2, 4) : (3,))
            end
        end
    end

    # Closed world: the reviewed Table-52 grid is exactly the registry's
    # ordinary ld/st inventory. The separate ld.red namespace is pinned
    # in host/nvptx_backend.jl.
    registry = Set(name for name in keys(PTX.NVVM.TABLE)
                   if (startswith(name, "llvm.nvvm.tcgen05.ld.") &&
                       !startswith(name, "llvm.nvvm.tcgen05.ld.red.")) ||
                      startswith(name, "llvm.nvvm.tcgen05.st."))
    @test reviewed == registry
    @test length(reviewed) == 74
end

@testset "tcgen05 ld/st rejects grammar and ABI misses" begin
    b2 = Symbol("16x32bx2")
    misses = (
        # Table 52 NA cells: per-lane register count would exceed 128.
        (_t5_mods(:ld, Symbol("16x128b"), 128), (UInt32,)),
        (_t5_mods(:st, Symbol("16x256b"), 64),
         (UInt32, NTuple{256, UInt32})),
        # pack is a load-side qualifier and unpack a store-side one.
        ((:ld, :sync, :aligned, Symbol("16x64b"), :x2,
          Symbol("unpack::16b"), :b32), (UInt32,)),
        ((:st, :sync, :aligned, Symbol("16x64b"), :x2,
          Symbol("pack::16b"), :b32), (UInt32, NTuple{2, UInt32})),
        # Canonical modifier order: shape.num.pack.b32.
        ((:ld, :sync, :aligned, Symbol("16x64b"), Symbol("pack::16b"),
          :x2, :b32), (UInt32,)),
        # immHalfSplitoff is mandatory for 16x32bx2, compile-time only,
        # and illegal elsewhere.
        (_t5_mods(:ld, b2, 2), (UInt32,)),
        (_t5_mods(:ld, b2, 2), (UInt32, Int64)),
        (_t5_mods(:st, b2, 2), (UInt32, NTuple{2, UInt32})),
        (_t5_mods(:ld, Symbol("16x64b"), 2), (UInt32, Val{8})),
        # Data tuple width is pinned by shape×count.
        (_t5_mods(:st, Symbol("16x128b"), 2), (UInt32, NTuple{2, UInt32})),
        (_t5_mods(:st, b2, 2), (UInt32, Val{8}, NTuple{4, UInt32})),
        # Load-with-reduction returns its scalar reduction alongside the
        # loaded registers; it does not take a scalar accumulator input.
        ((:ld, :red, :sync, :aligned, Symbol("32x32b"), :x2, :min, :f32),
         (Float32, UInt32)),
    )

    for (mods, args) in misses
        _t5_forbidden(mods, args)
    end
end

# --- cp (PTX 9.3 §9.7.17.9) -------------------------------------------------------

# Independent PTX 9.3 §9.7.17.9 inventory: shape (with its mandatory
# multicast pairing where the ISA requires one) × optional decompression.
# The production methods are intentionally not the source of this oracle.
const _T5_CP_SHAPES = [
    ((Symbol("128x256b"),),                          "128x256b"),
    ((Symbol("4x256b"),),                            "4x256b"),
    ((Symbol("128x128b"),),                          "128x128b"),
    ((Symbol("64x128b"), Symbol("warpx2::02_13")),   "64x128b_warpx2_02_13"),
    ((Symbol("64x128b"), Symbol("warpx2::01_23")),   "64x128b_warpx2_01_23"),
    ((Symbol("32x128b"), :warpx4),                   "32x128b_warpx4"),
]
const _T5_CP_FMTS = [((), ""), ((:b8x16, :b6x16_p32), "b6x16_p32"),
                     ((:b8x16, :b4x16_p64), "b4x16_p64")]

@testset "tcgen05.cp closed callable surface" begin
    reviewed = Set{String}()

    for (shapemods, stem) in _T5_CP_SHAPES, (fmtmods, fmt) in _T5_CP_FMTS,
            cg in 1:2
        intrinsic = fmt == "" ? "llvm.nvvm.tcgen05.cp.$stem.cg$cg" :
                                "llvm.nvvm.tcgen05.cp.$stem.$fmt.cg$cg"
        push!(reviewed, intrinsic)
        mods = (:cp, Symbol("cta_group::", cg), shapemods..., fmtmods...)
        # Single-route asm since the demotion (see wrappers/tcgen05.jl):
        # every form is convergent inline asm with the full clobber.
        info = lowering(Operation{:tcgen05, mods}(), (UInt32, UInt64))
        @test info.tier === :asm
        @test info.rettype === Nothing
        @test isempty(info.intrinsics)

        # The unwrapped intrinsic records stay in the pinned registry; the
        # attribute review below is what the demotion walked away from
        # (convergent + argmem — no stronger promise than the asm route).
        record = PTX.NVVM.intrinsic(intrinsic)
        @test record.ret === ()
        @test :convergent in record.props
        @test :inaccessiblemem_or_argmemonly in record.props
        @test (1, :nocapture) in record.argattrs
    end

    # Closed world: the reviewed grid is exactly the registry's cp
    # inventory — a registry addition must force a review here.
    registry = Set(name for name in keys(PTX.NVVM.TABLE)
                   if startswith(name, "llvm.nvvm.tcgen05.cp."))
    @test reviewed == registry
    @test length(reviewed) == 36
end

@testset "tcgen05.cp rejects grammar and ABI misses" begin
    cg1 = Symbol("cta_group::1")
    misses = (
        # The ISA couples multicast to shape: 64x128b requires a warpx2
        # pairing, 32x128b requires warpx4, and the other shapes take none.
        ((:cp, cg1, Symbol("64x128b")), (UInt32, UInt64)),
        ((:cp, cg1, Symbol("32x128b")), (UInt32, UInt64)),
        ((:cp, cg1, Symbol("64x128b"), :warpx4), (UInt32, UInt64)),
        ((:cp, cg1, Symbol("32x128b"), Symbol("warpx2::02_13")),
         (UInt32, UInt64)),
        ((:cp, cg1, Symbol("128x256b"), :warpx4), (UInt32, UInt64)),
        # dst_fmt and src_fmt are an ordered pair.
        ((:cp, cg1, Symbol("128x256b"), :b6x16_p32, :b8x16),
         (UInt32, UInt64)),
        ((:cp, cg1, Symbol("128x256b"), :b8x16), (UInt32, UInt64)),
        ((:cp, cg1, Symbol("128x256b"), :b6x16_p32), (UInt32, UInt64)),
        # Multicast must precede the format pair (ISA modifier order).
        ((:cp, cg1, Symbol("64x128b"), :b8x16, :b6x16_p32,
          Symbol("warpx2::02_13")), (UInt32, UInt64)),
        # Operand carriers are pinned: TMEM address is UInt32, the SMEM
        # descriptor is UInt64.
        ((:cp, cg1, Symbol("128x256b")), (UInt64, UInt64)),
        ((:cp, cg1, Symbol("128x256b")), (UInt32, UInt32)),
        ((:cp, cg1, Symbol("128x256b")), (UInt32,)),
    )

    for (mods, args) in misses
        _t5_forbidden(mods, args)
    end
end

# --- ld.red (PTX 8.8 §9.7.18.8) ---------------------------------------------------

# tcgen05.ld.red — single-route asm family (PTX 8.8 §9.7.18.8): independent
# grid oracle against the wrapper registry, plus lowered-code assertions for
# the collective-asm contract and the (data..., redval) return ABI. The
# assembler leg (sm_103f assembles; sm_100a refuses the instruction) lives in
# ptxas/tcgen05_ldst.jl; runtime evidence needs a CC 10.3+ device.

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

# --- ld{.red}.spcompress (PTX 9.4 §9.7.18.8, sm_107a) -----------------------------

# tcgen05.ld{.red}.spcompress — single-route asm family (PTX 9.4 §9.7.18.8,
# sm_107a): independent grid oracle against the wrapper registry, plus
# lowered-code assertions for the collective-asm contract and the
# (mdata, cdata[, redval]) return ABI. The assembler leg (sm_107a assembles;
# sm_107f and sm_100a refuse the `.sp::2:4` qualifier) lives in
# ptxas/tcgen05_ldst.jl; runtime evidence needs a CC 10.7 device.

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
