using PTX: Operation

# Independent PTX 9.3 §9.7.16.5.2 inventory.  Do not derive these from the
# production constants: changing a source grid or dtype tuple must force a
# review of both this closed-world oracle and the exact dispatch surface.
const EXPECTED_WGMMA_FLOAT_NS = (
    8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
    136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240,
    248, 256,
)
const EXPECTED_WGMMA_INT_NS = (
    8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224,
)
const EXPECTED_WGMMA_FLOAT_VARIANTS = (
    (:f32, :bf16, :bf16, 16, true),
    (:f32, :f16,  :f16,  16, true),
    (:f32, :tf32, :tf32, 8,  false),
    (:f32, :e4m3, :e4m3, 32, false),
    (:f32, :e4m3, :e5m2, 32, false),
    (:f32, :e5m2, :e4m3, 32, false),
    (:f32, :e5m2, :e5m2, 32, false),
    (:f16, :f16,  :f16,  16, true),
    (:f16, :e4m3, :e4m3, 32, false),
    (:f16, :e4m3, :e5m2, 32, false),
    (:f16, :e5m2, :e4m3, 32, false),
    (:f16, :e5m2, :e5m2, 32, false),
)
const EXPECTED_WGMMA_INT_VARIANTS = (
    (:s32, :s8, :s8, 32, false),
    (:s32, :u8, :u8, 32, false),
    (:s32, :s8, :u8, 32, false),
    (:s32, :u8, :s8, 32, false),
)
# Sparse mirrors dense with doubled K (§9.7.16.6); SS + runtime-scale_d only.
const EXPECTED_WGMMA_SP_VARIANTS = (
    (:f32, :bf16, :bf16, 32, true),
    (:f32, :f16,  :f16,  32, true),
    (:f16, :f16,  :f16,  32, true),
    (:f32, :tf32, :tf32, 16, false),
    (:f32, :e4m3, :e4m3, 64, false),
    (:f32, :e4m3, :e5m2, 64, false),
    (:f32, :e5m2, :e4m3, 64, false),
    (:f32, :e5m2, :e5m2, 64, false),
    (:f16, :e4m3, :e4m3, 64, false),
    (:f16, :e4m3, :e5m2, 64, false),
    (:f16, :e5m2, :e4m3, 64, false),
    (:f16, :e5m2, :e5m2, 64, false),
    (:s32, :s8, :s8, 64, false),
    (:s32, :u8, :u8, 64, false),
    (:s32, :s8, :u8, 64, false),
    (:s32, :u8, :s8, 64, false),
)

_wgmma_accumulator_type(dt_d::Symbol, n::Int) =
    dt_d === :f32 ? NTuple{n ÷ 2, Float32} :
    dt_d === :f16 ? NTuple{n ÷ 4, UInt32} :
    dt_d === :s32 ? NTuple{n ÷ 2, Int32} :
    error("unexpected test dtype $dt_d")

_wgmma_mods(dt_d, dt_a, dt_b, n, k) =
    (:mma_async, :sync, :aligned, Symbol("m64n", n, "k", k),
     dt_d, dt_a, dt_b)

@testset "dense WGMMA: closed-world dtype and N inventories" begin
    @test PTX._WGMMA_FLOAT_NS == EXPECTED_WGMMA_FLOAT_NS
    @test PTX._WGMMA_INT_NS == EXPECTED_WGMMA_INT_NS
    @test PTX._WGMMA_FLOAT_VARIANTS == EXPECTED_WGMMA_FLOAT_VARIANTS
    @test PTX._WGMMA_INT_VARIANTS == EXPECTED_WGMMA_INT_VARIANTS
    @test PTX._WGMMA_SP_VARIANTS == EXPECTED_WGMMA_SP_VARIANTS
    @test length(EXPECTED_WGMMA_FLOAT_VARIANTS) *
          length(EXPECTED_WGMMA_FLOAT_NS) == 384
    @test length(EXPECTED_WGMMA_INT_VARIANTS) *
          length(EXPECTED_WGMMA_INT_NS) == 64

    # Audit every N candidate on the floating eight-wide grid.  All four
    # surface signatures (SS runtime/Val and RF runtime) must exist exactly on
    # the relevant ISA grid, and every hole must stop at the typed-only guard.
    for (variants, legal_ns) in (
            (EXPECTED_WGMMA_FLOAT_VARIANTS, EXPECTED_WGMMA_FLOAT_NS),
            (EXPECTED_WGMMA_INT_VARIANTS, EXPECTED_WGMMA_INT_NS))
        for (dt_d, dt_a, dt_b, k, _has_trans) in variants, n in 8:8:256
            d_T = _wgmma_accumulator_type(dt_d, n)
            op = Operation{:wgmma, _wgmma_mods(dt_d, dt_a, dt_b, n, k)}()
            signatures = (
                (d_T, UInt64, UInt64, Bool),
                (d_T, UInt64, UInt64, Val{true}),
                (d_T, UInt64, UInt64, Val{false}),
                (d_T, NTuple{4, UInt32}, UInt64, Bool),
            )
            legal = n in legal_ns
            for argts in signatures
                method = which(op, argts)
                exact = !endswith(String(method.file), "entries.jl")
                @test exact == legal
                info = PTX.lowering(op, argts)
                if legal
                    @test method.module === PTX
                    @test info.tier === :asm
                    @test info.rettype === d_T
                else
                    @test info.tier === :forbidden
                    @test info.asm === nothing
                    @test endswith(String(method.file), "entries.jl")
                end
            end
        end
    end
end

@testset "dense WGMMA: illegal products fail loud" begin
    # These cover a floating-grid hole (n40), the two values beyond the dense
    # integer maximum that sparse WGMMA does support (n240/n256), wrong K, and
    # a cross-class dtype pair (fp8 × bf16) no §9.7.16.5 syntax block admits.
    misses = (
        ((:s32, :s8,   :s8,   40,  32), NTuple{20, Int32}),
        ((:s32, :u8,   :s8,   240, 32), NTuple{120, Int32}),
        ((:s32, :s8,   :u8,   256, 32), NTuple{128, Int32}),
        ((:s32, :s8,   :s8,   8,   16), NTuple{4, Int32}),
        ((:f32, :e4m3, :bf16, 8,   32), NTuple{4, Float32}),
    )
    for ((dt_d, dt_a, dt_b, n, k), d_T) in misses
        op = Operation{:wgmma, _wgmma_mods(dt_d, dt_a, dt_b, n, k)}()
        argts = (d_T, UInt64, UInt64, Bool)
        @test PTX.lowering(op, argts).tier === :forbidden
        @test endswith(String(which(op, argts).file), "entries.jl")
    end

    # Exercise generated dispatch itself, not only reflection.
    bad = Operation{:wgmma,
        (:mma_async, :sync, :aligned, :m64n40k32, :s32, :s8, :s8)}()
    args = (ntuple(_ -> Int32(0), Val(20)), UInt64(0), UInt64(0), false)
    err = try
        bad(args...)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("requires an exact typed wrapper", sprint(showerror, err))
end

# Independent Table 60 modifier surface.  Source location is an orthogonal
# dimension: each modifier accepts either a 64-bit shared-memory A descriptor
# or a bracketed 32-bit TMEM A address, while B remains a descriptor and both
# scale matrices are mandatory TMEM addresses.
const EXPECTED_TCGEN05_MX_ALIAS_ROWS = (
    (:mxf8f6f4, Symbol("scale_vec::1X"), :block32),
    (:mxf4,     Symbol("scale_vec::2X"), :block32),
    (:mxf4nvf4, Symbol("scale_vec::2X"), :block32),
    (:mxf4nvf4, Symbol("scale_vec::4X"), :block16),
)
const EXPECTED_TCGEN05_MX_KIND_SCALES = Set((
    (:mxf8f6f4, Symbol("scale_vec::1X")),
    (:mxf8f6f4, :block32),
    (:mxf4, Symbol("scale_vec::2X")),
    (:mxf4, :block32),
    (:mxf4nvf4, Symbol("scale_vec::2X")),
    (:mxf4nvf4, Symbol("scale_vec::4X")),
    (:mxf4nvf4, :block16),
    (:mxf4nvf4, :block32),
))

_tcgen_mx_mods(kind, scale, cg; sp = false, coll = nothing) =
    (:mma, (sp ? (:sp,) : ())..., Symbol("cta_group::", cg),
     Symbol("kind::", kind), :block_scale, scale,
     (coll === nothing ? () : (coll,))...)

const _TCGEN_MX_COLLECTORS = (nothing, Symbol("collector::a::lastuse"),
                              Symbol("collector::a::fill"),
                              Symbol("collector::a::use"))

@testset "tcgen05 MX: complete block-scale schema inventory" begin
    @test PTX._TCGEN05_MX_SCALE_VARIANTS == EXPECTED_TCGEN05_MX_ALIAS_ROWS

    actual = Set{Tuple{Symbol, Symbol}}()
    for (kind, scale_vec, block) in PTX._TCGEN05_MX_SCALE_VARIANTS
        push!(actual, (kind, scale_vec), (kind, block))
    end
    @test actual == EXPECTED_TCGEN05_MX_KIND_SCALES
    @test length(actual) == 8

    for (kind, scale) in EXPECTED_TCGEN05_MX_KIND_SCALES, cg in (1, 2),
            sp in (false, true), coll in _TCGEN_MX_COLLECTORS
        op = Operation{:tcgen05, _tcgen_mx_mods(kind, scale, cg; sp, coll)}()
        meta = sp ? (UInt32,) : ()
        ss = (UInt32, UInt64, UInt64, meta..., UInt32, UInt32, UInt32, Bool)
        ts = (UInt32, UInt32, UInt64, meta..., UInt32, UInt32, UInt32, Bool)
        for (argts, a_operand) in ((ss, "\$1"), (ts, "[\$1]"))
            @test which(op, argts).module === PTX
            info = PTX.lowering(op, argts)
            @test info.tier === :asm
            @test info.rettype === Nothing
            ci, _ = first(Base.code_typed(op, argts))
            typed = string(ci)
            @test occursin("tcgen05.mma" * (sp ? ".sp" : "") *
                           ".cta_group::$cg.kind::$kind" *
                           ".block_scale.$scale" *
                           (coll === nothing ? " " : ".$coll "), typed)
            # Julia's CodeInfo printer escapes the `$` operand sigils inside
            # the inline-assembly string.  Strip only that presentation-layer
            # escaping before checking the exact PTX operand schema.
            unescaped = replace(typed, "\\\$" => "\$")
            schema = sp ?
                "[\$0], $a_operand, \$2, [\$3], \$4, [\$5], [\$6], \$7;" :
                "[\$0], $a_operand, \$2, \$3, [\$4], [\$5], \$6;"
            @test occursin(schema, unescaped)
            @test occursin("asm sideeffect", typed)
            @test occursin("~{memory}", typed)
        end
        # The other arity is not a method: a dense call cannot silently
        # drop into an sp form (or vice versa) — the metadata operand is
        # load-bearing, not optional.
        other = sp ? (UInt32, UInt64, UInt64, UInt32, UInt32, UInt32, Bool) :
                     (UInt32, UInt64, UInt64, UInt32, UInt32, UInt32,
                      UInt32, Bool)
        @test PTX.lowering(op, other).tier === :forbidden
    end
end

@testset "tcgen05 MX: incomplete or illegal schemas fail loud" begin
    old_kinds = (:mxf8f6f4, :mxf4, :mxf4nvf4)
    old_argts = (UInt32, UInt64, UInt64, UInt32, Bool)
    for kind in old_kinds, cg in (1, 2)
        old = Operation{:tcgen05,
            (:mma, Symbol("cta_group::", cg), Symbol("kind::", kind))}()
        @test PTX.lowering(old, old_argts).tier === :forbidden
        @test endswith(String(which(old, old_argts).file), "entries.jl")
    end

    # Legal modifier, wrong arity/carrier: scale addresses cannot be omitted,
    # passed as descriptors, or silently reinterpreted.
    goodmods = _tcgen_mx_mods(:mxf8f6f4, Symbol("scale_vec::1X"), 1)
    goodop = Operation{:tcgen05, goodmods}()
    for argts in (
            old_argts,
            (UInt32, UInt64, UInt64, UInt32, UInt32, Bool),
            (UInt32, UInt64, UInt64, UInt32, UInt64, UInt64, Bool),
            (UInt32, UInt64, UInt32, UInt32, UInt32, UInt32, Bool))
        @test PTX.lowering(goodop, argts).tier === :forbidden
        @test endswith(String(which(goodop, argts).file), "entries.jl")
    end

    # Modifier pairs outside Table 60 and the ambiguous no-scale spelling are
    # intentionally not methods.
    for (kind, scale) in (
            (:mxf8f6f4, Symbol("scale_vec::2X")),
            (:mxf4, Symbol("scale_vec::1X")),
            (:mxf4nvf4, Symbol("scale_vec::1X")))
        op = Operation{:tcgen05, _tcgen_mx_mods(kind, scale, 1)}()
        argts = (UInt32, UInt64, UInt64, UInt32, UInt32, UInt32, Bool)
        @test PTX.lowering(op, argts).tier === :forbidden
    end
    ambiguous = Operation{:tcgen05,
        (:mma, Symbol("cta_group::1"), Symbol("kind::mxf4"), :block_scale)}()
    argts = (UInt32, UInt64, UInt64, UInt32, UInt32, UInt32, Bool)
    @test PTX.lowering(ambiguous, argts).tier === :forbidden

    old = Operation{:tcgen05,
        (:mma, Symbol("cta_group::1"), Symbol("kind::mxf8f6f4"))}()
    err = try
        old(UInt32(0), UInt64(0), UInt64(0), UInt32(0), false)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("requires an exact typed wrapper", sprint(showerror, err))
end
