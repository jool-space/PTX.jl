using PTX
using PTX: Operation, lowering

# Independent PTX 9.3 §9.7.17.10.9.3/.4 weight-stationary oracle:
# cta_group::1 only, B-side addressed collector (b0..b3 × op, default
# b0::discard), optional runtime zero-column-mask descriptor, sp
# metadata between the B descriptor and idesc. The production spec
# tables are intentionally not the source of this oracle.
_t5_ws_colls() = begin
    out = Any[nothing]
    for buf in 0:3, op in (:discard, :lastuse, :fill, :use)
        buf == 0 && op === :discard && continue
        push!(out, Symbol("collector::b$buf::$op"))
    end
    out
end

_t5_ws_mods(kind; sp = false, coll = nothing) =
    (:mma, :ws, (sp ? (:sp,) : ())..., Symbol("cta_group::1"),
     Symbol("kind::", kind),
     (coll === nothing ? () : (coll,))...)

@testset "tcgen05 ws mma closed callable surface" begin
    reviewed = Set{String}()

    for kind in (:f16, :tf32, :f8f6f4, :i8), sp in (false, true),
            tmem_a in (false, true), coll in _t5_ws_colls()
        mods = _t5_ws_mods(kind; sp, coll)
        aT = tmem_a ? UInt32 : UInt64
        meta = sp ? (UInt32,) : ()
        stem = "llvm.nvvm.tcgen05.mma.ws." * (sp ? "sp." : "") *
               (tmem_a ? "tensor" : "shared")

        for (args, intrinsic) in (
                ((UInt32, aT, UInt64, meta..., UInt32, Bool), stem),
                ((UInt32, aT, UInt64, meta..., UInt32, Bool, UInt64),
                 stem * ".zero_col_mask"))
            push!(reviewed, intrinsic)
            info = lowering(Operation{:tcgen05, mods}(), args)
            @test info.tier === :intrinsic
            @test info.rettype === Nothing
            @test intrinsic in info.intrinsics
        end
    end

    registry = Set(name for name in keys(PTX.NVVM.TABLE)
                   if startswith(name, "llvm.nvvm.tcgen05.mma.ws."))
    @test reviewed == registry
    @test reviewed == Set(PTX.wrapper_intrinsic_names(:tcgen05_mma_ws))
    @test length(reviewed) == 8
end

@testset "tcgen05 ws mma rejects grammar and ABI misses" begin
    misses = (
        # ws is cta_group::1-only.
        ((:mma, :ws, Symbol("cta_group::2"), Symbol("kind::f16")),
         (UInt32, UInt64, UInt64, UInt32, Bool)),
        # The collector buffer must be addressed: the dense a-side
        # spelling and a bare op are not ws grammar.
        (_t5_ws_mods(:f16; coll = Symbol("collector::a::fill")),
         (UInt32, UInt64, UInt64, UInt32, Bool)),
        (_t5_ws_mods(:f16; coll = Symbol("collector::fill")),
         (UInt32, UInt64, UInt64, UInt32, Bool)),
        ((:mma, :ws, Symbol("cta_group::1"), Symbol("kind::f16"),
          Symbol("collector::b4::fill")),
         (UInt32, UInt64, UInt64, UInt32, Bool)),
        # The zero-column-mask descriptor is a runtime 64-bit operand,
        # not an immediate, and not a 32-bit value.
        (_t5_ws_mods(:f16),
         (UInt32, UInt64, UInt64, UInt32, Bool, Val{0})),
        (_t5_ws_mods(:f16),
         (UInt32, UInt64, UInt64, UInt32, Bool, UInt32)),
        # sp metadata is mandatory for ws.sp and sits before idesc.
        (_t5_ws_mods(:f16; sp = true),
         (UInt32, UInt64, UInt64, UInt32, Bool)),
        (_t5_ws_mods(:f16; sp = true),
         (UInt32, UInt64, UInt64, UInt32, Bool, UInt32)),
        # ws has no ashift, disable-output-lane, or scale-input-d.
        ((:mma, :ws, Symbol("cta_group::1"), Symbol("kind::f16"), :ashift),
         (UInt32, UInt32, UInt64, UInt32, Bool)),
        (_t5_ws_mods(:f16),
         (UInt32, UInt64, UInt64, UInt32, NTuple{4, UInt32}, Bool)),
        (_t5_ws_mods(:f16),
         (UInt32, UInt64, UInt64, UInt32, Bool, Val{5})),
    )

    for (mods, args) in misses
        @test lowering(Operation{:tcgen05, mods}(), args).tier === :forbidden
        @test_throws ArgumentError PTX.build_call(:tcgen05, mods, args)
    end
end
