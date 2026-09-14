using PTX
using PTX: Operation, lowering

# Independent tcgen05.mma host oracles. The PTX 9.3 §9.7.17.10 dense, sparse,
# and weight-stationary grids lower through tier-2 intrinsics; the PTX 9.4
# §9.7.18.10 `.kind::ti16`, `.collector::b::*`, and `.decompress::lut::b`
# grids are single-route asm. The production spec tables are intentionally
# not the source of any inventory here.

# The asm-tier operand schema, re-derived from §9.7.18.10:
#
#   [d], a-desc|[a-tmem], b-desc{, [meta]}, idesc{, [scale-A], [scale-B]}
#      {, {mask}}, enable-input-d{, scale-input-d}{, zero-column-mask}

const _T5_COLL_A = (nothing, Symbol("collector::a::lastuse"),
                    Symbol("collector::a::fill"), Symbol("collector::a::use"))
const _T5_COLL_B = (nothing, Symbol("collector::b::fill"),
                    Symbol("collector::b::use"), Symbol("collector::b::lastuse"))

_t5_opt(x) = x === nothing ? () : (x,)

# The expected PTX operand text for one asm-tier shape.
function _t5_asm_schema(; a_tmem::Bool, meta::Bool = false, mx::Bool = false,
                        maskN::Int = 0, zero_col::Bool = false,
                        scale = nothing)
    parts = String["[\$0]", a_tmem ? "[\$1]" : "\$1", "\$2"]
    k = 3
    if meta
        push!(parts, "[\$$k]"); k += 1
    end
    push!(parts, "\$$k"); k += 1
    if mx
        push!(parts, "[\$$k]", "[\$$(k + 1)]"); k += 2
    end
    if maskN > 0
        push!(parts, "{" * join(("\$$(k + i)" for i in 0:maskN - 1), ", ") * "}")
        k += maskN
    end
    push!(parts, "\$$k"); k += 1
    zero_col && (push!(parts, "\$$k"); k += 1)
    scale === nothing || push!(parts, string(scale))
    join(parts, ", ") * ";"
end

# One asm-tier method: exact dispatch, asm tier, no result, no intrinsic,
# the exact head + operand schema, sideeffect + memory clobber, and no
# convergence contract (tcgen05.mma is non-convergent in FORMS).
function _t5_asm_check(mods, args, schema)
    op = Operation{:tcgen05, mods}()
    @test which(op, args).module === PTX
    info = lowering(op, args)
    @test info.tier === :asm
    @test info.rettype === Nothing
    @test isempty(info.intrinsics)
    ci, _ = first(Base.code_typed(op, args))
    typed = replace(string(ci), "\\\$" => "\$")
    head = "tcgen05." * join(String.(mods), ".")
    @test occursin(head * " " * schema, typed)
    @test occursin("asm sideeffect", typed)
    @test occursin("~{memory}", typed)
    @test !occursin("convergent", typed)
    nothing
end

# A grammar or ABI miss: no reviewed method, and the call builder refuses.
function _t5_forbidden(mods, args)
    @test lowering(Operation{:tcgen05, mods}(), args).tier === :forbidden
    @test_throws ArgumentError PTX.build_call(:tcgen05, mods, args)
    nothing
end

# The ws B-side addressed collectors (b0..b3 × op, default b0::discard).
_t5_ws_colls() = begin
    out = Any[nothing]
    for buf in 0:3, op in (:discard, :lastuse, :fill, :use)
        buf == 0 && op === :discard && continue
        push!(out, Symbol("collector::b$buf::$op"))
    end
    out
end

# --- dense (PTX 9.3 §9.7.17.10) ------------------------------------------------

# Independent PTX 9.3 §9.7.17.10 dense-mma oracle: kinds with their NVVM
# immarg values and scale-input-d legality, the empirical collector enum
# (0=discard, 1=lastuse, 2=fill, 3=use), and the ashift restrictions
# (TMEM A only; collector limited to discard/lastuse). The production
# spec tables are intentionally not the source of this oracle.
const _T5_DENSE_KINDS =
    ((:f16, 0, true), (:tf32, 1, true), (:f8f6f4, 2, false), (:i8, 3, false))
const _T5_DENSE_COLLECTORS =
    ((nothing, 0), (Symbol("collector::a::lastuse"), 1),
     (Symbol("collector::a::fill"), 2), (Symbol("collector::a::use"), 3))

_t5_dense_mods(cg, kind; ashift = false, coll = nothing) =
    (:mma, Symbol("cta_group::", cg), Symbol("kind::", kind),
     (ashift ? (:ashift,) : ())...,
     (coll === nothing ? () : (coll,))...)

@testset "tcgen05 dense mma closed callable surface" begin
    reviewed = Set{String}()

    for (kind, kindval, scale_ok) in _T5_DENSE_KINDS, cg in (1, 2),
            (coll, collval) in _T5_DENSE_COLLECTORS,
            (tmem_a, ashift) in ((false, false), (true, false), (true, true))
        ashift && collval >= 2 && continue
        mods = _t5_dense_mods(cg, kind; ashift, coll)
        aT = tmem_a ? UInt32 : UInt64
        maskT = NTuple{cg == 1 ? 4 : 8, UInt32}
        stem = "llvm.nvvm.tcgen05.mma." * (tmem_a ? "tensor" : "shared")
        sh = ashift ? ".ashift" : ""

        shapes = [
            ((UInt32, aT, UInt64, UInt32, Bool), stem * sh),
            ((UInt32, aT, UInt64, UInt32, maskT, Bool),
             stem * ".disable_output_lane.cg$cg" * sh),
        ]
        scale_ok && append!(shapes, [
            ((UInt32, aT, UInt64, UInt32, Bool, Val{5}),
             stem * ".scale_d" * sh),
            ((UInt32, aT, UInt64, UInt32, maskT, Bool, Val{5}),
             stem * ".scale_d.disable_output_lane.cg$cg" * sh),
        ])

        for (args, intrinsic) in shapes
            push!(reviewed, intrinsic)
            info = lowering(Operation{:tcgen05, mods}(), args)
            @test info.tier === :intrinsic
            @test info.rettype === Nothing
            @test intrinsic in info.intrinsics
        end
    end

    # Closed world: the reviewed grid is exactly the registry's dense
    # inventory and the generated family's name table.
    registry = Set(name for name in keys(PTX.NVVM.TABLE)
                   if startswith(name, "llvm.nvvm.tcgen05.mma.") &&
                      !occursin(".sp.", name) && !occursin(".ws.", name) &&
                      !occursin("block_scale", name))
    @test reviewed == registry
    @test reviewed == Set(PTX.wrapper_intrinsic_names(:tcgen05_mma_dense))
    @test length(reviewed) == 18

    # The ashift-legal collector subset is pinned by the registry ranges:
    # every .ashift record restricts the collector immarg to [0, 2).
    for name in reviewed
        record = PTX.NVVM.intrinsic(name)
        hi = last(record.ranges[end])
        @test hi == (endswith(name, ".ashift") ? 2 : 4)
    end
end

@testset "tcgen05 dense mma rejects grammar and ABI misses" begin
    f16_1 = _t5_dense_mods(1, :f16)
    misses = (
        # ashift is TMEM-A only (a-desc form has no row shift).
        (_t5_dense_mods(1, :f16; ashift = true),
         (UInt32, UInt64, UInt64, UInt32, Bool)),
        # ashift forbids collector fill/use (ISA; registry range [0, 2)).
        (_t5_dense_mods(1, :f16; ashift = true,
                        coll = Symbol("collector::a::fill")),
         (UInt32, UInt32, UInt64, UInt32, Bool)),
        (_t5_dense_mods(1, :f16; ashift = true,
                        coll = Symbol("collector::a::use")),
         (UInt32, UInt32, UInt64, UInt32, Bool)),
        # Modifier order is kind{.ashift}{.collector}; collector-first is
        # not a reviewed spelling.
        ((:mma, Symbol("cta_group::1"), Symbol("kind::f16"),
          Symbol("collector::a::lastuse"), :ashift),
         (UInt32, UInt32, UInt64, UInt32, Bool)),
        # scale-input-d exists only for f16/tf32.
        (_t5_dense_mods(1, :i8),
         (UInt32, UInt64, UInt64, UInt32, Bool, Val{5})),
        (_t5_dense_mods(1, :f8f6f4),
         (UInt32, UInt64, UInt64, UInt32, Bool, Val{5})),
        # ... and must be a compile-time immediate.
        (f16_1, (UInt32, UInt64, UInt64, UInt32, Bool, Int64)),
        # The mask width is pinned by cta_group (4 vs 8 words) and sits
        # before enable-input-d.
        (f16_1, (UInt32, UInt64, UInt64, UInt32, NTuple{8, UInt32}, Bool)),
        (_t5_dense_mods(2, :f16),
         (UInt32, UInt64, UInt64, UInt32, NTuple{4, UInt32}, Bool)),
        (f16_1, (UInt32, UInt64, UInt64, UInt32, Bool, NTuple{4, UInt32})),
        # Operand carriers are pinned (no widening of the B descriptor).
        (f16_1, (UInt32, UInt64, UInt32, UInt32, Bool)),
    )

    for (mods, args) in misses
        _t5_forbidden(mods, args)
    end
end

# --- sparse (PTX 9.3 §9.7.17.10.9.2) ------------------------------------------

# Independent PTX 9.3 §9.7.17.10.9.2 sparse-mma oracle: the dense grid
# plus the sparsity-metadata TMEM operand between the B descriptor and
# idesc. The production spec tables are intentionally not the source.
const _T5_SP_KINDS =
    ((:f16, true), (:tf32, true), (:f8f6f4, false), (:i8, false))

_t5_sp_mods(cg, kind; ashift = false, coll = nothing) =
    (:mma, :sp, Symbol("cta_group::", cg), Symbol("kind::", kind),
     (ashift ? (:ashift,) : ())...,
     (coll === nothing ? () : (coll,))...)

@testset "tcgen05 sparse mma closed callable surface" begin
    reviewed = Set{String}()

    for (kind, scale_ok) in _T5_SP_KINDS, cg in (1, 2),
            (ci, coll) in enumerate(_T5_COLL_A),
            (tmem_a, ashift) in ((false, false), (true, false), (true, true))
        ashift && ci > 2 && continue
        mods = _t5_sp_mods(cg, kind; ashift, coll)
        aT = tmem_a ? UInt32 : UInt64
        maskT = NTuple{cg == 1 ? 4 : 8, UInt32}
        stem = "llvm.nvvm.tcgen05.mma.sp." * (tmem_a ? "tensor" : "shared")
        sh = ashift ? ".ashift" : ""

        shapes = [
            ((UInt32, aT, UInt64, UInt32, UInt32, Bool), stem * sh),
            ((UInt32, aT, UInt64, UInt32, UInt32, maskT, Bool),
             stem * ".disable_output_lane.cg$cg" * sh),
        ]
        scale_ok && append!(shapes, [
            ((UInt32, aT, UInt64, UInt32, UInt32, Bool, Val{5}),
             stem * ".scale_d" * sh),
            ((UInt32, aT, UInt64, UInt32, UInt32, maskT, Bool, Val{5}),
             stem * ".scale_d.disable_output_lane.cg$cg" * sh),
        ])

        for (args, intrinsic) in shapes
            push!(reviewed, intrinsic)
            info = lowering(Operation{:tcgen05, mods}(), args)
            @test info.tier === :intrinsic
            @test info.rettype === Nothing
            @test intrinsic in info.intrinsics
        end
    end

    # Closed world against the registry's non-block-scale sp inventory;
    # the sp block-scale (MX) records stay outside the wrapper surface
    # like their dense counterparts.
    registry = Set(name for name in keys(PTX.NVVM.TABLE)
                   if startswith(name, "llvm.nvvm.tcgen05.mma.sp.") &&
                      !occursin("block_scale", name))
    @test reviewed == registry
    @test reviewed == Set(PTX.wrapper_intrinsic_names(:tcgen05_mma_sp))
    @test length(reviewed) == 18

    # ashift records restrict the collector immarg to [0, 2).
    for name in reviewed
        record = PTX.NVVM.intrinsic(name)
        hi = last(record.ranges[end])
        @test hi == (endswith(name, ".ashift") ? 2 : 4)
    end
end

@testset "tcgen05 sparse mma rejects grammar and ABI misses" begin
    f16_1 = _t5_sp_mods(1, :f16)
    misses = (
        # The sparsity-metadata operand is mandatory and is a TMEM
        # address (UInt32), not a descriptor.
        (f16_1, (UInt32, UInt64, UInt64, UInt32, Bool)),
        (f16_1, (UInt32, UInt64, UInt64, UInt64, UInt32, Bool)),
        # ... and sits between the B descriptor and idesc, not trailing.
        (f16_1, (UInt32, UInt64, UInt64, UInt32, Bool, UInt32)),
        # ashift is TMEM-A only and forbids collector fill/use.
        (_t5_sp_mods(1, :f16; ashift = true),
         (UInt32, UInt64, UInt64, UInt32, UInt32, Bool)),
        (_t5_sp_mods(1, :f16; ashift = true,
                     coll = Symbol("collector::a::fill")),
         (UInt32, UInt32, UInt64, UInt32, UInt32, Bool)),
        # scale-input-d exists only for f16/tf32.
        (_t5_sp_mods(2, :i8),
         (UInt32, UInt64, UInt64, UInt32, UInt32, Bool, Val{5})),
        # Mask width is pinned by cta_group.
        (_t5_sp_mods(1, :f16),
         (UInt32, UInt64, UInt64, UInt32, UInt32, NTuple{8, UInt32}, Bool)),
        # sp block-scale (MX) is a wrapped asm family (matrix_api_safety
        # pins its schema); a kind × scale pair outside Table 60 stays a
        # grammar miss even with the correct sp arity.
        ((:mma, :sp, Symbol("cta_group::1"), Symbol("kind::mxf8f6f4"),
          :block_scale, Symbol("scale_vec::2X")),
         (UInt32, UInt64, UInt64, UInt32, UInt32, UInt32, UInt32, Bool)),
    )

    for (mods, args) in misses
        _t5_forbidden(mods, args)
    end
end

# --- weight-stationary (PTX 9.3 §9.7.17.10.9.3/.4) ------------------------------

# Independent PTX 9.3 §9.7.17.10.9.3/.4 weight-stationary oracle:
# cta_group::1 only, B-side addressed collector (b0..b3 × op, default
# b0::discard), optional runtime zero-column-mask descriptor, sp
# metadata between the B descriptor and idesc. The production spec
# tables are intentionally not the source of this oracle.
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
        _t5_forbidden(mods, args)
    end
end

# --- .kind::ti16 (PTX 9.4 §9.7.18.10) ----------------------------------------------

# Independent PTX ISA 9.4 §9.7.18.10 `.kind::ti16` oracle: the integer
# schema (no scale-input-d) with an optional B-side collector on dense and
# sparse forms, the addressed `collector::bN::op` on ws/ws.sp, and the
# dense/sp ashift restrictions. The production grid is not the source.
const _T5_TI16 = Symbol("kind::ti16")

_t5_ti16_mods(cg; sp = false, ashift = false, a = nothing, b = nothing) =
    (:mma, (sp ? (:sp,) : ())..., Symbol("cta_group::", cg), _T5_TI16,
     (ashift ? (:ashift,) : ())..., _t5_opt(a)..., _t5_opt(b)...)

@testset "tcgen05 ti16 mma closed callable surface" begin
    expected = Set{Tuple{Vararg{Symbol}}}()
    methods_checked = 0
    for cg in (1, 2), sp in (false, true), ashift in (false, true),
            a in (ashift ? _T5_COLL_A[1:2] : _T5_COLL_A),
            b in _T5_COLL_B
        mods = _t5_ti16_mods(cg; sp, ashift, a, b)
        push!(expected, mods)
        maskN = cg == 1 ? 4 : 8
        meta = sp ? (UInt32,) : ()
        for a_tmem in (ashift ? (true,) : (false, true))
            aT = a_tmem ? UInt32 : UInt64
            _t5_asm_check(mods, (UInt32, aT, UInt64, meta..., UInt32, Bool),
                         _t5_asm_schema(; a_tmem, meta = sp))
            _t5_asm_check(mods,
                         (UInt32, aT, UInt64, meta..., UInt32,
                          NTuple{maskN, UInt32}, Bool),
                         _t5_asm_schema(; a_tmem, meta = sp, maskN))
            methods_checked += 2
        end
    end
    for sp in (false, true), coll in _t5_ws_colls()
        mods = (:mma, :ws, (sp ? (:sp,) : ())..., Symbol("cta_group::1"),
                _T5_TI16, _t5_opt(coll)...)
        push!(expected, mods)
        meta = sp ? (UInt32,) : ()
        for a_tmem in (false, true)
            aT = a_tmem ? UInt32 : UInt64
            _t5_asm_check(mods, (UInt32, aT, UInt64, meta..., UInt32, Bool),
                         _t5_asm_schema(; a_tmem, meta = sp))
            _t5_asm_check(mods,
                         (UInt32, aT, UInt64, meta..., UInt32, Bool, UInt64),
                         _t5_asm_schema(; a_tmem, meta = sp, zero_col = true))
            methods_checked += 2
        end
    end
    @test length(expected) == 128
    @test methods_checked == 448
    @test Set(PTX.wrapper_asm_forms(:tcgen05_mma_ti16)) == expected
    @test count(m -> :ws in m, expected) == 32
    @test count(m -> :sp in m && :ws ∉ m, expected) == 48
end

@testset "tcgen05 ti16 mma rejects grammar and ABI misses" begin
    base = _t5_ti16_mods(1; b = Symbol("collector::b::fill"))
    misses = (
        # No scale-input-d on the integer kinds.
        (base, (UInt32, UInt64, UInt64, UInt32, Bool, Val{5})),
        # ashift is TMEM-A only and forbids A fill/use.
        (_t5_ti16_mods(1; ashift = true), (UInt32, UInt64, UInt64, UInt32, Bool)),
        (_t5_ti16_mods(1; ashift = true, a = Symbol("collector::a::fill")),
         (UInt32, UInt32, UInt64, UInt32, Bool)),
        # The B collector follows the A collector; discard is never spelled.
        ((:mma, Symbol("cta_group::1"), _T5_TI16, Symbol("collector::b::fill"),
          Symbol("collector::a::fill")), (UInt32, UInt64, UInt64, UInt32, Bool)),
        (_t5_ti16_mods(1; b = Symbol("collector::b::discard")),
         (UInt32, UInt64, UInt64, UInt32, Bool)),
        # .kind::i8 has no B collector (§9.7.18.10 blocks 5/6).
        ((:mma, Symbol("cta_group::1"), Symbol("kind::i8"),
          Symbol("collector::b::fill")), (UInt32, UInt64, UInt64, UInt32, Bool)),
        # sp metadata is mandatory and sits before idesc.
        (_t5_ti16_mods(1; sp = true), (UInt32, UInt64, UInt64, UInt32, Bool)),
        # Mask width follows cta_group.
        (base, (UInt32, UInt64, UInt64, UInt32, NTuple{8, UInt32}, Bool)),
        (_t5_ti16_mods(2; b = Symbol("collector::b::use")),
         (UInt32, UInt64, UInt64, UInt32, NTuple{4, UInt32}, Bool)),
        # ws is cta_group::1 with the addressed collector; the dense
        # spelling is not ws grammar and ws has no mask.
        ((:mma, :ws, Symbol("cta_group::2"), _T5_TI16),
         (UInt32, UInt64, UInt64, UInt32, Bool)),
        ((:mma, :ws, Symbol("cta_group::1"), _T5_TI16,
          Symbol("collector::b::fill")), (UInt32, UInt64, UInt64, UInt32, Bool)),
        ((:mma, :ws, Symbol("cta_group::1"), _T5_TI16),
         (UInt32, UInt64, UInt64, UInt32, NTuple{4, UInt32}, Bool)),
        # Block-scale has no ti16 kind.
        ((:mma, Symbol("cta_group::1"), _T5_TI16, :block_scale, :block32),
         (UInt32, UInt64, UInt64, UInt32, UInt32, UInt32, Bool)),
    )
    for (mods, args) in misses
        _t5_forbidden(mods, args)
    end
end

# --- .collector::b::* on the float and block-scale kinds (PTX 9.4 §9.7.18.10) ---

# Independent PTX ISA 9.4 §9.7.18.10 `.collector::b::*` oracle for the
# non-ws kinds: the float kinds (blocks 3, dense and sp) with every A
# collector, the ashift restriction, scale-input-d on f16/tf32 only, and
# the block-scale family spellings (block 4). The production grid is not
# the source.
const _T5_CB = _T5_COLL_B[2:end]
const _T5_CB_MX = ((:mxf8f6f4, :block32), (:mxf4, :block32),
                   (:mxf4nvf4, :block32), (:mxf4nvf4, :block16))

_t5_cb_mods(kind, cg; sp = false, ashift = false, a = nothing, b) =
    (:mma, (sp ? (:sp,) : ())..., Symbol("cta_group::", cg),
     Symbol("kind::", kind), (ashift ? (:ashift,) : ())...,
     _t5_opt(a)..., b)

_t5_cb_mx_mods(kind, block, cg; sp = false, a = nothing, b) =
    (:mma, (sp ? (:sp,) : ())..., Symbol("cta_group::", cg),
     Symbol("kind::", kind), :block_scale, block, _t5_opt(a)..., b)

@testset "tcgen05 collector::b mma closed callable surface" begin
    expected = Set{Tuple{Vararg{Symbol}}}()
    methods_checked = 0
    for kind in (:f16, :tf32, :f8f6f4), cg in (1, 2), sp in (false, true),
            ashift in (false, true),
            a in (ashift ? _T5_COLL_A[1:2] : _T5_COLL_A), b in _T5_CB
        mods = _t5_cb_mods(kind, cg; sp, ashift, a, b)
        push!(expected, mods)
        maskN = cg == 1 ? 4 : 8
        meta = sp ? (UInt32,) : ()
        scale_ok = kind in (:f16, :tf32)
        for a_tmem in (ashift ? (true,) : (false, true))
            aT = a_tmem ? UInt32 : UInt64
            _t5_asm_check(mods, (UInt32, aT, UInt64, meta..., UInt32, Bool),
                         _t5_asm_schema(; a_tmem, meta = sp))
            _t5_asm_check(mods,
                         (UInt32, aT, UInt64, meta..., UInt32,
                          NTuple{maskN, UInt32}, Bool),
                         _t5_asm_schema(; a_tmem, meta = sp, maskN))
            methods_checked += 2
            scale_ok || continue
            _t5_asm_check(mods,
                         (UInt32, aT, UInt64, meta..., UInt32, Bool, Val{5}),
                         _t5_asm_schema(; a_tmem, meta = sp, scale = 5))
            _t5_asm_check(mods,
                         (UInt32, aT, UInt64, meta..., UInt32,
                          NTuple{maskN, UInt32}, Bool, Val{15}),
                         _t5_asm_schema(; a_tmem, meta = sp, maskN, scale = 15))
            methods_checked += 2
        end
    end
    @test length(expected) == 216
    @test methods_checked == 1200

    mx_expected = Set{Tuple{Vararg{Symbol}}}()
    for (kind, block) in _T5_CB_MX, cg in (1, 2), sp in (false, true),
            a in _T5_COLL_A, b in _T5_CB
        mods = _t5_cb_mx_mods(kind, block, cg; sp, a, b)
        push!(mx_expected, mods)
        meta = sp ? (UInt32,) : ()
        for a_tmem in (false, true)
            aT = a_tmem ? UInt32 : UInt64
            _t5_asm_check(mods,
                         (UInt32, aT, UInt64, meta..., UInt32, UInt32, UInt32,
                          Bool),
                         _t5_asm_schema(; a_tmem, meta = sp, mx = true))
            methods_checked += 1
        end
    end
    @test length(mx_expected) == 192
    @test methods_checked == 1584
    @test Set(PTX.wrapper_asm_forms(:tcgen05_mma_collb)) ==
          union(expected, mx_expected)
    # The pre-9.4 block-scale inventory is untouched.
    @test length(PTX.wrapper_asm_forms(:tcgen05_mx)) == 128
end

@testset "tcgen05 collector::b mma rejects grammar and ABI misses" begin
    fill_b = Symbol("collector::b::fill")
    f16 = _t5_cb_mods(:f16, 1; b = fill_b)
    misses = (
        # .kind::i8 has no B collector.
        (_t5_cb_mods(:i8, 1; b = fill_b), (UInt32, UInt64, UInt64, UInt32, Bool)),
        # discard is the unspelled default; the B collector follows A.
        (_t5_cb_mods(:f16, 1; b = Symbol("collector::b::discard")),
         (UInt32, UInt64, UInt64, UInt32, Bool)),
        ((:mma, Symbol("cta_group::1"), Symbol("kind::f16"), fill_b,
          Symbol("collector::a::fill")), (UInt32, UInt64, UInt64, UInt32, Bool)),
        # ashift: TMEM A only, no A fill/use.
        (_t5_cb_mods(:f16, 1; ashift = true, b = fill_b),
         (UInt32, UInt64, UInt64, UInt32, Bool)),
        (_t5_cb_mods(:f16, 1; ashift = true, a = Symbol("collector::a::use"),
                     b = fill_b), (UInt32, UInt32, UInt64, UInt32, Bool)),
        # scale-input-d: f16/tf32 only, an immediate in 0:15.
        (_t5_cb_mods(:f8f6f4, 1; b = fill_b),
         (UInt32, UInt64, UInt64, UInt32, Bool, Val{5})),
        (f16, (UInt32, UInt64, UInt64, UInt32, Bool, Int64)),
        # Mask width follows cta_group and precedes enable-input-d.
        (f16, (UInt32, UInt64, UInt64, UInt32, NTuple{8, UInt32}, Bool)),
        (f16, (UInt32, UInt64, UInt64, UInt32, Bool, NTuple{4, UInt32})),
        # sp metadata is mandatory.
        (_t5_cb_mods(:f16, 1; sp = true, b = fill_b),
         (UInt32, UInt64, UInt64, UInt32, Bool)),
        # Block-scale: the family spellings only, both scale addresses
        # mandatory, no mask.
        (_t5_cb_mx_mods(:mxf8f6f4, Symbol("scale_vec::1X"), 1; b = fill_b),
         (UInt32, UInt64, UInt64, UInt32, UInt32, UInt32, Bool)),
        (_t5_cb_mx_mods(:mxf4nvf4, Symbol("scale_vec::4X"), 1; b = fill_b),
         (UInt32, UInt64, UInt64, UInt32, UInt32, UInt32, Bool)),
        (_t5_cb_mx_mods(:mxf4, :block16, 1; b = fill_b),
         (UInt32, UInt64, UInt64, UInt32, UInt32, UInt32, Bool)),
        (_t5_cb_mx_mods(:mxf8f6f4, :block32, 1; b = fill_b),
         (UInt32, UInt64, UInt64, UInt32, Bool)),
        (_t5_cb_mx_mods(:mxf8f6f4, :block32, 1; b = fill_b),
         (UInt32, UInt64, UInt64, UInt32, NTuple{4, UInt32}, UInt32, UInt32,
          Bool)),
    )
    for (mods, args) in misses
        _t5_forbidden(mods, args)
    end

    # An out-of-range scale immediate is rejected before any code is
    # emitted.
    op = Operation{:tcgen05, f16}()
    @test_throws ArgumentError op(UInt32(0), UInt64(0), UInt64(0), UInt32(0),
                                  false, Val(16))
    @test_throws ArgumentError op(UInt32(0), UInt64(0), UInt64(0), UInt32(0),
                                  false, Val(-1))
end

# --- .decompress::lut::b (PTX 9.4 §9.7.18.10 block 7) ------------------------------

# Independent PTX ISA 9.4 §9.7.18.10 `.decompress::lut::b` oracle (syntax
# block 7): dense only, `.kind::f8f6f4` and `.kind::mxf8f6f4.block_scale`
# with the family `.block32` spelling, both collectors optional, and the
# LUT's TMEM address after the compressed B descriptor. The production
# grid is not the source.
const _T5_LUT = Symbol("decompress::lut::b")

_t5_lut_mods(cg; a = nothing, b = nothing) =
    (:mma, Symbol("cta_group::", cg), Symbol("kind::f8f6f4"), _T5_LUT,
     _t5_opt(a)..., _t5_opt(b)...)
_t5_lut_mx_mods(cg; a = nothing, b = nothing) =
    (:mma, Symbol("cta_group::", cg), Symbol("kind::mxf8f6f4"), :block_scale,
     _T5_LUT, :block32, _t5_opt(a)..., _t5_opt(b)...)

@testset "tcgen05 lut::b mma closed callable surface" begin
    expected = Set{Tuple{Vararg{Symbol}}}()
    methods_checked = 0
    for cg in (1, 2), a in _T5_COLL_A, b in _T5_COLL_B
        maskN = cg == 1 ? 4 : 8
        mods = _t5_lut_mods(cg; a, b)
        push!(expected, mods)
        for a_tmem in (false, true)
            aT = a_tmem ? UInt32 : UInt64
            _t5_asm_check(mods, (UInt32, aT, UInt64, UInt32, UInt32, Bool),
                         _t5_asm_schema(; a_tmem, meta = true))
            _t5_asm_check(mods,
                         (UInt32, aT, UInt64, UInt32, UInt32,
                          NTuple{maskN, UInt32}, Bool),
                         _t5_asm_schema(; a_tmem, meta = true, maskN))
            methods_checked += 2
        end
        mods = _t5_lut_mx_mods(cg; a, b)
        push!(expected, mods)
        for a_tmem in (false, true)
            aT = a_tmem ? UInt32 : UInt64
            _t5_asm_check(mods,
                         (UInt32, aT, UInt64, UInt32, UInt32, UInt32, UInt32,
                          Bool),
                         _t5_asm_schema(; a_tmem, meta = true, mx = true))
            methods_checked += 1
        end
    end
    @test length(expected) == 64
    @test methods_checked == 192
    @test Set(PTX.wrapper_asm_forms(:tcgen05_mma_lut)) == expected
end

@testset "tcgen05 lut::b mma rejects grammar and ABI misses" begin
    plain = _t5_lut_mods(1)
    misses = (
        # Dense only: no sp, no ashift, no scale-input-d.
        ((:mma, :sp, Symbol("cta_group::1"), Symbol("kind::f8f6f4"), _T5_LUT),
         (UInt32, UInt64, UInt64, UInt32, UInt32, Bool)),
        ((:mma, Symbol("cta_group::1"), Symbol("kind::f8f6f4"), _T5_LUT,
          :ashift), (UInt32, UInt32, UInt64, UInt32, UInt32, Bool)),
        (plain, (UInt32, UInt64, UInt64, UInt32, UInt32, Bool, Val{5})),
        # The LUT metadata address is mandatory.
        (plain, (UInt32, UInt64, UInt64, UInt32, Bool)),
        # Only f8f6f4 and mxf8f6f4 decompress B; the block-scale form uses
        # the family spelling only.
        ((:mma, Symbol("cta_group::1"), Symbol("kind::f16"), _T5_LUT),
         (UInt32, UInt64, UInt64, UInt32, UInt32, Bool)),
        ((:mma, Symbol("cta_group::1"), Symbol("kind::mxf4"), :block_scale,
          _T5_LUT, :block32),
         (UInt32, UInt64, UInt64, UInt32, UInt32, UInt32, UInt32, Bool)),
        ((:mma, Symbol("cta_group::1"), Symbol("kind::mxf8f6f4"), :block_scale,
          _T5_LUT, Symbol("scale_vec::1X")),
         (UInt32, UInt64, UInt64, UInt32, UInt32, UInt32, UInt32, Bool)),
        # Block-scale lut takes no mask and both scale addresses.
        (_t5_lut_mx_mods(1),
         (UInt32, UInt64, UInt64, UInt32, UInt32, NTuple{4, UInt32}, UInt32,
          UInt32, Bool)),
        (_t5_lut_mx_mods(1), (UInt32, UInt64, UInt64, UInt32, UInt32, Bool)),
        # Modifier order: lut before the collectors.
        ((:mma, Symbol("cta_group::1"), Symbol("kind::f8f6f4"),
          Symbol("collector::b::fill"), _T5_LUT),
         (UInt32, UInt64, UInt64, UInt32, UInt32, Bool)),
    )
    for (mods, args) in misses
        _t5_forbidden(mods, args)
    end
end
