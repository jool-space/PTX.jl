using PTX
using PTX: Operation, RawOperation, Address, lowering, build_call
include(joinpath(@__DIR__, "..", "tma_defs.jl"))

# Closed-world host oracles for the tensor-copy (TMA) wrapper surface: the
# PTX 9.3 §9.7.9.26.5.4 tile and base-im2col prefetch inventories, and the
# PTX ISA 9.4 §9.7.10.28/§9.7.10.19 additions whose inventory tma_defs.jl
# re-derives from the ISA syntax blocks. The wrapper generators are not the
# source of any count here.

# --- tile prefetch (PTX 9.3 §9.7.9.26.5.4) ------------------------------------

# Independent PTX 9.3 §9.7.9.26.5.4 tile-only inventory. The production
# methods are intentionally not the source of this oracle.
const _TMA_PREFETCH_FORMS = [
    (Symbol("1d"), 1, "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.1d"),
    (Symbol("2d"), 2, "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.2d"),
    (Symbol("3d"), 3, "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.3d"),
    (Symbol("4d"), 4, "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.4d"),
    (Symbol("5d"), 5, "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.5d"),
]

_tma_prefetch_mods(rank; hint = false) =
    (:async, :bulk, :prefetch, :tensor, rank, :L2, :global, :tile,
     (hint ? (Symbol("L2::cache_hint"),) : ())...)

@testset "TMA tile-prefetch closed callable surface" begin
    tmap = PTX.TMADescriptorPtr
    reviewed = Set{String}()

    for (rank, ncoords, intrinsic) in _TMA_PREFETCH_FORMS
        push!(reviewed, intrinsic)
        coords = ntuple(_ -> Int32, ncoords)
        for hint in (false, true)
            mods = _tma_prefetch_mods(rank; hint)
            args = hint ? (tmap, coords..., UInt64) : (tmap, coords...)
            # Single-route asm since the demotion (see wrappers/tma.jl):
            # every form is convergent inline asm with the full clobber.
            info = lowering(Operation{:cp, mods}(), args)
            @test info.tier === :asm
            @test info.rettype === Nothing
            @test isempty(info.intrinsics)

            # The unwrapped intrinsic records stay in the pinned registry;
            # the attribute review below is what the demotion walked away
            # from (convergent + a readonly descriptor operand — no
            # stronger promise than the asm route).
            record = PTX.NVVM.intrinsic(intrinsic)
            @test record.ret == ()
            @test :convergent in record.props
            @test !(:nomem in record.props)
            @test record.immargs == (ncoords + 3,)
            @test (1, :readonly) in record.argattrs
        end
    end

    registry = Set(name for name in keys(PTX.NVVM.TABLE)
                   if startswith(name,
                       "llvm.nvvm.cp.async.bulk.tensor.prefetch.tile.") &&
                      !occursin("gather", name))
    @test reviewed == registry
    @test length(reviewed) == 5
end

@testset "TMA tile-prefetch rejects grammar and ABI misses" begin
    tmap = PTX.TMADescriptorPtr
    pglobal = Core.LLVMPtr{UInt8, PTX.AS.Global}
    base2 = _tma_prefetch_mods(Symbol("2d"))
    hint2 = _tma_prefetch_mods(Symbol("2d"); hint = true)

    misses = (
        # Coordinate count is exactly the declared rank.
        (base2, (tmap, Int32)),
        (base2, (tmap, Int32, Int32, Int32)),
        # Cache qualifier and u64 policy operand are an inseparable pair.
        (base2, (tmap, Int32, Int32, UInt64)),
        (hint2, (tmap, Int32, Int32)),
        (hint2, (tmap, Int32, Int32, UInt32)),
        (hint2, (tmap, Int32, Int32, Int64)),
        # The package convention requires a descriptor carrier, not an AS1
        # data pointer. The wrapper raw-retypes AS.Const to generic AS0.
        (base2, (pglobal, Int32, Int32)),
        # Rank and canonical modifier order are closed.
        ((:async, :bulk, :prefetch, :tensor, Symbol("0d"), :L2,
          :global, :tile), (tmap,)),
        ((:async, :bulk, :prefetch, :tensor, Symbol("6d"), :L2,
          :global, :tile), (tmap, Int32, Int32, Int32, Int32, Int32, Int32)),
        ((:async, :bulk, :prefetch, :tensor, Symbol("2d"), :global,
          :L2, :tile), (tmap, Int32, Int32)),
        ((:async, :bulk, :prefetch, :tensor, Symbol("2d"), :L2,
          :global, Symbol("L2::cache_hint"), :tile),
         (tmap, Int32, Int32, UInt64)),
        # Other load-mode islands stay deliberately unimplemented here.
        ((:async, :bulk, :prefetch, :tensor, Symbol("2d"), :L2,
          :global, Symbol("tile::gather4")),
         (tmap, Int32, Int32, Int32, Int32, Int32)),
        ((:async, :bulk, :prefetch, :tensor, Symbol("3d"), :L2,
          :global, Symbol("im2col::w")),
         (tmap, Int32, Int32, Int32, UInt16)),
    )

    for (mods, args) in misses
        @test lowering(Operation{:cp, mods}(), args).tier === :forbidden
        @test_throws ArgumentError PTX.build_call(:cp, mods, args)
    end
end

# --- base-im2col prefetch (PTX 9.3 §9.7.9.26.5.4) -----------------------------

# Independent PTX 9.3 §9.7.9.26.5.4 base-im2col inventory. Do not derive
# ranks, coordinate/offset counts, or intrinsic names from the wrapper methods.
const _TMA_IM2COL_PREFETCH_FORMS = [
    (Symbol("3d"), 3, 1,
     "llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.3d"),
    (Symbol("4d"), 4, 2,
     "llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.4d"),
    (Symbol("5d"), 5, 3,
     "llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.5d"),
]

_tma_im2col_prefetch_mods(rank; hint = false) =
    (:async, :bulk, :prefetch, :tensor, rank, :L2, :global, :im2col,
     (hint ? (Symbol("L2::cache_hint"),) : ())...)

@testset "TMA base-im2col prefetch closed callable surface" begin
    reviewed = Set{String}()

    for (rank, ncoords, noffsets, intrinsic) in _TMA_IM2COL_PREFETCH_FORMS
        push!(reviewed, intrinsic)
        coords = ntuple(_ -> Int32, ncoords)
        offsets = ntuple(_ -> Int16, noffsets)
        for hint in (false, true)
            mods = _tma_im2col_prefetch_mods(rank; hint)
            args = hint ?
                (PTX.TMADescriptorPtr, coords..., offsets..., UInt64) :
                (PTX.TMADescriptorPtr, coords..., offsets...)
            op = Operation{:cp, mods}()
            @test which(op, args).module === PTX

            # Single-route asm since the demotion (see wrappers/tma.jl):
            # every form is convergent inline asm with the full clobber.
            info = lowering(op, args)
            @test info.tier === :asm
            @test info.rettype === Nothing
            @test isempty(info.intrinsics)

            # The unwrapped intrinsic records stay in the pinned registry;
            # the attribute review below is what the demotion walked away
            # from (convergent + a readonly descriptor operand — no
            # stronger promise than the asm route).
            record = PTX.NVVM.intrinsic(intrinsic)
            @test record.ret == ()
            @test :convergent in record.props
            @test !(:nomem in record.props)
            @test record.immargs == (2ncoords + 1,)
            @test (1, :readonly) in record.argattrs
        end
    end

    registry = Set(name for name in keys(PTX.NVVM.TABLE)
                   if startswith(name,
                       "llvm.nvvm.cp.async.bulk.tensor.prefetch.im2col.") &&
                      !occursin(".im2col.w.", name))
    @test reviewed == registry
    @test length(reviewed) == 3
end

@testset "TMA base-im2col prefetch rejects grammar and ABI misses" begin
    rank3 = Symbol("3d")
    base3 = _tma_im2col_prefetch_mods(rank3)
    hint3 = _tma_im2col_prefetch_mods(rank3; hint = true)
    pglobal = Core.LLVMPtr{UInt8, PTX.AS.Global}
    pconst16 = Core.LLVMPtr{UInt16, PTX.AS.Const}

    misses = (
        # Rank 3 requires exactly three s32 coordinates and one s16 offset.
        (base3, (PTX.TMADescriptorPtr, Int32, Int32, Int16)),
        (base3, (PTX.TMADescriptorPtr, Int32, Int32, Int32,
                 Int32, Int16)),
        (base3, (PTX.TMADescriptorPtr, UInt32, Int32, Int32, Int16)),
        (base3, (PTX.TMADescriptorPtr, Int64, Int32, Int32, Int16)),
        (base3, (PTX.TMADescriptorPtr, Int32, Int32, Int32, UInt16)),
        (base3, (PTX.TMADescriptorPtr, Int32, Int32, Int32, Int32)),
        # Cache qualifier and exact u64 policy carrier are inseparable.
        (base3, (PTX.TMADescriptorPtr, Int32, Int32, Int32,
                 Int16, UInt64)),
        (hint3, (PTX.TMADescriptorPtr, Int32, Int32, Int32, Int16)),
        (hint3, (PTX.TMADescriptorPtr, Int32, Int32, Int32,
                 Int16, UInt32)),
        # A tensor-map descriptor is the package's AS.Const UInt8 carrier.
        (base3, (pglobal, Int32, Int32, Int32, Int16)),
        (base3, (pconst16, Int32, Int32, Int32, Int16)),
        # Rank and canonical modifier order are closed.
        ((:async, :bulk, :prefetch, :tensor, Symbol("2d"), :L2,
          :global, :im2col),
         (PTX.TMADescriptorPtr, Int32, Int32)),
        ((:async, :bulk, :prefetch, :tensor, Symbol("6d"), :L2,
          :global, :im2col),
         (PTX.TMADescriptorPtr, Int32, Int32, Int32, Int32, Int32, Int32,
          Int16, Int16, Int16, Int16)),
        ((:async, :bulk, :prefetch, :tensor, rank3, :global, :L2, :im2col),
         (PTX.TMADescriptorPtr, Int32, Int32, Int32, Int16)),
        ((:async, :bulk, :prefetch, :tensor, rank3, :L2, :global,
          Symbol("L2::cache_hint"), :im2col),
         (PTX.TMADescriptorPtr, Int32, Int32, Int32, Int16, UInt64)),
        # Later modes have distinct ABIs/targets and remain deliberately absent.
        ((:async, :bulk, :prefetch, :tensor, rank3, :L2, :global,
          Symbol("im2col::w")),
         (PTX.TMADescriptorPtr, Int32, Int32, Int32, Int16, Int16)),
        ((:async, :bulk, :prefetch, :tensor, rank3, :L2, :global,
          Symbol("im2col::w::128")),
         (PTX.TMADescriptorPtr, Int32, Int32, Int32, Int16, Int16)),
        ((:async, :bulk, :prefetch, :tensor, Symbol("2d"), :L2,
          :global, Symbol("tile::gather4")),
         (PTX.TMADescriptorPtr, Int32, Int32, Int32, Int32, Int32)),
    )

    for (mods, args) in misses
        info = lowering(Operation{:cp, mods}(), args)
        @test info.tier === :forbidden
        @test_throws ArgumentError PTX.build_call(:cp, mods, args)
    end
end

# --- PTX ISA 9.4 tensor-copy forms (§9.7.10.28, §9.7.10.19) --------------------

@testset "PTX ISA 9.4 tensor-copy closed callable surface" begin
    expected = (A = 20, B = 60, C = 20, D = 30, E = 10, F = 6, G = 10, H = 8,
                I = 10, J = 18, K = 40, L = 48, M = 80)
    for (name, count) in pairs(expected)
        entries = getproperty(_TMA94_LOOPS, name)
        @test length(entries) == count
        for (op, mods, kinds, n) in entries
            argtypes = _tma94_argtypes((op, mods, kinds, n))
            o = Operation{op, mods}()
            @test which(o, argtypes).module === PTX
            info = lowering(o, argtypes)
            @test info.tier === :asm
            @test info.rettype === Nothing
            @test isempty(info.intrinsics)
            ci, rt = first(Base.code_typed(o, argtypes))
            code = string(ci)
            @test rt === Nothing
            @test occursin(PTX.build_head(op, mods) * " [", code)
            @test occursin("~{memory}", code)
            @test !occursin("llvm.nvvm", code)
        end
    end
    total = sum(length(getproperty(_TMA94_LOOPS, k)) for k in keys(expected))
    @test total == 360
    @test length(unique((e[1], e[2]) for k in keys(expected)
                        for e in getproperty(_TMA94_LOOPS, k))) == total

    # Six report spellings, each on both load directions at every rank.
    reports = vcat(_TMA94_LOOPS.B, _TMA94_LOOPS.D)
    @test count(e -> any(m -> startswith(String(m), "mbarrier::report::"),
                         e[2]), reports) == 90
    @test length(unique(m for e in reports for m in e[2]
                        if startswith(String(m), "mbarrier::report::"))) == 6
end

# The typed IR quotes the asm template, so `$` slots print escaped.
_tma94_code(o, argtypes) =
    replace(string(first(Base.code_typed(o, argtypes))[1]), "\\\$" => "\$")

@testset "PTX ISA 9.4 tensor-copy operand rendering" begin
    tmap = PTX.TMADescriptorPtr
    pS = Core.LLVMPtr{UInt16, PTX.AS.Shared}
    pM = Core.LLVMPtr{UInt64, PTX.AS.Shared}
    pG = Core.LLVMPtr{UInt8, PTX.AS.Global}
    cluster = _TMA94_CLUSTER
    complete = _TMA94_COMPLETE

    # Attribute override on a 2d cluster load: address, two dims, one lower
    # stride, packed upper strides, then the coordinates; ::32b mask last.
    mods = (:async, :bulk, :tensor, Symbol("2d"), cluster, :global, :tile,
            complete, _TMA94_MC32, _TMA94_OVADDR,
            Symbol("override::global_dim_stride"))
    code = _tma94_code(Operation{:cp, mods}(),
        (pS, tmap, pG, NTuple{2, UInt8}, NTuple{1, UInt32}, UInt16,
         Int32, Int32, pM, UInt32))
    @test occursin("[\$0], [\$1, \$2, {\$3, \$4}, {\$5}, \$6, {\$7, \$8}], [\$9], \$10;",
                   code)
    @test occursin("r,l,l,h,h,r,h,r,r,r,r,~{memory}", code)

    # 1d attribute override: one dimension byte, no strides.
    mods = (:async, :bulk, :tensor, Symbol("1d"), _TMA94_CTA, :global, :tile,
            complete, _TMA94_OVADDR, Symbol("override::global_dim"))
    code = _tma94_code(Operation{:cp, mods}(),
                       (pS, tmap, pG, NTuple{1, UInt8}, Int32, pM))
    @test occursin("[\$0], [\$1, \$2, {\$3}, {\$4}], [\$5];", code)

    # Store with the address override only; im2col eviction hint with offsets.
    mods = (:async, :bulk, :tensor, Symbol("1d"), :global, _TMA94_CTA, :tile,
            :bulk_group, _TMA94_OVADDR)
    code = _tma94_code(Operation{:cp, mods}(), (tmap, pG, Int32, pS))
    @test occursin("bulk_group.override::global_address [\$0, \$1, {\$2}], [\$3];",
                   code)
    mods = (:async, :bulk, :tensor, Symbol("3d"), :global, :bulk_group,
            :im2col, _TMA94_EVICT_NORMAL)
    code = _tma94_code(Operation{:applypriority, mods}(),
                       (tmap, Int32, Int32, Int32, Int16))
    @test occursin("applypriority.async.bulk.tensor.3d.global.bulk_group.im2col.L2::evict_normal [\$0, {\$1, \$2, \$3}], {\$4};",
                   code)

    # Mask carriers follow the multicast width.
    mods16 = (:async, :bulk, :tensor, Symbol("2d"), cluster, :global, :tile,
              complete, _TMA94_MC16)
    mods32 = (:async, :bulk, :tensor, Symbol("2d"), cluster, :global, :tile,
              complete, _TMA94_MC32)
    @test occursin("r,l,r,r,r,h,~{memory}",
                   string(first(Base.code_typed(Operation{:cp, mods16}(),
                                                (pS, tmap, Int32, Int32, pM, UInt16)))[1]))
    @test occursin("r,l,r,r,r,r,~{memory}",
                   string(first(Base.code_typed(Operation{:cp, mods32}(),
                                                (pS, tmap, Int32, Int32, pM, UInt32)))[1]))
end

@testset "PTX ISA 9.4 tensor-copy rejects grammar and ABI misses" begin
    tmap = PTX.TMADescriptorPtr
    pS = Core.LLVMPtr{UInt16, PTX.AS.Shared}
    pM = Core.LLVMPtr{UInt64, PTX.AS.Shared}
    pG = Core.LLVMPtr{UInt8, PTX.AS.Global}
    A32, A64 = Address{UInt32}, Address{UInt64}
    cluster, cta, complete = _TMA94_CLUSTER, _TMA94_CTA, _TMA94_COMPLETE
    d1, d2, d3 = Symbol("1d"), Symbol("2d"), Symbol("3d")
    dim, dimstride = Symbol("override::global_dim"), Symbol("override::global_dim_stride")

    # Not in the exact surface: the wrapper has no method, and an integer
    # address routes the miss to the structured-address boundary.
    not_wrapped = (
        # The attribute override is rank-determined (Table 34).
        ((:async, :bulk, :tensor, d1, cluster, :global, :tile, complete,
          _TMA94_OVADDR, dimstride), (A32, A64, A64, NTuple{1, UInt8}, Int32, A32)),
        ((:async, :bulk, :tensor, d2, cluster, :global, :tile, complete,
          _TMA94_OVADDR, dim), (A32, A64, A64, NTuple{2, UInt8}, Int32, Int32, A32)),
        # The attribute override requires the address override.
        ((:async, :bulk, :tensor, d2, cluster, :global, :tile, complete,
          dimstride), (A32, A64, NTuple{2, UInt8}, NTuple{1, UInt32}, UInt16,
                       Int32, Int32, A32)),
        # A ::16b multicast cannot carry an override.
        ((:async, :bulk, :tensor, d2, cluster, :global, :tile, complete,
          _TMA94_MC16, _TMA94_OVADDR), (A32, A64, A64, Int32, Int32, A32, UInt16)),
        # Report mechanisms exist on loads only.
        ((:async, :bulk, :tensor, d2, :global, cta, :tile, :bulk_group,
          _TMA94_REPORTS[1]), (A64, Int32, Int32, A32)),
        # Report × override is not generated.
        ((:async, :bulk, :tensor, d1, cta, :global, :tile, complete,
          _TMA94_REPORTS[1], _TMA94_OVADDR), (A32, A64, A64, Int32, A32)),
        # Overrides are tile-only, and no-offset modes are 3d and up.
        ((:async, :bulk, :tensor, d3, :global, cta, Symbol("im2col_no_offs"),
          :bulk_group, _TMA94_OVADDR), (A64, A64, Int32, Int32, Int32, A32)),
        ((:async, :bulk, :tensor, d2, :global, cta, Symbol("im2col_no_offs::w"),
          :bulk_group), (A64, Int32, Int32, A32)),
        # Prefetch: the two L2 qualifiers are separate syntax lines.
        ((:async, :bulk, :prefetch, :tensor, d2, :L2, :global, :tile,
          Symbol("L2::cache_hint"), _TMA94_EVICT_LAST), (tmap, Int32, Int32, UInt64)),
        ((:async, :bulk, :prefetch, :tensor, d2, :L2, :global, :tile,
          _TMA94_EVICT_LAST, Symbol("L2::cache_hint")), (tmap, Int32, Int32, UInt64)),
        ((:async, :bulk, :prefetch, :tensor, d2, :L2, :global, :tile,
          _TMA94_OVADDR, Symbol("L2::cache_hint")), (tmap, pG, Int32, Int32, UInt64)),
    )
    for (mods, args) in not_wrapped
        @test !hasmethod(Operation{:cp, mods}(), args) ||
              which(Operation{:cp, mods}(), args).module !== PTX ||
              endswith(String(which(Operation{:cp, mods}(), args).file), "entries.jl")
        @test lowering(Operation{:cp, mods}(), args).tier === :forbidden
        @test_throws ArgumentError build_call(:cp, mods, args)
    end

    # Wrong mask carriers miss the exact methods (Integer widths convert,
    # but the LLVMPtr-free chain has no reviewed grammar for these).
    mods32 = (:async, :bulk, :tensor, d2, cluster, :global, :tile, complete,
              _TMA94_MC32)
    @test hasmethod(Operation{:cp, mods32}(), (pS, tmap, Int32, Int32, pM, UInt16))
    @test lowering(Operation{:cp, mods32}(), (A32, A64, Int32, Int32, A32, UInt16)).tier ===
          :forbidden

    # applypriority.async.bulk.tensor misses fail before rendering.
    apply = (:async, :bulk, :tensor, d2, :global, :bulk_group, :tile,
             _TMA94_EVICT_NORMAL)
    for args in ((tmap, Int32), (tmap, Int32, Int32, Int32), (pG, Int32, Int32))
        @test lowering(Operation{:applypriority, apply}(), args).tier === :forbidden
        @test_throws ArgumentError build_call(:applypriority, apply, args)
    end
    gather = (:async, :bulk, :tensor, d2, :global, :bulk_group,
              Symbol("tile::gather4"), _TMA94_EVICT_NORMAL)
    @test lowering(Operation{:applypriority, gather}(),
                   (tmap, Int32, Int32, Int32, Int32, Int32)).tier === :forbidden
    @test PTX.form_contract(:applypriority, apply).returns == false
end

@testset "PTX ISA 9.4 bulk chain spellings (family-gated)" begin
    pS = Core.LLVMPtr{UInt8, PTX.AS.Shared}
    pG = Core.LLVMPtr{UInt8, PTX.AS.Global}
    pM = Core.LLVMPtr{UInt64, PTX.AS.Shared}
    complete = _TMA94_COMPLETE

    for (mods, args, tail) in (
            ((:async, :bulk, _TMA94_CLUSTER, :global, complete, _TMA94_MC16),
             (pS, pG, UInt32, pM, UInt16), "[\$0], [\$1], \$2, [\$3], \$4;"),
            ((:async, :bulk, _TMA94_CLUSTER, :global, complete, _TMA94_MC32),
             (pS, pG, UInt32, pM, UInt32), "[\$0], [\$1], \$2, [\$3], \$4;"),
            ((:async, :bulk, _TMA94_CLUSTER, :global, complete,
              _TMA94_REPORTS[6], _TMA94_MC32),
             (pS, pG, UInt32, pM, UInt32), "[\$0], [\$1], \$2, [\$3], \$4;"),
            ((:async, :bulk, :relaxed, :gpu, _TMA94_CTA, :global, complete,
              _TMA94_REPORTS[2], :b128),
             (pS, pG, UInt32, pM), "[\$0], [\$1], \$2, [\$3];"),
            ((:async, :bulk, :prefetch, :L2, :global, _TMA94_EVICT_LAST),
             (pG, UInt32), "[\$0], \$1;"))
        spec = build_call(:cp, mods, args)
        @test spec.asm == PTX.build_head(:cp, mods) * " " * tail
        @test spec.rettype === Nothing
        @test spec.side_effects
        @test occursin("~{memory}", spec.constraints)
    end
    mask_letters = build_call(:cp, (:async, :bulk, _TMA94_CLUSTER, :global,
                                    complete, _TMA94_MC16),
                              (pS, pG, UInt32, pM, UInt16)).constraints
    @test endswith(mask_letters, ",h,~{memory}")
    mask_letters = build_call(:cp, (:async, :bulk, _TMA94_CLUSTER, :global,
                                    complete, _TMA94_MC32),
                              (pS, pG, UInt32, pM, UInt32)).constraints
    @test endswith(mask_letters, ",r,~{memory}")

    apply = (:async, :bulk, :global, :bulk_group, _TMA94_EVICT_NORMAL)
    spec = build_call(:applypriority, apply, (pG, UInt32))
    @test spec.asm == "applypriority.async.bulk.global.bulk_group.L2::evict_normal [\$0], \$1;"
    @test spec.rettype === Nothing
    spec = build_call(:applypriority, apply, (pG, Val{128}))
    @test spec.asm == "applypriority.async.bulk.global.bulk_group.L2::evict_normal [\$0], 128;"
    @test spec.side_effects && occursin("~{memory}", spec.constraints)
end
