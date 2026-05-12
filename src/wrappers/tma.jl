# `cp.async.bulk.tensor.<N>d.*` — TMA wrappers (Hopper sm_90+). Hand-written:
# the `[base, {c0, c1, ...}]` tensor-coord operand has no chain-default
# rendering, three pointer constraint shapes per call (shared `r`, tensormap
# `l`, mbarrier `r`), and the `<N>d` rank must pin both the asm string and
# coord arity at the registration boundary.

# `direction = :load` emits the G2S form `[dst], [tmap, {coords}], [mbar]`.
# `direction = :store` emits the S2G form `[tmap, {coords}], [src]`.
function tma_spec(ndim::Int, direction::Symbol, dst_ss::Symbol)
    1 <= ndim <= 5 || error("tma_spec: ndim must be 1..5, got $ndim")

    if direction === :load
        dst_ss in (Symbol("shared::cluster"), Symbol("shared::cta")) ||
            error("tma_spec: load dst_ss must be Symbol(\"shared::cluster\") or Symbol(\"shared::cta\"), got :$dst_ss")
        ss_text = dst_ss === Symbol("shared::cluster") ? "shared::cluster" : "shared::cta"
        head = "cp.async.bulk.tensor.$(ndim)d.$ss_text.global.tile.mbarrier::complete_tx::bytes"
        coord_slots = join(("\$$(i+1)" for i in 1:ndim), ", ")
        mbar_slot   = ndim + 2
        asm = "$head [\$0], [\$1, {$coord_slots}], [\$$mbar_slot];"
        constraints = "r,l," * join(fill("r", ndim), ",") * ",r,~{memory}"
        return (; asm, constraints)
    elseif direction === :load_multicast
        # Multicast load: cluster-target only (`.multicast::cluster` distributes
        # one global read across the CTAs whose bits are set in the u16 mask).
        # Same operand layout as `:load` but appends the mask as a scalar
        # operand after the mbar address (no brackets).
        head = "cp.async.bulk.tensor.$(ndim)d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
        coord_slots = join(("\$$(i+1)" for i in 1:ndim), ", ")
        mbar_slot   = ndim + 2
        mask_slot   = ndim + 3
        asm = "$head [\$0], [\$1, {$coord_slots}], [\$$mbar_slot], \$$mask_slot;"
        constraints = "r,l," * join(fill("r", ndim), ",") * ",r,h,~{memory}"
        return (; asm, constraints)
    elseif direction === :store
        head = "cp.async.bulk.tensor.$(ndim)d.global.shared::cta.tile.bulk_group"
        coord_slots = join(("\$$i" for i in 1:ndim), ", ")
        src_slot    = ndim + 1
        asm = "$head [\$0, {$coord_slots}], [\$$src_slot];"
        constraints = "l," * join(fill("r", ndim), ",") * ",r,~{memory}"
        return (; asm, constraints)
    else
        error("tma_spec: direction must be :load or :store, got :$direction")
    end
end

# Coord param accepts any `Integer`; the wrapper converts to `Int32` at the
# @asmcall boundary so callers don't have to spell out `Int32(0)` etc.
_tma_coord_param(c::Symbol) = Expr(:(::), c, :Integer)

function _tma_load_register(ndim::Int, dst_ss::Symbol)
    spec = tma_spec(ndim, :load, dst_ss)
    mods = (:async, :bulk, :tensor, Symbol("$(ndim)d"),
            dst_ss, :global, :tile, Symbol("mbarrier::complete_tx::bytes"))
    coords = [Symbol("c$i") for i in 1:ndim]
    coord_params = [_tma_coord_param(c) for c in coords]
    coord_types  = fill(:Int32, ndim)
    coord_casts  = [:(Int32($c)) for c in coords]

    fdef = quote
        function (::Operation{:cp, $mods})(
                dst::Core.LLVMPtr{T,AS.Shared},
                tmap::Core.LLVMPtr{S,AS.Const},
                $(coord_params...),
                mbar::Core.LLVMPtr{U,AS.Shared}) where {T,S,U}
            Base.@inline
            @asmcall($(spec.asm), $(spec.constraints), true, Nothing,
                     Tuple{Core.LLVMPtr{T,AS.Shared}, Core.LLVMPtr{S,AS.Const},
                           $(coord_types...),
                           Core.LLVMPtr{U,AS.Shared}},
                     dst, tmap, $(coord_casts...), mbar)
            nothing
        end
    end
    eval(fdef)
end

function _tma_load_multicast_register(ndim::Int)
    spec = tma_spec(ndim, :load_multicast, Symbol("shared::cluster"))
    mods = (:async, :bulk, :tensor, Symbol("$(ndim)d"),
            Symbol("shared::cluster"), :global, :tile,
            Symbol("mbarrier::complete_tx::bytes"),
            Symbol("multicast::cluster"))
    coords = [Symbol("c$i") for i in 1:ndim]
    coord_params = [_tma_coord_param(c) for c in coords]
    coord_types  = fill(:Int32, ndim)
    coord_casts  = [:(Int32($c)) for c in coords]

    fdef = quote
        function (::Operation{:cp, $mods})(
                dst::Core.LLVMPtr{T,AS.Shared},
                tmap::Core.LLVMPtr{S,AS.Const},
                $(coord_params...),
                mbar::Core.LLVMPtr{U,AS.Shared},
                mask::Integer) where {T,S,U}
            Base.@inline
            @asmcall($(spec.asm), $(spec.constraints), true, Nothing,
                     Tuple{Core.LLVMPtr{T,AS.Shared}, Core.LLVMPtr{S,AS.Const},
                           $(coord_types...),
                           Core.LLVMPtr{U,AS.Shared}, UInt16},
                     dst, tmap, $(coord_casts...), mbar, UInt16(mask))
            nothing
        end
    end
    eval(fdef)
end

function _tma_store_register(ndim::Int)
    spec = tma_spec(ndim, :store, :_)
    mods = (:async, :bulk, :tensor, Symbol("$(ndim)d"),
            :global, Symbol("shared::cta"), :tile, :bulk_group)
    coords = [Symbol("c$i") for i in 1:ndim]
    coord_params = [_tma_coord_param(c) for c in coords]
    coord_types  = fill(:Int32, ndim)
    coord_casts  = [:(Int32($c)) for c in coords]

    fdef = quote
        function (::Operation{:cp, $mods})(
                tmap::Core.LLVMPtr{S,AS.Const},
                $(coord_params...),
                src::Core.LLVMPtr{T,AS.Shared}) where {S,T}
            Base.@inline
            @asmcall($(spec.asm), $(spec.constraints), true, Nothing,
                     Tuple{Core.LLVMPtr{S,AS.Const},
                           $(coord_types...),
                           Core.LLVMPtr{T,AS.Shared}},
                     tmap, $(coord_casts...), src)
            nothing
        end
    end
    eval(fdef)
end

for ndim in 1:5, dst_ss in (Symbol("shared::cluster"), Symbol("shared::cta"))
    _tma_load_register(ndim, dst_ss)
end

for ndim in 1:5
    _tma_load_multicast_register(ndim)
end

for ndim in 1:5
    _tma_store_register(ndim)
end
