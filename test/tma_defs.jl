# Independent inventory of the PTX ISA 9.4 tensor-copy forms (§9.7.10.28,
# §9.7.10.19), re-derived from the ISA syntax blocks rather than from the
# wrapper generators, plus the sink kernels the offline legs assemble.
# Included by host/tma.jl and ptxas/tma.jl so both legs see one list.
#
# Each entry is `(op, mods, kinds, n)`; `kinds` names every Julia argument
# in call order with the vocabulary below, and `n` is the tensor rank.

const _TMA94_REPORTS = (
    Symbol("mbarrier::report::disabled"),
    Symbol("mbarrier::report::validity::per_16bytes::80000000"),
    Symbol("mbarrier::report::validity::per_16bytes::8000"),
    Symbol("mbarrier::report::validity::per_16bytes::80"),
    Symbol("mbarrier::report::validity::per_16bytes::8"),
    Symbol("mbarrier::report::validity::per_element::ff"),
)
const _TMA94_REDOPS = (:add, :min, :max, :inc, :dec, :and, :or, :xor)
const _TMA94_MC16 = Symbol("multicast::cluster::16b")
const _TMA94_MC32 = Symbol("multicast::cluster::32b")
const _TMA94_CLUSTER = Symbol("shared::cluster")
const _TMA94_CTA = Symbol("shared::cta")
const _TMA94_COMPLETE = Symbol("mbarrier::complete_tx::bytes")
const _TMA94_OVADDR = Symbol("override::global_address")
const _TMA94_EVICT_LAST = Symbol("L2::evict_last")
const _TMA94_EVICT_NORMAL = Symbol("L2::evict_normal")

_tma94_rank(n) = Symbol("$(n)d")
_tma94_attr(n) = n == 1 ? Symbol("override::global_dim") :
                          Symbol("override::global_dim_stride")
# Override arguments: the global address, then (attribute form only) the
# dimension bytes, the lower strides (rank ≥ 2), and the packed upper
# strides (rank ≥ 2).
_tma94_override_kinds(n, attr) =
    attr ? (n == 1 ? [:gaddr, :dims] : [:gaddr, :dims, :lower, :upper]) : [:gaddr]
_tma94_coords(n) = fill(:coord, n)

# Loop A: multicast widths on the cluster loads.
function _tma94_multicast_widths()
    out = []
    for n in 1:5, cg2 in (false, true), (mc, mask) in ((_TMA94_MC16, :mask16),
                                                       (_TMA94_MC32, :mask32))
        mods = (:async, :bulk, :tensor, _tma94_rank(n),
                (cg2 ? (Symbol("cta_group::2"),) : ())...,
                _TMA94_CLUSTER, :global, :tile, _TMA94_COMPLETE, mc)
        push!(out, (:cp, mods, [:dst, :tmap, _tma94_coords(n)..., :mbar, mask], n))
    end
    out
end

# Loop B: report mechanisms on the cluster loads.
function _tma94_cluster_reports()
    out = []
    for n in 1:5, mc32 in (false, true), report in _TMA94_REPORTS
        mods = (:async, :bulk, :tensor, _tma94_rank(n), _TMA94_CLUSTER, :global,
                :tile, _TMA94_COMPLETE, report, (mc32 ? (_TMA94_MC32,) : ())...)
        kinds = [:dst, :tmap, _tma94_coords(n)..., :mbar,
                 (mc32 ? (:mask32,) : ())...]
        push!(out, (:cp, mods, kinds, n))
    end
    out
end

# Loop C: overrides on the cluster loads.
function _tma94_cluster_overrides()
    out = []
    for n in 1:5, mc32 in (false, true), attr in (false, true)
        mods = (:async, :bulk, :tensor, _tma94_rank(n), _TMA94_CLUSTER, :global,
                :tile, _TMA94_COMPLETE, (mc32 ? (_TMA94_MC32,) : ())...,
                _TMA94_OVADDR, (attr ? (_tma94_attr(n),) : ())...)
        kinds = [:dst, :tmap, _tma94_override_kinds(n, attr)...,
                 _tma94_coords(n)..., :mbar, (mc32 ? (:mask32,) : ())...]
        push!(out, (:cp, mods, kinds, n))
    end
    out
end

# Loop D: report mechanisms on the shared::cta loads.
function _tma94_cta_reports()
    out = []
    for n in 1:5, report in _TMA94_REPORTS
        mods = (:async, :bulk, :tensor, _tma94_rank(n), _TMA94_CTA, :global,
                :tile, _TMA94_COMPLETE, report)
        push!(out, (:cp, mods, [:dst, :tmap, _tma94_coords(n)..., :mbar], n))
    end
    out
end

# Loop E: overrides on the shared::cta loads.
function _tma94_cta_overrides()
    out = []
    for n in 1:5, attr in (false, true)
        mods = (:async, :bulk, :tensor, _tma94_rank(n), _TMA94_CTA, :global,
                :tile, _TMA94_COMPLETE, _TMA94_OVADDR,
                (attr ? (_tma94_attr(n),) : ())...)
        kinds = [:dst, :tmap, _tma94_override_kinds(n, attr)...,
                 _tma94_coords(n)..., :mbar]
        push!(out, (:cp, mods, kinds, n))
    end
    out
end

# Loop F: no-offset im2col store modes (base sm_90, ::w sm_107f).
function _tma94_store_no_offs(; w)
    out = []
    mode = w ? Symbol("im2col_no_offs::w") : Symbol("im2col_no_offs")
    for n in 3:5
        mods = (:async, :bulk, :tensor, _tma94_rank(n), :global, _TMA94_CTA,
                mode, :bulk_group)
        push!(out, (:cp, mods, [:tmap, _tma94_coords(n)..., :src], n))
    end
    out
end

# Loop G: overrides on the tile store.
function _tma94_store_overrides()
    out = []
    for n in 1:5, attr in (false, true)
        mods = (:async, :bulk, :tensor, _tma94_rank(n), :global, _TMA94_CTA,
                :tile, :bulk_group, _TMA94_OVADDR,
                (attr ? (_tma94_attr(n),) : ())...)
        kinds = [:tmap, _tma94_override_kinds(n, attr)..., _tma94_coords(n)...,
                 :src]
        push!(out, (:cp, mods, kinds, n))
    end
    out
end

# Loop H: eviction priority on the tile and base-im2col prefetches.
function _tma94_prefetch_evict()
    out = []
    for n in 1:5
        mods = (:async, :bulk, :prefetch, :tensor, _tma94_rank(n), :L2, :global,
                :tile, _TMA94_EVICT_LAST)
        push!(out, (:cp, mods, [:tmap, _tma94_coords(n)...], n))
    end
    for n in 3:5
        mods = (:async, :bulk, :prefetch, :tensor, _tma94_rank(n), :L2, :global,
                :im2col, _TMA94_EVICT_LAST)
        push!(out, (:cp, mods, [:tmap, _tma94_coords(n)..., fill(:offset, n - 2)...], n))
    end
    out
end

# Loop I: overrides on the tile prefetch.
function _tma94_prefetch_overrides()
    out = []
    for n in 1:5, attr in (false, true)
        mods = (:async, :bulk, :prefetch, :tensor, _tma94_rank(n), :L2, :global,
                :tile, _TMA94_OVADDR, (attr ? (_tma94_attr(n),) : ())...)
        kinds = [:tmap, _tma94_override_kinds(n, attr)..., _tma94_coords(n)...]
        push!(out, (:cp, mods, kinds, n))
    end
    out
end

# Loop J: applypriority.async.bulk.tensor — tile, base im2col, tile overrides.
function _tma94_applypriority()
    out = []
    for n in 1:5
        mods = (:async, :bulk, :tensor, _tma94_rank(n), :global, :bulk_group,
                :tile, _TMA94_EVICT_NORMAL)
        push!(out, (:applypriority, mods, [:tmap, _tma94_coords(n)...], n))
    end
    for n in 3:5
        mods = (:async, :bulk, :tensor, _tma94_rank(n), :global, :bulk_group,
                :im2col, _TMA94_EVICT_NORMAL)
        push!(out, (:applypriority, mods,
                    [:tmap, _tma94_coords(n)..., fill(:offset, n - 2)...], n))
    end
    for n in 1:5, attr in (false, true)
        mods = (:async, :bulk, :tensor, _tma94_rank(n), :global, :bulk_group,
                :tile, _TMA94_EVICT_NORMAL, _TMA94_OVADDR,
                (attr ? (_tma94_attr(n),) : ())...)
        kinds = [:tmap, _tma94_override_kinds(n, attr)..., _tma94_coords(n)...]
        push!(out, (:applypriority, mods, kinds, n))
    end
    out
end

# Loop K: cp.reduce.async.bulk.tensor tile forms (sm_90).
function _tma94_reduce_tile()
    out = []
    for n in 1:5, redop in _TMA94_REDOPS
        mods = (:reduce, :async, :bulk, :tensor, _tma94_rank(n), :global,
                _TMA94_CTA, redop, :tile, :bulk_group)
        push!(out, (:cp, mods, [:tmap, _tma94_coords(n)..., :src], n))
    end
    out
end

# Loop L: no-offset im2col reduce modes (base sm_90, ::w sm_107f).
function _tma94_reduce_no_offs(; w)
    out = []
    mode = w ? Symbol("im2col_no_offs::w") : Symbol("im2col_no_offs")
    for n in 3:5, redop in _TMA94_REDOPS
        mods = (:reduce, :async, :bulk, :tensor, _tma94_rank(n), :global,
                _TMA94_CTA, redop, mode, :bulk_group)
        push!(out, (:cp, mods, [:tmap, _tma94_coords(n)..., :src], n))
    end
    out
end

# Loop M: overrides on the tile reduce.
function _tma94_reduce_overrides()
    out = []
    for n in 1:5, redop in _TMA94_REDOPS, attr in (false, true)
        mods = (:reduce, :async, :bulk, :tensor, _tma94_rank(n), :global,
                _TMA94_CTA, redop, :tile, :bulk_group, _TMA94_OVADDR,
                (attr ? (_tma94_attr(n),) : ())...)
        kinds = [:tmap, _tma94_override_kinds(n, attr)..., _tma94_coords(n)...,
                 :src]
        push!(out, (:cp, mods, kinds, n))
    end
    out
end

const _TMA94_LOOPS = (
    A = _tma94_multicast_widths(),
    B = _tma94_cluster_reports(),
    C = _tma94_cluster_overrides(),
    D = _tma94_cta_reports(),
    E = _tma94_cta_overrides(),
    F = vcat(_tma94_store_no_offs(; w = false), _tma94_store_no_offs(; w = true)),
    G = _tma94_store_overrides(),
    H = _tma94_prefetch_evict(),
    I = _tma94_prefetch_overrides(),
    J = _tma94_applypriority(),
    K = _tma94_reduce_tile(),
    L = vcat(_tma94_reduce_no_offs(; w = false), _tma94_reduce_no_offs(; w = true)),
    M = _tma94_reduce_overrides(),
)

# Argument carriers per kind (the host oracle's dispatch types) and the
# literal operands the sink kernels pass.
function _tma94_kind_type(kind, n)
    kind === :dst && return Core.LLVMPtr{UInt16, PTX.AS.Shared}
    kind === :src && return Core.LLVMPtr{UInt16, PTX.AS.Shared}
    kind === :tmap && return PTX.TMADescriptorPtr
    kind === :mbar && return Core.LLVMPtr{UInt64, PTX.AS.Shared}
    kind === :gaddr && return Core.LLVMPtr{UInt8, PTX.AS.Global}
    kind === :dims && return NTuple{n, UInt8}
    kind === :lower && return NTuple{n - 1, UInt32}
    kind === :upper && return UInt16
    kind === :coord && return Int32
    kind === :mask16 && return UInt16
    kind === :mask32 && return UInt32
    kind === :offset && return Int16
    error("unknown operand kind $kind")
end
_tma94_argtypes(entry) = Tuple(_tma94_kind_type(k, entry[4]) for k in entry[3])

function _tma94_kind_value(kind, n)
    kind === :dst && return :dst
    kind === :src && return :dst
    kind === :tmap && return :tmap
    kind === :mbar && return :mbar
    kind === :gaddr && return :gaddr
    kind === :dims && return ntuple(_ -> 0x01, n)
    kind === :lower && return ntuple(_ -> UInt32(1), n - 1)
    kind === :upper && return UInt16(0)
    kind === :coord && return Int32(0)
    kind === :mask16 && return UInt16(0x3)
    kind === :mask32 && return UInt32(0x3)
    kind === :offset && return Int16(0)
    error("unknown operand kind $kind")
end

# Kernel groups by admitting target.
_tma94_is(entry, sym) = sym in entry[2]
_tma94_cg2(entry) = _tma94_is(entry, Symbol("cta_group::2"))
_tma94_w(entry) = _tma94_is(entry, Symbol("im2col_no_offs::w"))
const _TMA94_GROUPS = Dict(
    :cluster => vcat(filter(e -> _tma94_is(e, _TMA94_MC32) && !_tma94_cg2(e),
                            _TMA94_LOOPS.A),
                     _TMA94_LOOPS.B, _TMA94_LOOPS.C),
    :cluster_cg2 => filter(e -> _tma94_is(e, _TMA94_MC32) && _tma94_cg2(e),
                           _TMA94_LOOPS.A),
    :cta => vcat(_TMA94_LOOPS.D, _TMA94_LOOPS.E),
    :store => vcat(filter(_tma94_w, _TMA94_LOOPS.F), _TMA94_LOOPS.G),
    :reduce => vcat(filter(_tma94_w, _TMA94_LOOPS.L), _TMA94_LOOPS.M),
    :prefetch => vcat(_TMA94_LOOPS.H, _TMA94_LOOPS.I, _TMA94_LOOPS.J),
    :sm90 => vcat(filter(e -> _tma94_is(e, _TMA94_MC16) && !_tma94_cg2(e),
                         _TMA94_LOOPS.A),
                  filter(!_tma94_w, _TMA94_LOOPS.F), _TMA94_LOOPS.K,
                  filter(!_tma94_w, _TMA94_LOOPS.L)),
    :sm100a => filter(e -> _tma94_is(e, _TMA94_MC16) && _tma94_cg2(e),
                      _TMA94_LOOPS.A),
)

@generated function _tma94_sink!(::Val{group},
                                 dst::Core.LLVMPtr{UInt16, PTX.AS.Shared},
                                 tmap::PTX.TMADescriptorPtr,
                                 mbar::Core.LLVMPtr{UInt64, PTX.AS.Shared},
                                 gaddr::Core.LLVMPtr{UInt8, PTX.AS.Global}) where {group}
    body = Expr(:block)
    for (op, mods, kinds, n) in _TMA94_GROUPS[group]
        args = [_tma94_kind_value(k, n) for k in kinds]
        push!(body.args, :(PTX.Operation{$(QuoteNode(op)), $mods}()($(args...))))
    end
    push!(body.args, :(return nothing))
    body
end

_tma94_sink_types(group) = Tuple{Val{group}, Core.LLVMPtr{UInt16, PTX.AS.Shared},
                                 PTX.TMADescriptorPtr,
                                 Core.LLVMPtr{UInt64, PTX.AS.Shared},
                                 Core.LLVMPtr{UInt8, PTX.AS.Global}}

# Chain-rendered spellings: the non-tensor bulk copy widths and report, the
# bulk prefetch eviction priority, and the bulk eviction hint.
function _tma94_chain!(dst::Core.LLVMPtr{UInt8, PTX.AS.Shared},
                       g::Core.LLVMPtr{UInt8, PTX.AS.Global},
                       mbar::Core.LLVMPtr{UInt64, PTX.AS.Shared})
    ptx"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b"(
        dst, g, UInt32(16), mbar, UInt32(0x3))
    ptx"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.mbarrier::report::validity::per_element::ff.multicast::cluster::32b"(
        dst, g, UInt32(16), mbar, UInt32(0x3))
    ptx"cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.mbarrier::report::validity::per_16bytes::80000000"(
        dst, g, UInt32(16), mbar)
    ptx"cp.async.bulk.prefetch.L2.global.L2::evict_last"(g, UInt32(16))
    ptx"applypriority.async.bulk.global.bulk_group.L2::evict_normal"(g, Val(128))
    return nothing
end

const _TMA94_CHAIN_TYPES = Tuple{Core.LLVMPtr{UInt8, PTX.AS.Shared},
                                 Core.LLVMPtr{UInt8, PTX.AS.Global},
                                 Core.LLVMPtr{UInt64, PTX.AS.Shared}}
