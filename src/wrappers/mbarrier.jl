# mbarrier (PTX 9.3 §9.7.14.16) — single-route family: every form lowers
# through the exact schema in mbarrier_forms.jl to convergent inline asm.
# The family previously split ten pre-9.3 forms onto NVVM intrinsics; that
# split was retired deliberately:
#   - every mbarrier operation is an observable sync effect — there is no
#     CSE/LICM for intrinsic attributes to unlock, so the second route
#     bought bookkeeping (ISel cap-floor selection between legacy
#     `*.shared` and scoped `*.scope.cta.space.cta` names, typed-pointee
#     compat entries, per-intrinsic selection probes) and no optimization;
#   - the PTX 9.3 layout/phase_type/report forms and the cluster-space
#     forms were already asm (no intrinsics / `ptr addrspace(7)` ABI
#     mismatch at 22.1.7), and the PTX 9.4 multicast::cluster::32b forms
#     have no upstream intrinsic either — one route instead of three.
# Emitted-PTX delta vs the intrinsic route, reviewed at demotion: the
# standalone expect_tx spells legacy `mbarrier.expect_tx.shared.b64` (ISel
# printed the scoped `.relaxed.cta` spelling — semantically identical per
# ISA), and static-SMEM symbols materialize through mov/cvt instead of
# folding into the operand (ptxas folds these in SASS).
#
# Cluster-mapped addresses (from mapa.shared::cluster) are modeled as AS 3
# throughout the package; see the sink-destination forms below.

# Exact methods retain their integer-normalizing convenience signatures
# (the ledger deliberately rejects Int64 at :u32 operand positions), but
# delegate construction to the same closed schema as the generic and raw
# paths. This routes every form through `convergent_asm_ir`, so the whole
# family carries one call-site `convergent nomerge` + `~{memory}` contract.
@generated function _mbarrier_schema_call(
        ::Operation{:mbarrier, mods}, args::Vararg{Any,N}) where {mods,N}
    _chain_call_expr(build_call(:mbarrier, mods, args))
end

@inline optype"mbarrier.init.shared.b64"(
        mbar::Core.LLVMPtr{T, AS.Shared}, count::Integer) where T =
    _mbarrier_schema_call(ptx"mbarrier.init.shared.b64", mbar, UInt32(count))

@inline optype"mbarrier.inval.shared.b64"(
        mbar::Core.LLVMPtr{T, AS.Shared}) where T =
    _mbarrier_schema_call(ptx"mbarrier.inval.shared.b64", mbar)

@inline optype"mbarrier.arrive.shared.b64"(
        mbar::Core.LLVMPtr{T, AS.Shared}) where T =
    _mbarrier_schema_call(ptx"mbarrier.arrive.shared.b64", mbar)

# `count` is a per-thread arrive count (same as calling arrive `count`
# times). The caller must choose it so this noComplete operation cannot finish
# the phase; reaching completion is undefined behavior.
@inline optype"mbarrier.arrive.noComplete.shared.b64"(
        mbar::Core.LLVMPtr{T, AS.Shared}, count::Integer) where T =
    _mbarrier_schema_call(ptx"mbarrier.arrive.noComplete.shared.b64",
                          mbar, UInt32(count))

# (sm_90+) Fused expect+arrive: first increments tx-count by `tx_count`, then
# performs one arrive-on operation, decrementing pending arrivals by 1.
@inline optype"mbarrier.arrive.expect_tx.shared.b64"(
        mbar::Core.LLVMPtr{T, AS.Shared}, tx_count::Integer) where T =
    _mbarrier_schema_call(ptx"mbarrier.arrive.expect_tx.shared.b64",
                          mbar, UInt32(tx_count))

# (sm_90+) Standalone form: no arrive, no state output.
@inline optype"mbarrier.expect_tx.shared.b64"(
        mbar::Core.LLVMPtr{T, AS.Shared}, tx_count::Integer) where T =
    _mbarrier_schema_call(ptx"mbarrier.expect_tx.shared.b64",
                          mbar, UInt32(tx_count))

# Token form: pass the UInt64 returned by a prior arrive on the same mbar.
@inline optype"mbarrier.test_wait.shared.b64"(
        mbar::Core.LLVMPtr{T, AS.Shared}, state::Integer) where T =
    _mbarrier_schema_call(ptx"mbarrier.test_wait.shared.b64",
                          mbar, UInt64(state))

# Phase form: pass a 0/1 phase parity bit; returns true once a full arrive
# cycle has completed.
@inline optype"mbarrier.test_wait.parity.shared.b64"(
        mbar::Core.LLVMPtr{T, AS.Shared}, phase::Integer) where T =
    _mbarrier_schema_call(ptx"mbarrier.test_wait.parity.shared.b64",
                          mbar, UInt32(phase))

# (sm_90+) Suspend-allowing wait — hardware may park the warp on miss
# instead of returning false immediately.
@inline optype"mbarrier.try_wait.shared.b64"(
        mbar::Core.LLVMPtr{T, AS.Shared}, state::Integer) where T =
    _mbarrier_schema_call(ptx"mbarrier.try_wait.shared.b64",
                          mbar, UInt64(state))

@inline optype"mbarrier.try_wait.parity.shared.b64"(
        mbar::Core.LLVMPtr{T, AS.Shared}, phase::Integer) where T =
    _mbarrier_schema_call(ptx"mbarrier.try_wait.parity.shared.b64",
                          mbar, UInt32(phase))

# --- Cluster-scope variants (sm_90+), asm tier ------------------------------
# `mbarrier.arrive.shared::cluster.b64` requires a `_` (sink) destination
# operand — the cluster-scope arrive doesn't return a state token (cross-CTA
# state would be meaningless). Caller passes a cluster-mapped address from
# `mapa.shared::cluster` when arriving on a remote CTA's mbarrier.

@inline function optype"mbarrier.arrive.shared::cluster.b64"(
        mbar::Core.LLVMPtr{T, AS.Shared}) where T
    _mbarrier_schema_call(ptx"mbarrier.arrive.shared::cluster.b64", mbar)
end

@inline function optype"mbarrier.arrive.expect_tx.shared::cluster.b64"(
        mbar::Core.LLVMPtr{T, AS.Shared},
        tx_count::Integer) where T
    _mbarrier_schema_call(ptx"mbarrier.arrive.expect_tx.shared::cluster.b64",
                          mbar, UInt32(tx_count))
end

# --- PTX 9.4 cluster multicast (sm_107f), spelled-only until 13.4 ptxas ----
# `.multicast::cluster::32b` runs the operation on the mbarrier at the same
# CTA-relative offset in every cluster CTA selected by `cta_mask` (bit i =
# %cluster_ctarank i). The mask is the mandatory trailing operand.

@inline function optype"mbarrier.arrive.shared::cluster.multicast::cluster::32b.b64"(
        mbar::Core.LLVMPtr{T, AS.Shared}, cta_mask::Integer) where T
    _mbarrier_schema_call(
        ptx"mbarrier.arrive.shared::cluster.multicast::cluster::32b.b64",
        mbar, UInt32(cta_mask))
end

@inline function optype"mbarrier.arrive.expect_tx.shared::cluster.multicast::cluster::32b.b64"(
        mbar::Core.LLVMPtr{T, AS.Shared},
        tx_count::Integer, cta_mask::Integer) where T
    _mbarrier_schema_call(
        ptx"mbarrier.arrive.expect_tx.shared::cluster.multicast::cluster::32b.b64",
        mbar, UInt32(tx_count), UInt32(cta_mask))
end

@inline function optype"mbarrier.expect_tx.shared::cluster.multicast::cluster::32b.b64"(
        mbar::Core.LLVMPtr{T, AS.Shared},
        tx_count::Integer, cta_mask::Integer) where T
    _mbarrier_schema_call(
        ptx"mbarrier.expect_tx.shared::cluster.multicast::cluster::32b.b64",
        mbar, UInt32(tx_count), UInt32(cta_mask))
end

# --- PTX 9.3 extensions (layout / phase_type / report), asm tier ------------
# No NVVM intrinsics exist for any of these at 22.1.7 — asm tier by
# necessity, same as the cluster-space forms above.

# `mbarrier.init.layout::{v0,v1}.shared.b64 [mbar], count;` — explicit layout
# selector. layout::v0 is the historical default; layout::v1 enables fabric
# report tracking (see wrappers/fabric.jl). Same operand shape as plain
# `mbarrier.init`; only the asm head differs.

for lay in (:v0, :v1)
    mods = (:init, Symbol("layout::$lay"), :shared, :b64)
    @eval @inline function (op::Operation{:mbarrier, $mods})(
            mbar::Core.LLVMPtr{T, AS.Shared},
            count::Integer) where T
        _mbarrier_schema_call(op, mbar, UInt32(count))
    end
end

# `mbarrier.check_layout.layout::{v0,v1}.shared::cta.b64 p, [mbar];` — sets p=True
# iff the mbarrier's actual layout matches the qualifier. Lets a callee
# defensively verify a barrier passed in by a caller. sm_90+.

for lay in (:v0, :v1)
    mods = (:check_layout, Symbol("layout::$lay"), Symbol("shared::cta"), :b64)
    @eval @inline function (op::Operation{:mbarrier, $mods})(
            mbar::Core.LLVMPtr{T, AS.Shared}) where T
        _mbarrier_schema_call(op, mbar)
    end
end

# `.phase_type::primary` report forms (three outputs):
# `mbarrier.{test,try}_wait[.parity].phase_type::primary.shared.b64
#     waitComplete|reportPredicate, reportValue, [mbar], state-or-phase;`
# PTX's opaque reportValue is a `.b8` register. NVPTX has no i8 inline-asm
# constraint, so the wrapper returns its byte in the low half of a `UInt16`:
# `Tuple{Bool, Bool, UInt16}`. Report fields are undefined until waitComplete
# is true. Once complete, a set reportPredicate indicates an asynchronous
# operation reported an error or other condition; if it is clear, reportValue
# is guaranteed to be zero and the conditional phase advanced. Layout::v0
# always reports a zero predicate and value.
#
# `:report` is an audited synthetic result selector (no PTX counterpart) for
# the predicate pair plus reportValue; `:report_pred` in mbarrier_forms.jl
# selects the predicate pair without the optional value. PTX uses the same
# instruction head for all of these destination shapes.

for wait in (:test_wait, :try_wait)
    # token form: UInt64 state from a prior arrive
    mods = (wait, :report, Symbol("phase_type::primary"), :shared, :b64)
    @eval @inline function (op::Operation{:mbarrier, $mods})(
            mbar::Core.LLVMPtr{T, AS.Shared},
            state::Integer) where T
        _mbarrier_schema_call(op, mbar, UInt64(state))
    end

    # parity form: 0/1 phase bit
    mods = (wait, :report, :parity,
            Symbol("phase_type::primary"), :shared, :b64)
    @eval @inline function (op::Operation{:mbarrier, $mods})(
            mbar::Core.LLVMPtr{T, AS.Shared},
            phase::Integer) where T
        _mbarrier_schema_call(op, mbar, UInt32(phase))
    end
end

# `.phase_type::conditional` parity waits (single-output):
# `mbarrier.{test,try}_wait.parity.phase_type::conditional.shared.b64
#     waitComplete, [mbar], phaseParity;`
# Legal for both layouts. It is most useful with layout::v1, where conditional
# phase advances only when the payload report is zero (no fabric errors); for
# layout::v0 the conditional and primary phases advance in unison. `.parity`
# is mandatory per the PTX 9.3 syntax block.

for wait in (:test_wait, :try_wait)
    mods = (wait, :parity,
            Symbol("phase_type::conditional"), :shared, :b64)
    @eval @inline function (op::Operation{:mbarrier, $mods})(
            mbar::Core.LLVMPtr{T, AS.Shared},
            phase::Integer) where T
        _mbarrier_schema_call(op, mbar, UInt32(phase))
    end
end
