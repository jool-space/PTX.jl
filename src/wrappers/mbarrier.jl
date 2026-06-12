# mbarrier (PTX 9.2 §9.7.12.15) — second family migrated to tier-2
# intrinsic lowering (DESIGN.md). The notation surface is unchanged; the
# three return shapes (Nothing / UInt64 state / Bool pred) that needed
# hand-written asm constraints now fall out of the intrinsic signatures.
#
# Intrinsic choice per form is dictated by the cap floor: the new-style
# scoped intrinsics (`*.scope.cta.space.cta`) carry ISel predicates, and a
# form with an sm_90 operand cannot select below it — the count-form
# `arrive.scope.cta.space.cta` fails at sm_80 (verified against llc
# 22.1.7). So:
#   - sm_80 forms (init, inval, arrive, arrive.noComplete, test_wait) use
#     the legacy `*.shared` intrinsics, preserving the sm_80 floor;
#   - the parity waits use the scoped intrinsics (no legacy form exists at
#     22.1.7) — these select the unqualified sm_80 spelling, floor intact;
#   - sm_90-only forms (expect_tx, arrive.expect_tx, try_wait*) use the
#     scoped intrinsics.
#
# Cluster-space (`shared::cluster`) sink-destination forms stay on the asm
# tier below: their intrinsics take `ptr addrspace(7)` and the package
# currently models cluster-mapped addresses (from mapa.shared::cluster) as
# AS 3 — they migrate together with proper AS-7 modeling.

@inline (::Operation{:mbarrier, (:init, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared}, count::Integer) where T =
    nvvm"mbarrier.init.shared"(mbar, UInt32(count))

@inline (::Operation{:mbarrier, (:inval, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"mbarrier.inval.shared"(mbar)

@inline (::Operation{:mbarrier, (:arrive, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"mbarrier.arrive.shared"(mbar)

# `count` is a per-thread arrive count (same as calling arrive `count`
# times). Does NOT close the phase even if pending hits zero.
@inline (::Operation{:mbarrier, (:arrive, :noComplete, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared}, count::Integer) where T =
    nvvm"mbarrier.arrive.noComplete.shared"(mbar, UInt32(count))

# (sm_90+) Fused expect+arrive: records expected-tx-bytes and bumps pending
# by 1.
@inline (::Operation{:mbarrier, (:arrive, :expect_tx, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared}, tx_count::Integer) where T =
    nvvm"mbarrier.arrive.expect.tx.scope.cta.space.cta"(mbar, UInt32(tx_count))

# (sm_90+) Standalone form: no arrive, no state output.
@inline (::Operation{:mbarrier, (:expect_tx, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared}, tx_count::Integer) where T =
    nvvm"mbarrier.expect.tx.scope.cta.space.cta"(mbar, UInt32(tx_count))

# Token form: pass the UInt64 returned by a prior arrive on the same mbar.
@inline (::Operation{:mbarrier, (:test_wait, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared}, state::Integer) where T =
    nvvm"mbarrier.test.wait.shared"(mbar, UInt64(state))

# Phase form: pass a 0/1 phase parity bit; returns true once a full arrive
# cycle has completed.
@inline (::Operation{:mbarrier, (:test_wait, :parity, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared}, phase::Integer) where T =
    nvvm"mbarrier.test.wait.parity.scope.cta.space.cta"(mbar, UInt32(phase))

# (sm_90+) Suspend-allowing wait — hardware may park the warp on miss
# instead of returning false immediately.
@inline (::Operation{:mbarrier, (:try_wait, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared}, state::Integer) where T =
    nvvm"mbarrier.try.wait.scope.cta.space.cta"(mbar, UInt64(state))

@inline (::Operation{:mbarrier, (:try_wait, :parity, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared}, phase::Integer) where T =
    nvvm"mbarrier.try.wait.parity.scope.cta.space.cta"(mbar, UInt32(phase))

# --- Cluster-scope variants (sm_90+), asm tier ------------------------------
# `mbarrier.arrive.shared::cluster.b64` requires a `_` (sink) destination
# operand — the cluster-scope arrive doesn't return a state token (cross-CTA
# state would be meaningless). Caller passes a cluster-mapped address from
# `mapa.shared::cluster` when arriving on a remote CTA's mbarrier.

@generated function (::Operation{:mbarrier, (:arrive, Symbol("shared::cluster"), :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared}) where T
    quote
        Base.@inline
        @asmcall("mbarrier.arrive.shared::cluster.b64 _, [\$0];",
                 "r,~{memory}", true, Nothing,
                 Tuple{Core.LLVMPtr{$T, AS.Shared}},
                 mbar)
        nothing
    end
end

@generated function (::Operation{:mbarrier, (:arrive, :expect_tx, Symbol("shared::cluster"), :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        tx_count::Integer) where T
    quote
        Base.@inline
        @asmcall("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [\$0], \$1;",
                 "r,r,~{memory}", true, Nothing,
                 Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt32},
                 mbar, UInt32(tx_count))
        nothing
    end
end
