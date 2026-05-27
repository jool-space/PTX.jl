# Hand-written: three distinct return shapes (Nothing / UInt64 state / Bool
# pred) the chain can't infer from the trailing `.b64` modifier, plus a
# shared-AS pointer that needs the `r` (i32) constraint instead of the chain
# default `l`. PTX 9.2 §9.7.12.15.

# Operand-shape categories.
#   :addr_only                  — `[$0]`
#   :addr_count                 — `[$0], $1`
#   :state_addr                 — `$0, [$1]`
#   :state_addr_n               — `$0, [$1], $2`
#   :pred_addr                  — `$0, [$1]`            (pred output, single addr in)
#   :pred_addr_state            — `$0, [$1], $2`        (state input)
#   :pred_addr_phase            — `$0, [$1], $2`        (phase input)
#   :dual_pred_value_addr_state — `$0, $1, [$2], $3`    (Bool + UInt64 outputs, state in)
#   :dual_pred_value_addr_phase — `$0, $1, [$2], $3`    (Bool + UInt64 outputs, phase in)
#   :sink_addr                  — `_, [$0]`             (cluster-scope arrive: no state token)
#   :sink_addr_n                — `_, [$0], $1`         (cluster-scope arrive with count/tx)
#
# `mbar_layout` (PTX 9.3) splices `.layout::{v0,v1}` after `op` for instructions
# that take a `.layout` qualifier (`mbarrier.init`, `mbarrier.check_layout`).
# `phase_type` (PTX 9.3) splices `.phase_type::{primary,conditional}` after `op`
# for `mbarrier.{test,try}_wait` variants that observe layout::v1 report state.
function mbarrier_spec(op::Symbol, layout::Symbol;
                       ss::Symbol = :shared,
                       mbar_layout::Union{Nothing, Symbol} = nothing,
                       phase_type::Union{Nothing, Symbol} = nothing)
    quals = String[]
    mbar_layout === nothing || push!(quals, "layout::$(mbar_layout)")
    phase_type === nothing || push!(quals, "phase_type::$(phase_type)")
    qual_str = isempty(quals) ? "" : "." * join(quals, ".")
    head = "mbarrier.$op$qual_str.$ss.b64"
    if layout === :addr_only
        return (; asm = "$head [\$0];",
                  constraints = "r,~{memory}",
                  rettype = Nothing)
    elseif layout === :addr_count
        return (; asm = "$head [\$0], \$1;",
                  constraints = "r,r,~{memory}",
                  rettype = Nothing)
    elseif layout === :state_addr
        return (; asm = "$head \$0, [\$1];",
                  constraints = "=l,r,~{memory}",
                  rettype = UInt64)
    elseif layout === :state_addr_n
        return (; asm = "$head \$0, [\$1], \$2;",
                  constraints = "=l,r,r,~{memory}",
                  rettype = UInt64)
    elseif layout === :pred_addr
        return (; asm = "$head \$0, [\$1];",
                  constraints = "=b,r,~{memory}",
                  rettype = Bool)
    elseif layout === :pred_addr_state
        return (; asm = "$head \$0, [\$1], \$2;",
                  constraints = "=b,r,l,~{memory}",
                  rettype = Bool)
    elseif layout === :pred_addr_phase
        return (; asm = "$head \$0, [\$1], \$2;",
                  constraints = "=b,r,r,~{memory}",
                  rettype = Bool)
    elseif layout === :dual_pred_value_addr_state
        return (; asm = "$head \$0, \$1, [\$2], \$3;",
                  constraints = "=b,=l,r,l,~{memory}",
                  rettype = Tuple{Bool, UInt64})
    elseif layout === :dual_pred_value_addr_phase
        return (; asm = "$head \$0, \$1, [\$2], \$3;",
                  constraints = "=b,=l,r,r,~{memory}",
                  rettype = Tuple{Bool, UInt64})
    elseif layout === :sink_addr
        return (; asm = "$head _, [\$0];",
                  constraints = "r,~{memory}",
                  rettype = Nothing)
    elseif layout === :sink_addr_n
        return (; asm = "$head _, [\$0], \$1;",
                  constraints = "r,r,~{memory}",
                  rettype = Nothing)
    else
        error("mbarrier_spec: unknown layout $layout")
    end
end

@generated function (::Operation{:mbarrier, (:init, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        count::Integer) where T
    spec = mbarrier_spec(:init, :addr_count)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true, Nothing,
                 Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt32},
                 mbar, UInt32(count))
        nothing
    end
end

@generated function (::Operation{:mbarrier, (:inval, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared}) where T
    spec = mbarrier_spec(:inval, :addr_only)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true, Nothing,
                 Tuple{Core.LLVMPtr{$T, AS.Shared}},
                 mbar)
        nothing
    end
end

@generated function (::Operation{:mbarrier, (:arrive, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared}) where T
    spec = mbarrier_spec(:arrive, :state_addr)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true,
                 UInt64, Tuple{Core.LLVMPtr{$T, AS.Shared}},
                 mbar)
    end
end

# `count` is a per-thread arrive count (same as calling arrive `count` times).
# Does NOT close the phase even if pending hits zero.
@generated function (::Operation{:mbarrier, (:arrive, :noComplete, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        count::Integer) where T
    spec = mbarrier_spec(Symbol("arrive.noComplete"), :state_addr_n)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true,
                 UInt64, Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt32},
                 mbar, UInt32(count))
    end
end

# (sm_90+) Fused expect+arrive: records expected-tx-bytes and bumps pending by 1.
@generated function (::Operation{:mbarrier, (:arrive, :expect_tx, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        tx_count::Integer) where T
    spec = mbarrier_spec(Symbol("arrive.expect_tx"), :state_addr_n)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true,
                 UInt64, Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt32},
                 mbar, UInt32(tx_count))
    end
end

# (sm_90+) Standalone form: no arrive, no state output.
@generated function (::Operation{:mbarrier, (:expect_tx, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        tx_count::Integer) where T
    spec = mbarrier_spec(:expect_tx, :addr_count)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true, Nothing,
                 Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt32},
                 mbar, UInt32(tx_count))
        nothing
    end
end

# Token form: pass the UInt64 returned by a prior arrive on the same mbar.
@generated function (::Operation{:mbarrier, (:test_wait, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        state::Integer) where T
    spec = mbarrier_spec(:test_wait, :pred_addr_state)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true,
                 Bool, Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt64},
                 mbar, UInt64(state))
    end
end

# Phase form: pass a 0/1 phase parity bit; returns true once a full arrive
# cycle has completed.
@generated function (::Operation{:mbarrier, (:test_wait, :parity, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        phase::Integer) where T
    spec = mbarrier_spec(Symbol("test_wait.parity"), :pred_addr_phase)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true,
                 Bool, Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt32},
                 mbar, UInt32(phase))
    end
end

# (sm_90+) Suspend-allowing wait — hardware may park the warp on miss instead
# of returning false immediately.
@generated function (::Operation{:mbarrier, (:try_wait, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        state::Integer) where T
    spec = mbarrier_spec(:try_wait, :pred_addr_state)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true,
                 Bool, Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt64},
                 mbar, UInt64(state))
    end
end

@generated function (::Operation{:mbarrier, (:try_wait, :parity, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        phase::Integer) where T
    spec = mbarrier_spec(Symbol("try_wait.parity"), :pred_addr_phase)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true,
                 Bool, Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt32},
                 mbar, UInt32(phase))
    end
end

# --- Cluster-scope variants (sm_90+) ---------------------------------------
# `mbarrier.arrive.shared::cluster.b64` requires a `_` (sink) destination
# operand — the cluster-scope arrive doesn't return a state token (cross-CTA
# state would be meaningless). Caller passes a cluster-mapped address from
# `mapa.shared::cluster` when arriving on a remote CTA's mbarrier.

@generated function (::Operation{:mbarrier, (:arrive, Symbol("shared::cluster"), :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared}) where T
    spec = mbarrier_spec(:arrive, :sink_addr; ss = Symbol("shared::cluster"))
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true, Nothing,
                 Tuple{Core.LLVMPtr{$T, AS.Shared}},
                 mbar)
        nothing
    end
end

@generated function (::Operation{:mbarrier, (:arrive, :expect_tx, Symbol("shared::cluster"), :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        tx_count::Integer) where T
    spec = mbarrier_spec(Symbol("arrive.expect_tx"), :sink_addr_n;
                         ss = Symbol("shared::cluster"))
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true, Nothing,
                 Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt32},
                 mbar, UInt32(tx_count))
        nothing
    end
end

# --- PTX 9.3: `.layout::vN` on init -----------------------------------------
# `mbarrier.init.layout::{v0,v1}.shared.b64 [mbar], count;` — explicit layout
# selector. layout::v0 is the historical default; layout::v1 enables fabric
# report tracking (see fabric.try_* family). Same operand shape as plain
# `mbarrier.init`; only the asm head differs.

@generated function (::Operation{:mbarrier, (:init, Symbol("layout::v0"), :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        count::Integer) where T
    spec = mbarrier_spec(:init, :addr_count; mbar_layout = :v0)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true, Nothing,
                 Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt32},
                 mbar, UInt32(count))
        nothing
    end
end

@generated function (::Operation{:mbarrier, (:init, Symbol("layout::v1"), :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        count::Integer) where T
    spec = mbarrier_spec(:init, :addr_count; mbar_layout = :v1)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true, Nothing,
                 Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt32},
                 mbar, UInt32(count))
        nothing
    end
end

# --- PTX 9.3: `mbarrier.check_layout` ---------------------------------------
# `mbarrier.check_layout.layout::{v0,v1}.shared.b64 p, [mbar];` — sets p=True
# iff the mbarrier's actual layout matches the qualifier. Lets a callee
# defensively verify a barrier passed in by a caller. sm_90+.

@generated function (::Operation{:mbarrier, (:check_layout, Symbol("layout::v0"), :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared}) where T
    spec = mbarrier_spec(:check_layout, :pred_addr; mbar_layout = :v0)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true,
                 Bool, Tuple{Core.LLVMPtr{$T, AS.Shared}},
                 mbar)
    end
end

@generated function (::Operation{:mbarrier, (:check_layout, Symbol("layout::v1"), :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared}) where T
    spec = mbarrier_spec(:check_layout, :pred_addr; mbar_layout = :v1)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true,
                 Bool, Tuple{Core.LLVMPtr{$T, AS.Shared}},
                 mbar)
    end
end

# --- PTX 9.3: `.phase_type::primary` report (dual-output) -------------------
# `mbarrier.{test,try}_wait.[parity.]phase_type::primary.shared.b64
#     waitComplete, reportValue, [mbar], state-or-phase;`
# Returns `(reportPredicate, reportValue)` as `Tuple{Bool, UInt64}`. The
# predicate is True iff the primary phase completed AND the payload report
# is zero (i.e. no fabric op flagged an error). On layout::v0 mbarriers the
# value is always zero — the predicate then collapses to plain phase-completion.
#
# `:report` is a synthetic modifier (no PTX counterpart, mirrors `setp :dual`)
# that flags this dual-output dispatch — the chain has no other signal to
# choose between single-output and dual-output return shapes.

@generated function (::Operation{:mbarrier, (:test_wait, :report, Symbol("phase_type::primary"), :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        state::Integer) where T
    spec = mbarrier_spec(:test_wait, :dual_pred_value_addr_state;
                         phase_type = :primary)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true,
                 Tuple{Bool, UInt64}, Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt64},
                 mbar, UInt64(state))
    end
end

@generated function (::Operation{:mbarrier, (:test_wait, :report, :parity, Symbol("phase_type::primary"), :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        phase::Integer) where T
    spec = mbarrier_spec(Symbol("test_wait.parity"), :dual_pred_value_addr_phase;
                         phase_type = :primary)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true,
                 Tuple{Bool, UInt64}, Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt32},
                 mbar, UInt32(phase))
    end
end

@generated function (::Operation{:mbarrier, (:try_wait, :report, Symbol("phase_type::primary"), :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        state::Integer) where T
    spec = mbarrier_spec(:try_wait, :dual_pred_value_addr_state;
                         phase_type = :primary)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true,
                 Tuple{Bool, UInt64}, Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt64},
                 mbar, UInt64(state))
    end
end

@generated function (::Operation{:mbarrier, (:try_wait, :report, :parity, Symbol("phase_type::primary"), :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        phase::Integer) where T
    spec = mbarrier_spec(Symbol("try_wait.parity"), :dual_pred_value_addr_phase;
                         phase_type = :primary)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true,
                 Tuple{Bool, UInt64}, Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt32},
                 mbar, UInt32(phase))
    end
end

# --- PTX 9.3: `.phase_type::conditional` parity wait (single-output) --------
# `mbarrier.{test,try}_wait.parity.phase_type::conditional.shared.b64
#     waitComplete, [mbar], phaseParity;`
# Layout::v1 only. Observes conditional-phase advance — which only happens
# when the payload report is zero (no fabric errors). `.parity` is mandatory
# per the PTX 9.3 syntax block. layout::v0 conditional and primary phases
# advance in unison, so this variant is meaningful only with layout::v1.

@generated function (::Operation{:mbarrier, (:test_wait, :parity, Symbol("phase_type::conditional"), :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        phase::Integer) where T
    spec = mbarrier_spec(Symbol("test_wait.parity"), :pred_addr_phase;
                         phase_type = :conditional)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true,
                 Bool, Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt32},
                 mbar, UInt32(phase))
    end
end

@generated function (::Operation{:mbarrier, (:try_wait, :parity, Symbol("phase_type::conditional"), :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        phase::Integer) where T
    spec = mbarrier_spec(Symbol("try_wait.parity"), :pred_addr_phase;
                         phase_type = :conditional)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true,
                 Bool, Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt32},
                 mbar, UInt32(phase))
    end
end
