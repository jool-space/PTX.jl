# Hand-written: three distinct return shapes (Nothing / UInt64 state / Bool
# pred) the chain can't infer from the trailing `.b64` modifier, plus a
# shared-AS pointer that needs the `r` (i32) constraint instead of the chain
# default `l`. PTX 9.2 §9.7.12.15.

# Operand-shape categories.
#   :addr_only       — `[$0]`
#   :addr_count      — `[$0], $1`
#   :state_addr      — `$0, [$1]`
#   :state_addr_n    — `$0, [$1], $2`
#   :pred_addr_state — `$0, [$1], $2`  (state input)
#   :pred_addr_phase — `$0, [$1], $2`  (phase input)
#   :sink_addr       — `_, [$0]`        (cluster-scope arrive: no state token)
#   :sink_addr_n     — `_, [$0], $1`    (cluster-scope arrive with count/tx)
function mbarrier_spec(op::Symbol, layout::Symbol; ss::Symbol = :shared)
    head = "mbarrier.$op.$ss.b64"
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
    elseif layout === :pred_addr_state
        return (; asm = "$head \$0, [\$1], \$2;",
                  constraints = "=b,r,l,~{memory}",
                  rettype = Bool)
    elseif layout === :pred_addr_phase
        return (; asm = "$head \$0, [\$1], \$2;",
                  constraints = "=b,r,r,~{memory}",
                  rettype = Bool)
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
        count::UInt32) where T
    spec = mbarrier_spec(:init, :addr_count)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true, Nothing,
                 Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt32},
                 mbar, count)
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
        count::UInt32) where T
    spec = mbarrier_spec(Symbol("arrive.noComplete"), :state_addr_n)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true,
                 UInt64, Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt32},
                 mbar, count)
    end
end

# (sm_90+) Fused expect+arrive: records expected-tx-bytes and bumps pending by 1.
@generated function (::Operation{:mbarrier, (:arrive, :expect_tx, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        tx_count::UInt32) where T
    spec = mbarrier_spec(Symbol("arrive.expect_tx"), :state_addr_n)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true,
                 UInt64, Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt32},
                 mbar, tx_count)
    end
end

# (sm_90+) Standalone form: no arrive, no state output.
@generated function (::Operation{:mbarrier, (:expect_tx, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        tx_count::UInt32) where T
    spec = mbarrier_spec(:expect_tx, :addr_count)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true, Nothing,
                 Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt32},
                 mbar, tx_count)
        nothing
    end
end

# Token form: pass the UInt64 returned by a prior arrive on the same mbar.
@generated function (::Operation{:mbarrier, (:test_wait, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        state::UInt64) where T
    spec = mbarrier_spec(:test_wait, :pred_addr_state)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true,
                 Bool, Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt64},
                 mbar, state)
    end
end

# Phase form: pass a 0/1 phase parity bit; returns true once a full arrive
# cycle has completed.
@generated function (::Operation{:mbarrier, (:test_wait, :parity, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        phase::UInt32) where T
    spec = mbarrier_spec(Symbol("test_wait.parity"), :pred_addr_phase)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true,
                 Bool, Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt32},
                 mbar, phase)
    end
end

# (sm_90+) Suspend-allowing wait — hardware may park the warp on miss instead
# of returning false immediately.
@generated function (::Operation{:mbarrier, (:try_wait, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        state::UInt64) where T
    spec = mbarrier_spec(:try_wait, :pred_addr_state)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true,
                 Bool, Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt64},
                 mbar, state)
    end
end

@generated function (::Operation{:mbarrier, (:try_wait, :parity, :shared, :b64)})(
        mbar::Core.LLVMPtr{T, AS.Shared},
        phase::UInt32) where T
    spec = mbarrier_spec(Symbol("try_wait.parity"), :pred_addr_phase)
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true,
                 Bool, Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt32},
                 mbar, phase)
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
        tx_count::UInt32) where T
    spec = mbarrier_spec(Symbol("arrive.expect_tx"), :sink_addr_n;
                         ss = Symbol("shared::cluster"))
    quote
        Base.@inline
        @asmcall($(spec.asm), $(spec.constraints), true, Nothing,
                 Tuple{Core.LLVMPtr{$T, AS.Shared}, UInt32},
                 mbar, tx_count)
        nothing
    end
end
