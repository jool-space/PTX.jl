# Compile-touch sweep: every wrapper-owned Operation method must compile
# through its own Julia surface.
#
# The golden kernels are deliberately small and the gpu tiers use the
# workhorse forms, so a wrapper family's long tail (tcgen05 cta_group::2
# variants, TMA 1d/3d/4d/5d ranks, the ld/st shape×count grids) may never
# be device-compiled by any test — its intrinsic selection is pinned by
# conformance (which synthesizes IR directly, bypassing the wrapper), but
# the Julia-side glue (operand marshaling, address-space casts, tuple
# flattening) goes unverified. A busted argument list in a 4d TMA wrapper
# would surface for the first real user, not for CI.
#
# This sweep closes that: enumerate the wrapper method table, synthesize
# the argument-TYPE tuple per method (emit_ptx needs types, not values),
# and compile one trivial kernel per method at the family's cap floor. No
# hardware needed (backend compilation only), no goldens (nothing to
# maintain). As a side effect, GPUCompiler's compile-time coverage
# recording visits every touched wrapper line, so device-code coverage
# reflects the surface the package actually exposes.

# The kernel: dispatch the operation, discard the result. record_coverage
# fires at cache-populate time (pre-optimization), so even a pure form
# that DCEs away still verifies inference and records coverage.
_touch(o, args...) = (o(args...); return nothing)

const _WRAPPER_DIR = joinpath(dirname(dirname(@__DIR__)), "src", "wrappers")

# Cap floor per family — refined by mods where one opcode spans arch
# generations. Everything here is a *compilation* target; no device.
function _touch_target(op::Symbol, mods)
    has(s) = any(m -> occursin(s, String(m)), mods)
    op === :tcgen05 && return (v"10.0", :arch)
    op === :wgmma   && return (v"9.0", :arch)
    op === :mma     && (has("kind::") || :block_scale in mods) &&
        return (v"12.1", :arch)
    op === :mma     && return (v"9.0", :arch)
    (op === :ldmatrix || op === :stmatrix) && :b8 in mods &&
        return (v"10.0", :arch)
    # cta_group is a Blackwell cluster-pair feature; g2s with a nonzero
    # cta_group operand cannot ISel below sm_100 (ledger: validated sm_100a).
    op === :cp && has("cta_group::2") && return (v"10.0", :arch)
    op === :cvt && (has("e2m") || has("e3m") || has("e4m") || has("e5m") ||
                    has("ue8m0")) && return (v"12.1", :arch)
    (v"9.0", :arch)
end

# Argument-type synthesis from the method signature. Substitution rules:
#   - free TypeVars instantiate to their `_touch_tv` pick (LLVMPtr element
#     types and the like — UInt64 unless a failure teaches us otherwise)
#   - Union parameters take their first concrete member
_touch_tv(tv::TypeVar) = UInt64

function _touch_argtypes(m::Method)
    sig = m.sig
    while sig isa UnionAll
        sig = sig{_touch_tv(sig.var)}
    end
    out = Any[]
    for T in sig.parameters[2:end]
        if T isa Union
            members = Base.uniontypes(T)
            i = findfirst(isconcretetype, members)
            i === nothing && return nothing
            push!(out, members[i])
        elseif T === Integer
            # duck-typed coordinate/count/mask operands — any Integer works
            push!(out, Int32)
        elseif T isa DataType || T isa Core.TypeofBottom
            isconcretetype(T) || return nothing
            push!(out, T)
        else
            return nothing
        end
    end
    out
end

@testset "compile-touch: every wrapper method compiles through its surface" begin
    touched = 0
    failures = String[]
    unsynthesized = String[]
    PTX._visit_operation_methods() do m, op, mods
        String(m.file) == joinpath(_WRAPPER_DIR, basename(String(m.file))) ||
            startswith(String(m.file), _WRAPPER_DIR) || return
        argts = _touch_argtypes(m)
        label = "ptx\"$(join((String(op), String.(mods)...), '.'))\"" *
                "($(m.sig isa UnionAll ? "…" : join(m.sig.parameters[2:end], ", ")))"
        if argts === nothing
            push!(unsynthesized, label)
            return
        end
        cap, fs = _touch_target(op, mods)
        opT = Base.unwrap_unionall(m.sig).parameters[1]
        try
            emit_ptx(_touch, Tuple{opT, argts...}; cap, feature_set = fs)
            touched += 1
        catch err
            push!(failures, label * "  @sm_" * string(cap.major) * string(cap.minor) *
                            "\n      " * first(sprint(showerror, err), 200))
        end
    end
    isempty(failures) || foreach(f -> println("TOUCH FAILURE: ", f), failures)
    isempty(unsynthesized) ||
        foreach(u -> println("TOUCH UNSYNTHESIZED: ", u), unsynthesized)
    @test isempty(failures)
    # Argument synthesis must keep up with the wrapper surface: a method the
    # sweep cannot even construct types for is a hole in the sweep, not a
    # pass. Curate _touch_tv/_touch_argtypes when this fires.
    @test isempty(unsynthesized)
    # Regression floor: the sweep must actually be sweeping. Update when
    # the wrapper surface grows or shrinks deliberately.
    @test touched >= 250
    println("compile-touch: $touched wrapper methods compiled")
end
