# The effect ceiling — the reviewed form contracts are the maximum
# permissiveness any lowering route may hand the optimizer; the
# machine-extracted NVVM table may refine below the ceiling (speculatable,
# !range, param attrs, narrower memory location classes) but never exceed
# it. This suite is the tripwire: a backend bump that relaxes an
# intrinsic's properties, or a contract edit that tightens one, must
# surface here (and at every ceiled call site's generation) as a named
# failure demanding either a reviewed overlay or an ISA review — never a
# silent pass.

using PTX: NVVM
using PTX.NVVM: @nvvm_str

@testset "every intrinsic-tier wrapper record passes its ceiling" begin
    recs = [r for r in PTX.WRAPPER_REGISTRY if r.tier === :intrinsic]
    # Hand-pinned census: the generated mma/tcgen05 families. A new family
    # or a tier flip must update this deliberately.
    @test length(recs) == 1355
    failures = String[]
    for r in recs
        contract = PTX.form_contract(r.op, r.mods)
        contract === nothing &&
            (push!(failures, "$(r.intrinsic): no contract"); continue)
        try
            NVVM.check_ceiling(r.intrinsic, PTX._ceiling(contract))
        catch err
            push!(failures,
                  "$(r.family) $(r.op).$(join(r.mods, '.')) -> " *
                  "$(r.intrinsic): " * sprint(showerror, err))
        end
    end
    # The reviewed-exception list. Deliberately EMPTY: every divergence
    # found while building the ceiling was resolved by a reviewed overlay
    # (emit.jl) or a reviewed contract change (forms.jl). A future entry
    # here is a decision, not a suppression.
    expected_exceptions = String[]
    @test failures == expected_exceptions
end

@testset "every src nvvm literal is ceiled" begin
    # Source scan in the style of conformance's used_intrinsics(): a bare
    # `nvvm"..."` call site carries no ceiling, so inside the package every
    # literal must flow through `ceiled(nvvm"...", ptx"...")` (or the name
    # must be a non-registered docstring placeholder). The allowlist is
    # empty and stays empty.
    srcdir = joinpath(pkgdir(PTX), "src")
    offenders = String[]
    for (root, _, files) in walkdir(srcdir), file in files
        endswith(file, ".jl") || continue
        path = joinpath(root, file)
        source = read(path, String)
        for m in eachmatch(r"(ceiled\(\s*)?nvvm\"([^\"]*)\"", source)
            name = m.captures[2]
            full = startswith(name, "llvm.nvvm.") ? name :
                   "llvm.nvvm." * name
            NVVM.isintrinsic(full) || continue   # doc placeholders
            m.captures[1] === nothing &&
                push!(offenders, relpath(path, srcdir) * ": nvvm\"$name\"")
        end
    end
    allowlist = String[]
    @test sort(offenders) == allowlist
end

@testset "sreg route carries the :pure ceiling it claims" begin
    # Independent re-derivation: every intrinsic the sreg fast path can
    # emit must be :pure-class in the table (nomem, no sideeffects), or the
    # SREG_CEILING promise in dsl/sregs.jl is a lie.
    @test length(PTX.NVVM_SREG_U32) == 32
    for (sreg, suffix) in PTX.NVVM_SREG_U32
        i = NVVM.intrinsic("llvm.nvvm.read.ptx.sreg." * suffix)
        @test NVVM.observability_class(i) === :pure
        @test !NVVM.is_convergent(i)
    end
end

@testset "the ceiling check has teeth (negative controls)" begin
    # Class axis: a :pure-class intrinsic under a :clobbers ceiling is a
    # permissive-direction divergence and must throw.
    pure_name = "llvm.nvvm.add.rn.f"
    @test NVVM.observability_class(NVVM.intrinsic(pure_name)) === :pure
    @test_throws ErrorException NVVM.check_ceiling(
        pure_name, NVVM.Ceiling(0x01, false))
    # ... and passes under its true ceiling.
    @test NVVM.check_ceiling(pure_name, NVVM.Ceiling(0x03, false)) === nothing

    # :observable-class (nomem + sideeffects) under a :clobbers ceiling —
    # still more permissive than reviewed, still a violation.
    @test NVVM.observability_class(NVVM.intrinsic("llvm.nvvm.nanosleep")) ===
          :observable
    @test_throws ErrorException NVVM.check_ceiling(
        "llvm.nvvm.nanosleep", NVVM.Ceiling(0x01, false))
    @test NVVM.check_ceiling("llvm.nvvm.nanosleep",
                             NVVM.Ceiling(0x02, false)) === nothing

    # Convergence axis: a non-convergent table record under a convergent
    # ceiling must throw; the stricter direction is allowed by construction
    # (cp.async.bulk.tensor: table convergent, contract not).
    @test !NVVM.is_convergent(NVVM.intrinsic("llvm.nvvm.fence.proxy.async"))
    @test_throws ErrorException NVVM.check_ceiling(
        "llvm.nvvm.fence.proxy.async", NVVM.Ceiling(0x01, true))
    g2s = "llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d"
    @test NVVM.is_convergent(NVVM.intrinsic(g2s))
    tma_contract = PTX.form_contract(:cp, (:async, :bulk, :tensor))
    @test !tma_contract.convergent
    @test NVVM.check_ceiling(g2s, PTX._ceiling(tma_contract)) === nothing

    # The overlays feed the check: tcgen05.wait's widened memory props keep
    # it :memory-class (fits any ceiling); mma's overlaid convergence
    # satisfies the convergent + :pure mma contract.
    @test NVVM.observability_class(
        NVVM.intrinsic("llvm.nvvm.tcgen05.wait.ld")) === :memory
    mma_contract = PTX.form_contract(:mma, (:sync, :aligned))
    @test mma_contract.effects === :pure && mma_contract.convergent
    @test NVVM.check_ceiling("llvm.nvvm.mma.m16n8k16.row.col.bf16",
                             PTX._ceiling(mma_contract)) === nothing

    # No ceiling means no check — the bare macro's sentinel.
    @test NVVM.check_ceiling(pure_name, nothing) === nothing
    @test NVVM.callsite_ceiling(nvvm"add.rn.f") === nothing
end