# Static package-quality gates. Host-only, no GPU stack exercised.
#
# Aqua checks the package skeleton (method ambiguities, type piracy, stale
# deps, compat completeness, undefined exports); ExplicitImports checks that
# every cross-module name is either explicitly imported or deliberately
# qualified, so a dependency reshuffling its exports breaks loudly here
# instead of silently rebinding names.

using Aqua
using ExplicitImports
using PTX

# The persistent-tasks check loads PTX in a fresh process and waits for it to
# exit; the default 30 s is too tight when the GB10 runners are busy (a tag
# push runs CI twice on one commit). A task that never finishes still fails.
@testset "Aqua" begin
    Aqua.test_all(PTX; persistent_tasks = (; tmax = 120))
end

# check_all_explicit_imports_are_public is deliberately NOT enforced: the
# CUDACore extension imports PTX's own tensor-map internals (the normal
# parent-internals pattern for extensions) and CUDACore's libcuda driver
# bindings (CUtensorMap*, cuuint*_t), which CUDACore does not mark public.
# @enumx generates baremodules ExplicitImports cannot analyze; list them
# explicitly so a new unanalyzable module is still a loud failure.
const _ENUMX_MODULES = (PTX.IR.ScalarType, PTX.IR.StateSpace,
                        PTX.IR.VectorShape, PTX.IR.LinkingDirective,
                        PTX.Parser.TokenKind)

@testset "ExplicitImports" begin
    @test check_no_stale_explicit_imports(
        PTX; allow_unanalyzable = _ENUMX_MODULES) === nothing
    @test check_no_self_qualified_accesses(PTX) === nothing
end
