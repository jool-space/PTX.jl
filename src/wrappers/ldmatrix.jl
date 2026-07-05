# Warp-cooperative shared→register matrix load (PTX 9.2 §9.7.14.5.15) —
# tier-2 intrinsic lowering. The intrinsics carry the state space in the
# pointer's address space (llc emits the plain `.shared` spelling and can
# fold the shared symbol straight into the address operand), and carry
# `convergent` from the registry — the asm tier's sideeffect-only calls
# were duplicable across divergent branches, the collective-op miscompile
# class (CONCERNS.md, "Convergence on the asm tier").
#
# Surface:
#   - m8n8[.trans].b16 × x1/x2/x4 — tier 2.
#   - The explicit `shared::cta` chain forms keep their spelling via
#     convergent_asm_ir below: the intrinsics cannot spell `::cta`, and
#     the golden/text tests pin the qualified emission.
#   - m16n16.trans.b8 × x1/x2 (sm_100a family; the old "Hopper" banner
#     was wrong) — tier 2. Each 16×16×b8 tile is TWO regs per lane: x1
#     returns 2 registers, x2 returns 4.
#   - The old asm generator also emitted m16n16 non-trans and x4 forms
#     (ptxas-invalid: `.trans` is mandatory, count tops out at x2) at
#     1-reg-per-count arity (also ptxas-invalid). Never compilable, no
#     callers — removed, not migrated.
#
# Methods spell intrinsic names literally — test/host/conformance.jl
# scans for `nvvm"..."` literals and requires a selection probe for each.

@inline (::Operation{:ldmatrix, (:sync, :aligned, :m8n8, :x1, :shared, :b16)})(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m8n8.x1.b16"(addr)
@inline (::Operation{:ldmatrix, (:sync, :aligned, :m8n8, :x1, :trans, :shared, :b16)})(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m8n8.x1.trans.b16"(addr)
@inline (::Operation{:ldmatrix, (:sync, :aligned, :m8n8, :x2, :shared, :b16)})(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m8n8.x2.b16"(addr)
@inline (::Operation{:ldmatrix, (:sync, :aligned, :m8n8, :x2, :trans, :shared, :b16)})(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m8n8.x2.trans.b16"(addr)
@inline (::Operation{:ldmatrix, (:sync, :aligned, :m8n8, :x4, :shared, :b16)})(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m8n8.x4.b16"(addr)
@inline (::Operation{:ldmatrix, (:sync, :aligned, :m8n8, :x4, :trans, :shared, :b16)})(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m8n8.x4.trans.b16"(addr)

# b8 shapes (sm_100a family): 2 regs per count step.
@inline (::Operation{:ldmatrix, (:sync, :aligned, :m16n16, :x1, :trans, :shared, :b8)})(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m16n16.x1.trans.b8"(addr)
@inline (::Operation{:ldmatrix, (:sync, :aligned, :m16n16, :x2, :trans, :shared, :b8)})(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m16n16.x2.trans.b8"(addr)

# --- `shared::cta` spellings, asm tier ---------------------------------------
# Same instruction as `.shared` (the ::cta sub-qualifier is the explicit
# default); kept for callers that spell it. Emitted via convergent_asm_ir
# so the calls carry `convergent nomerge` like the intrinsic forms.
for (count, n) in ((:x1, 1), (:x2, 2), (:x4, 4)), trans in (false, true)
    mods = trans ?
        (:sync, :aligned, :m8n8, count, :trans, Symbol("shared::cta"), :b16) :
        (:sync, :aligned, :m8n8, count, Symbol("shared::cta"), :b16)
    ttext = trans ? ".trans" : ""
    outs = "{" * join(("\$$i" for i in 0:n-1), ", ") * "}"
    asm = "ldmatrix.sync.aligned.m8n8.$count$ttext.shared::cta.b16 $outs, [\$$n];"
    constraints = join(vcat(fill("=r", n), ["r", "~{memory}"]), ",")
    rettype = n == 1 ? UInt32 : NTuple{n, UInt32}
    ir = convergent_asm_ir(asm, constraints, rettype,
                           (Core.LLVMPtr{UInt16, AS.Shared},))
    @eval function (::Operation{:ldmatrix, $mods})(
            addr::Core.LLVMPtr{T, AS.Shared}) where T
        Base.@inline
        Base.llvmcall(($ir, "entry"), $rettype,
                      Tuple{Core.LLVMPtr{T, AS.Shared}}, addr)
    end
end
