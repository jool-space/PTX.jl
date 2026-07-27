# Warp-cooperative shared→register matrix load (PTX 9.3 §9.7.15.5.15) —
# tier-2 intrinsic lowering. The intrinsics carry the state space in the
# pointer's address space (llc emits the plain `.shared` spelling and can
# fold the shared symbol straight into the address operand), and carry
# `convergent` from the registry — the asm tier's sideeffect-only calls
# were duplicable across divergent branches, the collective-op miscompile
# class.
#
# Surface:
#   - m8n8[.trans].b16 × x1/x2/x4 — tier 2.
#   - The explicit `shared::cta` chain forms keep their spelling via
#     convergent_asm_ir below: the intrinsics cannot spell `::cta`, and
#     the golden/text tests pin the qualified emission.
#   - m16n16.trans.b8 × x1/x2 (sm_100a family; the old "Hopper" banner
#     was wrong) — tier 2. Each 16×16×b8 tile is TWO regs per lane: x1
#     returns 2 registers, x2 returns 4.
#   - Optional decompression (PTX 9.3 §9.7.15.5.15, PTX 8.6):
#       m8n16.x1/x2/x4.{b8x16.b6x16_p32,b8x16.b4x16_p64}
#       m16n16.x1/x2.trans.{b8x16.b6x16_p32,b8x16.b4x16_p64}
#     on the sm_100/sm_110/sm_120 architecture families. Figure 106 gives
#     one b32 result per matrix for m8n16; Figure 105 gives two per matrix for
#     m16n16. Both `.shared` and its explicit-default `.shared::cta` spelling
#     are exposed; generic addressing remains outside this typed surface.
#   - The old asm generator also emitted m16n16 non-trans and x4 forms
#     (ptxas-invalid: `.trans` is mandatory, count tops out at x2) at
#     1-reg-per-count arity (also ptxas-invalid). Never compilable, no
#     callers — removed, not migrated.
#
# Methods spell intrinsic names literally — test/host/conformance.jl
# scans for `nvvm"..."` literals and requires a selection probe for each.

@inline optype"ldmatrix.sync.aligned.m8n8.x1.shared.b16"(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m8n8.x1.b16"(addr)
@inline optype"ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16"(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m8n8.x1.trans.b16"(addr)
@inline optype"ldmatrix.sync.aligned.m8n8.x2.shared.b16"(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m8n8.x2.b16"(addr)
@inline optype"ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16"(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m8n8.x2.trans.b16"(addr)
@inline optype"ldmatrix.sync.aligned.m8n8.x4.shared.b16"(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m8n8.x4.b16"(addr)
@inline optype"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m8n8.x4.trans.b16"(addr)

# b8 shapes (sm_100a family): 2 regs per count step.
@inline optype"ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8"(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m16n16.x1.trans.b8"(addr)
@inline optype"ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8"(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m16n16.x2.trans.b8"(addr)

# Optional decompression from packed unsigned 4-/6-bit source rows. The
# pointer element type is deliberately unconstrained: PTX consumes an address,
# while the formats (not Julia's pointee type) define the 128-bit row payload.
@inline optype"ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b4x16_p64"(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m8n16.x1.b8x16.b4x16_p64"(addr)
@inline optype"ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b6x16_p32"(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m8n16.x1.b8x16.b6x16_p32"(addr)
@inline optype"ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64"(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m8n16.x2.b8x16.b4x16_p64"(addr)
@inline optype"ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b6x16_p32"(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m8n16.x2.b8x16.b6x16_p32"(addr)
@inline optype"ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64"(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m8n16.x4.b8x16.b4x16_p64"(addr)
@inline optype"ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b6x16_p32"(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m8n16.x4.b8x16.b6x16_p32"(addr)

@inline optype"ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16.b4x16_p64"(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m16n16.x1.trans.b8x16.b4x16_p64"(addr)
@inline optype"ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16.b6x16_p32"(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m16n16.x1.trans.b8x16.b6x16_p32"(addr)
@inline optype"ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8x16.b4x16_p64"(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m16n16.x2.trans.b8x16.b4x16_p64"(addr)
@inline optype"ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8x16.b6x16_p32"(
        addr::Core.LLVMPtr{T, AS.Shared}) where T =
    nvvm"ldmatrix.sync.aligned.m16n16.x2.trans.b8x16.b6x16_p32"(addr)

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

# Optional-decompression `shared::cta` forms. NVVM's intrinsic namespace
# cannot request the explicit default sub-qualifier, so these use the same
# convergent call-site barrier as the pre-existing m8n8 `shared::cta` forms.
for (shape, count, nout, trans) in (
        (:m8n16,  :x1, 1, false),
        (:m8n16,  :x2, 2, false),
        (:m8n16,  :x4, 4, false),
        (:m16n16, :x1, 2, true),
        (:m16n16, :x2, 4, true),
    ), src_fmt in (:b6x16_p32, :b4x16_p64)
    transmods = trans ? (:trans,) : ()
    mods = (:sync, :aligned, shape, count, transmods...,
            Symbol("shared::cta"), :b8x16, src_fmt)
    ttext = trans ? ".trans" : ""
    outs = "{" * join(("\$$i" for i in 0:nout-1), ", ") * "}"
    asm = "ldmatrix.sync.aligned.$shape.$count$ttext.shared::cta.b8x16.$src_fmt " *
          "$outs, [\$$nout];"
    constraints = join(vcat(fill("=r", nout), ["r", "~{memory}"]), ",")
    rettype = nout == 1 ? UInt32 : NTuple{nout, UInt32}
    ir = convergent_asm_ir(asm, constraints, rettype,
                           (Core.LLVMPtr{UInt8, AS.Shared},))
    @eval function (::Operation{:ldmatrix, $mods})(
            addr::Core.LLVMPtr{T, AS.Shared}) where T
        Base.@inline
        Base.llvmcall(($ir, "entry"), $rettype,
                      Tuple{Core.LLVMPtr{T, AS.Shared}}, addr)
    end
end
