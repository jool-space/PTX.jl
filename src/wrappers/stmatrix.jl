# Warp-cooperative register→shared matrix store (PTX 9.2 §9.7.14.5.16),
# sm_90+ — tier-2 intrinsic lowering, mirror of `wrappers/ldmatrix.jl`
# (state space via pointer AS, `convergent` from the registry; see that
# file's banner for the full rationale).
#
# Surface:
#   - m8n8[.trans].b16 × x1/x2/x4 — tier 2; x1 takes a bare UInt32, the
#     wider counts take NTuple{n, UInt32} (flattened at the call).
#   - `shared::cta` chain forms keep their spelling via convergent_asm_ir
#     below.
#   - m16n8.trans.b8 × x1/x2/x4 (sm_100a family) — tier 2. The old asm
#     generator's non-trans m16n8 forms were ptxas-invalid (`.trans` is
#     mandatory for the shape) — removed, not migrated.

@inline optype"stmatrix.sync.aligned.m8n8.x1.shared.b16"(
        addr::Core.LLVMPtr{T, AS.Shared}, reg::UInt32) where T =
    ceiled(nvvm"stmatrix.sync.aligned.m8n8.x1.b16",
           ptx"stmatrix.sync.aligned.m8n8.x1.shared.b16")(addr, reg)
@inline optype"stmatrix.sync.aligned.m8n8.x1.trans.shared.b16"(
        addr::Core.LLVMPtr{T, AS.Shared}, reg::UInt32) where T =
    ceiled(nvvm"stmatrix.sync.aligned.m8n8.x1.trans.b16",
           ptx"stmatrix.sync.aligned.m8n8.x1.trans.shared.b16")(addr, reg)
@inline optype"stmatrix.sync.aligned.m8n8.x2.shared.b16"(
        addr::Core.LLVMPtr{T, AS.Shared}, regs::NTuple{2, UInt32}) where T =
    ceiled(nvvm"stmatrix.sync.aligned.m8n8.x2.b16",
           ptx"stmatrix.sync.aligned.m8n8.x2.shared.b16")(addr, regs[1], regs[2])
@inline optype"stmatrix.sync.aligned.m8n8.x2.trans.shared.b16"(
        addr::Core.LLVMPtr{T, AS.Shared}, regs::NTuple{2, UInt32}) where T =
    ceiled(nvvm"stmatrix.sync.aligned.m8n8.x2.trans.b16",
           ptx"stmatrix.sync.aligned.m8n8.x2.trans.shared.b16")(addr, regs[1], regs[2])
@inline optype"stmatrix.sync.aligned.m8n8.x4.shared.b16"(
        addr::Core.LLVMPtr{T, AS.Shared}, regs::NTuple{4, UInt32}) where T =
    ceiled(nvvm"stmatrix.sync.aligned.m8n8.x4.b16",
           ptx"stmatrix.sync.aligned.m8n8.x4.shared.b16")(
        addr, regs[1], regs[2], regs[3], regs[4])
@inline optype"stmatrix.sync.aligned.m8n8.x4.trans.shared.b16"(
        addr::Core.LLVMPtr{T, AS.Shared}, regs::NTuple{4, UInt32}) where T =
    ceiled(nvvm"stmatrix.sync.aligned.m8n8.x4.trans.b16",
           ptx"stmatrix.sync.aligned.m8n8.x4.trans.shared.b16")(
        addr, regs[1], regs[2], regs[3], regs[4])

# b8 shape (sm_100a family).
@inline optype"stmatrix.sync.aligned.m16n8.x1.trans.shared.b8"(
        addr::Core.LLVMPtr{T, AS.Shared}, reg::UInt32) where T =
    ceiled(nvvm"stmatrix.sync.aligned.m16n8.x1.trans.b8",
           ptx"stmatrix.sync.aligned.m16n8.x1.trans.shared.b8")(addr, reg)
@inline optype"stmatrix.sync.aligned.m16n8.x2.trans.shared.b8"(
        addr::Core.LLVMPtr{T, AS.Shared}, regs::NTuple{2, UInt32}) where T =
    ceiled(nvvm"stmatrix.sync.aligned.m16n8.x2.trans.b8",
           ptx"stmatrix.sync.aligned.m16n8.x2.trans.shared.b8")(addr, regs[1], regs[2])
@inline optype"stmatrix.sync.aligned.m16n8.x4.trans.shared.b8"(
        addr::Core.LLVMPtr{T, AS.Shared}, regs::NTuple{4, UInt32}) where T =
    ceiled(nvvm"stmatrix.sync.aligned.m16n8.x4.trans.b8",
           ptx"stmatrix.sync.aligned.m16n8.x4.trans.shared.b8")(
        addr, regs[1], regs[2], regs[3], regs[4])

# --- `shared::cta` spellings, asm tier ---------------------------------------
# Same instruction as `.shared` (explicit-default sub-qualifier); via
# convergent_asm_ir so the calls carry `convergent nomerge`. Shared-AS
# pointer binds `r` (i32): NVPTX lowers shared-AS pointers to u32
# registers; `l` (i64) emits `%rd` and ptxas rejects.
for (count, n) in ((:x1, 1), (:x2, 2), (:x4, 4)), trans in (false, true)
    mods = trans ?
        (:sync, :aligned, :m8n8, count, :trans, Symbol("shared::cta"), :b16) :
        (:sync, :aligned, :m8n8, count, Symbol("shared::cta"), :b16)
    ttext = trans ? ".trans" : ""
    ins = "{" * join(("\$$i" for i in 1:n), ", ") * "}"
    asm = "stmatrix.sync.aligned.m8n8.$count$ttext.shared::cta.b16 [\$0], $ins;"
    constraints = join(vcat(["r"], fill("r", n), ["~{memory}"]), ",")
    argtypes = (Core.LLVMPtr{UInt16, AS.Shared}, ntuple(_ -> UInt32, n)...)
    ir = convergent_asm_ir(asm, constraints, Nothing, argtypes)
    if n == 1
        @eval function (::Operation{:stmatrix, $mods})(
                addr::Core.LLVMPtr{T, AS.Shared}, reg::UInt32) where T
            Base.@inline
            Base.llvmcall(($ir, "entry"), Nothing,
                          Tuple{Core.LLVMPtr{T, AS.Shared}, UInt32}, addr, reg)
            nothing
        end
    else
        reg_args = [:(regs[$i]) for i in 1:n]
        flat = fill(:UInt32, n)
        @eval function (::Operation{:stmatrix, $mods})(
                addr::Core.LLVMPtr{T, AS.Shared},
                regs::NTuple{$n, UInt32}) where T
            Base.@inline
            Base.llvmcall(($ir, "entry"), Nothing,
                          Tuple{Core.LLVMPtr{T, AS.Shared}, $(flat...)},
                          addr, $(reg_args...))
            nothing
        end
    end
end
