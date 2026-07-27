# `fabric.*` — CUDA Compute Fabric Transport (CFT) instructions, PTX 9.3 /
# sm_100+. Async NVLink-fabric ops; completion observed via `mbarrier` with
# `.layout::v1` (see wrappers/mbarrier.jl for the report-aware test_wait /
# try_wait variants).
#
# Asm tier by necessity: the NVVM registry (LLVM 22.1.7) has no fabric
# intrinsics. Hand-written rather than chain-default: the two-register CFT
# handle operand `[leId, off]` (u32 + u64 inside one bracketed group) has no
# chain-default rendering, so `:fabric` is deliberately NOT in the form
# registry (src/ledgers/forms.jl) — an unimplemented fabric form errors at the
# blessing boundary instead of rendering wrong asm; `ptx"..."raw` remains
# the explicit escape hatch.
#
# Scope: `submit`, `wait`, `try_get`, `try_put` (basic + .multimem).
# DEFERRED:
#   * `fabric.try_red`     — full redOp × type matrix
#   * `fabric.try_pullred` — full redOp × type matrix, multimem-only
#   * `try_put` `.cp_mask` (bytemask operand) and `.counted::bytes` (third
#     register `dstCounterOff` inside the handle bracket) forms
# All of these are extension-shaped: add another set of @generated methods
# alongside the existing ones; no core infrastructure changes needed.

# --- Zero-arg lifecycle ops -------------------------------------------------
# `fabric.submit;` and `fabric.submit.op_restrict::fetching;` — submit prior
# fabric operations issued by this thread so a subsequent mbarrier phase
# advance can observe their completion. Required before grid exit.
# `fabric.wait.sync_restrict::reads;` — wait on local shared-memory reads of
# submitted fabric ops; lets the program overwrite source smem before the
# entire op completes (fabric-read completion mechanism).
#
# All fabric ops carry sideeffect + `~{memory}`: they have observable
# cross-GPU visibility, and LLVM must not reorder them around the mbarrier
# submit/wait lifecycle.

@generated function optype"fabric.submit"()
    quote
        Base.@inline
        @asmcall("fabric.submit;",
                 "~{memory}", true, Nothing,
                 Tuple{})
        nothing
    end
end

@generated function optype"fabric.submit.op_restrict::fetching"()
    quote
        Base.@inline
        @asmcall("fabric.submit.op_restrict::fetching;",
                 "~{memory}", true, Nothing,
                 Tuple{})
        nothing
    end
end

@generated function optype"fabric.wait.sync_restrict::reads"()
    quote
        Base.@inline
        @asmcall("fabric.wait.sync_restrict::reads;",
                 "~{memory}", true, Nothing,
                 Tuple{})
        nothing
    end
end

# --- fabric.try_get ---------------------------------------------------------
# `fabric.try_get.async.shared::cta.mbarrier::complete_tx::bytes
#    .mbarrier::report::fabric.relaxed.sys.b128
#    [dstSmem], [srcLeId, srcDataOff], size, [mbar];`
#
# Operands:
#   $0 dst        shared::cta   ptr   (r)
#   $1 srcLeId    u32                 (r)
#   $2 srcDataOff u64                 (l)
#   $3 size       u32                 (r)
#   $4 mbar       shared::cta   ptr   (r)
#
# All qualifiers (sem/scope/dst/completion_mechanism/type) are mandatory per
# the PTX 9.3 syntax — there is no per-call variability, so the mod tuple is
# pinned at registration.

const _FABRIC_TRY_GET_HEAD =
    "fabric.try_get.async.shared::cta" *
    ".mbarrier::complete_tx::bytes.mbarrier::report::fabric" *
    ".relaxed.sys.b128"

@generated function optype"fabric.try_get.async.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.b128"(
        dst::Core.LLVMPtr{T, AS.Shared},
        srcLeId::Integer,
        srcDataOff::Integer,
        size::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}) where {T, U}
    asm = "$(_FABRIC_TRY_GET_HEAD) [\$0], [\$1, \$2], \$3, [\$4];"
    constraints = "r,r,l,r,r,~{memory}"
    quote
        Base.@inline
        @asmcall($asm, $constraints, true, Nothing,
                 Tuple{Core.LLVMPtr{$T, AS.Shared},
                       UInt32, UInt64, UInt32,
                       Core.LLVMPtr{$U, AS.Shared}},
                 dst, UInt32(srcLeId), UInt64(srcDataOff),
                 UInt32(size), mbar)
        nothing
    end
end

# --- fabric.try_put (basic + .multimem; no cp_mask / counted) ---------------
# `fabric.try_put.async{.multimem}.shared::cta.mbarrier::complete_tx::16B
#    .mbarrier::report::fabric.relaxed.sys.b128
#    [dstLeId, dstDataOff], [srcSmem], size, [mbar];`
#
# NOTE on completion mechanism: try_put uses `.mbarrier::complete_tx::16B`
# (count incremented per 16-byte chunk), distinct from try_get's
# `.mbarrier::complete_tx::bytes` (count = exact bytes).
#
# Operands ($0..$4):
#   $0 dstLeId    u32                 (r)
#   $1 dstDataOff u64                 (l)
#   $2 src        shared::cta  ptr    (r)
#   $3 size       u32                 (r)
#   $4 mbar       shared::cta  ptr    (r)

const _FABRIC_TRY_PUT_HEAD =
    "fabric.try_put.async.shared::cta" *
    ".mbarrier::complete_tx::16B.mbarrier::report::fabric" *
    ".relaxed.sys.b128"

const _FABRIC_TRY_PUT_MULTIMEM_HEAD =
    "fabric.try_put.async.multimem.shared::cta" *
    ".mbarrier::complete_tx::16B.mbarrier::report::fabric" *
    ".relaxed.sys.b128"

@generated function optype"fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.b128"(
        dstLeId::Integer,
        dstDataOff::Integer,
        src::Core.LLVMPtr{T, AS.Shared},
        size::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}) where {T, U}
    asm = "$(_FABRIC_TRY_PUT_HEAD) [\$0, \$1], [\$2], \$3, [\$4];"
    constraints = "r,l,r,r,r,~{memory}"
    quote
        Base.@inline
        @asmcall($asm, $constraints, true, Nothing,
                 Tuple{UInt32, UInt64,
                       Core.LLVMPtr{$T, AS.Shared},
                       UInt32, Core.LLVMPtr{$U, AS.Shared}},
                 UInt32(dstLeId), UInt64(dstDataOff), src,
                 UInt32(size), mbar)
        nothing
    end
end

@generated function optype"fabric.try_put.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.b128"(
        dstLeId::Integer,
        dstDataOff::Integer,
        src::Core.LLVMPtr{T, AS.Shared},
        size::Integer,
        mbar::Core.LLVMPtr{U, AS.Shared}) where {T, U}
    asm = "$(_FABRIC_TRY_PUT_MULTIMEM_HEAD) [\$0, \$1], [\$2], \$3, [\$4];"
    constraints = "r,l,r,r,r,~{memory}"
    quote
        Base.@inline
        @asmcall($asm, $constraints, true, Nothing,
                 Tuple{UInt32, UInt64,
                       Core.LLVMPtr{$T, AS.Shared},
                       UInt32, Core.LLVMPtr{$U, AS.Shared}},
                 UInt32(dstLeId), UInt64(dstDataOff), src,
                 UInt32(size), mbar)
        nothing
    end
end
