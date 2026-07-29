# NVVM-backed shortcut for `mov.u32 %reg, %sreg`. Routing through
# `llvm.nvvm.read.ptx.sreg.*` lets LLVM CSE redundant reads and propagate
# range metadata — neither expressible via `@asmcall` + `~{memory}`. Only
# invariant-per-thread sregs are listed; volatile ones (clock, clock64,
# globaltimer, pm0..7, smid, warpid, nsmid, gridid) deliberately fall through
# to the asm path so LLVM doesn't collapse repeated reads. `activemask` is a
# PTX instruction, not a special register.
# This deliberately is not a mirror of the generated backend registry: LLVM
# retains `llvm.nvvm.read.ptx.sreg.warpsize`, while the PTX source surface uses
# WARP_SZ and therefore excludes `%warpsize` here. Names must exist in the
# backend registry (asserted in test/host/inst.jl);
# the IN-PROCESS LLVM need not know them — emission goes through the tier-2
# IntrinsicCall (declare+call), not `ccall(name, llvmcall, ...)`, precisely
# because ccall resolves against the in-process intrinsic table and demotes
# unknown names (the cluster sregs on LLVM ≤ 16 / Julia ≤ 1.11) to a
# runtime trap.
const NVVM_SREG_U32 = Dict{Symbol, String}(
    Symbol("%tid.x")    => "tid.x",    Symbol("%tid.y")    => "tid.y",    Symbol("%tid.z")    => "tid.z",
    Symbol("%ntid.x")   => "ntid.x",   Symbol("%ntid.y")   => "ntid.y",   Symbol("%ntid.z")   => "ntid.z",
    Symbol("%ctaid.x")  => "ctaid.x",  Symbol("%ctaid.y")  => "ctaid.y",  Symbol("%ctaid.z")  => "ctaid.z",
    Symbol("%nctaid.x") => "nctaid.x", Symbol("%nctaid.y") => "nctaid.y", Symbol("%nctaid.z") => "nctaid.z",
    Symbol("%laneid")   => "laneid",
    Symbol("%lanemask_eq") => "lanemask.eq", Symbol("%lanemask_lt") => "lanemask.lt",
    Symbol("%lanemask_le") => "lanemask.le", Symbol("%lanemask_ge") => "lanemask.ge",
    Symbol("%lanemask_gt") => "lanemask.gt",
    Symbol("%cluster_ctaid.x")  => "cluster.ctaid.x",
    Symbol("%cluster_ctaid.y")  => "cluster.ctaid.y",
    Symbol("%cluster_ctaid.z")  => "cluster.ctaid.z",
    Symbol("%cluster_nctaid.x") => "cluster.nctaid.x",
    Symbol("%cluster_nctaid.y") => "cluster.nctaid.y",
    Symbol("%cluster_nctaid.z") => "cluster.nctaid.z",
    Symbol("%cluster_ctarank")  => "cluster.ctarank",
    Symbol("%cluster_nctarank") => "cluster.nctarank",
    Symbol("%clusterid.x")  => "clusterid.x",
    Symbol("%clusterid.y")  => "clusterid.y",
    Symbol("%clusterid.z")  => "clusterid.z",
    Symbol("%nclusterid.x") => "nclusterid.x",
    Symbol("%nclusterid.y") => "nclusterid.y",
    Symbol("%nclusterid.z") => "nclusterid.z",
)

@generated function (::Operation{:mov, (:u32,)})(::SpecialReg{S}) where {S}
    suffix = get(NVVM_SREG_U32, S, nothing)
    if suffix !== nothing
        intr = "llvm.nvvm.read.ptx.sreg." * suffix
        # Ceiling: invariant sreg reads are :pure (the table refines with
        # speculatable/noundef/!range below that ceiling).
        return :( $(NVVM.IntrinsicCall{Symbol(intr), _ceiling(_PURE)}())() )
    end
    spec = build_call(:mov, (:u32,), (SpecialReg{S},))
    quote
        Base.@inline $(LLVM.Interop).@asmcall(
            $(spec.asm), $(spec.constraints), $(spec.side_effects),
            $(spec.rettype),
            Tuple{$(spec.passthrough_argtypes...)})
    end
end
