# Shared kernels for the PTX ISA 9.4 bindings whose other coverage is the
# host-tier spelling oracle: cvt qualifiers, mbarrier cluster multicast, tcgen05 exclusive
# allocation and 9.4 commit forms, prefetch, ldmatrix .s8.s4, and the
# non-flushing f32 atomics. Every form with a result stores it so nothing
# is dead-code-eliminated; the offline (ptxas) and runtime legs both include
# this file so the exact same kernels are assembled and executed.

using PTX: smem_addr_u32

# --- cvt (sm_107f) -----------------------------------------------------------
# Independent enumeration from §9.7.10.24; the ledger is not consulted to
# build this list. Operand kinds come from the schema so the argument
# carriers match the reviewed ABI.
const _BIND_CVT94_FORMS = let forms = Tuple{Vararg{Symbol}}[]
    n1 = Symbol("scaled::n1::ue8m0")
    n2 = Symbol("scaled::n2::ue8m0")
    # .rz on the five 8-/6-/4-bit x2 destinations from f32, f16x2, bf16x2.
    for dst in (:e4m3x2, :e5m2x2, :e2m3x2, :e3m2x2, :e2m1x2),
            src in (:f32, :f16x2, :bf16x2)
        push!(forms, (:rz, :satfinite, dst, src))
    end
    # ue5m3x2: frnd4 down-converts from f32, rn from the packed halves,
    # up-converts into both packed halves, and the n2-scaled bf16x2 form.
    for rnd in (:rn, :rz, :rp)
        push!(forms, (rnd, :satfinite, :ue5m3x2, :f32))
    end
    for src in (:f16x2, :bf16x2)
        push!(forms, (:rn, :satfinite, :ue5m3x2, src))
    end
    push!(forms, (:rn, :f16x2, :ue5m3x2))
    push!(forms, (:rn, :bf16x2, :ue5m3x2))
    push!(forms, (:rn, :satfinite, n2, :bf16x2, :ue5m3x2))
    # .scaled::n1::ue8m0 on every narrow x2 destination from each source.
    for dst in (:e4m3x2, :e5m2x2, :e2m1x2, :e2m3x2, :e3m2x2, :ue5m3x2),
            src in (:f32, :f16x2, :bf16x2)
        push!(forms, (:rn, :satfinite, n1, dst, src))
    end
    # .pzo on float-to-narrower conversions, one per destination class.
    push!(forms, (:rn, :pzo, :f16, :f32))
    push!(forms, (:rn, :pzo, :bf16, :f32))
    push!(forms, (:rn, :pzo, :f16x2, :f32))
    push!(forms, (:rn, :pzo, :bf16x2, :f32))
    push!(forms, (:rn, :pzo, :tf32, :f32))
    push!(forms, (:rn, :satfinite, :pzo, :e4m3x2, :f32))
    push!(forms, (:rn, :satfinite, :pzo, :e2m1x2, :f16x2))
    push!(forms, (:rn, :satfinite, :pzo, n1, :e5m2x2, :bf16x2))
    forms
end

@generated function _bind_cvt94!(out::CuDeviceVector{UInt32, 1},
                                 f32a::Float32, f32b::Float32,
                                 u32::UInt32, u16::UInt16)
    body = Expr(:block)
    for (i, mods) in enumerate(_BIND_CVT94_FORMS)
        schema = PTX.schema(PTX.CvtLedger(), :cvt, mods)
        args = Symbol[]
        seen_f32 = 0
        for kind in schema.operands
            if kind === :f32
                seen_f32 += 1
                push!(args, seen_f32 == 1 ? :f32a : :f32b)
            elseif kind === :b32
                push!(args, :u32)
            elseif kind === :b16
                push!(args, :u16)
            else
                error("unexpected cvt operand kind $kind for $mods")
            end
        end
        argtypes = Tuple(a === :u32 ? UInt32 : a === :u16 ? UInt16 : Float32
                         for a in args)
        rettype = PTX.build_call(:cvt, mods, argtypes).rettype
        bits = sizeof(rettype) == 2 ? UInt16 : UInt32
        call = :(PTX.Operation{:cvt, $mods}()($(args...)))
        push!(body.args,
              :(Base.@inbounds out[$i] = UInt32(reinterpret($bits, $call))))
    end
    push!(body.args, :(return nothing))
    body
end

const _BIND_CVT94_TYPES = Tuple{CuDeviceVector{UInt32, 1}, Float32, Float32,
                                UInt32, UInt16}

# --- mbarrier .multicast::cluster::32b (sm_107f) -------------------------------
# Every reviewed schema carrying the 32-bit multicast qualifier, at every
# operand arity it admits. The mask is the mandatory trailing operand.
_bind_mbarrier32_schemas() =
    [s for s in PTX.MBARRIER_FORM_SCHEMAS
     if Symbol("multicast::cluster::32b") in s.mods]

@generated function _bind_mbarrier32!(mbar::Core.LLVMPtr{UInt64, PTX.AS.Shared},
                                      count::UInt32, mask::UInt32)
    body = Expr(:block)
    for s in _bind_mbarrier32_schemas(), v in s.variants
        args = length(v.operands) == 3 ? (:mbar, :count, :mask) : (:mbar, :mask)
        push!(body.args, :(PTX.Operation{:mbarrier, $(s.mods)}()($(args...))))
    end
    push!(body.args, :(return nothing))
    body
end

const _BIND_MBARRIER32_TYPES =
    Tuple{Core.LLVMPtr{UInt64, PTX.AS.Shared}, UInt32, UInt32}

# --- tcgen05 exclusive allocation and 9.4 commit forms -------------------------
# One kernel per cta_group; ptxas requires a single cta_group per kernel.
function _bind_tcgen05_alloc94_cg1!(slot::Core.LLVMPtr{UInt32, PTX.AS.Shared},
                                    ncols::UInt32)
    ptx"tcgen05.alloc.exclusive.cta_group::1.sync.aligned.b32"(slot, ncols)
    ptx"tcgen05.alloc.exclusive.cta_group::1.sync.aligned.shared::cta.b32"(
        smem_addr_u32(slot), ncols)
    taddr = @inbounds unsafe_load(slot)
    ptx"tcgen05.dealloc.exclusive.cta_group::1.sync.aligned.b32"(taddr, ncols)
    return nothing
end

function _bind_tcgen05_alloc94_cg2!(slot::Core.LLVMPtr{UInt32, PTX.AS.Shared},
                                    ncols::UInt32)
    ptx"tcgen05.alloc.exclusive.cta_group::2.sync.aligned.b32"(slot, ncols)
    ptx"tcgen05.alloc.exclusive.cta_group::2.sync.aligned.shared::cta.b32"(
        smem_addr_u32(slot), ncols)
    taddr = @inbounds unsafe_load(slot)
    ptx"tcgen05.dealloc.exclusive.cta_group::2.sync.aligned.b32"(taddr, ncols)
    return nothing
end

# The explicit ::16b spelling is the 9.4 name of the pre-9.4 default width
# and assembles on the sm_100 family; ::32b and the early A-read commit are
# sm_107f, so they live in their own kernels.
function _bind_tcgen05_commit16_cg1!(mbar::UInt32, mask::UInt16)
    ptx"tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster::16b.b64"(
        mbar, mask)
    return nothing
end

function _bind_tcgen05_commit16_cg2!(mbar::UInt32, mask::UInt16)
    ptx"tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster::16b.b64"(
        mbar, mask)
    return nothing
end

function _bind_tcgen05_commit94_cg1!(mbar::UInt32, mask::UInt32)
    ptx"tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster::32b.b64"(
        mbar, mask)
    ptx"tcgen05.commit.cta_group::1.mbarrier::arrive::one.sync_restrict::shared::read::mma::a.shared::cluster.b64"(
        mbar)
    ptx"tcgen05.commit.cta_group::1.mbarrier::arrive::one.sync_restrict::shared::read::mma::a.shared::cluster.multicast::cluster::32b.b64"(
        mbar, mask)
    return nothing
end

function _bind_tcgen05_commit94_cg2!(mbar::UInt32, mask::UInt32)
    ptx"tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster::32b.b64"(
        mbar, mask)
    ptx"tcgen05.commit.cta_group::2.mbarrier::arrive::one.sync_restrict::shared::read::mma::a.shared::cluster.b64"(
        mbar)
    ptx"tcgen05.commit.cta_group::2.mbarrier::arrive::one.sync_restrict::shared::read::mma::a.shared::cluster.multicast::cluster::32b.b64"(
        mbar, mask)
    return nothing
end

# --- prefetch .L1::32B.valid_addr and .add.noftz.f32 (sm_90) ------------------
function _bind_sm90_chain!(out::CuDeviceVector{Float32, 1},
                           g::Core.LLVMPtr{Float32, PTX.AS.Global},
                           value::Float32)
    generic = PTX.reinterpret_addrspace(Val(PTX.AS.Generic), g)
    ptx"prefetch.global.L1::32B.valid_addr"(g)
    ptx"prefetch.L1::32B.valid_addr"(generic)
    old_global = ptx"atom.global.add.noftz.f32"(g, value)
    old_generic = ptx"atom.add.noftz.f32"(generic, value)
    ptx"red.global.add.noftz.f32"(g, value)
    ptx"red.add.noftz.f32"(generic, value)
    @inbounds out[1] = old_global
    @inbounds out[2] = old_generic
    return nothing
end

const _BIND_SM90_CHAIN_TYPES = Tuple{CuDeviceVector{Float32, 1},
                                     Core.LLVMPtr{Float32, PTX.AS.Global},
                                     Float32}

# --- ldmatrix .m8n16 .s8.s4 (sm_90a and the sm_100f/sm_110f/sm_120f families) --
function _bind_ldmatrix_s8s4!(out::CuDeviceVector{UInt32, 1},
                              s::Core.LLVMPtr{UInt8, PTX.AS.Shared})
    a = ptx"ldmatrix.sync.aligned.m8n16.x1.shared.s8.s4"(s)
    b = ptx"ldmatrix.sync.aligned.m8n16.x2.shared.s8.s4"(s)
    c = ptx"ldmatrix.sync.aligned.m8n16.x4.shared.s8.s4"(s)
    d = ptx"ldmatrix.sync.aligned.m8n16.x1.shared::cta.s8.s4"(s)
    e = ptx"ldmatrix.sync.aligned.m8n16.x2.shared::cta.s8.s4"(s)
    f = ptx"ldmatrix.sync.aligned.m8n16.x4.shared::cta.s8.s4"(s)
    @inbounds begin
        out[1] = a
        out[2] = b[1]; out[3] = b[2]
        out[4] = c[1]; out[5] = c[2]; out[6] = c[3]; out[7] = c[4]
        out[8] = d
        out[9] = e[1]; out[10] = e[2]
        out[11] = f[1]; out[12] = f[2]; out[13] = f[3]; out[14] = f[4]
    end
    return nothing
end

const _BIND_LDMATRIX_TYPES = Tuple{CuDeviceVector{UInt32, 1},
                                   Core.LLVMPtr{UInt8, PTX.AS.Shared}}
