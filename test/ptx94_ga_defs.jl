# Shared numeric probe: each scalar load keeps its value live in global memory.
const _GA_LOAD_TYPES = (
    (:b8, UInt8), (:b16, UInt16), (:b32, UInt32), (:b64, UInt64),
    (:u8, UInt8), (:u16, UInt16), (:u32, UInt32), (:u64, UInt64),
    (:s8, Int8), (:s16, Int16), (:s32, Int32), (:s64, Int64),
    (:f32, Float32), (:f64, Float64),
)

@generated function _ga_readonly_loads!(out, input)
    body = Expr(:block, :(global_ptr = pointer(input)),
        :(generic_ptr = PTX.reinterpret_addrspace(Val(PTX.AS.Generic), global_ptr)))
    i = 0
    for raw in (false, true), space in ((), (:global,)), (kind, T) in _GA_LOAD_TYPES
        i += 1
        mods = (space..., kind, Symbol("proxy::readonly"))
        op = raw ? PTX.RawOperation : PTX.Operation
        ptr = isempty(space) ? :generic_ptr : :global_ptr
        bits = unsigned(T === Float32 ? UInt32 : T === Float64 ? UInt64 : T)
        call = :($op{:ld, $mods}()($ptr))
        push!(body.args, :(Base.@inbounds out[$i] = UInt64(reinterpret($bits, $call))))
    end
    push!(body.args, :(return nothing))
    body
end

# tcgen05.ld{.red}.spcompress (sm_107a): every registered form, results
# folded so each load stays live.
@generated function _ga_t5_ldspc!(out::CuDeviceVector{UInt32, 1}, taddr::UInt32)
    body = Expr(:block, :(acc = UInt32(0)), :(facc = 0.0f0))
    for mods in PTX.wrapper_asm_forms(:tcgen05_ldspc)
        red = mods[2] === :red
        push!(body.args, :(r = PTX.Operation{:tcgen05, $mods}()(taddr)))
        push!(body.args, :(acc ⊻= r[1][1] ⊻ r[2][1]))
        red && push!(body.args, :(facc += r[3]))
    end
    push!(body.args, :(ptx"tcgen05.wait::ld.sync.aligned"()))
    push!(body.args, :(Base.@inbounds out[1] = acc ⊻ reinterpret(UInt32, facc)))
    push!(body.args, :(return nothing))
    body
end

# spcompress / spdecompress (sm_107a): every registered form of one element
# width, every output register folded into the stored accumulator (the
# forms carry no sideeffect marker, so an unused result would be dropped).
_ga_sp_forms(family, elem) =
    [m for m in PTX.wrapper_asm_forms(family) if m[1] === elem]

_ga_sp_bits(s) = s === :b8 ? 8 : s === :b16 ? 16 : s === :b2 ? 2 : 4

@generated function _ga_spcompress!(out::CuDeviceVector{UInt32, 1},
                                    ::Val{elem}, seed::UInt32,
                                    spdesc::UInt32) where {elem}
    body = Expr(:block, :(acc = UInt32(0)))
    for mods in _ga_sp_forms(:spcompress, elem)
        num = parse(Int, String(mods[4])[2:end])
        data = Expr(:tuple, (:(seed + UInt32($i)) for i in 1:2 * num)...)
        push!(body.args, :(r = PTX.Operation{:spcompress, $mods}()($data, spdesc)))
        push!(body.args, :(acc ⊻= reduce(⊻, r[1]) ⊻ reduce(⊻, r[2])))
    end
    push!(body.args, :(Base.@inbounds out[1] = acc))
    push!(body.args, :(return nothing))
    body
end

# spdecompress: one kernel per form. The widest forms produce 128
# registers each, and consecutive forms sharing input values would keep
# more than the 255-register budget live in a single kernel.
@generated function _ga_spdecompress!(out::CuDeviceVector{UInt32, 1},
                                      ::Val{mods}, seed::UInt32) where {mods}
    e = _ga_sp_bits(mods[1])
    i = _ga_sp_bits(mods[2])
    s = parse(Int, split(String(mods[3])[5:end], ':')[1])
    num = parse(Int, String(mods[4])[2:end])
    nm = cld(s * i * num, 32)
    nc = cld(s * e * num, 32)
    mdata = Expr(:tuple, (:(seed & UInt32(0x11111111)) for _ in 1:nm)...)
    cdata = Expr(:tuple, (:(seed + UInt32($k)) for k in 1:nc)...)
    quote
        r = PTX.Operation{:spdecompress, $mods}()($mdata, $cdata)
        Base.@inbounds out[1] = reduce(⊻, r)
        return nothing
    end
end
# --- tcgen05 PTX ISA 9.4 mma forms (sm_107f) ----------------------------------
# One kernel per cta_group (ptxas rejects mixing cta_group::1/::2 in one
# function). Every form is a side-effecting asm call, so nothing is
# eliminated; the trailing store keeps the TMEM address live.
function _ga_t5_ti16_cg1!(out::Core.LLVMPtr{UInt32, 1}, d::UInt32,
                          adesc::UInt64, bdesc::UInt64, meta::UInt32,
                          idesc::UInt32, zcm::UInt64)
    mask = (UInt32(0), UInt32(0), UInt32(0), UInt32(0))
    ptx"tcgen05.mma.cta_group::1.kind::ti16"(d, adesc, bdesc, idesc, false)
    ptx"tcgen05.mma.cta_group::1.kind::ti16.collector::b::fill"(
        d, adesc, bdesc, idesc, mask, false)
    ptx"tcgen05.mma.cta_group::1.kind::ti16.collector::a::fill.collector::b::use"(
        d, d, bdesc, idesc, false)
    ptx"tcgen05.mma.cta_group::1.kind::ti16.ashift.collector::a::lastuse.collector::b::lastuse"(
        d, d, bdesc, idesc, mask, false)
    ptx"tcgen05.mma.sp.cta_group::1.kind::ti16.collector::b::fill"(
        d, adesc, bdesc, meta, idesc, false)
    ptx"tcgen05.mma.sp.cta_group::1.kind::ti16.ashift.collector::b::use"(
        d, d, bdesc, meta, idesc, mask, false)
    ptx"tcgen05.mma.ws.cta_group::1.kind::ti16"(d, adesc, bdesc, idesc, false)
    ptx"tcgen05.mma.ws.cta_group::1.kind::ti16.collector::b1::fill"(
        d, d, bdesc, idesc, false, zcm)
    ptx"tcgen05.mma.ws.sp.cta_group::1.kind::ti16.collector::b3::lastuse"(
        d, adesc, bdesc, meta, idesc, false)
    ptx"st.global.b32"(out, d)
    return nothing
end

function _ga_t5_ti16_cg2!(out::Core.LLVMPtr{UInt32, 1}, d::UInt32,
                          adesc::UInt64, bdesc::UInt64, meta::UInt32,
                          idesc::UInt32)
    mask = ntuple(_ -> UInt32(0), Val(8))
    ptx"tcgen05.mma.cta_group::2.kind::ti16.collector::b::lastuse"(
        d, adesc, bdesc, idesc, false)
    ptx"tcgen05.mma.cta_group::2.kind::ti16.collector::a::use.collector::b::fill"(
        d, adesc, bdesc, idesc, mask, false)
    ptx"tcgen05.mma.sp.cta_group::2.kind::ti16.ashift"(
        d, d, bdesc, meta, idesc, false)
    ptx"st.global.b32"(out, d)
    return nothing
end

function _ga_t5_collb_cg1!(out::Core.LLVMPtr{UInt32, 1}, d::UInt32,
                           adesc::UInt64, bdesc::UInt64, meta::UInt32,
                           idesc::UInt32, sa::UInt32, sb::UInt32)
    mask = (UInt32(0), UInt32(0), UInt32(0), UInt32(0))
    ptx"tcgen05.mma.cta_group::1.kind::f16.collector::b::fill"(
        d, adesc, bdesc, idesc, false)
    ptx"tcgen05.mma.cta_group::1.kind::f16.collector::a::fill.collector::b::use"(
        d, adesc, bdesc, idesc, mask, false, Val(5))
    ptx"tcgen05.mma.cta_group::1.kind::tf32.ashift.collector::a::lastuse.collector::b::lastuse"(
        d, d, bdesc, idesc, false, Val(1))
    ptx"tcgen05.mma.cta_group::1.kind::f8f6f4.collector::b::use"(
        d, d, bdesc, idesc, mask, false)
    ptx"tcgen05.mma.sp.cta_group::1.kind::f16.collector::b::fill"(
        d, adesc, bdesc, meta, idesc, false)
    ptx"tcgen05.mma.sp.cta_group::1.kind::tf32.ashift.collector::b::use"(
        d, d, bdesc, meta, idesc, mask, false, Val(15))
    ptx"tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::b::fill"(
        d, adesc, bdesc, idesc, sa, sb, false)
    ptx"tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::use.collector::b::lastuse"(
        d, d, bdesc, idesc, sa, sb, false)
    ptx"tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::fill.collector::b::use"(
        d, adesc, bdesc, meta, idesc, sa, sb, false)
    ptx"st.global.b32"(out, d)
    return nothing
end

# The sparse mxf4 kinds are a-variant-exclusive, so their B-collector forms
# assemble on sm_107a only.
function _ga_t5_collb_avariant!(out::Core.LLVMPtr{UInt32, 1}, d::UInt32,
                                adesc::UInt64, bdesc::UInt64, meta::UInt32,
                                idesc::UInt32, sa::UInt32, sb::UInt32)
    ptx"tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::fill.collector::b::fill"(
        d, adesc, bdesc, meta, idesc, sa, sb, false)
    ptx"tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::b::use"(
        d, d, bdesc, meta, idesc, sa, sb, false)
    ptx"st.global.b32"(out, d)
    return nothing
end

function _ga_t5_lut_cg1!(out::Core.LLVMPtr{UInt32, 1}, d::UInt32,
                         adesc::UInt64, bdesc::UInt64, lut::UInt32,
                         idesc::UInt32, sa::UInt32, sb::UInt32)
    mask = (UInt32(0), UInt32(0), UInt32(0), UInt32(0))
    ptx"tcgen05.mma.cta_group::1.kind::f8f6f4.decompress::lut::b"(
        d, adesc, bdesc, lut, idesc, false)
    ptx"tcgen05.mma.cta_group::1.kind::f8f6f4.decompress::lut::b.collector::a::fill.collector::b::use"(
        d, d, bdesc, lut, idesc, mask, false)
    ptx"tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.decompress::lut::b.block32"(
        d, adesc, bdesc, lut, idesc, sa, sb, false)
    ptx"tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.decompress::lut::b.block32.collector::b::lastuse"(
        d, d, bdesc, lut, idesc, sa, sb, false)
    ptx"st.global.b32"(out, d)
    return nothing
end

const _GA_T5_TI16_CG1_TT = Tuple{Core.LLVMPtr{UInt32, 1}, UInt32, UInt64,
                                 UInt64, UInt32, UInt32, UInt64}
const _GA_T5_TI16_CG2_TT = Tuple{Core.LLVMPtr{UInt32, 1}, UInt32, UInt64,
                                 UInt64, UInt32, UInt32}
const _GA_T5_MX_TT = Tuple{Core.LLVMPtr{UInt32, 1}, UInt32, UInt64, UInt64,
                           UInt32, UInt32, UInt32, UInt32}
