# Sparse warpgroup-level async MMA — `wgmma.mma_async.sp` (Hopper sm_90a).
#
# Matrix A is 2:4 structured sparse (1:2 for tf32): the MxK logical A is
# packed to Mx(K/2) non-zero elements, and a per-thread 32-bit metadata
# word supplies the position of each non-zero inside its 4-wide (2-wide
# for tf32) chunk. Relative to the dense chain (wgmma.jl) the instruction
# takes two extra operands (PTX 9.3 §9.7.16.6.3):
#
#   sp-meta — .b32 register. Which threads' metadata is consumed depends
#             on the shape: for k32/k16 shapes a thread-PAIR per quad
#             contributes (selected by sp-sel); for k64 shapes all
#             threads contribute. Layouts: Figures 176-181.
#   sp-sel  — 32-bit integer CONSTANT (0..3). Baked as an immediate via
#             `Val`: methods are generated for the values the spec
#             declares valid per shape — {0, 1} for k32/k16 (thread pair
#             T0,T1 or T2,T3), {0} for k64 (all threads contribute).
#             Passing anything else is a MethodError at compile time
#             instead of undefined behavior at run time.
#
# K is doubled relative to the dense shape for the same dtype (the packed
# A operand keeps the dense tile's byte size): bf16/f16 → k32, tf32 →
# k16, e4m3/e5m2/s8/u8 → k64. The A descriptor covers the PACKED Mx(K/2)
# tile — its SMEM layout is the canonical K-major layout of the
# equivalent dense Mx(K/2) tile, so `layout_for_a(dtype, m, k ÷ 2)` picks
# the right descriptor parameters.
#
# Only the SS form (both operands from SMEM descriptors) and the runtime
# `scale_d::Bool` variant are generated — mirroring which dense variants
# the corpus actually exercises; RF-A and baked-scale_d sparse forms can
# be added the same way when a kernel needs them.

# (dtype_d, dtype_a, dtype_b, k, has_trans) — sparse shapes. has_trans
# mirrors dense: only f16/bf16 SS descriptors accept `.trans`.
const _WGMMA_SP_VARIANTS = (
    (:f32, :bf16, :bf16, 32, true),
    (:f32, :f16,  :f16,  32, true),
    (:f16, :f16,  :f16,  32, true),
    (:f32, :tf32, :tf32, 16, false),
    (:f32, :e4m3, :e4m3, 64, false),
    (:f32, :e5m2, :e5m2, 64, false),
    (:f16, :e4m3, :e4m3, 64, false),
    (:f16, :e5m2, :e5m2, 64, false),
    (:s32, :s8, :s8, 64, false),
    (:s32, :u8, :u8, 64, false),
    (:s32, :s8, :u8, 64, false),
    (:s32, :u8, :s8, 64, false),
)

# Valid N values. Integer sparse shapes skip n40/n56/… (PTX 9.3
# §9.7.16.6.3 lists only these); FP shapes use the full dense list.
const _WGMMA_SP_INT_NS = (8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144,
                          160, 176, 192, 208, 224, 240, 256)

# Valid sp-sel immediates per K (§9.7.16.6.1): k64 = all threads
# contribute, selector must be 0; k32/k16 = thread-pair select.
_wgmma_sp_sels(k::Int) = k == 64 ? (0,) : (0, 1)

function wgmma_sp_spec(dtype_d::Symbol, dtype_a::Symbol, dtype_b::Symbol,
                       n::Int, k::Int, has_trans::Bool, sel::Int)
    nd, _, d_let = _wgmma_dvec_kind(dtype_d, n)
    head = "wgmma.mma_async.sp.sync.aligned.m64n$(n)k$(k).$dtype_d.$dtype_a.$dtype_b"
    d_slots = "{" * join(("\$$i" for i in 0:nd-1), ", ") * "}"
    imms_tail = if dtype_d === :s32
        ""
    elseif has_trans
        ", 1, 1, 0, 0"
    else
        ", 1, 1"
    end
    # Operand order (§9.7.16.6.3): d, a-desc, b-desc, sp-meta, sp-sel,
    # scale-d, imms. sp-sel is an integer constant — spliced literally.
    asm = "$head $d_slots, \$$(nd), \$$(nd+1), \$$(nd+2), $sel, \$$(nd+3)$imms_tail;"
    constraints = join(vcat(
        fill("=$d_let", nd),
        ["l", "l", "r", "b"],
        [string(i) for i in 0:nd-1],              # tied to outputs 0..nd-1
        ["~{memory}"],
    ), ",")
    (; nd, asm, constraints)
end

function _wgmma_sp_register(dtype_d::Symbol, dtype_a::Symbol, dtype_b::Symbol,
                            n::Int, k::Int, has_trans::Bool)
    _, d_J, _ = _wgmma_dvec_kind(dtype_d, n)
    mods = (:mma_async, :sp, :sync, :aligned,
            Symbol("m64n", n, "k", k),
            dtype_d, dtype_a, dtype_b)

    for sel in _wgmma_sp_sels(k)
        spec = wgmma_sp_spec(dtype_d, dtype_a, dtype_b, n, k, has_trans, sel)
        nd, asm, constraints = spec.nd, spec.asm, spec.constraints
        flat_argtypes = vcat([UInt64, UInt64, UInt32, Bool], fill(d_J, nd))
        ir = convergent_asm_ir(asm, constraints, NTuple{nd, d_J}, flat_argtypes)
        d_args = [:(d[$i]) for i in 1:nd]
        @eval function (::Operation{:wgmma, $mods})(
                d::NTuple{$nd, $d_J},
                a_desc::UInt64,
                b_desc::UInt64,
                sp_meta::UInt32,
                ::Val{$sel},
                scale_d::Bool)
            Base.@inline
            Base.llvmcall(($ir, "entry"),
                          NTuple{$nd, $d_J},
                          Tuple{$(flat_argtypes...)},
                          a_desc, b_desc, sp_meta, scale_d, $(d_args...))
        end
    end
    nothing
end

for (dt_d, dt_a, dt_b, k, has_trans) in _WGMMA_SP_VARIANTS
    for n in (dt_d === :s32 ? _WGMMA_SP_INT_NS : _WGMMA_NS)
        _wgmma_sp_register(dt_d, dt_a, dt_b, n, k, has_trans)
    end
end
