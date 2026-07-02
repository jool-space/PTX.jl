# `mma.sync.aligned` — fifth family migrated to tier-2 intrinsic lowering
# (DESIGN.md). Source of truth: PTX 9.2 §9.7.14.5.
#
# This family was already a table-driven generator on the asm tier, and
# stays one — but emitting `IntrinsicCall` instead of `@asmcall`. Unlike
# the hand-literal families (shfl/mbarrier/tma/tcgen05 verbs), the dtype
# cross-product here is far too large to spell as `nvvm"..."` literals
# (`kind::f8f6f4` alone is 5×5×2×2 = 100 structurally-identical forms), so
# the names are built from the dtype tuple. The conformance harness covers
# this family by a different mechanism than the src/ literal-scan: a
# registry-completeness assertion over every name the generator can
# produce, plus one selection probe per structural class (see
# test/host/conformance.jl, "mma generated-family coverage").
#
# Three data-type conventions, all hidden behind one notation surface
# (A/B and f16-accumulator C/D are packed `UInt32` fragments, matching the
# asm tier and pyptx):
#   - bf16 / tf32 / fp8 / kind::f8f6f4 inputs → intrinsic takes `i32` A/B;
#     the UInt32 fragments pass straight through.
#   - pure-f16 inputs (intrinsics named `.f16.f16` / `.f32.f32`) → intrinsic
#     takes `<2 x half>` A/B; each UInt32 is bitcast (free).
#   - C/D: f32 accumulator → `f32` (Float32 passes through); f16
#     accumulator → `<2 x half>`, bitcast to/from the packed UInt32.
#
# Layout: only `.row.col` exists (the sole layout modern m16n8k* shapes
# support — ptxas rejects the others, and the registry has no intrinsic
# for them). The asm tier registered all four layA×layB combos; three of
# every four were dead methods that emitted ptxas-rejected asm. The
# migration registers `.row.col` only, so a bogus-layout call is now a
# clean MethodError instead of a deferred ptxas failure.
#
# `kind::f8f6f4` (and the fp8 paths) are consumer-Blackwell (sm_120a+,
# the GB10 sub-byte FP accelerator — note the *opposite* gating from
# tcgen05, which is datacenter-only). ISel enforces the floor.

# Packed-UInt32 ⇄ <2 x half> bitcasts (free; the f16 fragment convention).
@inline _mma_u32_v2h(x::UInt32) = Base.llvmcall(
    ("""define <2 x half> @e(i32 %0) #0 {
          %v = bitcast i32 %0 to <2 x half>
          ret <2 x half> %v
        }
        attributes #0 = { alwaysinline }""", "e"),
    NTuple{2, VecElement{Float16}}, Tuple{UInt32}, x)
@inline _mma_v2h_u32(v::NTuple{2, VecElement{Float16}}) = Base.llvmcall(
    ("""define i32 @e(<2 x half> %0) #0 {
          %v = bitcast <2 x half> %0 to i32
          ret i32 %v
        }
        attributes #0 = { alwaysinline }""", "e"),
    UInt32, Tuple{NTuple{2, VecElement{Float16}}}, v)

# (shape, ab_dtype, cd_dtype) → (n_a_regs, n_b_regs, n_cd_regs) per lane.
# A/B fragments pack two 16-bit (or four 8-bit) values per UInt32. For
# `kind::f8f6f4` the per-lane count is fixed by the 8-bit container width
# (PTX 9.2 §9.7.14.5 line 1482) regardless of which sub-byte FP sits inside.
const MMA_SYNC_FRAGS = Dict{Tuple{Symbol, Symbol, Symbol}, NTuple{3, Int}}(
    (:m16n8k16, :bf16, :f32) => (4, 2, 4),
    (:m16n8k8,  :bf16, :f32) => (2, 1, 4),
    (:m16n8k16, :f16,  :f32) => (4, 2, 4),
    (:m16n8k8,  :f16,  :f32) => (2, 1, 4),
    (:m16n8k16, :f16,  :f16) => (4, 2, 2),
    (:m16n8k8,  :f16,  :f16) => (2, 1, 2),
    (:m16n8k8,  :tf32, :f32) => (4, 2, 4),
    (:m16n8k4,  :tf32, :f32) => (2, 1, 4),
    (:m16n8k16, :e4m3, :f32) => (2, 1, 4),
    (:m16n8k16, :e5m2, :f32) => (2, 1, 4),
    (:m16n8k16, :e3m2, :f32) => (2, 1, 4),
    (:m16n8k16, :e2m3, :f32) => (2, 1, 4),
    (:m16n8k16, :e2m1, :f32) => (2, 1, 4),
    (:m16n8k16, :e4m3, :f16) => (2, 1, 2),
    (:m16n8k16, :e5m2, :f16) => (2, 1, 2),
    (:m16n8k16, :e3m2, :f16) => (2, 1, 2),
    (:m16n8k16, :e2m3, :f16) => (2, 1, 2),
    (:m16n8k16, :e2m1, :f16) => (2, 1, 2),
    (:m16n8k32, :e4m3, :f32) => (4, 2, 4),
    (:m16n8k32, :e5m2, :f32) => (4, 2, 4),
    (:m16n8k32, :e3m2, :f32) => (4, 2, 4),
    (:m16n8k32, :e2m3, :f32) => (4, 2, 4),
    (:m16n8k32, :e2m1, :f32) => (4, 2, 4),
    (:m16n8k32, :e4m3, :f16) => (4, 2, 2),
    (:m16n8k32, :e5m2, :f16) => (4, 2, 2),
    (:m16n8k32, :e3m2, :f16) => (4, 2, 2),
    (:m16n8k32, :e2m3, :f16) => (4, 2, 2),
    (:m16n8k32, :e2m1, :f16) => (4, 2, 2),
)

# The intrinsic-name irregularity (LLVM IntrinsicsNVVM.td): bf16/tf32 carry
# only the input dtype; pure-f16 forms are named by accumulator
# (`.f16.f16` / `.f32.f32`); fp8 and kind forms carry the full dtype quad.
function _mma_intrinsic_name(shape::Symbol, a_ty::Symbol, b_ty::Symbol,
                             c_ty::Symbol, kind::Union{Symbol, Nothing})
    base = "mma.$shape.row.col"
    kind !== nothing && return "$base.kind.$kind.$c_ty.$a_ty.$b_ty.$c_ty"
    a_ty === :bf16 && return "$base.bf16"
    a_ty === :tf32 && return "$base.tf32"
    a_ty === :f16  && return c_ty === :f16 ? "$base.f16.f16" : "$base.f32.f32"
    return "$base.$c_ty.$a_ty.$b_ty.$c_ty"   # fp8 (e4m3/e5m2)
end

# Every name the generator routed to tier 2 — for the conformance
# completeness assertion (this is what protects the family in place of the
# literal scan). Forms with no intrinsic fall back to the asm tier and are
# listed in MMA_ASM_FORMS instead.
const MMA_INTRINSIC_NAMES = String[]
const MMA_ASM_FORMS = NTuple{5, Symbol}[]   # (shape, a, b, c, kind|:none)

function _mma_register(shape::Symbol, d_ty::Symbol, a_ty::Symbol,
                       b_ty::Symbol, c_ty::Symbol;
                       kind::Union{Symbol, Nothing} = nothing)
    haskey(MMA_SYNC_FRAGS, (shape, a_ty, c_ty)) || return nothing
    n_a, n_b, n_cd = MMA_SYNC_FRAGS[(shape, a_ty, c_ty)]

    mods_kind = kind === nothing ? () : (Symbol("kind::", kind),)
    mods = (:sync, :aligned, mods_kind...,
            shape, :row, :col, d_ty, a_ty, b_ty, c_ty)
    cd_J = c_ty === :f32 ? :Float32 : :UInt32

    name = _mma_intrinsic_name(shape, a_ty, b_ty, c_ty, kind)
    full = "llvm.nvvm." * name
    if !NVVM.isintrinsic(full)
        # No intrinsic at 22.1.7 (e.g. kind::f8f6f4 at m16n8k16) — asm tier.
        push!(MMA_ASM_FORMS, (shape, a_ty, b_ty, c_ty,
                              kind === nothing ? :none : kind))
        return _mma_register_asm(mods, shape, a_ty, b_ty, c_ty, kind,
                                 n_a, n_b, n_cd, cd_J)
    end
    push!(MMA_INTRINSIC_NAMES, full)
    call = NVVM.IntrinsicCall{Symbol(full)}()

    ab_vec = a_ty === :f16          # f16 inputs → <2 x half> A/B
    cd_vec = c_ty === :f16          # f16 accumulator → <2 x half> C/D

    a_in = [ab_vec ? :(_mma_u32_v2h(a[$i])) : :(a[$i]) for i in 1:n_a]
    b_in = [ab_vec ? :(_mma_u32_v2h(b[$i])) : :(b[$i]) for i in 1:n_b]
    c_in = [cd_vec ? :(_mma_u32_v2h(c[$i])) : :(c[$i]) for i in 1:n_cd]
    repack = cd_vec ? :(ntuple(i -> _mma_v2h_u32(d[i]), Val($n_cd))) : :d

    @eval function (::Operation{:mma, $mods})(
            a::NTuple{$n_a, UInt32},
            b::NTuple{$n_b, UInt32},
            c::NTuple{$n_cd, $cd_J})
        Base.@inline
        d = $call($(a_in...), $(b_in...), $(c_in...))
        $repack
    end
    nothing
end

# Asm-tier residue: forms LLVM 22.1.7 has no intrinsic for. Identical body
# to the pre-migration generator (UInt32 fragments, =f/f/r constraints).
function _mma_register_asm(mods, shape, a_ty, b_ty, c_ty, kind,
                           n_a, n_b, n_cd, cd_J)
    asm_kind = kind === nothing ? "" : "kind::$kind."
    cd_let = c_ty === :f32 ? "f" : "r"
    slots(off, n) = "{" * join(("\$$i" for i in off:off+n-1), ", ") * "}"
    asm = "mma.sync.aligned.$asm_kind$shape.row.col.$c_ty.$a_ty.$b_ty.$c_ty " *
          "$(slots(0, n_cd)), $(slots(n_cd, n_a)), " *
          "$(slots(n_cd + n_a, n_b)), $(slots(n_cd + n_a + n_b, n_cd));"
    constraints = join(vcat(fill("=$cd_let", n_cd), fill("r", n_a + n_b),
                            fill(cd_let, n_cd), ["~{memory}"]), ",")
    flat = vcat(fill(:UInt32, n_a + n_b), fill(cd_J, n_cd))
    # mma.sync is warp-collective — emitted with a `convergent` call-site
    # attribute, same reasoning as wgmma (see inst.jl, "convergent inline
    # asm"). @asmcall can't attach it.
    cdT = c_ty === :f32 ? Float32 : UInt32
    flat_types = vcat(fill(UInt32, n_a + n_b), fill(cdT, n_cd))
    ir = convergent_asm_ir(asm, constraints, NTuple{n_cd, cdT}, flat_types)
    a_args = [:(a[$i]) for i in 1:n_a]
    b_args = [:(b[$i]) for i in 1:n_b]
    c_args = [:(c[$i]) for i in 1:n_cd]
    @eval function (::Operation{:mma, $mods})(
            a::NTuple{$n_a, UInt32}, b::NTuple{$n_b, UInt32},
            c::NTuple{$n_cd, $cd_J})
        Base.@inline
        Base.llvmcall(($ir, "entry"),
                      NTuple{$n_cd, $cd_J}, Tuple{$(flat...)},
                      $(a_args...), $(b_args...), $(c_args...))
    end
    nothing
end

# Adding a dtype/shape combo = one entry in MMA_SYNC_FRAGS plus one line
# here. Layout is `.row.col` only (see header).
for shape in (:m16n8k16, :m16n8k8), ab_ty in (:bf16, :f16)
    _mma_register(shape, :f32, ab_ty, ab_ty, :f32)
end
for shape in (:m16n8k16, :m16n8k8)
    _mma_register(shape, :f16, :f16, :f16, :f16)
end
for shape in (:m16n8k8, :m16n8k4)
    _mma_register(shape, :f32, :tf32, :tf32, :f32)
end
# FP8 (Ada+) — m16n8k32 (PTX 8.0+) and m16n8k16 (PTX 8.7+).
for shape in (:m16n8k16, :m16n8k32), ab_ty in (:e4m3, :e5m2), c_ty in (:f32, :f16)
    _mma_register(shape, c_ty, ab_ty, ab_ty, c_ty)
end
# `kind::f8f6f4` (consumer-Blackwell sm_120a+) — mixed A/B legal because all
# five sub-byte FP types share the 8-bit container.
let f8f6f4 = (:e4m3, :e5m2, :e3m2, :e2m3, :e2m1)
    for shape in (:m16n8k16, :m16n8k32), a_ty in f8f6f4, b_ty in f8f6f4,
        c_ty in (:f32, :f16)
        _mma_register(shape, c_ty, a_ty, b_ty, c_ty; kind = :f8f6f4)
    end
end
