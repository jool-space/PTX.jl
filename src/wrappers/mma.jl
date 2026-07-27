# `mma.sync.aligned` — primarily tier-2 intrinsic lowering.
# Source of truth: PTX 9.3 §9.7.15.5.
#
# This family was already a table-driven generator on the asm tier, and
# stays one — normally emitting `IntrinsicCall` instead of `@asmcall`. Unlike
# the hand-literal families (shfl/mbarrier/tma/tcgen05 verbs), the dtype
# cross-product here is far too large to spell as `nvvm"..."` literals
# (`kind::f8f6f4` alone is 5×5×2×2 = 100 structurally-identical forms), so
# the names are built from the dtype tuple. The conformance harness covers
# this family by a different mechanism than the src/ literal-scan: a
# registry-completeness assertion over every name the generator can
# produce, plus one selection probe per structural class (see
# test/host/conformance.jl, "mma generated-family coverage").
#
# Four data-type conventions, all hidden behind one notation surface
# (A/B and f16-accumulator C/D are packed `UInt32` fragments, matching the
# asm tier and pyptx):
#   - bf16 / tf32 / fp8 / kind::f8f6f4 inputs → intrinsic takes `i32` A/B;
#     the UInt32 fragments pass straight through.
#   - pure-f16 inputs (intrinsics named `.f16.f16` / `.f32.f32`) → intrinsic
#     takes `<2 x half>` A/B; each UInt32 is bitcast (free).
#   - C/D: f32 accumulator → `f32` (Float32 passes through); f16
#     accumulator → `<2 x half>`, bitcast to/from the packed UInt32.
#   - integer A/B fragments are packed `.b32` (`UInt32`); the `.s32`
#     accumulator and result are `Int32`.  The wrapper reinterprets the
#     intrinsic's canonical unsigned i32 return without changing bits.
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
    (:m8n8k4,   :f64,  :f64) => (1, 1, 2),
    (:m16n8k4,  :f64,  :f64) => (2, 1, 4),
    (:m16n8k8,  :f64,  :f64) => (4, 2, 4),
    (:m16n8k16, :f64,  :f64) => (8, 4, 4),
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
    # PTX 9.3 §9.7.15.5.9–.11, Figures 84–96.  Signedness changes
    # element interpretation, not the packed .b32 fragment carrier.
    (:m16n8k16, :u8,   :s32) => (2, 1, 4),
    (:m16n8k16, :s8,   :s32) => (2, 1, 4),
    (:m16n8k32, :u8,   :s32) => (4, 2, 4),
    (:m16n8k32, :s8,   :s32) => (4, 2, 4),
    (:m16n8k32, :u4,   :s32) => (2, 1, 4),
    (:m16n8k32, :s4,   :s32) => (2, 1, 4),
    (:m16n8k64, :u4,   :s32) => (4, 2, 4),
    (:m16n8k64, :s4,   :s32) => (4, 2, 4),
    # PTX 9.3 §9.7.15.5.5, .12–.13, Figures 62–64 and 97–103.
    # Each .b32 carries 32 single-bit elements; C/D are signed accumulators.
    (:m8n8k128,  :b1, :s32) => (1, 1, 2),
    (:m16n8k128, :b1, :s32) => (2, 1, 4),
    (:m16n8k256, :b1, :s32) => (4, 2, 4),
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
    a_ty === :f64  && return "$base.f64"
    a_ty === :f16  && return c_ty === :f16 ? "$base.f16.f16" : "$base.f32.f32"
    return "$base.$c_ty.$a_ty.$b_ty.$c_ty"   # fp8 (e4m3/e5m2)
end

# Integer intrinsic names omit B's type when A and B match; `.satfinite`
# precedes the input type(s).  This is LLVM's IntrinsicsNVVM.td vocabulary,
# distinct from the complete PTX spelling exposed by Operation below.
function _mma_int_intrinsic_name(shape::Symbol, a_ty::Symbol, b_ty::Symbol,
                                 satfinite::Bool)
    base = "mma.$shape.row.col" * (satfinite ? ".satfinite" : "")
    base *= ".$a_ty"
    a_ty === b_ty ? base : base * ".$b_ty"
end

# Single-bit intrinsic names move the logical operation before `popc` and
# before the shape, unlike the complete PTX spelling where `.xor|and.popc`
# trails the type quartet.
_mma_b1_intrinsic_name(shape::Symbol, bitop::Symbol) =
    "mma.$bitop.popc.$shape.row.col.b1"

# Every name the generator routes to tier 2 lands in the wrapper registry
# (family :mma) — for the conformance completeness assertion (this is what
# protects the family in place of the literal scan). Forms with no intrinsic
# fall back to the asm tier and are recorded as :asm registry records instead.
function _mma_register(shape::Symbol, d_ty::Symbol, a_ty::Symbol,
                       b_ty::Symbol, c_ty::Symbol;
                       kind::Union{Symbol, Nothing} = nothing)
    haskey(MMA_SYNC_FRAGS, (shape, a_ty, c_ty)) || return nothing
    n_a, n_b, n_cd = MMA_SYNC_FRAGS[(shape, a_ty, c_ty)]

    mods_kind = kind === nothing ? () : (Symbol("kind::", kind),)
    mods = (:sync, :aligned, mods_kind...,
            shape, :row, :col, d_ty, a_ty, b_ty, c_ty)
    cd_J = c_ty === :f32 ? :Float32 : c_ty === :f64 ? :Float64 : :UInt32
    ab_J = a_ty === :f64 ? :Float64 : :UInt32     # f64 A/B are one f64/reg

    name = _mma_intrinsic_name(shape, a_ty, b_ty, c_ty, kind)
    full = "llvm.nvvm." * name
    if !NVVM.isintrinsic(full)
        # No intrinsic at 22.1.7 (e.g. kind::f8f6f4 at m16n8k16) — asm tier.
        return _mma_register_asm(mods, shape, a_ty, b_ty, c_ty, kind,
                                 n_a, n_b, n_cd, cd_J)
    end
    call = wrapper_intrinsic_call(:mma, :mma, mods, full)

    ab_vec = a_ty === :f16          # f16 inputs → <2 x half> A/B
    cd_vec = c_ty === :f16          # f16 accumulator → <2 x half> C/D

    a_in = [ab_vec ? :(_mma_u32_v2h(a[$i])) : :(a[$i]) for i in 1:n_a]
    b_in = [ab_vec ? :(_mma_u32_v2h(b[$i])) : :(b[$i]) for i in 1:n_b]
    c_in = [cd_vec ? :(_mma_u32_v2h(c[$i])) : :(c[$i]) for i in 1:n_cd]
    repack = cd_vec ? :(ntuple(i -> _mma_v2h_u32(d[i]), Val($n_cd))) : :d

    @eval function (::Operation{:mma, $mods})(
            a::NTuple{$n_a, $ab_J},
            b::NTuple{$n_b, $ab_J},
            c::NTuple{$n_cd, $cd_J})
        Base.@inline
        d = $call($(a_in...), $(b_in...), $(c_in...))
        $repack
    end
    nothing
end


# --- modern dense integer mma (PTX 7.0, sm_80+) ----------------------------
#
# PTX 9.3 §9.7.15.5.14 defines two independent products:
#   u8/s8: m16n8k16, m16n8k32
#   u4/s4: m16n8k32, m16n8k64
# Each permits every A×B signedness pair and optional `.satfinite`, for 32
# exact forms.  All use `.s32` accumulators/results and `.row.col`; the b1
# product is registered separately below because it has a distinct
# `.bitOp.popc` grammar and shape-dependent floors.
const MMA_INT_VARIANTS = (
    (:m16n8k16, :u8, :s8),
    (:m16n8k32, :u8, :s8),
    (:m16n8k32, :u4, :s4),
    (:m16n8k64, :u4, :s4),
)

function _mma_int_register(shape::Symbol, a_ty::Symbol, b_ty::Symbol,
                           satfinite::Bool)
    n_a, n_b, n_cd = MMA_SYNC_FRAGS[(shape, a_ty, :s32)]
    satmod = satfinite ? (:satfinite,) : ()
    mods = (:sync, :aligned, shape, :row, :col, satmod...,
            :s32, a_ty, b_ty, :s32)

    full = "llvm.nvvm." *
           _mma_int_intrinsic_name(shape, a_ty, b_ty, satfinite)
    call = wrapper_intrinsic_call(:mma, :mma, mods, full)

    a_in = [:(a[$i]) for i in 1:n_a]
    b_in = [:(b[$i]) for i in 1:n_b]
    c_in = [:(c[$i]) for i in 1:n_cd]
    @eval function (::Operation{:mma, $mods})(
            a::NTuple{$n_a, UInt32},
            b::NTuple{$n_b, UInt32},
            c::NTuple{$n_cd, Int32})
        Base.@inline
        d = $call($(a_in...), $(b_in...), $(c_in...))
        ntuple(i -> reinterpret(Int32, d[i]), Val($n_cd))
    end
    nothing
end

# Asm-tier residue: forms LLVM 22.1.7 has no intrinsic for. Identical body
# to the pre-migration generator (UInt32 fragments, =f/f/r constraints).
function _mma_register_asm(mods, shape, a_ty, b_ty, c_ty, kind,
                           n_a, n_b, n_cd, cd_J)
    register_wrapper!(:mma, :mma, mods, :asm)
    asm_kind = kind === nothing ? "" : "kind::$kind."
    cd_let = c_ty === :f32 ? "f" : c_ty === :f64 ? "d" : "r"
    ab_let = a_ty === :f64 ? "d" : "r"
    ab_J   = a_ty === :f64 ? :Float64 : :UInt32
    slots(off, n) = "{" * join(("\$$i" for i in off:off+n-1), ", ") * "}"
    asm = "mma.sync.aligned.$asm_kind$shape.row.col.$c_ty.$a_ty.$b_ty.$c_ty " *
          "$(slots(0, n_cd)), $(slots(n_cd, n_a)), " *
          "$(slots(n_cd + n_a, n_b)), $(slots(n_cd + n_a + n_b, n_cd));"
    constraints = join(vcat(fill("=$cd_let", n_cd), fill(ab_let, n_a + n_b),
                            fill(cd_let, n_cd), ["~{memory}"]), ",")
    flat = vcat(fill(ab_J, n_a + n_b), fill(cd_J, n_cd))
    # mma.sync is warp-collective — emitted with a `convergent` call-site
    # attribute, same reasoning as wgmma (see src/dsl/convergent_asm.jl,
    # "convergent inline asm"). @asmcall can't attach it.
    cdT = c_ty === :f32 ? Float32 : c_ty === :f64 ? Float64 : UInt32
    abT = a_ty === :f64 ? Float64 : UInt32
    flat_types = vcat(fill(abT, n_a + n_b), fill(cdT, n_cd))
    ir = convergent_asm_ir(asm, constraints, NTuple{n_cd, cdT}, flat_types)
    a_args = [:(a[$i]) for i in 1:n_a]
    b_args = [:(b[$i]) for i in 1:n_b]
    c_args = [:(c[$i]) for i in 1:n_cd]
    @eval function (::Operation{:mma, $mods})(
            a::NTuple{$n_a, $ab_J}, b::NTuple{$n_b, $ab_J},
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
# FP64 (DMMA, sm_80+) — m8n8k4 (PTX 7.0), m16n8k{4,8,16} (PTX 7.8). One f64
# per A/B register; f64 accumulators. Full-rate only on datacenter silicon
# (GA100/GH100/GB100 class); consumer parts execute it at a token rate.
for shape in (:m8n8k4, :m16n8k4, :m16n8k8, :m16n8k16)
    _mma_register(shape, :f64, :f64, :f64, :f64)
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

for (shape, u_ty, s_ty) in MMA_INT_VARIANTS,
        a_ty in (u_ty, s_ty), b_ty in (u_ty, s_ty),
        satfinite in (false, true)
    _mma_int_register(shape, a_ty, b_ty, satfinite)
end

# --- single-bit integer mma (PTX 7.0/7.1, sm_75/sm_80) --------------------
#
# PTX 9.3 §9.7.15.5.14 defines exactly three shapes × two logical
# operations.  XOR was introduced with b1 MMA in PTX 7.0; AND followed in
# PTX 7.1.  Only m8n8k128.xor reaches back to sm_75.  Every m16 shape and
# every AND form requires sm_80.  Target/version gating remains ISel/ptxas's
# responsibility; this boundary pins the operand and result ABI.
const MMA_B1_VARIANTS = (
    (:m8n8k128,  :xor),
    (:m8n8k128,  :and),
    (:m16n8k128, :xor),
    (:m16n8k128, :and),
    (:m16n8k256, :xor),
    (:m16n8k256, :and),
)

function _mma_b1_register_asm(shape::Symbol, bitop::Symbol,
                              n_a::Int, n_b::Int, n_cd::Int)
    slots(off, n) = "{" * join(("\$$i" for i in off:off+n-1), ", ") * "}"
    asm = "mma.sync.aligned.$shape.row.col.s32.b1.b1.s32.$bitop.popc " *
          "$(slots(0, n_cd)), $(slots(n_cd, n_a)), " *
          "$(slots(n_cd + n_a, n_b)), " *
          "$(slots(n_cd + n_a + n_b, n_cd));"
    # Early-clobber prevents an output from sharing a register with a packed
    # A/B source. No memory clobber: this is still pure per-warp arithmetic.
    constraints = join(vcat(fill("=&r", n_cd),
                            fill("r", n_a + n_b + n_cd)), ",")
    flat_types = vcat(fill(UInt32, n_a + n_b), fill(Int32, n_cd))
    ir = convergent_asm_ir(asm, constraints, NTuple{n_cd, Int32}, flat_types)
    a_in = [:(a[$i]) for i in 1:n_a]
    b_in = [:(b[$i]) for i in 1:n_b]
    c_in = [:(c[$i]) for i in 1:n_cd]
    mods = (:sync, :aligned, shape, :row, :col,
            :s32, :b1, :b1, :s32, bitop, :popc)
    register_wrapper!(:mma_b1, :mma, mods, :asm)
    @eval function (::Operation{:mma, $mods})(
            a::NTuple{$n_a, UInt32}, b::NTuple{$n_b, UInt32},
            c::NTuple{$n_cd, Int32})
        Base.@inline
        Base.llvmcall(($ir, "entry"), NTuple{$n_cd, Int32},
                      Tuple{$(flat_types...)},
                      $(a_in...), $(b_in...), $(c_in...))
    end
    nothing
end

function _mma_b1_register(shape::Symbol, bitop::Symbol)
    n_a, n_b, n_cd = MMA_SYNC_FRAGS[(shape, :b1, :s32)]
    # LLVM 22.1.7 carries the m8n8k128.xor intrinsic but its ISel predicate
    # incorrectly rejects the ISA-legal sm_75 floor (it first selects at
    # sm_80). Keep the public ISA floor with one typed convergent asm fallback;
    # the other five forms use their existing intrinsics.
    shape === :m8n8k128 && bitop === :xor &&
        return _mma_b1_register_asm(shape, bitop, n_a, n_b, n_cd)

    mods = (:sync, :aligned, shape, :row, :col,
            :s32, :b1, :b1, :s32, bitop, :popc)
    full = "llvm.nvvm." * _mma_b1_intrinsic_name(shape, bitop)
    call = wrapper_intrinsic_call(:mma_b1, :mma, mods, full)

    a_in = [:(a[$i]) for i in 1:n_a]
    b_in = [:(b[$i]) for i in 1:n_b]
    c_in = [:(c[$i]) for i in 1:n_cd]
    @eval function (::Operation{:mma, $mods})(
            a::NTuple{$n_a, UInt32},
            b::NTuple{$n_b, UInt32},
            c::NTuple{$n_cd, Int32})
        Base.@inline
        d = $call($(a_in...), $(b_in...), $(c_in...))
        ntuple(i -> reinterpret(Int32, d[i]), Val($n_cd))
    end
    nothing
end

for (shape, bitop) in MMA_B1_VARIANTS
    _mma_b1_register(shape, bitop)
end

# --- mma.sp — 2:4 structured-sparse mma (sm_80+, PTX 7.1+) -------------------
#
#   mma.sp.sync.aligned.<shape>.row.col.<d>.<a>.<b>.<c>  d, a, b, c, e, #sel
#
# A is stored compressed (PTX 9.3 §9.7.15.6 "Sparse matrix storage"): each
# 4-wide chunk of a row keeps its two non-zeros (tf32: 1:2, each 2-wide
# chunk keeps one), and the .b32 metadata operand `e` holds the 2-bit
# in-chunk position of every kept element. The sparsity selector — an
# immediate, passed as `Val(sel)` — names which thread (or thread pair) of
# each group of four contributes metadata; its legal range is shape/dtype
# specific and enforced by the intrinsic's immarg range at compile time
# (k16 16-bit & k8 tf32: one thread, 0..3; k32 16-bit & k16 tf32: a pair,
# 0..1; k64 fp8: all threads, must be 0). Integer products use the same
# exact boundary: k32 u8/s8 and k64 u4/s4 select 0..1; k64 u8/s8 and k128
# u4/s4 require 0.
#
# Surface (fragments per lane in MMA_SP_FRAGS):
#
#   ptx"mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32"(
#       a::NTuple{4, UInt32}, b::NTuple{4, UInt32}, c::NTuple{4, Float32},
#       e::UInt32, Val(0))
#
# Same intrinsic-name irregularity as dense: bf16/tf32 carry the input
# dtype, pure-f16 forms are named by accumulator, fp8 carries the quad.
# All registered forms have tier-2 intrinsics at the pinned backend; a
# form that loses its intrinsic on a backend bump lands in the registry as
# a :missing record and fails the conformance count instead of
# silently vanishing. `sp::ordered_metadata` (PTX 8.5+, sm_80+) is exposed for
# the 12 floating ABIs and all 32 integer products (shape/type/saturation) also
# exposed by the base form. The ISA recommends it because the unordered form
# may have substantially reduced performance on some targets. Its metadata
# nibbles must encode retained indices in increasing order; any other order has
# undefined behavior. Ordered forms and both integer variants use exact
# `Val{sel}` methods so the ISA's shape-dependent selector domain is a dispatch
# boundary, not merely an LLVM immarg diagnostic.

# (shape, ab_dtype, cd_dtype) → (n_a_regs, n_b_regs, n_cd_regs) per lane.
const MMA_SP_FRAGS = Dict{Tuple{Symbol, Symbol, Symbol}, NTuple{3, Int}}(
    (:m16n8k16, :f16,  :f16) => (2, 2, 2),
    (:m16n8k16, :f16,  :f32) => (2, 2, 4),
    (:m16n8k16, :bf16, :f32) => (2, 2, 4),
    (:m16n8k32, :f16,  :f16) => (4, 4, 2),
    (:m16n8k32, :f16,  :f32) => (4, 4, 4),
    (:m16n8k32, :bf16, :f32) => (4, 4, 4),
    (:m16n8k8,  :tf32, :f32) => (2, 2, 4),
    (:m16n8k16, :tf32, :f32) => (4, 4, 4),
    (:m16n8k64, :e4m3, :f32) => (4, 4, 4),
    (:m16n8k64, :e5m2, :f32) => (4, 4, 4),
    (:m16n8k32, :u8,   :s32) => (2, 2, 4),
    (:m16n8k32, :s8,   :s32) => (2, 2, 4),
    (:m16n8k64, :u8,   :s32) => (4, 4, 4),
    (:m16n8k64, :s8,   :s32) => (4, 4, 4),
    (:m16n8k64, :u4,   :s32) => (2, 2, 4),
    (:m16n8k64, :s4,   :s32) => (2, 2, 4),
    (:m16n8k128, :u4,  :s32) => (4, 4, 4),
    (:m16n8k128, :s4,  :s32) => (4, 4, 4),
)

# PTX 9.3 §9.7.15.6 integer products. Every row expands to four A/B
# signedness pairs, ordinary and `.satfinite`, then both metadata variants.
const MMA_SP_INT_VARIANTS = (
    (:m16n8k32,  :u8, :s8),
    (:m16n8k64,  :u8, :s8),
    (:m16n8k64,  :u4, :s4),
    (:m16n8k128, :u4, :s4),
)

function _mma_sp_intrinsic_name(shape::Symbol, a_ty::Symbol, b_ty::Symbol,
                                c_ty::Symbol; ordered::Bool = false)
    prefix = ordered ? "mma.sp.ordered.metadata" : "mma.sp"
    base = "$prefix.$shape.row.col"
    a_ty === :bf16 && return "$base.bf16"
    a_ty === :tf32 && return "$base.tf32"
    a_ty === :f16  && return c_ty === :f16 ? "$base.f16.f16" : "$base.f32.f32"
    return "$base.$c_ty.$a_ty.$b_ty.$c_ty"   # fp8 (e4m3/e5m2, mixed legal)
end

function _mma_sp_int_intrinsic_name(shape::Symbol, a_ty::Symbol,
                                    b_ty::Symbol, satfinite::Bool;
                                    ordered::Bool = false)
    prefix = ordered ? "mma.sp.ordered.metadata" : "mma.sp"
    base = "$prefix.$shape.row.col" * (satfinite ? ".satfinite" : "")
    base *= ".$a_ty"
    a_ty === b_ty ? base : base * ".$b_ty"
end

# PTX 9.3 §9.7.15.6: the selector identifies one thread, one thread pair, or
# the whole four-thread group depending on the sparse A fragment layout.
function _mma_sp_selectors(shape::Symbol, a_ty::Symbol)
    shape === :m16n8k128 && return (0,)
    (shape === :m16n8k64 && a_ty in (:u8, :s8, :e4m3, :e5m2)) &&
        return (0,)
    ((shape === :m16n8k64 && a_ty in (:u4, :s4)) ||
     shape === :m16n8k32 ||
     (shape === :m16n8k16 && a_ty === :tf32)) && return (0, 1)
    return (0, 1, 2, 3)
end

function _mma_sp_register(shape::Symbol, d_ty::Symbol, a_ty::Symbol,
                          b_ty::Symbol, c_ty::Symbol; ordered::Bool = false,
                          satfinite::Bool = false)
    satfinite && c_ty !== :s32 && return nothing
    haskey(MMA_SP_FRAGS, (shape, a_ty, c_ty)) || return nothing
    n_a, n_b, n_cd = MMA_SP_FRAGS[(shape, a_ty, c_ty)]
    integer = c_ty === :s32

    variant = ordered ? Symbol("sp::ordered_metadata") : :sp
    satmod = satfinite ? (:satfinite,) : ()
    mods = (variant, :sync, :aligned, shape, :row, :col, satmod...,
            d_ty, a_ty, b_ty, c_ty)
    cd_J = c_ty === :f32 ? :Float32 : c_ty === :s32 ? :Int32 : :UInt32

    intrinsic_name = integer ?
        _mma_sp_int_intrinsic_name(shape, a_ty, b_ty, satfinite; ordered) :
        _mma_sp_intrinsic_name(shape, a_ty, b_ty, c_ty; ordered)
    full = "llvm.nvvm." * intrinsic_name
    # No asm fallback here (every registered form has an intrinsic at the
    # pinned backend) — a miss becomes a :missing registry record and the
    # conformance counts fail loudly. The integer subset needs no separate
    # list: integer records are exactly those whose mods carry :s32.
    family = ordered ? :mma_sp_ordered : :mma_sp
    call = wrapper_intrinsic_call(family, :mma, mods, full; missing_ok = true)
    call === nothing && return nothing

    ab_vec = a_ty === :f16
    cd_vec = c_ty === :f16

    a_in = [ab_vec ? :(_mma_u32_v2h(a[$i])) : :(a[$i]) for i in 1:n_a]
    b_in = [ab_vec ? :(_mma_u32_v2h(b[$i])) : :(b[$i]) for i in 1:n_b]
    c_in = [cd_vec ? :(_mma_u32_v2h(c[$i])) : :(c[$i]) for i in 1:n_cd]
    repack = cd_vec ? :(ntuple(i -> _mma_v2h_u32(d[i]), Val($n_cd))) :
             integer ? :(ntuple(i -> reinterpret(Int32, d[i]), Val($n_cd))) : :d

    strict_selector = ordered || integer
    selectors = strict_selector ? _mma_sp_selectors(shape, a_ty) : (nothing,)
    for selector in selectors
        sel_type = strict_selector ? :(Val{$selector}) : :Val
        @eval function (::Operation{:mma, $mods})(
                a::NTuple{$n_a, UInt32},
                b::NTuple{$n_b, UInt32},
                c::NTuple{$n_cd, $cd_J},
                e::UInt32, sel::$sel_type)
            Base.@inline
            d = $call($(a_in...), $(b_in...), $(c_in...), e, sel)
            $repack
        end
    end
    nothing
end

for shape in (:m16n8k16, :m16n8k32)
    _mma_sp_register(shape, :f32, :f16, :f16, :f32)
    _mma_sp_register(shape, :f16, :f16, :f16, :f16)
    _mma_sp_register(shape, :f32, :bf16, :bf16, :f32)
    _mma_sp_register(shape, :f32, :f16, :f16, :f32; ordered = true)
    _mma_sp_register(shape, :f16, :f16, :f16, :f16; ordered = true)
    _mma_sp_register(shape, :f32, :bf16, :bf16, :f32; ordered = true)
end
for shape in (:m16n8k8, :m16n8k16)
    _mma_sp_register(shape, :f32, :tf32, :tf32, :f32)
    _mma_sp_register(shape, :f32, :tf32, :tf32, :f32; ordered = true)
end
# Sparse FP8 (PTX 8.4+) — mixed e4m3/e5m2 A/B is legal.
for a_ty in (:e4m3, :e5m2), b_ty in (:e4m3, :e5m2)
    _mma_sp_register(:m16n8k64, :f32, a_ty, b_ty, :f32)
    _mma_sp_register(:m16n8k64, :f32, a_ty, b_ty, :f32; ordered = true)
end

# Classic warp-level integer sparse MMA (PTX 7.1, sm_80). This is distinct
# from tcgen05.mma.sp and deliberately shares none of its descriptor surface.
for (shape, u_ty, s_ty) in MMA_SP_INT_VARIANTS,
        a_ty in (u_ty, s_ty), b_ty in (u_ty, s_ty),
        satfinite in (false, true), ordered in (false, true)
    _mma_sp_register(shape, :s32, a_ty, b_ty, :s32;
                     satfinite, ordered)
end
