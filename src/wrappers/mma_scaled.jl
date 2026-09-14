# Block-scaled `mma.sync.aligned` (Blackwell microscaling) — migrated to
# tier-2 intrinsic lowering alongside the dense mma family. Three kinds —
# `mxf4`, `mxf4nvf4`, `mxf8f6f4` — each computing
# `D = (A * scale_A) * (B * scale_B) + C`. Source of truth: PTX 9.2 §9.7.14.3.
#
# Operand order on the notation surface (unchanged): d, a, b, c, scale-a,
# {byte-id-a, thread-id-a}, scale-b, {byte-id-b, thread-id-b}. A/B/scale
# operands are packed UInt32; the byte/thread ids are UInt16. The
# intrinsic takes the same operands flat (i32×A/B, f32/i32×C, then sa, bid,
# tid, sb, bid, tid), so no repacking — A/B/C are i32/f32 here (no f16
# accumulator in the block-scale surface).
#
# Intrinsic name irregularity: the registry name carries scale_vec as
# `.scale.<n>x.` (e.g. `mma.block.scale.m16n8k32.row.col.mxf8f6f4.scale.1x.
# f32...`) and ISel renders the PTX qualifier order shape-before-kind
# (`mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::
# 1X.f32...`) where the asm tier put kind first. ptxas accepts both.
#
# m16n8k32 reuses the kind::f8f6f4 counts; m16n8k64 with .e2m1 is unique to
# the scaled mxf4* path (§9.7.14.5.11).
const MMA_SCALED_FRAGS = Dict{Tuple{Symbol, Symbol, Symbol}, NTuple{3, Int}}(
    (:m16n8k64, :e2m1, :f32) => (4, 2, 4),
    (:m16n8k32, :e4m3, :f32) => (4, 2, 4),
    (:m16n8k32, :e5m2, :f32) => (4, 2, 4),
    (:m16n8k32, :e3m2, :f32) => (4, 2, 4),
    (:m16n8k32, :e2m3, :f32) => (4, 2, 4),
    (:m16n8k32, :e2m1, :f32) => (4, 2, 4),
)

# scale_vec::NX → the registry's `.scale.nx.` infix.
const _MMA_SCALE_VEC_INFIX = Dict(Symbol("1X") => "scale.1x",
                                  Symbol("2X") => "scale.2x",
                                  Symbol("4X") => "scale.4x")

function _mma_scaled_register(kind::Symbol, scale_vec::Symbol,
                              shape::Symbol, layA::Symbol, layB::Symbol,
                              d_ty::Symbol, a_ty::Symbol, b_ty::Symbol,
                              c_ty::Symbol, s_ty::Symbol)
    haskey(MMA_SCALED_FRAGS, (shape, a_ty, c_ty)) || return nothing
    n_a, n_b, n_cd = MMA_SCALED_FRAGS[(shape, a_ty, c_ty)]

    mods = (:sync, :aligned, Symbol("kind::", kind), :block_scale,
            Symbol("scale_vec::", scale_vec),
            shape, layA, layB, d_ty, a_ty, b_ty, c_ty, s_ty)

    infix = _MMA_SCALE_VEC_INFIX[scale_vec]
    name = "mma.block.scale.$shape.$layA.$layB.$kind.$infix." *
           "$c_ty.$a_ty.$b_ty.$c_ty.$s_ty"
    full = "llvm.nvvm." * name

    call = wrapper_intrinsic_call(:mma_scaled, :mma, mods, full)
    cd_J = c_ty === :f32 ? :Float32 : :UInt32

    a_in = [:(a[$i]) for i in 1:n_a]
    b_in = [:(b[$i]) for i in 1:n_b]
    c_in = [:(c[$i]) for i in 1:n_cd]

    @eval function (::Operation{:mma, $mods})(
            a::NTuple{$n_a, UInt32}, b::NTuple{$n_b, UInt32},
            c::NTuple{$n_cd, $cd_J},
            sa::UInt32, bida::UInt16, tida::UInt16,
            sb::UInt32, bidb::UInt16, tidb::UInt16)
        Base.@inline
        $call($(a_in...), $(b_in...), $(c_in...),
              sa, bida, tida, sb, bidb, tidb)
    end
    nothing
end

# Per Table 36 of PTX 9.2 §9.7.14.3. Layout `.row.col` only.
_mma_scaled_register(:mxf4, Symbol("2X"), :m16n8k64, :row, :col,
                     :f32, :e2m1, :e2m1, :f32, :ue8m0)

_mma_scaled_register(:mxf4nvf4, Symbol("2X"), :m16n8k64, :row, :col,
                     :f32, :e2m1, :e2m1, :f32, :ue8m0)
_mma_scaled_register(:mxf4nvf4, Symbol("4X"), :m16n8k64, :row, :col,
                     :f32, :e2m1, :e2m1, :f32, :ue8m0)
_mma_scaled_register(:mxf4nvf4, Symbol("4X"), :m16n8k64, :row, :col,
                     :f32, :e2m1, :e2m1, :f32, :ue4m3)

let f8f6f4 = (:e4m3, :e5m2, :e3m2, :e2m3, :e2m1)
    for a_ty in f8f6f4, b_ty in f8f6f4
        _mma_scaled_register(:mxf8f6f4, Symbol("1X"), :m16n8k32, :row, :col,
                             :f32, a_ty, b_ty, :f32, :ue8m0)
    end
end
