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
# Residue: `mxf4nvf4` with `scale_vec::4X` and a `ue8m0` scale type has no
# intrinsic at 22.1.7 (only the `ue4m3` 4X and `ue8m0` 2X nvf4 forms are
# registered) — it falls back to the asm tier, same as the dense
# kind::f8f6f4 m16n8k16 forms.

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

const MMA_SCALED_INTRINSIC_NAMES = String[]
const MMA_SCALED_ASM_FORMS = NTuple{6, Symbol}[]   # (kind,svec,shape,a,b,stype)

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

    if !NVVM.isintrinsic(full)
        # Idempotent bookkeeping — see _mma_register.
        form = (kind, scale_vec, shape, a_ty, b_ty, s_ty)
        form in MMA_SCALED_ASM_FORMS || push!(MMA_SCALED_ASM_FORMS, form)
        return _mma_scaled_register_asm(mods, kind, scale_vec, shape,
                                        layA, layB, a_ty, b_ty, c_ty, s_ty,
                                        n_a, n_b, n_cd)
    end
    full in MMA_SCALED_INTRINSIC_NAMES || push!(MMA_SCALED_INTRINSIC_NAMES, full)
    call = NVVM.IntrinsicCall{Symbol(full)}()
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

# Asm-tier residue (forms with no intrinsic at 22.1.7). Identical body to
# the pre-migration generator.
function _mma_scaled_register_asm(mods, kind, scale_vec, shape, layA, layB,
                                  a_ty, b_ty, c_ty, s_ty, n_a, n_b, n_cd)
    slots(off, n) = "{" * join(("\$$i" for i in off:off+n-1), ", ") * "}"
    base = 2 * n_cd + n_a + n_b
    asm = "mma.sync.aligned.kind::$kind.block_scale.scale_vec::$scale_vec." *
          "$shape.$layA.$layB.$c_ty.$a_ty.$b_ty.$c_ty.$s_ty " *
          "$(slots(0, n_cd)), $(slots(n_cd, n_a)), " *
          "$(slots(n_cd + n_a, n_b)), $(slots(n_cd + n_a + n_b, n_cd)), " *
          "\$$(base + 0), {\$$(base + 1), \$$(base + 2)}, " *
          "\$$(base + 3), {\$$(base + 4), \$$(base + 5)};"
    cd_let = c_ty === :f32 ? "f" : "r"
    cd_J   = c_ty === :f32 ? :Float32 : :UInt32
    constraints = join(vcat(fill("=$cd_let", n_cd), fill("r", n_a + n_b),
                            fill(cd_let, n_cd),
                            ["r", "h", "h", "r", "h", "h"], ["~{memory}"]), ",")
    flat = vcat(fill(:UInt32, n_a + n_b), fill(cd_J, n_cd),
                [:UInt32, :UInt16, :UInt16, :UInt32, :UInt16, :UInt16])
    # mma.sync is warp-collective — emitted via convergent_asm_ir like the
    # dense fallbacks so the call carries `convergent nomerge` (aeff3ee
    # converted wgmma + dense mma but missed this file; @asmcall cannot
    # attach call-site attributes).
    cdT = c_ty === :f32 ? Float32 : UInt32
    flat_types = vcat(fill(UInt32, n_a + n_b), fill(cdT, n_cd),
                      [UInt32, UInt16, UInt16, UInt32, UInt16, UInt16])
    ir = convergent_asm_ir(asm, constraints, NTuple{n_cd, cdT}, flat_types)
    a_args = [:(a[$i]) for i in 1:n_a]
    b_args = [:(b[$i]) for i in 1:n_b]
    c_args = [:(c[$i]) for i in 1:n_cd]
    @eval function (::Operation{:mma, $mods})(
            a::NTuple{$n_a, UInt32}, b::NTuple{$n_b, UInt32},
            c::NTuple{$n_cd, $cd_J},
            sa::UInt32, bida::UInt16, tida::UInt16,
            sb::UInt32, bidb::UInt16, tidb::UInt16)
        Base.@inline
        Base.llvmcall(($ir, "entry"),
                      NTuple{$n_cd, $cd_J}, Tuple{$(flat...)},
                      $(a_args...), $(b_args...), $(c_args...),
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
                     :f32, :e2m1, :e2m1, :f32, :ue8m0)   # asm residue (no intrinsic)
_mma_scaled_register(:mxf4nvf4, Symbol("4X"), :m16n8k64, :row, :col,
                     :f32, :e2m1, :e2m1, :f32, :ue4m3)

let f8f6f4 = (:e4m3, :e5m2, :e3m2, :e2m3, :e2m1)
    for a_ty in f8f6f4, b_ty in f8f6f4
        _mma_scaled_register(:mxf8f6f4, Symbol("1X"), :m16n8k32, :row, :col,
                             :f32, a_ty, b_ty, :f32, :ue8m0)
    end
end
