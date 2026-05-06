# Block-scaled `mma.sync.aligned` (Blackwell). Three kinds — `mxf4`,
# `mxf4nvf4`, `mxf8f6f4` — each performing `D = (A * scale_A) * (B * scale_B)
# + C`. Operand order: d, a, b, c, scale-a-data, {byte-id-a, thread-id-a},
# scale-b-data, {byte-id-b, thread-id-b}. Source of truth: PTX 9.2 §9.7.14.3.

# m16n8k32 entries reuse the `kind::f8f6f4` counts; m16n8k64 with .e2m1 is
# unique to the scaled mxf4* path (§9.7.14.5.11).
const MMA_SCALED_FRAGS = Dict{Tuple{Symbol, Symbol, Symbol}, NTuple{3, Int}}(
    (:m16n8k64, :e2m1, :f32) => (4, 2, 4),
    (:m16n8k32, :e4m3, :f32) => (4, 2, 4),
    (:m16n8k32, :e5m2, :f32) => (4, 2, 4),
    (:m16n8k32, :e3m2, :f32) => (4, 2, 4),
    (:m16n8k32, :e2m3, :f32) => (4, 2, 4),
    (:m16n8k32, :e2m1, :f32) => (4, 2, 4),
)

function _mma_scaled_register(kind::Symbol, scale_vec::Symbol,
                              shape::Symbol, layA::Symbol, layB::Symbol,
                              d_ty::Symbol, a_ty::Symbol, b_ty::Symbol,
                              c_ty::Symbol, s_ty::Symbol)
    haskey(MMA_SCALED_FRAGS, (shape, a_ty, c_ty)) || return nothing
    n_a, n_b, n_cd = MMA_SCALED_FRAGS[(shape, a_ty, c_ty)]

    mods = (:sync, :aligned,
            Symbol("kind::", kind),
            :block_scale,
            Symbol("scale_vec::", scale_vec),
            shape, layA, layB, d_ty, a_ty, b_ty, c_ty, s_ty)

    # Slot order: D outputs, A, B, C, scale-a-data, {byte-id-a, thread-id-a},
    # scale-b-data, {byte-id-b, thread-id-b}.
    slots(off, n) = "{" * join(("\$$i" for i in off:off+n-1), ", ") * "}"
    d_slots = slots(0,                       n_cd)
    a_slots = slots(n_cd,                    n_a)
    b_slots = slots(n_cd + n_a,              n_b)
    c_slots = slots(n_cd + n_a + n_b,        n_cd)
    base    = 2 * n_cd + n_a + n_b
    sa_slot      = "\$" * string(base + 0)
    bida_slot    = "\$" * string(base + 1)
    tida_slot    = "\$" * string(base + 2)
    sb_slot      = "\$" * string(base + 3)
    bidb_slot    = "\$" * string(base + 4)
    tidb_slot    = "\$" * string(base + 5)

    asm = "mma.sync.aligned.kind::$kind.block_scale.scale_vec::$scale_vec." *
          "$shape.$layA.$layB.$d_ty.$a_ty.$b_ty.$c_ty.$s_ty " *
          "$d_slots, $a_slots, $b_slots, $c_slots, " *
          "$sa_slot, {$bida_slot, $tida_slot}, " *
          "$sb_slot, {$bidb_slot, $tidb_slot};"

    cd_let = c_ty === :f32 ? "f" : "r"
    cd_J   = c_ty === :f32 ? :Float32 : :UInt32

    constraints = join(vcat(
        fill("=$cd_let", n_cd),
        fill("r",        n_a + n_b),
        fill(cd_let,     n_cd),
        ["r", "h", "h", "r", "h", "h"],
        ["~{memory}"]
    ), ",")

    flat_argtypes = vcat(
        fill(:UInt32, n_a + n_b),
        fill(cd_J,    n_cd),
        [:UInt32, :UInt16, :UInt16, :UInt32, :UInt16, :UInt16],
    )

    a_args = [:(a[$i]) for i in 1:n_a]
    b_args = [:(b[$i]) for i in 1:n_b]
    c_args = [:(c[$i]) for i in 1:n_cd]

    @eval function (::Operation{:mma, $mods})(
            a::NTuple{$n_a, UInt32},
            b::NTuple{$n_b, UInt32},
            c::NTuple{$n_cd, $cd_J},
            sa::UInt32, bida::UInt16, tida::UInt16,
            sb::UInt32, bidb::UInt16, tidb::UInt16)
        Base.@inline
        @asmcall($asm, $constraints, true,
                 NTuple{$n_cd, $cd_J},
                 Tuple{$(flat_argtypes...)},
                 $(a_args...), $(b_args...), $(c_args...),
                 sa, bida, tida, sb, bidb, tidb)
    end
    nothing
end

# Per Table 36 of PTX 9.2 §9.7.14.3.
for layA in (:row, :col), layB in (:row, :col)
    _mma_scaled_register(:mxf4, Symbol("2X"), :m16n8k64, layA, layB,
                         :f32, :e2m1, :e2m1, :f32, :ue8m0)

    _mma_scaled_register(:mxf4nvf4, Symbol("2X"), :m16n8k64, layA, layB,
                         :f32, :e2m1, :e2m1, :f32, :ue8m0)
    _mma_scaled_register(:mxf4nvf4, Symbol("4X"), :m16n8k64, layA, layB,
                         :f32, :e2m1, :e2m1, :f32, :ue8m0)
    _mma_scaled_register(:mxf4nvf4, Symbol("4X"), :m16n8k64, layA, layB,
                         :f32, :e2m1, :e2m1, :f32, :ue4m3)

    let f8f6f4 = (:e4m3, :e5m2, :e3m2, :e2m3, :e2m1)
        for a_ty in f8f6f4, b_ty in f8f6f4
            _mma_scaled_register(:mxf8f6f4, Symbol("1X"),
                                 :m16n8k32, layA, layB,
                                 :f32, a_ty, b_ty, :f32, :ue8m0)
        end
    end
end
