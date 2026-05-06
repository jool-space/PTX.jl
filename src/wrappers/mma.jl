# Family generator for `mma.sync.aligned`. Adding a new dtype/shape combo =
# one entry in MMA_SYNC_FRAGS plus one line in the bulk-emit loop.
# Source of truth: PTX 9.2 §9.7.14.5.

# (shape, ab_dtype, cd_dtype) → (n_a_regs, n_b_regs, n_cd_regs) per lane.
# A/B fragments are UInt32 (packed). C/D are Float32 for f32 acc, UInt32 for f16.
# For `kind::f8f6f4` the per-lane count is fixed by the 8-bit container width
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

_mma_ab_letter(::Symbol) = "r"           # AB inputs always pack into UInt32
_mma_ab_julia()          = :UInt32

# CD type depends on accumulator: f32 → Float32 / `f`; f16 → UInt32 packed / `r`.
_mma_cd_letter(c_ty::Symbol) = c_ty === :f32 ? "f" : "r"
_mma_cd_julia(c_ty::Symbol)  = c_ty === :f32 ? :Float32 : :UInt32

function _mma_register(shape::Symbol, layA::Symbol, layB::Symbol,
                       d_ty::Symbol, a_ty::Symbol, b_ty::Symbol, c_ty::Symbol;
                       kind::Union{Symbol, Nothing} = nothing)
    haskey(MMA_SYNC_FRAGS, (shape, a_ty, c_ty)) || return nothing
    n_a, n_b, n_cd = MMA_SYNC_FRAGS[(shape, a_ty, c_ty)]

    mods_kind = kind === nothing ? () : (Symbol("kind::", kind),)
    mods = (:sync, :aligned, mods_kind...,
            shape, layA, layB, d_ty, a_ty, b_ty, c_ty)
    asm_kind = kind === nothing ? "" : "kind::$kind."

    cd_J     = _mma_cd_julia(c_ty)
    ab_let   = _mma_ab_letter(a_ty)
    cd_let   = _mma_cd_letter(c_ty)
    ab_J_sym = _mma_ab_julia()

    # Slot order: D outputs, A inputs, B inputs, C inputs.
    slots(off, n) = "{" * join(("\$$i" for i in off:off+n-1), ", ") * "}"
    d_slots = slots(0,           n_cd)
    a_slots = slots(n_cd,        n_a)
    b_slots = slots(n_cd + n_a,  n_b)
    c_slots = slots(n_cd + n_a + n_b, n_cd)

    asm = "mma.sync.aligned.$asm_kind$shape.$layA.$layB.$d_ty.$a_ty.$b_ty.$c_ty " *
          "$d_slots, $a_slots, $b_slots, $c_slots;"

    constraints = join(vcat(
        fill("=$cd_let", n_cd),
        fill(ab_let,     n_a + n_b),
        fill(cd_let,     n_cd),
        ["~{memory}"]
    ), ",")

    flat_argtypes = vcat(fill(ab_J_sym, n_a + n_b), fill(cd_J, n_cd))

    a_args = [:(a[$i]) for i in 1:n_a]
    b_args = [:(b[$i]) for i in 1:n_b]
    c_args = [:(c[$i]) for i in 1:n_cd]

    @eval function (::Operation{:mma, $mods})(
            a::NTuple{$n_a, UInt32},
            b::NTuple{$n_b, UInt32},
            c::NTuple{$n_cd, $cd_J})
        Base.@inline
        @asmcall($asm, $constraints, true,
                 NTuple{$n_cd, $cd_J},
                 Tuple{$(flat_argtypes...)},
                 $(a_args...), $(b_args...), $(c_args...))
    end
    nothing
end

for layA in (:row, :col), layB in (:row, :col)
    for shape in (:m16n8k16, :m16n8k8), ab_ty in (:bf16, :f16)
        _mma_register(shape, layA, layB, :f32, ab_ty, ab_ty, :f32)
    end
    for shape in (:m16n8k16, :m16n8k8)
        _mma_register(shape, layA, layB, :f16, :f16, :f16, :f16)
    end
    for shape in (:m16n8k8, :m16n8k4)
        _mma_register(shape, layA, layB, :f32, :tf32, :tf32, :f32)
    end
    # FP8 (Ada+) — m16n8k32 (PTX 8.0+) and m16n8k16 (PTX 8.7+).
    for shape in (:m16n8k16, :m16n8k32),
        ab_ty in (:e4m3, :e5m2),
        c_ty  in (:f32, :f16)
        _mma_register(shape, layA, layB, c_ty, ab_ty, ab_ty, c_ty)
    end
    # `kind::f8f6f4` (Blackwell sm_100a+) — mixed A/B legal because all five
    # share the 8-bit container.
    let f8f6f4 = (:e4m3, :e5m2, :e3m2, :e2m3, :e2m1)
        for shape in (:m16n8k16, :m16n8k32),
            a_ty in f8f6f4,
            b_ty in f8f6f4,
            c_ty in (:f32, :f16)
            _mma_register(shape, layA, layB, c_ty, a_ty, b_ty, c_ty;
                          kind = :f8f6f4)
        end
    end
end
