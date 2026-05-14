# Warpgroup-level async MMA (Hopper sm_90a). The accumulator `d` is read AND
# written: passed in as `NTuple{nd, T_d}`, returned as the call value. Wired
# via LLVM tied operands (`=X,...,0,1,...,nd-1`) so input and output bind to
# the same physical register file. `imm_scale_a/b` and `imm_trans_a/b` are
# baked. Sync ops (`wgmma.fence/commit_group/wait_group`) flow through the
# chain — `:wgmma` is in NONPURE_OPCODES. Source: PTX 9.2 §9.7.14.5.
#
# Three variants per shape:
#   1. `scale_d::Bool`        — runtime SREG (b constraint, $nd+2 slot).
#   2. `scale_d::Val{true}`   — bakes "1" immediate, keeps tied d input.
#   3. `scale_d::Val{false}`  — bakes "0" immediate AND drops the tied d
#      input. The HW ignores the accumulator when scale_d=0, so LLVM can
#      DCE upstream zero initialization. Use this for the first wgmma of
#      a tile to skip the per-tile zero ntuple.

# Valid N values for wgmma — step by 8 from 8 to 256.
const _WGMMA_NS = (8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112,
                   120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200,
                   208, 216, 224, 232, 240, 248, 256)

# (dtype_d, dtype_a, dtype_b, k, has_trans). k is fixed per ab-dtype:
# bf16/f16 → k16; tf32 → k8; e4m3/e5m2 → k32. has_trans is true only for
# f16/bf16 inputs (their SMEM descriptor accepts `.trans`).
const _WGMMA_VARIANTS = (
    (:f32, :bf16, :bf16, 16, true),
    (:f32, :f16,  :f16,  16, true),
    (:f32, :tf32, :tf32, 8,  false),
    (:f32, :e4m3, :e4m3, 32, false),
    (:f32, :e5m2, :e5m2, 32, false),
    (:f16, :f16,  :f16,  16, true),
    (:f16, :e4m3, :e4m3, 32, false),
    (:f16, :e5m2, :e5m2, 32, false),
    (:s32, :s8, :s8, 32, false),
    (:s32, :u8, :u8, 32, false),
    (:s32, :s8, :u8, 32, false),
    (:s32, :u8, :s8, 32, false),
)

function _wgmma_dvec_kind(dtype_d::Symbol, n::Int)
    dtype_d === :f32 && return (n >>> 1, Float32, "f")
    dtype_d === :f16 && return (n >>> 2, UInt32,  "r")
    dtype_d === :s32 && return (n >>> 1, Int32,   "r")
    error("wgmma: unsupported accumulator dtype: $dtype_d")
end

# `scale_d_imm`:
#   `nothing` → runtime scale_d slot (`b` constraint, tied d input)
#   `true`    → bake "1", tied d input retained
#   `false`   → bake "0", tied d input DROPPED (HW ignores acc on scale_d=0)
function wgmma_mma_async_spec(dtype_d::Symbol, dtype_a::Symbol, dtype_b::Symbol,
                              n::Int, k::Int, has_trans::Bool;
                              scale_d_imm::Union{Bool, Nothing} = nothing)
    nd, _, d_let = _wgmma_dvec_kind(dtype_d, n)
    head = "wgmma.mma_async.sync.aligned.m64n$(n)k$(k).$dtype_d.$dtype_a.$dtype_b"
    d_slots = "{" * join(("\$$i" for i in 0:nd-1), ", ") * "}"
    # PTX 9.2 §9.7.14.5.7: integer wgmma takes only `scale-d` (no scale-a/b
    # or trans-a/b). FP wgmma takes `scale-d, scale-a, scale-b` plus optional
    # `trans-a, trans-b` for f16/bf16/tf32 inputs (mantissa-shape A/B
    # descriptors). FP8 inputs have no trans bits.
    imms_tail = if dtype_d === :s32
        ""
    elseif has_trans
        ", 1, 1, 0, 0"
    else
        ", 1, 1"
    end
    if scale_d_imm === nothing
        asm = "$head $d_slots, \$$(nd), \$$(nd+1), \$$(nd+2)$imms_tail;"
        constraints = join(vcat(
            fill("=$d_let", nd),
            ["l", "l", "b"],
            [string(i) for i in 0:nd-1],          # tied to outputs 0..nd-1
            ["~{memory}"],
        ), ",")
    elseif scale_d_imm
        asm = "$head $d_slots, \$$(nd), \$$(nd+1), 1$imms_tail;"
        constraints = join(vcat(
            fill("=$d_let", nd),
            ["l", "l"],
            [string(i) for i in 0:nd-1],          # tied to outputs 0..nd-1
            ["~{memory}"],
        ), ",")
    else
        asm = "$head $d_slots, \$$(nd), \$$(nd+1), 0$imms_tail;"
        constraints = join(vcat(
            fill("=$d_let", nd),
            ["l", "l"],
            ["~{memory}"],
        ), ",")
    end
    (; nd, asm, constraints, d_let)
end

function _wgmma_mma_async_register(
        dtype_d::Symbol, dtype_a::Symbol, dtype_b::Symbol,
        n::Int, k::Int, has_trans::Bool)
    _, d_J, _ = _wgmma_dvec_kind(dtype_d, n)
    mods = (:mma_async, :sync, :aligned,
            Symbol("m64n", n, "k", k),
            dtype_d, dtype_a, dtype_b)

    # Variant 1: runtime scale_d::Bool (tied d input, b constraint).
    let spec = wgmma_mma_async_spec(dtype_d, dtype_a, dtype_b, n, k, has_trans)
        nd, asm, constraints = spec.nd, spec.asm, spec.constraints
        flat_argtypes = vcat([UInt64, UInt64, Bool], fill(d_J, nd))
        d_args = [:(d[$i]) for i in 1:nd]
        @eval function (::Operation{:wgmma, $mods})(
                d::NTuple{$nd, $d_J},
                a_desc::UInt64,
                b_desc::UInt64,
                scale_d::Bool)
            Base.@inline
            @asmcall($asm, $constraints, true,
                     NTuple{$nd, $d_J},
                     Tuple{$(flat_argtypes...)},
                     a_desc, b_desc, scale_d, $(d_args...))
        end
    end

    # Variant 2: compile-time scale_d=Val{true} — bakes "1", keeps tied d.
    let spec = wgmma_mma_async_spec(dtype_d, dtype_a, dtype_b, n, k, has_trans;
                                     scale_d_imm = true)
        nd, asm, constraints = spec.nd, spec.asm, spec.constraints
        flat_argtypes = vcat([UInt64, UInt64], fill(d_J, nd))
        d_args = [:(d[$i]) for i in 1:nd]
        @eval function (::Operation{:wgmma, $mods})(
                d::NTuple{$nd, $d_J},
                a_desc::UInt64,
                b_desc::UInt64,
                ::Val{true})
            Base.@inline
            @asmcall($asm, $constraints, true,
                     NTuple{$nd, $d_J},
                     Tuple{$(flat_argtypes...)},
                     a_desc, b_desc, $(d_args...))
        end
    end

    # Variant 3: compile-time scale_d=Val{false} — bakes "0", DROPS tied d.
    # The `_d` parameter is accepted for API uniformity but never reaches
    # @asmcall, so LLVM can DCE upstream zero initialization.
    let spec = wgmma_mma_async_spec(dtype_d, dtype_a, dtype_b, n, k, has_trans;
                                     scale_d_imm = false)
        nd, asm, constraints = spec.nd, spec.asm, spec.constraints
        @eval function (::Operation{:wgmma, $mods})(
                _d::NTuple{$nd, $d_J},
                a_desc::UInt64,
                b_desc::UInt64,
                ::Val{false})
            Base.@inline
            @asmcall($asm, $constraints, true,
                     NTuple{$nd, $d_J},
                     Tuple{UInt64, UInt64},
                     a_desc, b_desc)
        end
    end

    nothing
end

for (dt_d, dt_a, dt_b, k, has_trans) in _WGMMA_VARIANTS, n in _WGMMA_NS
    _wgmma_mma_async_register(dt_d, dt_a, dt_b, n, k, has_trans)
end
