# Warpgroup-level async MMA (Hopper sm_90a). The accumulator `d` is read AND
# written: passed in as `NTuple{nd, T_d}`, returned as the call value. Wired
# via LLVM tied operands (`=X,...,0,1,...,nd-1`) so input and output bind to
# the same physical register file. `imm_scale_a/b` and `imm_trans_a/b` are
# baked. Sync ops (`wgmma.fence/commit_group/wait_group`) flow through the
# chain — `:wgmma` is registered nonpure + convergent in the form
# registry (src/ledgers/forms.jl). Source: PTX 9.3 §9.7.16.5.
#
# Four variants per shape:
#   1. `scale_d::Bool`        — runtime SREG (b constraint, $nd+2 slot).
#   2. `scale_d::Val{true}`   — bakes "1" immediate, keeps tied d input.
#   3. `scale_d::Val{false}`  — bakes "0" immediate AND drops the tied d
#      input. The HW ignores the accumulator when scale_d=0, so LLVM can
#      DCE upstream zero initialization. Use this for the first wgmma of
#      a tile to skip the per-tile zero ntuple.
#   4. RF form — A from registers instead of an SMEM descriptor:
#      `(d, a::NTuple{4, UInt32}, b_desc, scale_d::Bool)`. Every ab-dtype
#      takes exactly four .b32 A registers per lane (two f16/bf16, four
#      tf32-bits, or four 8-bit elements each — PTX 9.3 §9.7.16.5.1).
#      A-in-registers has no `imm-trans-a` (there is no descriptor to
#      transpose), so the f16/bf16 imm tail shrinks to
#      `scale-a, scale-b, trans-b`. Only the runtime-Bool scale_d variant
#      is generated for RF — the Val forms exist for the SS descriptor
#      path's zero-init DCE, which RF mainloops (upconvert-in-registers,
#      CUTLASS mixed-dtype style) don't hit in practice.

# PTX 9.3 §9.7.16.5.2 does not give every dtype family the same N grid.
# Floating-point forms use every multiple of eight through 256.  Integer
# forms use 8,16,24,32 and then multiples of sixteen from 48 through 224.
# Thus every 8-mod-16 value after 32 (n40, n56, ..., n232, n248) and every
# value above 224 (including n240 and n256) is illegal for integer WGMMA.
# Keep the grids separate so registration cannot accidentally manufacture
# the floating×integer cross-product again.
const _WGMMA_FLOAT_NS = (8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96,
                         104, 112, 120, 128, 136, 144, 152, 160, 168, 176,
                         184, 192, 200, 208, 216, 224, 232, 240, 248, 256)
const _WGMMA_INT_NS = (8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160,
                       176, 192, 208, 224)

# (dtype_d, dtype_a, dtype_b, k, has_trans). k is fixed per ab-dtype:
# bf16/f16 → k16; tf32 → k8; e4m3/e5m2 → k32. has_trans is true only for
# f16/bf16 inputs (their SMEM descriptor accepts `.trans`).
const _WGMMA_FLOAT_VARIANTS = (
    (:f32, :bf16, :bf16, 16, true),
    (:f32, :f16,  :f16,  16, true),
    (:f32, :tf32, :tf32, 8,  false),
    (:f32, :e4m3, :e4m3, 32, false),
    (:f32, :e5m2, :e5m2, 32, false),
    (:f16, :f16,  :f16,  16, true),
    (:f16, :e4m3, :e4m3, 32, false),
    (:f16, :e5m2, :e5m2, 32, false),
)

const _WGMMA_INT_VARIANTS = (
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
    # PTX 9.3 §9.7.16.5.2: integer wgmma takes only `scale-d` (no scale-a/b
    # or trans-a/b). FP wgmma takes `scale-d, scale-a, scale-b` plus optional
    # `trans-a, trans-b` for f16/bf16 inputs. TF32 and FP8 inputs have no
    # transpose bits.
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

# RF form: A comes from four per-lane .b32 registers instead of an SMEM
# descriptor (PTX 9.3 §9.7.16.5 `d, a, b-desc, ...`). No `imm-trans-a` —
# a register operand has no descriptor to transpose — so the f16/bf16 imm
# tail is `scale-a, scale-b, trans-b`. Runtime scale_d only (see file top).
function wgmma_mma_async_rf_spec(dtype_d::Symbol, dtype_a::Symbol,
                                 dtype_b::Symbol, n::Int, k::Int,
                                 has_trans::Bool)
    nd, _, d_let = _wgmma_dvec_kind(dtype_d, n)
    head = "wgmma.mma_async.sync.aligned.m64n$(n)k$(k).$dtype_d.$dtype_a.$dtype_b"
    d_slots = "{" * join(("\$$i" for i in 0:nd-1), ", ") * "}"
    a_slots = "{" * join(("\$$(nd + i)" for i in 0:3), ", ") * "}"
    imms_tail = if dtype_d === :s32
        ""
    elseif has_trans
        ", 1, 1, 0"                               # scale-a, scale-b, trans-b
    else
        ", 1, 1"                                  # scale-a, scale-b
    end
    asm = "$head $d_slots, $a_slots, \$$(nd+4), \$$(nd+5)$imms_tail;"
    constraints = join(vcat(
        fill("=$d_let", nd),
        ["r", "r", "r", "r", "l", "b"],
        [string(i) for i in 0:nd-1],              # tied to outputs 0..nd-1
        ["~{memory}"],
    ), ",")
    (; nd, asm, constraints, d_let)
end

function _wgmma_mma_async_register(
        dtype_d::Symbol, dtype_a::Symbol, dtype_b::Symbol,
        n::Int, k::Int, has_trans::Bool)
    _, d_J, _ = _wgmma_dvec_kind(dtype_d, n)
    mods = (:mma_async, :sync, :aligned,
            Symbol("m64n", n, "k", k),
            dtype_d, dtype_a, dtype_b)
    register_wrapper!(:wgmma, :wgmma, mods, :asm)

    # wgmma.mma_async is warpgroup-collective: all 128 threads must execute
    # the same call site. Emitted via convergent_asm_ir (not @asmcall) so the
    # call carries `convergent` — sideeffect alone permits jump-threading
    # duplication across divergent branches, the active_mask miscompile class
    # (see src/dsl/convergent_asm.jl, "convergent inline asm").

    # Variant 1: runtime scale_d::Bool (tied d input, b constraint).
    let spec = wgmma_mma_async_spec(dtype_d, dtype_a, dtype_b, n, k, has_trans)
        nd, asm, constraints = spec.nd, spec.asm, spec.constraints
        flat_argtypes = vcat([UInt64, UInt64, Bool], fill(d_J, nd))
        ir = convergent_asm_ir(asm, constraints, NTuple{nd, d_J}, flat_argtypes)
        d_args = [:(d[$i]) for i in 1:nd]
        @eval function (::Operation{:wgmma, $mods})(
                d::NTuple{$nd, $d_J},
                a_desc::UInt64,
                b_desc::UInt64,
                scale_d::Bool)
            Base.@inline
            Base.llvmcall(($ir, "entry"),
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
        ir = convergent_asm_ir(asm, constraints, NTuple{nd, d_J}, flat_argtypes)
        d_args = [:(d[$i]) for i in 1:nd]
        @eval function (::Operation{:wgmma, $mods})(
                d::NTuple{$nd, $d_J},
                a_desc::UInt64,
                b_desc::UInt64,
                ::Val{true})
            Base.@inline
            Base.llvmcall(($ir, "entry"),
                          NTuple{$nd, $d_J},
                          Tuple{$(flat_argtypes...)},
                          a_desc, b_desc, $(d_args...))
        end
    end

    # Variant 3: compile-time scale_d=Val{false} — bakes "0", DROPS tied d.
    # The `_d` parameter is accepted for API uniformity but never reaches
    # the asm, so LLVM can DCE upstream zero initialization.
    let spec = wgmma_mma_async_spec(dtype_d, dtype_a, dtype_b, n, k, has_trans;
                                     scale_d_imm = false)
        nd, asm, constraints = spec.nd, spec.asm, spec.constraints
        ir = convergent_asm_ir(asm, constraints, NTuple{nd, d_J},
                               [UInt64, UInt64])
        @eval function (::Operation{:wgmma, $mods})(
                _d::NTuple{$nd, $d_J},
                a_desc::UInt64,
                b_desc::UInt64,
                ::Val{false})
            Base.@inline
            Base.llvmcall(($ir, "entry"),
                          NTuple{$nd, $d_J},
                          Tuple{UInt64, UInt64},
                          a_desc, b_desc)
        end
    end

    # Variant 4: RF form — A from four per-lane .b32 registers. Dispatches
    # on the second argument's type (NTuple{4, UInt32} vs UInt64 desc)
    # under the same Operation mods.
    let spec = wgmma_mma_async_rf_spec(dtype_d, dtype_a, dtype_b, n, k,
                                       has_trans)
        nd, asm, constraints = spec.nd, spec.asm, spec.constraints
        flat_argtypes = vcat(fill(UInt32, 4), [UInt64, Bool], fill(d_J, nd))
        ir = convergent_asm_ir(asm, constraints, NTuple{nd, d_J}, flat_argtypes)
        d_args = [:(d[$i]) for i in 1:nd]
        @eval function (::Operation{:wgmma, $mods})(
                d::NTuple{$nd, $d_J},
                a::NTuple{4, UInt32},
                b_desc::UInt64,
                scale_d::Bool)
            Base.@inline
            Base.llvmcall(($ir, "entry"),
                          NTuple{$nd, $d_J},
                          Tuple{$(flat_argtypes...)},
                          a[1], a[2], a[3], a[4], b_desc, scale_d,
                          $(d_args...))
        end
    end

    nothing
end

for (dt_d, dt_a, dt_b, k, has_trans) in _WGMMA_FLOAT_VARIANTS,
        n in _WGMMA_FLOAT_NS
    _wgmma_mma_async_register(dt_d, dt_a, dt_b, n, k, has_trans)
end
for (dt_d, dt_a, dt_b, k, has_trans) in _WGMMA_INT_VARIANTS,
        n in _WGMMA_INT_NS
    _wgmma_mma_async_register(dt_d, dt_a, dt_b, n, k, has_trans)
end
