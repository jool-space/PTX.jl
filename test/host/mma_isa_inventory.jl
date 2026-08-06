# Independent ISA-side inventory of the generated mma families.
#
# The conformance sweep proves the test-side and src-side registration
# loops agree by replaying them verbatim — deliberate for a generated
# family, but structurally blind to a form both copies omit (the mixed
# kind-less fp8 pairs were exactly such a hole: ISA-legal since PTX 8.4,
# absent from both loops, every count pin green). This file is the other
# oracle: the dtype/shape grids below are transcribed from PTX ISA 9.3 —
# §9.7.15.5 (dense), §9.7.15.6 (sparse A), Table 39 in §9.7.15.3 (block
# scaling) — never from src/ registration loops or their helpers, and are
# diffed against the wrapper registry in both directions.
#
# A form the grammar admits that the registry lacks must appear either in
# the expected sets (its absence is then a loud registration bug) or in a
# named exclusion below (a reviewed decision with a citation). A
# registered form the grammar does not admit must sit in the beyond-spec
# set with its own evidence. Growing any set is a review act, same as a
# FORMS entry.

# --- registry-notation spellings (mirror the ptx"" surface, not src) -------

_inv_dense(shape, d, a, b, c) =
    (:sync, :aligned, shape, :row, :col, d, a, b, c)
_inv_dense_kind(kind, shape, d, a, b, c) =
    (:sync, :aligned, kind, shape, :row, :col, d, a, b, c)
_inv_int(shape, a, b, sat) = sat ?
    (:sync, :aligned, shape, :row, :col, :satfinite, :s32, a, b, :s32) :
    (:sync, :aligned, shape, :row, :col, :s32, a, b, :s32)
_inv_b1(shape, bitop) =
    (:sync, :aligned, shape, :row, :col, :s32, :b1, :b1, :s32, bitop, :popc)
_inv_sp(variant, shape, d, a, b, c; sat = false) = sat ?
    (variant, :sync, :aligned, shape, :row, :col, :satfinite, d, a, b, c) :
    (variant, :sync, :aligned, shape, :row, :col, d, a, b, c)
_inv_scaled(kind, sv, shape, a, b, stype) =
    (:sync, :aligned, Symbol("kind::", kind), :block_scale,
     Symbol("scale_vec::", sv), shape, :row, :col,
     :f32, a, b, :f32, stype)

const _INV_FP8    = (:e4m3, :e5m2)
const _INV_F8F6F4 = (:e4m3, :e5m2, :e3m2, :e2m3, :e2m1)
# §9.7.15.5: for m16n8k8/k16/k32, dtype must equal ctype.
const _INV_ACC    = ((:f16, :f16), (:f32, :f32))

# --- dense (§9.7.15.5), family :mma ----------------------------------------

const EXPECTED_MMA = Set{Tuple}()
# f16: m16n8k8 and m16n8k16 (row.col; dtype = ctype). The legacy Volta
# m8n8k4 f16 grid (free alayout/blayout) is policy-excluded below.
for shape in (:m16n8k8, :m16n8k16), (d, c) in _INV_ACC
    push!(EXPECTED_MMA, _inv_dense(shape, d, :f16, :f16, c))
end
# bf16: m16n8k8 (atype/btype grid, atype = btype) and m16n8k16; f32 only.
for shape in (:m16n8k8, :m16n8k16)
    push!(EXPECTED_MMA, _inv_dense(shape, :f32, :bf16, :bf16, :f32))
end
# tf32: m16n8k4 and m16n8k8 (the m16n8k8 atype/btype grid); f32 only.
for shape in (:m16n8k4, :m16n8k8)
    push!(EXPECTED_MMA, _inv_dense(shape, :f32, :tf32, :tf32, :f32))
end
# f64: m8n8k4 plus the sm_90 m16n8k{4,8,16} shapes.
for shape in (:m8n8k4, :m16n8k4, :m16n8k8, :m16n8k16)
    push!(EXPECTED_MMA, _inv_dense(shape, :f64, :f64, :f64, :f64))
end
# Kind-less fp8: shape ∈ {m16n8k16, m16n8k32}; `.f8type` is spelled per
# operand, so A×B mixes freely; dtype = ctype ∈ {f16, f32}.
for shape in (:m16n8k16, :m16n8k32), a in _INV_FP8, b in _INV_FP8,
        (d, c) in _INV_ACC
    push!(EXPECTED_MMA, _inv_dense(shape, d, a, b, c))
end
# kind::f8f6f4: the 9.3 syntax spells the kind form at m16n8k32 only.
for a in _INV_F8F6F4, b in _INV_F8F6F4, (d, c) in _INV_ACC
    push!(EXPECTED_MMA,
          _inv_dense_kind(Symbol("kind::f8f6f4"), :m16n8k32, d, a, b, c))
end
# Integer: m16n8 shapes only (m8n8k16/m8n8k32 are policy-excluded below);
# signedness is per operand; `.satfinite` optional.
for (shape, u, s) in ((:m16n8k16, :u8, :s8), (:m16n8k32, :u8, :s8),
                      (:m16n8k32, :u4, :s4), (:m16n8k64, :u4, :s4)),
        a in (u, s), b in (u, s), sat in (false, true)
    push!(EXPECTED_MMA, _inv_int(shape, a, b, sat))
end

# Registered beyond the 9.3 grammar: kind::f8f6f4 at m16n8k16. ptxas
# accepts the spelling and selection is exercised at sm_121a
# (test/ptxas/sm121a.jl); the forms ride the asm tier (no LLVM intrinsic).
const BEYOND_SPEC_MMA = Set{Tuple}()
for a in _INV_F8F6F4, b in _INV_F8F6F4, (d, c) in _INV_ACC
    push!(BEYOND_SPEC_MMA,
          _inv_dense_kind(Symbol("kind::f8f6f4"), :m16n8k16, d, a, b, c))
end

# --- single-bit (§9.7.15.5), family :mma_b1 --------------------------------

const EXPECTED_MMA_B1 = Set{Tuple}(
    _inv_b1(shape, bitop)
    for shape in (:m8n8k128, :m16n8k128, :m16n8k256), bitop in (:xor, :and))

# --- sparse A (§9.7.15.6), families :mma_sp / :mma_sp_ordered --------------

function _expected_sp(variant)
    out = Set{Tuple}()
    for shape in (:m16n8k16, :m16n8k32), (d, c) in _INV_ACC
        push!(out, _inv_sp(variant, shape, d, :f16, :f16, c))
    end
    for shape in (:m16n8k16, :m16n8k32)
        push!(out, _inv_sp(variant, shape, :f32, :bf16, :bf16, :f32))
    end
    for shape in (:m16n8k8, :m16n8k16)
        push!(out, _inv_sp(variant, shape, :f32, :tf32, :tf32, :f32))
    end
    # Kind-less sparse fp8 is f32-accumulate at m16n8k64 in the 9.3
    # syntax; the sm_120 extensions live in the known-unwrapped set below.
    for a in _INV_FP8, b in _INV_FP8
        push!(out, _inv_sp(variant, :m16n8k64, :f32, a, b, :f32))
    end
    for (shape, u, s) in ((:m16n8k32, :u8, :s8), (:m16n8k64, :u8, :s8),
                          (:m16n8k64, :u4, :s4), (:m16n8k128, :u4, :s4)),
            a in (u, s), b in (u, s), sat in (false, true)
        push!(out, _inv_sp(variant, shape, :s32, a, b, :s32; sat))
    end
    out
end
const EXPECTED_MMA_SP = _expected_sp(:sp)
const EXPECTED_MMA_SP_ORDERED = _expected_sp(Symbol("sp::ordered_metadata"))

# --- block-scaled (§9.7.15.3 Table 39), family :mma_scaled -----------------

const EXPECTED_MMA_SCALED = Set{Tuple}()
for a in _INV_F8F6F4, b in _INV_F8F6F4
    push!(EXPECTED_MMA_SCALED,
          _inv_scaled(:mxf8f6f4, Symbol("1X"), :m16n8k32, a, b, :ue8m0))
end
push!(EXPECTED_MMA_SCALED,
      _inv_scaled(:mxf4, Symbol("2X"), :m16n8k64, :e2m1, :e2m1, :ue8m0))
push!(EXPECTED_MMA_SCALED,
      _inv_scaled(:mxf4nvf4, Symbol("2X"), :m16n8k64, :e2m1, :e2m1, :ue8m0))
push!(EXPECTED_MMA_SCALED,
      _inv_scaled(:mxf4nvf4, Symbol("4X"), :m16n8k64, :e2m1, :e2m1, :ue8m0))
push!(EXPECTED_MMA_SCALED,
      _inv_scaled(:mxf4nvf4, Symbol("4X"), :m16n8k64, :e2m1, :e2m1, :ue4m3))

# --- the diff, both directions ---------------------------------------------

function _inventory_diff(label, records, expected)
    got = Set{Tuple}(Tuple(r.mods) for r in records)
    missing_forms = sort!(collect(setdiff(expected, got)); by = string)
    extra_forms   = sort!(collect(setdiff(got, expected)); by = string)
    foreach(m -> println("$label MISSING (ISA-legal, unregistered): ", m),
            missing_forms)
    foreach(x -> println("$label EXTRA (registered, not in inventory): ", x),
            extra_forms)
    @test isempty(missing_forms)
    @test isempty(extra_forms)
end

@testset "dense mma registry matches the §9.7.15.5 grammar" begin
    @test length(EXPECTED_MMA) == 110
    @test length(BEYOND_SPEC_MMA) == 50
    _inventory_diff("mma", PTX.wrapper_records(:mma),
                    union(EXPECTED_MMA, BEYOND_SPEC_MMA))
    # The beyond-spec forms must stay asm-tier: an intrinsic appearing for
    # them on a backend bump means LLVM now models the form — re-check the
    # then-current ISA before promoting.
    beyond = [r for r in PTX.wrapper_records(:mma)
              if Tuple(r.mods) in BEYOND_SPEC_MMA]
    @test length(beyond) == 50
    @test all(r -> r.tier === :asm, beyond)
end

@testset "b1 mma registry matches the §9.7.15.5 grammar" begin
    @test length(EXPECTED_MMA_B1) == 6
    _inventory_diff("mma_b1", PTX.wrapper_records(:mma_b1), EXPECTED_MMA_B1)
end

@testset "sparse mma registries match the §9.7.15.6 grammar" begin
    @test length(EXPECTED_MMA_SP) == 44
    @test length(EXPECTED_MMA_SP_ORDERED) == 44
    _inventory_diff("mma_sp", PTX.wrapper_records(:mma_sp), EXPECTED_MMA_SP)
    _inventory_diff("mma_sp_ordered", PTX.wrapper_records(:mma_sp_ordered),
                    EXPECTED_MMA_SP_ORDERED)
end

@testset "block-scaled mma registry matches Table 39" begin
    @test length(EXPECTED_MMA_SCALED) == 29
    _inventory_diff("mma_scaled", PTX.wrapper_records(:mma_scaled),
                    EXPECTED_MMA_SCALED)
end

# --- reviewed absences ------------------------------------------------------
#
# Each block pins a surface the ISA admits and the registry deliberately
# lacks. Wrapping any of it must flip the corresponding assertion so the
# inventory is updated in the same change — silence stays impossible in
# both directions.

@testset "legacy m8n8 shapes stay unwrapped (policy)" begin
    # §9.7.15.5: m8n8k4 f16 (free alayout/blayout, Volta-optimized),
    # m8n8k16 u8/s8, m8n8k32 u4/s4. Dead shapes on the supported targets;
    # m8n8k4 remains wrapped for f64 only.
    recs = PTX.wrapper_records(:mma)
    @test all(r -> !(:m8n8k4 in r.mods && :f16 in r.mods), recs)
    @test all(r -> !(:m8n8k16 in r.mods) && !(:m8n8k32 in r.mods), recs)
end

@testset "sm_120 sparse fp8 extensions stay unwrapped (documented gap)" begin
    # §9.7.15.6 PTX ISA notes (8.7) add fp8 shape .m16n8k32 and .f16
    # dtype/ctype, target-gated sm_120; the syntax block spells only the
    # m16n8k64 f32 forms, which is what the registry carries.
    sp_recs = PTX.wrapper_records(:mma_sp, :mma_sp_ordered)
    fp8 = r -> any(t -> t in _INV_FP8, r.mods)
    @test all(r -> !(fp8(r) && :m16n8k32 in r.mods), sp_recs)
    @test all(r -> !(fp8(r) && :f16 in r.mods), sp_recs)
end

@testset "sparse kind/block_scale surfaces stay unwrapped (documented gap)" begin
    # §9.7.15.6: sp::ordered_metadata kind::f8f6f4 (m16n8k64) and the
    # sp::ordered_metadata block_scale kinds (m16n8k128 mxf4/mxf4nvf4,
    # m16n8k64 mxf8f6f4), all sm_120a-gated.
    sp_recs = PTX.wrapper_records(:mma_sp, :mma_sp_ordered)
    @test all(r -> !any(m -> occursin("kind::", String(m)), r.mods), sp_recs)
    @test all(r -> !(:block_scale in r.mods), sp_recs)
    scaled = PTX.wrapper_records(:mma_scaled)
    @test all(r -> !any(m -> occursin("sp", String(m)), r.mods), scaled)
end
