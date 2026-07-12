# TEST_TARGET: requires=toolkit evidence=compile
# Blackwell tcgen05 FlashAttention forward (bf16, head_dim 128) —
# algorithmic port of pyptx examples/blackwell/flash_attention_blackwell.py
# `build_flash_attention_blackwell` (the 1-CTA variant). The architecture
# follows FlashAttention-4 / CUTLASS 77_blackwell_fmha.
#
# DEFINITIONS ONLY (no testsets): included by gpu/blackwell/
# flash_attention.jl (the correctness tests, default config) and by
# bench/flash_attention.jl (the perf/debug workbench, which sweeps
# configs). The kernel is parameterized by a compile-time config
# NamedTuple passed as `Val(cfg)` — the Julia equivalent of pyptx's
# builder kwargs; each config compiles a fresh specialization:
#
#   scoreboard :: Bool  — drop the explicit `tcgen05.wait::ld` in the
#       softmax stream and lean on the SASS register scoreboard for the
#       RAW hazard (FA4 ships this: it emits 0 wait::ld). Default false
#       (spec-conservative, B200-validated).
#   beacon     :: Bool  — hang forensics: every mbarrier wait becomes a
#       bounded try_wait; on timeout the thread records
#       0xBEAC0000 | site<<8 | phase into its dbg slot (first blame only)
#       and PRETENDS SUCCESS so the pipeline drains and the kernel exits
#       instead of hanging. Decode with FAB_SITES. Default false.
#   nreg       :: NTuple{3,Int} — setmaxnreg split (softmax, correction,
#       mma/load warpgroup). Constraint: 256a + 128b + 128c ≤ 65536.
#       Default (136, 96, 144) = pyptx's split.
#
# Structure (faithful to pyptx):
#   * 256 query rows per CTA as two 128-row subtiles ("stages"); BN=128 KV
#     tiles staged through a 4-slot SMEM ring (K,V alternating) fed by TMA.
#   * tcgen05.mma computes S = Q@K^T into TMEM; P@V accumulates onto O in
#     TMEM with the A operand sourced directly FROM TMEM (the new
#     `tcgen05.mma` UInt32-A method → llvm.nvvm.tcgen05.mma.tensor).
#   * Warp-specialized, 16 warps: softmax (w0-7, two stage-parallel
#     warpgroups; thread t owns row t&127 and streams its 128 columns in
#     four 32-column chunks, double-buffered), correction (w8-11), MMA
#     dispatch (w12 lane 0), TMA load (w14 lane 0); w13/w15 idle.
#   * Stale-basis softmax: each tile exps against the PREVIOUS tile's
#     basis immediately; the row max / basis decision defers one tile and
#     runs in the next tile's TMEM-load shadow. Each 32-column quarter of
#     P publishes immediately (BAR_P_Q), so PV starts after 32 columns.
#   * FA4 "skip correction": O is rescaled in TMEM only when the running
#     row max moved by more than RESCALE_THRESHOLD in scaled-log2 units.
#     In the common no-rescale case the softmax warps release the O
#     barrier themselves (warp ballot); the correction warps re-derive
#     the same ballot from the alphas in shared memory, so barrier
#     arrival counts stay exact.
#   * BAR_O_RESC is split by KV-tile parity ([stage][tile&1]): the alpha
#     stream can run up to two tiles ahead of the MMA warp's consumption,
#     which would alias a single barrier's phase parity.
#   * ONE cross-work-item gate: the softmax pool's item start waits for
#     the previous epilogue's O readout (BAR_EPI) — softmax TMEM traffic
#     concurrent with the epilogue's O reads corrupts rows.
#   * Persistent scheduling: grid = one wave of CTAs, each looping over
#     (m_block, bh) work items; m_block varies fastest so consecutive
#     CTAs share K/V in L2.
#
# TMEM layout (512 columns of 32-bit):
#     cols   0-127   S0 (f32) — P0 (packed bf16) overwrites cols 0-63
#     cols 128-255   S1 (f32) — P1 overwrites cols 128-191
#     cols 256-383   O0 (f32 accumulator)
#     cols 384-511   O1 (f32 accumulator)
#
# Tensors are passed 2D as (batch*heads*seqlen, head_dim) bf16 row-major.
#
# Port notes vs pyptx (deliberate, all diagnosed — not casual):
#   * Explicit `tcgen05.wait::ld` in the softmax stream (pyptx default
#     `dbg_scoreboard_ld=True` drops it and leans on the SASS register
#     scoreboard, matching FA4's 0 wait::ld). Correct-by-spec first; the
#     scoreboard variant is a measured-perf follow-up.
#   * The FMA-pipe f32x2 exp2 emulation (`emu_pairs`) is skipped — all 16
#     pairs use `ex2.approx.ftz.f32`. Pure port-simplicity; the emu trick
#     only rebalances SFU/FMA pipe pressure.
#   * Debug scaffolding (waitmap / beacons / lockstep / s_f16) not ported.
#   * Shape parameters (seqlen, n_tiles, total_work, ...) are runtime
#     kernel arguments — one compiled kernel serves every shape (pyptx
#     rebuilds per shape). Static loop structure is unchanged: the pyptx
#     build-time `if trips > 0` becomes a runtime while that trips 0×.

using PTX: smem_addr_u32, tcgen05_descriptor, tcgen05_instr_desc_f16bf16_f32,
           BlackwellLayout, tensor_map_tile_2d, bf16x2_pack
using PTX.MBarriers: barrier_init, barrier_arrive, barrier_arrive_expect_tx,
                     barrier_try_wait
using PTX.NVVM                    # @nvvm_str for vote.ballot.sync
using CUDACore
using Random

const FAB_BM       = 128            # rows per Q subtile
const FAB_QSTAGE   = 2              # subtiles per CTA → 256 rows
const FAB_BN       = 128            # KV tile size
const FAB_HD       = 128            # head dim (fixed)
const FAB_KV_SLOTS = 4              # SMEM ring slots, each one K or V tile
const FAB_THREADS  = 512
const FAB_LOG2E    = 1.4426950408889634
const FAB_THRESH   = 256.0f0        # 2^8: FA4 skip-correction gate (post-exp2 domain)

const FAB_TILE_BYTES   = FAB_BN * FAB_HD * 2       # 32 KiB per K/V/Q tile
const FAB_STRIPE_BYTES = 16384                     # TMA 128B-swizzle stripe
const FAB_Q_BYTES      = FAB_QSTAGE * FAB_TILE_BYTES
const FAB_SLOT_UNITS   = FAB_TILE_BYTES ÷ 16       # ring-slot stride, 16B desc units

# SMEM map (byte offsets into dynamic SMEM)
const FAB_SMEM_Q     = 0
const FAB_SMEM_KV    = FAB_SMEM_Q + FAB_Q_BYTES                     # 65536
const FAB_SMEM_STATS = FAB_SMEM_KV + FAB_KV_SLOTS * FAB_TILE_BYTES  # 196608
#   alpha[2][128] @ +0 (f32, stage-major), sum[2][128] @ +1024
const FAB_SMEM_BARS  = FAB_SMEM_STATS + 2048

# mbarrier byte offsets relative to FAB_SMEM_BARS
const FAB_BAR_Q          = 0     # 2 × 8   (count 1, TMA tx)
const FAB_BAR_KV_FULL    = 16    # 4 × 8   (count 1, TMA tx)
const FAB_BAR_KV_FREE    = 48    # 4 × 8   (count 1, tcgen05.commit)
const FAB_BAR_S_FULL     = 80    # 2 × 8   (count 1, tcgen05.commit)
const FAB_BAR_P_Q        = 96    # 2 stages × 4 quarters × 8 (count 128)
const FAB_BAR_O_RESC     = 160   # 2 stages × 2 parities × 8 (count 128)
const FAB_BAR_O_DONE     = 192   # 8       (count 1, tcgen05.commit)
const FAB_BAR_STATS_FREE = 200   # 2 × 8   (count 128)
const FAB_BAR_PV_DONE    = 216   # 2 × 8   (count 1, tcgen05.commit)
const FAB_BAR_Q_FREE     = 232   # 8       (count 1, tcgen05.commit)
const FAB_BAR_EPI        = 240   # 8       (count 128, softmax item-start gate)
const FAB_BAR_BYTES      = 248

const FAB_SMEM_TMEM  = FAB_SMEM_BARS + FAB_BAR_BYTES
const FAB_SMEM_BYTES = FAB_SMEM_TMEM + 8

# TMEM columns: S/P at stage*128, O at 256 + stage*128.

const FAB_MMA_TID  = UInt32(384)   # warp 12 lane 0
const FAB_LOAD_TID = UInt32(448)   # warp 14 lane 0

# ── compile-time config ─────────────────────────────────────────────────

# Default: the B200-validated configuration (2026-07-12).
# nreg = pyptx's split (setmaxnreg is warpgroup-collective):
# 136*256 + 96*128 + 144*128 = 65536.
const FAB_CFG_DEFAULT = (scoreboard = false, beacon = false,
                         nreg = (136, 96, 144))

function fab_check_cfg(cfg)
    a, b, c = cfg.nreg
    pool = 256a + 128b + 128c
    pool <= 65536 ||
        error("nreg $(cfg.nreg): 256*$a + 128*$b + 128*$c = $pool > 65536")
    all(x -> 24 <= x <= 256 && x % 8 == 0, cfg.nreg) ||
        error("nreg values must be multiples of 8 in [24, 256]")
    cfg
end

# setmaxnreg through a Val function barrier: the immediate operand must
# be a compile-time constant, and the type parameter guarantees it
# survives any constprop weather between the config NamedTuple and the
# immarg.
@inline _fab_maxnreg_inc(::Val{N}) where {N} =
    ptx"setmaxnreg.inc.sync.aligned.u32"(Val(N))
@inline _fab_maxnreg_dec(::Val{N}) where {N} =
    ptx"setmaxnreg.dec.sync.aligned.u32"(Val(N))

# ── beacon plumbing (hang forensics) ────────────────────────────────────
# pyptx's `bwait` idea: waits are bounded; on timeout, record a
# first-blame code and pretend success so the pipeline drains and the
# host gets its buffer back instead of a hung kernel. One UInt32 slot
# per thread; 0 = never timed out.

struct FabDbg{BEACON}
    ptr::Core.LLVMPtr{UInt32, 1}
    tid::UInt32
end

const FAB_SITES = Dict(
    1  => "load:KV_FREE",   2  => "load:Q_FREE",
    3  => "mma:Q",          4  => "mma:KV_FULL",
    5  => "pv:P_Q",         6  => "pv:O_RESC",
    7  => "softmax:S_FULL", 8  => "softmax:STATS_FREE",
    9  => "softmax:EPI",    10 => "corr:PV_DONE",
    11 => "corr:O_DONE",
)

@inline _fab_wait(::FabDbg{false}, mbar, ph::UInt32, ::Val) =
    barrier_try_wait(mbar, ph)
@inline function _fab_wait(d::FabDbg{true}, mbar, ph::UInt32,
                           ::Val{SITE}) where {SITE}
    tries = UInt32(0)
    while !ptx"mbarrier.try_wait.parity.shared.b64"(mbar, ph)
        tries += UInt32(1)
        if tries > UInt32(3_000_000)
            slot = d.ptr + Int(d.tid) * 4
            if ptx"ld.global.b32"(slot) == UInt32(0)      # first blame only
                ptx"st.global.b32"(slot,
                    UInt32(0xBEAC0000) | (UInt32(SITE) << 8) | ph)
            end
            return nothing   # pretend success: let the pipeline drain
        end
    end
    nothing
end

# Per-work-item Q row base: m_block = w & mb_mask varies fastest,
# bh = w >> mb_shift.
@inline _fab_q_row0(w::UInt32, seqlen::UInt32, mb_shift::UInt32, mb_mask::UInt32) =
    (w >> mb_shift) * seqlen + (w & mb_mask) * UInt32(FAB_QSTAGE * FAB_BM)

# One 128×128 bf16 tile = two TMA 128-row × 64-col swizzle stripes.
@inline function _fab_load_tile(dst_ptr, byteoff::Int, tma, row::UInt32, mbar)
    for stripe in 0:1
        ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
            dst_ptr + byteoff + stripe * FAB_STRIPE_BYTES, tma,
            Int32(stripe * 64), reinterpret(Int32, row), mbar)
    end
end

# Ring-slot load: wait FREE (uniform accounting — the ring is self-credited
# once at role start), expect_tx, two stripe loads. Returns advanced row.
@inline function _fab_load_kv(d::FabDbg, ::Val{SLOT}, kv_ptr, tma, row::UInt32, mb,
                              ph::UInt32) where {SLOT}
    _fab_wait(d, mb + (FAB_BAR_KV_FREE + 8 * SLOT), ph, Val(1))
    barrier_arrive_expect_tx(mb + (FAB_BAR_KV_FULL + 8 * SLOT), FAB_TILE_BYTES)
    _fab_load_tile(kv_ptr, SLOT * FAB_TILE_BYTES, tma,
                   row, mb + (FAB_BAR_KV_FULL + 8 * SLOT))
    return row + UInt32(FAB_BN), ph ⊻ UInt32(1)
end

@inline _fab_commit(mbar_ptr) =
    ptx"tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64"(
        smem_addr_u32(mbar_ptr))

# Descriptor offset (16B units) for k-atom kk of a K-major 128-col bf16
# tile: k-atoms 0-3 live in stripe 0 (+32 B each), 4-7 in stripe 1 (+16 KiB).
@inline _fab_kmaj_off(kk::Int) = UInt64((kk ÷ 4) * 1024 + (kk % 4) * 2)

# S(stage) = Q(stage) @ K(k_idx)^T — 8 k16 UMMAs, accumulate within the tile.
@inline function _fab_qk_mma(::Val{STAGE}, ::Val{KIDX}, tmem::UInt32,
                             dq0::UInt64, dk0::UInt64, idesc::UInt32,
                             mb) where {STAGE, KIDX}
    d = tmem + UInt32(STAGE * 128)
    for kk in 0:7
        da = dq0 + UInt64(STAGE * FAB_SLOT_UNITS) + _fab_kmaj_off(kk)
        db = dk0 + UInt64(KIDX * 2 * FAB_SLOT_UNITS) + _fab_kmaj_off(kk)
        ptx"tcgen05.mma.cta_group::1.kind::f16"(d, da, db, idesc, kk != 0)
    end
    _fab_commit(mb + (FAB_BAR_S_FULL + 8 * STAGE))
end

# O(stage) += P(stage) @ V(v_idx) — A sourced from TMEM, consumed per
# 32-column quarter as the softmax publishes it (4-way split-P). PV(tile)
# is gated on O_RESC[stage][tile&1] at the first quarter.
@inline function _fab_pv_mma(dbg::FabDbg, ::Val{STAGE}, ::Val{VIDX}, first_accum::Bool,
                             ::Val{PAR}, tmem::UInt32, dv0::UInt64,
                             idesc::UInt32, mb,
                             ph_pq::UInt32, ph_or::UInt32) where {STAGE, VIDX, PAR}
    d = tmem + UInt32(256 + STAGE * 128)
    for q in 0:3
        _fab_wait(dbg, mb + (FAB_BAR_P_Q + 32 * STAGE + 8 * q), ph_pq, Val(5))
        if q == 0
            _fab_wait(dbg, mb + (FAB_BAR_O_RESC + 16 * STAGE + 8 * PAR), ph_or, Val(6))
        end
        ptx"tcgen05.fence::after_thread_sync"()
        for kk in (2q, 2q + 1)
            # `% UInt32`: kk is loop-carried through a tuple iteration, so
            # the checked UInt32() truncation survives LLVM's range
            # analysis as a live InexactError branch; the unchecked
            # conversion is exact by construction (kk ∈ 0:7).
            a = tmem + ((STAGE * 128 + kk * 8) % UInt32)
            db = dv0 + ((VIDX * 2 * FAB_SLOT_UNITS + kk * 128) % UInt64)
            ptx"tcgen05.mma.cta_group::1.kind::f16"(d, a, db, idesc,
                                                    !(first_accum & (kk == 0)))
        end
    end
    # deferred-rescale handshake: correction may only touch O after this
    # tile's PV has fully accumulated
    _fab_commit(mb + (FAB_BAR_PV_DONE + 8 * STAGE))
    return ph_pq ⊻ UInt32(1), ph_or ⊻ UInt32(1)
end

# ── softmax helpers (per-thread full row, 32-column chunks) ─────────────

@inline _fab_exp_chunk(v::NTuple{32, UInt32}, qk_scale::Float32, mneg::Float32) =
    ntuple(Val(32)) do i
        ptx"ex2.approx.ftz.f32"(fma(reinterpret(Float32, v[i]), qk_scale, mneg))
    end

# Fold a chunk into 4 strided accumulators (pyptx chunk_reduce shape).
# Written as explicit tuple expressions: an `ntuple ... do` closure of this
# size escapes the inliner and becomes a real device-side call.
@inline _fab_chunk_max(w::NTuple{32, Float32}) =
    (max(max(w[1], w[5], w[9]),  max(w[13], w[17]), max(w[21], w[25], w[29])),
     max(max(w[2], w[6], w[10]), max(w[14], w[18]), max(w[22], w[26], w[30])),
     max(max(w[3], w[7], w[11]), max(w[15], w[19]), max(w[23], w[27], w[31])),
     max(max(w[4], w[8], w[12]), max(w[16], w[20]), max(w[24], w[28], w[32])))
@inline _fab_chunk_sum(w::NTuple{32, Float32}) =
    (((w[1] + w[5]) + (w[9] + w[13])) + ((w[17] + w[21]) + (w[25] + w[29])),
     ((w[2] + w[6]) + (w[10] + w[14])) + ((w[18] + w[22]) + (w[26] + w[30])),
     ((w[3] + w[7]) + (w[11] + w[15])) + ((w[19] + w[23]) + (w[27] + w[31])),
     ((w[4] + w[8]) + (w[12] + w[16])) + ((w[20] + w[24]) + (w[28] + w[32])))

# exp → pack bf16 → store P quarter → publish BAR_P_Q[C] → fold accs.
# The wait::st round trip hides under the next chunk's in-flight load.
@inline function _fab_softmax_chunk(::Val{C}, v::NTuple{32, UInt32},
                                    macc::NTuple{4, Float32}, sacc::NTuple{4, Float32},
                                    sp_base::UInt32, pq_ptr,
                                    qk_scale::Float32, mneg::Float32) where {C}
    w = _fab_exp_chunk(v, qk_scale, mneg)
    pk = ntuple(k -> bf16x2_pack(w[2k - 1], w[2k]), Val(16))
    ptx"tcgen05.st.sync.aligned.32x32b.x16.b32"(sp_base + UInt32(C * 16), pk)
    ptx"tcgen05.wait::st.sync.aligned"()
    ptx"tcgen05.fence::before_thread_sync"()
    barrier_arrive(pq_ptr + 8 * C)
    cm = _fab_chunk_max(w)
    cs = _fab_chunk_sum(w)
    if C == 0
        return cm, cs
    else
        return ntuple(a -> max(macc[a], cm[a]), Val(4)),
               ntuple(a -> sacc[a] + cs[a], Val(4))
    end
end

# One KV tile of the softmax stream. Returns the updated per-thread state
# (ph_s, ph_stats, mneg, pmax_pend, sums_pend, l_run).
@inline function _fab_softmax_body(d::FabDbg, sb::Bool, first::Bool, oresc_par_ptr,
                                   sp_base::UInt32, sfull, sfree, pq_ptr,
                                   stats, alpha_idx::Int, stats_bar::UInt32,
                                   qk_scale::Float32,
                                   ph_s::UInt32, ph_stats::UInt32, mneg::Float32,
                                   pmax::Float32, sums::Float32, l_run::Float32)
    _fab_wait(d, sfull, ph_s, Val(7))
    ph_s ⊻= UInt32(1)
    ptx"tcgen05.fence::after_thread_sync"()

    # first chunk's load flies under the deferred tail
    v0 = ptx"tcgen05.ld.sync.aligned.32x32b.x32.b32"(sp_base)

    if !first
        # ── deferred tail of tile k-1: basis decision from this thread's
        # own row max on the exp'd values (exp2 is monotonic; pmax > 2^T
        # means the basis drifted by more than T scaled-log2 units) ──
        alpha = 1.0f0
        if pmax > FAB_THRESH
            alpha = ptx"rcp.approx.f32"(pmax)
            mneg -= ptx"lg2.approx.f32"(pmax)
        end
        l_run = (l_run + sums) * alpha

        # publish alpha_{k-1}: gates PV(k) at parity k&1. The common-case
        # O_RESC release goes first; the stats-slot publish (paced by
        # correction) follows.
        needs = alpha < 1.0f0
        blt = nvvm"vote.ballot.sync"(0xffffffff, needs)
        if blt == UInt32(0)
            barrier_arrive(oresc_par_ptr)
        end
        _fab_wait(d, sfree, ph_stats, Val(8))
        ph_stats ⊻= UInt32(1)
        @inbounds stats[alpha_idx] = alpha
        ptx"bar.arrive"(stats_bar, UInt32(64))
    end

    # ── stream 4 chunks: ld(c+1) issues as soon as ld(c) lands (or, in
    # scoreboard mode, immediately — up to two loads in flight, the RAW
    # hazard left to the SASS register scoreboard as FA4 does), and flies
    # under exp/pack/store(c) ──
    sb || ptx"tcgen05.wait::ld.sync.aligned"()
    v1 = ptx"tcgen05.ld.sync.aligned.32x32b.x32.b32"(sp_base + UInt32(32))
    macc, sacc = _fab_softmax_chunk(Val(0), v0, ntuple(_ -> 0.0f0, Val(4)),
                                    ntuple(_ -> 0.0f0, Val(4)),
                                    sp_base, pq_ptr, qk_scale, mneg)
    sb || ptx"tcgen05.wait::ld.sync.aligned"()
    v2 = ptx"tcgen05.ld.sync.aligned.32x32b.x32.b32"(sp_base + UInt32(64))
    macc, sacc = _fab_softmax_chunk(Val(1), v1, macc, sacc,
                                    sp_base, pq_ptr, qk_scale, mneg)
    sb || ptx"tcgen05.wait::ld.sync.aligned"()
    v3 = ptx"tcgen05.ld.sync.aligned.32x32b.x32.b32"(sp_base + UInt32(96))
    macc, sacc = _fab_softmax_chunk(Val(2), v2, macc, sacc,
                                    sp_base, pq_ptr, qk_scale, mneg)
    sb || ptx"tcgen05.wait::ld.sync.aligned"()
    macc, sacc = _fab_softmax_chunk(Val(3), v3, macc, sacc,
                                    sp_base, pq_ptr, qk_scale, mneg)

    # horizontal reduce into the pending carries
    pmax = max(max(macc[1], macc[2], macc[3]), macc[4])
    sums = (sacc[1] + sacc[2]) + (sacc[3] + sacc[4])
    return ph_s, ph_stats, mneg, pmax, sums, l_run
end

# ── correction helpers ──────────────────────────────────────────────────

# Element-wise O ops as top-level helpers: a closure over a variable that
# is reassigned in its enclosing loop gets boxed (dynamic dispatch in
# device code) — arguments don't.
@inline _fab_scale64(v::NTuple{64, UInt32}, s::Float32) =
    ntuple(i -> reinterpret(UInt32, reinterpret(Float32, v[i]) * s), Val(64))
# Q as a type parameter: `v[32q + ...]` with a runtime q is a dynamic
# NTuple{64} getindex, which lowers through an alloca that SROA cannot
# remove — the whole 64-register tuple demotes to local memory.
@inline _fab_pack16(v::NTuple{64, UInt32}, ::Val{Q}, s::Float32) where {Q} =
    ntuple(Val(16)) do k
        bf16x2_pack(reinterpret(Float32, v[32Q + 2k - 1]) * s,
                    reinterpret(Float32, v[32Q + 2k]) * s)
    end

# One tile of the correction stream for one stage: consume the alpha,
# rescale O in TMEM iff any row in this warp's band needs it. The PV_DONE
# wait happens EVERY tile (correction idles here anyway) so the phase var
# tracks the true completion count — a parity-only wait inside the rare
# branch could alias an older same-parity completion and race the
# in-flight PV.
@inline function _fab_corr_tile(d::FabDbg, ::Val{STAGE}, mb, stats, alpha_idx::Int,
                                stats_bar::UInt32, o_addr::UInt32,
                                ponc::UInt32, ph_pv::UInt32) where {STAGE}
    ptx"bar.sync"(stats_bar, UInt32(64))
    alpha = @inbounds stats[alpha_idx]
    barrier_arrive(mb + (FAB_BAR_STATS_FREE + 8 * STAGE))
    _fab_wait(d, mb + (FAB_BAR_PV_DONE + 8 * STAGE), ph_pv, Val(10))
    need = alpha < 1.0f0
    blt = nvvm"vote.ballot.sync"(0xffffffff, need)
    # softmax already released O_RESC when nothing rescales
    if blt != UInt32(0)
        ptx"tcgen05.fence::after_thread_sync"()
        for half in 0:1
            addr = o_addr + UInt32(half * 64)
            ov = ptx"tcgen05.ld.sync.aligned.32x32b.x64.b32"(addr)
            ptx"tcgen05.wait::ld.sync.aligned"()
            ptx"tcgen05.st.sync.aligned.32x32b.x64.b32"(addr, _fab_scale64(ov, alpha))
        end
        ptx"tcgen05.wait::st.sync.aligned"()
        ptx"tcgen05.fence::before_thread_sync"()
        barrier_arrive(mb + (FAB_BAR_O_RESC + 16 * STAGE) + Int(ponc))
    end
    return ph_pv ⊻ UInt32(1)
end

# Epilogue for one stage: normalize O by the row sum, pack bf16, store.
@inline function _fab_epi_stage(::Val{STAGE}, mb, stats, alpha_idx::Int,
                                stats_bar::UInt32, o_addr::UInt32,
                                out_row::UInt32, po) where {STAGE}
    ptx"bar.sync"(stats_bar, UInt32(64))          # streaming softmax's row sum
    l = @inbounds stats[alpha_idx + 256]
    inv_l = ptx"rcp.approx.f32"(l)

    row = out_row + UInt32(STAGE * FAB_BM)
    gptr = po + Int(row) * (FAB_HD * 2)
    for half in 0:1
        addr = o_addr + UInt32(half * 64)
        ov = ptx"tcgen05.ld.sync.aligned.32x32b.x64.b32"(addr)
        ptx"tcgen05.wait::ld.sync.aligned"()
        if STAGE == 1 && half == 1
            # the LAST TMEM read is complete: everything the gates protect
            # (the readout) is done, and the remaining normalize/pack/store
            # is register→global work. Release the next item now so it
            # overlaps the store tail.
            barrier_arrive(mb + (FAB_BAR_STATS_FREE + 8))
            barrier_arrive(mb + FAB_BAR_EPI)
            barrier_arrive(mb + FAB_BAR_O_RESC)          # stage 0, parity 0
            barrier_arrive(mb + (FAB_BAR_O_RESC + 16))   # stage 1, parity 0
        end
        pk = _fab_pack16(ov, Val(0), inv_l)
        for vec in 0:3
            ptx"st.global.v4.b32"(gptr + half * 128 + vec * 16,
                (pk[4vec + 1], pk[4vec + 2], pk[4vec + 3], pk[4vec + 4]))
        end
        pk = _fab_pack16(ov, Val(1), inv_l)
        for vec in 0:3
            ptx"st.global.v4.b32"(gptr + half * 128 + 64 + vec * 16,
                (pk[4vec + 1], pk[4vec + 2], pk[4vec + 3], pk[4vec + 4]))
        end
    end
    if STAGE == 0
        # stage-0 sums slot consumed: return the credit
        barrier_arrive(mb + FAB_BAR_STATS_FREE)
    end
end

# ── the kernel ──────────────────────────────────────────────────────────

function _fab_kernel!(
        O::CuDeviceVector{UInt16, 1},
        tma_Q::PTX.TMADescriptorPtr,
        tma_K::PTX.TMADescriptorPtr,
        tma_V::PTX.TMADescriptorPtr,
        seqlen::UInt32, mb_shift::UInt32, mb_mask::UInt32,
        n_tiles::UInt32, total_work::UInt32,
        qk_scale::Float32,
        dbg::CuDeviceVector{UInt32, 1},   # 512 slots; only written by beacon
        ::Val{CFG}) where {CFG}

    # @inbounds: CuDynamicSharedArray's @boundscheck plants
    # gpu_report_exception cold paths (and their register save/restore) in
    # the entry region. Not launch-critical — the 2026-07-12 B200 run
    # passed under Pkg.test's --check-bounds=yes, i.e. with these calls
    # present, so `.reqntid` alone is what makes ptxas honor setmaxnreg —
    # but the checks are dead weight in a hand-verified SMEM layout.
    smem_q    = @inbounds CuDynamicSharedArray(UInt16, FAB_Q_BYTES ÷ 2, FAB_SMEM_Q)
    smem_kv   = @inbounds CuDynamicSharedArray(UInt16, FAB_KV_SLOTS * FAB_TILE_BYTES ÷ 2,
                                               FAB_SMEM_KV)
    stats     = @inbounds CuDynamicSharedArray(Float32, 512, FAB_SMEM_STATS)
    bars      = @inbounds CuDynamicSharedArray(UInt64, FAB_BAR_BYTES ÷ 8, FAB_SMEM_BARS)
    tmem_slot = @inbounds CuDynamicSharedArray(UInt32, 1, FAB_SMEM_TMEM)

    q_ptr  = pointer(smem_q)
    kv_ptr = pointer(smem_kv)
    mb     = pointer(bars)

    tid      = ptx"mov.u32"(sreg"tid.x")
    cta_id   = ptx"mov.u32"(sreg"ctaid.x")
    num_ctas = ptx"mov.u32"(sreg"nctaid.x")
    warp   = tid >> UInt32(5)
    wig    = warp & UInt32(3)                     # warp in group
    row128 = tid & UInt32(127)
    lane_bits = (tid & UInt32(96)) << UInt32(16)  # TMEM lane bits: (warp%4)*32

    d  = FabDbg{CFG.beacon}(pointer(dbg), tid)
    sb = CFG.scoreboard

    # ── init barriers, allocate TMEM ──
    if tid == UInt32(0)
        for i in 0:1
            barrier_init(mb + (FAB_BAR_Q + 8i), 1)
            barrier_init(mb + (FAB_BAR_S_FULL + 8i), 1)
            for par in 0:1
                barrier_init(mb + (FAB_BAR_O_RESC + 16i + 8par), 128)
            end
            barrier_init(mb + (FAB_BAR_STATS_FREE + 8i), 128)
            barrier_init(mb + (FAB_BAR_PV_DONE + 8i), 1)
            for q in 0:3
                barrier_init(mb + (FAB_BAR_P_Q + 32i + 8q), 128)
            end
        end
        for s in 0:(FAB_KV_SLOTS - 1)
            barrier_init(mb + (FAB_BAR_KV_FULL + 8s), 1)
            barrier_init(mb + (FAB_BAR_KV_FREE + 8s), 1)
        end
        barrier_init(mb + FAB_BAR_O_DONE, 1)
        barrier_init(mb + FAB_BAR_Q_FREE, 1)
        barrier_init(mb + FAB_BAR_EPI, 128)       # arrived by 128 corr threads
        ptx"fence.proxy.async.shared::cta"()
    end
    if warp == UInt32(12)
        ptx"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"(
            smem_addr_u32(pointer(tmem_slot)), UInt32(512))
    end
    ptx"bar.sync"(Val(0))
    tmem = @inbounds tmem_slot[1]

    # =====================================================================
    # TMA load warp (single lane; warpgroup 12-15 takes NREG_OTHER)
    # =====================================================================
    if tid >= UInt32(384)
        _fab_maxnreg_inc(Val(CFG.nreg[3]))
    end

    if tid == FAB_LOAD_TID
        # self-credit the ring and the Q buffer once: every load below
        # (including each work item's prologue) waits FREE first, so the
        # accounting is uniform across persistent iterations
        for s in 0:(FAB_KV_SLOTS - 1)
            barrier_arrive(mb + (FAB_BAR_KV_FREE + 8s))
        end
        barrier_arrive(mb + FAB_BAR_Q_FREE)
        ph0 = UInt32(0); ph1 = UInt32(0); ph2 = UInt32(0); ph3 = UInt32(0)
        q_free_ph = UInt32(0)

        lw = cta_id
        while lw < total_work
            q_row0 = _fab_q_row0(lw, seqlen, mb_shift, mb_mask)
            kv_row = (lw >> mb_shift) * seqlen
            krow = kv_row
            vrow = kv_row

            # prologue, ordered by first use: QK(stage 0) needs only
            # Q0+K0, so K0 goes ahead of the second Q tile
            _fab_wait(d, mb + FAB_BAR_Q_FREE, q_free_ph, Val(2))
            q_free_ph ⊻= UInt32(1)
            barrier_arrive_expect_tx(mb + FAB_BAR_Q, FAB_TILE_BYTES)
            _fab_load_tile(q_ptr, 0, tma_Q, q_row0, mb + FAB_BAR_Q)
            krow, ph0 = _fab_load_kv(d, Val(0), kv_ptr, tma_K, krow, mb, ph0)
            barrier_arrive_expect_tx(mb + (FAB_BAR_Q + 8), FAB_TILE_BYTES)
            _fab_load_tile(q_ptr, FAB_TILE_BYTES, tma_Q, q_row0 + UInt32(FAB_BM),
                           mb + (FAB_BAR_Q + 8))
            vrow, ph1 = _fab_load_kv(d, Val(1), kv_ptr, tma_V, vrow, mb, ph1)
            krow, ph2 = _fab_load_kv(d, Val(2), kv_ptr, tma_K, krow, mb, ph2)
            vrow, ph3 = _fab_load_kv(d, Val(3), kv_ptr, tma_V, vrow, mb, ph3)

            # steady state: 4 loads per trip into slots 0..3
            trips = (n_tiles - UInt32(2)) >> 1
            trip = UInt32(0)
            while trip < trips
                krow, ph0 = _fab_load_kv(d, Val(0), kv_ptr, tma_K, krow, mb, ph0)
                vrow, ph1 = _fab_load_kv(d, Val(1), kv_ptr, tma_V, vrow, mb, ph1)
                krow, ph2 = _fab_load_kv(d, Val(2), kv_ptr, tma_K, krow, mb, ph2)
                vrow, ph3 = _fab_load_kv(d, Val(3), kv_ptr, tma_V, vrow, mb, ph3)
                trip += UInt32(1)
            end
            lw += num_ctas
        end
    end

    # =====================================================================
    # MMA dispatch warp (single thread)
    # =====================================================================
    if tid == FAB_MMA_TID
        idesc_qk = tcgen05_instr_desc_f16bf16_f32(
            m = 128, n = FAB_BN, ab_dtype = :bf16, a_major = :K, b_major = :K)
        idesc_pv = tcgen05_instr_desc_f16bf16_f32(
            m = 128, n = FAB_HD, ab_dtype = :bf16, a_major = :K, b_major = :MN)
        # K-major operands: pyptx masked_descriptor ≡ leading 16 / stride
        # 1024 / B128 (see tcgen05_descriptor.jl's tested equivalence).
        dq0 = tcgen05_descriptor(smem_addr_u32(q_ptr);
                                 leading_bytes = 16, stride_bytes = 1024,
                                 swizzle = BlackwellLayout.B128)
        dk0 = tcgen05_descriptor(smem_addr_u32(kv_ptr);
                                 leading_bytes = 16, stride_bytes = 1024,
                                 swizzle = BlackwellLayout.B128)
        dv0 = tcgen05_descriptor(smem_addr_u32(kv_ptr) + UInt32(FAB_TILE_BYTES);
                                 leading_bytes = 16384, stride_bytes = 1024,
                                 swizzle = BlackwellLayout.B128)

        kph0 = UInt32(0); kph1 = UInt32(0); kph2 = UInt32(0); kph3 = UInt32(0)
        ph_qa = UInt32(0); ph_qb = UInt32(0)
        ph_pq0 = UInt32(0); ph_pq1 = UInt32(0)      # all 4 quarters flip together
        ph_or00 = UInt32(0); ph_or01 = UInt32(0)    # [stage][parity]
        ph_or10 = UInt32(0); ph_or11 = UInt32(0)

        mw = cta_id
        while mw < total_work
            # ── prologue: S(0) for stage 0 as soon as Q0+K0 land; the
            # second Q tile's TMA overlaps the first QK ──
            _fab_wait(d, mb + FAB_BAR_Q, ph_qa, Val(3)); ph_qa ⊻= UInt32(1)
            _fab_wait(d, mb + FAB_BAR_KV_FULL, kph0, Val(4)); kph0 ⊻= UInt32(1)
            _fab_qk_mma(Val(0), Val(0), tmem, dq0, dk0, idesc_qk, mb)
            _fab_wait(d, mb + (FAB_BAR_Q + 8), ph_qb, Val(3)); ph_qb ⊻= UInt32(1)
            _fab_qk_mma(Val(1), Val(0), tmem, dq0, dk0, idesc_qk, mb)
            _fab_commit(mb + FAB_BAR_KV_FREE)

            # ── peeled j=0: PV(0) with V slot 1, S(1) with K slot 2 ──
            _fab_wait(d, mb + (FAB_BAR_KV_FULL + 8), kph1, Val(4)); kph1 ⊻= UInt32(1)
            _fab_wait(d, mb + (FAB_BAR_KV_FULL + 16), kph2, Val(4)); kph2 ⊻= UInt32(1)
            ph_pq0, ph_or00 = _fab_pv_mma(d, Val(0), Val(0), true, Val(0),
                                          tmem, dv0, idesc_pv, mb, ph_pq0, ph_or00)
            _fab_qk_mma(Val(0), Val(1), tmem, dq0, dk0, idesc_qk, mb)
            ph_pq1, ph_or10 = _fab_pv_mma(d, Val(1), Val(0), true, Val(0),
                                          tmem, dv0, idesc_pv, mb, ph_pq1, ph_or10)
            _fab_qk_mma(Val(1), Val(1), tmem, dq0, dk0, idesc_qk, mb)
            _fab_commit(mb + (FAB_BAR_KV_FREE + 16))
            _fab_commit(mb + (FAB_BAR_KV_FREE + 8))

            # ── steady: j = 1 .. n_tiles-2, two iterations per trip. KV
            # waits are hoisted ahead of the PV/QK pair: KV normally
            # arrives early while P quarters are the tight resource, so
            # the QK UMMAs queue behind the PVs and keep the tensor core
            # fed through quarter stalls. ──
            trips = (n_tiles - UInt32(2)) >> 1
            trip = UInt32(0)
            while trip < trips
                # j odd: V in slot 3, next K in slot 0
                _fab_wait(d, mb + (FAB_BAR_KV_FULL + 24), kph3, Val(4)); kph3 ⊻= UInt32(1)
                _fab_wait(d, mb + FAB_BAR_KV_FULL, kph0, Val(4)); kph0 ⊻= UInt32(1)
                ph_pq0, ph_or01 = _fab_pv_mma(d, Val(0), Val(1), false, Val(1),
                                              tmem, dv0, idesc_pv, mb, ph_pq0, ph_or01)
                _fab_qk_mma(Val(0), Val(0), tmem, dq0, dk0, idesc_qk, mb)
                ph_pq1, ph_or11 = _fab_pv_mma(d, Val(1), Val(1), false, Val(1),
                                              tmem, dv0, idesc_pv, mb, ph_pq1, ph_or11)
                _fab_qk_mma(Val(1), Val(0), tmem, dq0, dk0, idesc_qk, mb)
                _fab_commit(mb + FAB_BAR_KV_FREE)
                _fab_commit(mb + (FAB_BAR_KV_FREE + 24))
                # j even: V in slot 1, next K in slot 2
                _fab_wait(d, mb + (FAB_BAR_KV_FULL + 8), kph1, Val(4)); kph1 ⊻= UInt32(1)
                _fab_wait(d, mb + (FAB_BAR_KV_FULL + 16), kph2, Val(4)); kph2 ⊻= UInt32(1)
                ph_pq0, ph_or00 = _fab_pv_mma(d, Val(0), Val(0), false, Val(0),
                                              tmem, dv0, idesc_pv, mb, ph_pq0, ph_or00)
                _fab_qk_mma(Val(0), Val(1), tmem, dq0, dk0, idesc_qk, mb)
                ph_pq1, ph_or10 = _fab_pv_mma(d, Val(1), Val(0), false, Val(0),
                                              tmem, dv0, idesc_pv, mb, ph_pq1, ph_or10)
                _fab_qk_mma(Val(1), Val(1), tmem, dq0, dk0, idesc_qk, mb)
                _fab_commit(mb + (FAB_BAR_KV_FREE + 16))
                _fab_commit(mb + (FAB_BAR_KV_FREE + 8))
                trip += UInt32(1)
            end

            # all QKs for this work item are issued: release the Q buffer
            # so the next item's Q TMA overlaps the final PV and the
            # epilogue
            _fab_commit(mb + FAB_BAR_Q_FREE)

            # ── peeled j = n_tiles-1 (odd): PV only, V in slot 3 ──
            _fab_wait(d, mb + (FAB_BAR_KV_FULL + 24), kph3, Val(4)); kph3 ⊻= UInt32(1)
            ph_pq0, ph_or01 = _fab_pv_mma(d, Val(0), Val(1), false, Val(1),
                                          tmem, dv0, idesc_pv, mb, ph_pq0, ph_or01)
            ph_pq1, ph_or11 = _fab_pv_mma(d, Val(1), Val(1), false, Val(1),
                                          tmem, dv0, idesc_pv, mb, ph_pq1, ph_or11)
            _fab_commit(mb + (FAB_BAR_KV_FREE + 24))
            _fab_commit(mb + FAB_BAR_O_DONE)

            mw += num_ctas
        end

        # tail padding: correction's PV_DONE parity waits may lag the
        # completion stream while idling through skip tiles; extra
        # completions per stage guarantee every pending parity test still
        # sees a phase flip after the last real PV
        for _ in 1:8
            _fab_commit(mb + FAB_BAR_PV_DONE)
            _fab_commit(mb + (FAB_BAR_PV_DONE + 8))
        end
    end

    # =====================================================================
    # Softmax: two stage-parallel warpgroups (warps 0-3 → stage 0, warps
    # 4-7 → stage 1). Thread t owns row t&127 and streams its 128 columns
    # in four 32-column chunks, ping-ponging two register buffers so each
    # chunk's TMEM load flies under the previous chunk's exp.
    # =====================================================================
    if tid < UInt32(256)
        _fab_maxnreg_inc(Val(CFG.nreg[1]))
        stage = (tid & UInt32(128)) >> UInt32(7)

        sp_base = tmem + lane_bits + (stage << UInt32(7))
        sfull   = mb + FAB_BAR_S_FULL + 8 * Int(stage)
        sfree   = mb + FAB_BAR_STATS_FREE + 8 * Int(stage)
        pq_ptr  = mb + FAB_BAR_P_Q + 32 * Int(stage)
        oresc   = mb + FAB_BAR_O_RESC + 16 * Int(stage)
        stats_bar = UInt32(1) + wig + (stage << UInt32(2))  # named bar 1+w / 5+w
        alpha_idx = Int(stage) * 128 + Int(row128) + 1
        sum_idx   = alpha_idx + 256

        ph_s = UInt32(0); ph_stats = UInt32(0); epi_ph = UInt32(0)
        pmax = 0.0f0; sums = 0.0f0

        sw = cta_id
        while sw < total_work
            # gate the item start on the previous epilogue's O readout
            _fab_wait(d, mb + FAB_BAR_EPI, epi_ph, Val(9))
            epi_ph ⊻= UInt32(1)

            mneg = 0.0f0
            l_run = 0.0f0
            ph_s, ph_stats, mneg, pmax, sums, l_run = _fab_softmax_body(
                d, sb, true, oresc, sp_base, sfull, sfree, pq_ptr, stats, alpha_idx,
                stats_bar, qk_scale, ph_s, ph_stats, mneg, pmax, sums, l_run)
            it = UInt32(1)
            while it < n_tiles
                po = (it & UInt32(1)) << UInt32(3)
                ph_s, ph_stats, mneg, pmax, sums, l_run = _fab_softmax_body(
                    d, sb, false, oresc + Int(po), sp_base, sfull, sfree, pq_ptr,
                    stats, alpha_idx, stats_bar, qk_scale,
                    ph_s, ph_stats, mneg, pmax, sums, l_run)
                it += UInt32(1)
            end

            # drain: fold the last tile's sums into l (its basis decision
            # never runs — the pending rescale cancels between O and l in
            # the epilogue normalize), publish the row sum,
            # STATS_FREE-gated like every body publish.
            l_run += sums
            @inbounds stats[sum_idx] = l_run
            _fab_wait(d, sfree, ph_stats, Val(8))
            ph_stats ⊻= UInt32(1)
            ptx"bar.arrive"(stats_bar, UInt32(64))

            sw += num_ctas
        end
    end

    # =====================================================================
    # Correction warps (tids 256-383): O rescale + epilogue
    # =====================================================================
    if UInt32(256) <= tid < UInt32(384)
        _fab_maxnreg_dec(Val(CFG.nreg[2]))
        stats_bar0 = UInt32(1) + wig
        stats_bar1 = UInt32(5) + wig
        alpha_idx0 = Int(row128) + 1
        alpha_idx1 = 128 + Int(row128) + 1
        o_addr0 = tmem + UInt32(256) + lane_bits
        o_addr1 = tmem + UInt32(384) + lane_bits

        # first KV tile needs no correction: release both stages once
        # (parity-0 barrier gates PV(0)); also pre-arrive STATS_FREE so
        # the softmax's pre-write wait at tile 1 is already satisfied
        barrier_arrive(mb + FAB_BAR_O_RESC)
        barrier_arrive(mb + (FAB_BAR_O_RESC + 16))
        barrier_arrive(mb + FAB_BAR_STATS_FREE)
        barrier_arrive(mb + (FAB_BAR_STATS_FREE + 8))
        barrier_arrive(mb + FAB_BAR_EPI)

        ph_pv0 = UInt32(0); ph_pv1 = UInt32(0); done_ph = UInt32(0)
        po = pointer(O)

        cw = cta_id
        while cw < total_work
            # alpha published at tile i (i = 0..n_tiles-2; the last tile
            # publishes nothing) rescales O between PV(i) and PV(i+1)
            it = UInt32(0)
            while it < n_tiles - UInt32(1)
                ponc = ((it + UInt32(1)) & UInt32(1)) << UInt32(3)
                ph_pv0 = _fab_corr_tile(d, Val(0), mb, stats, alpha_idx0,
                                        stats_bar0, o_addr0, ponc, ph_pv0)
                ph_pv1 = _fab_corr_tile(d, Val(1), mb, stats, alpha_idx1,
                                        stats_bar1, o_addr1, ponc, ph_pv1)
                it += UInt32(1)
            end

            # drain: consume the last tile's PV_DONE completion so the
            # per-work-item wait/completion counts stay exactly balanced
            _fab_wait(d, mb + FAB_BAR_PV_DONE, ph_pv0, Val(10)); ph_pv0 ⊻= UInt32(1)
            _fab_wait(d, mb + (FAB_BAR_PV_DONE + 8), ph_pv1, Val(10)); ph_pv1 ⊻= UInt32(1)

            # ── epilogue: wait all PV done, normalize, store bf16 ──
            _fab_wait(d, mb + FAB_BAR_O_DONE, done_ph, Val(11)); done_ph ⊻= UInt32(1)
            ptx"tcgen05.fence::after_thread_sync"()

            out_row = _fab_q_row0(cw, seqlen, mb_shift, mb_mask) + row128
            _fab_epi_stage(Val(0), mb, stats, alpha_idx0, stats_bar0,
                           o_addr0, out_row, po)
            _fab_epi_stage(Val(1), mb, stats, alpha_idx1, stats_bar1,
                           o_addr1, out_row, po)

            cw += num_ctas
        end
    end

    ptx"bar.sync"(Val(0))
    if warp == UInt32(12)
        ptx"tcgen05.dealloc.cta_group::1.sync.aligned.b32"(tmem, UInt32(512))
        ptx"tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"()
    end
    return nothing
end


# ── host-side helpers (shared by the tests and the bench workbench) ─────

# f32 reference with bf16-quantized inputs, per (batch*head) slice.
function _fab_cpu_ref(Q::Matrix{Float32}, K::Matrix{Float32},
                      V::Matrix{Float32}, sm_scale::Float32)
    Qb = bf16_to_f32.(bf16_bits.(Q))
    Kb = bf16_to_f32.(bf16_bits.(K))
    Vb = bf16_to_f32.(bf16_bits.(V))
    scores = (Qb * Kb') .* sm_scale
    rmax = maximum(scores; dims = 2)
    ex = exp.(scores .- rmax)
    return (ex ./ sum(ex; dims = 2)) * Vb
end

# (rows, 128) f32 → bf16 bits, row-major (HD-fast) for TMA/global IO.
function _fab_pack(X::Matrix{Float32})
    rows, hd = size(X)
    out = Array{UInt16}(undef, hd, rows)
    for r in 1:rows, h in 1:hd
        @inbounds out[h, r] = bf16_bits(X[r, h])
    end
    out
end

# Decode a beacon buffer (one UInt32 per tid; 0 = never timed out).
function fab_decode_beacons(dbg::Vector{UInt32})
    hits = [(tid = t - 1, site = Int((v >> 8) & 0xff), phase = Int(v & 0xff))
            for (t, v) in enumerate(dbg) if v != 0]
    isempty(hits) && return nothing
    by_site = Dict{String, Vector{Int}}()
    for h in hits
        push!(get!(by_site, get(FAB_SITES, h.site, "site$(h.site)"), Int[]), h.tid)
    end
    by_site
end
