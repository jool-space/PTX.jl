# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc==10
# Blackwell tcgen05 FlashAttention forward — correctness tests for the
# kernel defined in flash_attention_defs.jl (see its header for the
# architecture and the compile-time config). Tests run the DEFAULT config
# (B200-validated 2026-07-12); the perf/debug workbench that sweeps
# configs lives in bench/flash_attention.jl.

include("flash_attention_defs.jl")

# ── Cross-arch ptxas validation ─────────────────────────────────────────

@testset "flash_attention_blackwell compiles at sm_100a" begin
    types(cfg) = Tuple{CuDeviceVector{UInt16, 1},
                       PTX.TMADescriptorPtr, PTX.TMADescriptorPtr, PTX.TMADescriptorPtr,
                       UInt32, UInt32, UInt32, UInt32, UInt32, Float32,
                       CuDeviceVector{UInt32, 1}, typeof(Val(cfg))}
    default = types(FAB_CFG_DEFAULT)
    # minthreads → .reqntid 512: without it ptxas cannot determine the
    # entry register count, IGNORES every setmaxnreg (C7508), and records
    # a flat 168 registers — which can't launch at 512 threads. With it
    # the kernel records 65536/512 = 128 and repartitions at runtime.
    @test ptxas_compiles(_fab_kernel!, default; cap = v"10.0", feature_set = :arch,
                         minthreads = 512)
    # Blackwell Ultra (CC 10.3) and the family target: sm_100a cubins load
    # ONLY on CC 10.0, so 10.3 devices recompile at sm_103a (the runtime
    # @cuda path auto-selects this); sm_100f is the one-cubin-per-family
    # alternative. Both must stay compilable.
    @test ptxas_compiles(_fab_kernel!, default; cap = v"10.3", feature_set = :arch,
                         minthreads = 512)
    @test ptxas_compiles(_fab_kernel!, default; cap = v"10.0", feature_set = :family,
                         minthreads = 512)

    ptx = emit_ptx(_fab_kernel!, default; cap = v"10.0", feature_set = :arch,
                   minthreads = 512)
    @test occursin(".reqntid 512, 1, 1", ptx)
    # QK: SMEM-descriptor A (i64 operand, no brackets)
    @test occursin(r"tcgen05\.mma\.cta_group::1\.kind::f16\.collector::a::discard \[%r\d+\], %rd\d+, %rd\d+", ptx)
    # PV: TMEM A (bracketed register operand) — the mma.tensor path
    @test occursin(r"tcgen05\.mma\.cta_group::1\.kind::f16\.collector::a::discard \[%r\d+\], \[%r\d+\], %rd\d+", ptx)
    @test occursin("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes", ptx)
    @test occursin("tcgen05.ld.sync.aligned.32x32b.x32.b32", ptx)
    @test occursin("tcgen05.ld.sync.aligned.32x32b.x64.b32", ptx)
    @test occursin("tcgen05.st.sync.aligned.32x32b.x16.b32", ptx)
    @test occursin("tcgen05.st.sync.aligned.32x32b.x64.b32", ptx)
    @test occursin("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64", ptx)
    @test occursin("tcgen05.fence::before_thread_sync", ptx)
    @test occursin("tcgen05.fence::after_thread_sync", ptx)
    @test occursin("mbarrier.try_wait.parity.shared.b64", ptx)
    @test occursin("mbarrier.arrive.expect_tx.shared.b64", ptx)
    # Default split (144, 80, 144): softmax and mma/load warpgroups both
    # raise to 144, the producer drops to 80.
    @test occursin("setmaxnreg.inc.sync.aligned.u32 144", ptx)
    @test occursin("setmaxnreg.dec.sync.aligned.u32 80", ptx)
    @test !occursin("setmaxnreg.dec.sync.aligned.u32 96", ptx)
    @test occursin("vote.sync.ballot.b32", ptx)
    @test occursin("bar.arrive", ptx)
    @test occursin("ex2.approx.ftz.f32", ptx)
    @test occursin("lg2.approx.f32", ptx)
    @test occursin("rcp.approx.f32", ptx)
    @test occursin("cvt.rn.bf16x2.f32", ptx)
    @test occursin("st.global.v4.b32", ptx)

    # config variants: the workbench's knobs must stay compilable, and
    # each must actually change the emitted PTX the way it claims.
    sb_cfg = fab_check_cfg((scoreboard = true, beacon = false,
                            nreg = (144, 80, 144), emu = ()))
    @test ptxas_compiles(_fab_kernel!, types(sb_cfg);
                         cap = v"10.0", feature_set = :arch, minthreads = 512)
    sb_ptx = emit_ptx(_fab_kernel!, types(sb_cfg);
                      cap = v"10.0", feature_set = :arch, minthreads = 512)
    # scoreboard mode drops the softmax stream's wait::ld (4 sites);
    # correction/epilogue keep theirs, so count strictly between 0 and
    # the default's
    n_waits(p) = length(collect(eachmatch(r"tcgen05\.wait::ld", p)))
    @test 0 < n_waits(sb_ptx) < n_waits(ptx)

    bc_cfg = fab_check_cfg((scoreboard = false, beacon = true,
                            nreg = (144, 80, 144), emu = ()))
    @test ptxas_compiles(_fab_kernel!, types(bc_cfg);
                         cap = v"10.0", feature_set = :arch, minthreads = 512)

    nr_cfg = fab_check_cfg((scoreboard = false, beacon = false,
                            nreg = (144, 88, 128), emu = ()))
    nr_ptx = emit_ptx(_fab_kernel!, types(nr_cfg);
                      cap = v"10.0", feature_set = :arch, minthreads = 512)
    @test occursin("setmaxnreg.inc.sync.aligned.u32 144", nr_ptx)
    @test occursin("setmaxnreg.dec.sync.aligned.u32 88", nr_ptx)

    # emu: pyptx's headline pair set moves 8 of 16 exp2 pairs per chunk
    # onto the packed-f32 FMA pipe; the other 8 stay on the SFU. The
    # default config must emit NO f32x2 arithmetic.
    emu_cfg = fab_check_cfg((scoreboard = false, beacon = false,
                             nreg = (144, 80, 144), emu = (1, 3, 5, 7)))
    @test ptxas_compiles(_fab_kernel!, types(emu_cfg);
                         cap = v"10.0", feature_set = :arch, minthreads = 512)
    emu_ptx = emit_ptx(_fab_kernel!, types(emu_cfg);
                       cap = v"10.0", feature_set = :arch, minthreads = 512)
    @test occursin("fma.rn.ftz.f32x2", emu_ptx)
    @test occursin("add.rm.ftz.f32x2", emu_ptx)
    @test occursin("sub.rn.ftz.f32x2", emu_ptx)
    @test occursin("max.ftz.f32", emu_ptx)
    @test occursin("ex2.approx.ftz.f32", emu_ptx)
    @test !occursin("f32x2", ptx)
end

# ── Runtime — datacenter Blackwell only (banner: cc==10) ────────────────
# (see tcgen05_smoke.jl rationale; the GB10 CI runner is CC 12.1)

if test_runtime_supported(@__FILE__)
    function _run_fab(B, H, S; input_scale = 0.5f0, atol = 5e-2)
        @assert S % (FAB_QSTAGE * FAB_BM) == 0
        n_tiles = S ÷ FAB_BN
        @assert n_tiles % 2 == 0 && n_tiles >= 2
        n_mblocks = S ÷ (FAB_QSTAGE * FAB_BM)
        @assert ispow2(n_mblocks)
        mb_shift = trailing_zeros(n_mblocks)
        bh = B * H
        total_work = n_mblocks * bh
        # persistent wave: one CTA per SM (each CTA takes the whole SM —
        # 512 threads, ~194 KiB SMEM, 512 TMEM columns). Queried, not
        # hardcoded: pyptx's num_sms=148 is B200-specific (sm_103 Ultra
        # parts have more SMs and would underfill).
        n_sms = CUDACore.attribute(CUDACore.device(),
                                   CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        num_ctas = min(total_work, n_sms)
        total_rows = bh * S
        sm_scale = 1.0f0 / sqrt(Float32(FAB_HD))

        rng = MersenneTwister(B * 7919 + H * 131 + S)
        Q = Float32.(randn(rng, total_rows, FAB_HD)) .* input_scale
        K = Float32.(randn(rng, total_rows, FAB_HD)) .* input_scale
        V = Float32.(randn(rng, total_rows, FAB_HD))

        Q_d = CuArray(_fab_pack(Q))
        K_d = CuArray(_fab_pack(K))
        V_d = CuArray(_fab_pack(V))
        tmaps = map((Q_d, K_d, V_d)) do X_d
            tensor_map_tile_2d(:bf16, pointer(X_d), total_rows, FAB_HD,
                               FAB_BM, 64; swizzle = :B128)
        end
        q_const = upload_tma_descriptor(tmaps[1])
        k_const = upload_tma_descriptor(tmaps[2])
        v_const = upload_tma_descriptor(tmaps[3])

        O_d = CUDACore.zeros(UInt16, total_rows * FAB_HD)
        dbg_d = CUDACore.zeros(UInt32, FAB_THREADS)
        args = (O_d, q_const.ptr, k_const.ptr, v_const.ptr,
                UInt32(S), UInt32(mb_shift), UInt32(n_mblocks - 1),
                UInt32(n_tiles), UInt32(total_work),
                sm_scale * Float32(FAB_LOG2E),
                dbg_d, Val(FAB_CFG_DEFAULT))
        kern = @cuda launch=false minthreads=512 _fab_kernel!(args...)
        attrs = CUDACore.attributes(kern.fun)
        attrs[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = FAB_SMEM_BYTES
        kern(args...; blocks = (num_ctas, 1, 1), threads = FAB_THREADS,
             shmem = FAB_SMEM_BYTES)
        CUDACore.synchronize()

        O_packed = reshape(Array(O_d), FAB_HD, total_rows)   # (HD, rows)
        maxdiff = 0.0f0
        for i in 1:bh
            r = ((i - 1) * S + 1):(i * S)
            O_got = permutedims(bf16_to_f32.(O_packed[:, r]))  # (S, HD)
            O_ref = _fab_cpu_ref(Q[r, :], K[r, :], V[r, :], sm_scale)
            maxdiff = max(maxdiff, maximum(abs.(O_got - O_ref)))
        end
        @info "flash_attention_blackwell" B H S maxdiff
        maxdiff < atol
    end

    @testset "FA forward B=$B H=$H S=$S" for (B, H, S) in [
            (1, 1, 256),     # 1 work item, n_tiles=2 (steady loops trip 0×)
            (1, 2, 512),     # 4 work items
            (2, 4, 1024),    # 32 work items — persistent multi-item per CTA
        ]
        @test _run_fab(B, H, S)
    end

    # At input_scale 0.5 the running row max NEVER drifts past
    # 2^RESCALE_THRESHOLD — CPU simulation of the stale-basis state machine
    # shows 0 rescale events across all shapes above, i.e. the correction
    # warps' TMEM rescale branch is dead code there. At 2.0 (same seed)
    # every row rescales at least once (1055 events) while only 58 of 96
    # warp-bands fire per tile — softmax skip-arrivals and correction
    # rescale-arrivals MIX within the same O_RESC phase, stressing the
    # exact-arrival-count invariant. Max per-tile row sum ~3e7: ample f32
    # headroom (scale 3 would push 4e16 — gratuitous).
    @testset "FA forward rescale path B=1 H=2 S=512 scale=2" begin
        @test _run_fab(1, 2, 512; input_scale = 2.0f0)
    end
end
