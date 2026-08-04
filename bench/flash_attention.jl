# Blackwell FlashAttention workbench — perf iteration and hang forensics
# for the kernel in test/gpu/blackwell/flash_attention_defs.jl. NOT part
# of the test suite; needs a CC 10.x device.
#
# Script mode (standard sweep + variant table):
#     julia --project=test bench/flash_attention.jl
#
# REPL mode (the iteration loop — edit defs, re-include, re-measure):
#     include("bench/flash_attention.jl")
#     fab_check(cfg)                     # correctness gate, rescale-heavy input
#     fab_time(cfg; H = 16, S = 4096)    # → (; ms, tflops)
#     fab_check((; FAB_CFG_DEFAULT..., emu = (1, 3, 5, 7)))  # emu sweep entry
#     fab_check((; FAB_CFG_DEFAULT..., splitp = false))  # quarter-granular P
#                                        # (the pre-split-P protocol)
#     fab_sweep()                        # default vs scoreboard, all shapes
#     fab_hang_debug(cfg; H = 2, S = 512)  # beacon run + site decode
#
# Configs are `(scoreboard = ..., beacon = ..., nreg = (...), emu = (...),
# splitp = ...)` NamedTuples
# (see the defs header); each distinct config compiles a fresh kernel
# specialization (~15 s first call).
#
# Measurement discipline (hardware-derived, both chips):
#   - The headline is the SATURATED shape: B*H*(S/256) well above the
#     148-CTA fill, e.g. B=8 H=8 S=8192. B=1 H=8 S=8192 is the
#     latency-sensitive case, and its distance from saturated is
#     chip-dependent — the two are not interchangeable.
#   - Cross-box TF deltas under ~8% are noise unless the configs were
#     A/B'd on the same box; cross-CHIP comparisons (B200-class
#     references vs B300 numbers) are not comparisons at all.

using PTX, CUDACore, Test, Random

include(joinpath(@__DIR__, "..", "test", "setup.jl"))
include(joinpath(@__DIR__, "..", "test", "gpu", "blackwell",
                 "flash_attention_defs.jl"))

# One prepared problem: device buffers + TMA descriptors + launch closure.
# Keep the returned NamedTuple alive for the duration of all launches (it
# owns the TMA descriptor blobs and device arrays).
function fab_setup(B, H, S; input_scale = 0.5f0, cfg = FAB_CFG_DEFAULT)
    fab_check_cfg(cfg)
    @assert S % (FAB_QSTAGE * FAB_BM) == 0
    n_tiles = S ÷ FAB_BN
    @assert n_tiles % 2 == 0 && n_tiles >= 2
    n_mblocks = S ÷ (FAB_QSTAGE * FAB_BM)
    @assert ispow2(n_mblocks)
    bh = B * H
    total_work = n_mblocks * bh
    n_sms = CUDACore.attribute(CUDACore.device(),
                               CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    num_ctas = min(total_work, n_sms)
    total_rows = bh * S
    sm_scale = 1.0f0 / sqrt(Float32(FAB_HD))

    rng = MersenneTwister(B * 7919 + H * 131 + S)
    Q = Float32.(randn(rng, total_rows, FAB_HD)) .* input_scale
    K = Float32.(randn(rng, total_rows, FAB_HD)) .* input_scale
    V = Float32.(randn(rng, total_rows, FAB_HD))

    Q_d = CuArray(fab_pack(Q)); K_d = CuArray(fab_pack(K))
    V_d = CuArray(fab_pack(V))
    tmaps = map((Q_d, K_d, V_d)) do X_d
        tensor_map_tile_2d(:bf16, pointer(X_d), total_rows, FAB_HD,
                           FAB_BM, 64; swizzle = :B128)
    end
    q_c = upload_tma_descriptor(tmaps[1])
    k_c = upload_tma_descriptor(tmaps[2])
    v_c = upload_tma_descriptor(tmaps[3])

    O_d = CUDACore.zeros(UInt16, total_rows * FAB_HD)
    dbg_d = CUDACore.zeros(UInt32, FAB_THREADS)
    args = (O_d, q_c.ptr, k_c.ptr, v_c.ptr,
            UInt32(S), UInt32(trailing_zeros(n_mblocks)), UInt32(n_mblocks - 1),
            UInt32(n_tiles), UInt32(total_work),
            sm_scale * Float32(FAB_LOG2E),
            dbg_d, Val(cfg))
    kern = @cuda launch=false minthreads=512 fab_kernel!(args...)
    attrs = CUDACore.attributes(kern.fun)
    attrs[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = FAB_SMEM_BYTES
    launch! = () -> kern(args...; blocks = (num_ctas, 1, 1),
                         threads = FAB_THREADS, shmem = FAB_SMEM_BYTES)
    (; launch!, O_d, dbg_d, Q, K, V, sm_scale, B, H, S, num_ctas,
       keepalive = (Q_d, K_d, V_d, q_c, k_c, v_c))
end

# Correctness gate. Defaults to the rescale-heavy input (scale 2.0) so a
# variant that breaks the correction path fails HERE, not in a timing run.
function fab_check(cfg = FAB_CFG_DEFAULT; B = 1, H = 2, S = 512,
                   input_scale = 2.0f0, atol = 5e-2)
    p = fab_setup(B, H, S; input_scale, cfg)
    p.launch!(); CUDACore.synchronize()
    O_packed = reshape(Array(p.O_d), FAB_HD, :)
    maxdiff = 0.0f0
    for i in 1:(B * H)
        r = ((i - 1) * S + 1):(i * S)
        O_got = permutedims(bf16_to_f32.(O_packed[:, r]))
        O_ref = fab_cpu_ref(p.Q[r, :], p.K[r, :], p.V[r, :], p.sm_scale)
        maxdiff = max(maxdiff, maximum(abs.(O_got - O_ref)))
    end
    ok = maxdiff < atol
    println("check $(cfg): B=$B H=$H S=$S scale=$input_scale ",
            "maxdiff=$maxdiff ", ok ? "OK" : "FAIL")
    ok
end

# Wall-clock timing. Forward-attention FLOPs = 4·bh·S²·D (2 matmuls of
# 2·S²·D each); softmax flops excluded by convention (matches FA papers).
function fab_time(cfg = FAB_CFG_DEFAULT; B = 1, H = 16, S = 4096,
                  input_scale = 0.5f0, iters = 20, warmup = 3,
                  warm_ms = 25.0)
    p = fab_setup(B, H, S; input_scale, cfg)
    # Warm the DEVICE, not just the code path. After the GPU idles (e.g.
    # through fab_check's CPU reference, or a REPL pause) clocks drop to an
    # idle P-state, and a few-launch warmup (~70 µs of work) does not ramp
    # them — the whole timed batch then executes at idle clocks and reads
    # ~40x slow (observed on B200: the sweep's first cell showed 7 TF for a
    # 380 TF shape, reproducibly, because it is always the first batch after
    # an idle gap). Spin launches for at least `warm_ms` of wall time, with
    # `warmup` as the minimum launch count.
    t_warm = time_ns()
    launches = 0
    while launches < warmup || (time_ns() - t_warm) < warm_ms * 1e6
        p.launch!()
        launches += 1
    end
    CUDACore.synchronize()
    t0 = time_ns()
    for _ in 1:iters
        p.launch!()
    end
    CUDACore.synchronize()
    ms = (time_ns() - t0) / 1e6 / iters
    flops = 4.0 * B * H * S^2 * FAB_HD
    (; ms, tflops = flops / (ms * 1e9), ctas = p.num_ctas)
end

# The standard sweep: shapes sized so total_work fills the SMs (S/256
# work items per bh); default vs scoreboard side by side.
function fab_sweep(; cfgs = [
        "default"    => FAB_CFG_DEFAULT,
        "scoreboard" => (scoreboard = true, beacon = false,
                         nreg = (144, 80, 144), emu = (), splitp = true),
    ], shapes = [(1, 16, 1024), (1, 16, 2048), (1, 16, 4096), (1, 8, 8192)])
    for (name, cfg) in cfgs
        fab_check(cfg) || (println("  skipping $name (failed check)"); continue)
    end
    println()
    println(rpad("shape", 22), join((rpad(n, 22) for (n, _) in cfgs)))
    for (B, H, S) in shapes
        row = rpad("B=$B H=$H S=$S", 22)
        for (_, cfg) in cfgs
            r = fab_time(cfg; B, H, S)
            row *= rpad("$(round(r.ms; digits=3)) ms  $(round(r.tflops; digits=1)) TF", 22)
        end
        println(row)
    end
end

# Hang forensics: run with the beacon config; if any wait timed out, the
# kernel drains instead of hanging and this decodes who blamed whom first.
# A healthy run prints "no beacons".
function fab_hang_debug(cfg = FAB_CFG_DEFAULT; B = 1, H = 2, S = 512,
                        input_scale = 2.0f0)
    bcfg = (scoreboard = cfg.scoreboard, beacon = true, nreg = cfg.nreg,
            emu = cfg.emu, splitp = cfg.splitp)
    p = fab_setup(B, H, S; input_scale, cfg = bcfg)
    p.launch!(); CUDACore.synchronize()
    hits = fab_decode_beacons(Array(p.dbg_d))
    if hits === nothing
        println("no beacons — every wait completed within the bound")
    else
        for (site, tids) in sort(collect(hits); by = first)
            println(rpad(site, 22), length(tids), " threads, e.g. tid ",
                    join(first(tids, 8), ","))
        end
    end
    hits
end

function main()
    cap = CUDACore.capability(CUDACore.device())
    if !(v"10.0" <= cap < v"11.0")
        @info "flash_attention workbench needs a datacenter-Blackwell device" cap
        return
    end
    @info "device" name=CUDACore.name(CUDACore.device()) cap
    fab_sweep()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
