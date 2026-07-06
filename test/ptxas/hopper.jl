# Hopper (sm_90a) wrapper validation — compile through ptxas without
# launching. Runs on any CUDA-functional box; the device's actual
# capability does not need to match (the cubin is never loaded).
#
# Each testset asserts two things:
#   1. `ptxas_compiles(...)` — the wrapper's inline-asm body, constraint
#      string, register classes, and operand encoding are all accepted by
#      ptxas as valid sm_90a PTX. This is the real correctness check.
#   2. `occursin(...)` smoke checks on the LLVM-emitted PTX text — confirm
#      the wrapper resolved to the expected opcode/shape/dtype, not just
#      that some valid PTX came out.
#
# These tests previously lived in gpu/exec.jl as compile-only string-match
# blocks; ptxas never ran. A typo in any wrapper opcode (e.g. wrong shape
# or illegal modifier combo) used to slip through silently — ptxas catches
# all of that.


# --- wgmma compile-only -----------------------------------------------------
#
# wgmma.mma_async.sync.aligned with various accumulator/operand dtypes.
# Validates: tied-operand constraint string, d-vec brace syntax, immediates
# (scaleD, scaleA, transA, transB) baked into the asm tail.

function _wgmma_compile_bf16!(out, a_desc::UInt64, b_desc::UInt64)
    d = ntuple(_ -> 0f0, Val(4))
    d_new = ptx"wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16"(
        d, a_desc, b_desc, false)
    @inbounds out[1] = d_new[1]
    return nothing
end

function _wgmma_compile_f16acc!(out, a_desc::UInt64, b_desc::UInt64)
    d = ntuple(_ -> UInt32(0), Val(4))
    d_new = ptx"wgmma.mma_async.sync.aligned.m64n16k16.f16.f16.f16"(
        d, a_desc, b_desc, true)
    @inbounds out[1] = d_new[1]
    return nothing
end

function _wgmma_compile_e4m3!(out, a_desc::UInt64, b_desc::UInt64)
    d = ntuple(_ -> 0f0, Val(8))
    d_new = ptx"wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3"(
        d, a_desc, b_desc, false)
    @inbounds out[1] = d_new[1]
    return nothing
end

@testset "wgmma m64n8k16 f32.bf16.bf16" begin
    types = Tuple{CuDeviceVector{Float32, 1}, UInt64, UInt64}
    @test ptxas_compiles(_wgmma_compile_bf16!, types;
                         cap = v"9.0", feature_set = :arch)
    ptx = emit_ptx(_wgmma_compile_bf16!, types;
                   cap = v"9.0", feature_set = :arch)
    @test occursin("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16", ptx)
    # 4 f32 d-regs in brace (%f, or untyped b32 %r on newer NVPTX backends).
    @test occursin(r"wgmma\.mma_async[^;]*\{%[fr]\d+, %[fr]\d+, %[fr]\d+, %[fr]\d+\}", ptx)
    # Imms baked in (scaleD=1, scaleA=1, transA=0, transB=0).
    @test occursin("1, 1, 0, 0;", ptx)
end

@testset "wgmma m64n16k16 f16.f16.f16" begin
    types = Tuple{CuDeviceVector{UInt32, 1}, UInt64, UInt64}
    @test ptxas_compiles(_wgmma_compile_f16acc!, types;
                         cap = v"9.0", feature_set = :arch)
    ptx = emit_ptx(_wgmma_compile_f16acc!, types;
                   cap = v"9.0", feature_set = :arch)
    @test occursin("wgmma.mma_async.sync.aligned.m64n16k16.f16.f16.f16", ptx)
    @test occursin(r"\{%r\d+, %r\d+, %r\d+, %r\d+\}", ptx)
end

@testset "wgmma m64n16k32 f32.e4m3.e4m3" begin
    types = Tuple{CuDeviceVector{Float32, 1}, UInt64, UInt64}
    @test ptxas_compiles(_wgmma_compile_e4m3!, types;
                         cap = v"9.0", feature_set = :arch)
    ptx = emit_ptx(_wgmma_compile_e4m3!, types;
                   cap = v"9.0", feature_set = :arch)
    @test occursin("wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3", ptx)
    # 8 d-regs in brace (%f, or untyped b32 %r on newer NVPTX backends).
    @test occursin(r"e4m3\.e4m3 \{%[fr]\d+(?:, %[fr]\d+){7}\}", ptx)
    # FP8 has no transA/transB imms — only scaleD, scaleA.
    @test occursin(", 1, 1;", ptx)
end


# --- WGMMA parametric sweep across _WGMMA_VARIANTS × {N=8, N=256} ---------
#
# Mints one compile-only kernel per (dtype_d, dtype_a, dtype_b, k) variant
# in `PTX._WGMMA_VARIANTS` at both ends of the N grid (8 and 256). Catches
# brace-count, tied-operand, and dispatch regressions across the full
# registered surface that the three hand-rolled testsets above don't reach.
#
# Encoder cross-reference: pyptx's `_Wgmma.mma_async()`
# (pyptx/pyptx/ptx.py:817-910). Verbatim corpus reference for
# m64n256k16.f32.e4m3.e4m3:
# pyptx/tests/corpus/wgmma_simple.ptx:10. The Hopper full pipeline below
# anchors the m64n8k16.f32.bf16.bf16 shape end-to-end.

const _WGMMA_SWEEP_NS = (8, 256)

# Emit one kernel per variant × N. Function name encodes the combo so
# ptxas error messages identify the failing variant.
for (dt_d, dt_a, dt_b, k, _has_trans) in PTX._WGMMA_VARIANTS,
        n in _WGMMA_SWEEP_NS
    nd    = dt_d === :f16 ? n >>> 2 : n >>> 1
    acc_J = dt_d === :f32 ? Float32 :
            dt_d === :f16 ? UInt32  : Int32
    shape = Symbol("m64n", n, "k", k)
    mods  = (:mma_async, :sync, :aligned, shape, dt_d, dt_a, dt_b)
    fname = Symbol("_wgmma_sweep_", dt_d, "_", dt_a, "_", dt_b, "_n", n, "!")

    @eval function $fname(out::CuDeviceVector{$acc_J, 1},
                          a_desc::UInt64, b_desc::UInt64)
        d = ntuple(_ -> zero($acc_J), Val($nd))
        d_new = $(PTX.Operation{:wgmma, mods}())(d, a_desc, b_desc, false)
        @inbounds out[1] = d_new[1]
        return nothing
    end
end

@testset "wgmma sweep $(dt_d).$(dt_a).$(dt_b) m64n$(n)k$(k)" for
        (dt_d, dt_a, dt_b, k, _has_trans) in PTX._WGMMA_VARIANTS,
        n in _WGMMA_SWEEP_NS

    acc_J = dt_d === :f32 ? Float32 :
            dt_d === :f16 ? UInt32  : Int32
    fname = Symbol("_wgmma_sweep_", dt_d, "_", dt_a, "_", dt_b, "_n", n, "!")
    f     = getfield(@__MODULE__, fname)
    types = Tuple{CuDeviceVector{acc_J, 1}, UInt64, UInt64}

    @test ptxas_compiles(f, types; cap = v"9.0", feature_set = :arch)
    ptx = emit_ptx(f, types; cap = v"9.0", feature_set = :arch)
    @test occursin(
        "wgmma.mma_async.sync.aligned.m64n$(n)k$(k).$dt_d.$dt_a.$dt_b", ptx)
end


# --- Hopper full pipeline ---------------------------------------------------
#
# End-to-end Hopper kernel: cp.async load → fence.proxy.async →
# wgmma.fence/mma_async/commit_group/wait_group → store. Exercises the
# canonical Hopper warpgroup-MMA pipeline. We can't launch this on Ada,
# but ptxas validates that every step lowers to a sm_90a-acceptable PTX.

function _hopper_pipeline_kernel!(D, A_bits, B_bits)
    smem_A = CuStaticSharedArray(UInt16, 1024)        # 64×16 bf16 (m64k16)
    smem_B = CuStaticSharedArray(UInt16, 128)         # 16×8 bf16 (k16n8)
    tid = ptx"mov.u32"(sreg"tid.x")

    # 1. cp.async load (cooperative, 16-byte chunks).
    src_A = pointer(A_bits) + Int(tid) * 16
    dst_A = pointer(smem_A) + Int(tid) * 16
    ptx"cp.async.ca.shared.global"(dst_A, src_A, Val(16))
    if tid < UInt32(8)
        src_B = pointer(B_bits) + Int(tid) * 16
        dst_B = pointer(smem_B) + Int(tid) * 16
        ptx"cp.async.ca.shared.global"(dst_B, src_B, Val(16))
    end
    ptx"cp.async.commit_group"()
    ptx"cp.async.wait_all"()
    ptx"bar.sync"(Val(0))

    # 2. Cross-proxy fence: cp.async writes via async-proxy, wgmma reads via
    #    async-proxy too on Hopper, but the ordering between generic-region
    #    SMEM stores and async-proxy reads needs this.
    ptx"fence.proxy.async"()
    ptx"wgmma.fence.sync.aligned"()

    # 3. Build descriptors. K-major bf16 16-element-wide tile:
    #    leading = 16 bytes (one row), stride = 128 bytes (8 rows).
    a_addr = PTX.smem_addr_u32(pointer(smem_A))
    b_addr = PTX.smem_addr_u32(pointer(smem_B))
    a_desc = PTX.wgmma_descriptor(a_addr;
                                   leading_byte_offset = 16,
                                   stride_byte_offset  = 128)
    b_desc = PTX.wgmma_descriptor(b_addr;
                                   leading_byte_offset = 16,
                                   stride_byte_offset  = 128)

    # 4. wgmma m64n8k16.f32.bf16.bf16: 4 d-regs per lane, 128-lane warpgroup.
    d = ntuple(_ -> 0f0, Val(4))
    d = ptx"wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16"(
        d, a_desc, b_desc, false)

    # 5. Drain the warp-group async pipeline.
    ptx"wgmma.commit_group.sync.aligned"()
    ptx"wgmma.wait_group.sync.aligned"(Val(0))

    # 6. Lane 0 stashes one element so the kernel isn't dead-coded away.
    if tid == UInt32(0)
        @inbounds D[1] = d[1]
    end
    return nothing
end

@testset "Hopper full pipeline (cp.async + wgmma)" begin
    types = Tuple{CuDeviceVector{Float32, 1},
                  CuDeviceVector{UInt16, 1},
                  CuDeviceVector{UInt16, 1}}
    @test ptxas_compiles(_hopper_pipeline_kernel!, types;
                         cap = v"9.0", feature_set = :arch)
    ptx = emit_ptx(_hopper_pipeline_kernel!, types;
                   cap = v"9.0", feature_set = :arch)
    # Every step of the canonical Hopper pipeline appears in the emitted PTX.
    @test occursin("cp.async.ca.shared.global",                       ptx)
    @test occursin("cp.async.commit_group",                           ptx)
    @test occursin("cp.async.wait_all",                               ptx)
    # tier-2 barrier: ISel formats with a tab where inline asm had a space
    @test occursin(r"bar\.sync \s*0",                                 ptx)
    @test occursin("fence.proxy.async",                               ptx)
    @test occursin("wgmma.fence.sync.aligned",                        ptx)
    @test occursin("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16", ptx)
    @test occursin("wgmma.commit_group.sync.aligned",                 ptx)
    @test occursin("wgmma.wait_group.sync.aligned 0",                 ptx)
    # Descriptor build constant-folds: leading/stride/swizzle bits encode
    # at compile time, then OR into the runtime SMEM address.
    @test occursin(r"or\.b64", ptx)
end


# --- stmatrix --------------------------------------------------------------
#
# stmatrix was added in PTX 8.0 / sm_90 (Hopper). ptxas rejects it on Ada
# and lower. Validates instruction text + operand shape through real ptxas.
# Since the tier-2 migration the address may be a 64-bit register (%rd, the
# intrinsic takes the pointer directly) instead of the asm tier's forced
# 32-bit %r — ptxas accepts both for .shared instructions.

function _stmatrix_compile_x1!(addr::Core.LLVMPtr{UInt16, PTX.AS.Shared},
                                v::UInt32)
    ptx"stmatrix.sync.aligned.m8n8.x1.shared.b16"(addr, v)
    return nothing
end

function _stmatrix_compile_x4!(addr::Core.LLVMPtr{UInt16, PTX.AS.Shared},
                                v0::UInt32, v1::UInt32, v2::UInt32, v3::UInt32)
    ptx"stmatrix.sync.aligned.m8n8.x4.shared.b16"(addr, (v0, v1, v2, v3))
    return nothing
end

function _stmatrix_compile_x4_trans!(addr::Core.LLVMPtr{UInt16, PTX.AS.Shared},
                                      v0::UInt32, v1::UInt32, v2::UInt32, v3::UInt32)
    ptx"stmatrix.sync.aligned.m8n8.x4.trans.shared.b16"(addr, (v0, v1, v2, v3))
    return nothing
end

@testset "stmatrix m8n8 x1/x4/x4.trans" begin
    types_x1 = Tuple{Core.LLVMPtr{UInt16, PTX.AS.Shared}, UInt32}
    @test ptxas_compiles(_stmatrix_compile_x1!, types_x1;
                         cap = v"9.0", feature_set = :arch)
    ptx_x1 = emit_ptx(_stmatrix_compile_x1!, types_x1;
                      cap = v"9.0", feature_set = :arch)
    @test occursin("stmatrix.sync.aligned.m8n8.x1.shared.b16", ptx_x1)
    # Operand order: address first, then 1-element brace.
    @test occursin(r"stmatrix\.sync\.aligned\.m8n8\.x1\.shared\.b16\s+\[%rd?\d+\]\s*,\s*\{%r\d+\}", ptx_x1)

    types_x4 = Tuple{Core.LLVMPtr{UInt16, PTX.AS.Shared}, UInt32, UInt32, UInt32, UInt32}
    @test ptxas_compiles(_stmatrix_compile_x4!, types_x4;
                         cap = v"9.0", feature_set = :arch)
    ptx_x4 = emit_ptx(_stmatrix_compile_x4!, types_x4;
                      cap = v"9.0", feature_set = :arch)
    @test occursin("stmatrix.sync.aligned.m8n8.x4.shared.b16", ptx_x4)
    # 4-element brace input.
    @test occursin(r"\[%rd?\d+\]\s*,\s*\{%r\d+(?:, %r\d+){3}\}", ptx_x4)

    @test ptxas_compiles(_stmatrix_compile_x4_trans!, types_x4;
                         cap = v"9.0", feature_set = :arch)
    ptx_xt = emit_ptx(_stmatrix_compile_x4_trans!, types_x4;
                      cap = v"9.0", feature_set = :arch)
    @test occursin("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16", ptx_xt)
end


# --- mbarrier Hopper-only ops -----------------------------------------------
#
# `expect_tx`, `arrive.expect_tx`, `try_wait`, and `try_wait.parity` were
# added in PTX 8.0 (sm_90). ptxas rejects them on Ada.

function _mbarrier_compile_expect_tx!(mbar::Core.LLVMPtr{UInt64, PTX.AS.Shared},
                                       tx::UInt32)
    ptx"mbarrier.expect_tx.shared.b64"(mbar, tx)
    return nothing
end

function _mbarrier_compile_arrive_expect_tx!(out,
        mbar::Core.LLVMPtr{UInt64, PTX.AS.Shared}, tx::UInt32)
    state = ptx"mbarrier.arrive.expect_tx.shared.b64"(mbar, tx)
    @inbounds out[1] = state
    return nothing
end

function _mbarrier_compile_try_wait!(out,
        mbar::Core.LLVMPtr{UInt64, PTX.AS.Shared}, state::UInt64)
    p = ptx"mbarrier.try_wait.shared.b64"(mbar, state)
    @inbounds out[1] = p ? UInt32(1) : UInt32(0)
    return nothing
end

function _mbarrier_compile_try_wait_parity!(out,
        mbar::Core.LLVMPtr{UInt64, PTX.AS.Shared}, phase::UInt32)
    p = ptx"mbarrier.try_wait.parity.shared.b64"(mbar, phase)
    @inbounds out[1] = p ? UInt32(1) : UInt32(0)
    return nothing
end

@testset "mbarrier sm_90 ops (expect_tx, try_wait, try_wait.parity)" begin
    types_etx = Tuple{Core.LLVMPtr{UInt64, PTX.AS.Shared}, UInt32}
    @test ptxas_compiles(_mbarrier_compile_expect_tx!, types_etx;
                         cap = v"9.0", feature_set = :arch)
    ptx = emit_ptx(_mbarrier_compile_expect_tx!, types_etx;
                   cap = v"9.0", feature_set = :arch)
    # since the tier-2 migration the backend spells the defaults out:
    # .relaxed is expect_tx's only legal sem, .cta the default scope —
    # the same operation in explicit form
    @test occursin("mbarrier.expect_tx.relaxed.cta.shared.b64", ptx)

    types_aetx = Tuple{CuDeviceVector{UInt64, 1},
                        Core.LLVMPtr{UInt64, PTX.AS.Shared}, UInt32}
    @test ptxas_compiles(_mbarrier_compile_arrive_expect_tx!, types_aetx;
                         cap = v"9.0", feature_set = :arch)
    ptx = emit_ptx(_mbarrier_compile_arrive_expect_tx!, types_aetx;
                   cap = v"9.0", feature_set = :arch)
    @test occursin("mbarrier.arrive.expect_tx.shared.b64", ptx)

    types_tw = Tuple{CuDeviceVector{UInt32, 1},
                      Core.LLVMPtr{UInt64, PTX.AS.Shared}, UInt64}
    @test ptxas_compiles(_mbarrier_compile_try_wait!, types_tw;
                         cap = v"9.0", feature_set = :arch)
    ptx = emit_ptx(_mbarrier_compile_try_wait!, types_tw;
                   cap = v"9.0", feature_set = :arch)
    @test occursin("mbarrier.try_wait.shared.b64", ptx)

    types_twp = Tuple{CuDeviceVector{UInt32, 1},
                       Core.LLVMPtr{UInt64, PTX.AS.Shared}, UInt32}
    @test ptxas_compiles(_mbarrier_compile_try_wait_parity!, types_twp;
                         cap = v"9.0", feature_set = :arch)
    ptx = emit_ptx(_mbarrier_compile_try_wait_parity!, types_twp;
                   cap = v"9.0", feature_set = :arch)
    @test occursin("mbarrier.try_wait.parity.shared.b64", ptx)
end


# --- TMA (cp.async.bulk.tensor) ---------------------------------------------
#
# Hopper TMA: one thread issues a bulk async tile copy with a tensor-map
# descriptor and coordinate vector. Validates the wrapper's [base, {coords}]
# operand encoding and per-arity constraint string assembly.

function _tma_load_2d!(dst::Core.LLVMPtr{UInt16, PTX.AS.Shared},
                       tmap::Core.LLVMPtr{UInt8, PTX.AS.Const},
                       c0::Int32, c1::Int32,
                       mbar::Core.LLVMPtr{UInt64, PTX.AS.Shared})
    ptx"cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"(
        dst, tmap, c0, c1, mbar)
    return nothing
end

function _tma_store_2d!(tmap::Core.LLVMPtr{UInt8, PTX.AS.Const},
                        c0::Int32, c1::Int32,
                        src::Core.LLVMPtr{UInt16, PTX.AS.Shared})
    ptx"cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"(
        tmap, c0, c1, src)
    return nothing
end

@testset "TMA cp.async.bulk.tensor.2d load + store" begin
    load_types = Tuple{Core.LLVMPtr{UInt16, PTX.AS.Shared},
                       Core.LLVMPtr{UInt8,  PTX.AS.Const},
                       Int32, Int32,
                       Core.LLVMPtr{UInt64, PTX.AS.Shared}}
    @test ptxas_compiles(_tma_load_2d!, load_types;
                         cap = v"9.0", feature_set = :arch)
    ptx = emit_ptx(_tma_load_2d!, load_types;
                   cap = v"9.0", feature_set = :arch)
    @test occursin("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes", ptx)

    store_types = Tuple{Core.LLVMPtr{UInt8,  PTX.AS.Const},
                        Int32, Int32,
                        Core.LLVMPtr{UInt16, PTX.AS.Shared}}
    @test ptxas_compiles(_tma_store_2d!, store_types;
                         cap = v"9.0", feature_set = :arch)
    ptx = emit_ptx(_tma_store_2d!, store_types;
                   cap = v"9.0", feature_set = :arch)
    @test occursin("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group", ptx)
end


# --- TMA bulk_group commit / wait + warp specialization helpers ----------
#
# Triton-corpus audit: matmul_tma_*sm9{0,12}a.ptx + fa_ws_pingpong_sm90a.ptx
# emit these as the connective tissue around TMA loads. All chain-default —
# `cp.async.bulk.{commit_group,wait_group.read}` take immediate args (or
# none); `setmaxnreg.{inc,dec}.sync.aligned.u32` take an immediate count;
# `fence.proxy.async.shared::cta` is no-arg.

function _hopper_async_helpers!()
    ptx"cp.async.bulk.commit_group"()
    ptx"cp.async.bulk.wait_group.read"(Val(0))
    ptx"fence.proxy.async.shared::cta"()
    ptx"setmaxnreg.dec.sync.aligned.u32"(Val(40))
    ptx"setmaxnreg.inc.sync.aligned.u32"(Val(232))
    return nothing
end

@testset "Hopper async helpers (commit/wait/fence/setmaxnreg)" begin
    @test ptxas_compiles(_hopper_async_helpers!, Tuple{};
                         cap = v"9.0", feature_set = :arch)
    ptx = emit_ptx(_hopper_async_helpers!, Tuple{};
                   cap = v"9.0", feature_set = :arch)
    @test occursin("cp.async.bulk.commit_group",          ptx)
    @test occursin("cp.async.bulk.wait_group.read",       ptx)
    @test occursin("fence.proxy.async.shared::cta",       ptx)
    @test occursin("setmaxnreg.dec.sync.aligned.u32 40",  ptx)
    @test occursin("setmaxnreg.inc.sync.aligned.u32 232", ptx)
end


# --- wgmma convergence through the optimizer ---------------------------------
# The spike shape (spikes/raw_asm_attrs.jl): identical collective calls
# leading both arms of a divergent branch, checked on the OPTIMIZED module.
# Guards two things the straight-line goldens cannot see:
#   - the `convergent` attribute group survives to the optimized module and
#     is bound to the asm call site (its loss is invisible to llc/ptxas —
#     the attribute only ever binds in the in-process middle end);
#   - both call sites are still distinct calls (not merged/hoisted/sunk into
#     one, not duplicated further).

function _hopper_wgmma_divergent!(out::CuDeviceVector{Float32, 1},
                                  a_desc::UInt64, b_desc::UInt64)
    tid = ptx"mov.u32"(sreg"tid.x")
    zero4 = (0f0, 0f0, 0f0, 0f0)
    d = if tid < UInt32(64)
        ptx"wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16"(
            zero4, a_desc, b_desc, Val(true))
    else
        ptx"wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16"(
            zero4, a_desc, b_desc, Val(true))
    end
    @inbounds out[tid + 1] = d[1]
    return nothing
end

@testset "wgmma convergent attribute survives optimization" begin
    types = Tuple{CuDeviceVector{Float32, 1}, UInt64, UInt64}
    llvm = sprint() do io
        CUDATools.code_llvm(io, _hopper_wgmma_divergent!, types;
                            arch = SMVersion(9, 0, :arch), kernel = true,
                            dump_module = true)
    end
    # two distinct call sites, each carrying an attribute group
    sites = collect(eachmatch(r"call [^\n]*asm sideeffect \"wgmma\.mma_async[^\n]*", llvm))
    @test length(sites) == 2
    @test all(m -> occursin(r"#\d+", m.match), sites)
    # and the group they reference includes `convergent`
    @test occursin(r"attributes #\d+ = \{[^}]*convergent", llvm)
end


# --- cluster-scope generic fences (tier-1 core IR, sm_90+ ISel floor) --------
# fence.{sc,acq_rel}.cluster lower to `fence syncscope("cluster") ...`; ISel
# rejects them below sm_90 with a loud error, so their pipeline validation
# lives here rather than in ptxas/baseline.jl. Stores between the fences
# keep each one's position observable.

function _hopper_cluster_fences!(out)
    @inbounds begin
        out[1] = UInt32(1)
        ptx"fence.sc.cluster"()
        out[2] = UInt32(2)
        ptx"fence.acq_rel.cluster"()
        out[3] = UInt32(3)
    end
    return nothing
end

@testset "cluster-scope generic fences at sm_90" begin
    types = Tuple{CuDeviceVector{UInt32, 1}}
    @test ptxas_compiles(_hopper_cluster_fences!, types;
                         cap = v"9.0", feature_set = :arch)
    ptx = emit_ptx(_hopper_cluster_fences!, types;
                   cap = v"9.0", feature_set = :arch)
    @test occursin("fence.sc.cluster;",      ptx)
    @test occursin("fence.acq_rel.cluster;", ptx)
end


# --- tensormap descriptor mutation (host-side TMA descriptor build) ------
#
# Triton's matmul_tma_sm120a kernel patches a shared-memory copy of the TMA
# descriptor with `tensormap.replace.tile.<field>.shared::cta.b1024.<wty>`
# before publishing it via `tensormap.cp_fenceproxy.*`. PTX 9.2 §9.7.9.10.
# All chain-default — first arg is the [shared-mem descriptor pointer] (the
# `tensormap` brackets=true contract in src/forms.jl), then a mix
# of immediate field indices and value registers. Each field has a fixed
# write width (b32 for most, b64 for global_address / global_stride).

function _hopper_tensormap_replace!(desc::Core.LLVMPtr{UInt8, PTX.AS.Shared},
                                     gaddr::UInt64, gdim::UInt32, gstride::UInt64,
                                     bdim::UInt32, estride::UInt32)
    ptx"tensormap.replace.tile.global_address.shared::cta.b1024.b64"(desc, gaddr)
    ptx"tensormap.replace.tile.rank.shared::cta.b1024.b32"(desc, Val(1))
    ptx"tensormap.replace.tile.box_dim.shared::cta.b1024.b32"(desc, Val(0), bdim)
    ptx"tensormap.replace.tile.global_dim.shared::cta.b1024.b32"(desc, Val(0), gdim)
    ptx"tensormap.replace.tile.global_stride.shared::cta.b1024.b64"(desc, Val(0), gstride)
    ptx"tensormap.replace.tile.element_stride.shared::cta.b1024.b32"(desc, Val(0), estride)
    ptx"tensormap.replace.tile.elemtype.shared::cta.b1024.b32"(desc, Val(0))
    ptx"tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32"(desc, Val(0))
    ptx"tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32"(desc, Val(2))
    ptx"tensormap.replace.tile.fill_mode.shared::cta.b1024.b32"(desc, Val(0))
    return nothing
end

@testset "tensormap.replace.tile.* (TMA descriptor mutation)" begin
    types = Tuple{Core.LLVMPtr{UInt8, PTX.AS.Shared},
                  UInt64, UInt32, UInt64, UInt32, UInt32}
    @test ptxas_compiles(_hopper_tensormap_replace!, types;
                         cap = v"9.0", feature_set = :arch)
    ptx = emit_ptx(_hopper_tensormap_replace!, types;
                   cap = v"9.0", feature_set = :arch)
    @test occursin("tensormap.replace.tile.global_address.shared::cta.b1024.b64",   ptx)
    @test occursin("tensormap.replace.tile.global_stride.shared::cta.b1024.b64",    ptx)
    @test occursin("tensormap.replace.tile.box_dim.shared::cta.b1024.b32",          ptx)
    @test occursin("tensormap.replace.tile.elemtype.shared::cta.b1024.b32",         ptx)
    @test occursin("tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32",     ptx)
    @test occursin("tensormap.replace.tile.fill_mode.shared::cta.b1024.b32",        ptx)
    @test occursin("tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32", ptx)
end

# tensormap.cp_fenceproxy + fence.proxy.tensormap::generic.acquire — the
# release/acquire pair that publishes the patched descriptor to TMA-engine-
# visible memory. Both take a [global-ptr] (the live descriptor) and an
# immediate size (128 = sizeof tensormap). cp_fenceproxy also takes the
# [shared-ptr] of the patched copy as source.

function _hopper_tensormap_publish!(live::Core.LLVMPtr{UInt8, PTX.AS.Global},
                                     scratch::Core.LLVMPtr{UInt8, PTX.AS.Shared})
    ptx"tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned"(
        live, scratch, Val(128))
    ptx"fence.proxy.tensormap::generic.acquire.gpu"(live, Val(128))
    return nothing
end

@testset "tensormap publish (cp_fenceproxy + fence.proxy.tensormap.acquire)" begin
    types = Tuple{Core.LLVMPtr{UInt8, PTX.AS.Global},
                  Core.LLVMPtr{UInt8, PTX.AS.Shared}}
    @test ptxas_compiles(_hopper_tensormap_publish!, types;
                         cap = v"9.0", feature_set = :arch)
    ptx = emit_ptx(_hopper_tensormap_publish!, types;
                   cap = v"9.0", feature_set = :arch)
    @test occursin("tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned",
                   ptx)
    @test occursin("fence.proxy.tensormap::generic.acquire.gpu", ptx)
end


# --- wgmma dtype combos used by Triton matmul corpus ----------------------
#
# Confirms the wgmma.jl L2 table covers the specific shapes Triton emits in
# matmul_tma_v3{3..5}_sm90a.ptx, fa_ws_pingpong_sm90a.ptx, and
# matmul_tma_sm120a.ptx. Each kernel issues the typed wrapper exactly once
# — the goal is dispatch coverage, not data path.

function _wgmma_compile_f16_n128!(out, a_desc::UInt64, b_desc::UInt64)
    d = ntuple(_ -> 0f0, Val(64))   # N/2 = 64 f32 regs for m64n128k16.f32
    d_new = ptx"wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16"(
        d, a_desc, b_desc, false)
    @inbounds out[1] = d_new[1]
    return nothing
end

function _wgmma_compile_f16_n64!(out, a_desc::UInt64, b_desc::UInt64)
    d = ntuple(_ -> 0f0, Val(32))   # N/2 = 32 f32 regs for m64n64k16.f32
    d_new = ptx"wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16"(
        d, a_desc, b_desc, false)
    @inbounds out[1] = d_new[1]
    return nothing
end

function _wgmma_compile_e5m2_n64!(out, a_desc::UInt64, b_desc::UInt64)
    d = ntuple(_ -> 0f0, Val(32))   # N/2 = 32 f32 regs for m64n64k32.f32
    d_new = ptx"wgmma.mma_async.sync.aligned.m64n64k32.f32.e5m2.e5m2"(
        d, a_desc, b_desc, false)
    @inbounds out[1] = d_new[1]
    return nothing
end

@testset "wgmma dtype combos (Triton corpus: m64n128k16 f16, m64n64k{16,32} f16/e5m2)" begin
    types = Tuple{CuDeviceVector{Float32, 1}, UInt64, UInt64}
    for (kf, asm) in (
            (_wgmma_compile_f16_n128!,
             "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16"),
            (_wgmma_compile_f16_n64!,
             "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16"),
            (_wgmma_compile_e5m2_n64!,
             "wgmma.mma_async.sync.aligned.m64n64k32.f32.e5m2.e5m2"))
        @test ptxas_compiles(kf, types; cap = v"9.0", feature_set = :arch)
        @test occursin(asm, emit_ptx(kf, types; cap = v"9.0", feature_set = :arch))
    end
end

# --- Pipeline / Barrier abstractions --------------------------------------
#
# The verb-style API in src/pipeline.jl is a thin layer over the raw
# `ptx"mbarrier.*"` wrappers. These tests are smoke kernels that
# exercise the abstraction end-to-end (init + cursor + waits + arrives)
# and confirm the emitted PTX matches the same sequence as the
# hand-rolled callers used to produce.

using PTX.MBarriers: BarrierArray, barrier_init, barrier_arrive,
                     barrier_arrive_expect_tx, barrier_wait,
                     barrier_arrive_cluster
using PTX.Pipelines: Pipeline, pipeline_cursor, pipeline_stage,
                     pipeline_init!

function _pipe_smoke_cta!(C::CuDeviceVector{UInt64, 1}, K::Int32)
    mbar_full  = CuStaticSharedArray(UInt64, 2)
    mbar_empty = CuStaticSharedArray(UInt64, 2)
    full  = BarrierArray{2}(pointer(mbar_full))
    empty = BarrierArray{2}(pointer(mbar_empty))

    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        pipeline_init!(full, empty, Val(1), Val(false))
    end
    ptx"bar.sync"(Val(0))

    if tid == UInt32(0)
        @inbounds for k_iter in Int32(0):(K - Int32(1))
            stage, phase = pipeline_cursor(Pipeline{2}, k_iter)
            barrier_wait(empty[stage], phase)
            barrier_arrive_expect_tx(full[stage], 64)
            barrier_arrive(empty[stage])
            barrier_wait(full[stage], phase)
        end
        C[1] = UInt64(K)
    end
    return nothing
end

@testset "pipeline_init! + barrier verbs (CTA-scope)" begin
    types = Tuple{CuDeviceVector{UInt64, 1}, Int32}
    @test ptxas_compiles(_pipe_smoke_cta!, types;
                         cap = v"9.0", feature_set = :arch)
    ptx = emit_ptx(_pipe_smoke_cta!, types;
                   cap = v"9.0", feature_set = :arch)
    @test occursin("mbarrier.init.shared.b64",             ptx)
    @test occursin("mbarrier.arrive.shared.b64",           ptx)
    @test occursin("mbarrier.arrive.expect_tx.shared.b64", ptx)
    @test occursin("mbarrier.test_wait.parity.shared.b64", ptx)
    @test occursin("fence.proxy.async.shared::cta",        ptx)
    @test !occursin("fence.mbarrier_init.release.cluster", ptx)
end

function _pipe_smoke_cluster!(C::CuDeviceVector{UInt64, 1})
    mbar_full  = CuStaticSharedArray(UInt64, 2)
    mbar_empty = CuStaticSharedArray(UInt64, 2)
    full  = BarrierArray{2}(pointer(mbar_full))
    empty = BarrierArray{2}(pointer(mbar_empty))

    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        # 2 consumers × 2 clusters = 4 pre-arrives per stage.
        pipeline_init!(full, empty, Val(4), Val(true))
    end
    ptx"bar.sync"(Val(0))

    stage = pipeline_stage(Pipeline{2}, Int32(0))
    if tid < UInt32(2)
        barrier_arrive_cluster(empty[stage], tid)
    end
    C[1] = UInt64(stage)
    return nothing
end

@testset "pipeline_init! + barrier verbs (cluster-scope)" begin
    types = Tuple{CuDeviceVector{UInt64, 1}}
    @test ptxas_compiles(_pipe_smoke_cluster!, types;
                         cap = v"9.0", feature_set = :arch)
    ptx = emit_ptx(_pipe_smoke_cluster!, types;
                   cap = v"9.0", feature_set = :arch)
    @test occursin("fence.mbarrier_init.release.cluster", ptx)
    @test occursin("mapa.shared::cluster.u32",            ptx)
    @test occursin("mbarrier.arrive.shared::cluster.b64", ptx)
end

# --- bf16x2_pack ergonomic helper -----------------------------------------
#
# `bf16x2_pack(lo, hi)` calls the fused `cvt.rn.bf16x2.f32` with swapped
# args so the Julia caller passes (lo, hi) and PTX gets (a=hi, b=lo)
# per its first-arg-upper convention (PTX 9.2 §9.7.9.21).

using PTX: bf16x2_pack

function _bf16x2_pack_compile!(out::CuDeviceVector{UInt32, 1},
                                a::Float32, b::Float32)
    @inbounds out[1] = bf16x2_pack(a, b)
    return nothing
end

@testset "bf16x2_pack lowers to fused cvt.rn.bf16x2.f32" begin
    types = Tuple{CuDeviceVector{UInt32, 1}, Float32, Float32}
    @test ptxas_compiles(_bf16x2_pack_compile!, types;
                         cap = v"8.0", feature_set = :baseline)
    ptx = emit_ptx(_bf16x2_pack_compile!, types;
                   cap = v"8.0", feature_set = :baseline)
    @test occursin("cvt.rn.bf16x2.f32", ptx)
end
