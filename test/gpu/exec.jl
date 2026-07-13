# TEST_TARGET: requires=gpu evidence=runtime target=sm_89
#
# End-to-end tests: kernels launch on a real device and produce correct
# numerical results. Compile-only Hopper/Blackwell tests live under
# `ptxas/`; raw-PTX corpus tests live under `gpu/corpus_compile.jl`.

# --- add.f32 ----------------------------------------------------------------

function _exec_add_f32!(out, a, b)
    @inbounds out[1] = ptx"add.f32"(a, b)
    return nothing
end

@testset "add.f32 executes" begin
    out = CUDACore.zeros(Float32, 1)
    @cuda threads=1 _exec_add_f32!(out, 1.5f0, 2.5f0)
    CUDACore.synchronize()
    @test Array(out)[1] == 4.0f0
end

# --- f32 ALU expansion ------------------------------------------------------

function _exec_alu_f32!(out, a, b, c)
    @inbounds out[1] = ptx"sub.f32"(a, b)
    @inbounds out[2] = ptx"mul.f32"(a, b)
    @inbounds out[3] = ptx"fma.rn.f32"(a, b, c)
    @inbounds out[4] = ptx"min.f32"(a, b)
    @inbounds out[5] = ptx"max.f32"(a, b)
    return nothing
end

@testset "f32 ALU executes" begin
    out = CUDACore.zeros(Float32, 5)
    a, b, c = 3.0f0, 5.0f0, 7.0f0
    @cuda threads=1 _exec_alu_f32!(out, a, b, c)
    CUDACore.synchronize()
    got = Array(out)
    @test got[1] == a - b           # sub
    @test got[2] == a * b           # mul
    @test got[3] == fma(a, b, c)    # fma.rn (round-to-nearest is Julia default)
    @test got[4] == min(a, b)       # min
    @test got[5] == max(a, b)       # max
end

# --- mov.u32 (every thread reads its own tid.x) ----------------------------

function _exec_mov_tid_x!(out)
    tid = ptx"mov.u32"(sreg"tid.x")
    @inbounds out[tid + 1] = tid
    return nothing
end

@testset "mov.u32 %tid.x executes" begin
    out = CUDACore.zeros(UInt32, 32)
    @cuda threads=32 _exec_mov_tid_x!(out)
    CUDACore.synchronize()
    @test Array(out) == UInt32.(0:31)
end

function _exec_mov_ntid_x!(out)
    tid = ptx"mov.u32"(sreg"tid.x")
    nt  = ptx"mov.u32"(sreg"ntid.x")
    @inbounds out[tid + 1] = nt
    return nothing
end

@testset "mov.u32 %ntid.x executes" begin
    out = CUDACore.zeros(UInt32, 16)
    @cuda threads=16 _exec_mov_ntid_x!(out)
    CUDACore.synchronize()
    @test all(Array(out) .== 0x10)
end

# --- bar.sync (cross-thread shared-memory communication) -------------------

function _exec_bar_sync!(out)
    smem = CuStaticSharedArray(Float32, 32)
    tid = ptx"mov.u32"(sreg"tid.x")
    @inbounds smem[tid + 1] = Float32(tid)
    ptx"bar.sync"(Val(0))
    # After the barrier every thread sees the writes from every other thread.
    @inbounds out[tid + 1] = smem[((tid + 1) % 32) + 1]
    return nothing
end

@testset "bar.sync synchronizes block" begin
    out = CUDACore.zeros(Float32, 32)
    @cuda threads=32 _exec_bar_sync!(out)
    CUDACore.synchronize()
    expected = Float32.(((0:31) .+ 1) .% 32)
    @test Array(out) == expected
end

# --- atom.add.gpu.u32 -------------------------------------------------------

function _exec_atom_add_gpu!(counter)
    ptx"atom.add.gpu.u32"(pointer(counter), UInt32(1))
    return nothing
end

@testset "atom.add.gpu.u32 executes" begin
    counter = CUDACore.zeros(UInt32, 1)
    nblocks, nthreads = 4, 128
    @cuda blocks=nblocks threads=nthreads _exec_atom_add_gpu!(counter)
    CUDACore.synchronize()
    @test Array(counter)[1] == UInt32(nblocks * nthreads)
end

# --- cp.async sync ops (smoke: no in-flight copies, must be a clean no-op) ---
# Real correctness coverage waits for cp.async.ca.shared.global in step 4.

function _exec_cp_async_sync!(out)
    ptx"cp.async.commit_group"()
    ptx"cp.async.wait_group"(Val(0))
    ptx"cp.async.wait_all"()
    @inbounds out[1] = 42f0
    return nothing
end

@testset "cp.async sync ops smoke" begin
    out = CUDACore.zeros(Float32, 1)
    @cuda threads=1 _exec_cp_async_sync!(out)
    CUDACore.synchronize()
    @test Array(out)[1] == 42f0
end

# --- cp.async.ca.shared.global round-trip ----------------------------------
# Single thread copies 16B (4 Float32s) from global to shared via cp.async,
# commits + waits, then writes the shared values back to a different global
# buffer. Verifies the payload survives the async copy.

function _exec_cp_async_ca!(dst, src)
    smem = CuStaticSharedArray(Float32, 4)
    if ptx"mov.u32"(sreg"tid.x") == 0
        ptx"cp.async.ca.shared.global"(pointer(smem), pointer(src), Val(16))
        ptx"cp.async.commit_group"()
        ptx"cp.async.wait_group"(Val(0))
        for i in 1:4
            @inbounds dst[i] = smem[i]
        end
    end
    return nothing
end

@testset "cp.async.ca.shared.global round-trips" begin
    src = CuArray(Float32[10.0, 20.0, 30.0, 40.0])
    dst = CUDACore.zeros(Float32, 4)
    @cuda threads=32 _exec_cp_async_ca!(dst, src)
    CUDACore.synchronize()
    @test Array(dst) == [10.0f0, 20.0f0, 30.0f0, 40.0f0]
end

# --- mma.sync.aligned.m16n8k16 bf16 smoke -----------------------------------
# Whole-zero A and B, identity C — every lane should observe d == c.
# Verifies the warp-cooperative op compiles, launches, and the I/O register
# layout from our hand-rolled IR matches NVPTX's expectations.

function _exec_mma_bf16_smoke!(out)
    a = (UInt32(0), UInt32(0), UInt32(0), UInt32(0))
    b = (UInt32(0), UInt32(0))
    tid = ptx"mov.u32"(sreg"tid.x")
    base = Float32(tid) * 4f0
    c = (base, base + 1f0, base + 2f0, base + 3f0)
    d = ptx"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"(a, b, c)
    off = Int(tid) * 4
    @inbounds out[off + 1] = d[1]
    @inbounds out[off + 2] = d[2]
    @inbounds out[off + 3] = d[3]
    @inbounds out[off + 4] = d[4]
    return nothing
end

@testset "mma.sync bf16 smoke (a=b=0 → d == c)" begin
    out = CUDACore.zeros(Float32, 32 * 4)
    @cuda threads=32 _exec_mma_bf16_smoke!(out)
    CUDACore.synchronize()
    got = Array(out)
    expected = [Float32(t) * 4f0 + Float32(j) for t in 0:31 for j in 0:3]
    @test got == expected
end

# Real correctness: A and B are all 1.0, C is zero. Each output element of
# the 16x8 result is a dot-product of 16 ones with 16 ones, plus 0 → 16.0.
# bf16 1.0 packs as 0x3F80; two packed = 0x3F803F80 in a UInt32. Every lane
# holds the same uniform fragment, so the lane-to-element mapping doesn't
# matter — the entire output should be 16.0.

function _exec_mma_bf16_correct!(out)
    one_packed = UInt32(0x3F80_3F80)
    a = (one_packed, one_packed, one_packed, one_packed)
    b = (one_packed, one_packed)
    c = (0f0, 0f0, 0f0, 0f0)
    d = ptx"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"(a, b, c)
    tid = ptx"mov.u32"(sreg"tid.x")
    off = Int(tid) * 4
    @inbounds out[off + 1] = d[1]
    @inbounds out[off + 2] = d[2]
    @inbounds out[off + 3] = d[3]
    @inbounds out[off + 4] = d[4]
    return nothing
end

@testset "mma.sync bf16 correctness (A=B=1.0, C=0 → D==16.0)" begin
    out = CUDACore.zeros(Float32, 32 * 4)
    @cuda threads=32 _exec_mma_bf16_correct!(out)
    CUDACore.synchronize()
    @test all(Array(out) .== 16.0f0)
end

# m16n8k8 (smaller k) — same identity check, expected sum is 8 instead of 16.
function _exec_mma_bf16_k8!(out)
    one_packed = UInt32(0x3F80_3F80)     # bf16 1.0 packed × 2
    a = (one_packed, one_packed)
    b = (one_packed,)
    c = (0f0, 0f0, 0f0, 0f0)
    d = ptx"mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32"(a, b, c)
    tid = ptx"mov.u32"(sreg"tid.x")
    off = Int(tid) * 4
    @inbounds out[off + 1] = d[1]
    @inbounds out[off + 2] = d[2]
    @inbounds out[off + 3] = d[3]
    @inbounds out[off + 4] = d[4]
    return nothing
end

@testset "mma.sync bf16 m16n8k8 correctness (A=B=1.0, C=0 → D==8.0)" begin
    out = CUDACore.zeros(Float32, 32 * 4)
    @cuda threads=32 _exec_mma_bf16_k8!(out)
    CUDACore.synchronize()
    @test all(Array(out) .== 8.0f0)
end

# f16 inputs, f32 accumulator — same shape, different dtype.
function _exec_mma_f16!(out)
    one_packed = UInt32(0x3C00_3C00)     # f16 1.0 packed × 2
    a = (one_packed, one_packed, one_packed, one_packed)
    b = (one_packed, one_packed)
    c = (0f0, 0f0, 0f0, 0f0)
    d = ptx"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"(a, b, c)
    tid = ptx"mov.u32"(sreg"tid.x")
    off = Int(tid) * 4
    @inbounds out[off + 1] = d[1]
    @inbounds out[off + 2] = d[2]
    @inbounds out[off + 3] = d[3]
    @inbounds out[off + 4] = d[4]
    return nothing
end

@testset "mma.sync f16/f32 correctness (A=B=1.0, C=0 → D==16.0)" begin
    out = CUDACore.zeros(Float32, 32 * 4)
    @cuda threads=32 _exec_mma_f16!(out)
    CUDACore.synchronize()
    @test all(Array(out) .== 16.0f0)
end

# --- End-to-end pipeline: cp.async + mma.sync -------------------------------
# Real pipeline integration on Ada (sm_89). One warp:
#   1. cp.async.ca.shared.global  loads A (16×16 bf16) and B (16×8 bf16)
#      from global into shared memory (32 threads × 16B = 512B for A;
#      16 threads × 16B = 256B for B).
#   2. cp.async.commit_group + wait_group(0) flushes the in-flight copies.
#   3. bar.sync makes the shared writes visible warp-wide.
#   4. Each lane reads its 4 a-frags + 2 b-frags from shared (uniform 0x3F80
#      bf16 in this test, so layout-correctness isn't being checked — only
#      that the data flowed end-to-end).
#   5. mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 produces D.
#   6. Each lane writes its 4 D-frags to a flat global output.
#
# Uniform A=B=1.0 means every D element is 16.0; layout of which lane holds
# which (row, col) doesn't matter for correctness. (A non-uniform variant
# requires the full PTX m16n8k16 fragment-distribution table; that's a
# follow-on once we want to test ldmatrix layouts directly.)

function _exec_gemm_pipeline!(D::CuDeviceArray{Float32},
                              A::CuDeviceArray{UInt32},
                              B::CuDeviceArray{UInt32})
    A_smem = CuStaticSharedArray(UInt32, 128)  # 256 bf16
    B_smem = CuStaticSharedArray(UInt32, 64)   # 128 bf16

    tid = ptx"mov.u32"(sreg"tid.x")

    # cp.async A: 32 lanes × 16B = 512B (the whole 256-bf16 A)
    a_dst = pointer(A_smem, Int(tid) * 4 + 1)
    a_src = pointer(A,      Int(tid) * 4 + 1)
    ptx"cp.async.ca.shared.global"(a_dst, a_src, Val(16))

    # cp.async B: 16 lanes × 16B = 256B (the whole 128-bf16 B)
    if tid < UInt32(16)
        b_dst = pointer(B_smem, Int(tid) * 4 + 1)
        b_src = pointer(B,      Int(tid) * 4 + 1)
        ptx"cp.async.ca.shared.global"(b_dst, b_src, Val(16))
    end

    ptx"cp.async.commit_group"()
    ptx"cp.async.wait_group"(Val(0))
    ptx"bar.sync"(Val(0))

    # Read fragments from shared (uniform here, so any 4/2 UInt32 read works).
    @inbounds a = (A_smem[1], A_smem[2], A_smem[3], A_smem[4])
    @inbounds b = (B_smem[1], B_smem[2])
    c = (0f0, 0f0, 0f0, 0f0)

    d = ptx"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"(a, b, c)

    off = Int(tid) * 4
    @inbounds D[off + 1] = d[1]
    @inbounds D[off + 2] = d[2]
    @inbounds D[off + 3] = d[3]
    @inbounds D[off + 4] = d[4]
    return nothing
end

@testset "End-to-end Ada pipeline: cp.async → mma.sync (A=B=1.0 → D=16.0)" begin
    one_packed = UInt32(0x3F80_3F80)        # bf16 1.0 packed × 2
    A = CUDACore.fill(one_packed, 128)          # 16×16 bf16
    B = CUDACore.fill(one_packed, 64)           # 16×8  bf16
    D = CUDACore.zeros(Float32, 32 * 4)
    @cuda threads=32 _exec_gemm_pipeline!(D, A, B)
    CUDACore.synchronize()
    @test all(Array(D) .== 16.0f0)
end

# --- cvt correctness --------------------------------------------------------

# f32 → f16 round-trip: PTX rounds to nearest, exact for representable values.
function _exec_cvt_f16_f32!(out, x)
    @inbounds out[1] = ptx"cvt.rn.f16.f32"(x)
    return nothing
end

@testset "cvt.rn.f16.f32 round-trip (1.5 ↔ Float16(1.5))" begin
    out = CUDACore.zeros(Float16, 1)
    @cuda threads=1 _exec_cvt_f16_f32!(out, 1.5f0)
    CUDACore.synchronize()
    @test Array(out)[1] === Float16(1.5)
end

# f16 → f32 widening — PTX rejects rounding modes for widening, so the
# wrapper is registered without `.rn`: `cvt.f32.f16`.
function _exec_cvt_f32_f16!(out, x)
    @inbounds out[1] = ptx"cvt.f32.f16"(x)
    return nothing
end

@testset "cvt.f32.f16 widening (Float16(1.5) → 1.5f0)" begin
    out = CUDACore.zeros(Float32, 1)
    @cuda threads=1 _exec_cvt_f32_f16!(out, Float16(1.5))
    CUDACore.synchronize()
    @test Array(out)[1] === 1.5f0
end

# bf16 → f32 widening: bf16(1.0) bit pattern 0x3F80, expect Float32(1.0).
function _exec_cvt_f32_bf16!(out, x)
    @inbounds out[1] = ptx"cvt.f32.bf16"(x)
    return nothing
end

@testset "cvt.f32.bf16 (0x3F80 → 1.0f0)" begin
    out = CUDACore.zeros(Float32, 1)
    @cuda threads=1 _exec_cvt_f32_bf16!(out, UInt16(0x3F80))
    CUDACore.synchronize()
    @test Array(out)[1] === 1.0f0
end

# Packed e4m3x2: pack two known f32 values into a UInt16. PTX packs `a` in
# the HIGH byte and `b` in the LOW byte (verified empirically against ptxas).
# e4m3(1.0) = 0x38, e4m3(2.0) = 0x40 → packed (a=1.0, b=2.0) is 0x3840.
function _exec_cvt_e4m3x2!(out, a, b)
    @inbounds out[1] = ptx"cvt.rn.satfinite.e4m3x2.f32"(a, b)
    return nothing
end

@testset "cvt.rn.satfinite.e4m3x2.f32 packs (1.0, 2.0) → 0x3840" begin
    out = CUDACore.zeros(UInt16, 1)
    @cuda threads=1 _exec_cvt_e4m3x2!(out, 1.0f0, 2.0f0)
    CUDACore.synchronize()
    @test Array(out)[1] == UInt16(0x3840)
end

# --- shfl.sync correctness -------------------------------------------------

# .idx with c = 0x1F: every lane reads from lane `b`. All 32 lanes that wrote
# their tid should read back the value held by lane 5.
function _exec_shfl_idx!(out)
    tid = ptx"mov.u32"(sreg"tid.x")
    src = tid + UInt32(100)
    val = ptx"shfl.sync.idx.b32"(src, UInt32(5), UInt32(0x1F), UInt32(0xFFFFFFFF))
    @inbounds out[Int(tid) + 1] = val
    return nothing
end

@testset "shfl.sync.idx.b32 (every lane reads lane 5)" begin
    out = CUDACore.zeros(UInt32, 32)
    @cuda threads=32 _exec_shfl_idx!(out)
    CUDACore.synchronize()
    @test all(Array(out) .== UInt32(105))    # lane 5's source = 5 + 100
end

# .bfly with mask 0x10, c = 0x1F: lane T receives from lane (T XOR 16).
function _exec_shfl_bfly!(out)
    tid = ptx"mov.u32"(sreg"tid.x")
    val = ptx"shfl.sync.bfly.b32"(tid, UInt32(0x10), UInt32(0x1F), UInt32(0xFFFFFFFF))
    @inbounds out[Int(tid) + 1] = val
    return nothing
end

@testset "shfl.sync.bfly.b32 (XOR-16 swap halves)" begin
    out = CUDACore.zeros(UInt32, 32)
    @cuda threads=32 _exec_shfl_bfly!(out)
    CUDACore.synchronize()
    expected = UInt32.(xor.(0:31, 0x10))
    @test Array(out) == expected
end

# --- fence + bar.warp.sync execute cleanly ---------------------------------

function _exec_fences!(out)
    ptx"fence.sc.gpu"()
    ptx"fence.acq_rel.cta"()
    @inbounds out[1] = 42f0
    return nothing
end

@testset "fence variants execute (smoke)" begin
    out = CUDACore.zeros(Float32, 1)
    @cuda threads=1 _exec_fences!(out)
    CUDACore.synchronize()
    @test Array(out)[1] == 42f0
end

# bar.warp.sync correctness: same shape as the existing bar.sync test, but
# warp-scoped. 32 lanes write tid to shared memory; warp barrier; each lane
# reads its neighbor's slot.
function _exec_bar_warp_sync!(out)
    smem = CuStaticSharedArray(Float32, 32)
    tid = ptx"mov.u32"(sreg"tid.x")
    @inbounds smem[tid + 1] = Float32(tid)
    ptx"bar.warp.sync"(UInt32(0xFFFFFFFF))
    @inbounds out[tid + 1] = smem[((tid + 1) % 32) + 1]
    return nothing
end

@testset "bar.warp.sync synchronizes warp" begin
    out = CUDACore.zeros(Float32, 32)
    @cuda threads=32 _exec_bar_warp_sync!(out)
    CUDACore.synchronize()
    expected = Float32.(((0:31) .+ 1) .% 32)
    @test Array(out) == expected
end

# --- setp single-pred -------------------------------------------------------
#
# Verifies the i1↔i8 bridge end-to-end: setp's `=b` constraint produces a
# `.pred` register, the Bool result feeds Julia's `if`, the compiler emits
# the right `selp`/branch on the consumer side.

function _exec_setp_eq!(out, x::Int32, y::Int32)
    p = ptx"setp.eq.s32"(x, y)
    @inbounds out[1] = p ? Int32(111) : Int32(222)
    return nothing
end

@testset "setp.eq.s32 → Bool drives if" begin
    out = CUDACore.zeros(Int32, 1)
    @cuda threads=1 _exec_setp_eq!(out, Int32(7), Int32(7))
    CUDACore.synchronize()
    @test Array(out)[1] == 111   # equal → true branch

    @cuda threads=1 _exec_setp_eq!(out, Int32(7), Int32(8))
    CUDACore.synchronize()
    @test Array(out)[1] == 222   # not equal → false branch
end

# Per-thread guard, exercising the predicated-control-flow pattern transpilers
# emit. Each thread writes only when its tid passes the predicate.
function _exec_setp_lt_threaded!(out, threshold::UInt32)
    tid = ptx"mov.u32"(sreg"tid.x")
    p = ptx"setp.lt.u32"(tid, threshold)
    if p
        @inbounds out[tid + 1] = tid
    end
    return nothing
end

@testset "setp.lt.u32 gates per-thread store" begin
    out = CUDACore.fill(typemax(UInt32), 32)
    @cuda threads=32 _exec_setp_lt_threaded!(out, UInt32(8))
    CUDACore.synchronize()
    got = Array(out)
    @test got[1:8]   == UInt32.(0:7)             # under threshold: wrote tid
    @test all(got[9:32] .== typemax(UInt32))     # over threshold: untouched
end

# --- vote.sync.{ballot,all,any}.pred ---------------------------------------
#
# vote.sync.ballot.b32: pred in (Bool), u32 mask out (one bit per lane).
# vote.sync.all.pred / .any.pred: pred in, pred out — round-trips Bool
# through a `.pred` register on both sides.

function _exec_vote_ballot!(out, threshold::UInt32)
    tid = ptx"mov.u32"(sreg"tid.x")
    p = ptx"setp.lt.u32"(tid, threshold)
    mask = ptx"vote.sync.ballot.b32"(p, UInt32(0xFFFFFFFF))
    if tid == UInt32(0)
        @inbounds out[1] = mask
    end
    return nothing
end

@testset "vote.sync.ballot.b32 collects per-lane preds" begin
    out = CUDACore.zeros(UInt32, 1)
    @cuda threads=32 _exec_vote_ballot!(out, UInt32(8))
    CUDACore.synchronize()
    @test Array(out)[1] == UInt32(0x000000FF)    # lanes 0..7 set
end

function _exec_vote_all!(out, threshold::UInt32)
    tid = ptx"mov.u32"(sreg"tid.x")
    p = ptx"setp.lt.u32"(tid, threshold)
    all_p = ptx"vote.sync.all.pred"(p, UInt32(0xFFFFFFFF))
    any_p = ptx"vote.sync.any.pred"(p, UInt32(0xFFFFFFFF))
    if tid == UInt32(0)
        @inbounds out[1] = all_p ? Int32(1) : Int32(0)
        @inbounds out[2] = any_p ? Int32(1) : Int32(0)
    end
    return nothing
end

@testset "vote.sync.all/any.pred round-trip Bool through .pred" begin
    out = CUDACore.zeros(Int32, 2)
    @cuda threads=32 _exec_vote_all!(out, UInt32(32))   # all lanes < 32
    CUDACore.synchronize()
    @test Array(out) == [Int32(1), Int32(1)]            # all=true, any=true

    @cuda threads=32 _exec_vote_all!(out, UInt32(8))    # only 0..7 < 8
    CUDACore.synchronize()
    @test Array(out) == [Int32(0), Int32(1)]            # all=false, any=true

    @cuda threads=32 _exec_vote_all!(out, UInt32(0))    # no lanes < 0
    CUDACore.synchronize()
    @test Array(out) == [Int32(0), Int32(0)]            # all=false, any=false
end

# --- setp dual-pred ---------------------------------------------------------
#
# Verifies the multi-output struct-bridge path (LLVM.jl#531) end-to-end:
# `Tuple{Bool, Bool}` rettype + `=b,=b,r,r` constraints produces
# `setp.<cmp>.<dt> %pN|%pM, ...` with both predicate slots populated.

function _exec_setp_dual_int!(out_p, out_q, x::Int32, y::Int32)
    p, q = ptx"setp.dual.lt.s32"(x, y)
    @inbounds out_p[1] = p ? Int32(1) : Int32(0)
    @inbounds out_q[1] = q ? Int32(1) : Int32(0)
    return nothing
end

@testset "setp.dual.lt.s32 → (Bool, Bool)" begin
    out_p, out_q = CUDACore.zeros(Int32, 1), CUDACore.zeros(Int32, 1)
    @cuda threads=1 _exec_setp_dual_int!(out_p, out_q, Int32(3), Int32(5))
    CUDACore.synchronize()
    @test Array(out_p)[1] == 1 && Array(out_q)[1] == 0    # 3<5 → (true, false)

    @cuda threads=1 _exec_setp_dual_int!(out_p, out_q, Int32(7), Int32(5))
    CUDACore.synchronize()
    @test Array(out_p)[1] == 0 && Array(out_q)[1] == 1    # 7<5 → (false, true)
end

function _exec_setp_dual_float!(out_p, out_q, x::Float32, y::Float32)
    p, q = ptx"setp.dual.eq.f32"(x, y)
    @inbounds out_p[1] = p ? Int32(1) : Int32(0)
    @inbounds out_q[1] = q ? Int32(1) : Int32(0)
    return nothing
end

@testset "setp.dual.eq.f32 → (Bool, Bool)" begin
    out_p, out_q = CUDACore.zeros(Int32, 1), CUDACore.zeros(Int32, 1)
    @cuda threads=1 _exec_setp_dual_float!(out_p, out_q, 1f0, 1f0)
    CUDACore.synchronize()
    @test Array(out_p)[1] == 1 && Array(out_q)[1] == 0    # equal → (true, false)

    @cuda threads=1 _exec_setp_dual_float!(out_p, out_q, 1f0, 2f0)
    CUDACore.synchronize()
    @test Array(out_p)[1] == 0 && Array(out_q)[1] == 1    # !equal → (false, true)
end

# --- shfl.sync with pred output --------------------------------------------
#
# `shfl.sync.idx.b32 d|p, src, lane, 0x1f, mask;` returns (data, in_range_pred).
# When the source lane is masked out by the membermask, p=false and d is
# undefined. Tests both the in-range and out-of-range cases.

function _exec_shfl_idx_pred!(out_v, out_p, src_lane::UInt32)
    tid = ptx"mov.u32"(sreg"tid.x")
    v, p = ptx"shfl.sync.idx.b32.pred"(tid, src_lane, UInt32(0x1f),
                                       UInt32(0x000000FF))   # only lanes 0..7 active
    @inbounds out_v[tid + 1] = v
    @inbounds out_p[tid + 1] = p ? Int32(1) : Int32(0)
    return nothing
end

@testset "shfl.sync.idx.b32.pred returns (data, in_range)" begin
    out_v = CUDACore.zeros(UInt32, 32)
    out_p = CUDACore.zeros(Int32, 32)
    @cuda threads=32 _exec_shfl_idx_pred!(out_v, out_p, UInt32(3))   # all read lane 3
    CUDACore.synchronize()
    got_v = Array(out_v)
    got_p = Array(out_p)
    # Lanes 0..7 are active in the membermask, lane 3 is the source.
    # PTX semantics: lanes inside the mask read lane 3's value; lanes
    # outside the mask are NOT participating (out_p comes from the active
    # lanes only — the test reads them all but only 0..7 should have run).
    @test got_v[1:8] == fill(UInt32(3), 8)            # all in-mask read tid=3
    @test got_p[1:8] == fill(Int32(1), 8)             # in-range true
end

# --- smem_addr_u32 + wgmma_descriptor (Hopper-prep, work on Ada too) -------
#
# smem_addr_u32 surfaces the 32-bit in-CTA SMEM offset of a shared-AS
# pointer. NVPTX represents shared-AS pointers as 32-bit registers natively;
# the helper just exposes that via `mov.u32` with the `r` constraint. Works
# on any sm_70+ — only wgmma itself is sm_90a, not the address conversion.
#
# wgmma_descriptor is pure UInt64 arithmetic; this confirms it inlines on
# device and constant-folds the compile-time fields.

function _exec_smem_addr!(out_lane0, out_diff)
    smem = CuStaticSharedArray(UInt32, 8)             # 32 bytes
    a = PTX.smem_addr_u32(pointer(smem))
    b = PTX.smem_addr_u32(pointer(smem) + 16)         # 16 bytes ahead
    tid = ptx"mov.u32"(sreg"tid.x")
    if tid == UInt32(0)
        @inbounds out_lane0[1] = a
        @inbounds out_diff[1]  = b - a
    end
    return nothing
end

@testset "smem_addr_u32 lowers + offsets correctly" begin
    out_lane0 = CUDACore.zeros(UInt32, 1)
    out_diff  = CUDACore.zeros(UInt32, 1)
    @cuda threads=32 _exec_smem_addr!(out_lane0, out_diff)
    CUDACore.synchronize()
    # We can't predict the absolute SMEM offset (depends on CUDA's allocator),
    # but the 16-byte ptr offset must produce exactly a 16-byte address delta.
    @test Array(out_diff)[1] == UInt32(16)
end

function _exec_wgmma_descriptor!(out)
    # Canonical bf16 16×8 K-major tile descriptor:
    #   addr = 0x200, leading = 16, stride = 128, swizzle = NONE
    #   expected = 0x20 | (1 << 16) | (8 << 32) = 0x0000_0008_0001_0020
    d = PTX.wgmma_descriptor(UInt32(0x200);
                              leading_byte_offset = 16,
                              stride_byte_offset  = 128)
    @inbounds out[1] = d
    return nothing
end

@testset "wgmma_descriptor inlines on device" begin
    out = CUDACore.zeros(UInt64, 1)
    @cuda threads=1 _exec_wgmma_descriptor!(out)
    CUDACore.synchronize()
    expected = UInt64(0x20) | (UInt64(1) << 16) | (UInt64(8) << 32)
    @test Array(out)[1] === expected
end


# --- mbarrier init / arrive / test_wait round-trip (Ada+) -----------------
#
# Two-warp kernel: each thread writes its tid into shared memory, then all
# threads rendezvous on a shared mbarrier; after the wait, each thread reads
# its neighbor's shared slot and copies it to global out. mbarrier provides
# the inter-warp ordering — without it, the second-pass reads would race
# the first-pass writes since warps schedule independently on Volta+.
#
# The init/arrive uses one shared u64 slot for the mbarrier; the data array
# is a separate shared buffer. Phase 0 → arrive → phase flips → test_wait
# returns true → safe to read.

function _exec_mbarrier_rendezvous!(out)
    bar_smem  = CuStaticSharedArray(UInt64, 1)
    data_smem = CuStaticSharedArray(Int32, 64)
    mbar = pointer(bar_smem)
    tid  = ptx"mov.u32"(sreg"tid.x")

    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mbar, UInt32(64))
    end
    ptx"bar.sync"(Val(0))    # init must be visible before any thread arrives

    @inbounds data_smem[tid + 1] = Int32(tid)

    ptx"mbarrier.arrive.shared.b64"(mbar)

    while !ptx"mbarrier.test_wait.parity.shared.b64"(mbar, UInt32(0))
        # spin until phase flips from 0 → 1 (all 64 arrivals observed)
    end

    nbr = (tid + UInt32(1)) % UInt32(64)
    @inbounds out[tid + 1] = data_smem[nbr + 1]
    return nothing
end

@testset "mbarrier init/arrive/test_wait synchronizes 2 warps" begin
    out = CUDACore.zeros(Int32, 64)
    @cuda threads=64 _exec_mbarrier_rendezvous!(out)
    CUDACore.synchronize()
    expected = Int32.(((0:63) .+ 1) .% 64)
    @test Array(out) == expected
end
