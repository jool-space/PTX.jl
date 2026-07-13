# TEST_TARGET: requires=toolkit evidence=mixed runtime=cc==9.0
# Ptr-array batched Hopper GEMM — ported from
# cutlass/examples/56_hopper_ptr_array_batched_gemm (NVIDIA CUTLASS,
# BSD-3-Clause).
#
# The CUTLASS example runs L independent same-shape GEMMs whose operand
# pointers come from device-resident pointer arrays (ptr-array batching, à
# la gemm_batched's array-of-pointers mode) — unlike strided batching, the
# batches need not be a fixed stride apart, or even come from one
# allocation. CUTLASS's headline feature there is on-the-fly TMA-descriptor
# modification (the kernel patches the descriptor's base address when it
# moves between batches).
#
# What's ported: the ptr-array indirection itself. The host builds one TMA
# descriptor per batch per operand (each batch's A/B are genuinely separate
# CuArray allocations — nothing is a fixed stride apart), uploads the
# 128-byte blobs, and passes device arrays OF DESCRIPTOR ADDRESSES. Each
# CTA picks its batch off `ctaid.x`, loads the UInt64 descriptor address
# from the array, and reinterprets it to `PTX.TMADescriptorPtr`
# (`Core.LLVMPtr{UInt8, AS.Const}` is primitive — the reinterpret is the
# same bitcast upload_tma_descriptor does host-side). The GEMM body is the
# single-warpgroup TMA → mbarrier → wgmma pipeline of gemm_warpgroup.jl.
#
# Not ported: descriptor patching in-kernel (`tensormap.replace` — no
# wrapper yet; one descriptor per batch sidesteps it), the cooperative
# multi-CTA tile scheduler, and the alpha/beta epilogue.

using PTX: wgmma_descriptor, smem_addr_u32, layout_for_a, tensor_map_tile_2d
using CUDACore
using Random

const PAB_BM = 64
const PAB_BN = 8
const PAB_BK = 16
const PAB_THREADS = 128                          # one warpgroup
const PAB_L = 4                                  # batch count (runtime test)

# Total TMA bytes the mbarrier waits on per batch (A tile + B tile, bf16).
const PAB_LOAD_BYTES = PAB_BM * PAB_BK * 2 + PAB_BK * PAB_BN * 2

function _pab_gemm_kernel!(
        D::CuDeviceVector{Float32, 1},
        descs_A::CuDeviceVector{UInt64, 1},
        descs_B::CuDeviceVector{UInt64, 1})

    smem_A = CuStaticSharedArray(UInt16, PAB_BM * PAB_BK)
    smem_B = CuStaticSharedArray(UInt16, PAB_BK * PAB_BN)
    mbar   = CuStaticSharedArray(UInt64, 1)

    a_ptr  = pointer(smem_A)
    b_ptr  = pointer(smem_B)
    mb_ptr = pointer(mbar)
    a_addr = smem_addr_u32(a_ptr)
    b_addr = smem_addr_u32(b_ptr)

    tid   = ptx"mov.u32"(sreg"tid.x")
    batch = ptx"mov.u32"(sreg"ctaid.x")

    # Ptr-array indirection: every thread loads its batch's descriptor
    # ADDRESSES from global memory, then treats them as TMA descriptor
    # pointers. (The descriptor blob itself stays in global memory; only
    # lane 0 ever dereferences it, inside the TMA issue.)
    tma_A = reinterpret(PTX.TMADescriptorPtr, @inbounds descs_A[Int(batch) + 1])
    tma_B = reinterpret(PTX.TMADescriptorPtr, @inbounds descs_B[Int(batch) + 1])

    # Same thread-0 issue block as gemm_warpgroup.jl: init + fence +
    # arrive.expect_tx + both bulk loads, then one CTA-wide bar.sync.
    if tid == UInt32(0)
        ptx"mbarrier.init.shared.b64"(mb_ptr, UInt32(1))
        ptx"fence.proxy.async.shared::cta"()
        ptx"mbarrier.arrive.expect_tx.shared.b64"(mb_ptr, UInt32(PAB_LOAD_BYTES))
        ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
            a_ptr, tma_A, Int32(0), Int32(0), mb_ptr)
        ptx"cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"(
            b_ptr, tma_B, Int32(0), Int32(0), mb_ptr)
    end
    ptx"bar.sync"(Val(0))

    while !ptx"mbarrier.test_wait.parity.shared.b64"(mb_ptr, UInt32(0))
    end

    ptx"fence.proxy.async.shared::cta"()
    ptx"wgmma.fence.sync.aligned"()

    # Both tiles K-fast in SMEM; K-major canonical layout for both
    # descriptors (see gemm_warpgroup.jl for the trans_b=0 reasoning).
    la = layout_for_a(dtype = :bf16, m = PAB_BM, k = PAB_BK)
    lb = layout_for_a(dtype = :bf16, m = PAB_BN, k = PAB_BK)
    a_desc = wgmma_descriptor(a_addr;
        leading_byte_offset = la.leading_byte_offset,
        stride_byte_offset  = la.stride_byte_offset,
        swizzle             = la.layout_type)
    b_desc = wgmma_descriptor(b_addr;
        leading_byte_offset = lb.leading_byte_offset,
        stride_byte_offset  = lb.stride_byte_offset,
        swizzle             = lb.layout_type)

    d = ntuple(_ -> 0f0, Val(4))
    d = ptx"wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16"(
        d, a_desc, b_desc, false)

    ptx"wgmma.commit_group.sync.aligned"()
    ptx"wgmma.wait_group.sync.aligned"(Val(0))

    # Epilogue: standard m64n8 f32 frag layout (gemm_warpgroup.jl), with the
    # batch's D slab at `batch * BM * BN` f32s.
    wid      = tid >> UInt32(5)
    lane     = tid & UInt32(31)
    frag_row = (wid << UInt32(4)) + (lane >> UInt32(2))
    frag_col = (lane & UInt32(3)) << UInt32(1)
    pd       = pointer(D)
    d_base   = batch * UInt32(PAB_BM * PAB_BN)
    off_a    = (d_base + frag_row * UInt32(PAB_BN) + frag_col) * UInt32(4)
    off_b    = (d_base + (frag_row + UInt32(8)) * UInt32(PAB_BN) + frag_col) * UInt32(4)
    ptx"st.global.v2.f32"(pd + Int(off_a), (d[1], d[2]))
    ptx"st.global.v2.f32"(pd + Int(off_b), (d[3], d[4]))
    return nothing
end

# ── Cross-arch ptxas validation ────────────────────────────────────────

@testset "ptr-array batched Hopper GEMM compiles at sm_90a" begin
    types = Tuple{CuDeviceVector{Float32, 1},
                  CuDeviceVector{UInt64, 1},
                  CuDeviceVector{UInt64, 1}}
    @test ptxas_compiles(_pab_gemm_kernel!, types;
                         cap = v"9.0", feature_set = :arch)

    ptx = emit_ptx(_pab_gemm_kernel!, types; cap = v"9.0", feature_set = :arch)
    @test occursin("mbarrier.arrive.expect_tx.shared.b64",                ptx)
    @test occursin("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes", ptx)
    @test occursin("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16", ptx)
    # The ptr-array indirection: a 64-bit global load of the descriptor
    # address must survive into the PTX (not get folded away).
    @test occursin("ld.global.u64", ptx) || occursin("ld.global.b64", ptx)
end

# ── Runtime — Hopper hardware ──────────────────────────────────────────

if test_runtime_supported(@__FILE__)
    @testset "ptr-array batched GEMM (L=$PAB_L separate allocations)" begin
        rng = MersenneTwister(0xba7c4)

        # Per-batch hosts + separately-allocated device arrays. The
        # allocations are genuinely independent — nothing about their
        # addresses is uniform-stride (that's the ptr-array point).
        A_f32s = Vector{Matrix{Float32}}(undef, PAB_L)
        B_f32s = Vector{Matrix{Float32}}(undef, PAB_L)
        A_ds   = Vector{CuArray{UInt16, 2}}(undef, PAB_L)
        B_ds   = Vector{CuArray{UInt16, 2}}(undef, PAB_L)
        # upload_tma_descriptor blobs MUST stay alive through the launch.
        ups_A  = Vector{Any}(undef, PAB_L)
        ups_B  = Vector{Any}(undef, PAB_L)

        for l in 1:PAB_L
            A_f32 = Float32.(randn(rng, PAB_BM, PAB_BK)) .* 0.1f0
            B_f32 = Float32.(randn(rng, PAB_BK, PAB_BN)) .* 0.1f0
            A_f32s[l] = A_f32
            B_f32s[l] = B_f32

            # K-fast packing, same convention as gemm_warpgroup.jl.
            A_packed = Array{UInt16}(undef, PAB_BK, PAB_BM)
            B_packed = Array{UInt16}(undef, PAB_BK, PAB_BN)
            for m in 1:PAB_BM, k in 1:PAB_BK
                A_packed[k, m] = bf16_bits(A_f32[m, k])
            end
            for n in 1:PAB_BN, k in 1:PAB_BK
                B_packed[k, n] = bf16_bits(B_f32[k, n])
            end
            A_ds[l] = CuArray(A_packed)
            B_ds[l] = CuArray(B_packed)

            tmap_A = tensor_map_tile_2d(:bf16, pointer(A_ds[l]),
                PAB_BM, PAB_BK, PAB_BM, PAB_BK; swizzle = :B32)
            tmap_B = tensor_map_tile_2d(:bf16, pointer(B_ds[l]),
                PAB_BN, PAB_BK, PAB_BN, PAB_BK; swizzle = :B32)
            ups_A[l] = upload_tma_descriptor(tmap_A)
            ups_B[l] = upload_tma_descriptor(tmap_B)
        end

        # Device-resident pointer arrays: one UInt64 descriptor address per
        # batch.
        descs_A = CuArray(UInt64[UInt64(UInt(pointer(ups_A[l].blob))) for l in 1:PAB_L])
        descs_B = CuArray(UInt64[UInt64(UInt(pointer(ups_B[l].blob))) for l in 1:PAB_L])

        D = CUDACore.zeros(Float32, PAB_L * PAB_BM * PAB_BN)
        @cuda threads = PAB_THREADS blocks = PAB_L _pab_gemm_kernel!(
            D, descs_A, descs_B)
        CUDACore.synchronize()

        D_host = Array(D)
        for l in 1:PAB_L
            slab = reshape(D_host[(l - 1) * PAB_BM * PAB_BN + 1 : l * PAB_BM * PAB_BN],
                           PAB_BN, PAB_BM)
            D_got = Array{Float32}(undef, PAB_BM, PAB_BN)
            for m in 1:PAB_BM, n in 1:PAB_BN
                D_got[m, n] = slab[n, m]
            end
            D_ref = bf16_gemm_ref(A_f32s[l], B_f32s[l])
            @test isapprox(D_got, D_ref; rtol = 1e-3, atol = 1e-3)
        end
    end
end
