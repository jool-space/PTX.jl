# Aggregate-return spike — hardware validation (GB10, 2026-06-12) of struct
# returns through Base.llvmcall; the suite inherited the oracle
# (test/gpu/nvvm.jl).
#
# Question: can tier-2 emission return multi-result intrinsics (struct returns
# unpacked to NTuple) through Base.llvmcall, surviving Julia codegen and the
# GPU pipeline? Subject: ldmatrix.sync.aligned.m8n8.x4.b16 — returns
# {i32,i32,i32,i32}, convergent, executable on the GB10 (sm_75+).
#
# The llvmcall entry repacks the struct into [4 x i32] (Julia's mapping for
# NTuple{4,UInt32}) via extractvalue/insertvalue — exactly the pattern the
# registry generator will emit for tcgen05.ld and friends.
#
# Run: julia --project=/tmp/convspike spikes/aggregate_return.jl

using CUDACore
using GPUCompiler: CompilerJob, methodinstance
import GPUCompiler

@inline ldmatrix_x4(p::Core.LLVMPtr{UInt16,3}) =
    Base.llvmcall(("""
        declare { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16(ptr addrspace(3)) #0
        define [4 x i32] @entry(ptr addrspace(3) %p) #1 {
            %s = call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16(ptr addrspace(3) %p)
            %v0 = extractvalue { i32, i32, i32, i32 } %s, 0
            %v1 = extractvalue { i32, i32, i32, i32 } %s, 1
            %v2 = extractvalue { i32, i32, i32, i32 } %s, 2
            %v3 = extractvalue { i32, i32, i32, i32 } %s, 3
            %a0 = insertvalue [4 x i32] undef, i32 %v0, 0
            %a1 = insertvalue [4 x i32] %a0, i32 %v1, 1
            %a2 = insertvalue [4 x i32] %a1, i32 %v2, 2
            %a3 = insertvalue [4 x i32] %a2, i32 %v3, 3
            ret [4 x i32] %a3
        }
        attributes #0 = { convergent nounwind memory(argmem: read) }
        attributes #1 = { alwaysinline }
        """, "entry"), NTuple{4,UInt32}, Tuple{Core.LLVMPtr{UInt16,3}}, p)

function kernel!(out)
    smem = CuStaticSharedArray(UInt16, 256)   # four 8x8 b16 matrices
    t = threadIdx().x                          # 1-based lane
    for j in 1:8                               # fill: smem[i] = i-1, row-major
        @inbounds smem[(t-1)*8 + j] = UInt16((t-1)*8 + j - 1)
    end
    sync_threads()
    # lane l supplies the address of 16-byte row l (lanes 0-7 → matrix 0, ...)
    p = pointer(smem) + (t-1)*16
    r = ldmatrix_x4(p)
    @inbounds for m in 1:4
        out[t, m] = r[m]
    end
    return
end

out = CUDACore.zeros(UInt32, 32, 4)
@cuda threads=32 kernel!(out)
h = Array(out)

source = methodinstance(typeof(kernel!), Tuple{CuDeviceMatrix{UInt32,1}})
config = CUDACore.compiler_config(CUDACore.device(); kernel=true)
ptx = sprint(io -> GPUCompiler.code_native(io, CompilerJob(source, config)))

# mechanics: the instruction was selected with 4 destination registers
inst = match(r"ldmatrix\.sync\.aligned\.m8n8\.x4\.shared\.b16\s+\{[^}]*\}", ptx)
println("PTX instruction: ", inst === nothing ? "MISSING" : inst.match)
nregs = inst === nothing ? 0 : count(',', inst.match) + 1
println("destination registers: ", nregs)

# semantics: standard m8n8 fragment layout — thread t (0-based), matrix m
# holds the b16 pair at row t÷4, cols 2(t%4)..2(t%4)+1, i.e. elements
# 64m + 8(t÷4) + 2(t%4) (+1 in the high half)
expected(t, m) = let e = UInt32(64m + 8(t ÷ 4) + 2(t % 4))
    (e + UInt32(1)) << 16 | e
end
ok = all(h[t+1, m+1] == expected(t, m) for t in 0:31, m in 0:3)
println("fragment values match m8n8 layout for all 32×4: ", ok)
println("sample (thread 0): ", [string(x, base=16, pad=8) for x in h[1, :]])

println(nregs == 4 && ok ? "RESULT: aggregate returns VALIDATED" :
                           "RESULT: needs investigation")
