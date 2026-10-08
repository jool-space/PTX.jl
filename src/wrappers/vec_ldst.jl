# Vectorized `ld.global.v{2,4}.{f32,b32,b16}` and `st.*` counterparts.
#
# TIER-1 (core LLVM IR, no NVVM intrinsic): a plain `load <N x T>` /
# `store <N x T>` on a global (addrspace 1) pointer lowers to the same
# `ld.global.v{2,4}` / `st.global.v{2,4}` ptxas emits, provided the access
# carries an alignment >= N*sizeof(T) (otherwise the backend splits it into
# scalar accesses). The prior asm form added a `~{memory}` clobber — an
# optimization barrier; the core-IR form drops it, so these become real,
# reorderable, CSE-able memory ops. The v{2,4} alignment was already a hard
# hardware requirement of the asm instruction, so asserting `align` here adds
# no new caller obligation.
#
# Hand-written (not the chain default) because the surface is `NTuple{N, T}`:
# the result/argument vector must be repacked between the LLVM vector type
# `<N x T>` (what `load`/`store` want) and the LLVM array type `[N x T]` (how
# Julia represents a homogeneous `NTuple`). The extract/insert pairs are free
# — ISel coalesces them into the vector instruction's register group.

# (count, dtype, Julia type)
const _VEC_LDST_VARIANTS = (
    (2, :f32, Float32),
    (4, :f32, Float32),
    (2, :b32, UInt32),
    (4, :b32, UInt32),
    (2, :b16, UInt16),
    (4, :b16, UInt16),
)

# Load `<N x T>` from the global pointer, then repack into the `[N x T]`
# Julia tuple representation.
@llvmgenerated builder function _vec_load(addr::Core.LLVMPtr{S, AS.Global},
        ::Type{T}, ::Val{N})::NTuple{N, T} where {S, T, N}
    vec = LLVM.VectorType(convert(LLVMType, T), N)
    if supports_typed_pointers(LLVM.context())
        addr = bitcast!(builder, addr, LLVM.PointerType(vec, AS.Global))
    end
    v = load!(builder, vec, addr; align = N * sizeof(T))
    tup = LLVM.UndefValue(convert(LLVMType, NTuple{N, T}))
    for i in 0:N-1
        elem = extract_element!(builder, v, LLVM.ConstantInt(Int32(i)))
        tup = insert_value!(builder, tup, elem, i)
    end
    tup
end

# Unpack the `[N x T]` tuple into `<N x T>` and store it to the global
# pointer.
@llvmgenerated builder function _vec_store(addr::Core.LLVMPtr{S, AS.Global},
        vals::Tuple{T, Vararg{T, M}})::Nothing where {S, T, M}
    N = M + 1
    vec = LLVM.VectorType(convert(LLVMType, T), N)
    v = LLVM.UndefValue(vec)
    for i in 0:N-1
        elem = extract_value!(builder, vals, i)
        v = insert_element!(builder, v, elem, LLVM.ConstantInt(Int32(i)))
    end
    if supports_typed_pointers(LLVM.context())
        addr = bitcast!(builder, addr, LLVM.PointerType(vec, AS.Global))
    end
    store!(builder, v, addr; align = N * sizeof(T))
    nothing
end

function _vec_ld_register(n::Int, dtype::Symbol, T)
    mods = (:global, Symbol("v", n), dtype)
    register_wrapper!(:vec_ldst, :ld, mods, :core_ir)
    load = device_only(:(_vec_load(addr, $T, Val($n))), "ld.global.v$n.$dtype")
    @eval @inline (::Operation{:ld, $mods})(addr::Core.LLVMPtr{S, AS.Global}) where S =
        $load
    nothing
end

function _vec_st_register(n::Int, dtype::Symbol, T)
    mods = (:global, Symbol("v", n), dtype)
    register_wrapper!(:vec_ldst, :st, mods, :core_ir)
    store = device_only(:(_vec_store(addr, vals)), "st.global.v$n.$dtype")
    @eval @inline function (::Operation{:st, $mods})(
            addr::Core.LLVMPtr{S, AS.Global}, vals::NTuple{$n, $T}) where S
        $store
        nothing
    end
    nothing
end

for (n, dt, T) in _VEC_LDST_VARIANTS
    _vec_ld_register(n, dt, T)
    _vec_st_register(n, dt, T)
end
