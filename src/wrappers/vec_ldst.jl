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

# PTX vector dtype → LLVM element type.
_vec_llvm_elt(dtype::Symbol) =
    dtype === :f32 ? "float" :
    dtype === :b32 ? "i32"   :
    dtype === :b16 ? "i16"   :
    error("unsupported vec ld/st dtype $dtype")

# Body IR for `Base.llvmcall`: load `<N x ET>` from the global pointer (%0),
# then repack into the `[N x ET]` Julia tuple representation.
function vec_ld_ir(n::Int, dtype::Symbol, align::Int)
    et  = _vec_llvm_elt(dtype)
    vec = "<$n x $et>"
    arr = "[$n x $et]"
    lines = ["%v = load $vec, ptr addrspace(1) %0, align $align"]
    for i in 0:n-1
        push!(lines, "%e$i = extractelement $vec %v, i32 $i")
    end
    prev = "undef"
    for i in 0:n-1
        push!(lines, "%r$i = insertvalue $arr $prev, $et %e$i, $i")
        prev = "%r$i"
    end
    push!(lines, "ret $arr $prev")
    join(lines, "\n")
end

# Body IR for `Base.llvmcall`: unpack the `[N x ET]` tuple argument (%1) into
# `<N x ET>` and store it to the global pointer (%0).
function vec_st_ir(n::Int, dtype::Symbol, align::Int)
    et  = _vec_llvm_elt(dtype)
    vec = "<$n x $et>"
    arr = "[$n x $et]"
    lines = String[]
    for i in 0:n-1
        push!(lines, "%a$i = extractvalue $arr %1, $i")
    end
    prev = "undef"
    for i in 0:n-1
        push!(lines, "%v$i = insertelement $vec $prev, $et %a$i, i32 $i")
        prev = "%v$i"
    end
    push!(lines, "store $vec $prev, ptr addrspace(1) %0, align $align")
    push!(lines, "ret void")
    join(lines, "\n")
end

function _vec_ld_register(n::Int, dtype::Symbol, T)
    mods = (:global, Symbol("v", n), dtype)
    ir   = vec_ld_ir(n, dtype, n * sizeof(T))
    @eval @generated function (::Operation{:ld, $mods})(
            addr::Core.LLVMPtr{S, AS.Global}) where S
        quote
            Base.@inline
            Base.llvmcall($($ir), NTuple{$($n), $($T)},
                          Tuple{Core.LLVMPtr{$S, AS.Global}}, addr)
        end
    end
    nothing
end

function _vec_st_register(n::Int, dtype::Symbol, T)
    mods = (:global, Symbol("v", n), dtype)
    ir   = vec_st_ir(n, dtype, n * sizeof(T))
    @eval function (::Operation{:st, $mods})(
            addr::Core.LLVMPtr{S, AS.Global},
            vals::NTuple{$n, $T}) where S
        Base.@inline
        Base.llvmcall($ir, Nothing,
                      Tuple{Core.LLVMPtr{S, AS.Global}, NTuple{$n, $T}},
                      addr, vals)
        nothing
    end
    nothing
end

for (n, dt, T) in _VEC_LDST_VARIANTS
    _vec_ld_register(n, dt, T)
    _vec_st_register(n, dt, T)
end
