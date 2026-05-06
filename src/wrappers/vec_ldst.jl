# Vectorized `ld.{state-space}.v{2,4}.{dtype}` and `st.*` counterparts. Hand-
# written because the chain default infers the wrong rettype (`last(parts)`
# as a scalar dtype, not `NTuple{4, T}`) and can't emit braced register-vector
# operands per arg.

# (count, dtype, Julia type, constraint letter)
const _VEC_LDST_VARIANTS = (
    (2, :f32, Float32, "f"),
    (4, :f32, Float32, "f"),
    (2, :b32, UInt32,  "r"),
    (4, :b32, UInt32,  "r"),
    (2, :b16, UInt16,  "h"),
    (4, :b16, UInt16,  "h"),
)

function vec_ld_spec(n::Int, dtype::Symbol, letter::String)
    asm = "ld.global.v$n.$dtype {" *
          join(("\$$i" for i in 0:n-1), ", ") * "}, [\$$n];"
    constraints = join(vcat(fill("=$letter", n), ["l", "~{memory}"]), ",")
    (; asm, constraints)
end

function vec_st_spec(n::Int, dtype::Symbol, letter::String)
    asm = "st.global.v$n.$dtype [\$0], {" *
          join(("\$$i" for i in 1:n), ", ") * "};"
    constraints = join(vcat(["l"], fill(letter, n), ["~{memory}"]), ",")
    (; asm, constraints)
end

function _vec_ld_register(n::Int, dtype::Symbol, T, letter::String)
    mods = (:global, Symbol("v", n), dtype)
    spec = vec_ld_spec(n, dtype, letter)
    asm, constraints = spec.asm, spec.constraints
    @eval @generated function (::Operation{:ld, $mods})(
            addr::Core.LLVMPtr{S, AS.Global}) where S
        quote
            Base.@inline
            @asmcall($($asm), $($constraints), true,
                     NTuple{$($n), $($T)},
                     Tuple{Core.LLVMPtr{$S, AS.Global}},
                     addr)
        end
    end
    nothing
end

function _vec_st_register(n::Int, dtype::Symbol, T, letter::String)
    mods = (:global, Symbol("v", n), dtype)
    spec = vec_st_spec(n, dtype, letter)
    asm, constraints = spec.asm, spec.constraints
    flat_argtypes = vcat([:(Core.LLVMPtr{S, AS.Global})], fill(T, n))
    val_args = [:(vals[$i]) for i in 1:n]
    @eval function (::Operation{:st, $mods})(
            addr::Core.LLVMPtr{S, AS.Global},
            vals::NTuple{$n, $T}) where S
        Base.@inline
        @asmcall($asm, $constraints, true, Nothing,
                 Tuple{$(flat_argtypes...)},
                 addr, $(val_args...))
        nothing
    end
    nothing
end

for (n, dt, T, letter) in _VEC_LDST_VARIANTS
    _vec_ld_register(n, dt, T, letter)
    _vec_st_register(n, dt, T, letter)
end
