# Optimizer-safe extended-precision arithmetic (PTX 9.3 §9.7.2).
#
# `CC.CF` is implicit PTX architectural state: LLVM inline-asm constraints
# cannot name it, it is not preserved across calls, and the ISA intends it for
# straight-line carry sequences.  Consequently no standalone instruction in
# this family may use the generic/raw chain.  The typed scalar methods below
# expose carry-in/out as Bool values and seed/materialize CC.CF inside one asm
# block.  The aggregate helpers keep a complete multi-limb operation in one
# block, which is both safer and cheaper for production use.

const ExtendedPrecisionWord = Union{UInt32, Int32, UInt64, Int64}

_extended_precision_dtype(::Type{UInt32}) = (:u32, "r")
_extended_precision_dtype(::Type{Int32})  = (:s32, "r")
_extended_precision_dtype(::Type{UInt64}) = (:u64, "l")
_extended_precision_dtype(::Type{Int64})  = (:s64, "l")

# LLVM.Interop.@asmcall requires literal template/constraint strings at macro
# expansion time.  Returning the macro call from a generated body makes the
# aggregate length N literal while staying on LLVM.jl's public API.
function _extended_asmcall_expr(asm::String, constraints::String,
                                rettype::Type, argtype::Type,
                                args::AbstractVector)
    :( $(LLVM.Interop).@asmcall(
        $asm, $constraints, true, $rettype, $argtype, $(args...)) )
end

function _scalar_extended_spec(head::String, ::Type{T}, value_inputs::Int;
                               carry_in::Bool, carry_out::Bool,
                               flag_kind::Symbol = :carry) where {T<:ExtendedPrecisionWord}
    flag_kind in (:carry, :borrow) ||
        throw(ArgumentError("expected :carry or :borrow, got $flag_kind"))
    dt, letter = _extended_precision_dtype(T)
    output_count = carry_out ? 2 : 1
    value_slots = output_count .+ (0:value_inputs-1)
    carry_slot = output_count + value_inputs

    lines = String[]
    if carry_in
        # Turn the explicit predicate into CC.CF without depending on incoming
        # architectural state. Stay within the matching add/sub family so the
        # conversion directly mirrors PTX carry or borrow semantics instead of
        # relying on a cross-family representation of the implicit flag.
        push!(lines, "selp.$dt cc_tmp, 1, 0, \$$carry_slot;")
        if flag_kind === :carry
            push!(lines, "add.cc.$dt cc_tmp, cc_tmp, -1;")
        else
            push!(lines, "sub.cc.$dt cc_tmp, 0, cc_tmp;")
        end
    end
    push!(lines, "$head \$0, " * join(("\$$slot" for slot in value_slots), ", ") * ";")
    if carry_out
        # No PTX instruction reads CC.CF into a predicate directly. Materialize
        # with the matching consumer, then convert its nonzero result to Bool.
        # addc yields 0/1; subc yields 0/-1 for no-borrow/borrow.
        materialize = flag_kind === :carry ? "addc" : "subc"
        push!(lines, "$materialize.$dt cc_tmp, 0, 0;")
        push!(lines, "setp.ne.$dt \$1, cc_tmp, 0;")
    end

    asm = "{ .reg .$dt cc_tmp; " * join(lines, " ") * " }"
    outputs = carry_out ? ["=&$letter", "=b"] : ["=&$letter"]
    inputs = vcat(fill(letter, value_inputs), carry_in ? ["b"] : String[])
    # The NVPTX backend accepts `~{cc}` but does not model CC as a physical
    # register, so `sideeffect=true` remains the load-bearing optimizer barrier.
    # Do not claim a memory clobber: these per-thread arithmetic blocks neither
    # access nor synchronize memory.
    constraints = join(vcat(outputs, inputs, ["~{cc}"]), ",")
    rettype = carry_out ? Tuple{T, Bool} : T
    argtype = Tuple{ntuple(_ -> T, value_inputs)...,
                    (carry_in ? (Bool,) : ())...}
    (; asm, constraints, rettype, argtype)
end

function _register_scalar_extended(op::Symbol, mods::Tuple{Vararg{Symbol}},
                                   head::String, ::Type{T}, value_inputs::Int;
                                   carry_in::Bool, carry_out::Bool,
                                   flag_kind::Symbol = :carry) where {T<:ExtendedPrecisionWord}
    spec = _scalar_extended_spec(head, T, value_inputs;
                                 carry_in, carry_out, flag_kind)
    asm, constraints, rettype, argtype =
        spec.asm, spec.constraints, spec.rettype, spec.argtype
    args = [Symbol(:a, i) for i in 1:value_inputs]
    signature = [:( $(args[i])::$T ) for i in 1:value_inputs]
    carry_in && push!(signature, :(carry::Bool))
    callargs = Any[args...]
    carry_in && push!(callargs, :carry)
    register_wrapper!(:extended_precision, op, mods, :asm)
    @eval function (::Operation{$(QuoteNode(op)), $mods})($(signature...))
        Base.@inline
        @asmcall($asm, $constraints, true, $rettype, $argtype, $(callargs...))
    end
    nothing
end

for (dt, T) in ((:u32, UInt32), (:s32, Int32), (:u64, UInt64), (:s64, Int64))
    _register_scalar_extended(:add, (:cc, dt), "add.cc.$dt", T, 2;
                              carry_in = false, carry_out = true)
    _register_scalar_extended(:addc, (dt,), "addc.$dt", T, 2;
                              carry_in = true, carry_out = false)
    _register_scalar_extended(:addc, (:cc, dt), "addc.cc.$dt", T, 2;
                              carry_in = true, carry_out = true)

    _register_scalar_extended(:sub, (:cc, dt), "sub.cc.$dt", T, 2;
                              carry_in = false, carry_out = true,
                              flag_kind = :borrow)
    _register_scalar_extended(:subc, (dt,), "subc.$dt", T, 2;
                              carry_in = true, carry_out = false,
                              flag_kind = :borrow)
    _register_scalar_extended(:subc, (:cc, dt), "subc.cc.$dt", T, 2;
                              carry_in = true, carry_out = true,
                              flag_kind = :borrow)

    for half in (:lo, :hi)
        _register_scalar_extended(:mad, (half, :cc, dt), "mad.$half.cc.$dt", T, 3;
                                  carry_in = false, carry_out = true)
        _register_scalar_extended(:madc, (half, dt), "madc.$half.$dt", T, 3;
                                  carry_in = true, carry_out = false)
        _register_scalar_extended(:madc, (half, :cc, dt), "madc.$half.cc.$dt", T, 3;
                                  carry_in = true, carry_out = true)
    end
end

function _addsub_aggregate_spec(kind::Symbol, N::Int, ::Type{T},
                                carry_in::Bool) where {T<:ExtendedPrecisionWord}
    N > 0 || throw(ArgumentError("extended-precision operands must contain at least one limb"))
    kind in (:add, :sub) || throw(ArgumentError("expected :add or :sub, got $kind"))
    dt, letter = _extended_precision_dtype(T)
    consumer = kind === :add ? "addc" : "subc"
    output_count = N + 1                         # N words + predicate
    a0 = output_count
    b0 = output_count + N
    carry_slot = output_count + 2N

    lines = String[]
    if carry_in
        push!(lines, "selp.$dt cc_tmp, 1, 0, \$$carry_slot;")
        if kind === :add
            push!(lines, "add.cc.$dt cc_tmp, cc_tmp, -1;")
        else
            push!(lines, "sub.cc.$dt cc_tmp, 0, cc_tmp;")
        end
        push!(lines, "$consumer.cc.$dt \$0, \$$a0, \$$b0;")
    else
        push!(lines, "$kind.cc.$dt \$0, \$$a0, \$$b0;")
    end
    for i in 2:N
        push!(lines, "$consumer.cc.$dt \$$(i - 1), " *
                     "\$$(a0 + i - 1), \$$(b0 + i - 1);")
    end
    materialize = kind === :add ? "addc" : "subc"
    push!(lines, "$materialize.$dt cc_tmp, 0, 0;")
    push!(lines, "setp.ne.$dt \$$N, cc_tmp, 0;")

    asm = "{ .reg .$dt cc_tmp; " * join(lines, " ") * " }"
    # Every word output is early-clobber: low limbs are written while later
    # high-limb inputs are still live inside this multi-instruction template.
    outputs = vcat(fill("=&$letter", N), ["=b"])
    inputs = vcat(fill(letter, 2N), carry_in ? ["b"] : String[])
    constraints = join(vcat(outputs, inputs, ["~{cc}"]), ",")
    rettype = Tuple{ntuple(_ -> T, N)..., Bool}
    argtype = Tuple{ntuple(_ -> T, 2N)...,
                    (carry_in ? (Bool,) : ())...}
    (; asm, constraints, rettype, argtype)
end

function _addsub_aggregate_expr(kind::Symbol, N::Int, ::Type{T},
                                carry_in::Bool,
                                carry_name::Symbol = :carry) where {T<:ExtendedPrecisionWord}
    spec = _addsub_aggregate_spec(kind, N, T, carry_in)
    args = Any[:(a[$i]) for i in 1:N]
    append!(args, [:(b[$i]) for i in 1:N])
    carry_in && push!(args, carry_name)
    call = _extended_asmcall_expr(spec.asm, spec.constraints,
                                  spec.rettype, spec.argtype, args)
    words = Expr(:tuple, [:(flat[$i]) for i in 1:N]...)
    quote
        Base.@inline
        flat = $call
        ($words, flat[$(N + 1)])
    end
end

# The methods spell `Tuple{T, Vararg{T, N}}` rather than `NTuple{N, T}`: the
# empty tuple is outside the domain (no limbs, nothing to add), and this
# binds `T` from the first limb so no method has an unbound type parameter
# (Aqua). NOTE: a comment must not sit between the docstring and the first
# definition — it silently detaches the docstring from the binding.
"""
    add_with_carry(a::NTuple{N,T}, b::NTuple{N,T}[, carry::Bool])
        -> (words::NTuple{N,T}, carry_out::Bool)

Add two little-endian limb tuples and return the same-width sum plus carry-out.
An optional explicit carry-in is accepted. `T` may be `UInt32`, `Int32`,
`UInt64`, or `Int64`; PTX defines identical carry behavior for signed and
unsigned add. The complete straight-line sequence is emitted as one opaque,
non-convergent inline-assembly call so `CC.CF` never crosses an LLVM boundary.
"""
@generated add_with_carry(a::Tuple{T, Vararg{T, N}}, b::Tuple{T, Vararg{T, N}}) where
        {N,T<:ExtendedPrecisionWord} = _addsub_aggregate_expr(:add, N + 1, T, false)

@generated add_with_carry(a::Tuple{T, Vararg{T, N}}, b::Tuple{T, Vararg{T, N}},
                          carry::Bool) where
        {N,T<:ExtendedPrecisionWord} = _addsub_aggregate_expr(:add, N + 1, T, true)

"""
    sub_with_borrow(a::NTuple{N,T}, b::NTuple{N,T}[, borrow::Bool])
        -> (words::NTuple{N,T}, borrow_out::Bool)

Subtract little-endian limb tuple `b` from `a`, returning the same-width
difference and borrow-out. An optional explicit borrow-in is accepted. The
entire sequence is one side-effecting, non-convergent asm block; see
[`add_with_carry`](@ref) for supported limb types.
"""
@generated sub_with_borrow(a::Tuple{T, Vararg{T, N}}, b::Tuple{T, Vararg{T, N}}) where
        {N,T<:ExtendedPrecisionWord} = _addsub_aggregate_expr(:sub, N + 1, T, false)

@generated sub_with_borrow(a::Tuple{T, Vararg{T, N}}, b::Tuple{T, Vararg{T, N}},
                           borrow::Bool) where
        {N,T<:ExtendedPrecisionWord} =
    _addsub_aggregate_expr(:sub, N + 1, T, true, :borrow)

function _mul_wide_spec(::Type{T}) where {T<:Union{UInt32, UInt64}}
    dt, letter = _extended_precision_dtype(T)
    # PTX 9.3 §9.7.2.6 documents this 2x2-limb sequence for u32. It computes a
    # full unsigned product [r3,r2,r1,r0] from little-endian [a1,a0] and
    # [b1,b0]; substituting the independently legal u64 forms generalizes it
    # to 128x128 -> 256 bits.
    asm = "{ " *
        "mul.lo.$dt \$0, \$4, \$6; " *
        "mul.hi.$dt \$1, \$4, \$6; " *
        "mad.lo.cc.$dt \$1, \$5, \$6, \$1; " *
        "madc.hi.$dt \$2, \$5, \$6, 0; " *
        "mad.lo.cc.$dt \$1, \$4, \$7, \$1; " *
        "madc.hi.cc.$dt \$2, \$4, \$7, \$2; " *
        "addc.$dt \$3, 0, 0; " *
        "mad.lo.cc.$dt \$2, \$5, \$7, \$2; " *
        "madc.hi.$dt \$3, \$5, \$7, \$3; }"
    constraints = join(vcat(fill("=&$letter", 4), fill(letter, 4),
                            ["~{cc}"]), ",")
    (; asm, constraints, rettype = NTuple{4,T}, argtype = NTuple{4,T})
end

"""
    mul_wide(a::NTuple{2,T}, b::NTuple{2,T}) -> NTuple{4,T}

Multiply two little-endian, two-limb unsigned integers and return the full
four-limb product. `T` may be `UInt32` or `UInt64`. This embeds the PTX
§9.7.2.6 `UInt32` `mad.cc`/`madc` sequence, generalized to the legal `UInt64`
forms, in one side-effecting, non-convergent asm block. Signed multi-limb
multiplication is intentionally not inferred from per-limb signed `mad.hi`
semantics.
"""
@generated function mul_wide(a::NTuple{2,T}, b::NTuple{2,T}) where
        {T<:Union{UInt32, UInt64}}
    spec = _mul_wide_spec(T)
    args = Expr[:(a[1]), :(a[2]), :(b[1]), :(b[2])]
    call = _extended_asmcall_expr(spec.asm, spec.constraints,
                                  spec.rettype, spec.argtype, args)
    quote
        Base.@inline
        $call
    end
end
