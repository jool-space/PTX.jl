# --- convergent inline asm ---------------------------------------------------
#
# `@asmcall` cannot attach call-site attributes, and `sideeffect` alone does
# NOT forbid duplicating a call site across a divergent branch (jump
# threading, tail duplication) — only `convergent` does. For warp-/warpgroup-
# collective instructions (wgmma.mma_async and mma.sync fallbacks), a split
# call site means different lanes execute different copies: the `active_mask`
# class of miscompile reproduced on hardware. Mbarrier asm uses this path too,
# matching the convergence contract on the complete llvm.nvvm.mbarrier.*
# surface regardless of dispatch tier. The attribute binds in the in-process
# middle end only — llc neither checks nor needs it — so tests assert emitted
# llvmcall IR rather than ptxas acceptance.
#
# Mechanism validated by spikes/raw_asm_attrs.jl: a `convergent` attribute
# group on an inline-asm call site parses through Base.llvmcall and survives
# the optimized module. The builder emits the same shape `@asmcall` does —
# asm callee returns a scalar or literal struct, entry returns Julia's
# lowering of the return type — plus the call-site attributes. `nomerge`
# accompanies `convergent`: LLVM ≤ 16 (Julia ≤ 1.11) hoists identical
# convergent calls from both arms of a divergent branch into one site — the
# collective-op miscompile. See NVVM.fnattrs.
#
# Keep the complete implementation above `_chain_call_expr`: Julia 1.12
# requires every global called by a generated-function generator to exist in
# the generator's definition world, not merely by the time it is invoked.

# Expression calling side-effecting inline `asm` on the argument expressions
# `args` (of Julia types `argtypes`), returning `rettype`, with the call site
# marked `convergent nomerge nounwind`.
Base.@nospecializeinfer convergent_asmcall(
        asm::String, constraints::String, @nospecialize(rettype::Type),
        @nospecialize(argtypes), @nospecialize(args...)) =
    _asmcall(asm, constraints, rettype, argtypes, args;
             sideeffect = true, attrs = ("convergent", "nomerge", "nounwind"))

# Same call without the convergence contract, for forms whose FORMS entry is
# deliberately non-convergent (tcgen05.mma) or pure (register-only data
# movement). `sideeffect = false` lets LLVM drop or merge the call like any
# other pure computation; pass true for observable operations.
Base.@nospecializeinfer plain_asmcall(
        asm::String, constraints::String, @nospecialize(rettype::Type),
        @nospecialize(argtypes), @nospecialize(args...); sideeffect::Bool = true) =
    _asmcall(asm, constraints, rettype, argtypes, args;
             sideeffect, attrs = ("nounwind",))

# Asm pointer operands have no pointee, so every LLVMPtr is retyped to a
# UInt8 pointee in the same address space (a no-op bitcast). Wrappers that
# are generic over the element type then share one IR per address space.
_asm_operand(@nospecialize(T::Type), @nospecialize(ex)) =
    T <: Core.LLVMPtr ?
        (Core.LLVMPtr{UInt8, T.parameters[2]},
         :(reinterpret(Core.LLVMPtr{UInt8, $(T.parameters[2])}, $ex))) :
        (T, ex)

# Static description of one asm call site, carried as type parameters so the
# emitter's generator depends on nothing but its argument types.
struct AsmSpec{asm, constraints, sideeffect, attrs} end

# Declared type of an emitter argument `x` standing for the `n` operands
# `x[1], …, x[n]`, each of type `T`. Wrappers spread accumulator tuples into
# up to 128 operands; passing the tuple keeps their bodies small, and the
# generator expands it back into the same operands.
struct _Spread{n, T} end

# `ex` is `x[k]` for a variable `x`: return `x`.
_spread_base(ex, k::Int) =
    ex isa Expr && ex.head === :ref && length(ex.args) == 2 &&
    ex.args[1] isa Symbol && ex.args[2] === k ? ex.args[1] : nothing

# Fold each run of operands `x[1], …, x[n]` (n ≥ 2) of one declared type into
# the single argument `x`, declared `_Spread{n, T}`.
function _fold_spreads(@nospecialize(argtypes), @nospecialize(args))
    types = Any[]
    exprs = Any[]
    i = 1
    while i <= length(args)
        x = _spread_base(args[i], 1)
        j = i
        if x !== nothing
            while j < length(args) && argtypes[j + 1] === argtypes[i] &&
                  _spread_base(args[j + 1], j - i + 2) === x
                j += 1
            end
        end
        if j > i
            push!(types, _Spread{j - i + 1, argtypes[i]})
            push!(exprs, x)
        else
            push!(types, argtypes[i])
            push!(exprs, args[i])
        end
        i = j + 1
    end
    types, exprs
end

# The expression a wrapper body embeds: a call to the lazy emitter, which
# builds the `llvmcall` IR when the wrapper is first compiled. Building the IR
# here, for every wrapper as it is defined, is most of the precompile time and
# the package image of the asm-tier families, which define thousands of
# wrappers a kernel uses a handful of.
function _asmcall(asm::String, constraints::String, @nospecialize(rettype::Type),
                  @nospecialize(argtypes), @nospecialize(args); sideeffect::Bool,
                  attrs::Tuple{Vararg{String}})
    length(argtypes) == length(args) ||
        error("asm call: $(length(args)) arguments for $(length(argtypes)) argument types")
    spec = AsmSpec{Symbol(asm), Symbol(constraints), sideeffect, map(Symbol, attrs)}()
    types, exprs = _fold_spreads(argtypes, args)
    call = Expr(:call, GlobalRef(@__MODULE__, :_lazy_asmcall), spec, rettype,
                Tuple{types...}, exprs...)
    device_only(call, asm)
end

# The `llvmcall` expression for an asm call on the argument expressions
# `args`, built by `_lazy_asmcall`'s generator. An argument type
# `_Spread{n, T}` passes the `NTuple{n, T}` its expression evaluates to as one
# `llvmcall` argument, which the IR unpacks into `n` asm operands of type `T`.
# Lowering a call on hundreds of operand expressions (`x[1], …, x[128]`) costs
# several times more than building its IR, so the tuple is not spread in Julia.
function _asm_llvmcall(asm::String, constraints::String, @nospecialize(rettype::Type),
                       @nospecialize(argtypes), @nospecialize(args); sideeffect::Bool,
                       attrs::Tuple{Vararg{String}})
    length(argtypes) == length(args) ||
        error("asm call: $(length(args)) arguments for $(length(argtypes)) argument types")
    # Index instead of `collect`ing: the call sites pass hundreds of distinct
    # tuple types, each of which would compile its own specialization.
    abitypes = Any[]
    argexprs = Any[]
    widths = Int[]
    for i in 1:length(args)
        T = argtypes[i]
        if T <: _Spread
            n, E = T.parameters
            push!(abitypes, NTuple{n, E})
            push!(argexprs, args[i])
            push!(widths, n)
        else
            (Base.isType(T) || Base.issingletontype(T)) &&
                throw(ArgumentError("asm call: argument $i of type $T has no LLVM " *
                                    "representation"))
            T, ex = _asm_operand(T, args[i])
            push!(abitypes, T)
            push!(argexprs, ex)
            push!(widths, 0)
        end
    end
    # The IR is for Julia's code generator, so its types come from Julia's
    # LLVM, also when another library is LLVM.jl's current one.
    ir = LLVM.library!(:julia) do
        _asm_ir(asm, constraints, rettype, abitypes, widths, sideeffect, attrs)
    end
    llvmcall = Expr(:call, GlobalRef(Core.Intrinsics, :llvmcall),
                    Expr(:tuple, ir, "entry"), rettype, Tuple{abitypes...})
    append!(llvmcall.args, argexprs)
    Expr(:block, Expr(:meta, :inline), llvmcall)
end

# The struct a multi-output asm call returns. Textual IR spells a literal
# struct type out at every use, once per output's extractvalue, so its size
# grows with the square of the output count: wgmma's 128 outputs make it most
# of the IR string the method body embeds. Wide structs are named, which
# spells the elements once; narrow ones stay literal, keeping the element
# types readable on the call line.
const _NAMED_ASM_OUTPUTS_MIN = 16

# Textual IR of the `llvmcall` module: an `entry` function calling the inline
# asm, laid out as LLVM prints a module, since tests match it. Only the types
# go through LLVM, taken from Julia's own lowering of each Julia type, which
# fixes the pointer spelling (typed on Julia 1.10) and the tuple lowering.
# Building a module to print and parse it again costs three times as much.
function _asm_ir(asm::String, constraints::String, @nospecialize(rettype::Type),
                 abitypes::Vector{Any}, widths::Vector{Int}, sideeffect::Bool,
                 attrs::Tuple{Vararg{String}})
    @dispose ctx=LLVM.Context() begin
        lowered = IdDict{Any, LLVMType}()
        lower(@nospecialize(T)) = get!(() -> convert(LLVMType, T), lowered, T)
        T_ret = lower(rettype)
        # LLVM dictates the asm's return shape from the number of outputs:
        # one returns the scalar, several a struct. Julia lowers a homogeneous
        # tuple to an array instead, so that shape is repacked.
        comps = rettype <: Tuple ?
            LLVMType[lower(T) for T in rettype.parameters] : nothing
        named = comps !== nothing && length(comps) >= _NAMED_ASM_OUTPUTS_MIN
        T_asm = rettype === Nothing ? LLVM.VoidType() :
                comps === nothing ? T_ret :
                length(comps) == 1 ? only(comps) :
                named ? nothing : LLVM.StructType(comps)
        repack = rettype !== Nothing && T_asm != T_ret
        s_ret = string(T_ret)
        s_asm = T_asm === nothing ? "%asm.outputs" : string(T_asm)

        io = IOBuffer()
        print(io, "; ModuleID = 'llvmcall'\nsource_filename = \"llvmcall\"\n\n")
        if named
            print(io, "%asm.outputs = type { ")
            join(io, (string(T) for T in comps), ", ")
            print(io, " }\n\n")
        end
        print(io, "; Function Attrs: alwaysinline\ndefine ", s_ret, " @entry(")
        s_args = String[string(lower(T)) for T in abitypes]
        for i in eachindex(s_args)
            i > 1 && print(io, ", ")
            print(io, s_args[i], " %", i - 1)
        end
        print(io, ") #0 {\nentry:\n")
        # Unnamed values number on from the parameters.
        nextval = length(abitypes)
        operands = String[]
        for i in eachindex(abitypes)
            if widths[i] == 0
                push!(operands, string(s_args[i], " %", i - 1))
                continue
            end
            T_tuple = lower(abitypes[i])
            T_tuple isa LLVM.ArrayType ||
                error("asm call: spread operand type $(abitypes[i]) lowers to " *
                      "$(string(T_tuple)), not an array")
            s_elem = string(T_tuple.element_type)
            for j in 0:widths[i]-1
                print(io, "  %", nextval, " = extractvalue ", s_args[i], " %", i - 1,
                      ", ", j, "\n")
                push!(operands, string(s_elem, " %", nextval))
                nextval += 1
            end
        end
        _check_constraints(constraints, rettype === Nothing ? 0 :
                                        comps === nothing ? 1 : length(comps),
                           length(operands))
        print(io, "  ")
        if rettype !== Nothing
            call = nextval
            print(io, "%", call, " = ")
            nextval += 1
        end
        print(io, "call ", rettype === Nothing ? "void" : s_asm, " asm ",
              sideeffect ? "sideeffect " : "")
        _escaped_string(io, asm)
        print(io, ", ")
        _escaped_string(io, constraints)
        print(io, "(")
        join(io, operands, ", ")
        print(io, ")", isempty(attrs) ? "" : " #1", "\n")
        if rettype === Nothing
            print(io, "  ret void\n")
        elseif !repack
            print(io, "  ret ", s_ret, " %", call, "\n")
        else
            agg = "undef"
            for i in 0:length(comps)-1
                s_elem = string(comps[i+1])
                elem = call
                if length(comps) > 1
                    print(io, "  %", nextval, " = extractvalue ", s_asm, " %", call,
                          ", ", i, "\n")
                    elem = nextval
                    nextval += 1
                end
                print(io, "  %", nextval, " = insertvalue ", s_ret, " ", agg, ", ",
                      s_elem, " %", elem, ", ", i, "\n")
                agg = string("%", nextval)
                nextval += 1
            end
            print(io, "  ret ", s_ret, " ", agg, "\n")
        end
        print(io, "}\n\nattributes #0 = { alwaysinline }\n")
        # LLVM prints a group's enum attributes in alphabetical order.
        isempty(attrs) ||
            print(io, "attributes #1 = { ", join(sort!(collect(attrs)), " "), " }\n")
        String(take!(io))
    end
end

# Julia parses the IR only when it compiles the `llvmcall`, and turns a parse
# error into a call that throws at run time, which a device compile rejects
# without the parser's message. So check here the mismatch the parser rejects
# an asm call for: a constraint list naming `=` outputs (other than indirect
# `=*` ones, which take an operand) and inputs (`~` clobbers are neither) in
# numbers other than the call returns and passes.
function _check_constraints(constraints::String, nout::Int, nin::Int)
    outs = ins = 0
    for c in eachsplit(constraints, ',')
        if startswith(c, "=*")
            ins += 1
        elseif startswith(c, '=')
            outs += 1
        elseif !startswith(c, '~')
            ins += 1
        end
    end
    outs == nout && ins == nin && return nothing
    error("asm call: constraints \"$constraints\" name $outs outputs and $ins " *
          "inputs, but the call returns $nout and passes $nin")
end

# A string constant as LLVM prints one: `\` doubled, printable ASCII other
# than `"` as is, every other byte as `\` and two uppercase hex digits.
function _escaped_string(io::IO, s::String)
    write(io, '"')
    for b in codeunits(s)
        if b == UInt8('\\')
            write(io, "\\\\")
        elseif 0x20 <= b <= 0x7e && b != UInt8('"')
            write(io, b)
        else
            write(io, '\\', "0123456789ABCDEF"[(b >> 4) + 1],
                  "0123456789ABCDEF"[(b & 0x0f) + 1])
        end
    end
    write(io, '"')
end

# A spread argument passes whole when its value is the operand tuple exactly:
# a tuple of another type is indexed, and LLVMPtr operands are retyped one by
# one.
_passes_whole(@nospecialize(T), n::Int, @nospecialize(E)) =
    T === NTuple{n, E} && isprimitivetype(E) && !(E <: Core.LLVMPtr)

@generated function _lazy_asmcall(::AsmSpec{asm, constraints, sideeffect, attrs},
                                  ::Type{R}, ::Type{AT}, args...) where
                                 {asm, constraints, sideeffect, attrs, R, AT}
    types = Any[]
    exprs = Any[]
    for (k, T) in enumerate(AT.parameters)
        if T <: _Spread && !_passes_whole(args[k], T.parameters[1], T.parameters[2])
            n, E = T.parameters
            for i in 1:n
                push!(types, E)
                push!(exprs, :(args[$k][$i]))
            end
        else
            push!(types, T)
            push!(exprs, :(args[$k]))
        end
    end
    _asm_llvmcall(String(asm), String(constraints), R, types, exprs;
                  sideeffect, attrs = map(String, attrs))
end
