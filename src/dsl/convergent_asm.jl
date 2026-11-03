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
convergent_asmcall(asm::String, constraints::String, @nospecialize(rettype::Type),
                   @nospecialize(argtypes), @nospecialize(args...)) =
    _asmcall(asm, constraints, rettype, argtypes, args;
             sideeffect = true, attrs = ("convergent", "nomerge", "nounwind"))

# Same call without the convergence contract, for forms whose FORMS entry is
# deliberately non-convergent (tcgen05.mma) or pure (register-only data
# movement). `sideeffect = false` lets LLVM drop or merge the call like any
# other pure computation; pass true for observable operations.
plain_asmcall(asm::String, constraints::String, @nospecialize(rettype::Type),
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

function _asmcall(asm::String, constraints::String, @nospecialize(rettype::Type),
                  @nospecialize(argtypes), @nospecialize(args); sideeffect::Bool,
                  attrs::Tuple{Vararg{String}})
    length(argtypes) == length(args) ||
        error("asm call: $(length(args)) arguments for $(length(argtypes)) argument types")
    # Index instead of `collect`ing: the call sites pass hundreds of distinct
    # tuple types, each of which would compile its own specialization.
    abitypes = Any[]
    argexprs = Any[]
    for i in 1:length(args)
        T, ex = _asm_operand(argtypes[i], args[i])
        push!(abitypes, T)
        push!(argexprs, ex)
    end
    call = generate_llvmcall(rettype, Tuple{abitypes...},
                             argexprs...) do builder, @nospecialize(params...)
        T_ret = convert(LLVMType, rettype)
        # LLVM dictates the asm's return shape from the number of outputs:
        # one returns the scalar, several a literal struct. Julia lowers a
        # homogeneous tuple to an array instead, so that shape is repacked.
        comps = rettype <: Tuple ? LLVMType[convert(LLVMType, T)
                                            for T in rettype.parameters] : nothing
        T_asm = rettype === Nothing ? LLVM.VoidType() :
                comps === nothing ? T_ret :
                length(comps) == 1 ? only(comps) : _asm_outputs_type(comps)
        values = Value[params[i] for i in 1:length(params)]
        asm_ft = LLVM.FunctionType(T_asm, LLVMType[v.value_type for v in values])
        call = call!(builder, asm_ft,
                     InlineAsm(asm_ft, asm, constraints, sideeffect), values)
        for attr in attrs
            push!(call.function_attributes, EnumAttribute(attr))
        end
        rettype === Nothing && return nothing
        T_asm == T_ret && return call
        ret = LLVM.UndefValue(T_ret)
        for i in 0:length(comps)-1
            elem = length(comps) == 1 ? call : extract_value!(builder, call, i)
            ret = insert_value!(builder, ret, elem, i)
        end
        ret
    end
    device_only(_inline_arguments(call), asm)
end

# The struct a multi-output asm call returns. Textual IR spells a literal
# struct type out at every use, once per output's extractvalue, so its size
# grows with the square of the output count: wgmma's 128 outputs make it most
# of the IR string the method body embeds. Wide structs are named, which
# spells the elements once; narrow ones stay literal, keeping the element
# types readable on the call line.
const _NAMED_ASM_OUTPUTS_MIN = 16

function _asm_outputs_type(comps::Vector{LLVMType})
    length(comps) < _NAMED_ASM_OUTPUTS_MIN && return LLVM.StructType(comps)
    T = LLVM.StructType("asm.outputs")
    LLVM.elements!(T, comps)
    T
end

# `generate_llvmcall` binds each argument expression to a temporary so it is
# evaluated once and in order. Passing the expressions to the `llvmcall`
# directly keeps both guarantees, since call arguments evaluate once each,
# left to right, and halves the statements a many-operand body (wgmma's up to
# 128 accumulators) lowers to. A temporary the `llvmcall` does not consume (a
# statically known argument) is dropped only when it is a bare variable;
# anything else keeps the block as generated.
function _inline_arguments(block::Expr)
    block.head === :block || return block
    stmts = block.args
    call = stmts[end]
    call isa Expr && call.head === :call || return block
    meta = Any[]
    binds = Dict{Symbol, Any}()
    for st in @view stmts[1:end-1]
        st isa Expr || return block
        if st.head === :meta
            push!(meta, st)
        elseif st.head === :(=) && st.args[1] isa Symbol
            binds[st.args[1]] = st.args[2]
        else
            return block
        end
    end
    used = Set{Symbol}(a for a in call.args if a isa Symbol && haskey(binds, a))
    for (tmp, ex) in binds
        tmp in used || ex isa Symbol || return block
    end
    args = Any[a isa Symbol && haskey(binds, a) ? binds[a] : a for a in call.args]
    Expr(:block, meta..., Expr(:call, args...))
end
