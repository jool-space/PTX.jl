# Shared emission body for Operation (registry contract) and RawOperation
# (RAW_CONTRACT). Convergent forms route through convergent_asm_ir so the
# call site carries `convergent nomerge` — @asmcall cannot attach call-site
# attributes, and `sideeffect` alone permits duplicating a collective op
# across a divergent branch (the activemask miscompile class).

function _chain_call_expr(spec)
    arg_exprs = (
        unwrap_address ? :(getfield(args[$i], :value)) :
        lane === nothing ? :(args[$i]) : :(args[$i][$lane])
        for ((i, lane), unwrap_address) in
            zip(spec.passthrough_indices, spec.passthrough_unwrap_address)
    )
    if spec.convergent
        ir = convergent_asm_ir(spec.asm, spec.constraints, spec.rettype,
                               spec.passthrough_argtypes)
        quote
            Base.@inline
            Base.llvmcall(($ir, "entry"), $(spec.rettype),
                          Tuple{$(spec.passthrough_argtypes...)},
                          $(arg_exprs...))
        end
    else
        quote
            Base.@inline $(LLVM.Interop).@asmcall(
                $(spec.asm), $(spec.constraints), $(spec.side_effects),
                $(spec.rettype),
                Tuple{$(spec.passthrough_argtypes...)},
                $(arg_exprs...))
        end
    end
end

@generated function (::Operation{op, mods})(args::Vararg{Any,N}) where {op, mods, N}
    _chain_call_expr(build_call(op, mods, args))
end

"""
    vector_load(ptx"ld....vN.type", address, Val(mask))

Emit a wide PTX vector load with an explicit destination-lane mask. `true`
returns that lane and `false` emits PTX's `_` sink, which guarantees the
corresponding memory location is not read. Sink lanes are supported only for
the ISA's wide `ld.v8.{b32,u32,s32,f32}` and
`ld.v4.{b64,u64,s64,f64}` forms. The result is the homogeneous tuple of live
lanes in source order. All-false masks are rejected: current ptxas cannot
assemble an all-`_` destination, and eliding a load would require a separate
memory-model audit.
"""
@generated function vector_load(::Operation{:ld, mods}, addr::A,
                                ::Val{mask}) where {mods, A, mask}
    schema = vector_result_schema(:ld, mods)
    schema === nothing && throw(vector_result_schema_miss(:ld, mods))
    spec = _build_vector_result_call(schema, (A,), form_contract(:ld, mods);
                                     sink_mask = mask)
    body = _chain_call_expr(spec)
    quote
        Base.@inline
        args = (addr,)
        $body
    end
end


@generated function vector_load(::Operation{:ld, mods}, addr::A, policy::P,
                                ::Val{mask}) where {mods, A, P, mask}
    schema = vector_result_schema(:ld, mods)
    schema === nothing && throw(vector_result_schema_miss(:ld, mods))
    spec = _build_vector_result_call(schema, (A, P), form_contract(:ld, mods);
                                     sink_mask = mask)
    body = _chain_call_expr(spec)
    quote
        Base.@inline
        args = (addr, policy)
        $body
    end
end

@generated function (::RawOperation{op, mods})(args::Vararg{Any,N}) where {op, mods, N}
    _chain_call_expr(build_call(op, mods, args; raw = true))
end
