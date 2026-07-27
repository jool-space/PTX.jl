# --- Lowering reflection ------------------------------------------------------
#
# `lowering(op, argtypes)` answers "what will this call actually become?"
# without compiling for a device. Chain-vs-wrapper is a dispatch question
# (`which`); wrapper tier is a typed-IR question: tier-2 wrappers reference
# an `NVVM.IntrinsicCall{name}` singleton whose name is a *type parameter*,
# recoverable structurally, and the remaining wrappers split on whether the
# body's llvmcall IR contains an inline-`asm` call (asm tier) or plain
# target-independent instructions (tier-1 core IR).

# The two chain-default methods, captured for identity comparison. `add.f32`
# never gets a wrapper, so `which` resolves both at load time.
const _CHAIN_METHOD     = which(Operation{:add, (:f32,)}(),    (Float32, Float32))
const _RAW_CHAIN_METHOD = which(RawOperation{:add, (:f32,)}(), (Float32, Float32))

function _walk_intrinsics!(names::Vector{String}, @nospecialize(x))
    if x isa NVVM.IntrinsicCall
        push!(names, String(typeof(x).parameters[1]))
    elseif x isa QuoteNode
        _walk_intrinsics!(names, x.value)
    elseif x isa GlobalRef
        if isdefined(x.mod, x.name)
            v = getglobal(x.mod, x.name)
            v isa NVVM.IntrinsicCall &&
                push!(names, String(typeof(v).parameters[1]))
        end
    elseif x isa Expr
        for a in x.args
            _walk_intrinsics!(names, a)
        end
    end
    return nothing
end

"""
    lowering(op::Operation, argtypes) -> NamedTuple

Reflect on how calling `op` with arguments of the given types will lower,
without a device. `argtypes` is a tuple of types or a `Tuple{...}` type.
Returns `(; tier, method, rettype, intrinsics, asm)`:

- `tier = :intrinsic` — a wrapper routes to `llvm.nvvm.*` intrinsics (tier 2);
  their names are in `intrinsics`, and `NVVM.intrinsic(name)` has each record.
- `tier = :core` — a wrapper emits target-independent LLVM IR (tier 1):
  a real `fence`, `load`/`store`, etc. No intrinsic, no asm.
- `tier = :asm` — a hand-written wrapper embeds inline PTX asm (no intrinsic
  or core-IR spelling exists at the pinned backend).
- `tier = :chain_asm` — no wrapper: the chain default renders inline asm
  from the mods and argument types under the form registry's contract
  (`RAW_CONTRACT` for a `RawOperation`); the text is in `asm`.
- `tier = :unregistered` — no wrapper and the opcode is not in the form
  registry: the call errors at the blessing boundary.
- `tier = :forbidden` — the selected generic path is unsafe: a spelling
  accesses implicit architectural state that cannot cross the call boundary,
  a typed-wrapper-only form missed its exact method, or a closed structured-,
  vector-, scalar-result, or immediate grammar island missed its audited ABI
  or constant domain. Explicit raw remains available for the typed-wrapper
  structural case, but not hidden state, an unknown result ABI, or an invalid
  ISA-required immediate.

Binding is not selectability: an `:intrinsic` form can still fail ISel below
its capability floor — that gate lives in the backend, not the registry.
"""
function lowering(o::Union{Operation, RawOperation}, @nospecialize(argtypes))
    argts = argtypes isa Type ? Tuple(argtypes.parameters) : Tuple(argtypes)
    op, mods = typeof(o).parameters
    m = which(o, Tuple{argts...})
    if m === _CHAIN_METHOD || m === _RAW_CHAIN_METHOD
        raw = m === _RAW_CHAIN_METHOD
        has_invalid_address_marker(argts) &&
            return (; tier = :forbidden, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        b128 = b128_form_schema(op, mods)
        address_rule = has_integer_address(argts) &&
                       vector_result_schema(op, mods) === nothing &&
                       b128 === nothing ?
                       structured_address_fallback_rule(op, mods) : nothing
        address_rule === nothing ||
            return (; tier = :forbidden, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        uses_implicit_cc(op, mods) &&
            return (; tier = :forbidden, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        immediate = immediate_form_contract(op, mods)
        immediate === nothing && requires_immediate_form_contract(op) &&
            return (; tier = :forbidden, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        if immediate !== nothing && !immediate.delegated
            try
                validate_immediate_form_args(immediate, argts)
            catch err
                err isa ArgumentError || rethrow()
                return (; tier = :forbidden, method = m, rettype = nothing,
                          intrinsics = String[], asm = nothing)
            end
        end
        if requires_mbarrier_schema(op)
            mbarrier = mbarrier_form_schema(op, mods)
            mbarrier === nothing &&
                return (; tier = :forbidden, method = m, rettype = nothing,
                          intrinsics = String[], asm = nothing)
            try
                validate_mbarrier_args(mbarrier, argts)
            catch err
                err isa ArgumentError || rethrow()
                return (; tier = :forbidden, method = m, rettype = nothing,
                          intrinsics = String[], asm = nothing)
            end
        end
        structured = structured_result_schema(op, mods)
        structured === nothing && requires_structured_result_schema(op) &&
            return (; tier = :forbidden, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        if structured !== nothing
            try
                validate_structured_result_args(structured, argts)
            catch err
                err isa ArgumentError || rethrow()
                return (; tier = :forbidden, method = m, rettype = nothing,
                          intrinsics = String[], asm = nothing)
            end
        end
        vector_result_schema(op, mods) === nothing &&
            requires_vector_result_schema(op, mods) &&
            return (; tier = :forbidden, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        vector = vector_result_schema(op, mods)
        if vector !== nothing
            try
                validate_vector_result_args(vector, argts)
            catch err
                err isa ArgumentError || rethrow()
                return (; tier = :forbidden, method = m, rettype = nothing,
                          intrinsics = String[], asm = nothing)
            end
        end
        b128 === nothing && requires_b128_form_schema(op, mods) &&
            return (; tier = :forbidden, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        if b128 !== nothing
            try
                validate_b128_form_args(b128, argts)
            catch err
                err isa ArgumentError || rethrow()
                return (; tier = :forbidden, method = m, rettype = nothing,
                          intrinsics = String[], asm = nothing)
            end
        end
        !raw && requires_typed_wrapper(op, mods) &&
            return (; tier = :forbidden, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        scalar_result_schema(op, mods) === nothing &&
            requires_scalar_result_schema(op, mods) &&
            return (; tier = :forbidden, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        clc_schema = op === :clusterlaunchcontrol ?
                     clc_try_cancel_schema(mods) : nothing
        if op === :clusterlaunchcontrol
            clc_schema === nothing && requires_clc_try_cancel_schema(op, mods) &&
                return (; tier = :forbidden, method = m, rettype = nothing,
                          intrinsics = String[], asm = nothing)
            if clc_schema !== nothing
                try
                    validate_clc_try_cancel_args(clc_schema, argts)
                catch err
                    err isa ArgumentError || rethrow()
                    return (; tier = :forbidden, method = m, rettype = nothing,
                              intrinsics = String[], asm = nothing)
                end
            end
        end
        schema = scalar_result_schema(op, mods)
        if schema !== nothing
            try
                validate_scalar_result_args(schema, argts)
            catch err
                err isa ArgumentError || rethrow()
                return (; tier = :forbidden, method = m, rettype = nothing,
                          intrinsics = String[], asm = nothing)
            end
        end
        contract = raw ? RAW_CONTRACT : form_contract(op, mods)
        contract === nothing &&
            return (; tier = :unregistered, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        has_explicit_address(argts) && !contract.brackets &&
            return (; tier = :forbidden, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        if b128 === nothing && _result_abi_error(op, mods) !== nothing
            return (; tier = :forbidden, method = m, rettype = nothing,
                      intrinsics = String[], asm = nothing)
        end
        # All policy failures have been classified above. Do not catch a broad
        # ArgumentError here: rendering/constraint bugs must remain visible to
        # callers of this introspection API rather than masquerading as policy.
        spec = raw ? build_call(op, mods, argts; raw = true) :
                     build_call(op, mods, argts; contract)
        return (; tier = :chain_asm, method = m, rettype = spec.rettype,
                  intrinsics = String[], asm = spec.asm)
    end
    # Structural pass: tier-2 wrappers reference IntrinsicCall singletons in
    # their unoptimized body — the intrinsic name is a type parameter.
    ci, rettype = only(Base.code_typed(o, argts; optimize = false))
    names = String[]
    for st in ci.code
        _walk_intrinsics!(names, st)
    end
    # Optimized pass: inlining exposes the llvmcall IR text — catches
    # intrinsics declared in literal IR (belt and braces) and, for the
    # remaining wrappers, splits asm tier from tier-1 core IR. (`@asmcall`
    # bodies only surface their asm string after inlining.)
    s = string(first(only(Base.code_typed(o, argts))))
    for match in eachmatch(r"llvm\.nvvm\.[A-Za-z0-9_.]+", s)
        push!(names, match.match)
    end
    unique!(names)
    isempty(names) ||
        return (; tier = :intrinsic, method = m, rettype,
                  intrinsics = names, asm = nothing)
    # `call <ty> asm ...` is the LLVM inline-asm spelling inside llvmcall IR.
    tier = occursin(" asm ", s) ? :asm : :core
    return (; tier, method = m, rettype, intrinsics = names, asm = nothing)
end
