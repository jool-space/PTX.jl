# Purity / clobber / convergence / bracket classification lives in the form
# registry (src/ledgers/forms.jl) — one central table, opcode defaults with
# mods-prefix overrides. `build_call` consumes it as a FormContract.

# Reading a special register is observable.
function has_special_reg(argtypes)
    for T in argtypes
        T <: SpecialReg && return true
    end
    false
end

# (constraint_letter, argtype, lane) — `lane` is set for tuple args that emit
# a braced register-vector group (one slot per lane).
const InputSlot = Tuple{String, Type, Union{Nothing, Int}, Bool}

function render_arg(::Type{P}, slot::Int, bracket::Bool) where {P <: Core.LLVMPtr}
    op_str = bracket ? "[\$" * string(slot) * "]" : "\$" * string(slot)
    op_str, InputSlot[("l", P, nothing, false)], slot + 1
end

function render_arg(::Type{Address{T}}, slot::Int, ::Bool) where {T}
    is_ptx_address_integer_type(T) || throw(ArgumentError(
        "PTX.Address{$T} is invalid: Address is reserved for 32-/64-bit " *
        "integer carriers; Core.LLVMPtr values remain unwrapped"))
    # An explicit role marker always brackets, independently of the opcode's
    # legacy pointer-wide contract.  Pass the payload type/constraint to LLVM;
    # `_chain_call_expr` unwraps `.value` before the asm call.
    letter = constraint_letter(T)
    "[\$$(slot)]", InputSlot[(letter, T, nothing, true)], slot + 1
end

function render_arg(::Type{SpecialReg{S}}, slot::Int, ::Bool) where {S}
    # `SpecialReg` is intentionally not exported, but direct construction
    # should not resurrect the obsolete `%warpsize` pseudo-register spelling.
    String(S) == IR.LEGACY_WARP_SIZE_SREG &&
        return string(IR.PREDEFINED_IMMEDIATES["WARP_SZ"]), InputSlot[], slot
    return String(S), InputSlot[], slot
end

function render_arg(::Type{Val{V}}, slot::Int, ::Bool) where {V}
    V isa Integer || error("PTX: Val{} arg must be an Integer, got ", V)
    return string(V), InputSlot[], slot
end

function render_arg(::Type{T}, slot::Int, ::Bool) where {T <: Tuple}
    n = fieldcount(T)
    n == 0 && error("PTX: empty tuple arg has no operand mapping")
    types = fieldtypes(T)
    et = first(types)
    all(t -> t === et, types) ||
        error("PTX: heterogeneous tuple args not supported (got $T)")
    letter = constraint_letter(et)
    op_str = "{" * join(("\$" * string(slot + i - 1) for i in 1:n), ", ") * "}"
    slots = InputSlot[(letter, et, i, false) for i in 1:n]
    op_str, slots, slot + n
end

function render_arg(::Type{T}, slot::Int, ::Bool) where {T}
    op_str = "\$" * string(slot)
    op_str, InputSlot[(constraint_letter(T), T, nothing, false)], slot + 1
end

build_head(op::Symbol, mods::Tuple{Vararg{Symbol}}) =
    isempty(mods) ? string(op) : string(op) * "." * join(string.(mods), ".")

function build_ledger_call(::StructuredLedger, schema::StructuredResultSchema,
                           @nospecialize(argtypes),
                           contract::FormContract)
    validate_structured_result_args(schema, argtypes)
    output_types = map(_structured_output_type, schema.outputs)
    rettype = result_type(schema)
    output_operand = length(output_types) == 1 ? "\$0" :
                     join(("\$" * string(i - 1) for i in eachindex(output_types)), "|")
    output_letters = ["=" * constraint_letter(T) for T in output_types]

    operand_strs = String[]
    input_letters = String[]
    passthrough = Type[]
    passthrough_ix = Tuple{Int, Union{Nothing, Int}}[]
    passthrough_unwrap = Bool[]
    slot = length(output_types)
    for (i, T) in enumerate(argtypes)
        # The closed schema knows these are register/immediate operands
        # (isspacep's generic-address value included — §9.7.9.20 takes the
        # address UNbracketed). RAW_CONTRACT's generic pointer-bracketing
        # guess must not turn an otherwise exact raw structured-result call
        # into invalid PTX.
        op_str, slots, slot = render_arg(T, slot, false)
        push!(operand_strs, op_str)
        for (letter, atype, lane, unwrap_address) in slots
            push!(input_letters, letter)
            push!(passthrough, atype)
            push!(passthrough_ix, (i, lane))
            push!(passthrough_unwrap, unwrap_address)
        end
    end

    head = build_head(schema.op, schema.ptxmods)
    asm = head * " " * join([output_operand; operand_strs], ", ") * ";"
    nonpure = !contract.pure || has_special_reg(argtypes)
    cparts = [output_letters; input_letters]
    nonpure && push!(cparts, "~{memory}")
    return (; asm, constraints = join(cparts, ","), side_effects = nonpure,
              convergent = contract.convergent, rettype,
              passthrough_argtypes = Tuple(passthrough),
              passthrough_indices = Tuple(passthrough_ix),
              passthrough_unwrap_address = Tuple(passthrough_unwrap))
end

function build_ledger_call(::MBarrierLedger, schema::MBarrierFormSchema,
                           @nospecialize(argtypes),
                           contract::FormContract)
    variant = validate_mbarrier_args(schema, argtypes)
    rettype = result_type(schema)
    output_operands, output_letters, slot =
        schema.destination === :none ? (String[], String[], 0) :
        schema.destination in (:sink, :remote_sink) ? (["_"], String[], 0) :
        schema.destination === :state ? (["\$0"], ["=l"], 1) :
        schema.destination === :predicate ? (["\$0"], ["=b"], 1) :
        schema.destination === :count ? (["\$0"], ["=r"], 1) :
        schema.destination === :report_pred ?
            (["\$0|\$1"], ["=b", "=b"], 2) :
        schema.destination === :report ?
            (["\$0|\$1", "report_value"], ["=b", "=b", "=h"], 3) :
        error("invalid mbarrier destination shape: ", schema.destination)

    operand_strs = String[]
    input_letters = String[]
    passthrough = Type[]
    passthrough_ix = Tuple{Int, Union{Nothing, Int}}[]
    passthrough_unwrap = Bool[]
    for (i, (kind, T)) in enumerate(zip(variant.operands, argtypes))
        op_str, slots, slot = render_arg(T, slot, false)
        # Address already renders its payload in brackets. LLVMPtr and the
        # legacy bare integer carriers need the schema to supply brackets.
        kind === :address && !(T <: Address) &&
            (op_str = "[" * op_str * "]")
        push!(operand_strs, op_str)
        for (letter, atype, lane, unwrap_address) in slots
            # LLVM retains the addrspace(3) pointer value, while NVPTX's `r`
            # inline-asm constraint selects the 32-bit PTX register required
            # for an explicitly shared address. This is the same intentional
            # exception used by the exact mbarrier wrappers; generic addresses
            # retain the ordinary 64-bit `l` pointer constraint.
            kind === :address && T <: Core.LLVMPtr &&
                schema.space !== :generic && (letter = "r")
            push!(input_letters, letter)
            push!(passthrough, atype)
            push!(passthrough_ix, (i, lane))
            push!(passthrough_unwrap, unwrap_address)
        end
    end

    head = build_head(:mbarrier, schema.ptxmods)
    operands = [output_operands; operand_strs]
    asm = isempty(operands) ? head * ";" :
          head * " " * join(operands, ", ") * ";"
    # PTX 9.3's opaque reportValue is a `.b8` destination (the CUDA API
    # describes mbarrier.layout::v1 status as 1-byte wide). NVPTX has no i8
    # inline-asm constraint, so bridge it through the low byte of a UInt16.
    schema.destination === :report &&
        (asm = "{ .reg .b8 report_value; " * asm *
               " mov.b16 \$2, {report_value, 0}; }")
    constraints = join([output_letters; input_letters; "~{memory}"], ",")
    return (; asm, constraints, side_effects = true,
              convergent = contract.convergent, rettype,
              passthrough_argtypes = Tuple(passthrough),
              passthrough_indices = Tuple(passthrough_ix),
              passthrough_unwrap_address = Tuple(passthrough_unwrap))
end

function build_ledger_call(::VectorLedger, schema::VectorResultSchema,
                           @nospecialize(argtypes),
                           contract::FormContract;
                           sink_mask = nothing)
    validate_vector_result_args(schema, argtypes)
    n = schema.form.lanes
    mask = sink_mask === nothing ? ntuple(_ -> true, n) :
           validate_vector_result_mask(schema, sink_mask)
    live = count(identity, mask)
    lane_type = schema.form.lane_type
    rettype = NTuple{live, lane_type}

    output_operands = String[]
    output_letters = String[]
    output_slot = 0
    b8_bridge = schema.form.op === :multimem &&
                schema.form.lane_kind in (:e4m3, :e5m2)
    bridge_index = 0
    for keep in mask
        if keep
            push!(output_operands,
                  b8_bridge ? "vector_result_lane" * string(bridge_index) :
                              "\$" * string(output_slot))
            push!(output_letters, "=" * constraint_letter(lane_type))
            output_slot += 1
        else
            push!(output_operands, "_")
        end
        bridge_index += 1
    end
    destination = "{" * join(output_operands, ", ") * "}"

    operand_strs = String[]
    input_letters = String[]
    passthrough = Type[]
    passthrough_ix = Tuple{Int, Union{Nothing, Int}}[]
    passthrough_unwrap = Bool[]
    slot = live
    roles = something(vector_result_operand_roles(schema, length(argtypes)))
    for (i, (kind, T)) in enumerate(zip(roles, argtypes))
        op_str, slots, slot = render_arg(T, slot, false)
        # Address already renders its payload in brackets. LLVMPtr and the
        # legacy bare integer carriers need the schema to supply brackets.
        kind === :address && !(T <: Address) && (op_str = "[" * op_str * "]")
        push!(operand_strs, op_str)
        for (letter, atype, lane, unwrap_address) in slots
            push!(input_letters, letter)
            push!(passthrough, atype)
            push!(passthrough_ix, (i, lane))
            push!(passthrough_unwrap, unwrap_address)
        end
    end

    head = build_head(schema.form.op, schema.mods)
    asm = head * " " * join([destination; operand_strs], ", ") * ";"
    if b8_bridge
        # PTX FP8 scalar lanes require actual .b8 destination registers;
        # ptxas rejects the .b16 registers selected by LLVM's smallest (`h`)
        # inline-asm class. Keep the public Julia carrier UInt8, materialize
        # each architectural lane in a block-local .b8 register, then bridge
        # its low byte through a legal .b16 output exactly like report-value
        # mbarrier lowering. LLVM truncates the low byte back to i8.
        moves = ["mov.b16 \$$(i - 1), {vector_result_lane$(i - 1), 0};"
                 for i in 1:n]
        asm = "{ .reg .b8 vector_result_lane<$(n)>; " * asm * " " *
              join(moves, " ") * " }"
    end
    # These three families are observable memory operations even when a caller
    # passes a synthetic contract in a host rendering test. The clobber also
    # prevents raw from pretending that the vector result is pure.
    constraints = join([output_letters; input_letters; "~{memory}"], ",")
    return (; asm, constraints, side_effects = true,
              convergent = contract.convergent, rettype,
              passthrough_argtypes = Tuple(passthrough),
              passthrough_indices = Tuple(passthrough_ix),
              passthrough_unwrap_address = Tuple(passthrough_unwrap))
end

function build_ledger_call(::B128Ledger, schema::B128FormSchema,
                           @nospecialize(argtypes), contract::FormContract)
    validate_b128_form_args(schema, argtypes)
    output_types = schema.result === Nothing ? Type[] :
                   schema.result === B128 ? Type[UInt64, UInt64] :
                   schema.result <: Tuple ? Type[schema.result.parameters...] :
                   Type[schema.result]
    output_letters = ["=" * constraint_letter(T) for T in output_types]
    slot = length(output_types)
    operand_strs = String[]
    input_letters = String[]
    passthrough = Type[]
    passthrough_ix = Tuple{Int, Union{Nothing, Int}}[]
    passthrough_unwrap = Bool[]
    for (i, (kind, T)) in enumerate(zip(schema.operands, argtypes))
        op_str, slots, slot = render_arg(T, slot, false)
        kind === :address && !(T <: Address) && (op_str = "[" * op_str * "]")
        push!(operand_strs, op_str)
        for (letter, atype, lane, unwrap_address) in slots
            push!(input_letters, letter)
            push!(passthrough, atype)
            push!(passthrough_ix, (i, lane))
            push!(passthrough_unwrap, unwrap_address)
        end
    end

    head = build_head(schema.op, schema.mods)
    b128_inputs = findall(==(:b128), schema.operands)
    local_name(i) = "b128_value" * string(i)
    declarations = String[]
    setup = String[]
    for (n, index) in enumerate(b128_inputs)
        push!(declarations, ".reg .b128 " * local_name(n) * ";")
        push!(setup, "mov.b128 " * local_name(n) * ", " * operand_strs[index] * ";")
    end

    instruction = if schema.kind === :mov
        "mov.b128 b128_result, " * local_name(1) * ";"
    elseif schema.kind === :load
        head * " b128_result, " * join(operand_strs, ", ") * ";"
    elseif schema.kind === :store
        args = copy(operand_strs)
        args[findfirst(==(:b128), schema.operands)] = local_name(1)
        head * " " * join(args, ", ") * ";"
    elseif schema.kind in (:exch, :cas)
        args = copy(operand_strs)
        for (n, index) in enumerate(b128_inputs)
            args[index] = local_name(n)
        end
        head * " b128_result, " * join(args, ", ") * ";"
    elseif schema.kind === :query_pred || schema.kind === :query_dim
        head * " \$0, " * local_name(1) * ";"
    elseif schema.kind === :query_v4
        head * " {\$0, \$1, \$2, \$3}, " * local_name(1) * ";"
    else
        error("unknown b128 form kind: ", schema.kind)
    end

    teardown = String[]
    if schema.result === B128
        push!(declarations, ".reg .b128 b128_result;")
        push!(teardown, "mov.b128 {\$0, \$1}, b128_result;")
    end
    asm = isempty(declarations) ? instruction :
          "{ " * join([declarations; setup; instruction; teardown], " ") * " }"
    nonpure = !contract.pure || has_special_reg(argtypes)
    constraints = [output_letters; input_letters]
    nonpure && push!(constraints, "~{memory}")
    (; asm, constraints = join(constraints, ","), side_effects = nonpure,
       convergent = contract.convergent, rettype = schema.result,
       passthrough_argtypes = Tuple(passthrough),
       passthrough_indices = Tuple(passthrough_ix),
       passthrough_unwrap_address = Tuple(passthrough_unwrap))
end

# The `build_ledger_call` methods above are the builder dispatch family: the
# reviewed schema-specific renderers, defined directly on their island
# handles. Islands without a method (immediate, CLC, scalar) lower through
# build_call's generic tail. The builder dispatch order in build_call exists
# only to keep the non-island guards (integer-Address fallback rule, registry
# contract check, explicit-address bracket check) at their historical
# positions between specific builders — `island_of` already guarantees at
# most one builder can fire.

# Pure: no LLVM, no GPU, no @asmcall. Used by both the runtime call site and
# host-side golden tests. `contract` defaults to the registry lookup; pass one
# explicitly for host-side rendering tests, or select the raw tier with the
# distinct `raw=true` signal. Semantic guards are checked first.
function build_call(op::Symbol, mods::Tuple{Vararg{Symbol}}, @nospecialize(argtypes);
                    contract::Union{FormContract, Nothing, Missing} = missing,
                    raw::Bool = false)
    raw && contract !== missing && throw(ArgumentError(
        "PTX.build_call: raw=true selects RAW_CONTRACT internally; " *
        "do not also pass contract=" * repr(contract)))
    # THE island partition (protocol.jl). Historical throw order is kept: the
    # immediate/mbarrier/structured misses (and the immediate island's
    # consult-time argument validation) precede the invalid-Address-marker
    # guard; the CLC/vector/scalar/b128 misses and the CLC/b128 consult-time
    # validations follow it. Consult-time validation runs exactly where the
    # historical hand-written cascade validated (immediate/CLC/b128); the
    # other islands validate inside their dedicated builders or, for scalar,
    # at the generic tail below.
    island = island_of(op, mods)
    s = island === nothing ? nothing : schema(island, op, mods)
    if island isa Union{ImmediateLedger, MBarrierLedger, StructuredLedger}
        s === nothing && throw(miss(island, op, mods))
        island isa ImmediateLedger && validate_ledger_args(island, s, argtypes)
    end
    has_invalid_address_marker(argtypes) && throw(ArgumentError(
        "PTX.Address is reserved for Int32, UInt32, Int64, and UInt64; " *
        "Core.LLVMPtr already preserves its address role and exact typed " *
        "dispatch, so call address(pointer) instead of constructing " *
        "Address{<:Core.LLVMPtr}"))
    if island !== nothing
        s === nothing && throw(miss(island, op, mods))
        island isa Union{CLCLedger, B128Ledger} &&
            validate_ledger_args(island, s, argtypes)
    end
    selected_contract = raw ? RAW_CONTRACT :
                        contract === missing ? form_contract(op, mods) : contract
    uses_implicit_cc(op, mods) && throw(ArgumentError(
        "ptx\"$(build_head(op, mods))\" accesses the implicit PTX CC.CF " *
        "flag, which cannot safely cross an LLVM inline-asm call boundary. " *
        "The raw tier does not repair that hidden dependency. Use the typed " *
        "wrapper with explicit Bool carry/borrow, or PTX.add_with_carry, " *
        "PTX.sub_with_borrow, or PTX.mul_wide for a fused operation."))
    rule = typed_wrapper_only_rule(op, mods)
    rule !== nothing && !raw && throw(ArgumentError(
        "ptx\"$(build_head(op, mods))\" requires an exact typed wrapper: " *
        rule.detail * ". No typed method matched this call, and the generic " *
        "scalar chain cannot preserve its operand/result structure. Check the " *
        "modifier spelling, arity, tuple widths, and carrier types, or use " *
        "ptx\"$(build_head(op, mods))\"raw for an explicit conservative " *
        "textual escape hatch."))
    # The closed mbarrier schema is authoritative for its result and operand
    # roles. Only non-mbarrier chain fallbacks reach the generic structured-
    # address and contract checks below.
    if island isa MBarrierLedger
        selected_contract === nothing && error(
            "mbarrier is missing its reviewed form contract")
        return build_ledger_call(MBarrierLedger(), s, argtypes,
                                 selected_contract)
    end
    island isa B128Ledger &&
        return build_ledger_call(B128Ledger(), s, argtypes,
                                 selected_contract)
    # The audited vector-result schema is likewise authoritative: it validates
    # and brackets its address operand itself, so an integer Address routed to
    # a reviewed vector form must not hit the exact-wrapper-only fallback rule.
    address_rule = has_integer_address(argtypes) && !(island isa VectorLedger) ?
                   structured_address_fallback_rule(op, mods) : nothing
    address_rule === nothing ||
        throw(structured_address_fallback_error(op, mods, address_rule))
    selected_contract === nothing && error(
        "ptx\"$(build_head(op, mods))\": opcode :$op is not in the form registry " *
        "(src/ledgers/forms.jl). The chain default makes optimizer promises (purity, " *
        "memory, convergence) that must be reviewed per form — add a registry " *
        "entry, or use ptx\"...\"raw for the maximally-conservative contract " *
        "(sideeffect + memory clobber + convergent; pointer operands bracketed).")
    island isa StructuredLedger &&
        return build_ledger_call(StructuredLedger(), s, argtypes,
                                 selected_contract)
    island isa VectorLedger &&
        return build_ledger_call(VectorLedger(), s, argtypes,
                                 selected_contract)
    has_explicit_address(argtypes) && !selected_contract.brackets &&
        throw(ArgumentError(
            "ptx\"$(build_head(op, mods))\" has no reviewed bracketed-address " *
             "operand role; PTX.address(...) is only accepted by memory/address " *
             "forms. Passing it here would emit invalid square brackets."))
    island isa ScalarLedger && validate_scalar_result_args(s, argtypes)
    # Immediate side-effect forms have no PTX destination even on the raw
    # tier, whose generic contract otherwise assumes a scalar result.
    rettype = island isa ImmediateLedger && !s.returns ? Nothing :
              selected_contract.returns ? infer_rettype(op, mods) : Nothing
    nonpure = !selected_contract.pure || has_special_reg(argtypes)
    bracket = selected_contract.brackets
    head = build_head(op, mods)

    operand_strs   = String[]
    input_letters  = String[]
    passthrough    = Type[]
    passthrough_ix = Tuple{Int, Union{Nothing, Int}}[]
    passthrough_unwrap = Bool[]
    slot = (rettype === Nothing) ? 0 : 1     # `$0` reserved for output

    for (i, T) in enumerate(argtypes)
        op_str, slots, slot = render_arg(T, slot, bracket)
        push!(operand_strs, op_str)
        for (letter, atype, lane, unwrap_address) in slots
            push!(input_letters, letter)
            push!(passthrough, atype)
            push!(passthrough_ix, (i, lane))
            push!(passthrough_unwrap, unwrap_address)
        end
    end

    full_operands = rettype === Nothing ? operand_strs : ["\$0"; operand_strs]
    asm = isempty(full_operands) ? head * ";" : head * " " * join(full_operands, ", ") * ";"

    cparts = String[]
    rettype === Nothing || push!(cparts, "=" * constraint_letter(rettype))
    append!(cparts, input_letters)
    nonpure && push!(cparts, "~{memory}")
    constraints = join(cparts, ",")

    return (; asm, constraints, side_effects = nonpure,
              convergent = selected_contract.convergent, rettype,
              passthrough_argtypes = Tuple(passthrough),
              passthrough_indices  = Tuple(passthrough_ix),
              passthrough_unwrap_address = Tuple(passthrough_unwrap))
end
