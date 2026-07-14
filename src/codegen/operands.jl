# Source-side operand rendering. Destination operands are rendered separately
# (see `render_dst` in instruction.jl) because they may need destructuring
# (PipeOperand) or hoisted `local` decls.

# `type_hint` (a dtype Symbol like `:f32`/`:s32`) wraps integer literals with
# the matching Julia type so chain dispatch matches; other operand kinds ignore.
function _predefined_immediate_expr(name::AbstractString)
    value = get(IR.PREDEFINED_IMMEDIATES, String(name), nothing)
    value === nothing ? nothing : "Val($value)"
end

# Constant expressions are held as raw IR text, so a predefined identifier
# inside one does not reach the standalone LabelOperand path below. Match PTX
# identifier boundaries (which include `$`) rather than a bare substring: a
# user identifier such as `WARP_SZ_limit` must remain untouched.
const _PTX_PREDEFINED_IDENTIFIER = r"^[A-Za-z][A-Za-z0-9_$]*$"
const _LEGACY_WARP_SIZE_TOKEN = r"(?<![A-Za-z0-9_$])%warpsize(?![A-Za-z0-9_$])"

function _predefined_immediate_token_regex(name::AbstractString)
    occursin(_PTX_PREDEFINED_IDENTIFIER, name) ||
        error("PTX predefined immediate $(repr(String(name))) is not an identifier")
    Regex("(?<![A-Za-z0-9_\\\$])" * name * "(?![A-Za-z0-9_\\\$])")
end

function _replace_predefined_immediate_tokens(
        text::AbstractString,
        immediates::AbstractDict{<:AbstractString,<:Integer} = IR.PREDEFINED_IMMEDIATES)
    out = String(text)
    for (name, value) in immediates
        out = replace(out, _predefined_immediate_token_regex(name) => string(value))
    end
    # `%warpsize` is the one legacy pseudo-register spelling. It maps to the
    # standard WARP_SZ immediate rather than belonging in the generic table.
    warp_size = string(get(immediates, "WARP_SZ", IR.PREDEFINED_IMMEDIATES["WARP_SZ"]))
    replace(out, _LEGACY_WARP_SIZE_TOKEN => warp_size)
end

function render_operand(op::RegisterOperand, cg::CodeGenState;
                        type_hint::Union{Symbol, Nothing} = nothing)
    name = op.name
    # Compatibility spelling from older NVVM-facing code. PTX itself spells
    # this standard immediate WARP_SZ, so lower it as an immediate in every
    # instruction position rather than emitting invalid %warpsize PTX.
    name == IR.LEGACY_WARP_SIZE_SREG && return _predefined_immediate_expr("WARP_SZ")
    name in SPECIAL_REGS && return sreg_val_expr(name)
    name in IR.V4_SPECIAL_REG_ROOTS &&
        throw(ArgumentError(
            "PTX codegen does not yet lower vector-valued special register $name; " *
            "use a scalar component such as $name.x or add vector IR/lowering support"))
    jname = julia_var(name)
    haskey(cg.pointer_aliases, jname) && return cg.pointer_aliases[jname]
    jname
end

function render_operand(op::ImmediateOperand, cg::CodeGenState;
                        type_hint::Union{Symbol, Nothing} = nothing)
    # PTX permits WARP_SZ wherever an immediate is allowed, including inside
    # constant expressions such as `(WARP_SZ >> 1)`. Replace the predefined
    # token before turning the raw PTX expression into Julia source.
    text = _replace_predefined_immediate_tokens(op.text)
    # PTX f32 hex literal `0fXXXXXXXX` — decode bit-exactly.
    if length(text) == 10 && lowercase(text[1:2]) == "0f"
        bits = tryparse(UInt32, text[3:end], base = 16)
        bits !== nothing &&
            return "reinterpret(Float32, 0x" *
                   string(bits, base = 16, pad = 8) * ")"
    end
    # PTX f64 hex literal `0dXXXXXXXXXXXXXXXX`.
    if length(text) == 18 && lowercase(text[1:2]) == "0d"
        bits = tryparse(UInt64, text[3:end], base = 16)
        bits !== nothing &&
            return "reinterpret(Float64, 0x" *
                   string(bits, base = 16, pad = 16) * ")"
    end
    # `.b{N}` is a bit-pattern type; PTX accepts negative literals as
    # sign-extended bit patterns (`mov.b32 %r, -1` → 0xFFFFFFFF). Julia's
    # `UInt32(-1)` throws InexactError, so route through reinterpret.
    if type_hint !== nothing
        jt = get(MODIFIER_TO_JULIA_TYPE, type_hint, nothing)
        if jt !== nothing
            hint_str = string(type_hint)
            if startswith(hint_str, "b") && startswith(text, "-")
                signed_t = "Int" * hint_str[2:end]
                return "reinterpret($jt, $signed_t($text))"
            end
            return "$jt($text)"
        end
    end
    text
end

function render_operand(op::LabelOperand, cg::CodeGenState;
                        type_hint::Union{Symbol, Nothing} = nothing)
    # WARP_SZ is a PTX predefined identifier (Table 3), so this precedence
    # cannot shadow a user-defined symbol.
    predefined = _predefined_immediate_expr(op.name)
    predefined !== nothing && return predefined
    # The lexer represents legal bare-name registers as LabelOperand. Once a
    # preceding declaration proves that namespace, use the register renderer
    # everywhere so source operands cannot turn into @label spellings.
    _declared_register(cg, op) === nothing ||
        return render_operand(RegisterOperand(op.name), cg; type_hint)
    # State-space symbols (e.g. a `.shared` decl referenced by name) come
    # through as LabelOperand. If we've translated the symbol, substitute.
    jname = julia_var(op.name)
    haskey(cg.pointer_aliases, jname) && return cg.pointer_aliases[jname]
    julia_label(op.name)
end

function render_operand(op::VectorOperand, cg::CodeGenState;
                        type_hint::Union{Symbol, Nothing} = nothing)
    inner = join((render_operand(e, cg; type_hint) for e in op.elements), ", ")
    "(" * inner * ")"
end

function render_operand(op::AddressOperand, cg::CodeGenState;
                        type_hint::Union{Symbol, Nothing} = nothing)
    base_expr = if startswith(op.base, "%")
        render_operand(RegisterOperand(op.base), cg)
    else
        jb = julia_var(op.base)
        get(cg.pointer_aliases, jb, jb)
    end
    # Preserve brackets as a first-class Julia-side role. `address` marks an
    # integer `%r`/`%rd`, but is identity for LLVMPtr expressions so an
    # address-space-specific typed wrapper still receives its exact pointer
    # type. Structured integer forms without a reviewed Address method fail
    # before the scalar fallback can guess their ABI.
    op.coords === nothing || return base_expr
    op.offset === nothing && return "address(" * base_expr * ")"
    off = String(op.offset)
    # TMA tensor-coord form: `[%rd, {%c0, %c1, ...}]` → render as a Julia
    # tuple `(rd, (c0, c1, ...))` so the output parses.
    stripped = lstrip(off, [',', ' '])
    if startswith(stripped, "{") && endswith(stripped, "}")
        inner = stripped[nextind(stripped, 1):prevind(stripped, lastindex(stripped))]
        coords = [julia_var(strip(c)) for c in eachsplit(inner, ',')]
        return "(" * base_expr * ", (" * join(coords, ", ") * "))"
    end
    "address(" * base_expr * " + " * off * ")"
end

function render_operand(op::ParenthesizedOperand, cg::CodeGenState;
                        type_hint::Union{Symbol, Nothing} = nothing)
    inner = join((render_operand(e, cg; type_hint) for e in op.elements), ", ")
    length(op.elements) == 1 ? "(" * inner * ",)" : "(" * inner * ")"
end

function render_operand(op::NegatedOperand, cg::CodeGenState;
                        type_hint::Union{Symbol, Nothing} = nothing)
    "!" * render_operand(op.operand, cg; type_hint)
end

# PipeOperand is only valid as a destination; if it sneaks into a source
# context, emit a tuple to avoid surprising the user.
function render_operand(op::PipeOperand, cg::CodeGenState;
                        type_hint::Union{Symbol, Nothing} = nothing)
    "(" * render_operand(op.left, cg; type_hint) * ", " *
          render_operand(op.right, cg; type_hint) * ")"
end
