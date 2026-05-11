# Source-side operand rendering. Destination operands are rendered separately
# (see `render_dst` in instruction.jl) because they may need destructuring
# (PipeOperand) or hoisted `local` decls.

# `type_hint` (a dtype Symbol like `:f32`/`:s32`) wraps integer literals with
# the matching Julia type so chain dispatch matches; other operand kinds ignore.
function render_operand(op::RegisterOperand, cg::CodeGenState;
                        type_hint::Union{Symbol, Nothing} = nothing)
    name = op.name
    name in SPECIAL_REGS && return sreg_val_expr(name)
    jname = julia_var(name)
    haskey(cg.pointer_aliases, jname) && return cg.pointer_aliases[jname]
    jname
end

function render_operand(op::ImmediateOperand, cg::CodeGenState;
                        type_hint::Union{Symbol, Nothing} = nothing)
    text = op.text
    # PTX f32 hex literal `0fXXXXXXXX` — decode bit-exactly.
    if length(text) == 10 && startswith(text, "0f")
        bits = tryparse(UInt32, text[3:end], base = 16)
        bits !== nothing &&
            return "reinterpret(Float32, 0x" *
                   string(bits, base = 16, pad = 8) * ")"
    end
    # PTX f64 hex literal `0dXXXXXXXXXXXXXXXX`.
    if length(text) == 18 && startswith(text, "0d")
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
    op.offset === nothing && return base_expr
    off = String(op.offset)
    # TMA tensor-coord form: `[%rd, {%c0, %c1, ...}]` → render as a Julia
    # tuple `(rd, (c0, c1, ...))` so the output parses.
    stripped = lstrip(off, [',', ' '])
    if startswith(stripped, "{") && endswith(stripped, "}")
        inner = stripped[nextind(stripped, 1):prevind(stripped, lastindex(stripped))]
        coords = [julia_var(strip(c)) for c in eachsplit(inner, ',')]
        return "(" * base_expr * ", (" * join(coords, ", ") * "))"
    end
    base_expr * " + " * off
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
