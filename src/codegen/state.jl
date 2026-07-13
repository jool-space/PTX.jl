mutable struct CodeGenState
    io::IOBuffer
    indent::Int
    reg_decls::Dict{String, RegDecl}     # base name "%r" → decl (count, type)
    declared::Set{String}                 # Julia local-var names already bound
    predicated_assigns::Set{String}       # regs assigned under predication — need `local` hoist
    param_names::Vector{String}           # PTX param order → Julia arg name
    # Names of `.shared` / `.shared::cta` VarDecls translated to
    # `CuStaticSharedArray`. Stored stripped of any `%`/`.` mangling — i.e.
    # the form `julia_var` produces.
    shared_vars::Set{String}
    # Register-name (stripped form) → Julia expression string. Populated when
    # `mov`/`add` stash a translated shared-pointer (or offset thereof) into a
    # register; consulted at every operand-render site so subsequent uses of
    # the register flatten to the substituted expression and the defining
    # instruction is dropped entirely.
    pointer_aliases::Dict{String, String}
end

CodeGenState() = CodeGenState(
    IOBuffer(), 0,
    Dict{String, RegDecl}(),
    Set{String}(),
    Set{String}(),
    String[],
    Set{String}(),
    Dict{String, String}(),
)

emit!(cg::CodeGenState, line::AbstractString) =
    println(cg.io, "    "^cg.indent, line)

blank!(cg::CodeGenState) = println(cg.io)

indent!(cg::CodeGenState)   = (cg.indent += 1; nothing)
dedent!(cg::CodeGenState)   = (cg.indent -= 1; nothing)

const JULIA_RESERVED = Set{String}([
    "function", "end", "module", "baremodule", "begin", "if", "else", "elseif",
    "for", "while", "do", "try", "catch", "finally", "return", "break", "continue",
    "let", "local", "global", "const", "quote", "true", "false",
    "import", "using", "export", "abstract", "primitive", "struct", "mutable",
    "type", "macro", "in", "isa", "where",
])

function julia_var(name::AbstractString)
    s = lstrip(name, '%')
    s = replace(s, '.' => '_')
    # Compiler-generated symbol names (e.g. `shmem5_$_0`) carry `$` which is
    # parse-error in Julia outside string interpolation. Stripping is safe:
    # PTX never gives two different symbols the same `$`-stripped spelling.
    s = replace(s, '$' => "")
    s = _demangle_julia_name(s)
    s in JULIA_RESERVED && (s *= "_")
    s
end

# CUDA.jl mangles names as `julia_<sanitized>_<hash>` (param suffix
# `_param_<N>`). Strip them so output reads as `rms_norm_v4_kernel(...)`.
# Edge case: user code naming a variable `julia_foo_123` would also be
# stripped — accepted as a sugar trade-off.
function _demangle_julia_name(name::AbstractString)
    m = match(r"^julia_+.+?_+\d+_param_(\d+)$", name)
    m === nothing || return "param_" * String(m.captures[1])
    m = match(r"^julia_+(.+?)_+(\d+)$", name)
    m === nothing || return String(m.captures[1])
    String(name)
end

function julia_label(name::AbstractString)
    s = replace(name, '$' => "")
    s = replace(s, '.' => '_')
    isempty(s) && return "L_anon"
    if !isletter(first(s)) && first(s) != '_'
        s = "L_" * s
    end
    s
end

# The reviewed PTX 9.3 ledger lives in IR. Canonicalization uses its full
# inventory; codegen intentionally uses only scalar/component spellings until
# vector-valued PTX operands have structured parsing and lowering support.
const SPECIAL_REGS = IR.SCALAR_SPECIAL_REGS

sreg_val_expr(name::AbstractString) = "sreg\"" * name * "\""

const SCALAR_TO_JULIA = Dict{ScalarType.T, Symbol}(
    ScalarType.F64  => :Float64, ScalarType.F32  => :Float32, ScalarType.F16  => :Float16,
    ScalarType.BF16 => :Float16,
    ScalarType.U8   => :UInt8,   ScalarType.U16  => :UInt16,
    ScalarType.U32  => :UInt32,  ScalarType.U64  => :UInt64,
    ScalarType.S8   => :Int8,    ScalarType.S16  => :Int16,
    ScalarType.S32  => :Int32,   ScalarType.S64  => :Int64,
    ScalarType.B8   => :UInt8,   ScalarType.B16  => :UInt16,
    ScalarType.B32  => :UInt32,  ScalarType.B64  => :UInt64,
    ScalarType.PRED => :Bool,
)

scalar_to_julia(t::ScalarType.T) = get(SCALAR_TO_JULIA, t, :UInt32)

const MODIFIER_TO_JULIA_TYPE = Dict{Symbol, String}(
    :f64  => "Float64", :f32  => "Float32", :f16  => "Float16",
    :bf16 => "Float16",
    :u8   => "UInt8",   :u16  => "UInt16",  :u32  => "UInt32", :u64 => "UInt64",
    :s8   => "Int8",    :s16  => "Int16",   :s32  => "Int32",  :s64 => "Int64",
    :b8   => "UInt8",   :b16  => "UInt16",  :b32  => "UInt32", :b64 => "UInt64",
    :pred => "Bool",
)

# Opcodes whose trailing modifier is the destination dtype, not the source —
# wrapping immediates with it would be wrong.
const NO_IMMEDIATE_TYPE_HINT_OPCODES = Set{String}((
    "cvt",       # cvt.<dst>.<src>
))

function operand_type_hint(opcode::AbstractString,
                           modifiers::Tuple{Vararg{AbstractString}})
    opcode in NO_IMMEDIATE_TYPE_HINT_OPCODES && return nothing
    isempty(modifiers) && return nothing
    sym = Symbol(lstrip(modifiers[end], '.'))
    haskey(MODIFIER_TO_JULIA_TYPE, sym) ? sym : nothing
end
