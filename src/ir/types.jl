# Ported from pyptx/pyptx/ir/types.py (https://github.com/patrick-toulme/pyptx).
# Copyright 2026 Patrick Toulmé. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Translated to Julia and adapted.

# `@enumx` nests values under their enum's name (`ScalarType.F32`,
# `StateSpace.SHARED`) — avoids the bare-name pollution `Base.@enum` would
# inflict (would clash with `Base.EOF`, `Base.CONST`, etc.).
using EnumX: @enumx

@enumx ScalarType begin
    B8;  B16;  B32;  B64;  B128
    U8;  U16;  U32;  U64
    S8;  S16;  S32;  S64
    F16; F16X2
    BF16; BF16X2
    TF32
    F32; F64
    E4M3; E5M2
    PRED
end

const SCALAR_TYPE_NAMES = Dict{ScalarType.T, String}(
    ScalarType.B8 => "b8", ScalarType.B16 => "b16", ScalarType.B32 => "b32",
    ScalarType.B64 => "b64", ScalarType.B128 => "b128",
    ScalarType.U8 => "u8", ScalarType.U16 => "u16",
    ScalarType.U32 => "u32", ScalarType.U64 => "u64",
    ScalarType.S8 => "s8", ScalarType.S16 => "s16",
    ScalarType.S32 => "s32", ScalarType.S64 => "s64",
    ScalarType.F16 => "f16", ScalarType.F16X2 => "f16x2",
    ScalarType.BF16 => "bf16", ScalarType.BF16X2 => "bf16x2",
    ScalarType.TF32 => "tf32",
    ScalarType.F32 => "f32", ScalarType.F64 => "f64",
    ScalarType.E4M3 => "e4m3", ScalarType.E5M2 => "e5m2",
    ScalarType.PRED => "pred",
)

const SCALAR_TYPE_FROM_PTX = Dict{String, ScalarType.T}(v => k for (k, v) in SCALAR_TYPE_NAMES)

ptx(t::ScalarType.T) = "." * SCALAR_TYPE_NAMES[t]

function scalar_type_from_ptx(text::AbstractString)
    raw = startswith(text, ".") ? SubString(text, 2) : text
    haskey(SCALAR_TYPE_FROM_PTX, String(raw)) ||
        throw(ArgumentError("unknown scalar type: $text"))
    SCALAR_TYPE_FROM_PTX[String(raw)]
end

@enumx StateSpace begin
    REG; SREG; CONST; GLOBAL; LOCAL; PARAM
    SHARED; SHARED_CTA; SHARED_CLUSTER
end

const STATE_SPACE_NAMES = Dict{StateSpace.T, String}(
    StateSpace.REG => "reg", StateSpace.SREG => "sreg",
    StateSpace.CONST => "const", StateSpace.GLOBAL => "global",
    StateSpace.LOCAL => "local", StateSpace.PARAM => "param",
    StateSpace.SHARED => "shared",
    StateSpace.SHARED_CTA => "shared::cta",
    StateSpace.SHARED_CLUSTER => "shared::cluster",
)

const STATE_SPACE_FROM_PTX = Dict{String, StateSpace.T}(v => k for (k, v) in STATE_SPACE_NAMES)

ptx(s::StateSpace.T) = "." * STATE_SPACE_NAMES[s]

function state_space_from_ptx(text::AbstractString)
    raw = startswith(text, ".") ? SubString(text, 2) : text
    haskey(STATE_SPACE_FROM_PTX, String(raw)) ||
        throw(ArgumentError("unknown state space: $text"))
    STATE_SPACE_FROM_PTX[String(raw)]
end

# PTX ISA 9.3 §5.4.2 defines declaration vectors as an independent shape
# modifier, not as part of the scalar element type.  Keep the reviewed matrix
# explicit: alternate floating-point formats are deliberately absent because
# §5.2.3 says they are not fundamental types, even when their storage width
# would fit the vector-size limit.
@enumx VectorShape begin
    V2; V4
end

const VECTOR_SHAPE_NAMES = Dict{VectorShape.T, String}(
    VectorShape.V2 => "v2",
    VectorShape.V4 => "v4",
)
const VECTOR_SHAPE_FROM_PTX =
    Dict{String, VectorShape.T}(v => k for (k, v) in VECTOR_SHAPE_NAMES)

ptx(s::VectorShape.T) = "." * VECTOR_SHAPE_NAMES[s]

function vector_shape_from_ptx(text::AbstractString)
    raw = startswith(text, ".") ? SubString(text, 2) : text
    haskey(VECTOR_SHAPE_FROM_PTX, String(raw)) ||
        throw(ArgumentError("unknown vector shape: $text"))
    VECTOR_SHAPE_FROM_PTX[String(raw)]
end

const VECTOR_DECLARATION_TYPES = Dict{VectorShape.T, Tuple}(
    VectorShape.V2 => (
        ScalarType.B8, ScalarType.B16, ScalarType.B32, ScalarType.B64,
        ScalarType.U8, ScalarType.U16, ScalarType.U32, ScalarType.U64,
        ScalarType.S8, ScalarType.S16, ScalarType.S32, ScalarType.S64,
        ScalarType.F16, ScalarType.F16X2, ScalarType.F32, ScalarType.F64,
    ),
    VectorShape.V4 => (
        ScalarType.B8, ScalarType.B16, ScalarType.B32,
        ScalarType.U8, ScalarType.U16, ScalarType.U32,
        ScalarType.S8, ScalarType.S16, ScalarType.S32,
        ScalarType.F16, ScalarType.F16X2, ScalarType.F32,
    ),
)

const VECTOR_DECLARATION_STATE_SPACES = (
    StateSpace.REG, StateSpace.GLOBAL, StateSpace.CONST,
    StateSpace.LOCAL, StateSpace.SHARED,
)

vector_declaration_legal(shape::VectorShape.T, type::ScalarType.T,
                         state_space::StateSpace.T) =
    type in VECTOR_DECLARATION_TYPES[shape] &&
    state_space in VECTOR_DECLARATION_STATE_SPACES

function validate_vector_declaration(shape::VectorShape.T,
                                     type::ScalarType.T,
                                     state_space::StateSpace.T)
    vector_declaration_legal(shape, type, state_space) && return nothing
    throw(ArgumentError("illegal PTX vector declaration $(ptx(state_space)) " *
                        "$(ptx(shape)) $(ptx(type)); PTX ISA 9.3 §5.4.2 " *
                        "requires v2/v4 fundamental non-predicate elements, " *
                        "at most 128 bits overall, in a declaration state space"))
end

@enumx LinkingDirective begin
    VISIBLE; EXTERN; WEAK; COMMON
end

const LINKING_NAMES = Dict{LinkingDirective.T, String}(
    LinkingDirective.VISIBLE => "visible",
    LinkingDirective.EXTERN  => "extern",
    LinkingDirective.WEAK    => "weak",
    LinkingDirective.COMMON  => "common",
)

const LINKING_FROM_PTX = Dict{String, LinkingDirective.T}(v => k for (k, v) in LINKING_NAMES)

ptx(l::LinkingDirective.T) = "." * LINKING_NAMES[l]

function linking_from_ptx(text::AbstractString)
    raw = startswith(text, ".") ? SubString(text, 2) : text
    haskey(LINKING_FROM_PTX, String(raw)) ||
        throw(ArgumentError("unknown linking directive: $text"))
    LINKING_FROM_PTX[String(raw)]
end
