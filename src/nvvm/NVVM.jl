"""
    NVVM

The registry of `llvm.nvvm.*` intrinsics understood by the pinned external
NVPTX backend. This module holds the
hand-written half: the record type, the type-token vocabulary, and the query
API. The data itself lives in `table.jl`, machine-generated from
`IntrinsicsNVVM.td` by `gen/generate_registry.jl` — never edited by hand.

Names and signatures here are stable only within a backend JLL major, by
explicit policy: this vocabulary is upstream's, surfaced honestly.
"""
module NVVM

export Intrinsic, intrinsic, isintrinsic

# --- Type tokens ------------------------------------------------------------
#
# Signatures are stored as tuples of tokens:
#   Symbol            scalar or vector by tblgen VT name (:i32, :f32, :bf16,
#                     :v2f16, :v4i32, ..., :Metadata)
#   PtrTok            pointer in a fixed address space — `ptr(3)` is shared
#   AnyTok            an overload slot (anyptr/anyint/anyfloat); the concrete
#                     type is chosen at the call site and the intrinsic's
#                     callsite name takes a mangling suffix (.p3, .f32, ...)
#   SlotTok           `slot(n)` — repeats the concrete type bound to overload
#                     slot n (LLVMMatchType; contributes no mangling suffix)
#
# Overload slots are numbered by order of AnyTok appearance across
# (ret..., params...).

struct PtrTok; addrspace::Int; end
struct AnyTok; kind::Symbol; end
struct SlotTok; slot::Int; end

ptr(as::Integer) = PtrTok(Int(as))
slot(n::Integer) = SlotTok(Int(n))
const anyptr   = AnyTok(:ptr)
const anyint   = AnyTok(:int)
const anyfloat = AnyTok(:float)

const TypeTok = Union{Symbol, PtrTok, AnyTok, SlotTok}

"""
    llvmtype(tok) -> String

LLVM IR spelling of a concrete type token. Overload tokens (AnyTok, SlotTok)
have no spelling of their own — binding them to concrete types is the
emission layer's job — so they error here.
"""
function llvmtype(tok::Symbol)
    s = get(SCALAR_IR, tok, nothing)
    s !== nothing && return s
    m = Base.match(r"^v(\d+)(i\d+|f\d+|bf16)$", String(tok))
    m !== nothing && return "<" * m.captures[1] * " x " * SCALAR_IR[Symbol(m.captures[2])] * ">"
    error("unknown type token :$tok")
end
# Pointers are spelled TYPED (`i8 addrspace(3)*`), not opaque
# (`ptr addrspace(3)`): Julia ≤ 1.11 parses llvmcall IR in the *device*
# context with typed pointers — the opaque spelling fails to parse there
# and Julia demotes the call site to a runtime trap (a stub kernel, no
# compile error). Julia ≥ 1.12 auto-upgrades typed spellings on parse, so
# one spelling serves every supported version; `i8` matches Julia's own
# LLVMPtr lowering in typed mode, so @entry parameters bind without glue.
# Known-to-old-LLVM intrinsics need an exact pointee — see TYPED_POINTEE
# in emit.jl. Revert to opaque wholesale when Julia ≤ 1.11 support ends.
llvmtype(tok::PtrTok) =
    tok.addrspace == 0 ? "i8*" : "i8 addrspace($(tok.addrspace))*"
llvmtype(tok::AnyTok) =
    error("overload slot ($(tok.kind)) has no fixed LLVM type; bind a concrete type first")
llvmtype(tok::SlotTok) =
    error("slot($(tok.slot)) repeats an overload slot; bind a concrete type first")

const SCALAR_IR = Dict{Symbol,String}(
    :i1 => "i1", :i8 => "i8", :i16 => "i16", :i32 => "i32", :i64 => "i64",
    :i128 => "i128", :f16 => "half", :bf16 => "bfloat", :f32 => "float",
    :f64 => "double", :Metadata => "metadata")

# --- The record -------------------------------------------------------------

"""
One intrinsic record, as extracted from the backend's `IntrinsicsNVVM.td`.

- `name`: the LLVM intrinsic name (base name — overloaded intrinsics take a
  mangling suffix at the call site).
- `ret`, `params`: type-token tuples. Multiple `ret` entries mean a literal
  struct return.
- `props`: intrinsic-level properties, lowercased from tblgen
  (`:convergent`, `:nomem`, `:readmem`, `:writemem`, `:argmemonly`,
  `:inaccessiblememonly`, `:inaccessiblemem_or_argmemonly`, `:speculatable`,
  `:sideeffects`, `:noreturn`, `:nocallback`, `:nofree`, `:willreturn`,
  `:nocreateundefpoison`, `:commutative`). These drive the attribute groups
  the emission layer attaches; `:convergent` is the load-bearing one.
- `immargs`: parameter positions (1-based) that must be immediate constants
  in the emitted IR, not SSA values.
- `ranges`: `(pos, lo, hi)` triples — value is in `[lo, hi)`; position 0 is
  the return value.
- `argnames`: `(pos, name)` pairs — operand names from the .td, where
  present (documentation metadata).
- `argattrs`: `(pos, attr)` pairs for per-position attributes (`:nocapture`,
  `:noalias`, `:noundef`, `:readonly`, `:writeonly`); position 0 is the
  return value.
"""
struct Intrinsic
    name::String
    ret::Tuple{Vararg{TypeTok}}
    params::Tuple{Vararg{TypeTok}}
    props::Tuple{Vararg{Symbol}}
    immargs::Tuple{Vararg{Int}}
    ranges::Tuple{Vararg{Tuple{Int,Int,Int}}}
    argnames::Tuple{Vararg{Tuple{Int,Symbol}}}
    argattrs::Tuple{Vararg{Tuple{Int,Symbol}}}
end

"""
    overloaded(i::Intrinsic) -> Bool

Whether the intrinsic has overload slots, i.e. needs a mangled callsite name.
"""
overloaded(i::Intrinsic) =
    any(t -> t isa AnyTok, i.ret) || any(t -> t isa AnyTok, i.params)

# --- The table --------------------------------------------------------------

const TABLE = Dict{String,Intrinsic}()

function intr(name::String, ret::Tuple, params::Tuple, props::Tuple;
              immargs::Tuple=(), ranges::Tuple=(), argnames::Tuple=(),
              argattrs::Tuple=())
    TABLE[name] = Intrinsic(name, ret, params, props,
                            immargs, ranges, argnames, argattrs)
    return nothing
end

include("table.jl")

# --- Queries ----------------------------------------------------------------

"""
    isintrinsic(name) -> Bool

Whether `name` is in the backend's intrinsic table (by base name, without
any mangling suffix).
"""
isintrinsic(name::AbstractString) = haskey(TABLE, String(name))

"""
    intrinsic(name) -> Intrinsic

Look up an intrinsic by its base name. On a miss, the error suggests
registered names sharing the longest matching dotted prefix — a renamed
intrinsic after a backend bump typically keeps most of its segments.
"""
function intrinsic(name::AbstractString)
    name = String(name)
    i = get(TABLE, name, nothing)
    i !== nothing && return i
    error(_miss_message(name))
end

function _miss_message(name::String)
    segs = split(name, '.')
    for n in length(segs)-1:-1:2
        prefix = join(segs[1:n], '.') * "."
        near = sort!(filter(startswith(prefix), collect(keys(TABLE))))
        if !isempty(near)
            shown = join("  " .* first(near, 12), "\n")
            more = length(near) > 12 ? "\n  … $(length(near) - 12) more" : ""
            return """
                "$name" is not in the backend's intrinsic table.
                Registered names under "$prefix":
                $shown$more"""
        end
    end
    return "\"$name\" is not in the backend's intrinsic table " *
           "(no registered names share a prefix with it)."
end

"""
    matching(prefix) -> Vector{String}

All registered intrinsic names starting with `prefix`, sorted.
"""
matching(prefix::AbstractString) =
    sort!(filter(startswith(String(prefix)), collect(keys(TABLE))))

include("emit.jl")

end # module NVVM
