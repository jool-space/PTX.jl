# Back half of the registry generator:
# consume the JSON produced by extract_intrinsics.sh and emit the committed
# Julia table at src/nvvm/table.jl. The output is checked in so a backend
# bump becomes a reviewable diff of renames/removals.
#
# Every mapping in here is closed-world: an unknown property, type token, or
# argument-property kind is an error, never a fallthrough. New vocabulary in
# a future .td must be looked at, not absorbed.
#
# Usage: julia --project=gen gen/generate_registry.jl [gen/nvvm_intrinsics_<v>.json]

using JSON

const GEN = @__DIR__
const SRC = joinpath(dirname(GEN), "src", "nvvm", "table.jl")

input = isempty(ARGS) ? only(filter(f -> occursin(r"^nvvm_intrinsics_.*\.json$", f),
                                    readdir(GEN))) |> f -> joinpath(GEN, f) : ARGS[1]
version = Base.match(r"nvvm_intrinsics_(.*)\.json$", basename(input)).captures[1]

records = JSON.parsefile(input)

# tblgen property name -> registry symbol. Closed set; see Intrinsic docs.
const PROPS = Dict(
    "IntrConvergent"                  => :convergent,
    "IntrNoMem"                       => :nomem,
    "IntrReadMem"                     => :readmem,
    "IntrWriteMem"                    => :writemem,
    "IntrArgMemOnly"                  => :argmemonly,
    "IntrInaccessibleMemOnly"         => :inaccessiblememonly,
    "IntrInaccessibleMemOrArgMemOnly" => :inaccessiblemem_or_argmemonly,
    "IntrSpeculatable"                => :speculatable,
    "IntrHasSideEffects"              => :sideeffects,
    "IntrNoReturn"                    => :noreturn,
    "IntrNoCallback"                  => :nocallback,
    "IntrNoFree"                      => :nofree,
    "IntrWillReturn"                  => :willreturn,
    "IntrNoCreateUndefOrPoison"       => :nocreateundefpoison,
    "Commutative"                     => :commutative,
)

# per-argument property kind -> registry symbol (ImmArg/Range/ArgName are
# routed to their own fields instead)
const ARGATTRS = Dict(
    "NoCapture" => :nocapture,
    "NoAlias"   => :noalias,
    "NoUndef"   => :noundef,
    "ReadOnly"  => :readonly,
    "WriteOnly" => :writeonly,
)

# tblgen VT names that pass through as scalar/vector Symbols; must stay in
# sync with NVVM.llvmtype
const VTS = Set(["i1", "i8", "i16", "i32", "i64", "i128",
                 "f16", "bf16", "f32", "f64", "Metadata",
                 "v2f16", "v2bf16", "v2i32", "v4i8", "v4i32", "v4f32",
                 "v8i32", "v16i32", "v32i32", "v64i32", "v128i32"])

const ANYS = Dict("pAny" => "anyptr", "iAny" => "anyint", "fAny" => "anyfloat")

function token(t)
    if t isa String
        t == "MetadataVT" && (t = "Metadata")
        t in VTS || error("unknown VT \"$t\" — extend VTS and NVVM.llvmtype")
        return ":" * t
    elseif haskey(t, "ptr")
        return "ptr($(t["ptr"]))"
    elseif haskey(t, "match")
        return "slot($(t["match"]))"
    elseif haskey(t, "any")
        return ANYS[t["any"]]
    end
    error("unrecognized type token $t")
end

tuptext(parts) = "(" * join(parts, ", ") * (length(parts) == 1 ? ",)" : ")")

# JSON positions are 0-based params / "ret"; the registry is 1-based with 0
# for the return value.
pos(a) = a == "ret" ? 0 : Int(a) + 1

derived_name(rec) = "llvm." * replace(chopprefix(rec, "int_"), "_" => ".")

lines = String[]
names = Set{String}()
for r in records
    name = isempty(r["llvm_name"]) ? derived_name(r["record"]) : r["llvm_name"]
    name in names && error("duplicate intrinsic name $name")
    push!(names, name)

    props = Symbol[]
    for p in r["props"]
        haskey(PROPS, p) || error("unknown property \"$p\" on $name — extend PROPS")
        push!(props, PROPS[p])
    end

    immargs, ranges, argnames, argattrs = String[], String[], String[], String[]
    for ap in r["arg_props"]
        kind = ap["kind"]
        if kind == "ImmArg"
            ap["arg"] == "ret" && error("ImmArg on return of $name")
            push!(immargs, string(pos(ap["arg"])))
        elseif kind == "Range"
            push!(ranges, "($(pos(ap["arg"])), $(ap["lower"]), $(ap["upper"]))")
        elseif kind == "ArgName"
            push!(argnames, "($(pos(ap["arg"])), :$(ap["name"]))")
        elseif haskey(ARGATTRS, kind)
            push!(argattrs, "($(pos(ap["arg"])), :$(ARGATTRS[kind]))")
        else
            error("unknown argument property \"$kind\" on $name — extend ARGATTRS")
        end
    end

    kw = String[]
    isempty(immargs)  || push!(kw, "immargs=" * tuptext(immargs))
    isempty(ranges)   || push!(kw, "ranges=" * tuptext(ranges))
    isempty(argnames) || push!(kw, "argnames=" * tuptext(argnames))
    isempty(argattrs) || push!(kw, "argattrs=" * tuptext(argattrs))

    push!(lines, "intr(\"$name\", " *
                 tuptext(token.(r["ret"])) * ", " *
                 tuptext(token.(r["params"])) * ", " *
                 tuptext(":" .* string.(props)) *
                 (isempty(kw) ? "" : "; " * join(kw, ", ")) * ")")
end
sort!(lines)

open(SRC, "w") do io
    print(io, """
        # Machine-generated by gen/generate_registry.jl from
        # gen/nvvm_intrinsics_$version.json — do not edit. To regenerate:
        #   gen/extract_intrinsics.sh llvmorg-$version
        #   julia --project=gen gen/generate_registry.jl
        #
        # $(length(lines)) intrinsics from IntrinsicsNVVM.td at llvmorg-$version,
        # name-conformant with the intrinsic table in that tag's llc binary.

        const BACKEND_LLVM_VERSION = v"$version"

        """)
    for l in lines
        println(io, l)
    end
end

println("wrote $SRC ($(length(lines)) intrinsics, backend $version)")
