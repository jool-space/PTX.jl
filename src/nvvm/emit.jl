# Tier-2 emission: registry-checked `Base.llvmcall` synthesis. Everything
# the spikes validated lands here (spikes/ was removed in ccdfb8a; view the
# scripts with `git show ccdfb8a~1:spikes/`): explicit
# attribute groups on declarations and load-bearing convergence attributes
# on call sites (the in-process LLVM may know and canonicalize some legacy
# intrinsics, so a declaration alone is not an optimizer boundary), aggregate
# returns unpacked to tuples
# (spikes/aggregate_return.jl), and canonical mangled names for overloaded
# intrinsics (llc accepts and silently remangles wrong spellings, so
# acceptance testing would never catch a mangling bug).

export @nvvm_str

# --- Julia-side type vocabulary ---------------------------------------------
#
# Per scalar token: LLVM IR spelling and the Julia types accepted at that
# position. The first accepted type is canonical, used for return values
# (integers come back unsigned: registers hold bit patterns; reinterpret is
# free). Bool is special-cased throughout: the token :i1 accepts Bool, whose
# llvmcall ABI is i8, so emission inserts trunc/zext glue at i1 positions.

# Core.BFloat16 arrived in Julia 1.11. On 1.10 bf16-typed intrinsic
# positions are simply unaccepted (clear compile-time error) — nothing is
# lost in practice: no wrapped intrinsic has one (bf16 mma moves packed
# i32 fragments).
const _BF16_TYPES = isdefined(Core, :BFloat16) ?
    (getfield(Core, :BFloat16),) : ()

const SCALARS = Dict{Symbol,Tuple{String,Tuple{Vararg{DataType}}}}(
    :i1   => ("i1",     (Bool,)),
    :i8   => ("i8",     (UInt8, Int8)),
    :i16  => ("i16",    (UInt16, Int16)),
    :i32  => ("i32",    (UInt32, Int32)),
    :i64  => ("i64",    (UInt64, Int64)),
    :i128 => ("i128",   (UInt128, Int128)),
    :f16  => ("half",   (Float16,)),
    :bf16 => ("bfloat", _BF16_TYPES),
    :f32  => ("float",  (Float32,)),
    :f64  => ("double", (Float64,)),
)

_vector(tok::Symbol) = Base.match(r"^v(\d+)(i\d+|f\d+|bf16)$", String(tok))

_vecelem(@nospecialize(T)) =
    T <: Tuple && length(T.parameters) > 0 && T.parameters[1] <: VecElement &&
    isconcretetype(T) ? T.parameters[1].parameters[1] : nothing

"Whether Julia type `T` is accepted at a position with concrete token `tok`."
function accepts(tok::Symbol, @nospecialize(T))::Bool
    haskey(SCALARS, tok) && return T in SCALARS[tok][2]
    m = _vector(tok)
    m === nothing && return false
    n, elem = parse(Int, m.captures[1]), Symbol(m.captures[2])
    E = _vecelem(T)
    return E !== nothing && length(T.parameters) == n && E in SCALARS[elem][2]
end
accepts(tok::PtrTok, @nospecialize(T)) =
    T <: Core.LLVMPtr && isconcretetype(T) && T.parameters[2] == tok.addrspace

"Human description of what a token accepts, for error messages."
function describe(tok::Symbol)
    haskey(SCALARS, tok) && return join(SCALARS[tok][2], "/")
    m = _vector(tok)
    m === nothing && return "?"
    return "NTuple{$(m.captures[1])," *
           "VecElement{$(join(SCALARS[Symbol(m.captures[2])][2], "/"))}}"
end
describe(tok::PtrTok)  = "Core.LLVMPtr{T,$(tok.addrspace)}"
describe(tok::AnyTok)  = tok.kind === :ptr ? "Core.LLVMPtr{T,AS}" :
                         tok.kind === :int ? "an Int/UInt type" : "a float type"
describe(tok::SlotTok) = "the type bound to overload slot $(tok.slot)"

"Canonical Julia type for a concrete token (used for returns)."
function canonical(tok::Symbol)
    haskey(SCALARS, tok) && return SCALARS[tok][2][1]
    m = _vector(tok)
    n, elem = parse(Int, m.captures[1]), Symbol(m.captures[2])
    return NTuple{n,VecElement{SCALARS[elem][2][1]}}
end
# pointer returns carry no element type in the intrinsic signature; UInt8 is
# the documented placeholder — reinterpret at the wrapper layer
canonical(tok::PtrTok) = Core.LLVMPtr{UInt8,tok.addrspace}

"llvmcall ABI spelling of an accepted Julia type (Bool lowers as i8)."
function abityp(@nospecialize(T))::String
    T === Bool && return "i8"
    T <: Core.LLVMPtr && return llvmtype(ptr(T.parameters[2]))
    E = _vecelem(T)
    E === nothing || return "<$(length(T.parameters)) x $(abityp(E))>"
    for (ir, types) in values(SCALARS)
        T in types && return ir
    end
    error("no llvmcall ABI mapping for $T")
end

"Mangling suffix contributed by the Julia type bound to an overload slot."
function mangle(@nospecialize(T))::String
    # Typed-pointer in-process LLVMs (≤ 16, Julia ≤ 1.11) mangle pointer
    # overloads with the pointee (`p3i8`); opaque (≥ 17) with the address
    # space alone (`p3`). The artifact llc remangles either canonically.
    T <: Core.LLVMPtr && return Base.libllvm_version < v"17" ?
        "p$(T.parameters[2])i8" : "p$(T.parameters[2])"
    E = _vecelem(T)
    E === nothing || return "v$(length(T.parameters))" * mangle(E)
    for (tok, (_, types)) in SCALARS
        T in types && return String(tok)
    end
    error("no mangling for $T")
end

slotaccepts(kind::Symbol, @nospecialize(T)) =
    kind === :ptr ? (T <: Core.LLVMPtr && isconcretetype(T)) :
    kind === :int ? T in (UInt8, Int8, UInt16, Int16, UInt32, Int32,
                          UInt64, Int64, UInt128, Int128) :
                    T in (Float16, _BF16_TYPES..., Float32, Float64)

# --- Attribute rendering ----------------------------------------------------
#
# The middle end reasons about attributes, never intrinsic semantics; what
# isn't stated here is permission (the convergence spike demonstrates the
# resulting miscompile). Spellings must parse under the *in-process* LLVM
# (Julia 1.12 = LLVM 18) — the external backend re-derives attributes from
# its own table and never sees ours.

function memory_attr(props)::Union{String,Nothing}
    # `IntrNoMem` describes memory access; `IntrHasSideEffects` independently
    # says the call is observable even without a result. LLVM IR has no
    # declaration attribute spelling for the latter. Claiming `readnone` /
    # `memory(none)` therefore throws away the only conservative signal an
    # in-process LLVM has when it does not recognize a newer NVVM intrinsic,
    # and lets ordinary DCE erase the call. Leave the memory effect unspecified
    # for this exact upstream-property combination. LLVMs that recognize the
    # intrinsic still recover their more precise built-in contract.
    :nomem in props && :sideeffects in props && return nothing

    rw = :readmem in props ? "read" : :writemem in props ? "write" : "readwrite"
    if Base.libllvm_version < v"16"
        # LLVM 15 (Julia 1.10) predates the memory(...) attribute; these
        # are the exact legacy spellings memory(...) replaced in LLVM 16 —
        # semantically identical, natively understood by 15's optimizer.
        suffix = rw == "read" ? " readonly" : rw == "write" ? " writeonly" : ""
        :nomem in props                         && return "readnone"
        :argmemonly in props                    && return "argmemonly" * suffix
        :inaccessiblememonly in props           && return "inaccessiblememonly" * suffix
        :inaccessiblemem_or_argmemonly in props && return "inaccessiblemem_or_argmemonly" * suffix
        :readmem in props                       && return "readonly"
        :writemem in props                      && return "writeonly"
        return nothing
    end
    :nomem in props                         && return "memory(none)"
    :argmemonly in props                    && return "memory(argmem: $rw)"
    :inaccessiblememonly in props           && return "memory(inaccessiblemem: $rw)"
    :inaccessiblemem_or_argmemonly in props && return "memory(argmem: $rw, inaccessiblemem: $rw)"
    :readmem in props                       && return "memory(read)"
    :writemem in props                      && return "memory(write)"
    return nothing
end

# Upstream-props corrections: IntrinsicsNVVM.td at 22.1.7 omits
# IntrConvergent from both warp-collective matrix namespaces:
#
#   - the existing `llvm.nvvm.mma.` namespace (390 pinned records);
#   - all 414 `wmma` load/mma/store records.
#
# PTX requires every lane to execute the same instruction in both families.
# The overlays are deliberately namespace-based and paired with closed-world
# registry tests: a future addition must be audited before a prefix can
# silently broaden. The WMMA test additionally pins its exact shape/operation
# matrix. `tcgen05.mma` is not included because it has single-thread issue
# semantics. Generated memory properties remain unchanged.
const CONVERGENT_OVERLAY_PREFIXES = ("llvm.nvvm.mma.", "llvm.nvvm.wmma.")

is_convergent(i::Intrinsic) = :convergent in i.props ||
    any(p -> startswith(i.name, p), CONVERGENT_OVERLAY_PREFIXES)

# LLVM ≤ 16 (Julia ≤ 1.11) does not derive merge protection from
# `convergent`: SimplifyCFG can hoist identical convergent calls from both
# arms of a divergent branch into one pre-branch site. `nomerge` closes that
# compatibility gap, so this pair must remain one shared policy for both
# declaration and call-site rendering.
const CONVERGENCE_ATTRS = ("convergent", "nomerge")

convergence_attrs(i::Intrinsic) =
    is_convergent(i) ? CONVERGENCE_ATTRS : ()

# These are intentionally the only declaration properties copied to a call
# site. In particular, declaration attributes such as `speculatable` are not
# universally legal or appropriate there. A dedicated group also survives
# LLVM's canonicalization of recognized legacy NVVM declarations.
callsiteattrs(i::Intrinsic)::String =
    join(convergence_attrs(i), " ")

entryattrs(i::Intrinsic)::String =
    is_convergent(i) ? "alwaysinline convergent" : "alwaysinline"

function fnattrs(i::Intrinsic)::String
    attrs = String[]
    append!(attrs, convergence_attrs(i))
    push!(attrs, "nounwind")  # LLVM intrinsics cannot unwind, categorically
    for (p, a) in ((:nocallback, "nocallback"), (:nofree, "nofree"),
                   (:willreturn, "willreturn"), (:speculatable, "speculatable"),
                   (:noreturn, "noreturn"))
        p in i.props && push!(attrs, a)
    end
    mem = memory_attr(i.props)
    mem === nothing || push!(attrs, mem)
    # :sideeffects, :commutative, :nocreateundefpoison have no IR-attribute
    # spelling (the first is conveyed by NOT claiming memory(none))
    # Convergence remains orthogonal to memory effects: pure matrix operations
    # keep `memory(none)` and ordinary same-block CSE remains legal.
    return join(attrs, " ")
end

const PARAM_ATTRS = Dict(:nocapture => "nocapture", :noalias => "noalias",
                         :noundef => "noundef", :readonly => "readonly",
                         :writeonly => "writeonly")

paramattrs(i::Intrinsic, pos::Int)::String =
    join((PARAM_ATTRS[a] for (p, a) in i.argattrs if p == pos), " ")

# Return-position properties use a different LLVM grammar from ordinary
# parameter attributes: they precede the result type on both declarations and
# calls. Keep the accepted registry vocabulary closed so a backend bump cannot
# silently drop a newly introduced return contract.
const RETURN_ATTRS = Dict(:noundef => "noundef")

function _stored_return_attrs(i::Intrinsic)::Vector{String}
    attrs = String[]
    for (pos, attr) in i.argattrs
        pos == 0 || continue
        spelling = get(RETURN_ATTRS, attr, nothing)
        spelling === nothing && error(
            "$(i.name): unsupported stored return attribute :$attr")
        push!(attrs, spelling)
    end
    unique!(attrs)
end

# Julia 1.10--1.12 use LLVM 15/16/18, where a stored return range is legally
# represented as !range metadata on the call result. This spelling remains
# supported by newer LLVMs as well, while the range(...) return attribute was
# introduced only in LLVM 19. Use the one cross-version encoding rather than
# emitting untested conditional declaration grammar.
const MIN_RETURN_RANGE_LLVM = v"15"

function _return_contract(i::Intrinsic, callret::String;
                          llvm_version::VersionNumber=Base.libllvm_version)
    attrs = _stored_return_attrs(i)
    ranges = [(lo, hi) for (pos, lo, hi) in i.ranges if pos == 0]

    if !isempty(attrs) && callret == "void"
        error("$(i.name): stored return attributes are illegal on a void return")
    end
    isempty(ranges) && return (
        attrs = join(attrs, " "), call_suffix = "", metadata = "")

    llvm_version >= MIN_RETURN_RANGE_LLVM || error(
        "$(i.name): stored return ranges require LLVM " *
        "$(MIN_RETURN_RANGE_LLVM.major)+, got $llvm_version")
    length(ranges) == 1 || error(
        "$(i.name): multiple stored return ranges need an explicit union encoding")
    length(i.ret) == 1 && occursin(r"^i[0-9]+$", callret) || error(
        "$(i.name): stored return range requires one scalar integer result, got $callret")

    lo, hi = only(ranges)
    lo != hi || error("$(i.name): stored return range must not be empty")
    return (
        attrs = join(attrs, " "),
        call_suffix = ", !range !0",
        metadata = "!0 = !{ $callret $lo, $callret $hi }",
    )
end

# --- Typed-pointer compatibility ----------------------------------------------
#
# All pointer types are spelled typed with an `i8` pointee (see
# NVVM.llvmtype) — parses natively on the typed-pointer in-process LLVMs
# (Julia ≤ 1.11 device context) and auto-upgrades to opaque on ≥ 1.12.
# The wrinkle: intrinsics KNOWN to LLVM 15/16 have their pointer params
# signature-verified against an exact pointee there, so those positions
# bitcast from the i8 ABI pointer at the call site (the opaque upgrade
# folds the cast away). Unknown intrinsics — the modern families — accept
# any pointee. Keyed (intrinsic name, 1-based param position); anything
# absent uses i8. Dies together with Julia ≤ 1.11 support.
# Currently empty: the only known-to-old-LLVM intrinsics the package called
# were the legacy mbarrier `*.shared` family, demoted to the asm tier when
# mbarrier became single-route. The mechanism stays for the next wrapper
# that reaches back to a 15/16-era intrinsic.
const TYPED_POINTEE = Dict{Tuple{String,Int},String}()

_typed_pointee(name::String, pos::Int) = get(TYPED_POINTEE, (name, pos), "i8")

_ptrspell(pointee::String, as::Int) =
    as == 0 ? "$pointee*" : "$pointee addrspace($as)*"

# --- Synthesis --------------------------------------------------------------

"""
    synthesize(name, argtypes) -> (; ir, rettype, tupletype, runtime)

Build the `Base.llvmcall` IR for calling intrinsic `name` with arguments of
the given Julia types. `runtime` is the subset of (1-based) argument
positions passed at run time — immediate-operand positions take `Val(x)`
and are spliced into the IR as constants instead, validated against the
registry's legal ranges. Overload slots are bound from the concrete
argument types and produce the canonical mangled callsite name. Every
check errors at compile time, naming the intrinsic and position.
"""
function synthesize(name::String, argtypes)
    i = intrinsic(name)
    err(msg) = error("nvvm\"$name\": $msg")

    any(t -> t === :Metadata, (i.ret..., i.params...)) &&
        err("metadata-typed intrinsics are not callable through nvvm\"\"")
    length(argtypes) == length(i.params) ||
        err("expects $(length(i.params)) arguments, got $(length(argtypes))")

    # overload slots: numbered by AnyTok appearance across (ret..., params...)
    slotkinds = Symbol[t.kind for t in (i.ret..., i.params...) if t isa AnyTok]
    bindings = Vector{Any}(nothing, length(slotkinds))
    function bind!(s::Int, @nospecialize(T), pos::Int)
        slotaccepts(slotkinds[s+1], T) ||
            err("argument $pos: expected $(describe(AnyTok(slotkinds[s+1]))) " *
                "for overload slot $s, got $T")
        bindings[s+1] = T
    end

    pslot = count(t -> t isa AnyTok, i.ret)  # ret slots precede param slots
    callargs = String[]   # operands of the intrinsic call
    entry    = String[]   # @entry parameters (runtime positions only)
    glue     = String[]   # pre-call conversions (Bool i8 -> i1)
    runtime  = Int[]

    for (pos, tok) in enumerate(i.params)
        T = argtypes[pos]
        if pos in i.immargs
            label = let names = [n for (p, n) in i.argnames if p == pos]
                isempty(names) ? "argument $pos" : "argument $pos (`$(names[1])`)"
            end
            T <: Val ||
                err("$label is an immediate operand: pass Val(x), got $T")
            v = T.parameters[1]
            v isa Union{Bool,Integer} || err("$label: Val($v) is not an integer")
            # IntrinsicsNVVM.td records only setmaxnreg's inclusive 24:256
            # bounds. PTX additionally requires a multiple of eight, so keep
            # this narrow semantic overlay beside registry range validation.
            # Reject Bool as well: it is an Integer subtype in Julia but not a
            # meaningful register-count literal.
            if name in ("llvm.nvvm.setmaxnreg.dec.sync.aligned.u32",
                        "llvm.nvvm.setmaxnreg.inc.sync.aligned.u32")
                v isa Integer && !(v isa Bool) ||
                    err("$label: Val($v) must contain a non-Bool integer")
                24 <= v <= 256 ||
                    err("$label: $v is outside the legal range [24, 256]")
                v % 8 == 0 ||
                    err("$label: $v must be a multiple of 8")
            end
            for (p, lo, hi) in i.ranges
                p == pos && !(lo <= Int(v) < hi) &&
                    err("$label: $v is outside the legal range [$lo, $(hi-1)]")
            end
            tok isa Symbol && haskey(SCALARS, tok) ||
                err("$label: unsupported immediate token $tok")
            lit = tok === :i1 ? string(Bool(v)) : string(Int(v))
            push!(callargs, "$(SCALARS[tok][1]) $lit")
            continue
        end

        T <: Val && err("argument $pos is not an immediate operand; got $T")
        k = length(entry)                 # 0-based @entry parameter index
        if tok isa AnyTok
            bind!(pslot, T, pos); pslot += 1
        elseif tok isa SlotTok
            if bindings[tok.slot+1] === nothing
                bind!(tok.slot, T, pos)
            elseif abityp(T) != abityp(bindings[tok.slot+1])
                err("argument $pos: must match overload slot $(tok.slot) " *
                    "($(bindings[tok.slot+1])), got $T")
            end
        elseif !accepts(tok, T)
            err("argument $pos: expected $(describe(tok)), got $T")
        end
        push!(runtime, pos)
        push!(entry, "$(abityp(T)) %a$k")
        if tok === :i1
            push!(glue, "  %b$k = trunc i8 %a$k to i1")
            push!(callargs, "i1 %b$k")
        elseif tok isa PtrTok && _typed_pointee(i.name, pos) != "i8"
            # Known-to-old-LLVM pointee bridge (see TYPED_POINTEE above).
            pt = _ptrspell(_typed_pointee(i.name, pos), tok.addrspace)
            push!(glue, "  %p$k = bitcast $(abityp(T)) %a$k to $pt")
            push!(callargs, "$pt %p$k")
        else
            push!(callargs, "$(abityp(T)) %a$k")
        end
    end

    for s in findall(isnothing, bindings)
        err("overload slot $(s-1) appears only in the return type and " *
            "cannot be inferred from the arguments (explicit slot binding " *
            "is not supported yet)")
    end

    # returns: Julia-side types and intrinsic-side IR types per component
    jts = Type[]
    irts = String[]
    rslot = 0
    for tok in i.ret
        if tok isa AnyTok
            push!(jts, bindings[rslot+1]); push!(irts, abityp(bindings[rslot+1]))
            rslot += 1
        elseif tok isa SlotTok
            push!(jts, bindings[tok.slot+1]); push!(irts, abityp(bindings[tok.slot+1]))
        else
            push!(jts, canonical(tok))
            push!(irts, llvmtype(tok))
        end
    end

    mangled = name * join("." .* mangle.(bindings))

    # declaration: intrinsic-side types for ALL positions (immargs included;
    # immarg positions are always concrete scalar tokens, checked above)
    decl = String[]
    for (pos, tok) in enumerate(i.params)
        t = tok isa AnyTok ? abityp(argtypes[pos]) :
            tok isa SlotTok ? abityp(bindings[tok.slot+1]) :
            tok isa PtrTok  ? _ptrspell(_typed_pointee(i.name, pos), tok.addrspace) :
            llvmtype(tok)
        pa = paramattrs(i, pos)
        push!(decl, isempty(pa) ? t : "$t $pa")
    end
    callret = isempty(irts) ? "void" :
              length(irts) == 1 ? irts[1] : "{ " * join(irts, ", ") * " }"

    retcontract = _return_contract(i, callret)
    retprefix = isempty(retcontract.attrs) ? "" : retcontract.attrs * " "

    # Attribute-group layout is fixed within synthesized modules:
    #   #0 intrinsic declaration; #1 always-inlined @entry; #2 call site.
    # Group #2 exists only for convergent calls.
    callattrs = callsiteattrs(i)
    callattrref = isempty(callattrs) ? "" : " #2"
    intrinsic_call = "call $retprefix$callret @\"$mangled\"(" *
                     join(callargs, ", ") * ")" * callattrref *
                     retcontract.call_suffix

    body = copy(glue)
    if isempty(jts)
        rettype = Nothing
        entryret = "void"
        push!(body, "  " * intrinsic_call)
        push!(body, "  ret void")
    elseif length(jts) == 1
        rettype = jts[1]
        push!(body, "  %r = " * intrinsic_call)
        if i.ret[1] === :i1
            rettype = Bool
            entryret = "i8"
            push!(body, "  %rz = zext i1 %r to i8")
            push!(body, "  ret i8 %rz")
        else
            entryret = callret
            push!(body, "  ret $callret %r")
        end
    else
        rettype = Tuple{jts...}
        comps = abityp.(jts)                       # Bool components are i8
        entryret = allequal(comps) ? "[$(length(comps)) x $(comps[1])]" :
                                     "{ " * join(comps, ", ") * " }"
        push!(body, "  %r = " * intrinsic_call)
        prev = "undef"
        for (k, tok) in enumerate(i.ret)
            push!(body, "  %e$k = extractvalue $callret %r, $(k-1)")
            v = "%e$k"
            if tok === :i1
                push!(body, "  %x$k = zext i1 %e$k to i8")
                v = "%x$k"
            end
            push!(body, "  %t$k = insertvalue $entryret $prev, $(comps[k]) $v, $(k-1)")
            prev = "%t$k"
        end
        push!(body, "  ret $entryret $prev")
    end

    callattrdef = isempty(callattrs) ? "" :
                  "attributes #2 = { $callattrs }"
    ir = """
        declare $retprefix$callret @"$mangled"($(join(decl, ", "))) #0
        define $entryret @entry($(join(entry, ", "))) #1 {
        $(join(body, "\n"))
        }
        attributes #0 = { $(fnattrs(i)) }
        attributes #1 = { $(entryattrs(i)) }
        $callattrdef
        $(retcontract.metadata)
        """

    return (; ir, rettype, tupletype=Tuple{argtypes[runtime]...}, runtime)
end

# --- The callable and the macro ---------------------------------------------

struct IntrinsicCall{name} end

Base.show(io::IO, ::IntrinsicCall{name}) where {name} =
    print(io, "nvvm\"", name, "\"")

@generated function (::IntrinsicCall{name})(args...) where {name}
    s = synthesize(String(name), args)
    call = Expr(:call, GlobalRef(Base, :llvmcall), (s.ir, "entry"),
                s.rettype, s.tupletype,
                (:(args[$i]) for i in s.runtime)...)
    return quote
        $(Expr(:meta, :inline))
        $call
    end
end

"""
    nvvm"llvm.nvvm.name"           # or nvvm"name" — the prefix is implied

A callable for a backend intrinsic, validated against the registry at macro
expansion. Calling it emits attribute-correct `Base.llvmcall` IR:

- Pointers are `Core.LLVMPtr{T,AS}`; the address space participates in
  overload resolution (mangled names are synthesized canonically).
- Immediate operands (`immargs` in the registry) are passed as `Val(x)`
  and range-checked at compile time.
- Vector operands are `NTuple{N,VecElement{T}}`.
- Integer returns are unsigned (`UInt32`, ...); multi-result intrinsics
  return tuples; `i1` maps to `Bool`.

This is tier-2 plumbing: names and signatures are stable
only within a backend JLL major, surfaced as-is from the LLVM table. The
PTX-vocabulary `ptx"..."` notation is the stable surface above this.
"""
macro nvvm_str(name::String)
    full = startswith(name, "llvm.nvvm.") ? name : "llvm.nvvm." * name
    isintrinsic(full) || error(_miss_message(full))
    return IntrinsicCall{Symbol(full)}()
end
