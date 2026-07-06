# The form registry — the blessing boundary's data (DESIGN.md, "A blessing
# boundary"; CONCERNS.md, "The chain default is permissive by default",
# retired by this table).
#
# Every promise the chain default makes to LLVM about a PTX form lives
# here, in one table, so the whole safety surface is auditable in one
# place and one diff. A promise is only ever a *permission* granted to the
# optimizer:
#
#   pure       — safe to delete if unused, CSE duplicates, reorder freely
#                (false ⇒ asm `sideeffect` + `~{memory}` clobber)
#   convergent — NOT safe to duplicate or merge across divergent branches:
#                warp-/warpgroup-collective (emitted via convergent_asm_ir
#                so the call carries `convergent nomerge`; implies !pure)
#   brackets   — pointer operands render as `[%addr]` (memory-op syntax)
#   returns    — the trailing-dtype rule may reserve a `$0` output register
#                (false for sink forms whose dtype tail names an *operand*)
#
# A FALSE promise in the permissive direction is a miscompile (a deleted
# trap, a reordered barrier, a merged activemask); in the conservative
# direction it only costs optimization. Chains under an opcode absent from
# this table ERROR at compile time — `ptx"..."raw` is the explicit opt-in
# that gets RAW_CONTRACT (maximally conservative) instead. Adding an entry
# here is a review act: check the ISA for memory effects, cross-lane
# semantics, and whether the chain tail names a result or an operand.

struct FormContract
    pure::Bool
    convergent::Bool
    brackets::Bool
    returns::Bool
    function FormContract(; pure::Bool = false, convergent::Bool = false,
                          brackets::Bool = false, returns::Bool = true)
        convergent && pure &&
            error("FormContract: convergent implies !pure (a collective op is observable)")
        new(pure, convergent, brackets, returns)
    end
end

# Maximally conservative: every restriction, no promises. What `ptx"..."raw`
# gets, and the only safe answer for a form nobody has reviewed. `brackets`
# is a text-level guess (most pointer-taking PTX instructions are memory
# ops wanting `[%addr]`); a raw chain that needs a bare pointer operand
# should pass the address as an integer instead — either way the failure
# is a loud ptxas reject, never a miscompile.
const RAW_CONTRACT = FormContract(pure = false, convergent = true,
                                  brackets = true, returns = true)

struct FormFamily
    default::FormContract
    # (mods-prefix => contract); longest matching prefix wins over default.
    overrides::Vector{Pair{Tuple{Vararg{Symbol}}, FormContract}}
end
FormFamily(default::FormContract) =
    FormFamily(default, Pair{Tuple{Vararg{Symbol}}, FormContract}[])

# Shorthands for table legibility.
const _PURE      = FormContract(pure = true)
const _SIDEFX    = FormContract()                          # sideeffect + clobber
const _SINK      = FormContract(returns = false)           # sideeffect, no output
const _MEM       = FormContract(brackets = true)           # memory op, returns
const _MEMSINK   = FormContract(brackets = true, returns = false)
const _COLL      = FormContract(convergent = true)         # collective, returns
const _COLLSINK  = FormContract(convergent = true, returns = false)
const _COLLMEM   = FormContract(convergent = true, brackets = true)

const FORMS = Dict{Symbol, FormFamily}(
    # ── Pure per-lane compute ────────────────────────────────────────────
    # Value ops with no memory access and no cross-lane semantics; the
    # trailing-dtype rule names their result correctly (cvt's dst-at-end-1
    # special case lives in infer_rettype; setp's Bool in PRED_RESULT).
    # Deliberately curated, not the whole ISA: an op joins this list only
    # after checking purity AND that its chain tail names the result (e.g.
    # `set.CmpOp.dtype.stype` and `testp` do NOT qualify — their tails name
    # the source type — so they stay unregistered until given entries that
    # handle their grammar).
    (op => FormFamily(_PURE) for op in (
        :mov, :add, :sub, :mul, :mad, :mul24, :mad24, :fma, :div, :rem,
        :abs, :neg, :min, :max, :and, :or, :xor, :not, :shl, :shr, :shf,
        :bfe, :bfi, :brev, :popc, :clz, :prmt, :lop3, :sad, :dp4a, :dp2a,
        :ex2, :lg2, :sin, :cos, :sqrt, :rsqrt, :rcp, :tanh, :copysign,
        :selp, :szext, :cvt, :cvta, :setp,
        # `clmad.{lo,hi}.type d, a, b, c;` — carryless multiply-add
        # (PTX 9.3, sm_80+). Pure ALU; tail names the result type.
        :clmad))...,

    # ── Non-collective side effects, no memory operand ───────────────────
    :membar            => FormFamily(_SIDEFX),
    :griddepcontrol    => FormFamily(_SIDEFX),
    :clusterlaunchcontrol => FormFamily(_SIDEFX),
    :exit              => FormFamily(_SIDEFX),
    :ret               => FormFamily(_SIDEFX),
    :trap              => FormFamily(_SIDEFX),
    :brkpt             => FormFamily(_SIDEFX),
    :pmevent           => FormFamily(_SIDEFX),
    :nanosleep         => FormFamily(_SINK),    # `.u32` names the duration operand
    # sm_90 cluster address queries: observable cross-CTA visibility.
    :mapa              => FormFamily(_SIDEFX),
    :getctarank        => FormFamily(_SIDEFX),
    :cluster           => FormFamily(_SIDEFX),

    # ── Memory ops (bracketed pointer operands) ──────────────────────────
    :ld       => FormFamily(_MEM),
    :st       => FormFamily(_MEMSINK),          # dtype tail = value written
    :atom     => FormFamily(_MEM),
    :red      => FormFamily(_MEMSINK),          # dtype tail = value written
    :prefetch => FormFamily(_MEM),
    :mbarrier => FormFamily(_MEM),
    :fence    => FormFamily(_MEM),              # brackets vacuous (no ptr args); kept as-was
    :tensormap => FormFamily(_MEMSINK),
    :discard  => FormFamily(_MEM),
    :applypriority => FormFamily(_MEM),
    :cp       => FormFamily(_MEM, [
        # `.b64` tail is the mbarrier address width, not a return.
        (:async, :mbarrier, :arrive) => _MEMSINK,
        # Bulk copies/reductions never return; PTX 9.3 added `.sem`/`.scope`
        # forms whose terminal `.type` (.b128, .add.u64, ...) is an operand
        # descriptor that DTYPE_RETTYPE would misread as a return slot.
        # Covers cp.async.bulk.{tensor,prefetch} too.
        (:async, :bulk)          => _MEMSINK,
        (:reduce, :async, :bulk) => _MEMSINK,
    ]),
    :multimem => FormFamily(_MEM, [
        (:st,)  => _MEMSINK,                    # dtype tail = value written
        (:red,) => _MEMSINK,
        # `multimem.cp.{async,reduce.async}.bulk` (PTX 9.3) — multicast bulk
        # copy/reduce; same no-return shape as the :cp entries above.
        (:cp,)  => _MEMSINK,
    ]),

    # ── Warp-/warpgroup-collective (convergent — every lane must reach the
    #    same call site; duplication across divergence is the activemask
    #    miscompile class) ──────────────────────────────────────────────
    :vote       => FormFamily(_COLL),
    :match      => FormFamily(_COLL),
    :redux      => FormFamily(_COLL),
    :elect      => FormFamily(_COLL),
    :activemask => FormFamily(_COLL),
    :shfl       => FormFamily(_COLL),
    :bar        => FormFamily(_COLL),           # bar.red returns; bar.sync tail is an id
    :barrier    => FormFamily(_COLL),
    :wgmma      => FormFamily(_COLL),
    :mma        => FormFamily(_COLL),
    :setmaxnreg => FormFamily(FormContract(convergent = true, returns = false)),
    :ldmatrix   => FormFamily(_COLLMEM),
    :stmatrix   => FormFamily(FormContract(convergent = true, brackets = true,
                                           returns = false)),
    :tcgen05    => FormFamily(_COLLMEM, [
        (:alloc,)  => FormContract(convergent = true, brackets = true, returns = false),
        (:commit,) => FormContract(convergent = true, brackets = true, returns = false),
        (:relinquish_alloc_permit,) =>
            FormContract(convergent = true, brackets = true, returns = false),
    ]),
)

"""
    form_contract(op, mods) -> Union{FormContract, Nothing}

The registry's contract for a chain form: the opcode's default, refined by
the longest matching mods-prefix override. `nothing` for an unregistered
opcode — callers decide whether that errors (the `ptx"..."` chain default)
or falls back explicitly (`ptx"..."raw` → RAW_CONTRACT).
"""
function form_contract(op::Symbol, mods::Tuple{Vararg{Symbol}})
    fam = get(FORMS, op, nothing)
    fam === nothing && return nothing
    best, bestlen = fam.default, 0
    for (prefix, c) in fam.overrides
        n = length(prefix)
        n <= length(mods) || continue
        ok = true
        for i in 1:n
            mods[i] === prefix[i] || (ok = false; break)
        end
        ok && n >= bestlen && (best = c; bestlen = n)
    end
    return best
end
