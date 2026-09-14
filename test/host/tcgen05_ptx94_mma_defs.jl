# Shared helpers for the PTX ISA 9.4 tcgen05.mma host oracles (ti16,
# collector::b, decompress::lut::b). The operand schema is re-derived here
# from §9.7.18.10 so the production generator is not its own witness:
#
#   [d], a-desc|[a-tmem], b-desc{, [meta]}, idesc{, [scale-A], [scale-B]}
#      {, {mask}}, enable-input-d{, scale-input-d}{, zero-column-mask}

using PTX
using PTX: Operation, lowering

const _T5_94_COLL_A = (nothing, Symbol("collector::a::lastuse"),
                       Symbol("collector::a::fill"), Symbol("collector::a::use"))
const _T5_94_COLL_B = (nothing, Symbol("collector::b::fill"),
                       Symbol("collector::b::use"), Symbol("collector::b::lastuse"))

_t5_94_opt(x) = x === nothing ? () : (x,)

# The expected PTX operand text for one shape.
function _t5_94_schema(; a_tmem::Bool, meta::Bool = false, mx::Bool = false,
                       maskN::Int = 0, zero_col::Bool = false,
                       scale = nothing)
    parts = String["[\$0]", a_tmem ? "[\$1]" : "\$1", "\$2"]
    k = 3
    if meta
        push!(parts, "[\$$k]"); k += 1
    end
    push!(parts, "\$$k"); k += 1
    if mx
        push!(parts, "[\$$k]", "[\$$(k + 1)]"); k += 2
    end
    if maskN > 0
        push!(parts, "{" * join(("\$$(k + i)" for i in 0:maskN - 1), ", ") * "}")
        k += maskN
    end
    push!(parts, "\$$k"); k += 1
    zero_col && (push!(parts, "\$$k"); k += 1)
    scale === nothing || push!(parts, string(scale))
    join(parts, ", ") * ";"
end

# One asm-tier method: exact dispatch, asm tier, no result, no intrinsic,
# the exact head + operand schema, sideeffect + memory clobber, and no
# convergence contract (tcgen05.mma is non-convergent in FORMS).
function _t5_94_check(mods, args, schema)
    op = Operation{:tcgen05, mods}()
    @test which(op, args).module === PTX
    info = lowering(op, args)
    @test info.tier === :asm
    @test info.rettype === Nothing
    @test isempty(info.intrinsics)
    ci, _ = first(Base.code_typed(op, args))
    typed = replace(string(ci), "\\\$" => "\$")
    head = "tcgen05." * join(String.(mods), ".")
    @test occursin(head * " " * schema, typed)
    @test occursin("asm sideeffect", typed)
    @test occursin("~{memory}", typed)
    @test !occursin("convergent", typed)
    nothing
end

function _t5_94_forbidden(mods, args)
    @test lowering(Operation{:tcgen05, mods}(), args).tier === :forbidden
    @test_throws ArgumentError PTX.build_call(:tcgen05, mods, args)
    nothing
end
