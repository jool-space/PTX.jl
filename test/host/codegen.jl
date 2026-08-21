using PTX: ptx_to_julia, ir_to_julia
using PTX.Parser: parse as parse_ptx
using PTX.IR
using PTX.Codegen: julia_var, julia_label, sreg_val_expr,
                   render_operand, CodeGenState, param_type_string

const CORPUS_DIR = joinpath(@__DIR__, "..", "corpus")
const CORPUS_FILES = sort(filter(p -> endswith(p, ".ptx"),
                                 readdir(CORPUS_DIR; join = true)))

const EXTERNAL_DIR = joinpath(CORPUS_DIR, "external")
function _gather_external_ptx(dir)
    files = String[]
    isdir(dir) || return files
    for entry in readdir(dir; join = true)
        if isdir(entry)
            append!(files, _gather_external_ptx(entry))
        elseif endswith(entry, ".ptx")
            push!(files, entry)
        end
    end
    sort(files)
end
const EXTERNAL_FILES = _gather_external_ptx(EXTERNAL_DIR)

# Compiler-emitted CLC corpus files that exercise mov.b128 carrier glue and
# the four exact query-cancel result shapes.
const EXTERNAL_B128_RELPATHS = Set((
    "llvm/clusterlaunchcontrol__nvvm_clusterlaunchcontrol_query_cancel_get_first_ctaid_x.ptx",
    "llvm/clusterlaunchcontrol__nvvm_clusterlaunchcontrol_query_cancel_get_first_ctaid_y.ptx",
    "llvm/clusterlaunchcontrol__nvvm_clusterlaunchcontrol_query_cancel_get_first_ctaid_z.ptx",
    "llvm/clusterlaunchcontrol__nvvm_clusterlaunchcontrol_query_cancel_is_canceled.ptx",
))
const EXTERNAL_B128_FILES = Set(
    normpath(joinpath(EXTERNAL_DIR, split(rel, '/')...))
    for rel in EXTERNAL_B128_RELPATHS)

# This is an acceptance boundary, not a coverage boast.  These compiler
# fixtures are the complete external subset whose declarations, operands, and
# instruction roles the transpiler currently preserves.  Every other fixture
# must fail before emission; adding a file cannot silently widen the subset.
const EXTERNAL_TRANSPILABLE_RELPATHS = Set((
    "llvm/cluster-dim__kernel_func_clusterxyz.ptx",
    "llvm/intrinsics-sm90__test_clusterid_x.ptx",
    "llvm/intrinsics-sm90__test_clusterid_y.ptx",
    "llvm/intrinsics-sm90__test_clusterid_z.ptx",
    "llvm/intrinsics-sm90__test_nclusterid_x.ptx",
    "llvm/intrinsics-sm90__test_nclusterid_y.ptx",
    "llvm/intrinsics-sm90__test_nclusterid_z.ptx",
    "llvm/mbarrier__barrierarrive.ptx",
    "llvm/mbarrier__barrierarrivedrop.ptx",
    "llvm/mbarrier__barrierarrivedropnoComplete.ptx",
    "llvm/mbarrier__barrierarrivedropnoCompleteshared.ptx",
    "llvm/mbarrier__barrierarrivedropshared.ptx",
    "llvm/mbarrier__barrierarrivenoComplete.ptx",
    "llvm/mbarrier__barrierarrivenoCompleteshared.ptx",
    "llvm/mbarrier__barrierarriveshared.ptx",
    "llvm/mbarrier__barrierinit.ptx",
    "llvm/mbarrier__barrierinitshared.ptx",
    "llvm/mbarrier__barrierinval.ptx",
    "llvm/mbarrier__barrierinvalshared.ptx",
    "llvm/mbarrier__barrierpendingcount.ptx",
    "llvm/mbarrier__barriertestwait.ptx",
    "llvm/mbarrier__barriertestwaitshared.ptx",
    "llvm/setmaxnreg-sm100a__test_set_maxn_reg_sm100a.ptx",
    "llvm/setmaxnreg__test_set_maxn_reg.ptx",
))
const EXTERNAL_TRANSPILABLE_FILES = Set(
    normpath(joinpath(EXTERNAL_DIR, split(rel, '/')...))
    for rel in EXTERNAL_TRANSPILABLE_RELPATHS)

# --- name-mangling unit tests ----------------------------------------------

@testset "julia_var" begin
    @test julia_var("%r5") == "r5"
    @test julia_var("%rd0") == "rd0"
    @test julia_var("%p1") == "p1"
    @test julia_var("param0") == "param0"
    @test julia_var("%tid.x") == "tid_x"             # falls back when not in SPECIAL_REGS dispatch
    @test julia_var("global") == "global_"           # reserved

    # CUDA.jl mangling: `julia_<sanitized>_<hash>` for function names,
    # appending `_param_<N>` for params. Demangle aggressively.
    @test julia_var("julia__rms_norm_v4_kernel__8414") == "rms_norm_v4_kernel"
    @test julia_var("julia__rms_norm_v4_kernel__8414_param_0") == "param_0"
    @test julia_var("julia__rms_norm_v4_kernel__8414_param_2") == "param_2"
    # Single-underscore variant (when original name has no leading `_`).
    @test julia_var("julia_foo_123") == "foo"
    @test julia_var("julia_foo_123_param_5") == "param_5"
    # Already-demangled names pass through.
    @test julia_var("vector_add") == "vector_add"
end

@testset "julia_label" begin
    @test julia_label("DONE") == "DONE"
    @test julia_label("\$L__BB13_2") == "L__BB13_2"
    @test julia_label("\$L.bb.4") == "L_bb_4"
    @test julia_label("123_target") == "L_123_target"
end

@testset "param_type_string" begin
    # .param .u64 param0
    @test param_type_string(IR.Param(state_space = IR.StateSpace.PARAM,
                                     type = IR.ScalarType.U64,
                                     name = "param0")) == "u64"
    # .param .u64 .ptr .global param1
    @test param_type_string(IR.Param(state_space = IR.StateSpace.PARAM,
                                     type = IR.ScalarType.U64,
                                     name = "param1",
                                     ptr_state_space = IR.StateSpace.GLOBAL)) ==
        "u64.ptr.global"
    # .param .u64 .ptr .global .align 8 param2
    @test param_type_string(IR.Param(state_space = IR.StateSpace.PARAM,
                                     type = IR.ScalarType.U64,
                                     name = "param2",
                                     ptr_state_space = IR.StateSpace.GLOBAL,
                                     ptr_alignment = 8)) ==
        "u64.ptr.global.palign8"
    # .param .align 64 .b8 param4[128]
    @test param_type_string(IR.Param(state_space = IR.StateSpace.PARAM,
                                     type = IR.ScalarType.B8,
                                     name = "param4",
                                     alignment = 64,
                                     array_size = 128)) ==
        "b8.align64.array128"
    # .ptr .shared::cta gets the :: → __ swap
    @test param_type_string(IR.Param(state_space = IR.StateSpace.PARAM,
                                     type = IR.ScalarType.U64,
                                     name = "p",
                                     ptr_state_space = IR.StateSpace.SHARED_CTA)) ==
        "u64.ptr.shared__cta"
end

@testset "sreg_val_expr" begin
    @test sreg_val_expr("%tid.x")     == "sreg\"%tid.x\""
    @test sreg_val_expr("%ctaid.y")   == "sreg\"%ctaid.y\""
    @test sreg_val_expr("%laneid")    == "sreg\"%laneid\""
    # Underscore-bearing names — `sreg"..."` preserves them verbatim.
    @test sreg_val_expr("%cluster_ctarank") == "sreg\"%cluster_ctarank\""
    @test sreg_val_expr("%lanemask_eq")     == "sreg\"%lanemask_eq\""
    @test sreg_val_expr("%cluster_ctaid.x") == "sreg\"%cluster_ctaid.x\""
end

@testset "PTX 9.4 scalar special-register lowering" begin
    # Codegen shares the reviewed scalar subset, while canonicalization keeps
    # the full inventory (including bare v4 roots) for structural comparison.
    @test PTX.Codegen.SPECIAL_REGS === PTX.IR.SCALAR_SPECIAL_REGS
    @test length(PTX.Codegen.SPECIAL_REGS) == 143
    @test !("%warpsize" in PTX.Codegen.SPECIAL_REGS)
    cg = CodeGenState()
    for reg in PTX.IR.SCALAR_SPECIAL_REGS
        @test render_operand(IR.RegisterOperand(reg), cg) == "sreg\"$reg\""
    end
    # PTX 9.3 spells the standard immediate WARP_SZ. Keep older parsed
    # %warpsize input working by lowering both spellings to the same literal.
    @test render_operand(IR.LabelOperand("WARP_SZ"), cg) == "Val(32)"
    @test render_operand(IR.RegisterOperand("%warpsize"), cg) == "Val(32)"

    # A whole v4 special-register value is legal PTX but not yet representable
    # by the scalar parser/lowering path. Reject it instead of emitting an
    # unbound Julia variable or malformed scalar inline asm.
    for root in PTX.IR.V4_SPECIAL_REG_ROOTS
        @test !(root in PTX.Codegen.SPECIAL_REGS)
        @test_throws ArgumentError render_operand(IR.RegisterOperand(root), cg)
    end
    @test_throws PTX.Codegen.TranspilerError ptx_to_julia(""".version 8.0
    .target sm_80
    .address_size 64
    .visible .entry vector_sreg_probe()
    {
    .reg .b32 %r0, %r1, %r2, %r3;
    mov.v4.u32 {%r0, %r1, %r2, %r3}, %tid;
    ret;
    }
    """)

    # The inventory intentionally knows more names than the generic transpiler
    # knows carrier types for.  Only the finite thread-index subset is admitted
    # until each additional special register has an explicit role entry.
    for reg in ("%tid.x", "%ntid.y", "%ctaid.z", "%nctaid.x", "%laneid")
        src = """.version 8.1
        .target sm_90
        .address_size 64
        .visible .entry sreg_probe()
        {
        .reg .u32 %r0;
        mov.u32 %r0, $reg;
        ret;
        }
        """
        out = ptx_to_julia(src)
        @test occursin("sreg\"$reg\"", out)
        parsed = Meta.parseall(out)
        @test !any(arg -> arg isa Expr && arg.head == :error, parsed.args)
    end
    for (decl, dst, op, reg) in (
            (".reg .u32 %r0;", "%r0", "mov.u32", "%pm4"),
            (".reg .b32 %r0;", "%r0", "mov.b32", "%reserved_smem_offset_0"),
            (".reg .u64 %rd0;", "%rd0", "mov.u64", "%current_graph_exec"),
            (".reg .pred %p0;", "%p0", "mov.pred", "%is_explicit_cluster"),
        )
        src = """.version 8.1
        .target sm_90
        .address_size 64
        .visible .entry sreg_probe()
        {
        $decl
        $op $dst, $reg;
        ret;
        }
        """
        err = try
            ptx_to_julia(src)
            nothing
        catch e
            e
        end
        @test err isa PTX.Codegen.TranspilerError
        @test err.category == :operand
        @test occursin("has no reviewed", sprint(showerror, err))
    end
    for spelling in ("WARP_SZ", "%warpsize")
        src = """.version 8.1
        .target sm_90
        .address_size 64
        .visible .entry warp_size_probe()
        {
        .reg .u32 %r0;
        mov.u32 %r0, $spelling;
        ret;
        }
        """
        out = ptx_to_julia(src)
        @test occursin("Val(32)", out)
        parsed = Meta.parseall(out)
        @test !any(arg -> arg isa Expr && arg.head == :error, parsed.args)
    end
    # Parenthesized expressions are opaque parser text. Token substitution is
    # still unit-tested above, but the closed generic operand contract rejects
    # the expression instead of copying PTX expression semantics into Julia.
    for spelling in ("WARP_SZ", "%warpsize")
        src = """.version 8.1
        .target sm_90
        .address_size 64
        .visible .entry warp_size_expr_probe()
        {
        .reg .u32 %r0;
        mov.u32 %r0, ($spelling >> 1);
        ret;
        }
        """
        @test_throws PTX.Codegen.TranspilerError ptx_to_julia(src)
    end
end

# --- operand rendering ------------------------------------------------------

@testset "render_operand: 8 kinds" begin
    cg = CodeGenState()
    @test render_operand(IR.RegisterOperand("%r5"), cg) == "r5"
    @test render_operand(IR.RegisterOperand("%tid.x"), cg) == "sreg\"%tid.x\""
    @test render_operand(IR.ImmediateOperand("42"), cg) == "42"
    @test render_operand(IR.ImmediateOperand("0xFF"), cg) == "0xFF"
    # PTX hex floats decode to bit-exact `reinterpret` calls.
    @test render_operand(IR.ImmediateOperand("0d3FF0000000000000"), cg) ==
          "reinterpret(Float64, 0x3ff0000000000000)"
    @test render_operand(IR.ImmediateOperand("0f3F800000"), cg) ==
          "reinterpret(Float32, 0x3f800000)"
    # With type hint, integer immediate gets wrapped.
    @test render_operand(IR.ImmediateOperand("42"), cg; type_hint = :s32) ==
          "Int32(42)"
    @test render_operand(IR.ImmediateOperand("0xFF"), cg; type_hint = :b32) ==
          "UInt32(0xFF)"
    @test render_operand(IR.ImmediateOperand("(WARP_SZ >> 1)"), cg;
                         type_hint = :u32) == "UInt32((32 >> 1))"
    @test render_operand(IR.ImmediateOperand("(%warpsize >> 1)"), cg;
                         type_hint = :u32) == "UInt32((32 >> 1))"
    # Every current/future predefined immediate in the ledger participates in
    # constant-expression substitution; this test uses a synthetic second
    # name so the production table need not grow merely to test the loop.
    @test PTX.Codegen._replace_predefined_immediate_tokens(
        "(WARP_SZ + FUTURE_CONST)",
        Dict("WARP_SZ" => 32, "FUTURE_CONST" => 7)) == "(32 + 7)"
    # Token boundaries matter: this is a distinct user identifier, not the
    # predefined WARP_SZ constant.
    @test render_operand(IR.ImmediateOperand("(WARP_SZ_limit >> 1)"), cg;
                         type_hint = :u32) == "UInt32((WARP_SZ_limit >> 1))"
    # `.b32` with negative literal → reinterpret(UInt32, Int32(-1)).
    @test render_operand(IR.ImmediateOperand("-1"), cg; type_hint = :b32) ==
          "reinterpret(UInt32, Int32(-1))"
    @test render_operand(IR.LabelOperand("DONE"), cg) == "DONE"
    @test render_operand(IR.VectorOperand((IR.RegisterOperand("%r0"),
                                            IR.RegisterOperand("%r1"))), cg) == "(r0, r1)"
    @test render_operand(IR.AddressOperand("%rd0", nothing), cg) == "address(rd0)"
    @test render_operand(IR.AddressOperand("%rd0", "16"), cg) == "address(rd0 + 16)"
    @test render_operand(IR.NegatedOperand(IR.RegisterOperand("%p0")), cg) == "!p0"
    @test render_operand(IR.PipeOperand(IR.RegisterOperand("%p0"),
                                         IR.RegisterOperand("%p1")), cg) == "(p0, p1)"
end

# --- closed corpus acceptance boundary ------------------------------------

const CURATED_TRANSPILABLE_NAMES = Set((
    "branches.ptx", "minimal.ptx", "predicates.ptx", "vector_add.ptx",
    "tileiras_vadd_sm121a.ptx",
))

@testset "ptx_to_julia: curated accept/reject manifest" begin
    for path in CORPUS_FILES
        if basename(path) in CURATED_TRANSPILABLE_NAMES
            expr = Meta.parseall(ptx_to_julia(read(path, String)))
            @test expr isa Expr && expr.head == :toplevel
            @test !any(a -> a isa Expr && a.head == :error, expr.args)
        else
            err = try
                ptx_to_julia(read(path, String))
                nothing
            catch e
                e
            end
            @test err isa PTX.Codegen.TranspilerError
            @test err.category in (:unsupported, :schema, :operand)
            @test occursin("PTX transpiler contract [", sprint(showerror, err))
        end
    end
end

@testset "ptx_to_julia: external closed acceptance boundary" begin
    actual = Set(relpath(path, EXTERNAL_DIR) for path in EXTERNAL_FILES)
    @test EXTERNAL_TRANSPILABLE_RELPATHS ⊆ actual
    for path in EXTERNAL_FILES
        rel = relpath(path, EXTERNAL_DIR)
        src = _external_parser_source(read(path, String))
        if rel in EXTERNAL_TRANSPILABLE_RELPATHS
            expr = Meta.parseall(ptx_to_julia(src))
            @test expr isa Expr && expr.head == :toplevel
            @test !any(a -> a isa Expr && a.head == :error, expr.args)
        else
            err = try
                ptx_to_julia(src)
                nothing
            catch e
                e
            end
            @test err isa PTX.Codegen.TranspilerError
            @test err.category in (:unsupported, :schema, :operand)
            @test occursin("PTX transpiler contract [", sprint(showerror, err))
        end
    end
end


@testset "ptx_to_julia: compiler b128 fixtures reject undeclared ABI slots" begin
    # Derive the corpus inventory independently so a fifth opaque-handle file
    # cannot silently bypass the common carrier and query-result tier.
    corpus_b128 = Set(relpath(path, EXTERNAL_DIR) for path in EXTERNAL_FILES
                      if occursin(r"\bmov\.b128\b", read(path, String)))
    @test corpus_b128 == EXTERNAL_B128_RELPATHS
    @test EXTERNAL_B128_FILES ⊆ Set(normpath.(EXTERNAL_FILES))

    for path in sort!(collect(EXTERNAL_B128_FILES))
        err = try
            ptx_to_julia(_external_parser_source(read(path, String)))
            nothing
        catch e
            e
        end
        @test err isa PTX.Codegen.TranspilerError
        @test err.category == :unsupported
        @test !isempty(err.path)
        @test !isempty(err.detail)
    end
end

@testset "ptx_to_julia: valid declared mov.b128/query carrier" begin
    source = """
    .version 8.7
    .target sm_100a
    .address_size 64
    .visible .entry declared_b128(
        .param .u64 lo,
        .param .u64 hi
    )
    {
        .reg .b64 %rd<2>;
        .reg .b128 %handle;
        .reg .b32 %r;
        ld.param.u64 %rd0, [lo];
        ld.param.u64 %rd1, [hi];
        mov.b128 %handle, {%rd0, %rd1};
        clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %r, %handle;
        ret;
    }
    """
    julia = ptx_to_julia(source)
    @test occursin("handle = ptx\"mov.b128\"((rd0, rd1))", julia)
    @test occursin("ptx\"clusterlaunchcontrol.query_cancel", julia)
    @test Meta.parseall(julia) isa Expr
end

# --- golden files ----------------------------------------------------------
#
# Hand-written expected output for the four canonical kernels. Byte-exact
# comparison. If the transpiler legitimately changes output format,
# update these — they're the spec.

const GOLDEN_MINIMAL = """\
# @ptx_kernel arch=sm_90a version=8.5
#   linking     = "visible"
function minimal()
    return nothing
end
"""

const GOLDEN_VECTOR_ADD = """\
# @ptx_kernel arch=sm_90a version=8.5
#   raw_params  = [("u64", "param0"), ("u64", "param1"), ("u64", "param2"), ("u32", "param3")]
#   linking     = "visible"
function vector_add(param0, param1, param2, param3)
    local r5 = zero(UInt32)
    local r6 = zero(UInt32)
    local r7 = zero(UInt32)
    local rd3 = zero(UInt64)
    local rd4 = zero(UInt64)
    local rd5 = zero(UInt64)
    local rd6 = zero(UInt64)
    local rd7 = zero(UInt64)
    rd0 = param0
    rd1 = param1
    rd2 = param2
    r0 = param3
    r1 = ptx"mov.u32"(sreg"%tid.x")
    r2 = ptx"mov.u32"(sreg"%ctaid.x")
    r3 = ptx"mov.u32"(sreg"%ntid.x")
    r4 = ptx"mad.lo.s32"(r2, r3, r1)
    p0 = ptx"setp.ge.u32"(r4, r0)
    if p0; @goto DONE; end
    rd3 = ptx"cvt.u64.u32"(r4)
    rd4 = ptx"shl.b64"(rd3, UInt32(2))
    rd5 = ptx"add.u64"(rd0, rd4)
    rd6 = ptx"add.u64"(rd1, rd4)
    rd7 = ptx"add.u64"(rd2, rd4)
    r5 = ptx"ld.global.f32"(address(rd5))
    r6 = ptx"ld.global.f32"(address(rd6))
    r7 = ptx"add.f32"(r5, r6)
    ptx"st.global.f32"(address(rd7), r7)
    @label DONE
    return nothing
end
"""

const GOLDEN_PREDICATES = """\
# @ptx_kernel arch=sm_90a version=8.5
#   linking     = "visible"
function pred_test()
    local r1 = zero(UInt32)
    p0 = ptx"setp.eq.s32"(r0, Int32(0))
    if p0; r1 = ptx"mov.b32"(UInt32(1)); end
    if !p0; r1 = ptx"mov.b32"(UInt32(0)); end
    (p1, p2) = ptx"setp.dual.lt.f32"(r2, r3)
    return nothing
end
"""

const GOLDEN_BRANCHES = """\
# @ptx_kernel arch=sm_90a version=8.5
#   linking     = "visible"
function branch_test()
    local r1 = zero(UInt32)
    p0 = ptx"setp.eq.s32"(r0, Int32(0))
    if p0; @goto THEN; end
    r1 = ptx"mov.b32"(UInt32(0))
    @goto ENDIF
    @label THEN
    r1 = ptx"mov.b32"(UInt32(1))
    @label ENDIF
    return nothing
end
"""

@testset "golden: $name" for (name, expected) in [
        ("minimal.ptx",       GOLDEN_MINIMAL),
        ("vector_add.ptx",    GOLDEN_VECTOR_ADD),
        ("predicates.ptx",    GOLDEN_PREDICATES),
        ("branches.ptx",      GOLDEN_BRANCHES),
    ]
    @test ptx_to_julia(read(joinpath(CORPUS_DIR, name), String)) == expected
end

# Synthetic case exercising the `add`-through-alias propagation path: ptxas
# often does `mov.u64 %rd0, smem; add.u64 %rd1, %rd0, off; ld.shared.b32
# [%rd1]` instead of folding the offset into the addressing mode.
@testset "shared-memory alias propagation through add" begin
    src = """
    .version 8.5
    .target sm_90a
    .address_size 64
    .visible .entry add_alias()
    {
        .reg .b32 %r<2>;
        .reg .b64 %rd<3>;
        .shared .b32 buf[16];
        mov.u64 %rd0, buf;
        add.u64 %rd1, %rd0, 8;
        ld.shared.b32 %r0, [%rd1];
        ret;
    }
    """
    expected = """\
    # @ptx_kernel arch=sm_90a version=8.5
    #   linking     = "visible"
    function add_alias()
        buf = CuStaticSharedArray(UInt32, 16)
        r0 = ptx"ld.shared.b32"(address((pointer(buf)) + UInt64(8)))
        return nothing
    end
    """
    @test ptx_to_julia(src) == expected
end

# --- tileiras (cuTile) compiler output ------------------------------------
#
# `tileiras_vadd_sm121a.ptx` is genuine tileiras output (cuTile.jl `code_ptx`
# of a Float32 vector-add tile kernel): `.language 7` provenance, a
# `___NV_TILE_LAUNCH_META_DATA___` module global read only by the CUDA driver,
# NVVM bit-typed param loads with a widened destination, %clusterid block
# indexing, and a trailing `.section .debug_str` block. One fixture exercises
# every accommodation the transpiler makes for this producer.
@testset "ptx_to_julia: tileiras vadd" begin
    src = read(joinpath(CORPUS_DIR, "tileiras_vadd_sm121a.ptx"), String)
    out = ptx_to_julia(src)
    expr = Meta.parseall(out)
    @test expr isa Expr && expr.head == :toplevel
    @test !any(a -> a isa Expr && a.head == :error, expr.args)
    # Launch metadata survives as header comments.
    @test occursin("(\"reqntid\", (128,))", out)
    @test occursin("(\"language\", (7,))", out)
    # `ld.param.b32 %rd1, [vadd_param_1]` zero-extends into the 64-bit
    # register (§9.4.1, Table 28) — the widening must be explicit.
    @test occursin("rd1 = UInt64(vadd_param_1)", out)
    # `ld.param.b64` into a matching-width register stays a plain rebind.
    @test occursin("rd7 = vadd_param_0", out)
    # %clusterid.x is an admitted u32 special-register carrier.
    @test occursin("ptx\"mov.u32\"(sreg\"%clusterid.x\")", out)
    # The unreferenced metadata global and the debug section are omitted.
    @test !occursin("NV_TILE_LAUNCH_META_DATA", out)
    @test !occursin("debug_str", out)
end

# The zero-init hoist recovers a register's carrier from its `.reg`
# declaration via the Julia variable name. Two lookup edges: PTX permits
# declarations without the `%` sigil (the re-prefixed lookup misses, the bare
# fallback hits), and `$`-carrying compiler names mangle irreversibly (both
# lookups miss — the hoist degrades to a bare `local`, the pre-zero-init
# behavior, rather than guessing a type).
@testset "zero-init hoist: declaration lookup edges" begin
    percentless = """.version 8.5
.target sm_90a
.address_size 64
.visible .entry percentless()
{
\t.reg .pred %p<2>;
\t.reg .b32 r<3>;
\tsetp.eq.s32 %p0, r0, 0;
\t@%p0 bra SKIP;
\tmov.b32 r1, 1;
SKIP:
\tret;
}
"""
    @test occursin("local r1 = zero(UInt32)", ptx_to_julia(percentless))

    mangled = replace(percentless,
        "percentless" => "mangled",
        ".reg .b32 r<3>;" => ".reg .b32 %r\$<3>;",
        "setp.eq.s32 %p0, r0, 0;" => "setp.eq.s32 %p0, %r\$0, 0;",
        "mov.b32 r1, 1;" => "mov.b32 %r\$1, 1;")
    out = ptx_to_julia(mangled)
    @test occursin(r"^    local r1$"m, out)
    @test !occursin("local r1 = zero", out)
end
