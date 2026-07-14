using PTX: ir_to_julia, ptx_to_julia
using PTX.IR

struct _UnreviewedTranspilerStatement <: IR.Statement end

function _contract_error(thunk; category = nothing)
    err = try
        thunk()
        nothing
    catch caught
        caught
    end
    @test err isa PTX.Codegen.TranspilerError
    err isa PTX.Codegen.TranspilerError || return nothing
    category === nothing || @test err.category == category
    @test !isempty(err.path)
    @test !isempty(err.detail)
    err
end

const _CONTRACT_RET = IR.Instruction("ret", (), ())

_contract_function(body::Tuple{Vararg{IR.Statement}} = (_CONTRACT_RET,); kwargs...) =
    IR.Function(; is_entry = true, name = "contract_probe", body, kwargs...)

function _contract_module(directives::Tuple{Vararg{IR.Statement}};
                          leading::Tuple{Vararg{IR.Statement}} = ())
    IR.Module(version = IR.Version(8, 5),
              target = IR.Target(("sm_90",)),
              address_size = IR.AddressSize(64),
              leading = leading, directives = directives)
end

_contract_module(func::IR.Function; kwargs...) =
    _contract_module((func,); kwargs...)

@testset "transpiler contract: module visitor is closed" begin
    func = _contract_function()
    address32 = IR.Module(version = IR.Version(8, 5),
                          target = IR.Target(("sm_90",)),
                          address_size = IR.AddressSize(32),
                          directives = (func,))
    rejected = (
        address32,
        _contract_module(func; leading = (IR.RawLine("#define X 1"),)),
        _contract_module((IR.VarDecl(state_space = IR.StateSpace.GLOBAL,
                                     type = IR.ScalarType.B32, name = "g"), func)),
        _contract_module((IR.PragmaDirective(value = "nounroll"), func)),
        _contract_module((IR.TargetDirective(target = IR.Target(("sm_90a",))), func)),
        _contract_module((_UnreviewedTranspilerStatement(), func)),
    )
    for mod in rejected
        err = _contract_error(() -> ir_to_julia(mod); category = :unsupported)
        @test startswith(err.path, "module.")
        mod === address32 && @test err.path == "module.address_size"
    end

    # Comments and blank lines are the only nonsemantic module statements.
    accepted = _contract_module((IR.Comment("kept"), IR.BlankLine(), func);
                                leading = (IR.Comment("license"), IR.BlankLine()))
    @test occursin("function contract_probe()", ir_to_julia(accepted))
end

@testset "transpiler contract: function and parameter ABI" begin
    retparam = IR.Param(state_space = IR.StateSpace.REG,
                        type = IR.ScalarType.B32, name = "rv")
    ptrparam = IR.Param(state_space = IR.StateSpace.PARAM,
                        type = IR.ScalarType.U64, name = "ptr",
                        ptr_state_space = IR.StateSpace.GLOBAL)
    aligned = IR.Param(state_space = IR.StateSpace.PARAM,
                       type = IR.ScalarType.B32, name = "x", alignment = 4)
    bf16 = IR.Param(state_space = IR.StateSpace.PARAM,
                    type = IR.ScalarType.BF16, name = "bf16")
    cases = (
        _contract_function(; return_params = (retparam,)),
        _contract_function(; linking = IR.LinkingDirective.WEAK),
        _contract_function(; directives = (IR.FunctionDirective(".maxnreg", (64,)),)),
        _contract_function(; params = (ptrparam,)),
        _contract_function(; params = (aligned,)),
        _contract_function(; params = (bf16,)),
        IR.Function(is_entry = false, name = "decl_only"),
    )
    for func in cases
        _contract_error(() -> ir_to_julia(_contract_module(func));
                        category = :unsupported)
    end

    # Function names and parameter names must remain unique after Julia name
    # mangling, not merely in their original PTX spelling.
    f1 = IR.Function(is_entry = true, name = "global", body = (_CONTRACT_RET,))
    f2 = IR.Function(is_entry = true, name = "global_", body = (_CONTRACT_RET,))
    _contract_error(() -> ir_to_julia(_contract_module((f1, f2)));
                    category = :unsupported)
    params = (IR.Param(state_space = IR.StateSpace.PARAM,
                       type = IR.ScalarType.U32, name = "global"),
              IR.Param(state_space = IR.StateSpace.PARAM,
                       type = IR.ScalarType.U32, name = "global_"))
    _contract_error(() -> ir_to_julia(_contract_module(
        _contract_function(; params))); category = :unsupported)
end

@testset "transpiler contract: body node and declaration policy" begin
    body_nodes = (
        IR.RawLine("opaque;"),
        IR.PragmaDirective(value = "nounroll"),
        IR.IntrinsicScope(name = "scope", args_repr = "x", body = ()),
        _UnreviewedTranspilerStatement(),
    )
    for node in body_nodes
        err = _contract_error(() -> ir_to_julia(_contract_module(
            _contract_function((node, _CONTRACT_RET)))); category = :unsupported)
        @test occursin(".body[1]", err.path)
    end

    declarations = (
        IR.RegDecl(type = IR.ScalarType.B32, name = "%rv",
                   vector_shape = IR.VectorShape.V2),
        IR.RegDecl(type = IR.ScalarType.BF16, name = "%bf16"),
        IR.RegDecl(type = IR.ScalarType.BF16X2, name = "%bf16x2"),
        IR.RegDecl(type = IR.ScalarType.TF32, name = "%tf32"),
        IR.VarDecl(state_space = IR.StateSpace.SHARED,
                   type = IR.ScalarType.B32, name = "sv", array_size = 2,
                   vector_shape = IR.VectorShape.V2),
        IR.VarDecl(state_space = IR.StateSpace.GLOBAL,
                   type = IR.ScalarType.B32, name = "global_storage"),
        IR.VarDecl(state_space = IR.StateSpace.LOCAL,
                   type = IR.ScalarType.B32, name = "local_storage"),
        IR.VarDecl(state_space = IR.StateSpace.CONST,
                   type = IR.ScalarType.B32, name = "const_storage"),
        IR.VarDecl(state_space = IR.StateSpace.SHARED_CTA,
                   type = IR.ScalarType.B32, name = "cta_storage"),
        IR.VarDecl(state_space = IR.StateSpace.SHARED_CLUSTER,
                   type = IR.ScalarType.B32, name = "cluster_storage"),
        IR.VarDecl(state_space = IR.StateSpace.SHARED,
                   type = IR.ScalarType.B32, name = "aligned", alignment = 16),
        IR.VarDecl(state_space = IR.StateSpace.SHARED,
                   type = IR.ScalarType.B32, name = "initialized",
                   initializer = ("1",)),
        IR.VarDecl(state_space = IR.StateSpace.SHARED,
                   type = IR.ScalarType.B32, name = "linked",
                   linking = IR.LinkingDirective.EXTERN),
    )
    for declaration in declarations
        _contract_error(() -> ir_to_julia(_contract_module(
            _contract_function((declaration, _CONTRACT_RET))));
            category = :unsupported)
    end
end

@testset "transpiler contract: declaration namespaces and block scope" begin
    collisions = (
        (IR.RegDecl(type = IR.ScalarType.B32, name = "%a.b"),
         IR.RegDecl(type = IR.ScalarType.B32, name = "%a_b")),
        (IR.RegDecl(type = IR.ScalarType.B32, name = "%r", count = 2),
         IR.RegDecl(type = IR.ScalarType.B32, name = "%r0")),
    )
    for declarations in collisions
        _contract_error(() -> ir_to_julia(_contract_module(
            _contract_function((declarations..., _CONTRACT_RET))));
            category = :unsupported)
    end

    param = IR.Param(state_space = IR.StateSpace.PARAM,
                     type = IR.ScalarType.U32, name = "x")
    reg = IR.RegDecl(type = IR.ScalarType.B32, name = "%x")
    _contract_error(() -> ir_to_julia(_contract_module(
        _contract_function((reg, _CONTRACT_RET); params = (param,))));
        category = :unsupported)

    inferred_collision = """
    .version 8.7
    .target sm_100a
    .address_size 64
    .visible .entry inferred_collision(.param .u64 x) {
        .reg .b64 %lo, %hi;
        mov.b128 x, {%lo, %hi};
        ret;
    }
    """
    _contract_error(() -> ptx_to_julia(inferred_collision);
                    category = :unsupported)

    # A block-local shared alias must not leak into a later bare-name register
    # with the same spelling after the PTX lexical scope has closed.
    scoped = """
    .version 8.5
    .target sm_90
    .address_size 64
    .visible .entry scoped_alias() {
        .reg .u64 %rd;
        { .shared .b64 scoped; }
        .reg .u64 scoped;
        mov.u64 scoped, 7;
        mov.u64 %rd, scoped;
        ret;
    }
    """
    scoped_julia = ptx_to_julia(scoped)
    @test occursin("scoped = ptx\"mov.u64\"(UInt64(7))", scoped_julia)
    @test occursin("rd = ptx\"mov.u64\"(scoped)", scoped_julia)

    outer_write = """
    .version 8.5
    .target sm_90
    .address_size 64
    .visible .entry outer_alias() {
        .shared .b32 s;
        .reg .u64 %rd;
        .reg .b32 %r;
        { mov.u64 %rd, s; }
        ld.shared.b32 %r, [%rd];
        ret;
    }
    """
    outer_julia = ptx_to_julia(outer_write)
    @test occursin("ptx\"ld.shared.b32\"(address(pointer(s)))", outer_julia)
    @test !occursin("address(rd)", outer_julia)

    cfg_alias = """
    .version 8.5
    .target sm_90
    .address_size 64
    .visible .entry cfg_alias() {
        .shared .b32 a;
        .shared .b32 b;
        .reg .u64 %rd;
        .reg .b32 %r;
        .reg .pred %p;
        mov.u64 %rd, a;
        @%p bra SKIP;
        mov.u64 %rd, b;
    SKIP:
        ld.shared.b32 %r, [%rd];
        ret;
    }
    """
    err = _contract_error(() -> ptx_to_julia(cfg_alias);
                          category = :unsupported)
    @test occursin("control-flow-aware", err.detail)
end

@testset "transpiler contract: control-flow scope" begin
    undefined = """
    .version 8.5
    .target sm_90
    .address_size 64
    .visible .entry bad_branch() { bra MISSING; ret; }
    """
    err = _contract_error(() -> ptx_to_julia(undefined); category = :unsupported)
    @test occursin("undefined", err.detail)

    cross_scope = _contract_function((
        IR.Block(body = (IR.Label("INNER"),)),
        IR.Instruction("bra", (), (IR.LabelOperand("INNER"),)),
        _CONTRACT_RET,
    ))
    err = _contract_error(() -> ir_to_julia(_contract_module(cross_scope));
                          category = :unsupported)
    @test occursin("crosses", err.detail)

    accepted = """
    .version 8.5
    .target sm_90
    .address_size 64
    .visible .entry good_branch() {
        .reg .pred %p;
        .reg .b32 %r;
        setp.eq.s32 %p, %r, 0;
        @%p bra DONE;
        mov.b32 %r, 1;
    DONE:
        ret;
    }
    """
    julia = ptx_to_julia(accepted)
    @test occursin("@goto DONE", julia)
    @test occursin("@label DONE", julia)
end

@testset "transpiler contract: finite instruction roles" begin
    rejected_sources = (
        "(%r0)", "{%r0, %r0}", "!%r0", "%r0|%r0", "(WARP_SZ >> 1)",
    )
    for source in rejected_sources
        ptx = """
        .version 8.5
        .target sm_90
        .address_size 64
        .visible .entry bad_operand() {
            .reg .b32 %r0, %r1;
            add.u32 %r1, %r0, $source;
            ret;
        }
        """
        _contract_error(() -> ptx_to_julia(ptx); category = :operand)
    end

    for instruction in (
            "unknown.op %r1, %r0;",
            "add.u32 %r1, %r0;",
            "add.u32 %missing, %r0, 1;",
            "add.u32 %r1, %p, 1;",
            "add.cc.u32 %r1, %r0, 1;",
            "call (%r1), helper, (%r0);",
        )
        ptx = """
        .version 8.5
        .target sm_90
        .address_size 64
        .visible .entry bad_form() {
            .reg .b32 %r0, %r1;
            .reg .pred %p;
            $instruction
            ret;
        }
        """
        _contract_error(() -> ptx_to_julia(ptx))
    end

    for instruction in (
            "add.f32 %f2, 1, %f1;",
            "add.f32 %f2, WARP_SZ, %f1;",
            "mov.pred %p, 1.5;",
        )
        ptx = """
        .version 8.5
        .target sm_90
        .address_size 64
        .visible .entry typed_constant() {
            .reg .f32 %f<3>;
            .reg .pred %p;
            $instruction
            ret;
        }
        """
        _contract_error(() -> ptx_to_julia(ptx); category = :operand)
    end
    compatible_predicate = """
    .version 8.5
    .target sm_90
    .address_size 64
    .visible .entry predicate_constant() {
        .reg .pred %p<3>;
        mov.pred %p0, -1;
        mov.pred %p1, 2;
        mov.pred %p2, 0;
        ret;
    }
    """
    predicate_julia = ptx_to_julia(compatible_predicate)
    @test occursin("p0 = ptx\"mov.pred\"(true)", predicate_julia)
    @test occursin("p1 = ptx\"mov.pred\"(true)", predicate_julia)
    @test occursin("p2 = ptx\"mov.pred\"(false)", predicate_julia)

    # `.file`/`.loc` are the one explicit debug-only omission; they do not
    # widen the semantic instruction ledger.
    debug_body = (IR.Instruction(".file", (), (IR.ImmediateOperand("1"),)),
                  IR.Instruction(".loc", (), (IR.ImmediateOperand("1"),)),
                  _CONTRACT_RET)
    @test occursin("return nothing", ir_to_julia(_contract_module(
        _contract_function(debug_body))))
end

@testset "transpiler contract: emitter consumes the reviewed roles" begin
    source = """
    .version 8.5
    .target sm_90
    .address_size 64
    .visible .entry role_probe() {
        .reg .pred %p;
        .reg .b64 %rd<5>;
        .shared .b64 buf[1];
        mov.u64 %rd0, 7;
        mov.u64 %rd1, buf;
        @%p mov.u64 %rd2, buf;
        mov.u64 %rd3, %rd0;
        shl.b64 %rd4, %rd3, 2;
        ret;
    }
    """
    julia = ptx_to_julia(source)
    @test occursin("rd0 = ptx\"mov.u64\"(UInt64(7))", julia)
    @test !occursin("rd1 = ptx\"mov.u64\"", julia)
    @test occursin("if p; rd2 = ptx\"mov.u64\"(pointer(buf)); end", julia)
    @test occursin("rd3 = ptx\"mov.u64\"(rd0)", julia)
    @test occursin("rd4 = ptx\"shl.b64\"(rd3, UInt32(2))", julia)
    @test !occursin("UInt64(2)", julia)
end

@testset "transpiler contract: exact schemas prove declarations" begin
    scalar_prefix = """
    .version 9.3
    .target sm_120f
    .address_size 64
    .visible .entry exact_scalar() {
    """
    scalar_suffix = """
        ret;
    }
    """
    for body in (
            ".reg .b64 %rd; popc.b64 %missing, %rd;",
            ".reg .u64 %dst; .reg .b64 %rd; popc.b64 %dst, %rd;",
            ".reg .u32 %dst; .reg .b32 %r; popc.b64 %dst, %r;",
            ".reg .u32 %dst; .reg .s32 %a; cvt.pack.sat.u8.s32.b32 %dst, %a, %a, %missing;",
            ".reg .u64 %dst; .reg .s32 %a; .reg .b32 %c; cvt.pack.sat.u8.s32.b32 %dst, %a, %a, %c;",
            ".reg .u32 %dst; .reg .s32 %a; cvt.pack.sat.u8.s32.b32 %dst, 1.5, %a, 0;",
            ".reg .s32 %dst, %a; dp4a.s32.s32 %dst, 1.5, %a, %a;",
        )
        _contract_error(() -> ptx_to_julia(scalar_prefix * body * scalar_suffix))
    end

    cvt_prefix = """
    .version 9.3
    .target sm_90
    .address_size 64
    .visible .entry exact_cvt() {
    """
    cvt_suffix = """
        ret;
    }
    """
    for body in (
            ".reg .u32 %r; cvt.rn.f32.u32 %missing, %r;",
            ".reg .u32 %dst, %r; cvt.rn.f32.u32 %dst, %r;",
            ".reg .f32 %dst; .reg .b64 %rd; cvt.rn.f32.u32 %dst, %rd;",
            ".reg .b32 %dst, %bits; cvt.rs.satfinite.e4m3x4.f32 %dst, {%missing, %missing, %missing, %missing}, %bits;",
        )
        _contract_error(() -> ptx_to_julia(cvt_prefix * body * cvt_suffix))
    end

    alternate = cvt_prefix * """
        .reg .b8 %fp4x2;
        .reg .b16 %fp6x2, %fp4x4;
        .reg .b32 %fp6x4, %bits;
        .reg .f32 %f<4>;
        cvt.rn.satfinite.e2m1x2.f32 %fp4x2, %f0, %f1;
        cvt.rn.satfinite.e2m3x2.f32 %fp6x2, %f0, %f1;
        cvt.rs.satfinite.e2m1x4.f32 %fp4x4, {%f0, %f1, %f2, %f3}, %bits;
        cvt.rs.satfinite.e2m3x4.f32 %fp6x4, {%f0, %f1, %f2, %f3}, %bits;
    """ * cvt_suffix
    alternate_julia = ptx_to_julia(alternate)
    @test occursin("e2m1x2.f32", alternate_julia)
    @test occursin("e2m1x4.f32", alternate_julia)
    for body in (
            ".reg .b16 %dst; .reg .f32 %f<2>; cvt.rn.satfinite.e2m1x2.f32 %dst, %f0, %f1;",
            ".reg .b32 %dst, %bits; .reg .f32 %f<4>; cvt.rs.satfinite.e2m1x4.f32 %dst, {%f0, %f1, %f2, %f3}, %bits;",
        )
        _contract_error(() -> ptx_to_julia(cvt_prefix * body * cvt_suffix))
    end

    clc = """
    .version 8.6
    .target sm_100a
    .address_size 64
    .visible .entry exact_clc() {
        .reg .u32 %r;
        clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%missing], [%r];
        ret;
    }
    """
    err = _contract_error(() -> ptx_to_julia(clc))
    @test occursin("address base", err.detail)
end

@testset "transpiler contract: address state and exact-schema provenance" begin
    for source in (
        """
        .version 8.5
        .target sm_90
        .address_size 64
        .visible .entry bad_param(.param .u32 p) {
            .reg .u64 %rd; ld.param.u64 %rd, [p]; ret;
        }
        """,
        """
        .version 8.5
        .target sm_90
        .address_size 64
        .visible .entry param_is_not_global(.param .f32 p) {
            .reg .b32 %r; ld.global.b32 %r, [p]; ret;
        }
        """,
        """
        .version 8.5
        .target sm_90
        .address_size 64
        .visible .entry shared_is_not_global() {
            .reg .b32 %r; .shared .b32 s; ld.global.b32 %r, [s]; ret;
        }
        """,
        """
        .version 9.0
        .target sm_90
        .address_size 64
        .visible .entry exact_mbarrier(.param .u64 p) {
            mbarrier.init.shared.b64 [p], 1; ret;
        }
        """,
        """
        .version 9.0
        .target sm_90
        .address_size 64
        .visible .entry exact_b128() {
            .reg .b128 %x; ld.global.b128 %x, [unknown]; ret;
        }
        """,
        """
        .version 9.0
        .target sm_90
        .address_size 64
        .visible .entry exact_vector() {
            .reg .b32 %r<2>; ld.global.v2.b32 {%r0, %r1}, [unknown]; ret;
        }
        """,
        """
        .version 9.0
        .target sm_90
        .address_size 64
        .visible .entry vector_policy() {
            .reg .b64 %rd;
            .reg .b32 %r<2>;
            ld.global.L2::cache_hint.v2.b32 {%r0, %r1}, [%rd], 1.5;
            ret;
        }
        """,
    )
        _contract_error(() -> ptx_to_julia(source))
    end

    accepted = """
    .version 8.5
    .target sm_90
    .address_size 64
    .visible .entry direct_shared() {
        .reg .b32 %r; .shared .b32 s; ld.shared.b32 %r, [s]; ret;
    }
    """
    @test occursin("ptx\"ld.shared.b32\"(address(pointer(s)))",
                   ptx_to_julia(accepted))

    generic_clc = """
    .version 8.6
    .target sm_100a
    .address_size 64
    .visible .entry generic_clc() {
        .reg .u64 %rd<2>;
        clusterlaunchcontrol.try_cancel.async.mbarrier::complete_tx::bytes.b128 [%rd0], [%rd1];
        ret;
    }
    """
    @test occursin("address(rd0), address(rd1)", ptx_to_julia(generic_clc))

    shared_clc = """
    .version 8.6
    .target sm_100a
    .address_size 64
    .visible .entry shared_clc() {
        .shared .b32 response;
        .shared .b32 mbar;
        clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [response], [mbar];
        ret;
    }
    """
    @test occursin("address(pointer(response)), address(pointer(mbar))",
                   ptx_to_julia(shared_clc))
end

@testset "transpiler contract: declared bare registers render as variables" begin
    source = """
    .version 9.0
    .target sm_90
    .address_size 64
    .visible .entry bare_regs() {
        .reg .u32 end, a, b;
        .reg .u64 wide;
        .reg .f32 c;
        .reg .pred module;
        popc.b64 end, wide;
        setp.eq.u32 module, a, b;
        cvt.rn.f32.u32 c, a;
        ret;
    }
    """
    julia = ptx_to_julia(source)
    @test occursin("end_ = ptx\"popc.b64\"(wide)", julia)
    @test occursin("module_ = ptx\"setp.eq.u32\"(a, b)", julia)
    @test occursin("c = ptx\"cvt.rn.f32.u32\"(a)", julia)
    @test !occursin("@label", julia)
end
