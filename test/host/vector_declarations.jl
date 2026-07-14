using PTX.IR: ScalarType, StateSpace, VectorShape, RegDecl, VarDecl,
              Function, RawLine, Instruction, RegisterOperand,
              VECTOR_DECLARATION_TYPES, VECTOR_DECLARATION_STATE_SPACES,
              vector_declaration_legal, vector_shape_from_ptx, ptx

# Independent transcription of PTX ISA 9.3 §5.4.2.  Do not derive this from
# storage widths: alternate floating-point formats fit some widths but §5.2.3
# explicitly says they are not fundamental types.
const EXPECTED_VECTOR_DECLARATION_TYPES = Dict(
    VectorShape.V2 => Set((
        ScalarType.B8, ScalarType.B16, ScalarType.B32, ScalarType.B64,
        ScalarType.U8, ScalarType.U16, ScalarType.U32, ScalarType.U64,
        ScalarType.S8, ScalarType.S16, ScalarType.S32, ScalarType.S64,
        ScalarType.F16, ScalarType.F16X2, ScalarType.F32, ScalarType.F64,
    )),
    VectorShape.V4 => Set((
        ScalarType.B8, ScalarType.B16, ScalarType.B32,
        ScalarType.U8, ScalarType.U16, ScalarType.U32,
        ScalarType.S8, ScalarType.S16, ScalarType.S32,
        ScalarType.F16, ScalarType.F16X2, ScalarType.F32,
    )),
)
const EXPECTED_VECTOR_DECLARATION_SPACES = Set((
    StateSpace.REG, StateSpace.GLOBAL, StateSpace.CONST,
    StateSpace.LOCAL, StateSpace.SHARED,
))

function _vector_decl_source(state_space, shape, type; alignment = nothing,
                             array_size = nothing, tail = "ret;")
    shape_text = ptx(shape)
    type_text = ptx(type)
    align_text = alignment === nothing ? "" : ".align $alignment "
    array_text = array_size === nothing ? "" : "[$array_size]"
    if state_space === StateSpace.REG
        decl = ".reg $shape_text $type_text %v7;"
        return """.version 9.3
        .target sm_75
        .address_size 64
        .entry vector_decl() {
          $decl
          $tail
        }
        """
    elseif state_space === StateSpace.LOCAL
        decl = ".local $align_text$shape_text $type_text v$array_text;"
        return """.version 9.3
        .target sm_75
        .address_size 64
        .entry vector_decl() {
          $decl
          $tail
        }
        """
    else
        decl = "$(ptx(state_space)) $align_text$shape_text $type_text v$array_text;"
        return """.version 9.3
        .target sm_75
        .address_size 64
        $decl
        .entry vector_decl() { $tail }
        """
    end
end

function _only_vector_decl(m, state_space)
    if state_space in (StateSpace.REG, StateSpace.LOCAL)
        f = only(d for d in m.directives if d isa Function)
        return only(d for d in f.body if d isa RegDecl || d isa VarDecl)
    end
    only(d for d in m.directives if d isa VarDecl)
end

@testset "closed PTX 9.3 vector declaration matrix" begin
    @test vector_shape_from_ptx(".v2") === VectorShape.V2
    @test vector_shape_from_ptx("v4") === VectorShape.V4
    @test_throws ArgumentError vector_shape_from_ptx(".v3")
    @test Set(keys(VECTOR_DECLARATION_TYPES)) == Set(keys(EXPECTED_VECTOR_DECLARATION_TYPES))
    for shape in keys(EXPECTED_VECTOR_DECLARATION_TYPES)
        @test Set(VECTOR_DECLARATION_TYPES[shape]) ==
              EXPECTED_VECTOR_DECLARATION_TYPES[shape]
    end
    @test sum(length, values(EXPECTED_VECTOR_DECLARATION_TYPES)) == 28
    @test Set(VECTOR_DECLARATION_STATE_SPACES) ==
          EXPECTED_VECTOR_DECLARATION_SPACES

    for shape in instances(VectorShape.T), type in instances(ScalarType.T),
        state_space in instances(StateSpace.T)
        expected = state_space in EXPECTED_VECTOR_DECLARATION_SPACES &&
                   type in EXPECTED_VECTOR_DECLARATION_TYPES[shape]
        @test vector_declaration_legal(shape, type, state_space) == expected
    end
end

@testset "all legal vector declarations parse and format structurally" begin
    for (shape, types) in EXPECTED_VECTOR_DECLARATION_TYPES,
        type in types, state_space in EXPECTED_VECTOR_DECLARATION_SPACES
        source = _vector_decl_source(state_space, shape, type)
        m = PTX.Parser.parse(source)
        decl = _only_vector_decl(m, state_space)
        @test decl.vector_shape === shape
        @test decl.type === type
        @test decl isa RegDecl ? decl.name == "%v7" : decl.name == "v"

        structural = PTX.IR.format(PTX.IR.normalize(m))
        @test occursin("$(ptx(state_space)) $(ptx(shape)) $(ptx(type))", structural)
        reparsed = PTX.Parser.parse(structural)
        @test isempty(PTX.IR.diff(PTX.IR.normalize(m),
                                  PTX.IR.normalize(reparsed)))
    end
end

@testset "vector declaration modifier order, alignment, and arrays" begin
    for state_space in (StateSpace.GLOBAL, StateSpace.CONST,
                        StateSpace.LOCAL, StateSpace.SHARED)
        source = _vector_decl_source(state_space, VectorShape.V4,
                                     ScalarType.U16;
                                     alignment = 16, array_size = 3)
        decl = _only_vector_decl(PTX.Parser.parse(source), state_space)
        @test decl.alignment == 16
        @test decl.array_size == 3
        structural = PTX.IR.format(PTX.IR.normalize(PTX.Parser.parse(source)))
        @test occursin("$(ptx(state_space)) .align 16 .v4 .u16 v[3];",
                       structural)
    end

    # The ISA's vector-size alignment is a semantic default. Parsing and
    # formatting must not manufacture an explicit `.align` modifier.
    source = _vector_decl_source(StateSpace.GLOBAL, VectorShape.V4,
                                 ScalarType.F32)
    decl = _only_vector_decl(PTX.Parser.parse(source), StateSpace.GLOBAL)
    @test decl.alignment === nothing
    @test !occursin(".align", PTX.IR.format(PTX.IR.normalize(
        PTX.Parser.parse(source))))

    linked = """.version 9.3
    .target sm_75
    .address_size 64
    .visible .global .align 8 .v2 .u32 linked_vectors[2];
    .entry linked_vector_decl() { ret; }
    """
    structural = PTX.IR.format(PTX.IR.normalize(PTX.Parser.parse(linked)))
    @test occursin(
        ".visible .global .align 8 .v2 .u32 linked_vectors[2];",
        structural)
end

@testset "programmatic vector declarations validate at construction" begin
    for shape in instances(VectorShape.T), type in instances(ScalarType.T)
        if type in EXPECTED_VECTOR_DECLARATION_TYPES[shape]
            @test RegDecl(type = type, name = "%v0",
                          vector_shape = shape) isa RegDecl
        else
            @test_throws ArgumentError RegDecl(type = type, name = "%v0",
                                               vector_shape = shape)
        end
    end
    for state_space in instances(StateSpace.T)
        if state_space in EXPECTED_VECTOR_DECLARATION_SPACES
            state_space === StateSpace.REG && continue
            @test VarDecl(state_space = state_space, type = ScalarType.U32,
                          name = "v", vector_shape = VectorShape.V2) isa VarDecl
        else
            @test_throws ArgumentError VarDecl(
                state_space = state_space, type = ScalarType.U32,
                name = "v", vector_shape = VectorShape.V2)
        end
    end
end

@testset "illegal vector declarations recover as RawLine" begin
    invalid = (
        ".reg .v2 .pred %v;",
        ".reg .v2 .b128 %v;",
        ".reg .v4 .u64 %v;",
        ".reg .v3 .u32 %v;",
        ".reg .v2 .bf16 %v;",
        ".reg .v2 .bf16x2 %v;",
        ".reg .v2 .tf32 %v;",
        ".reg .v2 .e4m3 %v;",
        ".reg .v2 .e5m2 %v;",
        ".param .v2 .u32 p;",
        ".shared::cta .v2 .u32 v;",
        ".shared::cluster .v2 .u32 v;",
    )
    for declaration in invalid
        source = """.version 9.3
        .target sm_90
        .address_size 64
        .entry invalid_vector() {
          $declaration
          ret;
        }
        """
        m = PTX.Parser.parse(source)
        f = only(d for d in m.directives if d isa Function)
        @test any(s -> s isa RawLine && occursin(strip(declaration), s.text), f.body)
        @test any(s -> s isa Instruction && s.opcode == "ret", f.body)
    end

    # Formal parameters are not declaration vectors. Pin an inline header so
    # line-local recovery consumes the complete malformed function.
    source = """.version 9.3
    .target sm_90
    .address_size 64
    .entry invalid_param(.param .v2 .u32 p) { ret; }
    .entry survivor() { ret; }
    """
    m = PTX.Parser.parse(source)
    @test any(s -> s isa RawLine && occursin("invalid_param", s.text), m.directives)
    @test any(s -> s isa Function && s.name == "survivor", m.directives)

    module_source = """.version 9.3
    .target sm_90
    .address_size 64
    .global .v4 .u64 invalid_global;
    .global .v2 .u32 valid_global;
    .entry survivor() { ret; }
    """
    module_ir = PTX.Parser.parse(module_source)
    @test any(s -> s isa RawLine && occursin("invalid_global", s.text),
              module_ir.directives)
    @test any(s -> s isa VarDecl && s.name == "valid_global" &&
                   s.vector_shape === VectorShape.V2, module_ir.directives)
end

@testset "vector shape survives normalize, canonicalize, and diff" begin
    source(root, shape = ".v4", type = ".u32") = """.version 9.3
    .target sm_75
    .address_size 64
    .entry vector_canon() {
      .reg $shape $type $root;
      .reg .u32 %r9;
      mov.u32 %r9, $root.x;
      add.u32 $root.y, %r9, $root.r;
      ret;
    }
    """
    a = PTX.Parser.parse(source("%v7"))
    b = PTX.Parser.parse(source("%v91"))
    ca = PTX.IR.canonicalize(a)
    cb = PTX.IR.canonicalize(b)
    @test ca == cb
    cfun = only(d for d in ca.directives if d isa Function)
    cdecl = only(d for d in cfun.body if d isa RegDecl)
    @test cdecl.name == "%v0"
    @test cdecl.vector_shape === VectorShape.V4
    operands = [o.name for s in cfun.body if s isa Instruction
                for o in s.operands if o isa RegisterOperand]
    @test "%v0.x" in operands
    @test "%v0.y" in operands
    @test "%v0.r" in operands

    v2 = PTX.Parser.parse(source("%v7", ".v2"))
    f32 = PTX.Parser.parse(source("%v7", ".v4", ".f32"))
    @test !isempty(PTX.IR.diff(a, v2))
    @test !isempty(PTX.IR.diff(a, f32))
    @test PTX.IR.canonicalize(a) != PTX.IR.canonicalize(v2)
    @test PTX.IR.canonicalize(a) != PTX.IR.canonicalize(f32)

    global_source(shape) = """.version 9.3
    .target sm_75
    .address_size 64
    .global $shape .u32 data;
    .entry vector_global() { ret; }
    """
    global_v2 = PTX.Parser.parse(global_source(".v2"))
    global_v4 = PTX.Parser.parse(global_source(".v4"))
    @test !isempty(PTX.IR.diff(global_v2, global_v4))
    @test PTX.IR.canonicalize(global_v2) != PTX.IR.canonicalize(global_v4)

    # Scalar allocator declarations retain the existing canonical behavior.
    @test count(s -> s isa RegDecl, cfun.body) == 1
end
