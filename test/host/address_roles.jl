using PTX: Address, address, build_call, format_call, Operation, RawOperation
using PTX.Parser: parse as parse_ptx
using PTX.IR: AddressOperand, Instruction, format
using InteractiveUtils: code_llvm

@noinline _address_payload_llvm_probe(x::UInt32) =
    ptx"ld.shared.u32"(address(x))

@noinline function _tcgen05_mx_shared_address_probe(
        d::UInt32, a_desc::UInt64, b_desc::UInt64, idesc::UInt32,
        scale_a::UInt32, scale_b::UInt32, enable::Bool)
    ptx"tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"(
        address(d), a_desc, b_desc, idesc,
        address(scale_a), address(scale_b), enable)
end

@noinline function _tcgen05_mx_tmem_address_probe(
        d::UInt32, a_tmem::UInt32, b_desc::UInt64, idesc::UInt32,
        scale_a::UInt32, scale_b::UInt32, enable::Bool)
    ptx"tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"(
        address(d), address(a_tmem), b_desc, idesc,
        address(scale_a), address(scale_b), enable)
end

# Independent PTX 9.3 §9.7.14.18 grammar oracle.  This deliberately does
# not derive from CLC_TRY_CANCEL_FORMS: changing production modifier order or
# broadening the grammar must force a review of these four spellings.
const EXPECTED_CLC_TRY_CANCEL_FORMS = (
    ((:try_cancel, :async, Symbol("mbarrier::complete_tx::bytes"), :b128),
     false, false),
    ((:try_cancel, :async, Symbol("shared::cta"),
      Symbol("mbarrier::complete_tx::bytes"), :b128),
     true, false),
    ((:try_cancel, :async, Symbol("mbarrier::complete_tx::bytes"),
      Symbol("multicast::cluster::all"), :b128),
     false, true),
    ((:try_cancel, :async, Symbol("shared::cta"),
      Symbol("mbarrier::complete_tx::bytes"),
      Symbol("multicast::cluster::all"), :b128),
     true, true),
)

# Independent closed oracle for every integer-address form that must never
# fall through to the scalar chain. Keys intentionally omit prose so message
# edits do not weaken or churn the semantic boundary.
const EXPECTED_STRUCTURED_ADDRESS_FALLBACK_RULES = Set([
    (:ld, (), :v2), (:ld, (), :v4),
    (:st, (), :v2), (:st, (), :v4),
    (:atom, (), :v2), (:atom, (), :v4), (:atom, (), :v8),
    (:red, (), :v2), (:red, (), :v4), (:red, (), :v8),
    (:multimem, (), :v2), (:multimem, (), :v4),
    (:multimem, (), :v8),
    (:ld, (), :b128), (:st, (), :b128), (:atom, (), :b128),
    (:red, (), :b128), (:multimem, (), :b128),
    (:ldmatrix, (), nothing), (:stmatrix, (), nothing),
    (:cp, (:async, :bulk, :tensor), nothing),
    (:cp, (:async, :bulk, :prefetch, :tensor), nothing),
    (:cp, (:reduce, :async, :bulk, :tensor), nothing),
    (:tcgen05, (), nothing), (:fabric, (), nothing),
])

function _expected_tcgen05_integer_address_forms()
    forms = Tuple{Vararg{Symbol}}[]
    for cg in 1:2
        cta = Symbol("cta_group::", cg)
        push!(forms, (:shift, cta, :down))
        for shape in (Symbol("128x256b"), Symbol("4x256b"),
                      Symbol("128x128b"))
            push!(forms, (:cp, cta, shape))
        end
    end
    for (shape, counts) in (
            (Symbol("16x64b"),  (1, 2, 4, 8, 16, 32, 64, 128)),
            (Symbol("32x32b"),  (1, 2, 4, 8, 16, 32, 64, 128)),
            (Symbol("16x128b"), (1, 2, 4, 8, 16, 32, 64)),
            (Symbol("16x256b"), (1, 2, 4, 8, 16, 32)))
        for count in counts, op in (:ld, :st)
            push!(forms, (op, :sync, :aligned, shape,
                          Symbol("x", count), :b32))
        end
    end
    for cg in 1:2
        cta = Symbol("cta_group::", cg)
        push!(forms, (:alloc, cta, :sync, :aligned,
                      Symbol("shared::cta"), :b32))
        for space in (Symbol("shared::cta"), Symbol("shared::cluster"))
            push!(forms, (:commit, cta, Symbol("mbarrier::arrive::one"),
                          space, :b64))
        end
        push!(forms, (:commit, cta, Symbol("mbarrier::arrive::one"),
                      Symbol("multicast::cluster"),
                      Symbol("shared::cluster"), :b64))
        for kind in (:f16, :tf32, :f8f6f4, :i8), sp in ((), (:sp,))
            for coll in (nothing, Symbol("collector::a::lastuse"),
                         Symbol("collector::a::fill"),
                         Symbol("collector::a::use"))
                push!(forms, (:mma, sp..., cta, Symbol("kind::", kind),
                              (coll === nothing ? () : (coll,))...))
            end
            for coll in (nothing, Symbol("collector::a::lastuse"))
                push!(forms, (:mma, sp..., cta, Symbol("kind::", kind),
                              :ashift,
                              (coll === nothing ? () : (coll,))...))
            end
        end
    end
    variants = (
        (:mxf8f6f4, Symbol("scale_vec::1X"), :block32),
        (:mxf4, Symbol("scale_vec::2X"), :block32),
        (:mxf4nvf4, Symbol("scale_vec::2X"), :block32),
        (:mxf4nvf4, Symbol("scale_vec::4X"), :block16),
    )
    for (kind, scale_vec, block) in variants,
            scale in (scale_vec, block), cg in 1:2
        push!(forms, (:mma, Symbol("cta_group::", cg),
                      Symbol("kind::", kind), :block_scale, scale))
    end
    for kind in (:f16, :tf32, :f8f6f4, :i8), sp in ((), (:sp,))
        colls = Any[nothing]
        for buf in 0:3, op in (:discard, :lastuse, :fill, :use)
            buf == 0 && op === :discard && continue
            push!(colls, Symbol("collector::b$buf::$op"))
        end
        for coll in colls
            push!(forms, (:mma, :ws, sp..., Symbol("cta_group::1"),
                          Symbol("kind::", kind),
                          (coll === nothing ? () : (coll,))...))
        end
    end
    Set(forms)
end

function _expected_tcgen05_integer_address_adapters()
    specs = Set{Tuple{Tuple{Vararg{Symbol}}, Tuple{Vararg{Type}}}}()
    A32 = Address{UInt32}
    for mods in _expected_tcgen05_integer_address_forms()
        if first(mods) === :mma && mods[2] === :ws
            # weight-stationary: two A carriers × {base, +zero-col-mask
            # descriptor}; sp inserts the metadata TMEM address
            sp = mods[3] === :sp
            meta = sp ? (A32,) : ()
            for aT in (UInt64, A32)
                push!(specs, (mods, (A32, aT, UInt64, meta..., UInt32, Bool)))
                push!(specs, (mods,
                    (A32, aT, UInt64, meta..., UInt32, Bool, UInt64)))
            end
            continue
        end
        if first(mods) === :mma && !(:block_scale in mods)
            # dense/sparse mma: two A carriers (SMEM descriptor / TMEM
            # address) × {base, +disable-output-lane mask} operand shapes,
            # plus {+scale, +mask+scale} for f16/tf32; ashift is
            # TMEM-only. The sp forms insert the sparsity-metadata TMEM
            # address between the B descriptor and idesc.
            sp = mods[2] === :sp
            off = sp ? 1 : 0
            cg = mods[2 + off] === Symbol("cta_group::1") ? 1 : 2
            maskT = NTuple{cg == 1 ? 4 : 8, UInt32}
            scale_ok = mods[3 + off] in
                (Symbol("kind::f16"), Symbol("kind::tf32"))
            meta = sp ? (A32,) : ()
            for aT in (:ashift in mods ? (A32,) : (UInt64, A32))
                push!(specs, (mods, (A32, aT, UInt64, meta..., UInt32, Bool)))
                push!(specs,
                      (mods, (A32, aT, UInt64, meta..., UInt32, maskT, Bool)))
                if scale_ok
                    push!(specs, (mods,
                        (A32, aT, UInt64, meta..., UInt32, Bool, Val)))
                    push!(specs, (mods,
                        (A32, aT, UInt64, meta..., UInt32, maskT, Bool, Val)))
                end
            end
            continue
        end
        argtypes = if first(mods) === :shift || first(mods) === :ld
            (A32,)
        elseif first(mods) === :alloc
            (A32, UInt32)
        elseif first(mods) === :cp
            (A32, UInt64)
        elseif first(mods) === :st
            base = Dict(Symbol("16x64b") => 1, Symbol("32x32b") => 1,
                        Symbol("16x128b") => 2,
                        Symbol("16x256b") => 4)[mods[4]]
            count = parse(Int, String(mods[5])[2:end])
            (A32, NTuple{base * count, UInt32})
        elseif first(mods) === :commit
            Symbol("multicast::cluster") in mods ? (A32, Integer) : (A32,)
        elseif first(mods) === :mma && :block_scale in mods
            (A32, UInt64, UInt64, UInt32, A32, A32, Bool)
        else
            error("unclassified tcgen05 adapter form: $mods")
        end
        push!(specs, (mods, argtypes))
        if first(mods) === :mma && :block_scale in mods
            push!(specs, (mods,
                (A32, A32, UInt64, UInt32, A32, A32, Bool)))
        end
    end
    specs
end

@testset "explicit PTX address role carrier" begin
    for T in (Int32, UInt32, Int64, UInt64)
        marked = address(zero(T))
        @test marked isa Address{T}
        @test marked.value === zero(T)
        @test isbitstype(typeof(marked))
        @test address(marked) === marked

        spec = build_call(:ld, (:global, :u32), (typeof(marked),))
        @test spec.asm == "ld.global.u32 \$0, [\$1];"
        @test spec.passthrough_argtypes === (T,)
        @test spec.passthrough_indices === ((1, nothing),)
        @test spec.passthrough_unwrap_address === (true,)
        @test spec.constraints == "=r,$(PTX.constraint_letter(T)),~{memory}"
    end

    # A role marker cannot inject invalid brackets into an ordinary value
    # form. The form contract must explicitly admit bracketed addresses.
    marked = address(UInt32(7))
    @test_throws ArgumentError build_call(:add, (:u32,),
                                          (UInt32, typeof(marked)))
    @test PTX.lowering(ptx"add.u32",
                       (UInt32, typeof(marked))).tier === :forbidden

    # Pointers are already role- and address-space-bearing. `address` is type
    # identity so it cannot hide an exact wrapper behind Address dispatch.
    pG = Core.LLVMPtr{UInt32, PTX.AS.Global}
    @test Core.Compiler.return_type(address, Tuple{pG}) === pG
    @test format_call(ptx"ld.global.u32", Tuple{pG}) ==
          "ld.global.u32 \$0, [\$1];"
    @test format_call(ptx"mov.u64", Tuple{pG}) == "mov.u64 \$0, \$1;"
    @test PTX.lowering(ptx"ld.global.v2.b32", (pG,)).rettype ===
          NTuple{2, UInt32}
    @test PTX.lowering(ptx"ld.global.v2.b32", (Address{pG},)).tier ===
          :forbidden
    @test_throws ArgumentError build_call(:ld, (:global, :u32),
                                          (Address{pG},))

    # Nested role markers cannot be constructed; the public helper is the
    # only idempotent path. Non-address carrier classes fail at construction.
    @test_throws MethodError Address{typeof(marked)}(marked)
    for x in (false, Int8(1), UInt8(1), Int16(1), UInt16(1),
              1.0f0, 1.0, Int128(1), UInt128(1), (UInt32(1),))
        @test_throws ArgumentError address(x)
    end

    # Optimized LLVM proves that the marker itself is not passed to inline
    # asm: the call receives the bare i32 payload and no aggregate carrier.
    llvm = sprint() do io
        code_llvm(io, _address_payload_llvm_probe, Tuple{UInt32};
                  optimize = true, raw = true, debuginfo = :none,
                  dump_module = true)
    end
    @test occursin(
        "asm sideeffect \"ld.shared.u32 \$0, [\$1];\", " *
        "\"=r,r,~{memory}\"(i32",
        llvm)
    @test !occursin("PTX.Address", llvm)
    @test haskey(Base.Docs.meta(PTX), Base.Docs.Binding(PTX, :address))
end

@testset "structured integer-address fallback boundary" begin
    actual = Set((r.op, r.prefix, r.marker)
                 for r in PTX.STRUCTURED_ADDRESS_FALLBACK_RULES)
    @test actual == EXPECTED_STRUCTURED_ADDRESS_FALLBACK_RULES
    @test length(actual) == 25

    A32, A64 = Address{UInt32}, Address{UInt64}
    cases = (
        (ptx"st.global.v4.b32", (A64, NTuple{4, UInt32})),
        (ptx"atom.global.add.v8.u32", (A64, NTuple{8, UInt32})),
        (ptx"red.global.add.v2.u32", (A64, NTuple{2, UInt32})),
        (ptx"multimem.ld_reduce.weak.gpu.global.add.v4.u32", (A64,)),
        (ptx"atom.global.exch.b128", (A64, NTuple{4, UInt32})),
        (ptx"ldmatrix.sync.aligned.m8n8.x2.shared.b16", (A32,)),
        (ptx"stmatrix.sync.aligned.m8n8.x2.shared.b16",
         (A32, NTuple{2, UInt32})),
        (ptx"cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes",
         (A32, A64, Int32, A32)),
        (ptx"fabric.try_get.async.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.b128",
         (A32, UInt32, UInt64, UInt32, A32)),
    )
    for (op, argtypes) in cases
        op_sym, mods = typeof(op).parameters
        raw = RawOperation{op_sym, mods}()
        @test PTX.lowering(op, argtypes).tier === :forbidden
        @test PTX.lowering(raw, argtypes).tier === :forbidden
        @test_throws ArgumentError build_call(op_sym, mods, argtypes)
        @test_throws ArgumentError build_call(op_sym, mods, argtypes; raw = true)
    end

    # Scalar b128 loads have their own audited carrier schema. The address
    # marker is authoritative there just as it is for reviewed vector loads;
    # only the unreviewed aggregate spelling above remains forbidden.
    b128_ld = build_call(:ld, (:global, :b128), (A64,))
    @test b128_ld.rettype === B128
    @test PTX.lowering(ptx"ld.global.b128", (A64,)).tier === :chain_asm

    # An audited vector-result form is authoritative for its own address
    # operand: an integer Address routes through the vector schema (which
    # validates and brackets it) instead of the exact-wrapper-only fallback.
    vector_ld = build_call(:ld, (:global, :v2, :b32), (A64,))
    @test vector_ld.asm == "ld.global.v2.b32 {\$0, \$1}, [\$2];"
    @test vector_ld.rettype === NTuple{2, UInt32}
    @test PTX.lowering(ptx"ld.global.v2.b32", (A64,)).tier === :chain_asm

    # Classic cp.async is scalar-structure-safe with explicit integer address
    # widths: two bracketed addresses, one baked size, and no result.
    classic = build_call(:cp, (:async, :ca, :shared, :global),
                         (A32, A64, Val{16}))
    @test classic.asm ==
          "cp.async.ca.shared.global [\$0], [\$1], 16;"
    @test classic.rettype === Nothing
    @test classic.constraints == "r,l,~{memory}"
end

@testset "closed tcgen05 integer-address adapters" begin
    expected_forms = _expected_tcgen05_integer_address_forms()
    @test Set(PTX.TCGEN05_INTEGER_ADDRESS_FORMS) == expected_forms
    @test length(expected_forms) == 314
    expected = _expected_tcgen05_integer_address_adapters()
    actual = Set((s.mods, s.argtypes)
                 for s in PTX.TCGEN05_INTEGER_ADDRESS_ADAPTERS)
    @test actual == expected
    @test length(actual) == 1098
    for (mods, signature) in actual
        # Immediate specs are the abstract `Val` (dispatch admits any
        # immediate); lowering probes need a concrete instance, as every
        # real call site has.
        marked = Tuple(T === Integer ? UInt16 :
                       T === Val ? Val{8} : T for T in signature)
        payload = Tuple(T <: Address ? T.parameters[1] :
                        T === Integer ? UInt16 :
                        T === Val ? Val{8} : T for T in signature)
        op = Operation{:tcgen05, mods}()
        payload_info = PTX.lowering(op, payload)
        marked_info = PTX.lowering(op, marked)
        @test marked_info.tier === payload_info.tier
        @test marked_info.rettype === payload_info.rettype
        @test marked_info.tier !== :chain_asm
        @test which(op, Tuple{marked...}) !== PTX._CHAIN_METHOD
    end

    # An unreviewed modifier tuple cannot use the adapter or raw fallback.
    miss = ptx"tcgen05.ld.sync.aligned.16x32bx2.x1.b32"
    @test PTX.lowering(miss, (Address{UInt32},)).tier === :forbidden
    @test PTX.lowering(ptx"tcgen05.ld.sync.aligned.16x32bx2.x1.b32"raw,
                       (Address{UInt32},)).tier === :forbidden

    # PTX 9.3 §9.7.17.7.1 spells dealloc's taddr as a bare register. It must
    # retain the existing exact UInt32 method and reject an invented bracketed
    # Address role instead of silently unwrapping it.
    dealloc = ptx"tcgen05.dealloc.cta_group::1.sync.aligned.b32"
    bare_dealloc = PTX.lowering(dealloc, (UInt32, UInt32))
    @test bare_dealloc.tier === :intrinsic
    @test which(dealloc, Tuple{UInt32, UInt32}) !== PTX._CHAIN_METHOD
    @test PTX.lowering(dealloc, (Address{UInt32},)).tier === :forbidden
    @test PTX.lowering(dealloc,
                       (Address{UInt32}, UInt32)).tier === :forbidden
    @test PTX.lowering(dealloc, (Address{UInt32}, UInt64)).tier === :forbidden
    @test_throws ArgumentError dealloc(address(UInt32(0)))
    @test_throws ArgumentError dealloc(address(UInt32(0)), UInt32(1))
    @test_throws ArgumentError dealloc(address(UInt32(0)), UInt64(1))
    @test_throws ArgumentError build_call(:tcgen05,
        (:dealloc, Symbol("cta_group::1"), :sync, :aligned, :b32),
        (Address{UInt32}, UInt32))
    @test_throws ArgumentError build_call(:tcgen05,
        (:dealloc, Symbol("cta_group::1"), :sync, :aligned, :b32),
        (Address{UInt32}, UInt64))

    # The two MX grammar alternatives differ only in operand 2. Pin every
    # bracket role at the generated method boundary, then independently prove
    # the direct calls lower to the exact PTX syntax with no marker in LLVM IR.
    mxmods = (:mma, Symbol("cta_group::1"), Symbol("kind::mxf8f6f4"),
              :block_scale, Symbol("scale_vec::1X"))
    A32 = Address{UInt32}
    shared_sig = (A32, UInt64, UInt64, UInt32, A32, A32, Bool)
    tmem_sig = (A32, A32, UInt64, UInt32, A32, A32, Bool)
    @test (mxmods, shared_sig) in actual
    @test (mxmods, tmem_sig) in actual
    mxop = Operation{:tcgen05, mxmods}()
    for bad_signature in (
            (A32, UInt64, UInt64, UInt32, UInt32, A32, Bool),
            (A32, UInt32, UInt64, UInt32, A32, A32, Bool),
            (A32, A32, Address{UInt64}, UInt32, A32, A32, Bool))
        @test PTX.lowering(mxop, bad_signature).tier === :forbidden
        @test which(mxop, Tuple{bad_signature...}) === PTX._CHAIN_METHOD
    end
    for (probe, argtypes, asm) in (
            (_tcgen05_mx_shared_address_probe,
             Tuple{UInt32, UInt64, UInt64, UInt32,
                   UInt32, UInt32, Bool},
             "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale." *
             "scale_vec::1X [\$0], \$1, \$2, \$3, [\$4], [\$5], \$6;"),
            (_tcgen05_mx_tmem_address_probe,
             Tuple{UInt32, UInt32, UInt64, UInt32,
                   UInt32, UInt32, Bool},
             "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale." *
             "scale_vec::1X [\$0], [\$1], \$2, \$3, [\$4], [\$5], \$6;"))
        llvm = sprint() do io
            code_llvm(io, probe, argtypes; optimize = true, raw = true,
                      debuginfo = :none, dump_module = true)
        end
        @test occursin(asm, llvm)
        @test !occursin("PTX.Address", llvm)
    end
end

@testset "pointer address identity preserves exact wrappers" begin
    G32 = Core.LLVMPtr{UInt32, PTX.AS.Global}
    S16 = Core.LLVMPtr{UInt16, PTX.AS.Shared}
    S64 = Core.LLVMPtr{UInt64, PTX.AS.Shared}
    C8 = Core.LLVMPtr{UInt8, PTX.AS.Const}
    for P in (G32, S16, S64, C8)
        @test Core.Compiler.return_type(address, Tuple{P}) === P
    end

    cases = (
        (ptx"ld.global.v2.b32", (G32,), :core, NTuple{2, UInt32}),
        (ptx"st.global.v2.b32", (G32, NTuple{2, UInt32}), :core, Nothing),
        (ptx"ldmatrix.sync.aligned.m8n8.x2.shared.b16",
         (S16,), :intrinsic, NTuple{2, UInt32}),
        (ptx"stmatrix.sync.aligned.m8n8.x2.shared.b16",
         (S16, NTuple{2, UInt32}), :intrinsic, Nothing),
        (ptx"mbarrier.init.shared.b64", (S64, UInt32), :intrinsic, Nothing),
        (ptx"cp.async.ca.shared.global",
         (S16, G32, Val{16}), :asm, Nothing),
        (ptx"cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes",
         (S16, C8, Int32, S64), :intrinsic, Nothing),
    )
    for (op, argtypes, tier, rettype) in cases
        info = PTX.lowering(op, argtypes)
        @test info.tier === tier
        @test info.rettype === rettype
        @test which(op, Tuple{argtypes...}) !== PTX._CHAIN_METHOD
    end
end

@testset "transpiled aliases compile through reviewed address roles" begin
    src = """
    .version 9.0
    .target sm_90
    .address_size 64
    .visible .entry _address_alias_vec(.param .u64 p) {
        .reg .u64 %rd0;
        .reg .b32 %r<2>;
        ld.param.u64 %rd0, [p];
        ld.global.v2.b32 {%r0, %r1}, [%rd0];
        ret;
    }
    .visible .entry _address_alias_mbarrier(.param .u64 p) {
        .reg .u64 %rd0;
        ld.param.u64 %rd0, [p];
        mbarrier.init.shared.b64 [%rd0], 1;
        ret;
    }
    """
    julia = PTX.ptx_to_julia(src)
    @test count("address(rd0)", julia) == 2
    Core.eval(@__MODULE__, Meta.parseall(julia))

    cases = (
        (getfield(@__MODULE__, :_address_alias_vec),
         Tuple{Core.LLVMPtr{UInt32, PTX.AS.Global}}),
        (getfield(@__MODULE__, :_address_alias_mbarrier),
         Tuple{Core.LLVMPtr{UInt64, PTX.AS.Shared}}),
    )
    for (f, argtypes) in cases
        ci, rt = only(Base.code_typed(f, argtypes; optimize = true))
        @test rt === Nothing
        @test !occursin("PTX.Address", string(ci))
    end

    # The chain API has typed wrappers for these families, but the source
    # transpiler deliberately rejects them until it has finite parser-side
    # form and operand-role entries.  Do not infer address roles from operand
    # position merely because a direct wrapper happens to exist.
    unsupported = (
        """
        .version 9.0
        .target sm_100a
        .address_size 64
        .visible .entry _address_alias_ldmatrix(.param .u64 p) {
            .reg .u64 %rd0;
            .reg .b32 %r<2>;
            ld.param.u64 %rd0, [p];
            ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%r0, %r1}, [%rd0];
            ret;
        }
        """,
        """
        .version 9.0
        .target sm_100a
        .address_size 64
        .visible .entry _address_alias_tcgen(
                .param .u32 d, .param .u64 ad, .param .u64 bd,
                .param .u32 id, .param .u32 sa, .param .u32 sb) {
            .reg .b32 %r<4>;
            .reg .b64 %rd<2>;
            .reg .pred %p;
            ld.param.u32 %r0, [d];
            ld.param.u64 %rd0, [ad];
            ld.param.u64 %rd1, [bd];
            ld.param.u32 %r1, [id];
            ld.param.u32 %r2, [sa];
            ld.param.u32 %r3, [sb];
            setp.eq.u32 %p, %r1, 0;
            tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [%r0], %rd0, %rd1, %r1, [%r2], [%r3], %p;
            ret;
        }
        """,
    )
    for source in unsupported
        @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(source)
    end
end

@testset "clusterlaunchcontrol.try_cancel closed address schema" begin
    actual = Set((mods, schema.shared_cta, schema.multicast)
                 for (mods, schema) in PTX.CLC_TRY_CANCEL_FORMS)
    expected = Set(EXPECTED_CLC_TRY_CANCEL_FORMS)
    @test actual == expected
    @test length(PTX.CLC_TRY_CANCEL_FORMS) == 4
    for schema in values(PTX.CLC_TRY_CANCEL_FORMS)
        @test schema.introduced == v"8.6"
        @test schema.target_floor == v"10.0"
        @test occursin("9.7.14.18-parallel", schema.section)
        if schema.multicast
            @test schema.multicast_arch_targets ==
                  ("sm_100a", "sm_110a", "sm_120a")
            @test schema.multicast_family_targets ==
                  ("sm_100f", "sm_110f", "sm_120f")
            @test schema.multicast_family_since == v"8.8"
        else
            @test isempty(schema.multicast_arch_targets)
            @test isempty(schema.multicast_family_targets)
            @test schema.multicast_family_since === nothing
        end
    end

    A32 = Address{UInt32}
    A64 = Address{UInt64}
    for (mods, shared, multicast) in
            EXPECTED_CLC_TRY_CANCEL_FORMS
        schema = PTX.clc_try_cancel_schema(mods)
        @test schema !== nothing
        @test schema.shared_cta == shared
        @test schema.multicast == multicast

        # PTX §6.4.1 permits either address width and zero-extends as
        # needed. CUDA 13.3 ptxas independently validates both generic widths
        # below; retain both here so the schema cannot narrow them again.
        carriers = shared ? (A32,) : (A32, A64)
        for A in carriers
            op = Operation{:clusterlaunchcontrol, mods}()
            spec = build_call(:clusterlaunchcontrol, mods, (A, A))
            @test spec.asm ==
                  "clusterlaunchcontrol.$(join(mods, '.')) [\$0], [\$1];"
            @test spec.rettype === Nothing
            @test spec.side_effects
            @test !spec.convergent
            payload = A.parameters[1]
            letter = PTX.constraint_letter(payload)
            @test spec.constraints == "$letter,$letter,~{memory}"
            @test spec.passthrough_argtypes === (payload, payload)
            @test spec.passthrough_unwrap_address === (true, true)
            @test PTX.lowering(op, (A, A)).tier === :chain_asm

            # Raw retains its conservative convergence policy, but it cannot
            # bypass this form's exact grammar or address roles.
            raw = RawOperation{:clusterlaunchcontrol, mods}()
            rawspec = build_call(:clusterlaunchcontrol, mods, (A, A);
                                 raw = true)
            @test rawspec.asm == spec.asm
            @test rawspec.rettype === Nothing
            @test rawspec.convergent
            @test PTX.lowering(raw, (A, A)).tier === :chain_asm
        end
    end

    # Pointer address-space proof is checked when it exists. Integer role
    # carriers intentionally assert the PTX address role but carry no space.
    pS = Core.LLVMPtr{UInt8, PTX.AS.Shared}
    p0 = Core.LLVMPtr{UInt8, PTX.AS.Generic}
    pG = Core.LLVMPtr{UInt8, PTX.AS.Global}
    shared_mods = EXPECTED_CLC_TRY_CANCEL_FORMS[2][1]
    generic_mods = EXPECTED_CLC_TRY_CANCEL_FORMS[1][1]
    @test build_call(:clusterlaunchcontrol, shared_mods, (pS, pS)).rettype === Nothing
    @test build_call(:clusterlaunchcontrol, generic_mods, (p0, p0)).rettype === Nothing
    mixed = build_call(:clusterlaunchcontrol, shared_mods,
                       (pS, Address{UInt32}))
    @test mixed.asm ==
          "clusterlaunchcontrol.$(join(shared_mods, '.')) [\$0], [\$1];"
    @test mixed.passthrough_argtypes === (pS, UInt32)
    @test mixed.passthrough_unwrap_address === (false, true)
    @test_throws ArgumentError build_call(:clusterlaunchcontrol, shared_mods,
                                          (pG, pS))
    @test_throws ArgumentError build_call(:clusterlaunchcontrol, generic_mods,
                                          (pS, p0))
    @test build_call(:clusterlaunchcontrol, generic_mods,
                     (A32, A64)).passthrough_argtypes === (UInt32, UInt64)

    bad_modifiers = (
        (:try_cancel, :async, :b128),
        (:async, :try_cancel, Symbol("mbarrier::complete_tx::bytes"), :b128),
        (:try_cancel, :async, Symbol("mbarrier::complete_tx::bytes"), :b64),
        (:try_cancel, :async, Symbol("multicast::cluster::all"),
         Symbol("mbarrier::complete_tx::bytes"), :b128),
        (:try_cancel, :async, Symbol("shared::cluster"),
         Symbol("mbarrier::complete_tx::bytes"), :b128),
    )
    for mods in bad_modifiers
        op = Operation{:clusterlaunchcontrol, mods}()
        raw = RawOperation{:clusterlaunchcontrol, mods}()
        @test PTX.lowering(op, (A32, A32)).tier === :forbidden
        @test PTX.lowering(raw, (A32, A32)).tier === :forbidden
        @test_throws ArgumentError build_call(:clusterlaunchcontrol, mods,
                                              (A32, A32))
        @test_throws ArgumentError build_call(:clusterlaunchcontrol, mods,
                                              (A32, A32); raw = true)
    end

    for bad_args in ((UInt32, UInt32), (A32,), (A32, A32, A32),
                     (A32, UInt32), (Float32, A32))
        op = Operation{:clusterlaunchcontrol, shared_mods}()
        raw = RawOperation{:clusterlaunchcontrol, shared_mods}()
        @test PTX.lowering(op, bad_args).tier === :forbidden
        @test PTX.lowering(raw, bad_args).tier === :forbidden
        @test_throws ArgumentError build_call(:clusterlaunchcontrol,
                                              shared_mods, bad_args)
        @test_throws ArgumentError build_call(:clusterlaunchcontrol,
                                              shared_mods, bad_args; raw = true)
    end
end

@testset "parser/transpiler preserves simple address roles" begin
    src = """.version 8.6
    .target sm_100
    .address_size 64
    .visible .entry address_probe(
        .param .u64 p,
        .param .u32 response,
        .param .u32 mbar)
    {
        .reg .u64 %rd0;
        .reg .u32 %r0, %r1, %r2;
        ld.param.u64 %rd0, [p];
        ld.global.u32 %r0, [%rd0+16];
        ld.param.u32 %r1, [response];
        ld.param.u32 %r2, [mbar];
        clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%r1], [%r2];
        ret;
    }
    """
    mod = parse_ptx(src)
    insts = Instruction[d for d in only(d for d in mod.directives
                                        if d isa PTX.IR.Function).body
                        if d isa Instruction]
    @test insts[2].operands[2] == AddressOperand("%rd0", "16")
    @test insts[5].operands == (AddressOperand("%r1"), AddressOperand("%r2"))
    @test format(parse_ptx(format(mod))) == format(mod)

    julia = PTX.ptx_to_julia(src)
    @test occursin("ptx\"ld.global.u32\"(address(rd0 + 16))", julia)
    @test occursin(
        "ptx\"clusterlaunchcontrol.try_cancel.async.shared::cta." *
        "mbarrier::complete_tx::bytes.b128\"(address(r1), address(r2))",
        julia)
    parsed = Meta.parseall(julia)
    @test !any(x -> x isa Expr && x.head == :error, parsed.args)

    # A mandatory CLC address written without brackets is rejected by the
    # transpiler itself, before alias propagation or generated-call lowering.
    unbracketed = replace(src,
        "b128 [%r1], [%r2];" => "b128 %r1, [%r2];")
    @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(unbracketed)
    misspelled = replace(src,
        ".mbarrier::complete_tx::bytes.b128" => ".b128")
    @test_throws PTX.Codegen.TranspilerError PTX.ptx_to_julia(misspelled)

    # Non-address labels and immediates retain their existing roles.
    cg = PTX.Codegen.CodeGenState()
    @test PTX.Codegen.render_operand(PTX.IR.LabelOperand("DONE"), cg) == "DONE"
    @test PTX.Codegen.render_operand(PTX.IR.ImmediateOperand("7"), cg;
                                    type_hint = :u32) == "UInt32(7)"
end
