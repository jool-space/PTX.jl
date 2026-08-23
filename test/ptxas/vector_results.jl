# Offline compiler evidence for every audited vector-result core cell and each
# exact ptxas-compatible atom example. These kernels never load onto a GPU.

_vr_is_fp8(form) = form.lane_kind in
    (:e4m3, :e4m3x2, :e4m3x4, :e5m2, :e5m2x2, :e5m2x4)

_vr_ptxas_rejects_acc(form) =
    form.op === :multimem && form.coremods[2] in (:min, :max) &&
    any(mod -> mod in (Symbol("acc::f16"), Symbol("acc::f32")), form.coremods)

function _vr_partition(form)
    form.op === :ld && return form.target.min_sm == v"10.0" ? :ld100 : :ld75
    form.op === :atom && return :atom90
    form.op === :multimem && return _vr_is_fp8(form) ? :multimem_fp8 : :multimem90
    error("unpartitioned vector-result form $(form.op).$(join(form.coremods, '.'))")
end

# Ledger forms beyond the managed assembler's ISA ceiling are spelled-only
# and excluded from every assembly partition. The exclusion tracks the
# shipped toolkit (CUDA 13.3 → 9.3 skips the two 9.4 forms; a 13.4+ artifact
# assembles them) — the pinned per-ceiling skip count below fails loudly if
# the exclusion drifts. Resolved once at load: the partition bodies are
# built during @generated expansion, which must not touch artifact state.
const _VR_ASSEMBLER_ISA = _ptxas_isa()
_vr_spelled_only(form) = form.ptx_version > _VR_ASSEMBLER_ISA

@testset "forms beyond the assembler ceiling are excluded, and counted" begin
    skipped = [form for form in PTX.VECTOR_RESULT_CORE_FORMS
               if _vr_spelled_only(form)]
    @test length(skipped) == (_VR_ASSEMBLER_ISA >= v"9.4" ? 0 : 2)
    @test all(f -> f.coremods[1] === :add && :noftz in f.coremods &&
                   f.lane_kind === :f32, skipped)
end

function _vr_representative_mods(form)
    form.op === :ld && return form.target.min_sm == v"10.0" ?
        (:global, Symbol("L2::evict_last"), form.coremods...) :
        (:weak, :global, form.coremods...)
    form.coremods
end

function _vr_call_expr(form, mods = _vr_representative_mods(form))
    callee = :(PTX.Operation{$(QuoteNode(form.op)), $mods}())
    if form.op === :atom
        source = form.lane_kind === :f16 ? :f16 :
                 form.lane_kind === :bf16 ? :bf16 :
                 form.lane_kind === :f32 ? :f32 : :packed
        tuple = Expr(:tuple, fill(source, form.lanes)...)
        return Expr(:call, callee, :ptr, tuple)
    end
    Expr(:call, callee, :ptr)
end

function _vr_partition_body(partition; compat = false,
                            compiler_status = :accepted)
    compiler_status in (:accepted, :rejected) ||
        error("unknown vector-result compiler status: $compiler_status")
    body = Expr(:block)
    for form in PTX.VECTOR_RESULT_CORE_FORMS
        _vr_spelled_only(form) && continue
        _vr_partition(form) === partition || continue
        (_vr_ptxas_rejects_acc(form) == (compiler_status === :rejected)) || continue
        push!(body.args, _vr_call_expr(form))
    end
    if compat
        for record in PTX.VECTOR_RESULT_COMPAT_FORMS
            mods = only(record.spellings)
            schema = PTX.schema(PTX.VectorLedger(), :atom, mods)
            push!(body.args, _vr_call_expr(schema.form, mods))
        end
    end
    push!(body.args, :(return nothing))
    body
end

@generated function _vr_ld75!(ptr::Core.LLVMPtr{UInt8,PTX.AS.Global},
                              f16::Float16, bf16::UInt16,
                              f32::Float32, packed::UInt32)
    _vr_partition_body(:ld75)
end

@generated function _vr_ld100!(ptr::Core.LLVMPtr{UInt8,PTX.AS.Global},
                               f16::Float16, bf16::UInt16,
                               f32::Float32, packed::UInt32)
    _vr_partition_body(:ld100)
end

@generated function _vr_atom90!(ptr::Core.LLVMPtr{UInt8,PTX.AS.Global},
                                f16::Float16, bf16::UInt16,
                                f32::Float32, packed::UInt32)
    _vr_partition_body(:atom90; compat = true)
end

@generated function _vr_multimem90!(ptr::Core.LLVMPtr{UInt8,PTX.AS.Global},
                                    f16::Float16, bf16::UInt16,
                                    f32::Float32, packed::UInt32)
    _vr_partition_body(:multimem90)
end

@generated function _vr_multimem_fp8!(ptr::Core.LLVMPtr{UInt8,PTX.AS.Global},
                                      f16::Float16, bf16::UInt16,
                                      f32::Float32, packed::UInt32)
    _vr_partition_body(:multimem_fp8)
end

@generated function _vr_multimem90_rejected!(
        ptr::Core.LLVMPtr{UInt8,PTX.AS.Global}, f16::Float16, bf16::UInt16,
        f32::Float32, packed::UInt32)
    _vr_partition_body(:multimem90; compiler_status = :rejected)
end

@generated function _vr_multimem_fp8_rejected!(
        ptr::Core.LLVMPtr{UInt8,PTX.AS.Global}, f16::Float16, bf16::UInt16,
        f32::Float32, packed::UInt32)
    _vr_partition_body(:multimem_fp8; compiler_status = :rejected)
end

const _VR_KERNEL_TYPES = Tuple{
    Core.LLVMPtr{UInt8,PTX.AS.Global}, Float16, UInt16, Float32, UInt32,
}

function _vr_run_ptxas(ptx::String, target::String)
    mktempdir() do dir
        ptx_path = joinpath(dir, "vector_results.ptx")
        cubin_path = joinpath(dir, "vector_results.cubin")
        write(ptx_path, ptx)
        cmd = `$(CUDACore.CUDA_Compiler.ptxas()) --gpu-name $target --output-file $cubin_path $ptx_path`
        log = IOBuffer()
        proc = run(pipeline(ignorestatus(cmd), stdout = log, stderr = log))
        (; accepted = success(proc), log = String(take!(log)))
    end
end

# CUDACore 6.2's target database cannot emit sm_110, while CUDA 13.3 ptxas
# accepts both exact PTX 9.3 feature targets. Mirror the deliberately narrow
# oracle in blackwell.jl: the body is emitted at sm_100a/f and only the unique
# target directive changes, with an exact reverse-roundtrip assertion.
function _vr_retarget_sm100_to_sm110(ptx::String, source_target::String,
                                     target::String)
    @assert source_target in ("sm_100a", "sm_100f")
    @assert target == replace(source_target, "sm_100" => "sm_110")
    source_directive = ".target $source_target"
    target_directive = ".target $target"
    @assert length(findall(source_directive, ptx)) == 1
    retargeted = replace(ptx, source_directive => target_directive; count = 1)
    @assert replace(retargeted, target_directive => source_directive;
                    count = 1) == ptx
    retargeted
end

@testset "ptxas-accepted vector-result cells at retained targets" begin
    # The two 9.4 add.noftz.f32 forms join :atom90 once the assembler
    # accepts ISA 9.4 — both counts stay pinned per ceiling.
    atom90_count = _VR_ASSEMBLER_ISA >= v"9.4" ? 34 : 32
    partitions = (
        (_vr_ld75!, :ld75, 24, v"7.5", :baseline),
        (_vr_ld100!, :ld100, 8, v"10.0", :baseline),
        (_vr_atom90!, :atom90, atom90_count, v"9.0", :baseline),
        (_vr_multimem90!, :multimem90, 42, v"9.0", :baseline),
        (_vr_multimem_fp8!, :multimem_fp8, 56, v"10.0", :arch),
    )
    for (kernel, partition, expected_count, cap, feature_set) in partitions
        forms = filter(form -> !_vr_spelled_only(form) &&
                             _vr_partition(form) === partition &&
                             !_vr_ptxas_rejects_acc(form),
                       PTX.VECTOR_RESULT_CORE_FORMS)
        @test length(forms) == expected_count
        @test ptxas_compiles(kernel, _VR_KERNEL_TYPES; cap, feature_set)
        ptx = emit_ptx(kernel, _VR_KERNEL_TYPES; cap, feature_set)
        for form in forms
            head = PTX.build_head(form.op, _vr_representative_mods(form))
            @test occursin(head, ptx)
        end
        if partition === :atom90
            for record in PTX.VECTOR_RESULT_COMPAT_FORMS
                @test occursin(PTX.build_head(:atom, only(record.spellings)), ptx)
            end
        end
    end
end

@testset "FP8 multimem target feature sets" begin
    # PTX 9.3 lists four architecture-specific targets and two family-specific
    # targets for the FP8 forms. Compile all 56 forms accepted by current
    # ptxas at each one so numeric-minimum flattening cannot masquerade as
    # evidence. sm_110a/f use the exact-directive retarget oracle above.
    targets = (
        (v"10.0", :arch, "sm_100a"), (v"11.0", :arch, "sm_110a"),
        (v"12.0", :arch, "sm_120a"), (v"12.1", :arch, "sm_121a"),
        (v"10.0", :family, "sm_100f"), (v"11.0", :family, "sm_110f"),
    )
    for (cap, feature_set, target) in targets
        if cap == v"11.0"
            source_target = feature_set === :arch ? "sm_100a" : "sm_100f"
            source = emit_ptx(_vr_multimem_fp8!, _VR_KERNEL_TYPES;
                              cap = v"10.0", feature_set)
            ptx = _vr_retarget_sm100_to_sm110(source, source_target, target)
            result = _vr_run_ptxas(ptx, target)
            @test result.accepted
        else
            @test ptxas_compiles(_vr_multimem_fp8!, _VR_KERNEL_TYPES;
                                 cap, feature_set)
            ptx = emit_ptx(_vr_multimem_fp8!, _VR_KERNEL_TYPES;
                           cap, feature_set)
        end
        @test occursin(".target $target", ptx)
    end
end

@testset "spec-derived accumulated min/max compiler boundary" begin
    # PTX 9.3's syntax and independent legality tables admit these 48 cells,
    # but CUDA 12.9 and 13.3 ptxas reject them as illegal reductions. Preserve
    # them in the ISA ledger/API while pinning the current compiler evidence
    # separately, so future assembler support makes this test request review.
    rejected90 = filter(form -> _vr_partition(form) === :multimem90 &&
                                _vr_ptxas_rejects_acc(form),
                        PTX.VECTOR_RESULT_CORE_FORMS)
    rejected_fp8 = filter(form -> _vr_partition(form) === :multimem_fp8 &&
                                  _vr_ptxas_rejects_acc(form),
                          PTX.VECTOR_RESULT_CORE_FORMS)
    @test length(rejected90) == 20
    @test length(rejected_fp8) == 28

    ptx90 = emit_ptx(_vr_multimem90_rejected!, _VR_KERNEL_TYPES;
                     cap = v"9.0", feature_set = :baseline)
    for form in rejected90
        @test occursin(PTX.build_head(form.op, form.coremods), ptx90)
    end
    result90 = _vr_run_ptxas(ptx90, "sm_90")
    @test !result90.accepted
    message90 = "Illegal reduction operation for instruction " *
                "'multimem.ld_reduce.acc::f32'"
    @test count(line -> occursin(message90, line),
                eachline(IOBuffer(result90.log))) == 20

    # Exercise the FP8 rejection at every exact/family target listed by the
    # ISA. CUDACore emits exact 10.x/12.x targets; 11.x uses the narrow target
    # retarget because the installed target database predates sm_110.
    targets = (
        (v"10.0", :arch, "sm_100a"), (v"11.0", :arch, "sm_110a"),
        (v"12.0", :arch, "sm_120a"), (v"12.1", :arch, "sm_121a"),
        (v"10.0", :family, "sm_100f"), (v"11.0", :family, "sm_110f"),
    )
    for (cap, feature_set, target) in targets
        if cap == v"11.0"
            source_target = feature_set === :arch ? "sm_100a" : "sm_100f"
            source = emit_ptx(_vr_multimem_fp8_rejected!, _VR_KERNEL_TYPES;
                              cap = v"10.0", feature_set)
            ptx = _vr_retarget_sm100_to_sm110(source, source_target, target)
        else
            ptx = emit_ptx(_vr_multimem_fp8_rejected!, _VR_KERNEL_TYPES;
                           cap, feature_set)
        end
        for form in rejected_fp8
            @test occursin(PTX.build_head(form.op, form.coremods), ptx)
        end
        result = _vr_run_ptxas(ptx, target)
        @test !result.accepted
        message = "Illegal reduction operation for instruction " *
                  "'multimem.ld_reduce.acc::f16'"
        @test count(line -> occursin(message, line),
                    eachline(IOBuffer(result.log))) == 28
    end
end

function _vr_i8_output_probe(ptr::Core.LLVMPtr{UInt8,PTX.AS.Global})
    ptx"ld.global.v2.u8"(ptr)
    ptx"ld.global.v2.s8"(ptr)
    ptx"multimem.ld_reduce.add.v4.e5m2"(ptr)
    return nothing
end

@testset "i8 vector outputs bridge through legal PTX registers" begin
    types = Tuple{Core.LLVMPtr{UInt8,PTX.AS.Global}}
    @test ptxas_compiles(_vr_i8_output_probe, types;
                         cap = v"10.0", feature_set = :arch)
    llvm = emit_llvm(_vr_i8_output_probe, types;
                     cap = v"10.0", feature_set = :arch)
    @test occursin(r"call \{ i8, i8 \} asm sideeffect \"ld\.global\.v2\.u8", llvm)
    @test occursin(r"call \{ i8, i8 \} asm sideeffect \"ld\.global\.v2\.s8", llvm)
    @test occursin(r"call \{ i8, i8, i8, i8 \} asm sideeffect \"\{ \.reg \.b8 vector_result_lane<4>", llvm)
    @test occursin("mov.b16 \$0, {vector_result_lane0, 0}", llvm)
    ptx = emit_ptx(_vr_i8_output_probe, types;
                   cap = v"10.0", feature_set = :arch)
    @test occursin(r"\.reg \.b16\s+%rs", ptx)
    @test occursin(".reg .b8 vector_result_lane<4>", ptx)
    @test occursin("multimem.ld_reduce.add.v4.e5m2", ptx)
end

function _vr_hint_and_sink_probe(ptr::Core.LLVMPtr{UInt8,PTX.AS.Global},
                                 policy::UInt64,
                                 vals::NTuple{2,Float32})
    ptx"ld.global.L2::cache_hint.v2.u32"(ptr, policy)
    ptx"atom.global.add.L2::cache_hint.v2.f32"(ptr, vals, policy)
    vector_load(ptx"ld.global.L2::evict_last.v8.u32", ptr,
                Val((true, false, true, true, false, true, true, true)))
    return nothing
end

function _vr_cache_hint_without_policy_ptx(op::Symbol)
    instruction, registers = if op === :ld
        ("ld.global.L2::cache_hint.v2.u32 {%r0, %r1}, [%rd];",
         ".reg .b32 %r<2>;")
    elseif op === :atom
        ("atom.global.add.L2::cache_hint.v2.f32 " *
         "{%r0, %r1}, [%rd], {%r2, %r3};",
         ".reg .b32 %r<4>;")
    else
        error("unsupported cache-hint probe opcode: $op")
    end
    """
    .version 8.1
    .target sm_90
    .address_size 64
    .visible .entry cache_hint_probe(.param .u64 input) {
        .reg .b64 %rd;
        $registers
        ld.param.u64 %rd, [input];
        $instruction
        ret;
    }
    """
end

function _vr_all_sink_ptx()
    """
    .version 8.8
    .target sm_100
    .address_size 64
    .visible .entry all_sink_probe(.param .u64 input) {
        .reg .b64 %rd;
        ld.param.u64 %rd, [input];
        ld.global.L2::evict_last.v8.u32 {_, _, _, _, _, _, _, _}, [%rd];
        ret;
    }
    """
end

function _vr_atom_immediate_ptx(kind::Symbol, values::AbstractString)
    declaration = kind in (:f16, :bf16) ? ".reg .b16 %d<2>;" :
                  kind === :f16x2 ? ".reg .b32 %d<2>;" :
                  kind === :f32 ? ".reg .f32 %d<2>;" :
                  error("unsupported vector atom immediate kind: $kind")
    noftz = kind === :f32 ? "" : ".noftz"
    """
    .version 8.8
    .target sm_90
    .address_size 64
    .visible .entry atom_immediate_probe(.param .u64 input) {
        .reg .b64 %rd;
        $declaration
        ld.param.u64 %rd, [input];
        atom.global.add$noftz.v2.$kind {%d0, %d1}, [%rd], $values;
        ret;
    }
    """
end

function _vr_atom_sink_ptx()
    """
    .version 8.8
    .target sm_90
    .address_size 64
    .visible .entry atom_sink_probe(.param .u64 input) {
        .reg .b64 %rd;
        .reg .f32 %a<2>;
        ld.param.u64 %rd, [input];
        mov.f32 %a0, 0f3f800000;
        mov.f32 %a1, 0f40000000;
        atom.global.add.v2.f32 _, [%rd], {%a0, %a1};
        ret;
    }
    """
end

@testset "vector atom immediate and bit-bucket compiler boundary" begin
    # Vector atom source constants are format-specific. CUDA 12.9/13.3 ptxas
    # reject half, bfloat, packed-half, and integer-as-f32 immediate lanes.
    # Keep these raw assembler checks independent from transpiler validation.
    rejected = ((:f16, "{0, 0}"),
                (:f16, "{1.0, 0f40000000}"),
                (:bf16, "{0, 0}"),
                (:bf16, "{1.0, 0f40000000}"),
                (:f16x2, "{0, 0}"),
                (:f16x2, "{1.0, 0f40000000}"),
                (:f32, "{1, 2}"))
    for (kind, values) in rejected
        result = _vr_run_ptxas(_vr_atom_immediate_ptx(kind, values), "sm_90")
        @test !result.accepted
        @test occursin("Arguments mismatch for instruction 'atom'", result.log)
    end

    # f32 floating constants are accepted, including f32/f64 exact encodings
    # and decimal notation; the transpiler performs their f32 use-site cast.
    for values in ("{0f3f800000, 0d4000000000000000}",
                   "{0F3f800000, 0D4000000000000000}",
                   "{-1.5, 2.0e0}")
        result = _vr_run_ptxas(_vr_atom_immediate_ptx(:f32, values), "sm_90")
        @test result.accepted
    end

    # PTX's whole-result `_` is a valid vector atomic reduction. This differs
    # from the invalid per-lane `{_, _}` destination shape.
    sink = _vr_run_ptxas(_vr_atom_sink_ptx(), "sm_90")
    @test sink.accepted
end

@testset "cache-policy dependency and wide-load sinks assemble" begin
    types = Tuple{Core.LLVMPtr{UInt8,PTX.AS.Global}, UInt64,
                  NTuple{2,Float32}}
    @test ptxas_compiles(_vr_hint_and_sink_probe, types;
                         cap = v"10.0", feature_set = :baseline)
    ptx = emit_ptx(_vr_hint_and_sink_probe, types;
                   cap = v"10.0", feature_set = :baseline)
    lines = strip.(collect(eachline(IOBuffer(ptx))))
    @test count(line -> startswith(line, "ld.global.L2::cache_hint.v2.u32 "),
                lines) == 1
    @test count(line -> startswith(
                    line, "atom.global.add.L2::cache_hint.v2.f32 "),
                lines) == 1
    @test count(line -> startswith(
                    line, "ld.global.L2::evict_last.v8.u32 "),
                lines) == 1
    @test occursin(r"ld\.global\.L2::evict_last\.v8\.u32\s+\{[^}]*_[^}]*_[^}]*\}", ptx)
    @test !occursin("{_, _, _, _, _, _, _, _}", ptx)

    # PTX 9.3's per-lane sink prose does not forbid this spelling, but ptxas
    # cannot infer an element register type when no destination is live. The
    # production helper rejects it before emission rather than deleting a
    # potentially synchronizing load.
    all_sink = _vr_run_ptxas(_vr_all_sink_ptx(), "sm_100")
    @test !all_sink.accepted
    @test occursin("Unable to infer type of vector elements", all_sink.log)

    # The ISA grammar makes the hint and policy independently optional and
    # states policy => hint. NVIDIA's CUDA 12.9 and 13.3 assemblers enforce
    # the converse too. Pin that toolchain boundary independently of the API
    # fail-loud tests, using canonical scalar-equivalent vector spellings.
    for op in (:ld, :atom)
        result = _vr_run_ptxas(_vr_cache_hint_without_policy_ptx(op), "sm_90")
        @test !result.accepted
        @test occursin("Arguments mismatch for instruction '$op'", result.log)
    end
end
