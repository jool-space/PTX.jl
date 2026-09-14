# Exact-target compiler evidence for the sm_107a spcompress and spdecompress
# families: every registered form assembles on sm_107a with the operand
# shape the ISA prescribes, and the sm_107f family and sm_100a targets refuse
# the `.sp::2:4` qualifier. The host inventory and lowering contracts live in
# host/spcompress.jl.

_sp_forms(family, elem) =
    [m for m in PTX.wrapper_asm_forms(family) if m[1] === elem]

_sp_bits(s) = s === :b8 ? 8 : s === :b16 ? 16 : s === :b2 ? 2 : 4

# Every registered form of one element width, every output register folded
# into the stored accumulator (the forms carry no sideeffect marker, so an
# unused result would be dropped).
@generated function _spcompress_surface!(out::CuDeviceVector{UInt32, 1},
                                         ::Val{elem}, seed::UInt32,
                                         spdesc::UInt32) where {elem}
    body = Expr(:block, :(acc = UInt32(0)))
    for mods in _sp_forms(:spcompress, elem)
        num = parse(Int, String(mods[4])[2:end])
        data = Expr(:tuple, (:(seed + UInt32($i)) for i in 1:2 * num)...)
        push!(body.args, :(r = PTX.Operation{:spcompress, $mods}()($data, spdesc)))
        push!(body.args, :(acc ⊻= reduce(⊻, r[1]) ⊻ reduce(⊻, r[2])))
    end
    push!(body.args, :(Base.@inbounds out[1] = acc))
    push!(body.args, :(return nothing))
    body
end

# spdecompress: one kernel per form. The widest forms produce 128 registers
# each, and consecutive forms sharing input values would keep more than the
# 255-register budget live in a single kernel.
@generated function _spdecompress_surface!(out::CuDeviceVector{UInt32, 1},
                                           ::Val{mods}, seed::UInt32) where {mods}
    e = _sp_bits(mods[1])
    i = _sp_bits(mods[2])
    s = parse(Int, split(String(mods[3])[5:end], ':')[1])
    num = parse(Int, String(mods[4])[2:end])
    nm = cld(s * i * num, 32)
    nc = cld(s * e * num, 32)
    mdata = Expr(:tuple, (:(seed & UInt32(0x11111111)) for _ in 1:nm)...)
    cdata = Expr(:tuple, (:(seed + UInt32($k)) for k in 1:nc)...)
    quote
        r = PTX.Operation{:spdecompress, $mods}()($mdata, $cdata)
        Base.@inbounds out[1] = reduce(⊻, r)
        return nothing
    end
end

@testset "spcompress and spdecompress assemble on sm_107a only" begin
    if _ptxas_isa() < v"9.4"
        @test_skip "PTX 9.4 assembler required"
    else
        out = CuDeviceVector{UInt32, 1}
        for elem in (:b8, :b16)
            types = Tuple{out, Val{elem}, UInt32, UInt32}
            @test ptxas_compiles(_spcompress_surface!, types;
                                 cap = v"10.7", feature_set = :arch)
            ptx = emit_ptx(_spcompress_surface!, types;
                           cap = v"10.7", feature_set = :arch)
            for mods in _sp_forms(:spcompress, elem)
                @test occursin(PTX.build_head(:spcompress, mods) * " {", ptx)
            end
            @test ptxas_rejects(_spcompress_surface!, types; cap = v"10.7",
                                feature_set = :family, target = "sm_107f")
            @test ptxas_rejects(_spcompress_surface!, types; cap = v"10.0",
                                feature_set = :arch, target = "sm_100a")
        end
        # The ISA's own example spelling, exact operand shape.
        ptx = emit_ptx(_spcompress_surface!, Tuple{out, Val{:b8}, UInt32, UInt32};
                       cap = v"10.7", feature_set = :arch)
        @test occursin(r"spcompress\.b8\.b2\.sp::2:4\.x4 \{%r\d+\}, \{%r\d+, %r\d+, %r\d+, %r\d+\}, \{(%r\d+, ){7}%r\d+\}, %r\d+;",
                       ptx)

        for mods in PTX.wrapper_asm_forms(:spdecompress)
            types = Tuple{out, Val{mods}, UInt32}
            @test ptxas_compiles(_spdecompress_surface!, types;
                                 cap = v"10.7", feature_set = :arch)
            @test occursin(PTX.build_head(:spdecompress, mods) * " {",
                           emit_ptx(_spdecompress_surface!, types;
                                    cap = v"10.7", feature_set = :arch))
        end
        types = Tuple{out, Val{(:b8, :b2, Symbol("sp::2:4"), :x32)}, UInt32}
        @test ptxas_rejects(_spdecompress_surface!, types; cap = v"10.7",
                            feature_set = :family, target = "sm_107f")
        @test ptxas_rejects(_spdecompress_surface!, types; cap = v"10.0",
                            feature_set = :arch, target = "sm_100a")
        ptx = emit_ptx(_spdecompress_surface!, types;
                       cap = v"10.7", feature_set = :arch)
        @test occursin(r"spdecompress\.b8\.b2\.sp::2:4\.x32 \{(%r\d+, ){31}%r\d+\}, \{%r\d+, %r\d+, %r\d+, %r\d+\}, \{(%r\d+, ){15}%r\d+\};",
                       ptx)
    end
end
