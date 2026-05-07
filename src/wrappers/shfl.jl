# Warp-level register shuffle (PTX 9.2 §9.7.4). Both output forms — data only
# and (data, in-range-pred). The pred form takes a trailing `.pred` chain
# modifier; transpiler injects it when the parser produces a `PipeOperand`
# destination.

const SHFL_MODES = (:up, :down, :bfly, :idx)

function shfl_spec(mode::Symbol; pred::Bool = false)
    mode in SHFL_MODES || error("shfl_spec: unknown mode :$mode")
    if pred
        (; asm = "shfl.sync.$mode.b32 \$0|\$1, \$2, \$3, \$4, \$5;",
           constraints = "=r,=b,r,r,r,r,~{memory}",
           rettype = Tuple{UInt32, Bool})
    else
        (; asm = "shfl.sync.$mode.b32 \$0, \$1, \$2, \$3, \$4;",
           constraints = "=r,r,r,r,r,~{memory}",
           rettype = UInt32)
    end
end

function _shfl_register(mode::Symbol)
    mods = (:sync, mode, :b32)
    spec = shfl_spec(mode)
    asm, constraints = spec.asm, spec.constraints
    @eval function (::Operation{:shfl, $mods})(a::UInt32, b::UInt32, c::UInt32,
                                                membermask::UInt32)
        Base.@inline
        @asmcall($asm, $constraints, true,
                 UInt32, Tuple{UInt32, UInt32, UInt32, UInt32},
                 a, b, c, membermask)
    end

    mods_pred = (:sync, mode, :b32, :pred)
    spec_pred = shfl_spec(mode; pred = true)
    asm_pred, constraints_pred = spec_pred.asm, spec_pred.constraints
    @eval function (::Operation{:shfl, $mods_pred})(a::UInt32, b::UInt32, c::UInt32,
                                                     membermask::UInt32)
        Base.@inline
        @asmcall($asm_pred, $constraints_pred, true,
                 Tuple{UInt32, Bool}, Tuple{UInt32, UInt32, UInt32, UInt32},
                 a, b, c, membermask)
    end
    nothing
end

for mode in SHFL_MODES
    _shfl_register(mode)
end
