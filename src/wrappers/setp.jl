# Dual-pred form of `setp.<cmp>.<dtype>`. Hand-written because the chain has
# no signal to choose between single-pred (returns Bool) and dual-pred
# (returns Tuple{Bool, Bool}); the transpiler injects a `.dual` modifier
# when the parser produces a `PipeOperand` destination.

const SETP_INT_DTYPES = (
    (:b16, UInt16), (:b32, UInt32), (:b64, UInt64),
    (:u16, UInt16), (:u32, UInt32), (:u64, UInt64),
    (:s16, Int16),  (:s32, Int32),  (:s64, Int64),
)

const SETP_FLOAT_DTYPES = (
    (:f16, Float16), (:f32, Float32), (:f64, Float64),
)

const SETP_CMPS = (:eq, :ne, :lt, :gt, :le, :ge)

function setp_dual_spec(cmp::Symbol, dt::Symbol, JT::Type)
    asm = "setp.$cmp.$dt \$0|\$1, \$2, \$3;"
    letter = constraint_letter(JT)
    constraints = "=b,=b,$letter,$letter"
    (; asm, constraints, rettype = Tuple{Bool, Bool})
end

for (dt, JT) in (SETP_INT_DTYPES..., SETP_FLOAT_DTYPES...), cmp in SETP_CMPS
    mods = (:dual, cmp, dt)
    spec = setp_dual_spec(cmp, dt, JT)
    asm, constraints = spec.asm, spec.constraints
    @eval function (::Operation{:setp, $mods})(a::$JT, b::$JT)
        Base.@inline
        @asmcall($asm, $constraints, false,
                 Tuple{Bool, Bool}, Tuple{$JT, $JT}, a, b)
    end
end
