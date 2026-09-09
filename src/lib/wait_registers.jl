# Register dependencies belong on the wait's own asm call: returning the
# inputs after a separate void wait leaves their SSA uses independent.
const _RegisterWait = Union{typeof(ptx"tcgen05.wait::ld.sync.aligned"),
                            typeof(ptx"tcgen05.wait::st.sync.aligned")}
const _WaitRegister = Union{UInt32, Int32, Float32, UInt64, Int64, Float64}

"""
    wait_registers(wait_instruction, values)

Execute `wait_instruction` and return `values` with an explicit compiler
register dependency on that wait. Use the returned values for subsequent
computation. Their bits, types, and tuple structure are preserved.

Supported waits are `ptx"tcgen05.wait::ld.sync.aligned"` and
`ptx"tcgen05.wait::st.sync.aligned"`. Values may be a 32- or 64-bit integer
or floating-point scalar, or a homogeneous tuple of those scalars (including
an empty tuple). The instruction retains its full hardware wait scope;
listing values does not restrict which asynchronous operations it waits for.

```julia
values = ptx"tcgen05.ld.sync.aligned.32x32b.x32.b32"(tmem_addr)
# Independent work can run while the load is in flight.
values = PTX.wait_registers(ptx"tcgen05.wait::ld.sync.aligned", values)
# Consume the returned values here.
```

The wait remains side-effecting and convergent, with a memory clobber.
Tied input/output operands additionally express register dependencies;
a memory clobber alone does not express this dataflow. Adjacent 32-bit tuple
elements travel through paired 64-bit operands to preserve register packing.
The wait emits no arithmetic on the values.

The no-argument instruction call remains available. This API makes a
compiler dependency expressible; it does not imply a demonstrated
wrong-result bug in that existing call.
"""
@inline wait_registers(op::_RegisterWait, value::_WaitRegister) =
    only(wait_registers(op, (value,)))

@inline function wait_registers(op::_RegisterWait, ::Tuple{})
    op()
    ()
end

@generated function wait_registers(::W, values::Tuple{T,Vararg{T,M}}) where
        {W<:_RegisterWait, M, T<:_WaitRegister}
    N = M + 1
    # Packing is a bit reinterpretation, preserving NaNs and signed zeros.
    # Fully explicit tuple accesses keep wide fragments out of local memory.
    paired = sizeof(T) == 4
    npairs = paired ? N ÷ 2 : 0
    carriers = paired ? [fill(UInt64, npairs)..., fill(T, N % 2)...] : fill(T, N)
    args = paired ? [:(UInt64(reinterpret(UInt32, values[$(2i - 1)])) |
                       (UInt64(reinterpret(UInt32, values[$(2i)])) << 32))
                      for i in 1:npairs] : [:(values[$i]) for i in 1:N]
    paired && isodd(N) && push!(args, :(values[$N]))
    n = length(carriers)
    constraints = join([("=" * constraint_letter(C) for C in carriers)...,
                        string.(0:n-1)..., "~{memory}"], ",")
    asm = "tcgen05." * join(W.parameters[2], ".") * ";"
    rt = Tuple{carriers...}
    ir = convergent_asm_ir(asm, constraints, rt, carriers)
    outputs = paired ? [:(reinterpret($T, (result[$(cld(i, 2))] >> $(isodd(i) ? 0 : 32)) % UInt32))
                         for i in 1:2npairs] : [:(result[$i]) for i in 1:N]
    paired && isodd(N) && push!(outputs, :(result[$n]))
    quote
        Base.@inline
        result = Base.llvmcall(($ir, "entry"), $rt, Tuple{$(carriers...)}, $(args...))
        tuple($(outputs...))
    end
end
