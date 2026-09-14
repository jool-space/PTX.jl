# `spcompress` / `spdecompress` (PTX 9.4 §9.7.10.30–31, sm_107a): register-
# level structured-sparsity compression and expansion. Both are per-thread
# data movement over brace-enclosed b32 register vectors whose widths depend
# on every qualifier, so they are typed-wrapper-only (TYPED_WRAPPER_ONLY_RULES)
# and render through `plain_asm_ir` without a sideeffect marker: the FORMS
# contract is :pure, and an unused result may be dropped like any other
# computation. Every output is early-clobber (`=&r`): the ISA makes register
# reuse among the vector operands undefined behaviour.
#
# spcompress.elemsize.idxsize.sp::2:4.num mdata, cdata, data, spdesc
#   data  = 2·num registers (dense input), cdata = num (kept elements),
#   mdata = ceil(num·idxsize/elemsize) (selected indices); the descriptor
#   (see `spcompress_desc`) picks the element type and the selection rule.
#   Julia: (data::NTuple{2num, UInt32}, spdesc::UInt32) -> (mdata, cdata).
#
# spdecompress.elemsize.idxsize.sp::S:T.num data, mdata, cdata
#   mdata = ceil(S·idxsize·num/32), cdata = ceil(S·elemsize·num/32),
#   data  = ceil(T·elemsize·num/32); the ISA admits a tuple only when
#   S·elemsize ≤ 32, a `.b2` index can address T (T ≤ 4), and the output
#   spans 32..4096 bits (the 253-register cap never binds).
#   Julia: (mdata::NTuple, cdata::NTuple) -> data::NTuple.
#
# ptxas 13.4 admits both opcodes at sm_107a and rejects `.sp::2:4` at
# sm_107f; the Table 39 sizes above are the ones it accepts.

const _SPCOMPRESS_NUMS = (1, 2, 4, 8, 16, 32, 64)
const _SPDECOMPRESS_FACTORS = ((1, 2), (1, 4), (1, 8), (1, 16),
                               (2, 4), (2, 8), (2, 16), (4, 8), (4, 16))

_sp_bits(size::Symbol) = size === :b8 ? 8 : size === :b16 ? 16 :
                         size === :b2 ? 2 : size === :b4 ? 4 :
                         error("unknown spcompress width $size")

function _sp_regs(k::Int)
    join(("\$$(k0)" for k0 in k), ", ")
end

function _spcompress_register(elem::Symbol, idx::Symbol, num::Int)
    mods = (elem, idx, Symbol("sp::2:4"), Symbol("x", num))
    ndata = 2 * num
    ncdata = num
    nmdata = cld(num * _sp_bits(idx), _sp_bits(elem))
    nout = nmdata + ncdata
    outs = 0:nout - 1
    mregs = join(("\$$k" for k in 0:nmdata - 1), ", ")
    cregs = join(("\$$k" for k in nmdata:nout - 1), ", ")
    dregs = join(("\$$k" for k in nout:nout + ndata - 1), ", ")
    asm = "spcompress.$elem.$idx.sp::2:4.x$num {$mregs}, {$cregs}, " *
          "{$dregs}, \$$(nout + ndata);"
    constraints = join([fill("=&r", nout); fill("r", ndata + 1)], ",")
    flat = Tuple{fill(UInt32, nout)...}
    ir = plain_asm_ir(asm, constraints, flat,
                      (fill(UInt32, ndata)..., UInt32); sideeffect = false)
    argtypes = Tuple{fill(UInt32, ndata)..., UInt32}
    dvals = [:(data[$i]) for i in 1:ndata]
    mvals = [:(r[$i]) for i in 1:nmdata]
    cvals = [:(r[$i]) for i in nmdata + 1:nout]
    register_wrapper!(:spcompress, :spcompress, mods, :asm)
    @eval @inline function (::Operation{:spcompress, $mods})(
            data::NTuple{$ndata, UInt32}, spdesc::UInt32)
        r = Base.llvmcall(($ir, "entry"), $flat, $argtypes,
                          $(dvals...), spdesc)
        (($(mvals...),), ($(cvals...),))::Tuple{NTuple{$nmdata, UInt32},
                                               NTuple{$ncdata, UInt32}}
    end
    nothing
end

function _spdecompress_sizes(elem::Symbol, idx::Symbol, s::Int, t::Int,
                             num::Int)
    e = _sp_bits(elem)
    i = _sp_bits(idx)
    nmdata = cld(s * i * num, 32)
    ncdata = cld(s * e * num, 32)
    ndata = cld(t * e * num, 32)
    admitted = s * e <= 32 && (idx !== :b2 || t <= 4) &&
               t * e * num >= 32 && t * e * num <= 4096 &&
               nmdata + ncdata + ndata <= 253
    admitted, nmdata, ncdata, ndata
end

function _spdecompress_register(elem::Symbol, idx::Symbol, s::Int, t::Int,
                                num::Int)
    admitted, nmdata, ncdata, ndata = _spdecompress_sizes(elem, idx, s, t, num)
    admitted || return nothing
    mods = (elem, idx, Symbol("sp::$s:$t"), Symbol("x", num))
    nin = nmdata + ncdata
    dregs = join(("\$$k" for k in 0:ndata - 1), ", ")
    mregs = join(("\$$k" for k in ndata:ndata + nmdata - 1), ", ")
    cregs = join(("\$$k" for k in ndata + nmdata:ndata + nin - 1), ", ")
    asm = "spdecompress.$elem.$idx.sp::$s:$t.x$num {$dregs}, {$mregs}, " *
          "{$cregs};"
    constraints = join([fill("=&r", ndata); fill("r", nin)], ",")
    rt = NTuple{ndata, UInt32}
    ir = plain_asm_ir(asm, constraints, rt, Tuple(fill(UInt32, nin));
                      sideeffect = false)
    argtypes = Tuple{fill(UInt32, nin)...}
    mvals = [:(mdata[$i]) for i in 1:nmdata]
    cvals = [:(cdata[$i]) for i in 1:ncdata]
    register_wrapper!(:spdecompress, :spdecompress, mods, :asm)
    @eval @inline function (::Operation{:spdecompress, $mods})(
            mdata::NTuple{$nmdata, UInt32}, cdata::NTuple{$ncdata, UInt32})
        Base.llvmcall(($ir, "entry"), $rt, $argtypes,
                      $(mvals...), $(cvals...))
    end
    nothing
end

for elem in (:b8, :b16), idx in (:b2, :b4), num in _SPCOMPRESS_NUMS
    _spcompress_register(elem, idx, num)
end

for elem in (:b8, :b16), idx in (:b2, :b4),
        (s, t) in _SPDECOMPRESS_FACTORS, num in _SPCOMPRESS_NUMS
    _spdecompress_register(elem, idx, s, t, num)
end
