```@meta
CurrentModule = PTX
```

# Reference

The full public API. Everything else under `PTX`, `PTX.IR`,
`PTX.Codegen`, `PTX.Parser` is internal and may change without notice.

## Authoring

```@docs
@ptx_str
@sreg_str
PTX.@mod_str
```

## Pointers

```@docs
PTX.Address
PTX.address
PTX.reinterpret_addrspace
```

## Extended-precision arithmetic

```@docs
PTX.add_with_carry
PTX.sub_with_borrow
PTX.mul_wide
```

## Vector loads

```@docs
PTX.vector_load
```

## Transpiler

```@docs
ptx_to_julia
ir_to_julia
```

## Parser

```@docs
PTX.Parser.tokenize
PTX.Parser.Token
PTX.Parser.TokenKind
PTX.Parser.LexError
PTX.Parser.parse
PTX.Parser.ParseError
```

## IR

```@docs
PTX.IR.format
```

## Index

```@index
```
