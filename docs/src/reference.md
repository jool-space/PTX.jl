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
