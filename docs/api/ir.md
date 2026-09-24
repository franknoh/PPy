# IR

`ppy_compiler.ir` is the typed canonical IR that sits between analysis and
the backends. Its modules:

- `model`: the data structure
- `types`: the types
- `dialect`: the extension point
- `verify`: the rules
- `printer`, `parser`, `codec`: the text form

[The IR](../internals/ir.md) describes the format and the dialects.

## Model

::: ppy_compiler.ir.model

## Types

::: ppy_compiler.ir.types

## Dialects

::: ppy_compiler.ir.dialect

## Passes

::: ppy_compiler.ir.passes

## Patterns

::: ppy_compiler.ir.pattern

## Verification

::: ppy_compiler.ir.verify

## The linker

::: ppy_compiler.ir.linker

## Text

::: ppy_compiler.ir.codec

::: ppy_compiler.ir.printer

::: ppy_compiler.ir.parser

## Shared passes

::: ppy_compiler.ir.transforms
