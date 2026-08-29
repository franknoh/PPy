# Diagnostics

Every diagnostic has a stable code (spec 29.1). `ppy explain E1301` prints
the description; `--no-strict` downgrades the strict-mode errors that have a
sound fallback, and never the rest.

## Source and module structure

| code | meaning |
|---|---|
| `E1001` | The source could not be parsed by the configured CPython grammar. |
| `E1002` | The source file could not be read. |
| `E1003` | A module is provided by both a .py and a .ppy source under the same qualified name. |

## Names and imports

| code | meaning |
|---|---|
| `E1101` | A name is used before any binding reaches it. |
| `E1102` | An import target could not be resolved to a module in the project or environment. |
| `E1103` | A `from ... import *` makes the module namespace unanalyzable. |

## Annotations and attributes

| code | meaning |
|---|---|
| `E1201` | A parameter has no annotation and no inferable type, which would introduce an implicit `Any`. |
| `E1202` | An attribute could not be resolved on a statically known type. |
| `E1203` | A generic type was used bare, which would introduce an implicit `Any` element type. |
| `E1204` | A decorator applies an unknown transform, so the decorated signature is unknown. |
| `E1205` | A `ppy.` decorator names a directive the runtime does not export. |
| `E1206` | An attribute was read through a value that may be `None`. |

## Types

| code | meaning |
|---|---|
| `E1301` | An assignment or argument does not match the declared type. |
| `E1302` | An operator has no definition for the operand types. |
| `E1303` | A returned value does not match the declared return type. |
| `E1304` | A stable type could not be inferred for a parameter or local. |
| `E1305` | A call does not match the callee's parameter list. |
| `E1306` | A callable's signature is unknown, so the call cannot be typed. |

## Fixed-width contracts

| code | meaning |
|---|---|
| `E1401` | A value provably leaves the range promised by a fixed-width marker. |
| `E1402` | A fixed-width contract needs a runtime check that the selected contract mode forbids. |

## Dynamic features

| code | meaning |
|---|---|
| `E1501` | `eval`, `exec`, or runtime code-object construction is not statically analyzable. |
| `E1502` | Mutation of `globals()`, `locals()`, or frame locals is not statically analyzable. |
| `E1503` | An import target is not a compile-time constant. |
| `E1504` | A dynamic Python feature requires an explicit `ppy.dynamic` boundary. |
| `E1505` | A dynamic boundary is forbidden by the project configuration. |
| `E1506` | Attribute or class mutation after analysis is not supported. |
| `E1507` | A class is constructed dynamically, from computed bases or an unvouched metaclass. |

## Purity

| code | meaning |
|---|---|
| `E1601` | A function declared `@ppy.pure` performs a forbidden effect. |
| `E1602` | A function declared `@ppy.pure` calls a function with unknown effects. |

## Directive requirements

| code | meaning |
|---|---|
| `E1701` | `@ppy.parallel(require=True)` could not be satisfied for the selected backend. |
| `E1702` | `@ppy.native(require=True)` could not be satisfied without an opaque Python call. |

## Backends and tools

| code | meaning |
|---|---|
| `E1801` | The selected backend is unavailable in this environment. |
| `E1802` | A construct is not supported by the selected backend. |

## Remarks

| code | meaning |
|---|---|
| `R3001` | An optimization remark. |
| `R3002` | The converter promoted a list parameter to a borrowed buffer. |
| `R3003` | A list parameter is close to being a borrowed buffer but something blocks it. |

## Warnings

| code | meaning |
|---|---|
| `W2001` | A module is shadowed by a same-named source with a different extension. |
| `W2002` | A `bool` value takes part in arithmetic, which is legal but usually unintended. |
| `W2003` | Unknown `Annotated` metadata was preserved but not interpreted. |
| `W2004` | A directive had no effect for the selected backend. |
| `W2005` | Conversion left both a .py and a .ppy source for the same module. |
