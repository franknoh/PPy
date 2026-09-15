# Plugin API

The library plugin interface, version 2. A plugin extends the compiler's
semantics through explicit hooks and nothing else: it types the calls and
attributes of the modules it claims, declares their effects, says how each
recognized operation lowers — as a backend-neutral spec, never as backend
code — and may register dialects, passes, patterns, and lowerings for the
IR. Every hook has a no-op default, so the compiler calls them directly.
What each builtin plugin does is in [Plugins](../internals/plugins.md).

::: ppy_compiler.plugins.base

## Tensor ownership and backend types (PPy 0.3.3)

Plugin interface 2 remains compatible. Plugins using the APIs in this section
must require compiler version 0.3.3 or newer.

`CallResult.arguments` describes what a recognized call does with individual
arguments. Address a positional argument by its zero-based index or a keyword
argument by its name:

```python
CallResult(
    T.NONE,
    arguments=(
        CallArgument(0, ArgumentOwnership.BORROWED),
        CallArgument("out", ArgumentOwnership.MUT, "out must be writable"),
    ),
)
```

`BORROWED` reads an argument only for the duration of the call. `MUT` permits
call-scoped writes and requires a `ppy.Mut[...]` or `ppy.Owned[...]` value.
`OWNED` means the callee may retain the value and therefore requires
`ppy.Owned[...]`. Mutations follow aliases back to the function parameter.
Arguments without a contract keep the conservative behavior: the compiler
assumes the callee may retain them. Starred positional arguments and unpacked
keyword arguments are also conservative because their runtime positions or
names are unknown.

An ownership violation reports E1802 at the argument. Its explanation comes
from `CallArgument.reason`, then `CallResult.reason`, then a compiler default.
A rejected call uses `CallResult.reason` or `RejectSpec.reason` in its E1802
diagnostic.

`Plugin.lower_type_for_backend(type_, facts, backend) -> IRType | None` lets a
plugin claim a representation only when that backend is explicitly selected.
The default returns `None`. `PluginRegistry.lower_type` accepts the optional
`backend` argument, rejects competing backend claims, and otherwise preserves
the registration-order behavior of `Plugin.lower_type`. The shared
`ppy.Tensor` type has no legacy plugin-specific representation when no backend
is selected; its neutral canonical form is handled by the compiler.

## Canonical types and operations (PPy 0.3.2)

Plugin interface 2 remains compatible. Plugins using the following additions
must require compiler version 0.3.2 or newer.

`Plugin.lower_type(type_, facts) -> IRType | None` chooses a canonical IR
representation. The registry asks enabled plugins in registration order and
uses the first non-`None` answer. The frontend supplies the actual `Facts`,
including dtype and shape, for parameters, returns and plugin call results.
Returning `None` lets the builtin type conversion try next. Register every
custom type's dialect through `register_dialects`.

Canonical signatures, local values and function calls use IR types. A custom
tensor or pointer does not need a CPU `NativeSignature`; CPU emission checks
its own ABI and preserves Python fallback when that boundary is unavailable.
An explicit external backend request reports functions it cannot lower with
the function name, source location and reason.

A `CallResult` with `DialectOperationSpec` becomes the named operation. Positional
arguments become SSA operands in source order. The result type comes from
`lower_type`; a `CallResult` with the analysis type `T.NONE` produces zero
results, allowing a store as a statement.
Effects, guards and the call's source location are retained on the operation.
Observable writes cannot be removed by dead-code elimination.

Keyword arguments have explicit roles:

```python
DialectOperationSpec(
    "example",
    "scale",
    attributes=(("mode", "linear"),),
    keyword_operands=("factor",),
    keyword_attributes=("axis",),
)
```

For `scale(x, factor=n, axis=1)`, `x` and `n` are operands and `axis=1`
is an attribute. Keyword operand values are evaluated in Python source order
and appended in the declared order. Omitted keywords are omitted from the
operation; the plugin must supply any required defaults in its contract.
Keyword attributes must be literals or names with proven constant values.
Undeclared keyword roles, unpacked arguments and nonconstant attributes are
rejected. Fixed attributes must agree with corresponding keyword attributes.
Attribute values use the IR's serializable scalar, type, tuple and dictionary
forms. `guards` are contract metadata for backend validation/lowering; a
backend must implement required checks or reject the operation.
