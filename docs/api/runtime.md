# `ppy_runtime`

Everything a built artifact needs at launch, and nothing more. This package
never imports `ppy_compiler`: a compiled application depends on the
interpreter, this runtime, its native library, and its manifest, and keeps
working with the compiler uninstalled.

## Launching

::: ppy_runtime.launch

::: ppy_runtime.manifest

## Binding and the ABI

::: ppy_runtime.binding

::: ppy_runtime.abi

## Runtimes native code calls into

::: ppy_runtime.arrow

::: ppy_runtime.aio

::: ppy_runtime.cuda

::: ppy_runtime.xla

## Staged artifacts

::: ppy_runtime.exported

::: ppy_runtime.regions
