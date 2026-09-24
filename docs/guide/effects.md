# Effects and purity

The compiler infers an effect set for each function. This page lists the
effects, explains `@ppy.pure`, and describes the three paths a program can run
on.

## The effect vocabulary

All consumers share one vocabulary: purity, native and GPU eligibility, code
motion, inlining, parallelization, fusion, the async lowering, and plugin
contracts.

```text
alloc  read_object  write_object  read_memory  write_memory
read_global  write_global  io  network  random  time
thread  process  sync  atomic  may_raise
python_callback  python_dynamic  gpu_launch  device_memory  external_unknown
```

`read_memory` and `write_memory` are native memory a pointer or a buffer
reaches. `read_object` and `write_object` are Python objects.

The IR carries each function's effects, and passes read them. For example, an
unused call to a function that only allocates and reads is dead code.

## `@ppy.pure`

`@ppy.pure` asserts the set contains none of the forbidden effects. Everything
is forbidden except allocation, reads, and raising.

The checker proves it interprocedurally:

- a pure function calling something with unknown effects is `E1602`
- a callee that mutates the caller's argument is charged to the caller

## Purity and lowering

Purity and lowering are two different questions about one write. Purity asks
whether anything could observe it. Lowering asks whether CPython has to
perform it.

A function that fills memory it allocated and passes it on lowers, but is not
`@ppy.pure`, because the callee saw the object before the function returned.

## The three execution paths

| Command | What runs |
|---|---|
| `python f.ppy` | plain CPython. `import ppy` installs a `sys.meta_path` finder so `.py` files can import `.ppy` modules: natively when the compiler is installed and the module checks clean, as Python source otherwise. |
| `ppy f.ppy` | the optimized Python backend: AST-level optimization (folding, inlining, LICM, loop transforms) executed by CPython. |
| `ppy run f.ppy` | eligible functions compile through LLVM; everything else runs the Python body. |

Any observable difference between the three is a compiler bug. The test suite
and `examples/run_all.py` compare all three on every example.

`ppy build` produces the third path ahead of time. Its launcher runs through
`ppy_runtime` with machine code from the library built next to it.
`ppy build --standalone` goes further: a native executable with no CPython
inside, for programs whose reachable graph is entirely native.

## Guards and fallback

A guard that fails is a fallback, never a different answer: native code that
cannot keep a promise runs the Python body.

One exception stands above that rule. A check `--sanitize` inserted does not
fall back but raises `SanitizerFailure` ([CLI](../cli.md)).

Examples: [Effects and contracts](../howto/03_effects_and_contracts.md),
[Errors](../howto/18_errors.md).
