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

## Module globals

A function that reads a module global can still run natively under
`ppy run`, if the global is settled. A settled global is bound once by the
module's own top-level code and never bound again: no second assignment at
the top level, no `global` statement that assigns it, and no `del`.

```python
PRIMES: list[int] = [p for p in range(2, 100) if all(p % d for d in range(2, p))]
SEEN: dict[int, int] = {}


def count(n: int) -> int:
    total = 0
    for p in PRIMES:
        if n % p == 0:
            SEEN[p] = SEEN.get(p, 0) + 1
            total += 1
    return total
```

Here is how a settled global reaches native code:

- Native code takes each settled global the function reads as one more
  parameter, after its own.
- When Python calls the function, the boundary reads each global from the
  module, as it is at that moment. A container crosses as any container
  argument does, and a write to it (`SEEN[p] = ...`) is copied back into the
  module's object.
- A native caller passes on the globals it was given, so a function that
  calls `count` takes `PRIMES` and `SEEN` as well, even from another module.

The analysis cannot see every rebinding: `setattr(module, ...)` from
elsewhere, or a test that patches the name. For that reason the boundary
reads the global at every call instead of once:

- if a global was rebound to another object of the same type, the call uses
  the new object;
- if the name is gone, or now holds a different type, the Python body runs.

A literal constant (`LIMIT = 10`) is folded where it is read and needs none
of this.

A global that is not settled keeps its readers in Python. Neither kind
reaches a standalone build or a C export, and a method, a nested function,
a thread's body, and a function used as a value are not passed globals, so
they too stay in Python when they read one.

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
