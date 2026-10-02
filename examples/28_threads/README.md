# Threads

A native function releases the GIL, so compute in it scales across threads.
`busy` is a 200-million-iteration loop. Under plain CPython two threads take
as long as one; under `ppy run` they finish in half the time. It is the same
file and the same `threading.Thread`.

## Run it

```bash
python  threads.ppy
ppy run threads.ppy
```

<!-- outputs:start -->
## What it prints

**`python  threads.ppy`**

```text
1 thread    3314.1 ms
2 threads   6709.8 ms
scaling       0.99x
```

**`ppy run threads.ppy`**

```text
1 thread     130.7 ms
2 threads    147.4 ms
scaling       1.77x
```

<!-- outputs:end -->

The scaling line at the bottom of the output is the measurement, 1.95× here.

## The kernel

```python
@ppy.pure
@ppy.opt(3)
def busy(rounds: int) -> int:
    total: int = 0
    for i in range(rounds):
        total += i % 7
    return total
```

Once its arguments are unpacked, `busy` touches no Python object. The
generated wrapper therefore wraps the call in `Py_BEGIN_ALLOW_THREADS`.

## When the GIL stays held

- A function with an effect that can reach the interpreter keeps the GIL:
  one that prints, reads input, opens a file, or calls a Python function
  ([Effects in native code](../../docs/guide/native-effects.md)).
- A short straight-line function keeps it too. Releasing and retaking the
  GIL costs about 20 ns, more than such a body saves.
- Borrowed buffers get the same treatment NumPy gives them: the boundary
  pins the memory for the whole call.

## Threads inside native code

This example is Python threads calling native code. The other direction,
threads and shared memory inside native code, is
[`ppy.concurrent` and `ppy.atomic`](../34_atomics_and_threads/README.md).

Read on: [Atomics and threads](../../docs/guide/concurrency.md) ·
[Architecture: the boundary](../../docs/internals/architecture.md)

`threads.ppy` is hand-written; there is no `.py` source and no conversion step.
