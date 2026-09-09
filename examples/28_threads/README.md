# Threads

A native function releases the GIL, so compute in it scales across threads.
`busy` is a 200-million-iteration loop; under plain CPython two threads take
as long as one, under `ppy run` they finish in half the time. Same file,
same `threading.Thread`.

## Nothing to hold the GIL for

```python
@ppy.pure
@ppy.opt(3)
def busy(rounds: int) -> int:
    total: int = 0
    for i in range(rounds):
        total += i % 7
    return total
```

Once its arguments are unpacked, `busy` touches no Python object, so the
generated wrapper wraps the call in `Py_BEGIN_ALLOW_THREADS`. A function
with an effect that can reach the interpreter keeps the GIL, and one that
performs I/O is not lowered at all. Borrowed buffers get the same treatment
NumPy gives them: the boundary pins the memory for the whole call. The
scaling line at the bottom of the output is the measurement, 1.95× here.

This is Python threads calling native code. The other direction — threads
and shared memory inside native code — is
[`ppy.concurrent` and `ppy.atomic`](../34_atomics_and_threads/README.md).

## Run it

```bash
python  threads.ppy
ppy run threads.ppy
```

<!-- outputs:start -->
## What it prints

**`python  threads.ppy`**

```text
1 thread    3013.2 ms
2 threads   6009.0 ms
scaling       1.00x
```

**`ppy run threads.ppy`**

```text
1 thread     119.4 ms
2 threads    120.5 ms
scaling       1.98x
```

<!-- outputs:end -->

Read on: [Atomics and threads](../../docs/guide/concurrency.md) ·
[Architecture: the boundary](../../docs/internals/architecture.md)

`threads.ppy` is hand-written; there is no `.py` source and no conversion step.
