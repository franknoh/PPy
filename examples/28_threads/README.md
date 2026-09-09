# One native function, two threads, 1.95×

`busy` is a 200-million-iteration loop. Under plain CPython two threads
running it take as long as one, because the GIL serializes them. Under
`ppy run` the generated boundary releases the GIL for the whole native
call, and two threads finish in half the time. Same file, same
`threading.Thread`; the scaling line at the bottom is the difference.

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
wrapper wraps the call in `Py_BEGIN_ALLOW_THREADS`. A function with an
effect that can reach the interpreter keeps the GIL, and one that performs
I/O is not lowered at all. Borrowed buffers get the same treatment as
NumPy gives them: the boundary pins the memory for the whole call.

This is Python threads calling native code. The other direction — threads
and shared memory *inside* native code — is
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
1 thread    3304.6 ms
2 threads   6419.7 ms
scaling       1.03x
```

**`ppy run threads.ppy`**

```text
1 thread     124.8 ms
2 threads    132.3 ms
scaling       1.89x
```

<!-- outputs:end -->

## Read on

- [Atomics and threads](../../docs/guide/concurrency.md) — the native side.
- [Architecture: the boundary](../../docs/internals/architecture.md) — what the generated wrapper does per call.

`threads.ppy` is hand-written; there is no `.py` source and no conversion step.
