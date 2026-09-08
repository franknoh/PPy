# Four threads, one counter, one answer

Four workers each add 500 to a shared counter with `atomic.fetch_add` and
500 × 2 to a shared total behind a mutex. The program prints `2000 4000` on
plain CPython, on the Python backend, and natively — because `ppy.atomic`
and `ppy.concurrent` are the same operations on every path, over memory the
program owns.

## Atomics with C11's orders, and C11's rules

```python
def worker(counter: native.ptr[int], mutex: native.ptr[int], total: native.ptr[int], rounds: int) -> None:
    for _ in range(rounds):
        atomic.fetch_add(counter, 1, order="relaxed")
        concurrent.lock(mutex)
        native.store(total, native.load(total) + 2)
        concurrent.unlock(mutex)
```

`load`, `store`, `exchange`, `compare_exchange`, the `fetch_*` family, and
`fence` each take an `order` — `relaxed`, `acquire`, `release`, `acq_rel`,
`seq_cst` — and the checker holds what C11 holds: a load is never
`release`, a store never `acquire`, a relaxed fence orders nothing
(`E1641`). Natively each lowers to the instruction of that order; under
CPython the operations serialize under one lock, which is a valid
implementation of every order.

## Threads and what keeps them apart

`concurrent.spawn(f, *args)` runs a function of the module on a new thread
and hands back a handle; `join` waits. The synchronization objects are
memory the program owns: a mutex is one `int` slot, a condition one more
counting notifications, and `wait`/`notify` spin under the CPU's pause hint
over the same atomics on every path. `signal` shows the pattern — a waiter
blocks on a condition until the main thread sets a flag and notifies. A
thread that fails a guard fails its joiner, and the function falls back as
a whole. Native code links pthreads; a target without them is refused with
the reason.

## Run it

```bash
python  counters.ppy
ppy run counters.ppy
```

<!-- outputs:start -->
## What it prints

**`python  counters.ppy`**

```text
2000 4000
2000 True 7 1
40
```

**`ppy run counters.ppy`**

```text
2000 4000
2000 True 7 1
40
```

<!-- outputs:end -->

## Read on

- [Atomics and threads](../../docs/guide/concurrency.md) — the two namespaces in full.
- [Threads](../28_threads/README.md) — the other direction: Python threads calling native code.
- [Parallel range](../35_parallel_range/README.md) — when you want the split done for you.

`counters.ppy` is hand-written; there is no `.py` source and no conversion
step.
