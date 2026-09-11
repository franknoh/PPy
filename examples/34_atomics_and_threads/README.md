# Atomics and threads

`ppy.atomic` and `ppy.concurrent`: shared memory one operation at a time,
and threads with what keeps them apart, over memory the program owns. Four
workers each add 500 to a shared counter and 500 × 2 to a shared total
behind a mutex.

## Atomics with C11's orders

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
CPython the operations serialize under one lock, which implements every
order.

## Threads and what keeps them apart

`concurrent.spawn(f, *args)` runs a function of the module on a new thread
and hands back a handle; `join` waits. The synchronization objects are
memory the program owns: a mutex is one `int` slot, a condition one more
counting notifications, and `wait`/`notify` spin under the CPU's pause hint
over the same atomics on every path. `signal` shows the pattern — a waiter
blocks on a condition until the main thread sets a flag and notifies. A
thread that fails a guard fails its joiner, and the function falls back as a
whole. Native code links pthreads; a target without them is refused with the
reason.

## Run it

```bash
python  counters.ppy
ppy run counters.ppy
```

<!-- outputs:start -->
## What it prints

**`python  counters.ppy`**, **`ppy run counters.ppy`**

```text
2000 4000
2000 True 7 1
40
```

<!-- outputs:end -->

Read on: [Atomics and threads](../../docs/guide/concurrency.md) ·
[Threads](../28_threads/README.md) ·
[Parallel range](../35_parallel_range/README.md)

`counters.ppy` is hand-written; there is no `.py` source and no conversion
step.
