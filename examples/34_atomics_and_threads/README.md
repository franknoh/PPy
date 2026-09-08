# Atomics and threads

Shared memory one operation at a time, and threads with what keeps them
apart: `ppy.atomic` and `ppy.concurrent`, over memory the program owns.

## Provenance

Hand-written. `counters.ppy` is written directly; there is no `.py` source
and no conversion step involved.

## What it shows

- `atomic.fetch_add`, `compare_exchange`, `exchange`, `load`, and `fence`
  take an `order` the checker holds to what C11 holds: a relaxed fence
  orders nothing, a load is never `release`.
- `concurrent.spawn(f, *args)` runs a function of the module on a new
  thread and `join` waits; four workers add to one counter and to one total
  behind a mutex, and the answer is the same on every path because the
  operations are the same operations.
- A mutex is one `int` slot and a condition one more: `wait`, `notify`, and
  the spin under the CPU's pause hint are implemented the same way over the
  atomics on every path. Under CPython the atomics serialize under one lock.
- Native code links pthreads for these; a target without them is refused
  with the reason.

## Run it

```bash
python  counters.ppy
ppy run counters.ppy
```
