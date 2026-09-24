# Atomics and threads: `ppy.atomic` and `ppy.concurrent`

This page covers atomic operations on shared memory and the threads and
synchronization objects built on them. Both namespaces have a reference
implementation under CPython and a lowering to a dialect of the IR
([The IR](../internals/ir.md)), so a program using them runs the same on
every path.

## `ppy.atomic`

`ppy.atomic` is shared memory, one operation at a time, on a
`native.ptr[int]` or a byte pointer. `load`, `store`, and `exchange` also take
`float`.

The operations:

- `load`, `store`, `exchange`
- `compare_exchange`, which answers `(the value found, whether it swapped)`
- `fetch_add`, `fetch_sub`, `fetch_and`, `fetch_or`, `fetch_xor`
- `fence`

### Memory orders

Each operation takes `order="seq_cst"` unless told otherwise (`relaxed`,
`acquire`, `release`, `acq_rel`). The checker holds what C11 holds, and
reports a violation as `E1641`:

- a load is not `release`
- a store is not `acquire`
- a relaxed fence orders nothing

Under CPython the operations serialize under one lock.

## `ppy.concurrent`

`ppy.concurrent` is threads and what keeps them apart.

### Spawning and joining

`spawn(f, *args)` runs a function of the module on a new thread and hands back
its handle. The function returns nothing and has the parameters a native call
takes. `join(handle)` waits for it. A thread that failed a guard fails its
joiner, which falls back as a whole. `thread_id()` names the running thread.

### Synchronization objects

The synchronization objects are memory the program owns, so they are
pointers:

| object | memory | operations |
|---|---|---|
| mutex | one `int` slot; zero is unlocked | `lock`, `unlock` |
| condition | one `int` slot counting notifications | `wait(condition, mutex)`, `notify` |
| barrier | two `int` slots | `barrier(slots, parties)` |

Each path implements them the same way over the atomics, spinning with the
CPU's pause hint.

### Diagnostics and platform

`E1642` names a misuse. Native code links pthreads for these. A target
without them is refused with the reason.

## Native functions on Python threads

Generated wrappers release the GIL around native calls, so `@ppy.native`
functions scale on ordinary Python threads too. Measured: 1.95× on two
threads, against 0.98× for the same code on plain CPython
([Threads](../howto/28_threads.md)).

Examples: [Atomics and threads](../howto/34_atomics_and_threads.md),
[Threads](../howto/28_threads.md).
