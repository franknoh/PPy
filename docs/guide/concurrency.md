# Atomics and threads: `ppy.atomic` and `ppy.concurrent`

Both have a reference implementation under CPython and a lowering to a
dialect of the IR ([The IR](../internals/ir.md)), so a program using them
runs the same on every path.

## `ppy.atomic`

`ppy.atomic` is shared memory, one operation at a time, on a
`native.ptr[int]` (or a byte pointer; `load`, `store`, and `exchange` take
`float` too): `load`, `store`, `exchange`, `compare_exchange` -- which
answers `(the value found, whether it swapped)` -- `fetch_add`,
`fetch_sub`, `fetch_and`, `fetch_or`, `fetch_xor`, and `fence`. Each
takes `order="seq_cst"` unless said otherwise (`relaxed`, `acquire`,
`release`, `acq_rel`), and the checker holds what C11 holds: a load is
not `release`, a store is not `acquire`, a relaxed fence orders nothing
(`E1641`). Under CPython the operations serialize under one lock.

## `ppy.concurrent`

`ppy.concurrent` is threads and what keeps them apart. `spawn(f, *args)`
runs a function of the module -- one that returns nothing, with the
parameters a native call takes -- on a new thread and hands back its
handle; `join(handle)` waits for it, and a thread that failed a guard
fails its joiner, which falls back as a whole. The synchronization
objects are memory the program owns, so they are pointers: a mutex is one
`int` slot (`lock`, `unlock`; zero is unlocked), a condition is one `int`
slot counting notifications (`wait(condition, mutex)`, `notify`), a
barrier is two `int` slots (`barrier(slots, parties)`), and every path
implements them the same way over the atomics, spinning with the CPU's
pause hint. `thread_id()` names the running thread. `E1642` names a
misuse. Native code links pthreads for these; a target without them is
refused with the reason.

Generated wrappers release the GIL around native calls, so `@ppy.native`
functions scale on ordinary Python threads too: measured 1.95× on two
threads against 0.98× for the same code on plain CPython
([Threads](../howto/28_threads.md)).

Examples: [Atomics and threads](../howto/34_atomics_and_threads.md),
[Threads](../howto/28_threads.md).
