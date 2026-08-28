# Threads

A native region releases the GIL, so compute in it scales across threads.

## Provenance

Hand-written. `threads.ppy` is written directly; there is no `.py` source and
no conversion step involved.

## What it shows

- `busy` touches no Python object once its arguments are unpacked, so the
  generated boundary wraps the call in `Py_BEGIN_ALLOW_THREADS`.
- Under plain CPython the same file is bound by the GIL and two threads take
  twice as long as one.
- A function with an effect that can reach the interpreter keeps the GIL, and
  one that performs I/O is not lowered natively at all.

Borrowed buffers are covered too: the boundary pins the memory for the whole
call, which is the guarantee NumPy relies on when it does the same thing.

## Run it

```bash
python  threads.ppy
ppy run threads.ppy
```
