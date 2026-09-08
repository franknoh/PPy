# Coroutines

An echo server and its client in one program: `ppy.aio` is asyncio under
CPython and a native runtime -- one loop per process over epoll, timers,
and non-blocking sockets -- where the compiler lowers the coroutines.

## Provenance

Hand-written. `echo.ppy` is written directly; there is no `.py` source and
no conversion step involved.

## What it shows

- `aio.listen`, `port`, `accept`, `connect`, `read`, `write`, `close`, and
  `sleep`: a socket is an `int`, bytes move through a `native.ptr[ppy.u8]`,
  and every socket operation answers a negative errno rather than raising,
  so a compiled coroutine and the Python one say the same thing.
- `aio.spawn` starts the server as a task; `aio.run` drives the client to
  its value, and the answer folds the bytes echoed, the byte read back, and
  the doubled sleep into one number the three paths print alike.
- `aio.compiled(main)` says whether the native runtime took the coroutine
  here (Linux and a C compiler build it once into the cache); the line that
  prints it starts with `# `, the mark for what may differ by path.
- An `async def` the compiler cannot take -- an await it does not know --
  stays in Python; nothing is ever turned into a blocking call to look
  correct.

## Run it

```bash
python  echo.ppy
ppy run echo.ppy
ppy emit ir echo.ppy   # the async dialect: create, await, the IO operations
```
