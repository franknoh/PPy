# Coroutines: `ppy.aio`

`ppy.aio` gives you sockets and sleeps as awaitables. They run under asyncio
in CPython and on a native epoll loop when compiled.

## Example

```python
import ppy
from ppy import aio, native


async def echo_once(listening: int) -> int:
    client = await aio.accept(listening)
    room = native.stack_alloc[ppy.u8](64)
    got = await aio.read(client, room, 64)
    sent = await aio.write(client, room, got)
    aio.close(client)
    return sent


def serve() -> int:
    return aio.run(echo_once(aio.listen("127.0.0.1", 8000)))
```

## The operations

| kind | operations |
|---|---|
| awaitables | `aio.sleep(seconds)`, `accept(fd)`, `connect(host, port)`, `read(fd, p, n)`, `write(fd, p, n)` |
| task | `spawn(coroutine)` starts one as a task to await later |
| immediate | `listen(host, port)`, `port(fd)`, `close(fd)` |
| driver | `aio.run(coroutine)` drives it to its value |

A socket is an `int`, and bytes move through a `native.ptr[ppy.u8]`. Each
socket operation answers a negative errno rather than raising, so a compiled
coroutine and the Python one say the same thing. `E1645` names a misuse.

## Under CPython

The awaitables are asyncio's, and `run` is `asyncio.run`.

## Native coroutines

The compiler lowers an `async def` to the async dialect of the IR
([The IR](../internals/ir.md)) when its awaits are these and other
coroutines of the module, over scalars and pointers, returning one scalar or
nothing. The result is a state machine the native async runtime drives: one
loop per process over epoll, timers, and non-blocking sockets.

Calling such a coroutine hands back a future the runtime owns. `aio.run` runs
the loop until it completes. Awaiting it from asyncio steps the native loop
between the Python loop's turns. `aio.compiled(f)` says whether that is so
here.

## When a coroutine stays in Python

- An await the compiler does not know keeps the coroutine in Python.
- So does a machine without the runtime. The runtime needs Linux and a C
  compiler, and is built once into the cache.

Nothing is ever turned into a blocking call to look correct.

## Guard failures

A guard failing inside a running native coroutine cannot fall back to
Python. The future fails, and `aio.run` raises `aio.NativeGuardFailed`,
naming the coroutine. `--safeguards off`, or a body the prover clears, keeps
guards out of it.

Examples: [Coroutines](../howto/37_aio.md).
