# Coroutines: `ppy.aio`

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

`aio.sleep(seconds)`, `accept(fd)`, `connect(host, port)`, `read(fd, p, n)`,
and `write(fd, p, n)` are the awaitables; `spawn(coroutine)` starts one as
a task to await later; `listen(host, port)`, `port(fd)`, and `close(fd)`
are immediate. A socket is an `int`, bytes move through a
`native.ptr[ppy.u8]`, and every socket operation answers a negative errno
rather than raising, so a compiled coroutine and the Python one say the
same thing. `aio.run(coroutine)` drives it to its value. `E1645` names a
misuse.

Under CPython the awaitables are asyncio's and `run` is `asyncio.run`. The
compiler lowers an `async def` whose awaits are these and other coroutines
of the module -- of scalars and pointers, returning one scalar or nothing
-- to the async dialect of the IR ([The IR](../internals/ir.md)): a state
machine the native async runtime drives, one loop per process over epoll,
timers, and non-blocking sockets. Calling such a coroutine hands back a
future the runtime owns; `aio.run` runs the loop until it completes, and
awaiting it from asyncio steps the native loop between the Python loop's
turns. `aio.compiled(f)` says whether that is so here. An await the
compiler does not know keeps the coroutine in Python, as does a machine
without the runtime -- Linux and a C compiler build it once into the cache
-- and nothing is ever turned into a blocking call to look correct. A
guard failing inside a running native coroutine cannot fall back to
Python: the future fails and `aio.run` raises `aio.NativeGuardFailed`,
naming the coroutine; `--safeguards off`, or a body the prover clears,
keeps guards out of it.

Examples: [Coroutines](../howto/37_aio.md).
