# Coroutines

An echo server and its client in one program. `ppy.aio` is asyncio under
CPython and a native runtime — one loop per process over epoll, timers, and
non-blocking sockets — where the compiler lowers the coroutines.

## Sockets are integers, bytes are pointers

```python
async def echo_once(listening: int) -> int:
    client = await aio.accept(listening)
    room = native.stack_alloc[ppy.u8](8)
    got = await aio.read(client, room, 8)
    sent = await aio.write(client, room, got)
    aio.close(client)
    return sent
```

`aio.listen`, `port`, `accept`, `connect`, `read`, `write`, `close`, and
`sleep` are the vocabulary. A socket is an `int`, bytes move through a
`native.ptr[ppy.u8]`, and every socket operation answers a negative errno
rather than raising, which is what a state machine with no exception to
throw can do. `aio.spawn` starts the server as a task; `aio.run` drives the
client to its value, and the answer folds the bytes echoed, the byte read
back, and the doubled sleep into one number.

## What the native loop takes, and what it leaves

An `async def` whose awaits are these operations and other coroutines of
the module — over scalars and pointers, returning one scalar or nothing —
becomes a state machine: `lower-async` splits each coroutine into a starter
and a resume function, and a value read across an `await` lives in the
frame. An await the compiler does not know keeps the coroutine in Python,
as does a machine without the runtime (Linux and a C compiler build it once
into the cache). Nothing is ever turned into a blocking call to look
correct. `aio.compiled(main)` says which happened here; the line that
prints it starts with `# `, the mark for output that may differ by path.

## Compared with asyncio and uvloop

Twenty thousand eight-byte round trips over one loopback connection --
a client and an echo server in one process -- in [`compare/`](compare/):
[`echo_bench.ppy`](compare/echo_bench.ppy),
[`echo_asyncio.py`](compare/echo_asyncio.py),
[`echo_uvloop.py`](compare/echo_uvloop.py). Milliseconds for the whole run,
over five processes.

**PPY** is the coroutines as the example writes them, lowered to state
machines over the native loop; **asyncio** is the standard library's
streams, as an echo server is usually written; **uvloop** is the same
program on the libuv-backed loop:

```python
async def serve(listening: int, rounds: int) -> int:
    client = await aio.accept(listening)
    room = native.stack_alloc[ppy.u8](8)
    total = 0
    for _ in range(rounds):
        got = await aio.read(client, room, 8)
        total += await aio.write(client, room, got)
    aio.close(client)
    return total
```

```python
async def serve(reader, writer):
    total = 0
    for _ in range(ROUNDS):
        data = await reader.readexactly(8)
        writer.write(data)
        await writer.drain()
        total += len(data)
    writer.close()
    await writer.wait_closed()
    return total
```

<!-- compare:start -->
| | PPY `ppy.aio` | asyncio streams | uvloop |
|---|---:|---:|---:|
| 20000 round trips | **103.98 ± 2.18** | 810.53 ± 298.97 | 721.51 ± 358.05 |
<!-- compare:end -->

A round trip is four socket operations and four resumptions of a
coroutine. Under asyncio each of those is a Python frame, a `Future`, and
a trip through the event loop's callback queue; uvloop moves the loop and
the transports into C and leaves the coroutines in Python; PPY's state
machines resume in native code and call `read` and `write` directly, so
what remains per round trip is the syscalls. The program pays for that in
vocabulary: sockets are integers, bytes go through pointers, and an error
is a negative errno, where asyncio's streams give you `bytes` and
exceptions.

Intel Core Ultra 9 386H; uvloop 0.22.1 on CPython 3.12.13, asyncio and PPY
on CPython 3.14.5, from a checkout on a native filesystem.

## Run it

```bash
python  echo.ppy
ppy run echo.ppy
ppy emit ir echo.ppy
```

<!-- outputs:start -->
## What it prints

**`python  echo.ppy`**

```text
42333003
# compiled coroutine here: False
```

**`ppy run echo.ppy`**

```text
42333003
# compiled coroutine here: True
```

**`ppy emit ir echo.ppy`**

*466 lines: [outputs/03-ppy-emit-ir-echo-ppy.txt](outputs/03-ppy-emit-ir-echo-ppy.txt)*

<!-- outputs:end -->

Read on: [Coroutines](../../docs/guide/aio.md) ·
[The IR: the async dialect](../../docs/internals/ir.md)

`echo.ppy` is hand-written; there is no `.py` source and no conversion step.
