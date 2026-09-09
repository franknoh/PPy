# Coroutines

An echo server and its client in one program. `ppy.aio` is asyncio under
CPython and a native runtime — one loop per process over epoll, timers, and
non-blocking sockets — where the compiler lowers the coroutines. Same
program, same number.

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
rather than raising, so a compiled coroutine and the Python one say the
same thing. `aio.spawn` starts the server as a task; `aio.run` drives the
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
