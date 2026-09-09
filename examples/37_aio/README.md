# An echo server and its client, as coroutines that compile

`echo.ppy` listens on a port, spawns a server task, connects a client,
writes `hi!`, reads it back, and folds the bytes into one number. Under
CPython `ppy.aio` is asyncio. Under `ppy run` the compiler lowers the
coroutines to the async dialect of the IR and a native runtime drives
them — one loop per process over epoll, timers, and non-blocking sockets.
Same program, same number.

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
client to its value.

## What the native loop takes, and what it leaves

An `async def` whose awaits are these operations and other coroutines of
the module — over scalars and pointers, returning one scalar or nothing —
becomes a state machine: `lower-async` splits each coroutine into a
starter and a resume function, and a value read across an `await` lives in
the frame. An await the compiler does not know keeps the coroutine in
Python, as does a machine without the runtime (Linux and a C compiler build
it once into the cache). Nothing is ever turned into a blocking call to
look correct. `aio.compiled(main)` says which happened here; the line that
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

```text
ppyir 1
module @echo
dialect async 1
dialect core 1
attrs {ppy.libraries = ["ppy_aio"]}

private global @ppy.str.0 : buffer<u8> = "127.0.0.1"
private global @ppy.str.1 : buffer<u8> = "127.0.0.1"

func @echo_wait_and_double(%n: i64) -> future<i64> attrs {effects = ["may_raise", "sync", "time"], ppy.abi = "ppy", ppy.async = true, ppy.async.lowered = true, ppy.qualname = "echo.wait_and_double", ppy.releases_gil = true, ppy.symbol = "ppy_echo_wait_and_double"} loc("examples/37_aio/echo.ppy":5:0) {
^entry:
    %frame = async.frame_new {slots = 7} : ptr<i64>
    %0 = core.const 3 : i64
    %1 = core.ptr_offset %frame, %0 : ptr<i64>
    core.store %n, %1
    %future = async.spawn %frame {callee = @echo_wait_and_double_resume} : future<i64>
    core.ret %future
}

func @echo_echo_once(%listening: i64) -> future<i64> attrs {effects = ["alloc", "network", "read_memory", "sync", "write_memory"], ppy.abi = "ppy", ppy.async = true, ppy.async.lowered = true, ppy.qualname = "echo.echo_once", ppy.releases_gil = true, ppy.symbol = "ppy_echo_echo_once"} loc("examples/37_aio/echo.ppy":13:0) {
```

*466 lines in all — [full output](outputs/03-ppy-emit-ir-echo-ppy.txt).*

<!-- outputs:end -->

## Read on

- [Coroutines](../../docs/guide/aio.md) — the awaitables, the runtime, and the guard rule.
- [The IR: the async dialect](../../docs/internals/ir.md) — `lower-async` and the frame.

`echo.ppy` is hand-written; there is no `.py` source and no conversion step.
