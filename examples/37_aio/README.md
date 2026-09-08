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
^entry:
    %frame = async.frame_new {slots = 17} : ptr<i64>
    %0 = core.const 3 : i64
    %1 = core.ptr_offset %frame, %0 : ptr<i64>
    core.store %listening, %1
    %future = async.spawn %frame {callee = @echo_echo_once_resume} : future<i64>
    core.ret %future
}

func @echo_send_hello(%port: i64) -> future<i64> attrs {effects = ["alloc", "network", "read_memory", "sync", "write_memory"], ppy.abi = "ppy", ppy.async = true, ppy.async.lowered = true, ppy.qualname = "echo.send_hello", ppy.releases_gil = true, ppy.symbol = "ppy_echo_send_hello"} loc("examples/37_aio/echo.ppy":22:0) {
^entry:
    %frame = async.frame_new {slots = 20} : ptr<i64>
    %0 = core.const 3 : i64
    %1 = core.ptr_offset %frame, %0 : ptr<i64>
    core.store %port, %1
    %future = async.spawn %frame {callee = @echo_send_hello_resume} : future<i64>
    core.ret %future
}

func @echo_main() -> future<i64> attrs {effects = ["alloc", "may_raise", "network", "read_memory", "sync", "time", "write_memory"], ppy.abi = "ppy", ppy.async = true, ppy.async.lowered = true, ppy.qualname = "echo.main", ppy.releases_gil = true, ppy.symbol = "ppy_echo_main"} loc("examples/37_aio/echo.ppy":35:0) {
… 426 more lines
```

<!-- outputs:end -->
