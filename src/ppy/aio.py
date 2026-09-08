"""`ppy.aio`: coroutines that sleep and speak on sockets, the same on every path.

```python
from ppy import aio, native


async def serve_one(listening: int) -> int:
    client = await aio.accept(listening)
    room = native.stack_alloc[ppy.u8](64)
    got = await aio.read(client, room, 64)
    sent = await aio.write(client, room, got)
    aio.close(client)
    return sent


aio.run(serve_one(aio.listen("127.0.0.1", 8000)))
```

`sleep`, `accept`, `connect`, `read`, and `write` are awaitables; `spawn`
starts a coroutine as a task to await later; `listen`, `port`, and `close`
are immediate. A socket is an `int`, and every socket
operation answers a negative errno rather than raising, so a compiled
coroutine and this one say the same thing. Under CPython the awaitables are
asyncio's, and `run` is `asyncio.run`; the compiler lowers an `async def`
whose awaits are these and other native coroutines to the async dialect
(spec 76), the native runtime drives it (spec 77), and `run` given what a
compiled coroutine hands back drives that. An await the compiler does not
know keeps the coroutine in Python (spec 78); `compiled(f)` says which
happened.
"""

from __future__ import annotations

import asyncio
import errno as _errno
import socket as _socket
from collections.abc import Awaitable, Callable
from typing import Any

from ppy_runtime.aio import NativeFuture, NativeGuardFailed

from ._native_api import Pointer, _layout

__all__ = [
    "NativeFuture",
    "NativeGuardFailed",
    "accept",
    "close",
    "compiled",
    "connect",
    "listen",
    "port",
    "read",
    "run",
    "sleep",
    "spawn",
    "write",
]

_sockets: dict[int, _socket.socket] = {}


def _register(sock: _socket.socket) -> int:
    sock.setblocking(False)
    _sockets[sock.fileno()] = sock
    return sock.fileno()


def _socket_of(fd: int) -> _socket.socket | None:
    return _sockets.get(fd)


def _negative(error: OSError) -> int:
    """The negative errno a failed socket operation answers with."""
    if isinstance(error, _socket.gaierror):
        return -_errno.EHOSTUNREACH
    return -(error.errno or _errno.EIO)


def _address(host: str, port_number: int, passive: bool) -> tuple[int, Any]:
    flags = _socket.AI_PASSIVE if passive else 0
    found = _socket.getaddrinfo(
        host or None, port_number, _socket.AF_UNSPEC, _socket.SOCK_STREAM, 0, flags
    )
    family, _type, _proto, _name, address = found[0]
    return family, address


async def sleep(seconds: float) -> None:
    """Wait `seconds`; other coroutines run meanwhile."""
    await asyncio.sleep(seconds)


def listen(host: str, port_number: int, backlog: int = 16) -> int:
    """A listening TCP socket bound to `host:port`, or a negative errno."""
    try:
        family, address = _address(host, port_number, passive=True)
        sock = _socket.socket(family, _socket.SOCK_STREAM)
        sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        sock.bind(address)
        sock.listen(backlog if backlog > 0 else 16)
    except OSError as error:
        return _negative(error)
    return _register(sock)


def port(fd: int) -> int:
    """The port a socket is bound to, or a negative errno."""
    sock = _socket_of(fd)
    if sock is None:
        return -_errno.EBADF
    try:
        return int(sock.getsockname()[1])
    except OSError as error:
        return _negative(error)


async def accept(fd: int) -> int:
    """The next connection on a listening socket, or a negative errno."""
    sock = _socket_of(fd)
    if sock is None:
        return -_errno.EBADF
    try:
        connection, _peer = await asyncio.get_running_loop().sock_accept(sock)
    except OSError as error:
        return _negative(error)
    return _register(connection)


async def connect(host: str, port_number: int) -> int:
    """A socket connected to `host:port`, or a negative errno."""
    try:
        family, address = _address(host, port_number, passive=False)
        sock = _socket.socket(family, _socket.SOCK_STREAM)
        sock.setblocking(False)
        await asyncio.get_running_loop().sock_connect(sock, address)
    except OSError as error:
        return _negative(error)
    return _register(sock)


def _view(buffer: Pointer[Any], count: int) -> memoryview:
    _code, width = _layout(buffer.element)
    start = buffer.index * width
    return memoryview(buffer.memory).cast("B")[start : start + count]


async def read(fd: int, buffer: Pointer[Any], count: int) -> int:
    """Up to `count` bytes into `buffer`: how many came, 0 at the end, negative errno on failure."""
    sock = _socket_of(fd)
    if sock is None:
        return -_errno.EBADF
    try:
        return await asyncio.get_running_loop().sock_recv_into(sock, _view(buffer, count))
    except OSError as error:
        return _negative(error)


async def write(fd: int, buffer: Pointer[Any], count: int) -> int:
    """All `count` bytes of `buffer` to the socket: `count`, or a negative errno."""
    sock = _socket_of(fd)
    if sock is None:
        return -_errno.EBADF
    try:
        await asyncio.get_running_loop().sock_sendall(sock, bytes(_view(buffer, count)))
    except OSError as error:
        return _negative(error)
    return count


def close(fd: int) -> None:
    sock = _sockets.pop(fd, None)
    if sock is not None:
        sock.close()


def spawn(awaitable: Awaitable[Any]) -> Awaitable[Any]:
    """Start `awaitable` now, as a task the loop runs alongside; await it for its value."""
    if isinstance(awaitable, NativeFuture):
        awaitable.start()
        return awaitable
    return asyncio.ensure_future(awaitable)  # type: ignore[arg-type]


def run(awaitable: Awaitable[Any]) -> Any:
    """`awaitable`'s value: the native loop for a compiled coroutine, asyncio otherwise."""
    if isinstance(awaitable, NativeFuture):
        return awaitable.result()
    return asyncio.run(awaitable)  # type: ignore[arg-type]


def compiled(function: Callable[..., Any]) -> bool:
    """Whether calling `function` starts a native coroutine here."""
    signature = getattr(function, "__ppy_native__", None)
    return bool(getattr(signature, "future", ""))
