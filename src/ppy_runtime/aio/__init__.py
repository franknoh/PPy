"""The native async runtime's Python face: its library, its futures, and how they run (spec 77).

The runtime is one C file beside this module. It is compiled once, on
first use, with the C compiler the machine has, into the user's cache, and
loaded into the process; compiled code the JIT makes and a built artifact
both reach the same functions, so a future one side started the other can
drive. Without a compiler, or off Linux, there is no runtime: `available()`
says so, `reason()` says why, and every coroutine runs under asyncio.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any

__all__ = [
    "KIND",
    "NativeFuture",
    "NativeGuardFailed",
    "Runtime",
    "available",
    "compiler",
    "header_path",
    "library_path",
    "reason",
    "runtime",
    "runtime_for",
    "source_path",
]

KIND = "ppy.aio"
_KINDS = {"int": int, "float": float, "bool": bool, "none": lambda _bits: None}


class NativeGuardFailed(RuntimeError):
    """A guard failed inside a native coroutine, where nothing can fall back.

    A compiled coroutine that has already slept or spoken on a socket cannot
    hand the work back to its Python definition; the honest answer is this
    error, naming the function. `--safeguards off` or a body the prover
    clears keeps the guard out of the coroutine.
    """


def source_path() -> Path:
    return Path(__file__).with_name("ppy_aio.c")


def header_path() -> Path:
    return Path(__file__).with_name("ppy_aio.h")


def compiler() -> str | None:
    """The C compiler that builds the runtime: `CC`, else the first of cc, gcc, clang."""
    spelled = os.environ.get("CC")
    if spelled:
        return spelled
    for candidate in ("cc", "gcc", "clang"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def _cache_directory() -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root) / "ppy" / "aio"


_lock = threading.Lock()
_library: Path | bool | None = None
_reason = ""


def reason() -> str:
    """Why the runtime is unavailable here; empty when it is."""
    library_path()
    return _reason


def library_path() -> Path | None:
    """The compiled runtime, built into the cache on first use; None with the reason kept."""
    global _library, _reason  # noqa: PLW0603 - one runtime per process
    with _lock:
        if _library is not None:
            return _library or None
        _library = False
        if os.environ.get("PPY_NO_AIO"):
            _reason = "PPY_NO_AIO is set"
            return None
        if platform.system() != "Linux":
            _reason = f"the native async runtime runs on Linux (epoll); this is {platform.system()}"
            return None
        cc = compiler()
        if cc is None:
            _reason = "no C compiler on PATH to build the async runtime with (set CC)"
            return None
        source = source_path()
        digest = hashlib.sha256(source.read_bytes() + header_path().read_bytes()).hexdigest()[:16]
        directory = _cache_directory()
        target = directory / f"libppy_aio-{digest}.so"
        if not target.is_file():
            directory.mkdir(parents=True, exist_ok=True)
            draft = target.with_name(f"{target.name}.{os.getpid()}.part")
            command = [cc, "-std=c11", "-O2", "-shared", "-fPIC", "-o", str(draft), str(source)]
            done = subprocess.run(command, capture_output=True, text=True, check=False)
            if done.returncode != 0:
                _reason = f"the async runtime did not compile: {done.stderr.strip()[:400]}"
                draft.unlink(missing_ok=True)
                return None
            draft.replace(target)
        _library = target
        return target


def available() -> bool:
    return library_path() is not None


class Runtime:
    """`ppy_aio_*` through ctypes, on whichever library holds them."""

    def __init__(self, library: Any) -> None:
        self.lib = library
        i64, i32 = ctypes.c_int64, ctypes.c_int32
        library.ppy_aio_run.restype = i32
        library.ppy_aio_run.argtypes = [i64]
        library.ppy_aio_step.restype = i32
        library.ppy_aio_step.argtypes = [i64]
        library.ppy_aio_start.argtypes = [i64]
        library.ppy_aio_step_started.restype = i32
        library.ppy_aio_step_started.argtypes = [i64]
        library.ppy_aio_state.restype = i32
        library.ppy_aio_state.argtypes = [i64]
        library.ppy_aio_take.restype = i64
        library.ppy_aio_take.argtypes = [i64]
        library.ppy_aio_failure.restype = i64
        library.ppy_aio_failure.argtypes = [i64]
        library.ppy_aio_platform.restype = ctypes.c_char_p

    def platform(self) -> str:
        return self.lib.ppy_aio_platform().decode("ascii")

    def run(self, handle: int) -> int:
        return int(self.lib.ppy_aio_run(handle))

    def step(self, timeout_ms: int) -> int:
        return int(self.lib.ppy_aio_step(timeout_ms))

    def start(self, handle: int) -> None:
        self.lib.ppy_aio_start(handle)

    def step_started(self, handle: int) -> int:
        return int(self.lib.ppy_aio_step_started(handle))

    def state(self, handle: int) -> int:
        return int(self.lib.ppy_aio_state(handle))

    def take(self, handle: int) -> int:
        return int(self.lib.ppy_aio_take(handle))

    def failure(self, handle: int) -> int:
        return int(self.lib.ppy_aio_failure(handle))


_runtime: Runtime | None = None


def runtime() -> Runtime | None:
    """The process's runtime, on the library `library_path` built; None without one."""
    global _runtime  # noqa: PLW0603 - one runtime per process
    if _runtime is None:
        path = library_path()
        if path is None:
            return None
        _runtime = Runtime(ctypes.CDLL(str(path)))
    return _runtime


def runtime_for(owner: Any) -> Runtime | None:
    """The runtime a native function's own library carries, else the process's."""
    if owner is not None and hasattr(owner, "ppy_aio_run"):
        return Runtime(owner)
    return runtime()


def _bits_to(kind: str, bits: int) -> Any:
    if kind == "float":
        return ctypes.c_double.from_buffer_copy(ctypes.c_int64(bits)).value
    if kind == "bool":
        return bool(bits & 0xFF)
    if kind == "none":
        return None
    return bits


class NativeFuture:
    """What a natively compiled `async def` hands back when called: a future the runtime owns.

    `ppy.aio.run` drives the loop until it completes; awaited from asyncio,
    it steps the native loop between the Python loop's turns. Its value is
    read once.
    """

    __slots__ = ("_kind", "_result", "_taken", "function", "handle", "runtime")

    def __init__(self, handle: int, kind: str, runtime_: Runtime, function: str = "") -> None:
        self.handle = handle
        self.runtime = runtime_
        self.function = function
        self._kind = kind
        self._taken = False
        self._result: Any = None

    def __repr__(self) -> str:
        return f"aio.NativeFuture({self.function or self.handle})"

    @property
    def done(self) -> bool:
        return self._taken or self.runtime.state(self.handle) != 0

    def _finish(self) -> Any:
        if self._taken:
            return self._result
        state = self.runtime.state(self.handle)
        if state == 2:
            code = self.runtime.take(self.handle)
            self._taken = True
            raise NativeGuardFailed(
                f"a guard failed inside the native coroutine `{self.function}` (site {code}); "
                "a coroutine cannot fall back to Python once it has run"
            )
        self._result = _bits_to(self._kind, self.runtime.take(self.handle))
        self._taken = True
        return self._result

    def start(self) -> None:
        """Run this coroutine from the loop's next turn on, without waiting for it."""
        if not self._taken:
            self.runtime.start(self.handle)

    def result(self) -> Any:
        """Run the loop until this future completes, and its value."""
        if not self._taken:
            status = self.runtime.run(self.handle)
            if status == 2:
                raise RuntimeError(
                    f"the native coroutine `{self.function}` waits for something nothing will "
                    "complete"
                )
        return self._finish()

    def __await__(self) -> Generator[Any, None, Any]:
        import asyncio

        while not self.done:
            if self.runtime.step_started(self.handle) == 0 and not self.done:
                raise RuntimeError(
                    f"the native coroutine `{self.function}` waits for something nothing will "
                    "complete"
                )
            yield from asyncio.sleep(0).__await__()
        return self._finish()
