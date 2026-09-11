"""Launching a compiled kernel: the CUDA driver through ctypes, no toolkit needed (spec 75).

The build stages a kernel as PTX with the kinds of its parameters; a
`Kernel` loads it into the device's primary context once and launches it.
Scalars are passed by value; a `native` pointer's whole array goes to the
device before the launch and comes back after it when the pointer is
mutable, so a launch means exactly what the reference launch means, only on
the device. Memory made by `cuda.device_alloc` already lives there: its
address is passed and nothing is copied. A machine without the driver, or
without a device, has no kernel to bind: the reference runs instead, and
the binding says why.
"""

from __future__ import annotations

import ctypes
import json
import os
import threading
from typing import Any

__all__ = ["KIND", "CudaError", "Driver", "Kernel", "available", "driver", "kernel_binding"]

#: The payload's `kind`, distinguishing it from an XLA or a JAX export.
KIND = "ppy.cuda"
_LIBRARIES = (
    "libcuda.so.1",
    "libcuda.so",
    "/usr/lib/wsl/lib/libcuda.so.1",
    "/usr/local/cuda/compat/libcuda.so.1",
    "nvcuda.dll",
)


class CudaError(RuntimeError):
    """The driver refused: the message carries its error string."""


class Driver:
    """The CUDA driver, loaded once, with the first device's primary context."""

    def __init__(self, library: Any) -> None:
        self.lib = library
        self.lock = threading.RLock()
        self._declare()
        self.check(library.cuInit(0), "cuInit")
        count = ctypes.c_int(0)
        self.check(library.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
        if count.value == 0:
            raise CudaError("no CUDA device is present")
        self.device = ctypes.c_int(0)
        self.check(library.cuDeviceGet(ctypes.byref(self.device), 0), "cuDeviceGet")
        self.context = ctypes.c_void_p(0)
        self.check(
            library.cuDevicePrimaryCtxRetain(ctypes.byref(self.context), self.device),
            "cuDevicePrimaryCtxRetain",
        )

    def _declare(self) -> None:
        lib = self.lib
        size, address = ctypes.c_size_t, ctypes.c_uint64
        lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(address), size]
        lib.cuMemFree_v2.argtypes = [address]
        lib.cuMemcpyHtoD_v2.argtypes = [address, ctypes.c_void_p, size]
        lib.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, address, size]
        lib.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        lib.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        lib.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            *([ctypes.c_uint] * 6),
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]

    def check(self, status: int, what: str) -> None:
        if status != 0:
            message = ctypes.c_char_p(None)
            self.lib.cuGetErrorString(status, ctypes.byref(message))
            raw = message.value
            spelled = raw.decode("utf-8", "replace") if raw is not None else str(status)
            raise CudaError(f"{what}: {spelled}")

    def current(self) -> None:
        self.check(self.lib.cuCtxSetCurrent(self.context), "cuCtxSetCurrent")

    def name(self) -> str:
        buffer = ctypes.create_string_buffer(256)
        self.check(self.lib.cuDeviceGetName(buffer, 256, self.device), "cuDeviceGetName")
        return buffer.value.decode("utf-8", "replace")

    def load(self, ptx: str) -> ctypes.c_void_p:
        module = ctypes.c_void_p(0)
        self.check(
            self.lib.cuModuleLoadData(ctypes.byref(module), ptx.encode("utf-8")), "cuModuleLoadData"
        )
        return module

    def function(self, module: ctypes.c_void_p, symbol: str) -> ctypes.c_void_p:
        function = ctypes.c_void_p(0)
        self.check(
            self.lib.cuModuleGetFunction(ctypes.byref(function), module, symbol.encode("utf-8")),
            f"cuModuleGetFunction({symbol})",
        )
        return function

    def alloc(self, nbytes: int) -> ctypes.c_uint64:
        pointer = ctypes.c_uint64(0)
        self.check(self.lib.cuMemAlloc_v2(ctypes.byref(pointer), max(nbytes, 1)), "cuMemAlloc")
        return pointer

    def free(self, pointer: ctypes.c_uint64) -> None:
        self.check(self.lib.cuMemFree_v2(pointer), "cuMemFree")

    def upload(self, pointer: ctypes.c_uint64, buffer: Any, nbytes: int) -> None:
        self.check(self.lib.cuMemcpyHtoD_v2(pointer, buffer, nbytes), "cuMemcpyHtoD")

    def download(self, buffer: Any, pointer: ctypes.c_uint64, nbytes: int) -> None:
        self.check(self.lib.cuMemcpyDtoH_v2(buffer, pointer, nbytes), "cuMemcpyDtoH")

    def launch(
        self,
        function: ctypes.c_void_p,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        parameters: Any,
    ) -> None:
        self.check(
            self.lib.cuLaunchKernel(function, *grid, *block, 0, None, parameters, None),
            "cuLaunchKernel",
        )
        self.check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")


_driver: Driver | bool | None = None
_driver_lock = threading.Lock()


def driver() -> Driver | None:
    """The driver, loaded on first use; None where there is none, or `PPY_NO_CUDA` is set."""
    global _driver  # noqa: PLW0603 - one driver per process
    with _driver_lock:
        if _driver is None:
            _driver = False
            if not os.environ.get("PPY_NO_CUDA"):
                for name in _LIBRARIES:
                    try:
                        library = ctypes.CDLL(name)
                    except OSError:
                        continue
                    try:
                        _driver = Driver(library)
                    except (CudaError, AttributeError):
                        _driver = False
                    break
        return _driver or None


def available() -> bool:
    return driver() is not None


class Kernel:
    """One compiled kernel: its PTX, its symbol, and the kinds of its parameters."""

    def __init__(
        self,
        function: str,
        symbol: str,
        ptx: str,
        params: list[dict[str, Any]],
        threads: int = 0,
    ) -> None:
        self.function = function
        self.symbol = symbol
        self.ptx = ptx
        self.params = params
        #: A tile kernel's block: the threads that share one program's tiles.
        self.threads = threads
        self._handle: ctypes.c_void_p | None = None

    def _loaded(self, d: Driver) -> ctypes.c_void_p:
        if self._handle is None:
            self._handle = d.function(d.load(self.ptx), self.symbol)
        return self._handle

    def launch(
        self, grid: tuple[int, int, int], block: tuple[int, int, int], arguments: tuple[Any, ...]
    ) -> None:
        """Run the kernel over `grid` blocks of `block` threads with `arguments`, and wait."""
        d = driver()
        if d is None:
            raise CudaError("no CUDA driver")
        if len(arguments) != len(self.params):
            raise TypeError(
                f"{self.function} takes {len(self.params)} argument(s), {len(arguments)} given"
            )
        with d.lock:
            d.current()
            handle = self._loaded(d)
            holders: list[Any] = []
            copies: list[tuple[ctypes.c_uint64, Any, int, bool]] = []
            written: list[Any] = []
            try:
                for value, described in zip(arguments, self.params, strict=True):
                    kind = described["kind"]
                    if kind == "int":
                        holders.append(ctypes.c_int64(int(value)))
                    elif kind == "float":
                        holders.append(ctypes.c_double(float(value)))
                    elif kind == "bool":
                        holders.append(ctypes.c_int8(1 if value else 0))
                    elif kind in {"ptr", "const_ptr"} and getattr(value, "home", None) is not None:
                        # Memory that lives on the device: its address, no copy.
                        home = value.home
                        base = home.device_address()
                        if base is None:
                            raise CudaError("device memory was made without a device")
                        holders.append(ctypes.c_uint64(base + value.index * home.width))
                        if kind == "ptr" and bool(getattr(value, "mutable", True)):
                            written.append(home)
                    elif kind in {"ptr", "const_ptr"}:
                        memory = value.memory
                        nbytes = len(memory) * memory.itemsize
                        pointer = d.alloc(nbytes)
                        writable = kind == "ptr" and bool(getattr(value, "mutable", True))
                        copies.append((pointer, memory, nbytes, writable))
                        d.upload(pointer, _host_address(memory, nbytes), nbytes)
                        holders.append(
                            ctypes.c_uint64(pointer.value + value.index * memory.itemsize)
                        )
                    else:
                        raise TypeError(f"{self.function}: no device form for a `{kind}` argument")
                parameters = (ctypes.c_void_p * len(holders))(
                    *(ctypes.addressof(h) for h in holders)
                )
                d.launch(handle, grid, block, parameters)
                for home in written:
                    home.device_written()
                for pointer, memory, nbytes, writable in copies:
                    if writable:
                        d.download((ctypes.c_char * nbytes).from_buffer(memory), pointer, nbytes)
            finally:
                for pointer, _memory, _nbytes, _writable in copies:
                    d.free(pointer)


def _host_address(memory: memoryview, nbytes: int) -> Any:
    """What the driver reads a host array from: the array itself, not a copy of it.

    A copy into a ctypes buffer costs more than the transfer -- 90 ms against
    12 for 128 MB here -- so a writable buffer hands over its own address.
    Only a read-only view, which ctypes cannot address in place, is copied.
    """
    try:
        return ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(memory)))
    except TypeError:
        return (ctypes.c_char * nbytes).from_buffer_copy(memory)


def kernel_binding(function: str, payload: bytes, fallback: Any):  # type: ignore[no-untyped-def]
    """The staged kernel behind `function`, for `ppy.cuda.launch` to find.

    Called directly, a kernel runs its own definition, as it would under
    CPython; launched, it runs on the device. Without a driver the binding
    is the fallback and says why.
    """
    from .exported import ExportedBinding

    binding = ExportedBinding(function, fallback, fallback)
    if driver() is None:
        binding.reason = "no CUDA driver or device; the reference launch runs"
        return binding
    described = json.loads(payload.decode("utf-8"))
    if described.get("kind") != KIND:
        raise ValueError("not a ppy.cuda payload")
    kernel = Kernel(
        described["function"],
        described["symbol"],
        described["ptx"],
        described["params"],
        int(described.get("threads", 0)),
    )

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return fallback(*args, **kwargs)

    wrapper.__name__ = function
    wrapper.__ppy_exported__ = True  # type: ignore[attr-defined]
    wrapper.__ppy_fallback__ = fallback  # type: ignore[attr-defined]
    wrapper.__ppy_kernel__ = kernel  # type: ignore[attr-defined]
    wrapper.__ppy_binding__ = binding  # type: ignore[attr-defined]
    binding.wrapper = wrapper
    return binding
