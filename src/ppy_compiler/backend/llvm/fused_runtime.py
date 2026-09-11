"""Guarded dispatch for fused kernels (spec 19.3, 19.5, 19.10, 19.11).

One kernel serves every library whose arrays it was built over: a NumPy
`ndarray` and a CPU torch `Tensor` are both `float64` storage behind a
pointer, so the same loop runs over either once the guards hold. A PyArrow
`Array` is Arrow's layout -- a values buffer and a validity bitmap -- and
its kernel is columnar IR over exactly that, so the array is read where it
lies and the result is an Arrow array built over the buffers the kernel
filled; a pandas `Series` is that Arrow array when Arrow-backed, or its
NumPy values behind a bitmap of ones, and comes back as a Series over its
own index. What the guards are, how a pointer is taken, and what the result
is wrapped in are the storage's business; the kernel's is the arithmetic.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .fusion import FusedLoop
from .parallel import chunk_bounds, offset, pool

__all__ = ["FusedBinding", "bind_fused"]

_DOUBLE_SIZE = ctypes.sizeof(ctypes.c_double)
_STATUS_OK = 0

_SAFE_ERROR_STATES = frozenset({"ignore"})


@dataclass(slots=True)
class FusedBinding:
    loop: FusedLoop
    wrapper: Callable[..., object]
    fallback: Callable[..., object]
    calls: int = 0
    fallbacks: int = 0
    parallel_calls: int = 0
    reason: str = ""


class _Storage:
    """How one library's arrays are checked, read, and made."""

    name = ""

    def accepts(self, value: object) -> bool:
        raise NotImplementedError

    def shape(self, value: Any) -> tuple[int, ...]:
        return tuple(value.shape)

    def size(self, value: Any) -> int:
        raise NotImplementedError

    def pointer(self, value: Any, kind: Any) -> Any:
        raise NotImplementedError

    def empty(self, shape: tuple[int, ...]) -> Any:
        raise NotImplementedError

    def scalar(self, value: float) -> object:
        return value

    def strict(self) -> bool:
        """Must a result be checked for a floating-point condition?"""
        return False

    def finite(self, result: Any, arrays: tuple[Any, ...]) -> bool:
        return True

    def finite_slice(self, result: Any, arrays: tuple[Any, ...], start: int, stop: int) -> bool:
        return self.finite(result, arrays)


class _NumPy(_Storage):
    name = "numpy"

    def __init__(self, numpy: Any) -> None:
        self.numpy = numpy
        self.float64 = numpy.dtype("float64")

    def accepts(self, value: object) -> bool:
        # An exact ndarray only: a subclass may override dispatch entirely.
        if type(value) is not self.numpy.ndarray:
            return False
        if value.dtype != self.float64 or not value.dtype.isnative:  # type: ignore[attr-defined]
            return False
        return bool(value.flags["C_CONTIGUOUS"])  # type: ignore[attr-defined]

    def size(self, value: Any) -> int:
        return int(value.size)

    def pointer(self, value: Any, kind: Any) -> Any:
        return ctypes.cast(value.ctypes.data, kind)

    def empty(self, shape: tuple[int, ...]) -> Any:
        return self.numpy.empty(shape, dtype=self.float64)

    def strict(self) -> bool:
        """A generated loop raises no NumPy warning of its own, so unless every
        error category is ignored the result must be checked before it is
        trusted (spec 19.11)."""
        state = self.numpy.geterr()
        return not all(value in _SAFE_ERROR_STATES for value in state.values())

    def finite(self, result: Any, arrays: tuple[Any, ...]) -> bool:
        """Confirm the kernel raised no floating-point condition NumPy would
        report. Divide-by-zero, overflow, and invalid operations each leave a
        non-finite value where they happened, and it propagates to the
        result, so the result finite means none occurred; otherwise the
        Python path re-runs and reports exactly what NumPy would. The inputs
        need no pass of their own: a non-finite input that NumPy would warn
        about (`inf * 0`, `inf - inf`) is a non-finite result, and one it
        would not warn about is not the kernel's to report."""
        del arrays
        return bool(self.numpy.isfinite(result).all())

    def finite_slice(self, result: Any, arrays: tuple[Any, ...], start: int, stop: int) -> bool:
        """The same confirmation over one chunk, `[start, stop)` of the result
        flattened: the worker that just wrote the chunk checks it while it is
        still in its cache, so the pass costs a fraction of a serial one."""
        del arrays
        if result is None:
            return True
        return bool(self.numpy.isfinite(result.reshape(-1)[start:stop]).all())


class _Torch(_Storage):
    name = "torch"

    def __init__(self, torch: Any) -> None:
        self.torch = torch

    def accepts(self, value: object) -> bool:
        # An exact Tensor on the CPU, float64, contiguous, and outside
        # autograd: a tensor that records its history needs the dispatcher.
        torch = self.torch
        if type(value) is not torch.Tensor:
            return False
        if value.dtype is not torch.float64 or value.device.type != "cpu":  # type: ignore[attr-defined]
            return False
        if value.requires_grad:  # type: ignore[attr-defined]
            return False
        return bool(value.is_contiguous())  # type: ignore[attr-defined]

    def size(self, value: Any) -> int:
        return int(value.numel())

    def pointer(self, value: Any, kind: Any) -> Any:
        return ctypes.cast(value.data_ptr(), kind)

    def empty(self, shape: tuple[int, ...]) -> Any:
        return self.torch.empty(shape, dtype=self.torch.float64)

    def scalar(self, value: float) -> object:
        # A torch reduction is a tensor of rank 0, as the library answers.
        return self.torch.tensor(value, dtype=self.torch.float64)


def _storage(name: str) -> _Storage | None:
    try:
        if name == "torch":
            import torch

            return _Torch(torch)
        import numpy

        return _NumPy(numpy)
    except ImportError:
        return None


def bind_fused(
    loop: FusedLoop,
    address: int,
    fallback: Callable[..., object],
    *,
    parallel: bool = False,
    threads: str | int = "auto",
    nan_address: int = 0,
) -> FusedBinding:
    """Wrap one fused kernel in the guards its fast-path domain requires.

    A columnar kernel has a twin at `nan_address` for NumPy-backed pandas
    Series, whose nulls are NaNs and whose masks are bytes; without one,
    such Series run pandas.
    """
    if loop.storage in {"pyarrow", "pandas"}:
        return _bind_columnar(loop, address, fallback, nan_address)
    storage = _storage(loop.storage)
    if storage is None:
        return FusedBinding(loop, fallback, fallback, reason=f"{loop.storage} is not importable")

    double = ctypes.c_double
    pointer = ctypes.POINTER(double)
    array_count = len(loop.arrays)
    scalar_count = len(loop.scalars)

    # The kernel is an IR function: an `i32` status back, its result -- or a
    # placeholder, for a map -- through a pointer after the arguments.
    if loop.returns_scalar:
        prototype = ctypes.CFUNCTYPE(
            ctypes.c_int32,
            *([pointer] * array_count),
            *([double] * scalar_count),
            ctypes.c_int64,
            pointer,
        )
    else:
        prototype = ctypes.CFUNCTYPE(
            ctypes.c_int32,
            pointer,
            *([pointer] * array_count),
            *([double] * scalar_count),
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int64),
        )
    native = prototype(address)

    binding = FusedBinding(loop, lambda *a: None, fallback)

    splittable = parallel and _splittable(loop)
    workers = pool(threads) if parallel else None

    def reduce_chunk(pointers: list, widened: list[float], start: int, stop: int) -> float:
        result = double(0.0)
        status = native(
            *[offset(p, _DOUBLE_SIZE, start) for p in pointers], *widened, stop - start, result
        )
        if status != _STATUS_OK:
            raise _Failed
        return result.value

    def map_chunk(out: Any, pointers: list, widened: list[float], start: int, stop: int) -> None:
        placeholder = ctypes.c_int64(0)
        status = native(
            offset(out, _DOUBLE_SIZE, start),
            *[offset(p, _DOUBLE_SIZE, start) for p in pointers],
            *widened,
            stop - start,
            placeholder,
        )
        if status != _STATUS_OK:
            raise _Failed

    def wrapper(*args: object) -> object:
        if len(args) != array_count + scalar_count:
            return fallback(*args)
        arrays = args[:array_count]
        scalars = args[array_count:]

        shape = None
        for value in arrays:
            if not storage.accepts(value):
                binding.fallbacks += 1
                return fallback(*args)
            if shape is None:
                shape = storage.shape(value)
            elif storage.shape(value) != shape:
                # Array-to-array broadcasting is left to the library.
                binding.fallbacks += 1
                return fallback(*args)
        if shape is None:
            binding.fallbacks += 1
            return fallback(*args)

        for value in scalars:
            if type(value) not in (int, float):
                binding.fallbacks += 1
                return fallback(*args)

        if loop.reduction in {"max", "min"} and storage.size(arrays[0]) == 0:
            # An empty min/max has no identity; the library raises, so let it.
            binding.fallbacks += 1
            return fallback(*args)

        pointers = [storage.pointer(a, pointer) for a in arrays]
        widened = [float(s) for s in scalars]  # type: ignore[arg-type]
        length = storage.size(arrays[0])
        strict = storage.strict()

        bounds = (
            chunk_bounds(length, workers.threads)
            if workers is not None and splittable
            else [(0, length)]
        )
        parallelized = len(bounds) > 1

        try:
            if loop.returns_scalar:
                if parallelized:
                    partials = workers.map_chunks(  # type: ignore[union-attr]
                        lambda start, stop: reduce_chunk(pointers, widened, start, stop), length
                    )
                    result = _combine(loop.reduction, [float(p) for p in partials])
                    finite = not strict or storage.finite(result, arrays)
                else:
                    result = reduce_chunk(pointers, widened, 0, length)
                    finite = not strict or storage.finite(result, arrays)
                if strict and not finite:
                    binding.fallbacks += 1
                    return fallback(*args)
                binding.calls += 1
                binding.parallel_calls += int(parallelized)
                return storage.scalar(result)

            out = storage.empty(shape)
            out_pointer = storage.pointer(out, pointer)
            # A guarded kernel refuses a non-finite result itself, by its status;
            # any other is checked here, a chunk at a time where the loop split.
            check = strict and not loop.guarded
            if parallelized:

                def map_and_check(start: int, stop: int) -> bool:
                    map_chunk(out_pointer, pointers, widened, start, stop)
                    return not check or storage.finite_slice(out, arrays, start, stop)

                finite = all(workers.map_chunks(map_and_check, length))  # type: ignore[union-attr]
            else:
                map_chunk(out_pointer, pointers, widened, 0, length)
                finite = not check or storage.finite(out, arrays)
        except _Failed:
            binding.fallbacks += 1
            return fallback(*args)
        if strict and not finite:
            binding.fallbacks += 1
            return fallback(*args)
        binding.calls += 1
        binding.parallel_calls += int(parallelized)
        return out

    wrapper.__name__ = loop.symbol
    wrapper.__ppy_fused__ = loop  # type: ignore[attr-defined]
    binding.wrapper = wrapper
    return binding


class _Failed(Exception):
    """The kernel's own guard failed: the library's path answers."""


# -- Arrow -----------------------------------------------------------------------------------------


def _bind_columnar(
    loop: FusedLoop, address: int, fallback: Callable[..., object], nan_address: int = 0
) -> FusedBinding:
    """A columnar kernel over PyArrow arrays or pandas Series, read in place.

    The kernel takes the result's values and validity buffers, then each
    array's values and validity buffers, the scalars, and the row count. A
    PyArrow array is its buffers, and so is an Arrow-backed pandas Series;
    the answer is an Arrow array, or a Series over the same index. A
    NumPy-backed Series -- `float64` values whose nulls are NaNs, or a
    `bool` mask -- goes to the kernel's NaN-model twin, which reads and
    writes that layout directly; the answer is a Series over a fresh NumPy
    array, with no copy and no bitmap in between. An array without nulls
    lends a bitmap of ones. A bit-packed buffer sliced inside a byte, a
    chunked array, another type, Series whose indexes are not one index,
    or a shape the kernel does not take run the library's own compute.
    """
    try:
        import pyarrow
    except ImportError:
        return FusedBinding(loop, fallback, fallback, reason="pyarrow is not importable")
    pandas: Any = None
    numpy: Any = None
    if loop.storage == "pandas":
        try:
            import numpy
            import pandas
        except ImportError:
            return FusedBinding(loop, fallback, fallback, reason="pandas is not importable")

    byte_pointer = ctypes.POINTER(ctypes.c_uint8)
    array_count = len(loop.arrays)
    kinds = loop.kinds or ("f64",) * array_count
    prototype = ctypes.CFUNCTYPE(
        ctypes.c_int32,
        ctypes.c_void_p,
        byte_pointer,
        *([ctypes.c_void_p, byte_pointer] * array_count),
        *([ctypes.c_double] * len(loop.scalars)),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64),
    )
    native = prototype(address)
    nan_native = prototype(nan_address) if nan_address else None
    binding = FusedBinding(loop, lambda *a: None, fallback)
    ones = bytearray()
    types = {"f64": pyarrow.float64(), "bool": pyarrow.bool_()}
    numpy_kinds = {"f64": "float64", "bool": "bool"}

    def all_valid(length: int) -> Any:
        nonlocal ones
        needed = (length + 7) // 8
        if len(ones) < needed:
            ones = bytearray(b"\xff" * max(needed, 64))
        return (ctypes.c_uint8 * len(ones)).from_buffer(ones)

    def arrow_parts(value: Any, kind: str) -> tuple[int, Any] | None:
        """(values address, validity pointer) of an array the kernel takes, else None."""
        if not isinstance(value, pyarrow.Array) or value.type != types[kind]:
            return None
        validity, values = value.buffers()[:2]
        if values is None:
            return None
        if kind == "f64":
            address_ = values.address + value.offset * _DOUBLE_SIZE
        else:
            if value.offset % 8:
                return None
            address_ = values.address + value.offset // 8
        if validity is None or value.null_count == 0:
            bitmap = all_valid(len(value))
        elif value.offset % 8:
            return None
        else:
            bitmap = ctypes.cast(validity.address + value.offset // 8, byte_pointer)
        return address_, bitmap

    def series_parts(value: Any, kind: str, keep: list) -> tuple[int, Any, str] | None:
        """A Series' (values address, validity pointer, backing); None where the kernel stops."""
        if not isinstance(value, pandas.Series):
            return None
        storage = value.array
        if isinstance(storage, pandas.arrays.ArrowExtensionArray):
            chunked = storage.__arrow_array__()
            if chunked.num_chunks != 1:
                return None
            described = arrow_parts(chunked.chunk(0), kind)
            return None if described is None else (*described, "arrow")
        if nan_native is None or not isinstance(storage, pandas.arrays.NumpyExtensionArray):
            return None
        values = value.to_numpy()
        if str(values.dtype) != numpy_kinds[kind] or not values.flags["C_CONTIGUOUS"]:
            return None
        keep.append(values)
        # The NaN-model kernel reads no validity; the bitmap only fills the argument.
        return values.ctypes.data, all_valid(len(values)), "numpy"

    def one_index(series: list) -> Any:
        """The index every Series shares, or None: alignment is pandas' own business."""
        first = series[0].index
        for other in series[1:]:
            index = other.index
            if index is first:
                continue
            ranges = isinstance(first, pandas.RangeIndex) and isinstance(index, pandas.RangeIndex)
            if ranges and first.equals(index):
                continue
            return None
        return first

    def wrapper(*args: object) -> object:
        if len(args) != array_count + len(loop.scalars):
            return fallback(*args)
        arrays = args[:array_count]
        scalars = args[array_count:]
        if any(type(s) not in (int, float) for s in scalars):
            binding.fallbacks += 1
            return fallback(*args)
        length: int | None = None
        atoms: list = []
        backings: set[str] = set()
        keep: list = []
        for value, kind in zip(arrays, kinds, strict=True):
            if pandas is not None:
                described = series_parts(value, kind, keep)
                if described is not None:
                    backings.add(described[2])
                    described = described[:2]
            else:
                described = arrow_parts(value, kind)
            if described is None or (length is not None and len(value) != length):  # type: ignore[arg-type]
                binding.fallbacks += 1
                return fallback(*args)
            length = len(value)  # type: ignore[arg-type]
            atoms.extend(described)
        if length is None or len(backings) > 1:
            binding.fallbacks += 1
            return fallback(*args)
        index = None
        if pandas is not None:
            index = one_index(list(arrays))
            if index is None:
                binding.fallbacks += 1
                return fallback(*args)
        bitmap_bytes = (length + 7) // 8
        placeholder = ctypes.c_int64(0)
        widened = [float(s) for s in scalars]  # type: ignore[arg-type]
        if backings == {"numpy"}:
            assert nan_native is not None
            # NumPy's layout in and out: the answer is the array the Series wraps.
            answer = numpy.empty(length, dtype=numpy_kinds[loop.result])
            status = nan_native(
                answer.ctypes.data, all_valid(length), *atoms, *widened, length, placeholder
            )
            if status != _STATUS_OK:
                binding.fallbacks += 1
                return fallback(*args)
            binding.calls += 1
            names = {series.name for series in arrays}  # type: ignore[attr-defined]
            return pandas.Series(answer, index=index, name=names.pop() if len(names) == 1 else None)
        values_bytes = length * _DOUBLE_SIZE if loop.result == "f64" else bitmap_bytes
        out_values = pyarrow.allocate_buffer(values_bytes, resizable=False)
        out_validity = pyarrow.allocate_buffer(bitmap_bytes, resizable=False)
        status = native(
            out_values.address,
            ctypes.cast(out_validity.address, byte_pointer),
            *atoms,
            *widened,
            length,
            placeholder,
        )
        if status != _STATUS_OK:
            binding.fallbacks += 1
            return fallback(*args)
        binding.calls += 1
        validity_buffer = out_validity if loop.nullable else None
        result = pyarrow.Array.from_buffers(
            types[loop.result], length, [validity_buffer, out_values]
        )
        if pandas is None:
            return result
        names = {series.name for series in arrays}  # type: ignore[attr-defined]
        name = names.pop() if len(names) == 1 else None
        return pandas.Series(pandas.arrays.ArrowExtensionArray(result), index=index, name=name)

    wrapper.__name__ = loop.symbol
    wrapper.__ppy_fused__ = loop  # type: ignore[attr-defined]
    binding.wrapper = wrapper
    return binding


def _splittable(loop: FusedLoop) -> bool:
    """Can this kernel be split across workers without changing its result?

    An elementwise map writes disjoint output elements, so it always can.
    `min` and `max` are associative once NaN propagates. `add` and `mul`
    only reach here when the program already permitted reassociation, so
    splitting adds no new licence. `mean` is excluded: per-chunk means
    cannot be merged without weighting, and unequal final chunks make that
    wrong.
    """
    if not loop.returns_scalar:
        return True
    return loop.reduction in {"add", "mul", "max", "min"}


def _combine(reduction: str, partials: list[float]) -> float:
    """Merge per-chunk results for a recognized reduction (spec 17.2)."""
    if reduction == "mul":
        result = 1.0
        for value in partials:
            result *= value
        return result
    if reduction == "max":
        return max(partials)
    if reduction == "min":
        return min(partials)
    return sum(partials)
