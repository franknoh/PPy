"""Typed input for programs that read a lot of it (spec 28).

`input()` and `sys.stdin.read().split()` build a Python object for every
field, which is most of what a competitive-programming submission spends its
time on. `read_ints` fills a buffer straight from file descriptor 0 through a
small C reader, so the numbers never become objects on the way in.

The reader is compiled once and cached; without a C compiler the pure-Python
fallback below is used instead, and the only difference is speed.
"""

from __future__ import annotations

import array as _array
import ctypes
import os
import sys
from typing import get_args as _get_args
from typing import get_origin as _get_origin

__all__ = ["input", "read_ints", "read_token", "reader_available"]

#: The reader owns file descriptor 0 and buffers it itself, so it must not be
#: mixed with `input()` or `sys.stdin` in the same program.
_SOURCE = r"""
#include <stdint.h>
#include <unistd.h>

static char ppy_buffer[1 << 16];
static long ppy_filled = 0;
static long ppy_position = 0;

static int ppy_next(void) {
    if (ppy_position == ppy_filled) {
        long got = (long)read(0, ppy_buffer, sizeof ppy_buffer);
        if (got <= 0) {
            return -1;
        }
        ppy_filled = got;
        ppy_position = 0;
    }
    return (unsigned char)ppy_buffer[ppy_position++];
}

static int ppy_is_space(int c) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\f' || c == '\v';
}

int64_t ppy_rt_read_token(int8_t *data, int64_t capacity) {
    int c = ppy_next();
    while (c != -1 && ppy_is_space(c)) {
        c = ppy_next();
    }
    int64_t count = 0;
    while (c != -1 && !ppy_is_space(c)) {
        if (count < capacity) {
            data[count] = (int8_t)c;
        }
        count++;
        c = ppy_next();
    }
    return count < capacity ? count : capacity;
}

int64_t ppy_rt_read_token_wide(int64_t *data, int64_t capacity) {
    int c = ppy_next();
    while (c != -1 && ppy_is_space(c)) {
        c = ppy_next();
    }
    int64_t count = 0;
    while (c != -1 && !ppy_is_space(c)) {
        if (count < capacity) {
            data[count] = c;
        }
        count++;
        c = ppy_next();
    }
    return count < capacity ? count : capacity;
}

int64_t ppy_rt_read_ints(int64_t *data, int64_t capacity) {
    int64_t count = 0;
    int c = ppy_next();
    while (count < capacity) {
        while (c != -1 && (c < '0' || c > '9') && c != '-') {
            c = ppy_next();
        }
        if (c == -1) {
            break;
        }
        int negative = 0;
        if (c == '-') {
            negative = 1;
            c = ppy_next();
        }
        int64_t value = 0;
        while (c >= '0' && c <= '9') {
            value = value * 10 + (c - '0');
            c = ppy_next();
        }
        data[count++] = negative ? -value : value;
    }
    return count;
}
"""

_LOADED: list[object] = []

#: Scratch space reused by the scalar reads, so one number costs no allocation.
_SCALAR_SLOT = _array.array("q", [0])
_TOKEN_SLOT = _array.array("b", bytes(4096))


#: Bumped whenever `_SOURCE` changes, so a stale reader is never loaded.
#: Naming the artifact from this and the interpreter's own tag keeps the
#: fast path free of `hashlib` and `sysconfig`, which cost more to import
#: than everything else a launch does.
_READER_VERSION = 1


def _cache_directory() -> str:
    spelled = os.environ.get("PPY_CACHE_DIR")
    root = spelled or os.path.join(os.path.expanduser("~"), ".cache", "ppy")
    return os.path.join(root, "reader")


def _build() -> object | None:
    """Compile the reader once, and hand back the loaded library.

    Everything this needs is imported here rather than at module scope: the
    compiler runs at most once in a program's life, and the modules it takes
    to run it are a measurable part of every start that does not.
    """
    directory = _cache_directory()
    name = f"ppy_reader_{_READER_VERSION}_{sys.implementation.cache_tag}.so"
    library = os.path.join(directory, name)
    if not os.path.isfile(library) and not _compile(library):
        return None
    try:
        return ctypes.CDLL(library)
    except OSError:
        return None


def _compile(library: str) -> bool:
    """Build the reader. Everything this needs is imported here: it runs at
    most once in a machine's life, and importing it costs every start that
    finds the artifact already there."""
    import subprocess
    import tempfile

    compiler = os.environ.get("CC") or "cc"
    try:
        os.makedirs(os.path.dirname(library), exist_ok=True)
    except OSError:
        return False
    with tempfile.TemporaryDirectory() as scratch:
        source = os.path.join(scratch, "reader.c")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write(_SOURCE)
        staged = os.path.join(scratch, os.path.basename(library))
        done = subprocess.run(
            [compiler, "-O2", "-shared", "-fPIC", source, "-o", staged],
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode != 0 or not os.path.isfile(staged):
            return False
        try:
            # Replace atomically, so a half-written file is never loaded.
            os.replace(staged, library)
        except OSError:
            return False
    return True


def _reader():  # type: ignore[no-untyped-def]
    if not _LOADED:
        library = _build()
        if library is None:
            _LOADED.append(None)
        else:
            entry = library.ppy_rt_read_ints
            entry.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_int64]
            entry.restype = ctypes.c_int64
            token = library.ppy_rt_read_token
            token.argtypes = [ctypes.POINTER(ctypes.c_int8), ctypes.c_int64]
            token.restype = ctypes.c_int64
            wide = library.ppy_rt_read_token_wide
            wide.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_int64]
            wide.restype = ctypes.c_int64
            token = (token, wide)
            # The library is kept alongside the entry points: dropping it
            # would unload the code the pointers refer to.
            _LOADED.append((entry, token, library))
    return _LOADED[0]


def reader_available() -> bool:
    """Whether the compiled reader is in use rather than the Python fallback."""
    return _reader() is not None


#: Fields the fallback has read but not yet handed out. Successive calls
#: continue where the last one stopped, exactly as the C reader does.
_PENDING: list[str] = []
_DRAINED: list[bool] = []


def _fallback(buffer) -> int:  # type: ignore[no-untyped-def]
    """The same contract without a C compiler: correct, and slower."""
    if not _DRAINED:
        _PENDING.extend(reversed(sys.stdin.read().split()))
        _DRAINED.append(True)
    capacity = len(buffer)
    count = 0
    while count < capacity and _PENDING:
        buffer[count] = int(_PENDING.pop())
        count += 1
    return count


def read_ints(buffer) -> int:  # type: ignore[no-untyped-def]
    """Fill `buffer` with whitespace-separated integers from standard input.

    Returns how many were read, which is fewer than `len(buffer)` only when
    the input ran out. The buffer must be a writable contiguous buffer of
    64-bit integers -- `array.array("q", ...)` is the usual one.

    This reads file descriptor 0 directly, so a program that calls it must
    not also read `input()` or `sys.stdin`.
    """
    found = _reader()
    if found is None:
        return _fallback(buffer)
    entry, _tokens, _library = found
    view = _writable(buffer, 8, "read_ints needs a writable buffer of 64-bit integers")
    address = ctypes.addressof(ctypes.c_char.from_buffer(view))
    return int(entry(ctypes.cast(address, ctypes.POINTER(ctypes.c_int64)), view.shape[0]))


def _writable(buffer, itemsize: int, complaint: str):  # type: ignore[no-untyped-def]
    view = memoryview(buffer)
    if view.readonly or not view.c_contiguous or view.itemsize != itemsize:
        raise TypeError(complaint)
    return view


def _fallback_token(buffer) -> int:  # type: ignore[no-untyped-def]
    if not _DRAINED:
        _PENDING.extend(reversed(sys.stdin.read().split()))
        _DRAINED.append(True)
    if not _PENDING:
        return 0
    encoded = _PENDING.pop().encode("utf-8")[: len(buffer)]
    for index, byte in enumerate(encoded):
        buffer[index] = byte
    return len(encoded)


def read_token(buffer) -> int:  # type: ignore[no-untyped-def]
    """Read one whitespace-delimited token into `buffer`, one byte per slot.

    The buffer may hold bytes (`array.array("b", ...)`) or 64-bit integers
    (`array.array("q", ...)`); the wider one is what a native kernel indexes,
    since `Buffer[int]` is 64-bit. Returns how many slots were written, capped
    at `len(buffer)` -- the rest of an over-long token is consumed and dropped.
    """
    found = _reader()
    if found is None:
        return _fallback_token(buffer)
    _entry, (narrow, wide), _library = found
    view = memoryview(buffer)
    if view.readonly or not view.c_contiguous or view.itemsize not in (1, 8):
        raise TypeError("read_token needs a writable buffer of bytes or 64-bit integers")
    address = ctypes.addressof(ctypes.c_char.from_buffer(view))
    if view.itemsize == 1:
        return int(narrow(ctypes.cast(address, ctypes.POINTER(ctypes.c_int8)), view.shape[0]))
    return int(wide(ctypes.cast(address, ctypes.POINTER(ctypes.c_int64)), view.shape[0]))


def _one_int() -> int:
    slot = _SCALAR_SLOT
    if read_ints(slot) != 1:
        raise EOFError("the input ended where an integer was expected")
    return slot[0]


def _one_token() -> str:
    room = _TOKEN_SLOT
    length = read_token(room)
    if length == 0:
        raise EOFError("the input ended where a token was expected")
    return bytes(memoryview(room)[:length]).decode("utf-8")


def _one_float() -> float:
    return float(_one_token())


_SCALARS = {int: _one_int, float: _one_float, str: _one_token, bool: lambda: bool(_one_int())}


def _buffer_element(spec) -> object | None:  # type: ignore[no-untyped-def]
    """`ppy.Buffer[T]`'s element type, or None if `spec` is not one."""
    for item in getattr(spec, "__metadata__", ()):
        element = getattr(item, "element", None)
        if element is not None:
            return element
    return None


class _Reading:
    """One `ppy.input[T]`, waiting to be called."""

    __slots__ = ("_spec",)

    def __init__(self, spec) -> None:  # type: ignore[no-untyped-def]
        self._spec = spec

    def __repr__(self) -> str:
        return f"ppy.input[{getattr(self._spec, '__name__', self._spec)!r}]"

    def __call__(self, argument=None):  # type: ignore[no-untyped-def]
        spec = self._spec
        element = _buffer_element(spec)
        if element is not None:
            if not isinstance(argument, int):
                raise TypeError("reading a buffer needs how many values to read")
            if element is not int:
                raise TypeError("only `Buffer[int]` can be read for now")
            values = _array.array("q", bytes(8 * argument))
            read_ints(values)
            return values
        if argument is not None:
            if not isinstance(argument, str):
                raise TypeError("the argument to a scalar read is a prompt")
            sys.stdout.write(argument)
            sys.stdout.flush()
        return _read_one(spec)


def _read_one(spec):  # type: ignore[no-untyped-def]
    reader = _SCALARS.get(spec)
    if reader is not None:
        return reader()
    if _get_origin(spec) is tuple:
        parts = _get_args(spec)
        if not parts or Ellipsis in parts:
            raise TypeError("a tuple to read needs a fixed number of typed fields")
        return tuple(_read_one(part) for part in parts)
    raise TypeError(f"{spec!r} is not something `ppy.input` knows how to read")


class _TypedInput:
    """`ppy.input[T](...)`: read the next value as `T` says to read it.

    ```python
    n = ppy.input[int]()                   # one integer
    a, b = ppy.input[tuple[int, int]]()    # two fields, line breaks irrelevant
    word = ppy.input[str]("name? ")        # a token, after printing the prompt
    values = ppy.input[Buffer[int]](n)     # n integers, straight into a buffer
    ```

    Whitespace and newlines are the same thing to it, as they are to `scanf`.
    Reading goes into memory rather than through a Python object per field,
    so the buffer form is what takes a million numbers quickly.
    """

    __slots__ = ()

    def __getitem__(self, spec) -> _Reading:  # type: ignore[no-untyped-def]
        return _Reading(spec)

    def __call__(self, *_args, **_keywords):  # type: ignore[no-untyped-def]
        raise TypeError("`ppy.input` needs the type it is reading: `ppy.input[int]()`")


#: The name shadows the builtin on purpose: this is the typed one.
input = _TypedInput()  # pylint: disable=redefined-builtin
