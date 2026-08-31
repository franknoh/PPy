"""Typed input for programs that read a lot of it (spec 28).

`input()` and `sys.stdin.read().split()` build a Python object for every
field, which is most of what a competitive-programming submission spends its
time on. `read_ints` fills a buffer straight from file descriptor 0 through a
small C reader, so the numbers never become objects on the way in.

The reader is compiled once and cached; without a C compiler the pure-Python
fallback below is used instead, and the only difference is speed.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

__all__ = ["read_ints", "read_token", "reader_available"]

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


def _cache_directory() -> Path:
    spelled = os.environ.get("PPY_CACHE_DIR")
    root = Path(spelled) if spelled else Path.home() / ".cache" / "ppy"
    return root / "reader"


def _build() -> object | None:
    """Compile the reader once, and hand back the loaded library."""
    compiler = os.environ.get("CC") or "cc"
    tag = hashlib.sha256(
        (_SOURCE + sys.version + sysconfig.get_platform()).encode("utf-8")
    ).hexdigest()[:16]
    directory = _cache_directory()
    library = directory / f"ppy_reader_{tag}.so"
    if not library.is_file():
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "reader.c"
            source.write_text(_SOURCE, encoding="utf-8")
            staged = Path(scratch) / library.name
            done = subprocess.run(
                [compiler, "-O2", "-shared", "-fPIC", str(source), "-o", str(staged)],
                capture_output=True,
                text=True,
                check=False,
            )
            if done.returncode != 0 or not staged.is_file():
                return None
            try:
                # Rename into place, so a half-written file is never loaded.
                staged.replace(library)
            except OSError:
                return None
    try:
        return ctypes.CDLL(str(library))
    except OSError:
        return None


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
