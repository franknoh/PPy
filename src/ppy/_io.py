"""Typed input: lines the way `input()` reads them, tokens the way a scanner does (spec 28).

Two readers over one standard input, with one grammar between them:

```python
line = ppy.input[str]()              # one line, the newline removed, as input() gives it
n = ppy.input[int]()                 # one line, read as int(input()) reads it
a, b = ppy.input[tuple[int, int]]()  # one line split into exactly two fields

word = ppy.scan[str]()               # one whitespace-delimited token, lines irrelevant
k = ppy.scan[int]()                  # one integer token
values = ppy.scan[Buffer[int]](n)    # n integer tokens, straight into a buffer
```

`ppy.input` is line-oriented and means what the builtin `input()` means, so
a converted program reads what it read before. `ppy.scan` is the scanner:
whitespace and newlines are the same thing to it, as they are to `scanf`,
and reading goes into memory rather than through a Python object per
field, which is what takes a million numbers quickly. `ppy.read_ints` and
`ppy.read_token` are the scanner's low-level forms over a buffer the caller
already has.

The scanner is C compiled once and cached (`ppy_runtime.scanner`, the same
text a standalone binary links); without a C compiler the pure-Python
fallback below reads exactly the same grammar, and the only difference is
speed. Both own file descriptor 0 and buffer it, so a program that reads
through them must not also read `input()` or `sys.stdin`.
"""

from __future__ import annotations

import array as _array
import ctypes
import os
import re
import struct as _struct
import sys
from typing import Any
from typing import get_args as _get_args
from typing import get_origin as _get_origin

__all__ = ["input", "read_ints", "read_token", "reader_available", "scan"]

#: Room a token or a line is read into at a time; a longer one continues.
_CHUNK = 1 << 12
#: How much of an offending token an error message shows.
_BAD = 64
#: The integer-token grammar, the scanner's: a sign, digits, nothing else.
_INTEGER = re.compile(rb"[+-]?[0-9]+\Z")
#: An ASCII field as `int()` reads it: a sign, digits, single underscores between them.
_FIELD = re.compile(rb"[+-]?[0-9](?:_?[0-9])*\Z")
#: Where a line of integers starts: slots, doubled while the line goes on.
_LINE_SLOTS = 1024
_LIMIT = 1 << 63

#: Bumped whenever the scanner's C changes, so a stale reader is never loaded.
#: Naming the artifact from this and the interpreter's own tag keeps the
#: fast path free of `hashlib` and `sysconfig`, which cost more to import
#: than everything else a launch does.
_READER_VERSION = 4


def _cache_directory() -> str:
    spelled = os.environ.get("PPY_CACHE_DIR")
    root = spelled or os.path.join(os.path.expanduser("~"), ".cache", "ppy")
    return os.path.join(root, "reader")


def _build() -> object | None:
    """Compile the reader once, and hand back the loaded library."""
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

    from ppy_runtime.scanner import library_source

    compiler = os.environ.get("CC") or "cc"
    try:
        os.makedirs(os.path.dirname(library), exist_ok=True)
    except OSError:
        return False
    with tempfile.TemporaryDirectory() as scratch:
        source = os.path.join(scratch, "reader.c")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write(library_source())
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


def _malformed(token: bytes) -> ValueError:
    """The one message for a token that is not an integer, from either reader."""
    shown = token.decode("utf-8", "replace")
    if _INTEGER.match(token):
        return ValueError(f"the integer {shown} does not fit in 64 bits")
    return ValueError(f"expected an integer, got {shown!r}")


def _bad_field(token: bytes, overflow: bool) -> Exception:
    """What `array.array("q", map(int, line.split()))` raises for the field, from either reader."""
    if overflow:
        return OverflowError("int too big to convert")
    shown = token[: _BAD - 1].decode("utf-8", "replace")
    return ValueError(f"invalid literal for int() with base 10: {shown!r}")


def _address(view: memoryview, kind):  # type: ignore[no-untyped-def]
    return ctypes.cast(ctypes.addressof(ctypes.c_char.from_buffer(view)), ctypes.POINTER(kind))


class _Compiled:
    """The scanner's C functions, called through ctypes."""

    __slots__ = (
        "_bad",
        "_chunk",
        "_fixed",
        "_ints",
        "_library",
        "_line",
        "_line_ints",
        "_more",
        "_more_ref",
        "_token",
        "_wide",
    )
    _bad: Any
    _chunk: Any
    _fixed: Any
    _ints: Any
    _library: Any
    _line: Any
    _line_ints: Any
    _more: Any
    _more_ref: Any
    _token: Any
    _wide: Any

    def __init__(self, library) -> None:  # type: ignore[no-untyped-def]
        i8, i64 = ctypes.c_int8, ctypes.c_int64
        p8, p64 = ctypes.POINTER(i8), ctypes.POINTER(i64)
        self._token = library.ppy_rt_read_token
        self._token.argtypes = [p8, i64, i8, p8]
        self._token.restype = i64
        self._wide = library.ppy_rt_read_token_wide
        self._wide.argtypes = [p64, i64, i8, p8]
        self._wide.restype = i64
        self._line = library.ppy_rt_read_line
        self._line.argtypes = [p8, i64, i8, p8]
        self._line.restype = i64
        self._ints = library.ppy_rt_read_ints
        self._ints.argtypes = [p64, i64, p8, i64]
        self._ints.restype = i64
        self._line_ints = library.ppy_rt_input_ints
        self._line_ints.argtypes = [p64, i64, p8, i64, i8, p8]
        self._line_ints.restype = i64
        self._fixed = library.ppy_rt_input_fixed
        self._fixed.argtypes = [p64, i64, p8, i64]
        self._fixed.restype = i64
        # The library is kept beside the entry points: dropping it would
        # unload the code the pointers refer to.
        self._library = library
        self._chunk = (i8 * _CHUNK)()
        self._bad = (i8 * _BAD)()
        self._more = i8(0)
        self._more_ref = ctypes.byref(self._more)

    def _whole(self, function) -> bytes | None:  # type: ignore[no-untyped-def]
        """A token or a line in full, chunk by chunk; None where there is none."""
        parts: list[bytes] = []
        continuing = 0
        while True:
            got = int(function(self._chunk, _CHUNK, continuing, ctypes.byref(self._more)))
            if got < 0:
                return None
            parts.append(ctypes.string_at(self._chunk, got))
            if not self._more.value:
                return b"".join(parts)
            continuing = 1

    def token(self) -> bytes | None:
        found = self._whole(self._token)
        return found or None

    def line(self) -> bytes | None:
        # One call answers nearly every line; the chunk loop is for the rest.
        got = int(self._line(self._chunk, _CHUNK, 0, self._more_ref))
        if got < 0:
            return None
        if not self._more.value:
            return ctypes.string_at(self._chunk, got)
        first = ctypes.string_at(self._chunk, got)
        return first + (self._whole_continued(self._line) or b"")

    def _whole_continued(self, function) -> bytes:  # type: ignore[no-untyped-def]
        parts: list[bytes] = []
        while True:
            got = int(function(self._chunk, _CHUNK, 1, self._more_ref))
            parts.append(ctypes.string_at(self._chunk, got))
            if not self._more.value:
                return b"".join(parts)

    def token_into(self, view: memoryview) -> int:
        """`read_token`: fill `view`, then consume and drop the rest of the token."""
        function = self._token if view.itemsize == 1 else self._wide
        kind = ctypes.c_int8 if view.itemsize == 1 else ctypes.c_int64
        got = int(function(_address(view, kind), view.shape[0], 0, ctypes.byref(self._more)))
        while self._more.value:
            self._token(self._chunk, _CHUNK, 1, ctypes.byref(self._more))
        return got

    def ints_into(self, view: memoryview) -> int:
        got = int(self._ints(_address(view, ctypes.c_int64), view.shape[0], self._bad, _BAD))
        if got < 0:
            raise _malformed(ctypes.string_at(self._bad))
        return got

    def line_fields(self, _view: memoryview, address, room: int) -> int:  # type: ignore[no-untyped-def]
        """A line of integers into the `room` slots at `address`; how many the line held.

        Every field is checked, the ones past the room included, and the line
        is consumed whole; more fields than room is answered by their count.
        """
        got = self._fixed(address, room, self._bad, _BAD)
        if got >= 0:
            return got
        if got == -1:
            raise EOFError("EOF when reading a line")
        raise _bad_field(ctypes.string_at(self._bad), got == -3)

    def line_ints(self) -> _array.array:
        """One line of integers into a new buffer, however many the line holds."""
        values = _array.array("q", bytes(8 * _LINE_SLOTS))
        filled = 0
        continuing = 0
        while True:
            # The view is let go before the array is resized, which an export forbids.
            with memoryview(values)[filled:] as view:
                got = int(
                    self._line_ints(
                        _address(view, ctypes.c_int64),
                        view.shape[0],
                        self._bad,
                        _BAD,
                        continuing,
                        self._more_ref,
                    )
                )
            if got < 0:
                raise EOFError("EOF when reading a line")
            filled += got
            state = self._more.value
            if state >= 2:
                raise _bad_field(ctypes.string_at(self._bad), state == 3)
            if state == 0:
                break
            values.frombytes(bytes(8 * len(values)))
            continuing = 1
        del values[filled:]
        return values


_SPACE_BYTES = re.compile(rb"[ \t\n\r\f\v]")
_NON_SPACE = re.compile(rb"[^ \t\n\r\f\v]")
_NEWLINE = re.compile(rb"\n")


class _Fallback:
    """The same contract without a C compiler: correct, and slower."""

    __slots__ = ("_buffer", "_eof", "_position")
    _buffer: bytes
    _eof: bool
    _position: int

    def __init__(self) -> None:
        self._buffer = b""
        self._position = 0
        self._eof = False

    def _fill(self) -> bool:
        """More bytes after the position; False at the end of the input."""
        if self._eof:
            return False
        chunk = sys.stdin.buffer.read(1 << 16)
        if not chunk:
            self._eof = True
            return False
        self._buffer = self._buffer[self._position :] + chunk
        self._position = 0
        return True

    def _ensure(self) -> bool:
        return self._position < len(self._buffer) or self._fill()

    def _skip_space(self) -> None:
        while self._ensure():
            found = _NON_SPACE.search(self._buffer, self._position)
            if found is not None:
                self._position = found.start()
                return
            self._position = len(self._buffer)

    def _until(self, stop: re.Pattern[bytes]) -> bytes:
        """Bytes up to the first match of `stop` (not consumed), or to the end."""
        parts: list[bytes] = []
        while self._ensure():
            found = stop.search(self._buffer, self._position)
            if found is not None:
                parts.append(self._buffer[self._position : found.start()])
                self._position = found.start()
                break
            parts.append(self._buffer[self._position :])
            self._position = len(self._buffer)
        return b"".join(parts)

    def token(self) -> bytes | None:
        self._skip_space()
        found = self._until(_SPACE_BYTES)
        return found or None

    def line(self) -> bytes | None:
        if not self._ensure():
            return None
        found = self._until(_NEWLINE)
        if self._position < len(self._buffer):
            self._position += 1  # the newline
        return found

    def token_into(self, view: memoryview) -> int:
        found = self.token() or b""
        count = min(len(found), view.shape[0])
        target = view.cast("B") if view.itemsize == 1 else view
        for index in range(count):
            target[index] = found[index]
        return count

    def ints_into(self, view: memoryview) -> int:
        count = 0
        capacity = view.shape[0]
        while count < capacity:
            found = self.token()
            if found is None:
                break
            if not _INTEGER.match(found):
                raise _malformed(found)
            value = int(found)
            if not -_LIMIT <= value < _LIMIT:
                raise _malformed(found)
            view[count] = value
            count += 1
        return count

    def _fields(self) -> list[int]:
        """A line's integer fields, every one checked left to right, as the C reader does."""
        line = self.line()
        if line is None:
            raise EOFError("EOF when reading a line")
        values = []
        for field in line.split():
            if not _FIELD.match(field):
                raise _bad_field(field, False)
            value = int(field)
            if not -_LIMIT <= value < _LIMIT:
                raise _bad_field(field, True)
            values.append(value)
        return values

    def line_fields(self, view: memoryview, _address, room: int) -> int:  # type: ignore[no-untyped-def]
        values = self._fields()
        for index, value in enumerate(values[:room]):
            view[index] = value
        return len(values)

    def line_ints(self) -> _array.array:
        return _array.array("q", self._fields())


_SOURCE: list[object] = []


def _source():  # type: ignore[no-untyped-def]
    if not _SOURCE:
        library = _build()
        _SOURCE.append(_Fallback() if library is None else _Compiled(library))
    return _SOURCE[0]


def reader_available() -> bool:
    """Whether the compiled scanner is in use rather than the Python fallback."""
    return isinstance(_source(), _Compiled)


# -- the low-level scanner -------------------------------------------------------------


def read_ints(buffer) -> int:  # type: ignore[no-untyped-def]
    """Fill `buffer` with integer tokens from standard input.

    Returns how many were read, which is fewer than `len(buffer)` only when
    the input ran out. A token that is not an integer -- anything but an
    optional sign and digits, or a value outside 64 bits -- raises
    `ValueError` naming it, with the tokens before it already stored. The
    buffer must be a writable contiguous buffer of 64-bit integers;
    `array.array("q", ...)` is the usual one.

    This reads file descriptor 0 directly, so a program that calls it must
    not also read `input()` or `sys.stdin`.
    """
    view = memoryview(buffer)
    if view.readonly or not view.c_contiguous or view.itemsize != 8:
        raise TypeError("read_ints needs a writable buffer of 64-bit integers")
    return _source().ints_into(view)


def read_token(buffer) -> int:  # type: ignore[no-untyped-def]
    """Read one whitespace-delimited token into `buffer`, one byte per slot.

    The buffer may hold bytes (`array.array("b", ...)`) or 64-bit integers
    (`array.array("q", ...)`); the wider one is what a native kernel indexes,
    since `Buffer[int]` is 64-bit. Returns how many slots were written. The
    buffer's length is the capacity the caller chose: a longer token is cut
    there, and the rest of it is consumed and dropped. `ppy.scan[str]()` is
    the form with no such limit.
    """
    view = memoryview(buffer)
    if view.readonly or not view.c_contiguous or view.itemsize not in (1, 8):
        raise TypeError("read_token needs a writable buffer of bytes or 64-bit integers")
    return _source().token_into(view)


# -- scalar reads --------------------------------------------------------------------

_SCALAR_SLOT = _array.array("q", [0])


def _scan_int() -> int:
    slot = _SCALAR_SLOT
    if read_ints(slot) != 1:
        raise EOFError("the input ended where an integer was expected")
    return slot[0]


def _scan_token() -> str:
    found = _source().token()
    if found is None:
        raise EOFError("the input ended where a token was expected")
    return found.decode("utf-8")


def _scan_float() -> float:
    return float(_scan_token())


_SCANNED = {int: _scan_int, float: _scan_float, str: _scan_token, bool: lambda: bool(_scan_int())}


def _input_line() -> str:
    found = _source().line()
    if found is None:
        raise EOFError("EOF when reading a line")
    return found.removesuffix(b"\r").decode("utf-8")


_LINE_FIELDS = {int: int, float: float, str: str}


def _buffer_element(spec) -> object | None:  # type: ignore[no-untyped-def]
    """`ppy.Buffer[T]`'s element type, or None if `spec` is not one."""
    for item in getattr(spec, "__metadata__", ()):
        element = getattr(item, "element", None)
        if element is not None:
            return element
    return None


def _spelled(what: str, spec) -> str:  # type: ignore[no-untyped-def]
    return f"ppy.{what}[{getattr(spec, '__name__', spec)!r}]"


class _IntFields:
    """`ppy.input[tuple[int, ...]]()` with every field an integer: read in C, into one slot."""

    __slots__ = ("_address", "_count", "_slot", "_unpack", "_view")
    _address: Any
    _count: int
    _slot: _array.array
    _unpack: Any
    _view: memoryview

    def __init__(self, count: int) -> None:
        self._count = count
        # One slot more than the tuple, so a longer line is seen as one.
        self._slot = _array.array("q", bytes(8 * (count + 1)))
        self._view = memoryview(self._slot)
        self._address = _address(self._view, ctypes.c_int64)
        self._unpack = _struct.Struct("=" + "q" * count).unpack_from

    def __call__(self) -> tuple:
        got = _source().line_fields(self._view, self._address, self._count + 1)
        if got != self._count:
            raise ValueError(f"expected {self._count} field(s) on the line, got {got}")
        return self._unpack(self._slot)


class _Fields:
    """`ppy.input[tuple[...]]()` with mixed fields: the line split, each field converted."""

    __slots__ = ("_converters",)
    _converters: tuple

    def __init__(self, converters: tuple) -> None:
        self._converters = converters

    def __call__(self) -> tuple:
        fields = _input_line().split()
        if len(fields) != len(self._converters):
            raise ValueError(
                f"expected {len(self._converters)} field(s) on the line, got {len(fields)}"
            )
        return tuple(
            convert(field) for convert, field in zip(self._converters, fields, strict=True)
        )  # type: ignore[misc]


def _line_buffer() -> _array.array:
    return _source().line_ints()


class _LineRead:
    """One `ppy.input[T]`, planned once and called any number of times."""

    __slots__ = ("_buffer", "_run", "_spec")
    _buffer: bool
    _run: Any
    _spec: Any

    def __init__(self, spec) -> None:  # type: ignore[no-untyped-def]
        self._spec = spec
        self._buffer = False
        self._run = self._plan(spec)

    def __repr__(self) -> str:
        return _spelled("input", self._spec)

    def _plan(self, spec):  # type: ignore[no-untyped-def]
        element = _buffer_element(spec)
        if element is not None:
            if element is not int:
                shown = getattr(element, "__name__", element)
                raise TypeError(
                    f"a line of integers reads into `Buffer[int]`, not `Buffer[{shown}]`"
                )
            self._buffer = True
            return _line_buffer
        if spec is str:
            return _input_line
        convert = _LINE_FIELDS.get(spec)
        if convert is not None:
            return lambda: convert(_input_line())
        origin = _get_origin(spec)
        if origin is list:
            parts = _get_args(spec)
            element = _LINE_FIELDS.get(parts[0]) if len(parts) == 1 else None
            if element is None:
                raise TypeError(f"{self!r} reads a line as a list of int, float, or str")
            return lambda: [element(field) for field in _input_line().split()]
        if origin is tuple:
            parts = _get_args(spec)
            if not parts or Ellipsis in parts:
                raise TypeError("a tuple to read needs a fixed number of typed fields")
            converters = tuple(_LINE_FIELDS.get(part) for part in parts)
            if any(convert is None for convert in converters):
                raise TypeError(f"{self!r} reads fields of int, float, or str")
            if all(part is int for part in parts):
                return _IntFields(len(parts))
            return _Fields(converters)
        raise TypeError(f"{spec!r} is not something `ppy.input` knows how to read")

    def __call__(self, argument=None):  # type: ignore[no-untyped-def]
        if argument is not None:
            if self._buffer:
                raise TypeError(
                    "`ppy.input[Buffer[int]]()` reads the whole line and takes no count; "
                    "`ppy.scan[Buffer[int]](n)` reads n tokens"
                )
            raise TypeError("`ppy.input[T]()` takes no argument; print a prompt first, then read")
        return self._run()


class _TokenRead:
    """One `ppy.scan[T]`, waiting to be called."""

    __slots__ = ("_spec",)
    _spec: Any

    def __init__(self, spec) -> None:  # type: ignore[no-untyped-def]
        self._spec = spec

    def __repr__(self) -> str:
        return _spelled("scan", self._spec)

    def __call__(self, argument=None):  # type: ignore[no-untyped-def]
        spec = self._spec
        element = _buffer_element(spec)
        if element is not None:
            if not isinstance(argument, int) or isinstance(argument, bool):
                raise TypeError("reading a buffer needs how many values to read")
            if element is not int:
                raise TypeError("only `Buffer[int]` can be scanned for now")
            values = _array.array("q", bytes(8 * max(argument, 0)))
            read_ints(values)
            return values
        if argument is not None:
            raise TypeError("`ppy.scan[T]()` takes no argument; print a prompt first, then read")
        return _scan_one(spec)


def _scan_one(spec):  # type: ignore[no-untyped-def]
    reader = _SCANNED.get(spec)
    if reader is not None:
        return reader()
    if _get_origin(spec) is tuple:
        parts = _get_args(spec)
        if not parts or Ellipsis in parts:
            raise TypeError("a tuple to read needs a fixed number of typed fields")
        return tuple(_scan_one(part) for part in parts)
    raise TypeError(f"{spec!r} is not something `ppy.scan` knows how to read")


_LINE_READS: dict = {}
_TOKEN_READS: dict = {}


def _planned(cache: dict, kind, spec):  # type: ignore[no-untyped-def]
    """The reader for `spec`, made once and kept: a read in a loop plans nothing per call."""
    try:
        return cache.setdefault(spec, kind(spec))
    except TypeError:  # an unhashable spec is planned each time
        return kind(spec)


class _Input:
    """`ppy.input[T]()`: one line of standard input, read as `T` reads a line.

    ```python
    line = ppy.input[str]()              # the line, newline removed; "" for an empty line
    n = ppy.input[int]()                 # int(input()): the whole line is the number
    x = ppy.input[float]()               # float(input())
    a, b = ppy.input[tuple[int, int]]()  # input().split(), exactly two fields
    ```

    The line is what the builtin `input()` would return: the trailing
    newline is removed, spaces inside are kept, an empty line is `""`, the
    last line needs no newline, and the end of the input is `EOFError`. A
    line read as a number must be one number -- `1 2` is `ValueError`, as
    `int("1 2")` is -- and a tuple takes exactly its fields from one line,
    never from the next. A prompt is a `print` before the read.
    """

    __slots__ = ()

    def __getitem__(self, spec) -> _LineRead:  # type: ignore[no-untyped-def]
        try:
            return _LINE_READS[spec]
        except (KeyError, TypeError):
            return _planned(_LINE_READS, _LineRead, spec)

    def __call__(self, *_args, **_keywords):  # type: ignore[no-untyped-def]
        raise TypeError("`ppy.input` needs the type it is reading: `ppy.input[int]()`")


class _Scan:
    """`ppy.scan[T]()`: the next token of standard input, whatever line it is on.

    ```python
    n = ppy.scan[int]()                  # one integer token
    a, b = ppy.scan[tuple[int, int]]()   # two tokens, line breaks irrelevant
    word = ppy.scan[str]()               # one token, however long
    values = ppy.scan[Buffer[int]](n)    # n integer tokens, straight into a buffer
    ```

    Whitespace and newlines are the same thing to it, as they are to
    `scanf`; a token read stops before the whitespace after it, so a line
    read that follows starts there. An integer token is a sign and digits
    within 64 bits, and anything else where one is expected is `ValueError`.
    """

    __slots__ = ()

    def __getitem__(self, spec) -> _TokenRead:  # type: ignore[no-untyped-def]
        try:
            return _TOKEN_READS[spec]
        except (KeyError, TypeError):
            return _planned(_TOKEN_READS, _TokenRead, spec)

    def __call__(self, *_args, **_keywords):  # type: ignore[no-untyped-def]
        raise TypeError("`ppy.scan` needs the type it is reading: `ppy.scan[int]()`")


#: The name shadows the builtin on purpose: this is the typed one.
input = _Input()  # pylint: disable=redefined-builtin
scan = _Scan()
