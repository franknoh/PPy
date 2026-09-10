"""The input scanner, in C, defined once for everyone who reads standard input.

`ppy.scan` and `ppy.read_*` under CPython load these functions from a small
shared library compiled on first use; a standalone binary links the same
text in as its runtime shims. One grammar, one implementation, two homes,
so a token means the same thing whichever way a program was built.

The grammar:

* Whitespace is ASCII: space, tab, newline, carriage return, form feed,
  vertical tab -- what `bytes.split()` splits on.
* A token is a maximal run of non-whitespace bytes. Reading one stops
  *before* the whitespace that ends it, so the newline after a token is
  still there for a line read that follows.
* A line is every byte up to a newline, which is consumed and not part of
  the line; the last line of the input needs no newline.
* An integer token is `[+-]?[0-9]+` and fits in a signed 64-bit word.
  Anything else where an integer is expected is an error, never skipped.

The functions keep no state between calls beyond the byte buffer itself:
a token or a line that does not fit the room it was given is cut there,
`*more` says so, and the caller continues with `continuing = 1`, which
means "do not skip whitespace first". The Python fallback in `ppy._io`
implements exactly this contract.
"""

from __future__ import annotations

__all__ = ["FUNCTIONS", "INTERNAL", "STATEFUL", "library_source"]

#: The buffered byte source. `peek` looks without consuming. This is the one
#: function that owns state, which is why a header-only unit refuses it.
NEXT = """{
    static char room[1 << 16];
    static long filled = 0;
    static long position = 0;
    if (position == filled) {
        filled = (long)fread(room, 1, sizeof room, stdin);
        if (filled <= 0) {
            filled = 0;
            position = 0;
            return -1;
        }
        position = 0;
    }
    if (peek) {
        return (unsigned char)room[position];
    }
    return (unsigned char)room[position++];
}"""

_IS_SPACE = "(c == ' ' || c == '\\n' || c == '\\r' || c == '\\t' || c == '\\f' || c == '\\v')"


def _token(slot: str) -> str:
    return f"""{{
    int c = ppy_rt_next(1);
    if (!continuing) {{
        while (c != -1 && {_IS_SPACE}) {{
            ppy_rt_next(0);
            c = ppy_rt_next(1);
        }}
    }}
    int64_t count = 0;
    while (c != -1 && !{_IS_SPACE}) {{
        if (count == capacity) {{
            *more = 1;
            return count;
        }}
        data[count++] = ({slot})c;
        ppy_rt_next(0);
        c = ppy_rt_next(1);
    }}
    *more = 0;
    return count;
}}"""


READ_TOKEN = _token("int8_t")
READ_TOKEN_WIDE = _token("int64_t")

READ_LINE = """{
    int c = ppy_rt_next(1);
    if (c == -1 && !continuing) {
        *more = 0;
        return -1;
    }
    int64_t count = 0;
    while (c != -1 && c != '\\n') {
        if (count == capacity) {
            *more = 1;
            return count;
        }
        data[count++] = (int8_t)c;
        ppy_rt_next(0);
        c = ppy_rt_next(1);
    }
    if (c == '\\n') {
        ppy_rt_next(0);
    }
    *more = 0;
    return count;
}"""

#: `[+-]?[0-9]+` within a signed 64-bit word, or the offending token copied
#: into `bad` (cut to fit with its terminating NUL, the rest consumed) and
#: `~count` back.
READ_INTS = """{
    int64_t count = 0;
    while (count < capacity) {
        int c = ppy_rt_next(1);
        while (c != -1 && IS_SPACE) {
            ppy_rt_next(0);
            c = ppy_rt_next(1);
        }
        if (c == -1) {
            break;
        }
        int64_t seen = 0;
        int negative = 0;
        int digits = 0;
        int failed = 0;
        uint64_t magnitude = 0;
        if (c == '-' || c == '+') {
            negative = c == '-';
            if (seen < bad_capacity) {
                bad[seen] = (int8_t)c;
            }
            seen++;
            ppy_rt_next(0);
            c = ppy_rt_next(1);
        }
        while (c != -1 && !IS_SPACE) {
            if (seen < bad_capacity) {
                bad[seen] = (int8_t)c;
            }
            seen++;
            if (c < '0' || c > '9') {
                failed = 1;
            } else if (!failed) {
                uint64_t digit = (uint64_t)(c - '0');
                if (magnitude > 922337203685477580ULL
                    || (magnitude == 922337203685477580ULL && digit > (uint64_t)(7 + negative))) {
                    failed = 1;
                } else {
                    magnitude = magnitude * 10 + digit;
                    digits++;
                }
            }
            ppy_rt_next(0);
            c = ppy_rt_next(1);
        }
        if (failed || digits == 0) {
            bad[seen < bad_capacity - 1 ? seen : bad_capacity - 1] = 0;
            return ~count;
        }
        data[count++] = negative ? (int64_t)(0 - magnitude) : (int64_t)magnitude;
    }
    return count;
}""".replace("IS_SPACE", _IS_SPACE)

#: `ppy.input[Buffer[int]]()`: one line, every field an integer as Python's
#: `int()` reads an ASCII one -- a sign, digits, single underscores between
#: them -- into `data`. `*more` says how the call ended: 0 at the end of the
#: line (consumed), 1 with the line continuing past `capacity` (call again
#: with `continuing`), 2 with a field that is not an integer and 3 with one
#: outside 64 bits; for 2 the field is in `bad`, cut to fit with its NUL, and
#: for both the rest of the line is consumed, as `input()` had read it whole.
#: Returns -1 where the input ended before a line began.
INPUT_INTS = """{
    int c = ppy_rt_next(1);
    if (c == -1 && !continuing) {
        *more = 0;
        return -1;
    }
    int64_t count = 0;
    for (;;) {
        while (c != -1 && c != '\\n' && IS_SPACE) {
            ppy_rt_next(0);
            c = ppy_rt_next(1);
        }
        if (c == -1) {
            *more = 0;
            return count;
        }
        if (c == '\\n') {
            ppy_rt_next(0);
            *more = 0;
            return count;
        }
        if (count == capacity) {
            *more = 1;
            return count;
        }
        int64_t seen = 0;
        int negative = 0;
        int digits = 0;
        int failed = 0;
        int last_underscore = 1;
        uint64_t magnitude = 0;
        if (c == '-' || c == '+') {
            negative = c == '-';
            bad[seen++] = (int8_t)c;
            ppy_rt_next(0);
            c = ppy_rt_next(1);
        }
        while (c != -1 && !IS_SPACE) {
            if (seen < bad_capacity - 1) {
                bad[seen] = (int8_t)c;
            }
            seen++;
            if (c == '_') {
                if (last_underscore) {
                    failed = 2;
                }
                last_underscore = 1;
            } else if (c < '0' || c > '9') {
                failed = 2;
            } else {
                uint64_t digit = (uint64_t)(c - '0');
                if (magnitude > 922337203685477580ULL
                    || (magnitude == 922337203685477580ULL && digit > (uint64_t)(7 + negative))) {
                    failed = failed ? failed : 3;
                } else if (!failed) {
                    magnitude = magnitude * 10 + digit;
                }
                digits++;
                last_underscore = 0;
            }
            ppy_rt_next(0);
            c = ppy_rt_next(1);
        }
        if (digits == 0 || last_underscore) {
            failed = 2;
        }
        if (failed) {
            bad[seen < bad_capacity - 1 ? seen : bad_capacity - 1] = 0;
            while (c != -1 && c != '\\n') {
                ppy_rt_next(0);
                c = ppy_rt_next(1);
            }
            if (c == '\\n') {
                ppy_rt_next(0);
            }
            *more = (int8_t)failed;
            return count;
        }
        data[count++] = negative ? (int64_t)(0 - magnitude) : (int64_t)magnitude;
    }
}""".replace("IS_SPACE", _IS_SPACE)

#: `ppy.input[tuple[int, ...]]()`: a line of integers into the `room` slots at
#: `data`, every field checked, the line consumed whole. Returns how many
#: fields the line held (more than `room` is answered by the count, the rest
#: having gone into scratch), -1 where the input ended before a line, -2 for
#: a field that is not an integer (in `bad`), -3 for one outside 64 bits.
INPUT_FIXED = """{
    int64_t drain[64];
    int8_t more = 0;
    int64_t got = ppy_rt_input_ints(data, room, bad, bad_capacity, 0, &more);
    if (got < 0) {
        return -1;
    }
    while (more == 1) {
        got += ppy_rt_input_ints(drain, 64, bad, bad_capacity, 1, &more);
    }
    if (more >= 2) {
        return -(int64_t)more;
    }
    return got;
}"""

#: What a standalone binary does where Python would raise: say so and stop.
_STANDALONE_FAIL = """{
    fputs(message, stderr);
    fputc('\\n', stderr);
    exit(1);
}"""

#: `ppy.input[int]()` with no interpreter: one line, read as Python's `int()`
#: reads it -- surrounding whitespace, a sign, digits with single underscores
#: between them -- or an error that ends the program.
INPUT_INT = """{
    static int8_t line[1 << 16];
    int8_t more = 0;
    int64_t length = ppy_rt_read_line(line, (int64_t)sizeof line, 0, &more);
    if (length < 0) {
        ppy_rt_fail("ppy: EOFError: the input ended where a line was expected");
    }
    if (more) {
        ppy_rt_fail("ppy: ValueError: a line read as an integer is longer than the room for it");
    }
    int64_t start = 0;
    int64_t end = length;
    while (start < end && IS_SPACE_AT(start)) {
        start++;
    }
    while (end > start && IS_SPACE_AT(end - 1)) {
        end--;
    }
    int negative = 0;
    if (start < end && (line[start] == '-' || line[start] == '+')) {
        negative = line[start] == '-';
        start++;
    }
    int digits = 0;
    int last_underscore = 1;
    uint64_t magnitude = 0;
    for (int64_t i = start; i < end; i++) {
        int c = line[i];
        if (c == '_') {
            if (last_underscore) {
                ppy_rt_fail("ppy: ValueError: invalid literal for int()");
            }
            last_underscore = 1;
            continue;
        }
        if (c < '0' || c > '9') {
            ppy_rt_fail("ppy: ValueError: invalid literal for int()");
        }
        uint64_t digit = (uint64_t)(c - '0');
        if (magnitude > 922337203685477580ULL
            || (magnitude == 922337203685477580ULL && digit > (uint64_t)(7 + negative))) {
            ppy_rt_fail("ppy: OverflowError: the integer does not fit in 64 bits");
        }
        magnitude = magnitude * 10 + digit;
        digits++;
        last_underscore = 0;
    }
    if (digits == 0 || last_underscore) {
        ppy_rt_fail("ppy: ValueError: invalid literal for int()");
    }
    return negative ? (int64_t)(0 - magnitude) : (int64_t)magnitude;
}""".replace("IS_SPACE_AT(start)", _IS_SPACE.replace("c ==", "line[start] ==")).replace(
    "IS_SPACE_AT(end - 1)", _IS_SPACE.replace("c ==", "line[end - 1] ==")
)

#: `ppy.scan[Buffer[int]](n)` with no interpreter: `count` integers into
#: `data`, fewer where the input ends, an error that ends the program where
#: a token is not an integer.
FILL_INTS = """{
    int8_t bad[64];
    int64_t got = ppy_rt_read_ints(data, count, bad, (int64_t)sizeof bad);
    if (got < 0) {
        ppy_rt_fail("ppy: ValueError: expected an integer token");
    }
    return got;
}"""

#: `ppy.scan[int]()` with no interpreter.
SCAN_INT = """{
    int64_t slot = 0;
    int8_t bad[64];
    int64_t got = ppy_rt_read_ints(&slot, 1, bad, (int64_t)sizeof bad);
    if (got < 0) {
        ppy_rt_fail("ppy: ValueError: expected an integer token");
    }
    if (got == 0) {
        ppy_rt_fail("ppy: EOFError: the input ended where an integer was expected");
    }
    return slot;
}"""

#: name -> (result, parameters, body, headers, needs). The order is the
#: order a C file defines them in.
FUNCTIONS: dict[str, tuple[str, tuple[str, ...], str, tuple[str, ...], tuple[str, ...]]] = {
    "ppy_rt_next": ("int", ("int peek",), NEXT, ("stdio.h",), ()),
    "ppy_rt_read_token": (
        "int64_t",
        ("int8_t *data", "int64_t capacity", "int8_t continuing", "int8_t *more"),
        READ_TOKEN,
        ("stdint.h",),
        ("ppy_rt_next",),
    ),
    "ppy_rt_read_token_wide": (
        "int64_t",
        ("int64_t *data", "int64_t capacity", "int8_t continuing", "int8_t *more"),
        READ_TOKEN_WIDE,
        ("stdint.h",),
        ("ppy_rt_next",),
    ),
    "ppy_rt_read_line": (
        "int64_t",
        ("int8_t *data", "int64_t capacity", "int8_t continuing", "int8_t *more"),
        READ_LINE,
        ("stdint.h",),
        ("ppy_rt_next",),
    ),
    "ppy_rt_read_ints": (
        "int64_t",
        ("int64_t *data", "int64_t capacity", "int8_t *bad", "int64_t bad_capacity"),
        READ_INTS,
        ("stdint.h",),
        ("ppy_rt_next",),
    ),
    "ppy_rt_input_ints": (
        "int64_t",
        (
            "int64_t *data",
            "int64_t capacity",
            "int8_t *bad",
            "int64_t bad_capacity",
            "int8_t continuing",
            "int8_t *more",
        ),
        INPUT_INTS,
        ("stdint.h",),
        ("ppy_rt_next",),
    ),
    "ppy_rt_input_fixed": (
        "int64_t",
        ("int64_t *data", "int64_t room", "int8_t *bad", "int64_t bad_capacity"),
        INPUT_FIXED,
        ("stdint.h",),
        ("ppy_rt_input_ints",),
    ),
    "ppy_rt_fail": (
        "void",
        ("const char *message",),
        _STANDALONE_FAIL,
        ("stdio.h", "stdlib.h"),
        (),
    ),
    "ppy_rt_input_int": (
        "int64_t",
        (),
        INPUT_INT,
        ("stdint.h",),
        ("ppy_rt_read_line", "ppy_rt_fail"),
    ),
    "ppy_rt_fill_ints": (
        "int64_t",
        ("int64_t *data", "int64_t count"),
        FILL_INTS,
        ("stdint.h",),
        ("ppy_rt_read_ints", "ppy_rt_fail"),
    ),
    "ppy_rt_scan_int": (
        "int64_t",
        (),
        SCAN_INT,
        ("stdint.h",),
        ("ppy_rt_read_ints", "ppy_rt_fail"),
    ),
}

#: The functions that own process state.
STATEFUL = frozenset({"ppy_rt_next", "ppy_rt_input_int"})

#: The functions only the scanner itself calls: private to whichever unit carries them.
INTERNAL = frozenset({"ppy_rt_next"})

#: What the runtime's shared library carries: the scanner without the
#: standalone-only failure path, which has Python's exceptions instead.
LIBRARY = (
    "ppy_rt_next",
    "ppy_rt_read_token",
    "ppy_rt_read_token_wide",
    "ppy_rt_read_line",
    "ppy_rt_read_ints",
    "ppy_rt_input_ints",
    "ppy_rt_input_fixed",
)


def library_source() -> str:
    """The reader `ppy._io` compiles once and loads with ctypes."""
    lines = ["#include <stdint.h>", "#include <stdio.h>", ""]
    for name in LIBRARY:
        result, parameters, body, _headers, _needs = FUNCTIONS[name]
        storage = "static " if name in INTERNAL else ""
        spelled = result if result.endswith("*") else f"{result} "
        lines.append(f"{storage}{spelled}{name}({', '.join(parameters) or 'void'}) {body}")
        lines.append("")
    return "\n".join(lines)
