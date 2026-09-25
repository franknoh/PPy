"""The C runtime support a program with no interpreter under it links to.

Each shim is one C function the frontend calls by name from a
`core.call_extern`: printing, reading standard input, and the buffer a
standalone program makes for itself. The LLVM road compiles the whole set
into a standalone binary; the C backend carries into its output exactly
the shims a unit calls, and a header-only unit refuses the ones that keep
state, because a header cannot own a process's input buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

from ppy_runtime import collections as _collections
from ppy_runtime.scanner import FUNCTIONS, INTERNAL, STATEFUL

__all__ = ["SHIMS", "Shim", "definition", "program_main", "support_source"]


@dataclass(frozen=True, slots=True)
class Shim:
    """One runtime function: its C signature, its definition, what it needs."""

    result: str
    parameters: tuple[str, ...]
    source: str
    headers: tuple[str, ...] = ()
    #: The names of other shims this one's definition calls.
    needs: tuple[str, ...] = ()
    #: The function owns process state (a static input buffer): a header
    #: included from two translation units would own it twice.
    stateful: bool = False

    def prototype(self, name: str, storage: str = "") -> str:
        result = self.result if self.result.endswith("*") else f"{self.result} "
        return f"{storage}{result}{name}({', '.join(self.parameters) or 'void'})"


_ALLOC = """{
    if (count < 0) {
        /* CPython raises ValueError here; a standalone binary has no
         * exception, so it says what happened and stops. */
        fflush(stdout);
        fputs("ValueError: a buffer cannot hold fewer than no elements\\n", stderr);
        exit(1);
    }
    void *room = calloc(count > 0 ? (size_t)count : 1, (size_t)width);
    if (room == NULL) {
        fflush(stdout);
        fputs("MemoryError\\n", stderr);
        exit(1);
    }
    return (int8_t *)room;
}"""

#: A failed guard in a program with no Python under it: the line CPython's
#: traceback would end with, `{0}` and on filled in, and CPython's status.
_RAISE = """{
    int64_t values[4];
    values[0] = a;
    values[1] = b;
    values[2] = c;
    values[3] = d;
    fflush(stdout);
    for (const char *at = text; *at != '\\0'; at++) {
        if (at[0] == '{' && at[1] >= '0' && at[1] < '0' + count && at[2] == '}') {
            fprintf(stderr, "%" PRId64, values[at[1] - '0']);
            at += 2;
            continue;
        }
        fputc(*at, stderr);
    }
    fputc('\\n', stderr);
    exit(1);
}"""

SHIMS: dict[str, Shim] = {
    "ppy_rt_raise": Shim(
        "void",
        (
            "const char *text",
            "int64_t count",
            "int64_t a",
            "int64_t b",
            "int64_t c",
            "int64_t d",
        ),
        _RAISE,
        ("inttypes.h", "stdio.h", "stdlib.h"),
    ),
    "ppy_rt_print_i64": Shim(
        "void", ("int64_t value",), '{ printf("%" PRId64, value); }', ("inttypes.h", "stdio.h")
    ),
    "ppy_rt_print_bool": Shim(
        "void", ("int8_t value",), '{ fputs(value ? "True" : "False", stdout); }', ("stdio.h",)
    ),
    "ppy_rt_print_str": Shim(
        "void",
        ("const char *text", "int64_t length"),
        "{\n    fwrite(text, 1, (size_t)length, stdout);\n}",
        ("stdio.h",),
    ),
    "ppy_rt_print_sep": Shim("void", (), "{ fputc(' ', stdout); }", ("stdio.h",)),
    "ppy_rt_print_nl": Shim("void", (), "{ fputc('\\n', stdout); }", ("stdio.h",)),
    "ppy_rt_flush_stdout": Shim("void", (), "{ fflush(stdout); }", ("stdio.h",)),
    "ppy_rt_alloc": Shim(
        "int8_t *", ("int64_t count", "int64_t width"), _ALLOC, ("stdio.h", "stdlib.h")
    ),
}
# The scanner, the same text `ppy._io` compiles for a program under CPython.
SHIMS.update(
    {
        name: Shim(result, parameters, body, headers, needs=needs, stateful=name in STATEFUL)
        for name, (result, parameters, body, headers, needs) in FUNCTIONS.items()
    }
)

# The collections: `ppy.Vec` and the rest, the text `ppy run` loads compiled.
SHIMS.update(
    {
        name: Shim(result, parameters, body, _collections.HEADERS, needs=needs)
        for name, (result, parameters, body, needs) in _collections.FUNCTIONS.items()
    }
)

#: `ppy.input[str]()` and `input()` with no interpreter: one line, its newline
#: (and a carriage return before it) removed, read in pieces however long.
_INPUT_STR = """{
    int8_t chunk[4096];
    int8_t more = 0;
    int64_t length = ppy_rt_read_line(chunk, (int64_t)sizeof chunk, 0, &more);
    if (length < 0) {
        ppy_rt_fail("ppy: EOFError: EOF when reading a line");
    }
    int8_t *made = ppy_str_builder(length);
    ppy_str_add_bytes(made, chunk, length);
    while (more) {
        length = ppy_rt_read_line(chunk, (int64_t)sizeof chunk, 1, &more);
        if (length <= 0) {
            break;
        }
        ppy_str_add_bytes(made, chunk, length);
    }
    int64_t *header = (int64_t *)made;
    if (header[0] > 0 && ppy_str_raw(made)[header[0] - 1] == '\\r') {
        header[0]--;
        header[3]--;
    }
    if (!ppy_str_valid(ppy_str_raw(made), header[0])) {
        ppy_rt_fail("ppy: UnicodeDecodeError: the line is not UTF-8");
    }
    return ppy_str_finish(made);
}"""

#: `ppy.scan[str]()` with no interpreter: the next whitespace-delimited token.
_SCAN_STR = """{
    int8_t chunk[4096];
    int8_t more = 0;
    int64_t length = ppy_rt_read_token(chunk, (int64_t)sizeof chunk, 0, &more);
    if (length <= 0 && !more) {
        ppy_rt_fail("ppy: EOFError: the input ended where a token was expected");
    }
    int8_t *made = ppy_str_builder(length);
    ppy_str_add_bytes(made, chunk, length);
    while (more) {
        length = ppy_rt_read_token(chunk, (int64_t)sizeof chunk, 1, &more);
        ppy_str_add_bytes(made, chunk, length);
    }
    if (!ppy_str_valid(ppy_str_raw(made), ((int64_t *)made)[0])) {
        ppy_rt_fail("ppy: UnicodeDecodeError: the token is not UTF-8");
    }
    return ppy_str_finish(made);
}"""

_STRING_READ_NEEDS = (
    "ppy_rt_fail",
    "ppy_str_builder",
    "ppy_str_add_bytes",
    "ppy_str_raw",
    "ppy_str_valid",
    "ppy_str_finish",
)

SHIMS.update(
    {
        "ppy_rt_input_str": Shim(
            "int8_t *",
            (),
            _INPUT_STR,
            ("stdint.h",),
            needs=("ppy_rt_read_line", *_STRING_READ_NEEDS),
        ),
        "ppy_rt_scan_str": Shim(
            "int8_t *",
            (),
            _SCAN_STR,
            ("stdint.h",),
            needs=("ppy_rt_read_token", *_STRING_READ_NEEDS),
        ),
    }
)

_COMMENTS = {
    "ppy_rt_next": (
        "/* The buffered byte source behind `ppy.input` and `ppy.scan` with no\n"
        " * interpreter under them: the same scanner the runtime reader compiles,\n"
        " * reading standard input directly. */\n"
    ),
    "ppy_rt_fail": (
        "/* Where Python would raise, a standalone binary says what happened and\n"
        " * stops: there is no exception to raise and no caller to catch it. */\n"
    ),
    "ppy_rt_raise": (
        "/* A failed guard, where there is no Python to fall back to: the line\n"
        " * CPython's traceback would end with, and CPython's exit status. */\n"
    ),
    "ppy_rt_alloc": (
        "/* A buffer a standalone program makes for itself. There is no interpreter\n"
        " * to own it, and a program that exits is the only lifetime that matters. */\n"
    ),
}


def definition(name: str, storage: str = "") -> str:
    """The C definition of one shim, with the storage class asked for."""
    shim = SHIMS[name]
    internal = "static " if name in INTERNAL and not storage else storage
    return f"{_COMMENTS.get(name, '')}{shim.prototype(name, internal)} {shim.source}\n"


def support_source() -> str:
    """The whole support library as one C file, for a standalone build."""
    headers = sorted({h for shim in SHIMS.values() for h in shim.headers} | {"stdint.h"})
    lines = [f"#include <{h}>" for h in headers] + [""]
    # The collections and the strings call each other, so they are declared first.
    lines.extend(_collections.prototype(name) for name in _collections.FUNCTIONS)
    lines.append("")
    lines.extend(definition(name) for name in SHIMS)
    return "\n".join(lines).rstrip("\n") + "\n"


def program_main(symbol: str, fputs: str = "fputs", *, collect: bool = False) -> str:
    """A `main` that runs the program's entry and fails where a guard fails.

    A guard that knows what CPython would raise says so and exits where it
    fails (`ppy_rt_raise`). One that does not reaches here with a status,
    which is said once on standard error and ends the process with
    CPython's status for an uncaught exception.
    """
    # The cycles still waiting for a collection are freed before the process
    # ends, so a leak checker sees only what nothing can free.
    collected = "    ppy_coll_collect();\n" if collect else ""
    return f"""int main(void) {{
    int64_t out = 0;
    int32_t status = {symbol}(&out);
{collected}    if (status != 0) {{
        {fputs}("RuntimeError: a native guard failed with no Python to fall back to\\n", stderr);
        return 1;
    }}
    return 0;
}}
"""
