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


_NEXT = """{
    static char room[1 << 16];
    static long filled = 0;
    static long position = 0;
    if (position == filled) {
        filled = (long)fread(room, 1, sizeof room, stdin);
        if (filled <= 0) {
            return -1;
        }
        position = 0;
    }
    return (unsigned char)room[position++];
}"""

_ALLOC = """{
    if (count < 0) {
        /* CPython raises ValueError here; a standalone binary has no
         * exception, so it says what happened and stops. */
        fputs("ppy: a buffer cannot hold fewer than no elements\\n", stderr);
        exit(1);
    }
    void *room = calloc(count > 0 ? (size_t)count : 1, (size_t)width);
    if (room == NULL) {
        fputs("ppy: out of memory\\n", stderr);
        exit(1);
    }
    return (int64_t *)room;
}"""

_READ_INTS = """{
    int64_t count = 0;
    while (count < capacity) {
        int c = ppy_rt_next();
        while (c != -1 && (c < '0' || c > '9') && c != '-') {
            c = ppy_rt_next();
        }
        if (c == -1) {
            break;
        }
        int negative = 0;
        if (c == '-') {
            negative = 1;
            c = ppy_rt_next();
        }
        int64_t value = 0;
        while (c >= '0' && c <= '9') {
            value = value * 10 + (c - '0');
            c = ppy_rt_next();
        }
        data[count++] = negative ? -value : value;
    }
    return count;
}"""

_READ_INT = """{
    int c = ppy_rt_next();
    while (c != -1 && (c < '0' || c > '9') && c != '-') {
        c = ppy_rt_next();
    }
    if (c == -1) {
        return 0;
    }
    int negative = 0;
    if (c == '-') {
        negative = 1;
        c = ppy_rt_next();
    }
    int64_t value = 0;
    while (c >= '0' && c <= '9') {
        value = value * 10 + (c - '0');
        c = ppy_rt_next();
    }
    return negative ? -value : value;
}"""

SHIMS: dict[str, Shim] = {
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
    "ppy_rt_next": Shim("int", (), _NEXT, ("stdio.h",), stateful=True),
    "ppy_rt_alloc": Shim(
        "int64_t *", ("int64_t count", "int64_t width"), _ALLOC, ("stdio.h", "stdlib.h")
    ),
    "ppy_rt_read_ints": Shim(
        "int64_t",
        ("int64_t *data", "int64_t capacity"),
        _READ_INTS,
        ("stdio.h",),
        needs=("ppy_rt_next",),
        stateful=True,
    ),
    "ppy_rt_read_int": Shim(
        "int64_t", (), _READ_INT, ("stdio.h",), needs=("ppy_rt_next",), stateful=True
    ),
}

_COMMENTS = {
    "ppy_rt_next": (
        "/* `ppy.input[int]()` with no interpreter under it: the same buffered scan\n"
        " * the runtime reader does, reading standard input directly. At end of input\n"
        " * it answers 0, because a standalone binary has no exception to raise. */\n"
    ),
    "ppy_rt_alloc": (
        "/* A buffer a standalone program makes for itself. There is no interpreter\n"
        " * to own it, and a program that exits is the only lifetime that matters. */\n"
    ),
}


def definition(name: str, storage: str = "") -> str:
    """The C definition of one shim, with the storage class asked for."""
    shim = SHIMS[name]
    internal = "static " if name == "ppy_rt_next" and not storage else storage
    return f"{_COMMENTS.get(name, '')}{shim.prototype(name, internal)} {shim.source}\n"


def support_source() -> str:
    """The whole support library as one C file, for a standalone build."""
    headers = sorted({h for shim in SHIMS.values() for h in shim.headers} | {"stdint.h"})
    lines = [f"#include <{h}>" for h in headers] + [""]
    lines.extend(definition(name) for name in SHIMS)
    return "\n".join(lines).rstrip("\n") + "\n"


def program_main(symbol: str, fputs: str = "fputs") -> str:
    """A `main` that runs the program's entry and fails where a guard fails.

    There is no Python to fall back to, so a status other than success is
    the process's exit, said once on standard error.
    """
    return f"""int main(void) {{
    int64_t out = 0;
    int32_t status = {symbol}(&out);
    if (status != 0) {{
        {fputs}("ppy: a native guard failed and there is no Python to fall back to\\n", stderr);
        return 70;
    }}
    return 0;
}}
"""
