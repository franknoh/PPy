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
    "ppy_rt_alloc": Shim(
        "int64_t *", ("int64_t count", "int64_t width"), _ALLOC, ("stdio.h", "stdlib.h")
    ),
}
# The scanner, the same text `ppy._io` compiles for a program under CPython.
SHIMS.update(
    {
        name: Shim(result, parameters, body, headers, needs=needs, stateful=name in STATEFUL)
        for name, (result, parameters, body, headers, needs) in FUNCTIONS.items()
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
