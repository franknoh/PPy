"""The collections runtime: `Vec`, `Deque`, `Heap`, and `MaxHeap` in C.

Native code holds a collection as a handle, a pointer to four words:

```text
[0] length   [1] capacity   [2] the elements   [3] where a deque starts
```

The elements are eight bytes each, `int64_t` or `double`, so every function
that reads or writes one comes in two spellings, `_i64` and `_f64`. The
functions check nothing: the code that calls them guards an index or an
empty pop first, which is what lets a failed check fall back to Python
under `ppy run` and stop a standalone binary with the reason.

One text, three homes, as the scanner has: a standalone binary and emitted
C carry the functions they call as shims, and `ppy run` loads them from a
shared library compiled from the same text on first use.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import threading
from pathlib import Path

__all__ = ["FUNCTIONS", "library_path", "library_source", "source_path"]

#: The element kinds, and the C type of each.
ELEMENTS = {"i64": "int64_t", "f64": "double"}

_FAIL = """
        fputs("ppy: MemoryError: a collection could not grow\\n", stderr);
        exit(1);"""

#: A header and room for `count` elements, all zero.
NEW = (
    """{
    int64_t *header = (int64_t *)calloc(4, sizeof(int64_t));
    void *room = calloc(count > 0 ? (size_t)count : 1, 8);
    if (header == NULL || room == NULL) {"""
    + _FAIL
    + """
    }
    header[0] = count;
    header[1] = count > 0 ? count : 1;
    header[2] = (int64_t)(intptr_t)room;
    return (int8_t *)header;
}"""
)

#: Room for one more element: the capacity doubles, and a deque's elements
#: are laid out from its start again.
RESERVE = (
    """{
    int64_t *header = (int64_t *)handle;
    if (header[0] < header[1]) {
        return;
    }
    int64_t capacity = header[1] * 2;
    int64_t *old = (int64_t *)(intptr_t)header[2];
    int64_t *room = (int64_t *)calloc((size_t)capacity, 8);
    if (room == NULL) {"""
    + _FAIL
    + """
    }
    for (int64_t i = 0; i < header[0]; i++) {
        room[i] = old[(header[3] + i) % header[1]];
    }
    free(old);
    header[1] = capacity;
    header[2] = (int64_t)(intptr_t)room;
    header[3] = 0;
}"""
)

LENGTH = """{
    return ((int64_t *)handle)[0];
}"""

CLEAR = """{
    int64_t *header = (int64_t *)handle;
    header[0] = 0;
    header[3] = 0;
}"""

FREE = """{
    if (handle != NULL) {
        int64_t *header = (int64_t *)handle;
        free((void *)(intptr_t)header[2]);
        free(header);
    }
}"""


def _typed(element: str) -> dict[str, tuple[str, tuple[str, ...], str, tuple[str, ...]]]:
    """The functions that read or write an element of one kind."""
    c = ELEMENTS[element]
    at = f"(({c} *)(intptr_t)header[2])"
    slot = f"{at}[(header[3] + index) % header[1]]"
    less = "a < b"
    functions = {
        f"ppy_coll_get_{element}": (
            c,
            ("int8_t *handle", "int64_t index"),
            f"{{\n    int64_t *header = (int64_t *)handle;\n    return {slot};\n}}",
            (),
        ),
        f"ppy_coll_set_{element}": (
            "void",
            ("int8_t *handle", "int64_t index", f"{c} value"),
            f"{{\n    int64_t *header = (int64_t *)handle;\n    {slot} = value;\n}}",
            (),
        ),
        f"ppy_coll_push_back_{element}": (
            "void",
            ("int8_t *handle", f"{c} value"),
            f"""{{
    ppy_coll_reserve(handle);
    int64_t *header = (int64_t *)handle;
    {at}[(header[3] + header[0]) % header[1]] = value;
    header[0]++;
}}""",
            ("ppy_coll_reserve",),
        ),
        f"ppy_coll_push_front_{element}": (
            "void",
            ("int8_t *handle", f"{c} value"),
            f"""{{
    ppy_coll_reserve(handle);
    int64_t *header = (int64_t *)handle;
    header[3] = (header[3] + header[1] - 1) % header[1];
    {at}[header[3]] = value;
    header[0]++;
}}""",
            ("ppy_coll_reserve",),
        ),
        f"ppy_coll_pop_back_{element}": (
            c,
            ("int8_t *handle",),
            f"""{{
    int64_t *header = (int64_t *)handle;
    header[0]--;
    return {at}[(header[3] + header[0]) % header[1]];
}}""",
            (),
        ),
        f"ppy_coll_pop_front_{element}": (
            c,
            ("int8_t *handle",),
            f"""{{
    int64_t *header = (int64_t *)handle;
    {c} value = {at}[header[3]];
    header[3] = (header[3] + 1) % header[1];
    header[0]--;
    return value;
}}""",
            (),
        ),
        f"ppy_coll_reverse_{element}": (
            "void",
            ("int8_t *handle",),
            f"""{{
    int64_t *header = (int64_t *)handle;
    for (int64_t i = 0, j = header[0] - 1; i < j; i++, j--) {{
        int64_t index = i;
        {c} low = {slot};
        index = j;
        {c} high = {slot};
        {slot} = low;
        index = i;
        {slot} = high;
    }}
}}""",
            (),
        ),
        # A stable merge sort, so equal elements keep their order as
        # Python's sort keeps it: 0.0 and -0.0 compare equal and print apart.
        f"ppy_coll_sort_{element}": (
            "void",
            ("int8_t *handle",),
            f"""{{
    int64_t *header = (int64_t *)handle;
    int64_t n = header[0];
    if (n < 2) {{
        return;
    }}
    {c} *items = ({c} *)calloc((size_t)n, sizeof({c}));
    {c} *spare = ({c} *)calloc((size_t)n, sizeof({c}));
    if (items == NULL || spare == NULL) {{{_FAIL}
    }}
    for (int64_t index = 0; index < n; index++) {{
        items[index] = {slot};
    }}
    for (int64_t width = 1; width < n; width *= 2) {{
        for (int64_t low = 0; low < n; low += 2 * width) {{
            int64_t middle = low + width < n ? low + width : n;
            int64_t high = low + 2 * width < n ? low + 2 * width : n;
            int64_t i = low, j = middle, k = low;
            while (i < middle && j < high) {{
                {c} a = items[j];
                {c} b = items[i];
                spare[k++] = {less} ? items[j++] : items[i++];
            }}
            while (i < middle) {{
                spare[k++] = items[i++];
            }}
            while (j < high) {{
                spare[k++] = items[j++];
            }}
        }}
        {c} *swap = items;
        items = spare;
        spare = swap;
    }}
    free({at});
    free(spare);
    header[1] = n;
    header[2] = (int64_t)(intptr_t)items;
    header[3] = 0;
}}""",
            (),
        ),
    }
    for order, better in (("min", "a < b"), ("max", "a > b")):
        functions[f"ppy_heap_push_{order}_{element}"] = (
            "void",
            ("int8_t *handle", f"{c} value"),
            f"""{{
    ppy_coll_reserve(handle);
    int64_t *header = (int64_t *)handle;
    {c} *items = {at};
    int64_t i = header[0]++;
    while (i > 0) {{
        int64_t parent = (i - 1) / 2;
        {c} a = value;
        {c} b = items[parent];
        if (!({better})) {{
            break;
        }}
        items[i] = items[parent];
        i = parent;
    }}
    items[i] = value;
}}""",
            ("ppy_coll_reserve",),
        )
        functions[f"ppy_heap_pop_{order}_{element}"] = (
            c,
            ("int8_t *handle",),
            f"""{{
    int64_t *header = (int64_t *)handle;
    {c} *items = {at};
    {c} top = items[0];
    {c} last = items[--header[0]];
    int64_t n = header[0];
    int64_t i = 0;
    while (1) {{
        int64_t child = 2 * i + 1;
        if (child >= n) {{
            break;
        }}
        if (child + 1 < n) {{
            {c} a = items[child + 1];
            {c} b = items[child];
            if ({better}) {{
                child++;
            }}
        }}
        {c} a = items[child];
        {c} b = last;
        if (!({better})) {{
            break;
        }}
        items[i] = items[child];
        i = child;
    }}
    if (n > 0) {{
        items[i] = last;
    }}
    return top;
}}""",
            (),
        )
    return functions


#: name -> (result, parameters, body, needs). The order is the order a C file
#: defines them in: a function after everything it calls.
FUNCTIONS: dict[str, tuple[str, tuple[str, ...], str, tuple[str, ...]]] = {
    "ppy_coll_none": ("int8_t *", (), "{\n    return NULL;\n}", ()),
    "ppy_coll_new": ("int8_t *", ("int64_t count",), NEW, ()),
    "ppy_coll_reserve": ("void", ("int8_t *handle",), RESERVE, ()),
    "ppy_coll_len": ("int64_t", ("int8_t *handle",), LENGTH, ()),
    "ppy_coll_clear": ("void", ("int8_t *handle",), CLEAR, ()),
    "ppy_coll_free": ("void", ("int8_t *handle",), FREE, ()),
}
for _element in ELEMENTS:
    FUNCTIONS.update(_typed(_element))

#: What every function needs from the C library.
HEADERS = ("stdint.h", "stdio.h", "stdlib.h")


def library_source() -> str:
    """The whole runtime as one C file."""
    lines = [f"#include <{header}>" for header in HEADERS] + [""]
    for name, (result, parameters, body, _needs) in FUNCTIONS.items():
        spelled = result if result.endswith("*") else f"{result} "
        lines.append(f"{spelled}{name}({', '.join(parameters) or 'void'}) {body}")
        lines.append("")
    return "\n".join(lines)


def _cache_directory() -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root) / "ppy" / "collections"


_lock = threading.Lock()
_library: Path | bool | None = None


def source_path() -> Path:
    """The runtime's text on disk, for a build that compiles it in."""
    text = library_source()
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    target = _cache_directory() / f"ppy_collections-{digest}.c"
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        draft = target.with_name(f"{target.name}.{os.getpid()}.part")
        draft.write_text(text, encoding="utf-8")
        draft.replace(target)
    return target


def library_path() -> Path | None:
    """The runtime compiled for `ppy run`, built into the cache on first use."""
    global _library  # noqa: PLW0603 - one runtime per process
    with _lock:
        if _library is not None:
            return _library or None
        _library = False
        from .aio import compiler  # pylint: disable=import-outside-toplevel

        cc = compiler()
        if cc is None:
            return None
        source = source_path()
        target = source.with_suffix(".so")
        if not target.is_file():
            draft = target.with_name(f"{target.name}.{os.getpid()}.part")
            command = [cc, "-std=c11", "-O2", "-shared", "-fPIC", "-o", str(draft), str(source)]
            done = subprocess.run(command, capture_output=True, text=True, check=False)
            if done.returncode != 0:
                draft.unlink(missing_ok=True)
                return None
            draft.replace(target)
        _library = target
        return target
