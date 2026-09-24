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


#: name -> (result, parameters, body, needs).
Definitions = dict[str, tuple[str, tuple[str, ...], str, tuple[str, ...]]]


def _typed(element: str) -> Definitions:
    """The functions that read or write an element of one kind."""
    c = ELEMENTS[element]
    at = f"(({c} *)(intptr_t)header[2])"
    slot = f"{at}[(header[3] + index) % header[1]]"
    less = "a < b"
    functions: Definitions = {
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


def _load(c: str, where: str) -> str:
    return f"{c} value;\n    memcpy(&value, &{where}, 8);\n    return value;"


def _store(c: str, where: str) -> str:
    return f"{c} stored = value;\n    memcpy(&{where}, &stored, 8);"


# -- LinkedList: nodes of four words, named by index ----------------------------
#
# header: [0] length  [1] capacity  [2] nodes  [3] head  [4] tail
#         [5] the last freed node (-1)  [6] nodes ever made
# node:   [0] value   [1] prev      [2] next   [3] alive
# A freed node's `next` links the free stack, so ids come back last-freed-first,
# which is the order the reference hands them out in.

LIST_NEW = (
    """{
    int64_t *header = (int64_t *)calloc(8, sizeof(int64_t));
    int64_t *nodes = (int64_t *)calloc(4 * 4, sizeof(int64_t));
    if (header == NULL || nodes == NULL) {"""
    + _FAIL
    + """
    }
    header[1] = 4;
    header[2] = (int64_t)(intptr_t)nodes;
    header[3] = -1;
    header[4] = -1;
    header[5] = -1;
    return (int8_t *)header;
}"""
)

LIST_NODE = (
    """{
    int64_t *header = (int64_t *)handle;
    int64_t node = header[5];
    if (node >= 0) {
        int64_t *nodes = (int64_t *)(intptr_t)header[2];
        header[5] = nodes[4 * node + 2];
    } else {
        if (header[6] == header[1]) {
            int64_t capacity = header[1] * 2;
            int64_t *grown = (int64_t *)realloc((void *)(intptr_t)header[2],
                                                (size_t)(4 * capacity) * sizeof(int64_t));
            if (grown == NULL) {"""
    + _FAIL
    + """
            }
            header[1] = capacity;
            header[2] = (int64_t)(intptr_t)grown;
        }
        node = header[6]++;
    }
    int64_t *nodes = (int64_t *)(intptr_t)header[2];
    nodes[4 * node] = bits;
    nodes[4 * node + 1] = before;
    nodes[4 * node + 2] = after;
    nodes[4 * node + 3] = 1;
    if (before == -1) {
        header[3] = node;
    } else {
        nodes[4 * before + 2] = node;
    }
    if (after == -1) {
        header[4] = node;
    } else {
        nodes[4 * after + 1] = node;
    }
    header[0]++;
    return node;
}"""
)

LIST_VALID = """{
    int64_t *header = (int64_t *)handle;
    int64_t *nodes = (int64_t *)(intptr_t)header[2];
    return node >= 0 && node < header[6] && nodes[4 * node + 3] ? 1 : 0;
}"""

LIST_UNLINK = """{
    int64_t *header = (int64_t *)handle;
    int64_t *nodes = (int64_t *)(intptr_t)header[2];
    int64_t before = nodes[4 * node + 1];
    int64_t after = nodes[4 * node + 2];
    if (before == -1) {
        header[3] = after;
    } else {
        nodes[4 * before + 2] = after;
    }
    if (after == -1) {
        header[4] = before;
    } else {
        nodes[4 * after + 1] = before;
    }
    nodes[4 * node + 1] = -1;
    nodes[4 * node + 2] = header[5];
    nodes[4 * node + 3] = 0;
    header[5] = node;
    header[0]--;
    return nodes[4 * node];
}"""

LIST_STEP = """{
    int64_t *header = (int64_t *)handle;
    int64_t *nodes = (int64_t *)(intptr_t)header[2];
    return nodes[4 * node + field];
}"""

LIST_CLEAR = """{
    int64_t *header = (int64_t *)handle;
    header[0] = 0;
    header[3] = -1;
    header[4] = -1;
    header[5] = -1;
    header[6] = 0;
}"""

# -- HashMap and HashSet: entries in insertion order, and an index over them ----
#
# header: [0] live entries  [1] entry capacity  [2] entries  [3] entries used
#         [4] index         [5] index size (a power of two)  [6] version
# entry:  [0] key  [1] value  [2] alive
# index:  -1 empty, -2 a removed entry, else the entry it points at

MAP_HASH = """{
    uint64_t z = (uint64_t)key + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return (int64_t)((z ^ (z >> 31)) & (uint64_t)mask);
}"""

MAP_REINDEX = (
    """{
    int64_t *header = (int64_t *)handle;
    free((void *)(intptr_t)header[4]);
    int64_t *index = (int64_t *)malloc((size_t)size * sizeof(int64_t));
    if (index == NULL) {"""
    + _FAIL
    + """
    }
    for (int64_t i = 0; i < size; i++) {
        index[i] = -1;
    }
    int64_t *entries = (int64_t *)(intptr_t)header[2];
    for (int64_t e = 0; e < header[3]; e++) {
        if (entries[3 * e + 2]) {
            int64_t i = ppy_map_hash(entries[3 * e], size - 1);
            while (index[i] != -1) {
                i = (i + 1) & (size - 1);
            }
            index[i] = e;
        }
    }
    header[4] = (int64_t)(intptr_t)index;
    header[5] = size;
}"""
)

MAP_NEW = (
    """{
    int64_t *header = (int64_t *)calloc(8, sizeof(int64_t));
    int64_t *entries = (int64_t *)calloc(3 * 8, sizeof(int64_t));
    if (header == NULL || entries == NULL) {"""
    + _FAIL
    + """
    }
    header[1] = 8;
    header[2] = (int64_t)(intptr_t)entries;
    ppy_map_reindex((int8_t *)header, 16);
    return (int8_t *)header;
}"""
)

MAP_FIND = """{
    int64_t *header = (int64_t *)handle;
    int64_t *index = (int64_t *)(intptr_t)header[4];
    int64_t *entries = (int64_t *)(intptr_t)header[2];
    int64_t mask = header[5] - 1;
    int64_t i = ppy_map_hash(key, mask);
    while (index[i] != -1) {
        int64_t e = index[i];
        if (e >= 0 && entries[3 * e] == key) {
            return i;
        }
        i = (i + 1) & mask;
    }
    return -1;
}"""

MAP_ENTRY = """{
    int64_t *header = (int64_t *)handle;
    int64_t at = ppy_map_slot(handle, key);
    return at < 0 ? -1 : ((int64_t *)(intptr_t)header[4])[at];
}"""

# A new key: room for one more entry, compacting the removed ones out or
# doubling, then the entry at the end and its slot in the index.
MAP_INSERT = (
    """{
    int64_t *header = (int64_t *)handle;
    int64_t found = ppy_map_find(handle, key);
    if (found >= 0) {
        ((int64_t *)(intptr_t)header[2])[3 * found + 1] = bits;
        return;
    }
    if (header[3] == header[1]) {
        int64_t *entries = (int64_t *)(intptr_t)header[2];
        int64_t kept = 0;
        for (int64_t e = 0; e < header[3]; e++) {
            if (entries[3 * e + 2]) {
                entries[3 * kept] = entries[3 * e];
                entries[3 * kept + 1] = entries[3 * e + 1];
                entries[3 * kept + 2] = 1;
                kept++;
            }
        }
        header[3] = kept;
        if (kept * 2 > header[1]) {
            int64_t capacity = header[1] * 2;
            int64_t *grown = (int64_t *)realloc(entries, (size_t)(3 * capacity) * sizeof(int64_t));
            if (grown == NULL) {"""
    + _FAIL
    + """
            }
            header[1] = capacity;
            header[2] = (int64_t)(intptr_t)grown;
        }
        ppy_map_reindex(handle, header[1] * 2);
    }
    int64_t *entries = (int64_t *)(intptr_t)header[2];
    int64_t e = header[3]++;
    entries[3 * e] = key;
    entries[3 * e + 1] = bits;
    entries[3 * e + 2] = 1;
    int64_t *index = (int64_t *)(intptr_t)header[4];
    int64_t mask = header[5] - 1;
    int64_t i = ppy_map_hash(key, mask);
    while (index[i] >= 0) {
        i = (i + 1) & mask;
    }
    index[i] = e;
    header[0]++;
    header[6]++;
}"""
)

MAP_REMOVE = """{
    int64_t *header = (int64_t *)handle;
    int64_t at = ppy_map_slot(handle, key);
    if (at < 0) {
        return -1;
    }
    int64_t *index = (int64_t *)(intptr_t)header[4];
    int64_t e = index[at];
    index[at] = -2;
    ((int64_t *)(intptr_t)header[2])[3 * e + 2] = 0;
    header[0]--;
    header[6]++;
    return e;
}"""

MAP_FIELD = """{
    int64_t *header = (int64_t *)handle;
    return ((int64_t *)(intptr_t)header[2])[3 * entry + field];
}"""

#: One word of any collection's header.
FIELD = """{
    return ((int64_t *)handle)[field];
}"""

#: The node after `node` while walking the list, or -1: a removed node has none,
#: as the reference's walk ends at one.
LIST_AFTER = """{
    if (!ppy_list_valid(handle, node)) {
        return -1;
    }
    return ppy_list_step(handle, node, 2);
}"""

MAP_CLEAR = """{
    int64_t *header = (int64_t *)handle;
    header[0] = 0;
    header[3] = 0;
    header[6]++;
    ppy_map_reindex(handle, header[5]);
}"""

MAP_FREE = """{
    if (handle != NULL) {
        int64_t *header = (int64_t *)handle;
        free((void *)(intptr_t)header[2]);
        free((void *)(intptr_t)header[4]);
        free(header);
    }
}"""

# -- TreeMap and TreeSet: a treap over an array of nodes --------------------------
#
# header: [0] length  [1] capacity  [2] nodes  [3] root  [4] free list
#         [5] nodes ever made  [6] version  [7] the priority generator's state
# node:   [0] key  [1] value  [2] left  [3] right  [4] priority  [5] next free
# The priorities come from a fixed xorshift sequence, so a program builds the
# same tree every time it runs.

TREE_NEW = (
    """{
    int64_t *header = (int64_t *)calloc(8, sizeof(int64_t));
    int64_t *nodes = (int64_t *)calloc(6 * 8, sizeof(int64_t));
    if (header == NULL || nodes == NULL) {"""
    + _FAIL
    + """
    }
    header[1] = 8;
    header[2] = (int64_t)(intptr_t)nodes;
    header[3] = -1;
    header[4] = -1;
    header[7] = (int64_t)0x2545F4914F6CDD1DULL;
    return (int8_t *)header;
}"""
)

TREE_SPLIT = """{
    int64_t *nodes = (int64_t *)(intptr_t)((int64_t *)handle)[2];
    if (node == -1) {
        *low = -1;
        *high = -1;
        return;
    }
    if (nodes[6 * node] < key) {
        ppy_tree_split(handle, nodes[6 * node + 3], key, &nodes[6 * node + 3], high);
        *low = node;
    } else {
        ppy_tree_split(handle, nodes[6 * node + 2], key, low, &nodes[6 * node + 2]);
        *high = node;
    }
}"""

TREE_MERGE = """{
    int64_t *nodes = (int64_t *)(intptr_t)((int64_t *)handle)[2];
    if (low == -1) {
        return high;
    }
    if (high == -1) {
        return low;
    }
    if (nodes[6 * low + 4] > nodes[6 * high + 4]) {
        nodes[6 * low + 3] = ppy_tree_merge(handle, nodes[6 * low + 3], high);
        return low;
    }
    nodes[6 * high + 2] = ppy_tree_merge(handle, low, nodes[6 * high + 2]);
    return high;
}"""

TREE_ERASE = """{
    int64_t *header = (int64_t *)handle;
    int64_t *nodes = (int64_t *)(intptr_t)header[2];
    if (node == -1) {
        return -1;
    }
    if (nodes[6 * node] == key) {
        int64_t joined = ppy_tree_merge(handle, nodes[6 * node + 2], nodes[6 * node + 3]);
        nodes[6 * node + 5] = header[4];
        header[4] = node;
        return joined;
    }
    if (key < nodes[6 * node]) {
        nodes[6 * node + 2] = ppy_tree_erase(handle, nodes[6 * node + 2], key);
    } else {
        nodes[6 * node + 3] = ppy_tree_erase(handle, nodes[6 * node + 3], key);
    }
    return node;
}"""

TREE_FIND = """{
    int64_t *header = (int64_t *)handle;
    int64_t *nodes = (int64_t *)(intptr_t)header[2];
    int64_t node = header[3];
    while (node != -1 && nodes[6 * node] != key) {
        node = key < nodes[6 * node] ? nodes[6 * node + 2] : nodes[6 * node + 3];
    }
    return node;
}"""

TREE_INSERT = (
    """{
    int64_t *header = (int64_t *)handle;
    int64_t found = ppy_tree_find(handle, key);
    if (found >= 0) {
        ((int64_t *)(intptr_t)header[2])[6 * found + 1] = bits;
        return;
    }
    int64_t node = header[4];
    if (node >= 0) {
        header[4] = ((int64_t *)(intptr_t)header[2])[6 * node + 5];
    } else {
        if (header[5] == header[1]) {
            int64_t capacity = header[1] * 2;
            int64_t *grown = (int64_t *)realloc((void *)(intptr_t)header[2],
                                                (size_t)(6 * capacity) * sizeof(int64_t));
            if (grown == NULL) {"""
    + _FAIL
    + """
            }
            header[1] = capacity;
            header[2] = (int64_t)(intptr_t)grown;
        }
        node = header[5]++;
    }
    uint64_t x = (uint64_t)header[7];
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    header[7] = (int64_t)x;
    int64_t *nodes = (int64_t *)(intptr_t)header[2];
    nodes[6 * node] = key;
    nodes[6 * node + 1] = bits;
    nodes[6 * node + 2] = -1;
    nodes[6 * node + 3] = -1;
    nodes[6 * node + 4] = (int64_t)(x >> 1);
    int64_t low = -1;
    int64_t high = -1;
    ppy_tree_split(handle, header[3], key, &low, &high);
    header[3] = ppy_tree_merge(handle, ppy_tree_merge(handle, low, node), high);
    header[0]++;
    header[6]++;
}"""
)

TREE_REMOVE = """{
    int64_t *header = (int64_t *)handle;
    if (ppy_tree_find(handle, key) < 0) {
        return 0;
    }
    header[3] = ppy_tree_erase(handle, header[3], key);
    header[0]--;
    header[6]++;
    return 1;
}"""

# The nearest key: 0 at most `key`, 1 at least, 2 below, 3 above.
TREE_BOUND = """{
    int64_t *header = (int64_t *)handle;
    int64_t *nodes = (int64_t *)(intptr_t)header[2];
    int64_t node = header[3];
    int64_t best = -1;
    while (node != -1) {
        int64_t at = nodes[6 * node];
        int take = mode == 0 ? at <= key : mode == 1 ? at >= key : mode == 2 ? at < key : at > key;
        if (take) {
            best = node;
        }
        int right = mode == 0 || mode == 2 ? take : !take;
        node = right ? nodes[6 * node + 3] : nodes[6 * node + 2];
    }
    return best;
}"""

TREE_END = """{
    int64_t *header = (int64_t *)handle;
    int64_t *nodes = (int64_t *)(intptr_t)header[2];
    int64_t node = header[3];
    while (node != -1 && nodes[6 * node + 2 + last] != -1) {
        node = nodes[6 * node + 2 + last];
    }
    return node;
}"""

TREE_FIELD = """{
    int64_t *header = (int64_t *)handle;
    return ((int64_t *)(intptr_t)header[2])[6 * node + field];
}"""

TREE_CLEAR = """{
    int64_t *header = (int64_t *)handle;
    header[0] = 0;
    header[3] = -1;
    header[4] = -1;
    header[5] = 0;
    header[6]++;
}"""


def _structures(element: str) -> Definitions:
    """The linked list, map, and tree functions that carry a value of one kind."""
    c = ELEMENTS[element]
    functions: Definitions = {}
    bits = f"{c} stored = value;\n    int64_t bits;\n    memcpy(&bits, &stored, 8);"
    for name, before, after in (
        ("push_back", "((int64_t *)handle)[4]", "-1"),
        ("push_front", "-1", "((int64_t *)handle)[3]"),
        ("insert_after", "node", "ppy_list_step(handle, node, 2)"),
        ("insert_before", "ppy_list_step(handle, node, 1)", "node"),
    ):
        parameters = (
            "int8_t *handle",
            *(("int64_t node",) if "insert" in name else ()),
            f"{c} value",
        )
        functions[f"ppy_list_{name}_{element}"] = (
            "int64_t",
            parameters,
            f"{{\n    {bits}\n    return ppy_list_node(handle, bits, {before}, {after});\n}}",
            ("ppy_list_node", "ppy_list_step"),
        )
    functions[f"ppy_list_remove_{element}"] = (
        c,
        ("int8_t *handle", "int64_t node"),
        f"{{\n    int64_t bits = ppy_list_unlink(handle, node);\n    {_load(c, 'bits')}\n}}",
        ("ppy_list_unlink",),
    )
    functions[f"ppy_list_value_{element}"] = (
        c,
        ("int8_t *handle", "int64_t node"),
        f"{{\n    int64_t bits = ppy_list_step(handle, node, 0);\n    {_load(c, 'bits')}\n}}",
        ("ppy_list_step",),
    )
    functions[f"ppy_list_set_{element}"] = (
        "void",
        ("int8_t *handle", "int64_t node", f"{c} value"),
        (
            "{\n    int64_t *header = (int64_t *)handle;\n"
            f"    {_store(c, '((int64_t *)(intptr_t)header[2])[4 * node]')}\n}}"
        ),
        (),
    )
    functions[f"ppy_map_put_{element}"] = (
        "void",
        ("int8_t *handle", "int64_t key", f"{c} value"),
        f"{{\n    {bits}\n    ppy_map_insert(handle, key, bits);\n}}",
        ("ppy_map_insert",),
    )
    functions[f"ppy_map_value_{element}"] = (
        c,
        ("int8_t *handle", "int64_t entry"),
        f"{{\n    int64_t bits = ppy_map_field(handle, entry, 1);\n    {_load(c, 'bits')}\n}}",
        ("ppy_map_field",),
    )
    functions[f"ppy_tree_put_{element}"] = (
        "void",
        ("int8_t *handle", "int64_t key", f"{c} value"),
        f"{{\n    {bits}\n    ppy_tree_insert(handle, key, bits);\n}}",
        ("ppy_tree_insert",),
    )
    functions[f"ppy_tree_value_{element}"] = (
        c,
        ("int8_t *handle", "int64_t node"),
        f"{{\n    int64_t bits = ppy_tree_field(handle, node, 1);\n    {_load(c, 'bits')}\n}}",
        ("ppy_tree_field",),
    )
    return functions


STRUCTURES: Definitions = {
    "ppy_list_new": ("int8_t *", (), LIST_NEW, ()),
    "ppy_list_node": (
        "int64_t",
        ("int8_t *handle", "int64_t bits", "int64_t before", "int64_t after"),
        LIST_NODE,
        (),
    ),
    "ppy_list_valid": ("int64_t", ("int8_t *handle", "int64_t node"), LIST_VALID, ()),
    "ppy_list_unlink": ("int64_t", ("int8_t *handle", "int64_t node"), LIST_UNLINK, ()),
    "ppy_list_step": (
        "int64_t",
        ("int8_t *handle", "int64_t node", "int64_t field"),
        LIST_STEP,
        (),
    ),
    "ppy_list_after": (
        "int64_t",
        ("int8_t *handle", "int64_t node"),
        LIST_AFTER,
        ("ppy_list_valid", "ppy_list_step"),
    ),
    "ppy_list_clear": ("void", ("int8_t *handle",), LIST_CLEAR, ()),
    "ppy_map_hash": ("int64_t", ("int64_t key", "int64_t mask"), MAP_HASH, ()),
    "ppy_map_reindex": (
        "void",
        ("int8_t *handle", "int64_t size"),
        MAP_REINDEX,
        ("ppy_map_hash",),
    ),
    "ppy_map_new": ("int8_t *", (), MAP_NEW, ("ppy_map_reindex",)),
    "ppy_map_slot": ("int64_t", ("int8_t *handle", "int64_t key"), MAP_FIND, ("ppy_map_hash",)),
    "ppy_map_find": ("int64_t", ("int8_t *handle", "int64_t key"), MAP_ENTRY, ("ppy_map_slot",)),
    "ppy_map_insert": (
        "void",
        ("int8_t *handle", "int64_t key", "int64_t bits"),
        MAP_INSERT,
        ("ppy_map_find", "ppy_map_reindex", "ppy_map_hash"),
    ),
    "ppy_map_remove": ("int64_t", ("int8_t *handle", "int64_t key"), MAP_REMOVE, ("ppy_map_slot",)),
    "ppy_map_field": (
        "int64_t",
        ("int8_t *handle", "int64_t entry", "int64_t field"),
        MAP_FIELD,
        (),
    ),
    "ppy_map_clear": ("void", ("int8_t *handle",), MAP_CLEAR, ("ppy_map_reindex",)),
    "ppy_map_free": ("void", ("int8_t *handle",), MAP_FREE, ()),
    "ppy_tree_new": ("int8_t *", (), TREE_NEW, ()),
    "ppy_tree_split": (
        "void",
        ("int8_t *handle", "int64_t node", "int64_t key", "int64_t *low", "int64_t *high"),
        TREE_SPLIT,
        (),
    ),
    "ppy_tree_merge": (
        "int64_t",
        ("int8_t *handle", "int64_t low", "int64_t high"),
        TREE_MERGE,
        (),
    ),
    "ppy_tree_erase": (
        "int64_t",
        ("int8_t *handle", "int64_t node", "int64_t key"),
        TREE_ERASE,
        ("ppy_tree_merge",),
    ),
    "ppy_tree_find": ("int64_t", ("int8_t *handle", "int64_t key"), TREE_FIND, ()),
    "ppy_tree_insert": (
        "void",
        ("int8_t *handle", "int64_t key", "int64_t bits"),
        TREE_INSERT,
        ("ppy_tree_find", "ppy_tree_split", "ppy_tree_merge"),
    ),
    "ppy_tree_remove": (
        "int64_t",
        ("int8_t *handle", "int64_t key"),
        TREE_REMOVE,
        ("ppy_tree_find", "ppy_tree_erase"),
    ),
    "ppy_tree_bound": (
        "int64_t",
        ("int8_t *handle", "int64_t key", "int64_t mode"),
        TREE_BOUND,
        (),
    ),
    "ppy_tree_end": ("int64_t", ("int8_t *handle", "int64_t last"), TREE_END, ()),
    "ppy_tree_field": (
        "int64_t",
        ("int8_t *handle", "int64_t node", "int64_t field"),
        TREE_FIELD,
        (),
    ),
    "ppy_tree_clear": ("void", ("int8_t *handle",), TREE_CLEAR, ()),
}


#: name -> (result, parameters, body, needs). The order is the order a C file
#: defines them in: a function after everything it calls.
FUNCTIONS: Definitions = {
    "ppy_coll_none": ("int8_t *", (), "{\n    return NULL;\n}", ()),
    "ppy_coll_new": ("int8_t *", ("int64_t count",), NEW, ()),
    "ppy_coll_reserve": ("void", ("int8_t *handle",), RESERVE, ()),
    "ppy_coll_len": ("int64_t", ("int8_t *handle",), LENGTH, ()),
    "ppy_coll_field": ("int64_t", ("int8_t *handle", "int64_t field"), FIELD, ()),
    "ppy_coll_clear": ("void", ("int8_t *handle",), CLEAR, ()),
    "ppy_coll_free": ("void", ("int8_t *handle",), FREE, ()),
}
for _element in ELEMENTS:
    FUNCTIONS.update(_typed(_element))
FUNCTIONS.update(STRUCTURES)
for _element in ELEMENTS:
    FUNCTIONS.update(_structures(_element))

#: What every function needs from the C library.
HEADERS = ("stdint.h", "stdio.h", "stdlib.h", "string.h")


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
#: The compiled runtime once built, and whether building it was tried.
_built: Path | None = None
_tried = False


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
    global _built, _tried  # noqa: PLW0603 - one runtime per process
    with _lock:
        if _tried:
            return _built
        _tried = True
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
        _built = target
        return target
