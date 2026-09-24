"""The collections runtime on its own: `ppy_runtime/collections.c` over words.

Held to the reference classes in `ppy._collections` through ctypes, with
elements two words wide (`tuple[int, float]`) and keys of two (`tuple[int,
int]`), so the layout, the comparisons, and the orders are all exercised. A C
program then builds nested collections and lets go of them under
AddressSanitizer, which fails on a leak or on memory used after it was freed.
"""

from __future__ import annotations

import ctypes
import random
import struct
import subprocess
from pathlib import Path

import pytest

import ppy
from ppy_compiler.backend.c.runtime import support_source
from ppy_compiler.backend.llvm.link import c_compiler
from ppy_runtime import collections as runtime

requires_cc = pytest.mark.skipif(c_compiler() is None, reason="no C compiler on PATH")

_POINTERS = (
    "ppy_seq_new", "ppy_seq_at", "ppy_seq_push_back", "ppy_seq_push_front", "ppy_seq_pop_back",
    "ppy_seq_pop_front", "ppy_heap_pop", "ppy_coll_scratch", "ppy_map_new", "ppy_map_value_at",
    "ppy_map_key_at", "ppy_tree_new", "ppy_tree_key_at", "ppy_tree_value_at", "ppy_list_new",
    "ppy_list_at",
)  # fmt: skip
_WORDS = (
    "ppy_coll_len", "ppy_coll_field", "ppy_map_find", "ppy_map_put", "ppy_map_remove",
    "ppy_map_alive", "ppy_tree_find", "ppy_tree_put", "ppy_tree_remove", "ppy_tree_bound",
    "ppy_tree_end", "ppy_list_node", "ppy_list_valid", "ppy_list_step", "ppy_list_after",
)  # fmt: skip

#: The float mask of `tuple[int, float]`: the second word is a double.
_PAIR_FLOATS = 0b10


def _library() -> ctypes.CDLL:
    path = runtime.library_path()
    assert path is not None
    lib = ctypes.CDLL(str(path))
    for name in _POINTERS:
        getattr(lib, name).restype = ctypes.c_void_p
    for name in _WORDS:
        getattr(lib, name).restype = ctypes.c_int64
    return lib


def _put_pair(address: int, value: tuple[int, float]) -> None:
    ctypes.memmove(address, struct.pack("<qd", value[0], value[1]), 16)


def _pair(address: int) -> tuple[int, float]:
    return struct.unpack("<qd", ctypes.string_at(address, 16))


def _word(value: int) -> ctypes.c_int64:
    return ctypes.c_int64(value)


@requires_cc
def test_sequences_of_pairs_match_the_reference_deque_and_its_stable_sort():
    lib = _library()
    rng = random.Random(5)
    for _ in range(60):
        made = ctypes.c_void_p(lib.ppy_seq_new(_word(0), _word(2), _word(_PAIR_FLOATS), _word(0)))
        reference = ppy.Deque[tuple[int, float]]()
        for _ in range(200):
            op = rng.randrange(4)
            value = (rng.randrange(-5, 5), rng.choice([0.0, -0.0, 1.5, -2.0, rng.random()]))
            if op == 0:
                _put_pair(lib.ppy_seq_push_back(made), value)
                reference.push_back(value)
            elif op == 1:
                _put_pair(lib.ppy_seq_push_front(made), value)
                reference.push_front(value)
            elif op == 2 and reference:
                assert _pair(lib.ppy_seq_pop_back(made)) == reference.pop_back()
            elif op == 3 and reference:
                assert _pair(lib.ppy_seq_pop_front(made)) == reference.pop_front()
        count = len(reference)
        assert [_pair(lib.ppy_seq_at(made, _word(i))) for i in range(count)] == list(reference)
        lib.ppy_seq_sort(made)
        expected = ppy.Vec[tuple[int, float]]()
        for value in reference:
            expected.push(value)
        expected.sort()
        got = [repr(_pair(lib.ppy_seq_at(made, _word(i)))) for i in range(count)]
        assert got == [repr(value) for value in expected]
        lib.ppy_coll_release(made)


@requires_cc
@pytest.mark.parametrize("largest", [False, True])
def test_heaps_of_pairs_pop_in_the_reference_order(largest: bool):
    lib = _library()
    rng = random.Random(6)
    for _ in range(60):
        made = ctypes.c_void_p(lib.ppy_seq_new(_word(0), _word(2), _word(_PAIR_FLOATS), _word(0)))
        reference = (ppy.MaxHeap if largest else ppy.Heap)[tuple[int, float]]()
        for _ in range(300):
            if rng.random() < 0.6 or not reference:
                value = (rng.randrange(0, 4), float(rng.randrange(0, 4)))
                _put_pair(lib.ppy_coll_scratch(made), value)
                lib.ppy_heap_push(made, _word(int(largest)))
                reference.push(value)
            else:
                assert _pair(lib.ppy_heap_pop(made, _word(int(largest)))) == reference.pop()
        lib.ppy_coll_release(made)


@requires_cc
def test_maps_and_trees_keyed_by_pairs_match_the_references():
    lib = _library()
    rng = random.Random(7)
    key = ctypes.create_string_buffer(16)

    def at(pair: tuple[int, int]) -> ctypes.Array[ctypes.c_char]:
        ctypes.memmove(key, struct.pack("<qq", *pair), 16)
        return key

    def word(address: int) -> int:
        return struct.unpack("<q", ctypes.string_at(address, 8))[0]

    for _ in range(60):
        hashed = ctypes.c_void_p(lib.ppy_map_new(_word(2), _word(1), _word(0), _word(0)))
        tree = ctypes.c_void_p(lib.ppy_tree_new(_word(2), _word(1), _word(0), _word(0)))
        by_hash = ppy.HashMap[tuple[int, int], int]()
        by_order = ppy.TreeMap[tuple[int, int], int]()
        for _ in range(300):
            pair = (rng.randrange(-3, 3), rng.randrange(-3, 3))
            if rng.randrange(3) < 2:
                value = rng.randrange(100)
                entry = lib.ppy_map_put(hashed, at(pair))
                ctypes.memmove(
                    lib.ppy_map_value_at(hashed, _word(entry)), struct.pack("<q", value), 8
                )
                node = lib.ppy_tree_put(tree, at(pair))
                ctypes.memmove(
                    lib.ppy_tree_value_at(tree, _word(node)), struct.pack("<q", value), 8
                )
                by_hash[pair] = value
                by_order[pair] = value
            else:
                entry = lib.ppy_map_remove(hashed, at(pair))
                node = lib.ppy_tree_remove(tree, at(pair))
                if pair in by_hash:
                    assert word(lib.ppy_map_value_at(hashed, _word(entry))) == by_hash.pop(pair)
                    assert word(lib.ppy_tree_value_at(tree, _word(node))) == by_order.pop(pair)
                else:
                    assert entry == -1 and node == -1
            bounds = (by_order.floor, by_order.ceiling, by_order.lower, by_order.higher)
            for mode, bound in enumerate(bounds):
                node = lib.ppy_tree_bound(tree, at(pair), _word(mode))
                try:
                    expected = bound(pair)
                except KeyError:
                    expected = None
                got = struct.unpack(
                    "<qq", ctypes.string_at(lib.ppy_tree_key_at(tree, _word(node)), 16)
                )
                assert (None if node < 0 else got) == expected
        used = lib.ppy_coll_field(hashed, _word(3))
        order = [
            struct.unpack("<qq", ctypes.string_at(lib.ppy_map_key_at(hashed, _word(e)), 16))
            for e in range(used)
            if lib.ppy_map_alive(hashed, _word(e))
        ]
        assert order == list(by_hash)
        walked, node = [], lib.ppy_tree_end(tree, _word(0))
        while node >= 0:
            walked.append(
                struct.unpack("<qq", ctypes.string_at(lib.ppy_tree_key_at(tree, _word(node)), 16))
            )
            node = lib.ppy_tree_bound(tree, at(walked[-1]), _word(3))
        assert walked == list(by_order)
        lib.ppy_coll_release(hashed)
        lib.ppy_coll_release(tree)


@requires_cc
def test_the_linked_list_hands_out_the_references_ids():
    lib = _library()
    rng = random.Random(8)
    for _ in range(60):
        made = ctypes.c_void_p(lib.ppy_list_new(_word(1), _word(0), _word(0)))
        reference = ppy.LinkedList[int]()
        for _ in range(200):
            nodes = [n for n, alive in enumerate(reference._alive) if alive]
            op, value = rng.randrange(4), rng.randrange(99)
            if op == 0:
                tail = lib.ppy_coll_field(made, _word(4))
                node = lib.ppy_list_node(made, _word(tail), _word(-1))
                ctypes.memmove(lib.ppy_list_at(made, _word(node)), struct.pack("<q", value), 8)
                assert node == reference.push_back(value)
            elif op == 1 and nodes:
                before = rng.choice(nodes)
                after = lib.ppy_list_step(made, _word(before), _word(1))
                node = lib.ppy_list_node(made, _word(before), _word(after))
                ctypes.memmove(lib.ppy_list_at(made, _word(node)), struct.pack("<q", value), 8)
                assert node == reference.insert_after(before, value)
            elif nodes:
                gone = rng.choice(nodes)
                lib.ppy_list_unlink(made, _word(gone))
                removed = struct.unpack(
                    "<q", ctypes.string_at(lib.ppy_list_at(made, _word(gone)), 8)
                )
                assert removed[0] == reference.remove(gone)
        walked, node = [], lib.ppy_coll_field(made, _word(3))
        while node != -1:
            walked.append(
                struct.unpack("<q", ctypes.string_at(lib.ppy_list_at(made, _word(node)), 8))[0]
            )
            node = lib.ppy_list_after(made, _word(node))
        assert walked == list(reference)
        lib.ppy_coll_release(made)


OWNERSHIP = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int8_t *ppy_seq_new(int64_t, int64_t, int64_t, int64_t);
int8_t *ppy_seq_push_back(int8_t *);
int8_t *ppy_seq_at(int8_t *, int64_t);
int8_t *ppy_seq_pop_back(int8_t *);
void ppy_coll_retain(int8_t *);
void ppy_coll_release(int8_t *);
void ppy_seq_clear(int8_t *);
int8_t *ppy_map_new(int64_t, int64_t, int64_t, int64_t);
int64_t ppy_map_put(int8_t *, const int8_t *);
int8_t *ppy_map_value_at(int8_t *, int64_t);
int64_t ppy_map_remove(int8_t *, const int8_t *);
int8_t *ppy_tree_new(int64_t, int64_t, int64_t, int64_t);
int64_t ppy_tree_put(int8_t *, const int8_t *);
int8_t *ppy_tree_value_at(int8_t *, int64_t);
int8_t *ppy_list_new(int64_t, int64_t, int64_t);
int64_t ppy_list_node(int8_t *, int64_t, int64_t);
int8_t *ppy_list_at(int8_t *, int64_t);
int main(void) {
    /* Vec[Vec[int]] with 1000 inner vectors, one alias kept past the outer. */
    int8_t *outer = ppy_seq_new(0, 1, 0, 1);
    int8_t *kept = NULL;
    for (int i = 0; i < 1000; i++) {
        int8_t *inner = ppy_seq_new(0, 1, 0, 0);
        for (int j = 0; j < i % 7; j++) {
            int64_t v = j;
            memcpy(ppy_seq_push_back(inner), &v, 8);
        }
        memcpy(ppy_seq_push_back(outer), &inner, 8);   /* the outer takes the reference */
        if (i == 500) {
            kept = inner;
            ppy_coll_retain(kept);
        }
    }
    int8_t *popped;
    memcpy(&popped, ppy_seq_pop_back(outer), 8);     /* ownership moves to us */
    ppy_coll_release(popped);
    ppy_coll_release(outer);
    int64_t first;
    memcpy(&first, ppy_seq_at(kept, 0), 8);
    ppy_coll_release(kept);
    /* HashMap[int, Vec[int]], TreeMap[int, Vec[int]], LinkedList[Vec[int]], with removal and clear. */
    int8_t *map = ppy_map_new(1, 1, 0, 1);
    int8_t *tree = ppy_tree_new(1, 1, 0, 1);
    int8_t *list = ppy_list_new(1, 0, 1);
    for (int64_t k = 0; k < 300; k++) {
        int8_t *a = ppy_seq_new(3, 1, 0, 0), *b = ppy_seq_new(2, 1, 0, 0), *c = ppy_seq_new(1, 1, 0, 0);
        memcpy(ppy_map_value_at(map, ppy_map_put(map, (int8_t *)&k)), &a, 8);
        memcpy(ppy_tree_value_at(tree, ppy_tree_put(tree, (int8_t *)&k)), &b, 8);
        memcpy(ppy_list_at(list, ppy_list_node(list, -1, -1 + 0 * k)), &c, 8);
    }
    for (int64_t k = 0; k < 300; k += 3) {
        int64_t e = ppy_map_remove(map, (int8_t *)&k);
        int8_t *gone;
        memcpy(&gone, ppy_map_value_at(map, e), 8);
        ppy_coll_release(gone);
    }
    ppy_coll_release(map);
    ppy_coll_release(tree);
    ppy_coll_release(list);
    printf("ok %lld\n", (long long)first);
    return 0;
}
"""


def _sanitizes(compiler: str, directory: Path) -> bool:
    probe = directory / "probe.c"
    probe.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    done = subprocess.run(
        [compiler, "-fsanitize=address", str(probe), "-o", str(directory / "probe")],
        capture_output=True,
        check=False,
    )
    return done.returncode == 0


@requires_cc
def test_nested_collections_are_freed_once_and_never_used_after(tmp_path: Path):
    """References counted: an alias outlives its container, removed values are
    handed over, and letting go of the outermost frees everything, once."""
    compiler = c_compiler()
    assert compiler is not None
    if not _sanitizes(compiler, tmp_path):
        pytest.skip("the C compiler has no AddressSanitizer")
    (tmp_path / "support.c").write_text(support_source(), encoding="utf-8")
    (tmp_path / "ownership.c").write_text(OWNERSHIP, encoding="utf-8")
    binary = tmp_path / "ownership"
    subprocess.run(
        [
            compiler,
            "-g",
            "-fsanitize=address,undefined",
            "ownership.c",
            "support.c",
            "-o",
            str(binary),
        ],
        cwd=tmp_path,
        check=True,
    )
    ran = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        check=False,
        env={"ASAN_OPTIONS": "detect_leaks=1"},
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == "ok 0"
