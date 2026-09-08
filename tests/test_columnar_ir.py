"""The columnar and arrow dialects: nulls kept, loops that agree with NumPy, Arrow read in place."""

from __future__ import annotations

import ctypes
import subprocess

import numpy as np
import pytest

from ppy_compiler.backend.c import Language
from ppy_compiler.backend.c import emit_module as emit_c
from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.from_ir import emit_module
from ppy_compiler.backend.llvm.jit import JitEngine
from ppy_compiler.backend.llvm.link import c_compiler
from ppy_compiler.ir import (
    F64,
    I64,
    U8,
    BufferType,
    Builder,
    DialectType,
    IRModule,
    PtrType,
    decode,
    encode,
    verify,
)
from ppy_compiler.ir.dialects import arrow, columnar, core
from ppy_compiler.ir.passes import PassContext, PassManager
from ppy_compiler.ir.transforms import LowerTensor
from ppy_runtime.abi import STATUS_OK

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
requires_cc = pytest.mark.skipif(c_compiler() is None, reason="no C compiler on PATH")

F64S = BufferType(F64)
I64S = BufferType(I64)
BITS = BufferType(U8)


def _pack(valid: np.ndarray) -> np.ndarray:
    """A validity bitmap as Arrow lays it out: bit i of byte i // 8, least significant first."""
    return np.packbits(np.asarray(valid, dtype=bool), bitorder="little")


def _unpack(bits: np.ndarray, n: int) -> np.ndarray:
    return np.unpackbits(np.asarray(bits, dtype=np.uint8), bitorder="little")[:n].astype(bool)


# -- the module ----------------------------------------------------------------------------------


def _module() -> IRModule:
    module = IRModule("columns")
    module.require("columnar", 1)
    module.require("arrow", 1)

    def function(name: str, params):  # type: ignore[no-untyped-def]
        f = module.add_function(name, params, [I64])
        b = Builder(f.add_entry_block())
        return b, list(f.entry.arguments)

    # arith: (a * b) + fill_null(a, 1.0), a nullable
    b, (a, av, c, n, out, outv) = function(
        "arith", [("a", F64S), ("av", BITS), ("b", F64S), ("n", I64), ("out", F64S), ("outv", BITS)]
    )
    ca = columnar.from_parts(b, a, av, n, F64, "a")
    cb = columnar.from_parts(b, c, None, n, F64, "b")
    total = columnar.binary(
        b,
        "add",
        columnar.binary(b, "mul", ca, cb),
        columnar.fill_null(b, ca, core.const(b, 1.0, F64)),
    )
    columnar.store(b, total, out, outv)
    core.ret(b, columnar.length(b, total))

    # compare_filter: the rows where a > b, nulls out
    b, (a, av, c, n, out) = function(
        "compare_filter", [("a", F64S), ("av", BITS), ("b", F64S), ("n", I64), ("out", F64S)]
    )
    ca = columnar.from_parts(b, a, av, n, F64)
    cb = columnar.from_parts(b, c, None, n, F64)
    mask = columnar.binary(b, "greater", ca, cb)
    kept = columnar.filter_(b, columnar.fill_null(b, ca, core.const(b, 0.0, F64)), mask)
    columnar.store(b, kept, out)
    core.ret(b, columnar.length(b, kept))

    # sorting: the positions in ascending order, nulls last
    b, (a, av, n, order) = function(
        "sorting", [("a", F64S), ("av", BITS), ("n", I64), ("order", I64S)]
    )
    ca = columnar.from_parts(b, a, av, n, F64)
    columnar.store(b, columnar.sort_indices(b, ca), order)
    core.ret(b, n)

    # taking: rows by position, a null position a null row
    b, (a, av, n, idx, idxv, m, out, outv) = function(
        "taking",
        [
            ("a", F64S),
            ("av", BITS),
            ("n", I64),
            ("idx", I64S),
            ("idxv", BITS),
            ("m", I64),
            ("out", F64S),
            ("outv", BITS),
        ],
    )
    ca = columnar.from_parts(b, a, av, n, F64)
    positions = columnar.from_parts(b, idx, idxv, m, I64)
    taken = columnar.take(b, ca, positions)
    columnar.store(b, taken, out, outv)
    core.ret(b, columnar.length(b, taken))

    # aggregates: sum, mean, min, max concatenated into four rows; count apart
    b, (a, av, n, out, outv, cnt) = function(
        "aggregates",
        [("a", F64S), ("av", BITS), ("n", I64), ("out", F64S), ("outv", BITS), ("cnt", I64S)],
    )
    ca = columnar.from_parts(b, a, av, n, F64)
    folded = columnar.concat(
        b, tuple(columnar.aggregate(b, what, ca) for what in ("sum", "mean", "min", "max"))
    )
    columnar.store(b, folded, out, outv)
    columnar.store(b, columnar.aggregate(b, "count", ca), cnt)
    core.ret(b, columnar.length(b, folded))

    # grouping: by an integer key, sum / count / max of a nullable value
    b, (k, v, vv, n, ok, osum, osumv, ocnt, omax, omaxv) = function(
        "grouping",
        [
            ("k", I64S),
            ("v", F64S),
            ("vv", BITS),
            ("n", I64),
            ("ok", I64S),
            ("osum", F64S),
            ("osumv", BITS),
            ("ocnt", I64S),
            ("omax", F64S),
            ("omaxv", BITS),
        ],
    )
    table = columnar.make(
        b,
        ("k", "v"),
        (columnar.from_parts(b, k, None, n, I64), columnar.from_parts(b, v, vv, n, F64)),
    )
    grouped = columnar.group_by(b, table, "k", (("v", "sum"), ("v", "count"), ("v", "max")))
    columnar.store(b, columnar.column_of(b, grouped, "k"), ok)
    columnar.store(b, columnar.column_of(b, grouped, "v_sum"), osum, osumv)
    columnar.store(b, columnar.column_of(b, grouped, "v_count"), ocnt)
    columnar.store(b, columnar.column_of(b, grouped, "v_max"), omax, omaxv)
    core.ret(b, columnar.length(b, grouped))

    # joining: inner, on one integer key
    b, (kl, vl, nl, kr, wr, nr, jk, jv, jw) = function(
        "joining",
        [
            ("kl", I64S),
            ("vl", F64S),
            ("nl", I64),
            ("kr", I64S),
            ("wr", F64S),
            ("nr", I64),
            ("jk", I64S),
            ("jv", F64S),
            ("jw", F64S),
        ],
    )
    left = columnar.make(
        b,
        ("k", "v"),
        (columnar.from_parts(b, kl, None, nl, I64), columnar.from_parts(b, vl, None, nl, F64)),
    )
    right = columnar.make(
        b,
        ("k", "w"),
        (columnar.from_parts(b, kr, None, nr, I64), columnar.from_parts(b, wr, None, nr, F64)),
    )
    joined = columnar.join(b, left, right, "k")
    columnar.store(b, columnar.column_of(b, joined, "k"), jk)
    columnar.store(b, columnar.column_of(b, joined, "v"), jv)
    columnar.store(b, columnar.column_of(b, joined, "w"), jw)
    core.ret(b, columnar.length(b, joined))

    # from_arrow: an ArrowArray struct read in place, negated, written out
    b, (p, out, outv) = function("from_arrow", [("p", PtrType(U8)), ("out", F64S), ("outv", BITS)])
    imported = arrow.import_(b, p, F64)
    column = arrow.to_column(b, imported)
    columnar.store(b, columnar.unary(b, "negate", column), out, outv)
    core.ret(b, arrow.count(b, "length", imported))

    problems = verify(module)
    assert not problems, problems
    return module


def _lowered() -> IRModule:
    module = _module()
    PassManager(PassContext()).add(LowerTensor()).run(module)
    problems = verify(module)
    assert not problems, problems
    return module


def test_types_and_operations_verify_and_read_back():
    module = _module()
    text = encode(module)
    assert "columnar.column<f64, nullable>" in text
    assert "columnar.table<k, columnar.column<i64>, v, columnar.column<f64, nullable>>" in text
    assert 'aggregates = [["v", "sum"], ["v", "count"], ["v", "max"]], key = "k"' in text
    again = decode(text)
    assert not verify(again) and encode(again) == text
    lowered = encode(_lowered())
    assert "columnar." not in lowered.replace("columnar.column", "").replace("columnar.table", "")
    assert "arrow." not in lowered.replace("arrow.array", "")


def test_the_verifier_names_misuse():
    module = IRModule("bad")
    module.require("columnar", 1)
    f = module.add_function("f", [("a", I64S), ("b", F64S), ("n", I64)], [I64])
    b = Builder(f.add_entry_block())
    a, c, n = f.entry.arguments
    ints = columnar.from_parts(b, a, None, n, I64)
    floats = columnar.from_parts(b, c, None, n, F64)
    b.create("columnar.div", (ints, ints), (columnar.column_type(I64),))
    b.create("columnar.add", (ints, floats), (columnar.column_type(F64),))
    b.create("columnar.take", (floats, floats), (columnar.column_type(F64),))
    table = columnar.make(b, ("x",), (floats,))
    b.create(
        "columnar.group_by", (table,), (table.type,), {"key": "x", "aggregates": (("x", "sum"),)}
    )
    b.create("columnar.column_of", (table,), (columnar.column_type(F64),), {"name": "y"})
    core.ret(b, n)
    messages = [e.message for e in verify(module)]
    assert any("columnar.div takes floating-point columns" in m for m in messages)
    assert any("one element type, not i64 and f64" in m for m in messages)
    assert any("integer column of row positions" in m for m in messages)
    assert any("integer or bool column without nulls" in m for m in messages)
    assert any("has no column 'y'" in m for m in messages)
    twice = DialectType(
        "columnar", "table", ("a", columnar.column_type(F64), "a", columnar.column_type(F64))
    )
    assert columnar.verify_type(twice) == "a table names column 'a' twice"


# -- running -------------------------------------------------------------------------------------


def _call(address, name: str, *arguments):  # type: ignore[no-untyped-def]
    atoms: list = []
    kinds: list = []
    for argument in arguments:
        if isinstance(argument, np.ndarray):
            pointer = {
                np.dtype("float64"): ctypes.POINTER(ctypes.c_double),
                np.dtype("int64"): ctypes.POINTER(ctypes.c_int64),
                np.dtype("uint8"): ctypes.POINTER(ctypes.c_uint8),
            }[argument.dtype]
            atoms.extend([argument.ctypes.data_as(pointer), argument.size])
            kinds.extend([pointer, ctypes.c_int64])
        elif isinstance(argument, int):
            atoms.append(argument)
            kinds.append(ctypes.c_int64)
        else:
            atoms.append(argument)
            kinds.append(ctypes.c_void_p)
    out = ctypes.c_int64(0)
    function = ctypes.CFUNCTYPE(ctypes.c_int32, *kinds, ctypes.POINTER(ctypes.c_int64))(
        address(name)
    )
    status = function(*atoms, ctypes.byref(out))
    return status, out.value


class _ArrowArray(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_int64),
        ("null_count", ctypes.c_int64),
        ("offset", ctypes.c_int64),
        ("n_buffers", ctypes.c_int64),
        ("n_children", ctypes.c_int64),
        ("buffers", ctypes.POINTER(ctypes.c_void_p)),
        ("children", ctypes.c_void_p),
        ("dictionary", ctypes.c_void_p),
        ("release", ctypes.c_void_p),
        ("private_data", ctypes.c_void_p),
    ]


def _check(address) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(7)
    n = 13
    a = rng.normal(size=n)
    valid = np.array(
        [True, False, True, True, False, True, True, True, False, True, True, False, True]
    )
    c = rng.normal(size=n)
    out = np.zeros(n)
    outv = np.zeros(2, dtype=np.uint8)
    status, length = _call(address, "arith", a, _pack(valid), c, n, out, outv)
    assert status == STATUS_OK and length == n
    np.testing.assert_array_equal(_unpack(outv, n), valid)
    np.testing.assert_allclose(out[valid], (a * c + a)[valid])

    kept = np.zeros(n)
    status, count = _call(address, "compare_filter", a, _pack(valid), c, n, kept)
    expected = a[valid & (a > c)]
    assert status == STATUS_OK and count == len(expected)
    np.testing.assert_array_equal(kept[:count], expected)

    order = np.zeros(n, dtype=np.int64)
    status, _ = _call(address, "sorting", a, _pack(valid), n, order)
    assert status == STATUS_OK
    expected_order = sorted(range(n), key=lambda i: (not valid[i], a[i] if valid[i] else 0.0))
    assert order.tolist() == expected_order

    idx = np.array([3, 0, 12, 5, 1, 7], dtype=np.int64)
    idxv = np.array([True, True, True, False, True, True])
    taken = np.zeros(6)
    takenv = np.zeros(1, dtype=np.uint8)
    status, m = _call(address, "taking", a, _pack(valid), n, idx, _pack(idxv), 6, taken, takenv)
    assert status == STATUS_OK and m == 6
    expected_valid = idxv & valid[idx]
    np.testing.assert_array_equal(_unpack(takenv, 6), expected_valid)
    np.testing.assert_array_equal(taken[expected_valid], a[idx][expected_valid])
    bad = np.array([n], dtype=np.int64)
    status, _ = _call(
        address,
        "taking",
        a,
        _pack(valid),
        n,
        bad,
        _pack([True]),
        1,
        np.zeros(1),
        np.zeros(1, dtype=np.uint8),
    )
    assert status != STATUS_OK, "a position outside the column fails the guard"

    folded = np.zeros(4)
    foldedv = np.zeros(1, dtype=np.uint8)
    cnt = np.zeros(1, dtype=np.int64)
    status, rows = _call(address, "aggregates", a, _pack(valid), n, folded, foldedv, cnt)
    assert status == STATUS_OK and rows == 4
    seen = a[valid]
    np.testing.assert_allclose(folded, [seen.sum(), seen.mean(), seen.min(), seen.max()])
    assert _unpack(foldedv, 4).all() and cnt[0] == valid.sum()
    status, _ = _call(
        address, "aggregates", a, _pack(np.zeros(n, dtype=bool)), n, folded, foldedv, cnt
    )
    assert status == STATUS_OK and not _unpack(foldedv, 4).any() and cnt[0] == 0, "all null: null"

    keys = np.array([3, 1, 3, 2, 1, 3, 9, 2], dtype=np.int64)
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    values_valid = np.array([True, True, False, True, True, True, False, True])
    groups = np.zeros(8, dtype=np.int64)
    sums = np.zeros(8)
    sumsv = np.zeros(1, dtype=np.uint8)
    counts = np.zeros(8, dtype=np.int64)
    maxes = np.zeros(8)
    maxesv = np.zeros(1, dtype=np.uint8)
    status, g = _call(
        address,
        "grouping",
        keys,
        values,
        _pack(values_valid),
        8,
        groups,
        sums,
        sumsv,
        counts,
        maxes,
        maxesv,
    )
    assert status == STATUS_OK and g == 4
    assert groups[:4].tolist() == [1, 2, 3, 9]
    np.testing.assert_allclose(sums[:4], [7.0, 12.0, 7.0, 0.0])
    assert _unpack(sumsv, 4).tolist() == [True, True, True, False], "a group of nulls sums to null"
    assert counts[:4].tolist() == [2, 2, 2, 0]
    np.testing.assert_allclose(maxes[:3], [5.0, 8.0, 6.0])
    assert _unpack(maxesv, 4).tolist() == [True, True, True, False]

    kl = np.array([2, 5, 1, 2, 7], dtype=np.int64)
    vl = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    kr = np.array([5, 2, 2, 3], dtype=np.int64)
    wr = np.array([0.5, 0.25, 0.125, 0.0625])
    jk = np.zeros(8, dtype=np.int64)
    jv = np.zeros(8)
    jw = np.zeros(8)
    status, matches = _call(address, "joining", kl, vl, 5, kr, wr, 4, jk, jv, jw)
    assert status == STATUS_OK and matches == 5
    expected_pairs = [
        (kl[i], vl[i], wr[j])
        for i in sorted(range(5), key=lambda i: kl[i])
        for j in sorted(range(4), key=lambda j: kr[j])
        if kl[i] == kr[j]
    ]
    assert list(zip(jk[:5], jv[:5], jw[:5], strict=True)) == expected_pairs

    # An Arrow array by hand: seven values from offset 3, with two nulls.
    stored = np.arange(10, dtype=np.float64)
    stored_valid = np.array([True] * 10)
    stored_valid[[4, 8]] = False
    bits = _pack(stored_valid)
    buffers = (ctypes.c_void_p * 2)(bits.ctypes.data, stored.ctypes.data)
    array = _ArrowArray(7, 2, 3, 2, 0, buffers, None, None, None, None)
    negated = np.zeros(7)
    negatedv = np.zeros(1, dtype=np.uint8)
    status, length = _call(address, "from_arrow", ctypes.addressof(array), negated, negatedv)
    assert status == STATUS_OK and length == 7
    np.testing.assert_array_equal(_unpack(negatedv, 7), stored_valid[3:10])
    np.testing.assert_array_equal(negated[stored_valid[3:10]], -stored[3:10][stored_valid[3:10]])
    without = (ctypes.c_void_p * 2)(None, stored.ctypes.data)
    plain = _ArrowArray(10, 0, 0, 2, 0, without, None, None, None, None)
    negated = np.zeros(10)
    negatedv = np.zeros(2, dtype=np.uint8)
    status, length = _call(address, "from_arrow", ctypes.addressof(plain), negated, negatedv)
    assert status == STATUS_OK and length == 10 and _unpack(negatedv, 10).all()
    np.testing.assert_array_equal(negated, -stored)


@requires_llvm
def test_columns_agree_with_numpy_on_llvm():
    engine = JitEngine(opt_level=2).open()
    engine.add(emit_module(_lowered()))
    engine.finalize()
    _check(engine.address)


@requires_cc
def test_columns_agree_with_numpy_in_c(tmp_path):  # type: ignore[no-untyped-def]
    source = tmp_path / "columns.c"
    source.write_text(emit_c(_lowered(), Language.C), encoding="utf-8")
    library = tmp_path / "columns.so"
    compiler = c_compiler()
    assert compiler is not None
    result = subprocess.run(
        [compiler, "-std=c11", "-Wall", "-O2", "-shared", "-fPIC", "-o", str(library), str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "warning:" not in result.stderr, result.stderr
    handle = ctypes.CDLL(str(library))
    _check(lambda name: ctypes.cast(getattr(handle, name), ctypes.c_void_p).value)
