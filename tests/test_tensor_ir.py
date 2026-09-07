"""The tensor dialect: shapes inferred, memory made, loops that agree with NumPy."""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler
from ppy_compiler.ir import (
    F64,
    I64,
    BufferType,
    Builder,
    DialectType,
    IRModule,
    decode,
    encode,
    verify,
)
from ppy_compiler.ir import shape as shapes
from ppy_compiler.ir.dialects import core, layout, tensor
from ppy_compiler.ir.passes import PassContext, PassManager
from ppy_compiler.ir.transforms import DeadCodeElimination, LoweringError, LowerTensor, SimplifyCFG
from ppy_runtime.abi import STATUS_FALLBACK, STATUS_OK

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
requires_cc = pytest.mark.skipif(c_compiler() is None, reason="no C compiler on PATH")

F64S = BufferType(F64)
I64S = BufferType(I64)


def _module() -> IRModule:
    module = IRModule("tensors")
    module.require("tensor", 1)
    t43 = tensor.tensor_type(F64, (4, 3))
    t3 = tensor.tensor_type(F64, (3,))
    t32 = tensor.tensor_type(F64, (3, 2))
    t6 = tensor.tensor_type(F64, (6,))
    t4 = tensor.tensor_type(F64, (4,))

    f = module.add_function("elementwise", [("a", F64S), ("b", F64S), ("out", F64S)], [I64])
    b = Builder(f.add_entry_block())
    a, c, out = f.entry.arguments[0], f.entry.arguments[1], f.entry.arguments[2]
    ta = tensor.load(b, a, t43, "a")
    tb = tensor.load(b, c, t3, "b")
    total = tensor.elementwise(b, "add", ta, tb)
    scaled = tensor.elementwise(b, "mul", total, ta)
    tensor.store(b, tensor.elementwise(b, "div", scaled, tensor.broadcast(b, tb, (4, 3))), out)
    core.ret(b, core.const(b, 0, I64))

    g = module.add_function("reductions", [("a", F64S), ("rows", F64S), ("top", F64S)], [I64])
    b = Builder(g.add_entry_block())
    a, rows, top = g.entry.arguments[0], g.entry.arguments[1], g.entry.arguments[2]
    ta = tensor.load(b, a, t43, "a")
    tensor.store(b, tensor.reduce(b, ta, (1,), "add"), rows)
    tensor.store(b, tensor.reduce(b, ta, (0, 1), "max", keepdims=True), top)
    core.ret(b, core.const(b, 0, I64))

    h = module.add_function("matmul_t", [("a", F64S), ("b", F64S), ("out", F64S)], [I64])
    b = Builder(h.add_entry_block())
    a, c, out = h.entry.arguments[0], h.entry.arguments[1], h.entry.arguments[2]
    product = tensor.matmul(b, tensor.load(b, a, t43), tensor.load(b, c, t32))
    tensor.store(b, tensor.transpose(b, product, (1, 0)), out)
    core.ret(b, core.const(b, 0, I64))

    k = module.add_function("slices", [("a", F64S), ("out", F64S)], [I64])
    b = Builder(k.add_entry_block())
    a, out = k.entry.arguments[0], k.entry.arguments[1]
    ta = tensor.load(b, a, t6)
    odd = tensor.slice_(b, ta, (1,), (6,), (2,))
    head = tensor.slice_(b, ta, (0,), (3,), (1,))
    joined = tensor.concat(b, (odd, head), 0)
    matrix = tensor.reshape(b, joined, (2, 3))
    tensor.store(b, tensor.transpose(b, matrix, (1, 0)), out)
    core.ret(b, core.const(b, 0, I64))

    m = module.add_function("converts", [("a", F64S), ("out", I64S)], [I64])
    b = Builder(m.add_entry_block())
    a, out = m.entry.arguments[0], m.entry.arguments[1]
    doubled = tensor.elementwise(b, "add", tensor.load(b, a, t4), tensor.load(b, a, t4))
    tensor.store(b, tensor.convert(b, doubled, I64), out)
    core.ret(b, core.const(b, 0, I64))

    big = tensor.tensor_type(F64, (300, 300))
    n = module.add_function("large", [("a", F64S), ("out", F64S)], [I64])
    b = Builder(n.add_entry_block())
    a, out = n.entry.arguments[0], n.entry.arguments[1]
    ta = tensor.load(b, a, big)
    tensor.store(b, tensor.reduce(b, tensor.elementwise(b, "mul", ta, ta), (1,), "add"), out)
    core.ret(b, core.const(b, 0, I64))
    problems = verify(module)
    assert not problems, problems
    return module


def _lowered() -> IRModule:
    module = _module()
    manager = PassManager(PassContext())
    manager.add(LowerTensor())
    manager.add(SimplifyCFG())
    manager.add(DeadCodeElimination())
    manager.run(module)
    problems = verify(module)
    assert not problems, problems
    return module


def test_types_infer_shapes_and_read_back():
    n = shapes.Symbol("N")
    symbolic = tensor.tensor_type(
        F64, (n, shapes.mul(n, 2)), layout.strided((2, 1), offset=4, align=32)
    )
    assert (
        str(symbolic)
        == "tensor.tensor<f64, N, (2 * N), layout.strided<2, 1, offset, 4, align, 32>>"
    )
    info = tensor.describe(symbolic)
    assert info is not None and info.shape == (n, shapes.mul(n, 2)) and not info.static
    assert info.layout.strides == (2, 1) and info.layout.offset == 4 and info.layout.align == 32
    module = IRModule("shapes")
    module.require("tensor", 1)
    f = module.add_function("f", [("a", F64S), ("b", F64S)], [I64])
    b = Builder(f.add_entry_block())
    a, c = f.entry.arguments[0], f.entry.arguments[1]
    left = tensor.load(b, a, tensor.tensor_type(F64, (n, 1)))
    right = tensor.load(b, c, tensor.tensor_type(F64, (1, shapes.Symbol("M"))))
    both = tensor.elementwise(b, "add", left, right)
    assert tensor.describe(both.type).shape == (n, shapes.Symbol("M"))  # type: ignore[union-attr]
    core.ret(b, core.const(b, 0, I64))
    assert not verify(module)
    text = encode(module)
    again = decode(text)
    assert not verify(again) and encode(again) == text
    assert "tensor.tensor<f64, N, M>" in text
    with pytest.raises(LoweringError, match="symbolic shape"):
        PassManager(PassContext()).add(LowerTensor()).run(module)


def test_the_verifier_holds_shapes_and_the_lowering_refuses_tensor_calls():
    module = IRModule("bad")
    module.require("tensor", 1)
    f = module.add_function("f", [("a", F64S)], [I64])
    b = Builder(f.add_entry_block())
    (a,) = (f.entry.arguments[0],)
    t = tensor.load(b, a, tensor.tensor_type(F64, (4, 3)))
    u = tensor.load(b, a, tensor.tensor_type(F64, (2,)))
    b.create("tensor.add", (t, u), (tensor.tensor_type(F64, (4, 3)),))
    b.create("tensor.matmul", (t, t), (tensor.tensor_type(F64, (4, 4)),))
    b.create("tensor.transpose", (t,), (tensor.tensor_type(F64, (3, 4)),), {"perm": (0, 0)})
    b.create("tensor.reduce", (t,), (tensor.tensor_type(F64, (4,)),), {"axes": (1,), "op": "avg"})
    b.create("tensor.reshape", (t,), (tensor.tensor_type(F64, (5,)),))
    b.create("tensor.load", (a,), (DialectType("tensor", "tensor", (F64, -1)),))
    core.ret(b, core.const(b, 0, I64))
    messages = [e.message for e in verify(module)]
    assert any("do not broadcast" in m for m in messages)
    assert any("inner dimensions differ" in m for m in messages)
    assert any("permutation of its axes" in m for m in messages)
    assert any("`op` is one of add, mul, min, max" in m for m in messages)
    assert any("changes the element count" in m for m in messages)
    assert any("not negative" in m for m in messages)
    crossing = IRModule("crossing")
    crossing.require("tensor", 1)
    g = crossing.add_function("g", [("t", tensor.tensor_type(F64, (2,)))], [])
    core.ret(Builder(g.add_entry_block()))
    with pytest.raises(LoweringError, match="crosses a call as a buffer"):
        PassManager(PassContext()).add(LowerTensor()).run(crossing)


def test_lowering_makes_views_where_it_can_and_memory_where_it_must():
    text = encode(_lowered())
    assert "tensor." not in text.replace("tensor.tensor", "").replace("module @tensors", "")
    assert "core.alloca" in text, "small tensors live on the stack"
    assert 'core.call_extern {abi = "c", callee = "malloc"}' in text or "malloc" in text, (
        "a 300x300 tensor lives on the heap"
    )
    assert "free" in text
    assert text.count("core.guard") >= 6, "every load and store checks the buffer's length"


def _call(address, name: str, *arrays: np.ndarray):  # type: ignore[no-untyped-def]
    atoms = []
    for array in arrays:
        pointer = array.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double if array.dtype == np.float64 else ctypes.c_int64)
        )
        atoms.extend([pointer, array.size])
    out = ctypes.c_int64(0)
    kinds = []
    for array in arrays:
        kinds.extend(
            [
                ctypes.POINTER(ctypes.c_double if array.dtype == np.float64 else ctypes.c_int64),
                ctypes.c_int64,
            ]
        )
    function = ctypes.CFUNCTYPE(ctypes.c_int32, *kinds, ctypes.POINTER(ctypes.c_int64))(
        address(name)
    )
    return function(*atoms, ctypes.byref(out))


def _check(address) -> None:  # type: ignore[no-untyped-def]
    a = np.arange(12, dtype=np.float64).reshape(4, 3) + 0.5
    b = np.array([1.0, 2.0, 4.0])
    out = np.zeros((4, 3))
    assert _call(address, "elementwise", a, b, out) == STATUS_OK
    np.testing.assert_array_equal(out, (a + b) * a / b)
    rows = np.zeros(4)
    top = np.zeros(1)
    assert _call(address, "reductions", a, rows, top) == STATUS_OK
    np.testing.assert_array_equal(rows, a.sum(axis=1))
    assert top[0] == a.max()
    m = np.arange(6, dtype=np.float64).reshape(3, 2) - 1.5
    product = np.zeros((2, 4))
    assert _call(address, "matmul_t", a, m, product) == STATUS_OK
    np.testing.assert_allclose(product, (a @ m).T)
    v = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    sliced = np.zeros(6)
    assert _call(address, "slices", v, sliced) == STATUS_OK
    expected = np.concatenate([v[1:6:2], v[0:3]]).reshape(2, 3).T.reshape(-1)
    np.testing.assert_array_equal(sliced, expected)
    ints = np.zeros(4, dtype=np.int64)
    assert _call(address, "converts", np.array([0.5, 1.25, -2.75, 3.0]), ints) == STATUS_OK
    np.testing.assert_array_equal(ints, (np.array([0.5, 1.25, -2.75, 3.0]) * 2).astype(np.int64))
    big = np.linspace(0.0, 1.0, 90000).reshape(300, 300)
    sums = np.zeros(300)
    assert _call(address, "large", big, sums) == STATUS_OK
    np.testing.assert_allclose(sums, (big * big).sum(axis=1))
    short = np.zeros(2)
    assert _call(address, "reductions", short, rows, top) == STATUS_FALLBACK, (
        "a buffer too short fails the guard"
    )


@requires_llvm
def test_the_llvm_backend_agrees_with_numpy():
    from ppy_compiler.backend.llvm.from_ir import emit_module
    from ppy_compiler.backend.llvm.jit import JitEngine

    engine = JitEngine(opt_level=2).open()
    engine.add(emit_module(_lowered()))
    engine.finalize()
    _check(engine.address)


@requires_cc
def test_the_c_backend_agrees_with_numpy(tmp_path: Path):
    from ppy_compiler.backend.c import Language, emit_module

    source = tmp_path / "tensors.c"
    source.write_text(emit_module(_lowered(), Language.C), encoding="utf-8")
    library = tmp_path / "libtensors.so"
    done = subprocess.run(
        [
            c_compiler(),
            "-std=c11",
            "-Wall",
            "-O2",
            "-shared",
            "-fPIC",
            "-o",
            str(library),
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "warning:" not in done.stderr, done.stderr
    handle = ctypes.CDLL(str(library))
    _check(lambda name: ctypes.cast(getattr(handle, name), ctypes.c_void_p).value or 0)
