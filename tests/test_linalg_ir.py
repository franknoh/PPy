"""The linalg and special dialects: loops and LAPACK that agree with NumPy."""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler
from ppy_compiler.ir import F64, I64, BufferType, Builder, IRModule, decode, encode, verify
from ppy_compiler.ir.dialects import core, linalg, special, tensor
from ppy_compiler.ir.passes import PassContext, PassManager
from ppy_compiler.ir.transforms import DeadCodeElimination, LoweringError, LowerTensor, SimplifyCFG
from ppy_runtime.abi import STATUS_FALLBACK, STATUS_OK

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
requires_cc = pytest.mark.skipif(c_compiler() is None, reason="no C compiler on PATH")
_lapack = ctypes.util.find_library("lapack")
requires_lapack = pytest.mark.skipif(_lapack is None, reason="no LAPACK on this machine")

F64S = BufferType(F64)


def _loops_module() -> IRModule:
    module = IRModule("linalg")
    module.require("tensor", 1)
    module.require("linalg", 1)
    t3 = tensor.tensor_type(F64, (3,))
    t33 = tensor.tensor_type(F64, (3, 3))
    t32 = tensor.tensor_type(F64, (3, 2))

    f = module.add_function("dot", [("a", F64S), ("b", F64S), ("out", F64S)], [I64])
    b = Builder(f.add_entry_block())
    a, c, out = f.entry.arguments[0], f.entry.arguments[1], f.entry.arguments[2]
    tensor.store(b, linalg.dot(b, tensor.load(b, a, t3), tensor.load(b, c, t3)), out)
    core.ret(b, core.const(b, 0, I64))

    g = module.add_function("chol", [("a", F64S), ("out", F64S)], [I64])
    b = Builder(g.add_entry_block())
    a, out = g.entry.arguments[0], g.entry.arguments[1]
    tensor.store(b, linalg.cholesky(b, tensor.load(b, a, t33)), out)
    core.ret(b, core.const(b, 0, I64))

    h = module.add_function("trsolve", [("a", F64S), ("rhs", F64S), ("out", F64S)], [I64])
    b = Builder(h.add_entry_block())
    a, rhs, out = h.entry.arguments[0], h.entry.arguments[1], h.entry.arguments[2]
    lower = linalg.triangular_solve(b, tensor.load(b, a, t33), tensor.load(b, rhs, t32), lower=True)
    tensor.store(b, lower, out)
    core.ret(b, core.const(b, 0, I64))

    k = module.add_function("solve", [("a", F64S), ("rhs", F64S), ("out", F64S)], [I64])
    b = Builder(k.add_entry_block())
    a, rhs, out = k.entry.arguments[0], k.entry.arguments[1], k.entry.arguments[2]
    tensor.store(b, linalg.solve(b, tensor.load(b, a, t33), tensor.load(b, rhs, t3)), out)
    core.ret(b, core.const(b, 0, I64))

    m = module.add_function("upper", [("a", F64S), ("rhs", F64S), ("out", F64S)], [I64])
    b = Builder(m.add_entry_block())
    a, rhs, out = m.entry.arguments[0], m.entry.arguments[1], m.entry.arguments[2]
    upper = linalg.triangular_solve(b, tensor.load(b, a, t33), tensor.load(b, rhs, t3), lower=False)
    tensor.store(b, upper, out)
    core.ret(b, core.const(b, 0, I64))

    s = module.add_function("specials", [("x", F64S), ("out", F64S)], [I64])
    b = Builder(s.add_entry_block())
    x, out = s.entry.arguments[0], s.entry.arguments[1]
    value = core.buffer_load(b, x, core.const(b, 0, I64))
    names = ["erf", "erfc", "gamma", "gammaln", "ndtr", "logit", "bessel_j0", "bessel_y1"]
    for index, name in enumerate(names):
        core.buffer_store(b, special.call(b, name, value), out, core.const(b, index, I64))
    order = core.const(b, 2, I64)
    core.buffer_store(
        b, special.call(b, "bessel_jn", order, value), out, core.const(b, len(names), I64)
    )
    core.ret(b, core.const(b, 0, I64))
    module.require("special", 1)
    problems = verify(module)
    assert not problems, problems
    return module


def _lapack_module() -> IRModule:
    module = IRModule("factor")
    module.require("tensor", 1)
    module.require("linalg", 1)
    t43 = tensor.tensor_type(F64, (4, 3))
    t33 = tensor.tensor_type(F64, (3, 3))
    f = module.add_function("qr", [("a", F64S), ("q", F64S), ("r", F64S)], [I64])
    b = Builder(f.add_entry_block())
    a, q_out, r_out = f.entry.arguments[0], f.entry.arguments[1], f.entry.arguments[2]
    q, r = linalg.qr(b, tensor.load(b, a, t43))
    tensor.store(b, q, q_out)
    tensor.store(b, r, r_out)
    core.ret(b, core.const(b, 0, I64))
    g = module.add_function("svd", [("a", F64S), ("u", F64S), ("s", F64S), ("vt", F64S)], [I64])
    b = Builder(g.add_entry_block())
    a, u_out, s_out, vt_out = (g.entry.arguments[i] for i in range(4))
    u, s, vt = linalg.svd(b, tensor.load(b, a, t43))
    tensor.store(b, u, u_out)
    tensor.store(b, s, s_out)
    tensor.store(b, vt, vt_out)
    core.ret(b, core.const(b, 0, I64))
    h = module.add_function("eig", [("a", F64S), ("values", F64S), ("vectors", F64S)], [I64])
    b = Builder(h.add_entry_block())
    a, values_out, vectors_out = h.entry.arguments[0], h.entry.arguments[1], h.entry.arguments[2]
    values, vectors = linalg.eig(b, tensor.load(b, a, t33))
    tensor.store(b, values, values_out)
    tensor.store(b, vectors, vectors_out)
    core.ret(b, core.const(b, 0, I64))
    problems = verify(module)
    assert not problems, problems
    return module


def _lower(module: IRModule, lapack: bool = True) -> IRModule:
    manager = PassManager(PassContext())
    manager.add(LowerTensor(lapack=lapack))
    manager.add(SimplifyCFG())
    manager.add(DeadCodeElimination())
    manager.run(module)
    problems = verify(module)
    assert not problems, problems
    return module


def _call(address, name: str, *arrays: np.ndarray):  # type: ignore[no-untyped-def]
    atoms = []
    kinds = []
    for array in arrays:
        atoms.extend([array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), array.size])
        kinds.extend([ctypes.POINTER(ctypes.c_double), ctypes.c_int64])
    out = ctypes.c_int64(0)
    function = ctypes.CFUNCTYPE(ctypes.c_int32, *kinds, ctypes.POINTER(ctypes.c_int64))(
        address(name)
    )
    return function(*atoms, ctypes.byref(out))


def _check_loops(address) -> None:  # type: ignore[no-untyped-def]
    a = np.array([1.0, 2.0, 3.0])
    c = np.array([4.0, -5.0, 6.0])
    out = np.zeros(1)
    assert _call(address, "dot", a, c, out) == STATUS_OK and out[0] == a @ c
    spd = np.array([[4.0, 2.0, 0.6], [2.0, 5.0, 1.0], [0.6, 1.0, 3.0]])
    factor = np.zeros((3, 3))
    assert _call(address, "chol", spd, factor) == STATUS_OK
    np.testing.assert_allclose(factor, np.linalg.cholesky(spd))
    not_spd = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert _call(address, "chol", not_spd, factor) == STATUS_FALLBACK, (
        "not positive definite: the guard fails"
    )
    lower = np.tril(spd)
    rhs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    solved = np.zeros((3, 2))
    assert _call(address, "trsolve", lower, rhs, solved) == STATUS_OK
    np.testing.assert_allclose(solved, np.linalg.solve(lower, rhs))
    general = np.array([[0.0, 2.0, 1.0], [1.0, -2.0, 3.0], [4.0, 1.0, -1.0]])
    vector = np.array([1.0, 2.0, 3.0])
    x = np.zeros(3)
    assert _call(address, "solve", general, vector, x) == STATUS_OK
    np.testing.assert_allclose(x, np.linalg.solve(general, vector))
    singular = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]])
    assert _call(address, "solve", singular, vector, x) == STATUS_FALLBACK
    upper = np.triu(spd)
    assert _call(address, "upper", upper, vector, x) == STATUS_OK
    np.testing.assert_allclose(x, np.linalg.solve(upper, vector))
    specials = np.zeros(9)
    assert _call(address, "specials", np.array([0.7]), specials) == STATUS_OK
    expected = [
        math.erf(0.7),
        math.erfc(0.7),
        math.gamma(0.7),
        math.lgamma(0.7),
        0.5 * math.erfc(-0.7 / math.sqrt(2)),
        math.log(0.7 / 0.3),
    ]
    np.testing.assert_allclose(specials[:6], expected)
    assert abs(specials[6] - 0.8812008886) < 1e-6, "J0(0.7)"
    assert abs(specials[7] - (-1.1032498719)) < 1e-6, "Y1(0.7)"
    assert abs(specials[8] - 0.0587869) < 1e-6, "J2(0.7)"


def _check_lapack(address) -> None:  # type: ignore[no-untyped-def]
    a = np.array([[1.0, 2.0, 0.5], [3.0, -1.0, 2.0], [0.0, 1.5, 1.0], [2.0, 2.0, -3.0]])
    q = np.zeros((4, 3))
    r = np.zeros((3, 3))
    assert _call(address, "qr", a, q, r) == STATUS_OK
    np.testing.assert_allclose(q @ r, a, atol=1e-12)
    np.testing.assert_allclose(q.T @ q, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.tril(r, -1), 0.0, atol=1e-12)
    u = np.zeros((4, 3))
    s = np.zeros(3)
    vt = np.zeros((3, 3))
    assert _call(address, "svd", a, u, s, vt) == STATUS_OK
    np.testing.assert_allclose(u @ np.diag(s) @ vt, a, atol=1e-12)
    np.testing.assert_allclose(s, np.linalg.svd(a, compute_uv=False), atol=1e-12)
    m = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, -4.0], [0.0, 4.0, 3.0]])
    values = np.zeros((3, 2))
    vectors = np.zeros((3, 3, 2))
    assert _call(address, "eig", m, values, vectors) == STATUS_OK
    complex_values = values[:, 0] + 1j * values[:, 1]
    np.testing.assert_allclose(
        sorted(complex_values, key=lambda z: (z.real, z.imag)),
        sorted(np.linalg.eigvals(m), key=lambda z: (z.real, z.imag)),
    )
    complex_vectors = vectors[:, :, 0] + 1j * vectors[:, :, 1]
    for column in range(3):
        np.testing.assert_allclose(
            m @ complex_vectors[:, column],
            complex_values[column] * complex_vectors[:, column],
            atol=1e-12,
        )


def test_the_dialects_print_read_back_and_refuse_bad_shapes():
    module = _loops_module()
    text = encode(module)
    again = decode(text)
    assert not verify(again) and encode(again) == text
    assert "linalg.cholesky" in text and "special.bessel_jn" in text
    bad = IRModule("bad")
    bad.require("tensor", 1)
    bad.require("linalg", 1)
    f = bad.add_function("f", [("a", F64S)], [I64])
    b = Builder(f.add_entry_block())
    a = f.entry.arguments[0]
    wide = tensor.load(b, a, tensor.tensor_type(F64, (2, 3)))
    b.create("linalg.cholesky", (wide,), (wide.type,))
    b.create("linalg.solve", (wide, wide), (wide.type,))
    core.ret(b, core.const(b, 0, I64))
    messages = [e.message for e in verify(bad)]
    assert any("is square" in m for m in messages)
    without = _lapack_module()
    with pytest.raises(LoweringError, match="needs LAPACK"):
        _lower(without, lapack=False)


@requires_llvm
def test_loops_agree_with_numpy_on_llvm():
    from ppy_compiler.backend.llvm.from_ir import emit_module
    from ppy_compiler.backend.llvm.jit import JitEngine

    engine = JitEngine(opt_level=2).open()
    engine.add(emit_module(_lower(_loops_module())))
    engine.finalize()
    _check_loops(engine.address)


@requires_cc
def test_loops_agree_with_numpy_in_c(tmp_path: Path):
    from ppy_compiler.backend.c import Language, emit_module

    source = tmp_path / "linalg.c"
    source.write_text(emit_module(_lower(_loops_module()), Language.C), encoding="utf-8")
    library = tmp_path / "liblinalg.so"
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
            "-lm",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "warning:" not in done.stderr, done.stderr
    handle = ctypes.CDLL(str(library))
    _check_loops(lambda name: ctypes.cast(getattr(handle, name), ctypes.c_void_p).value or 0)


@requires_llvm
@requires_lapack
def test_the_factorizations_call_lapack():
    from ppy_compiler.backend.llvm.from_ir import emit_module
    from ppy_compiler.backend.llvm.jit import JitEngine

    module = _lower(_lapack_module())
    assert module.attributes.get("ppy.libraries") == ("lapack",)
    engine = JitEngine(opt_level=2).open()
    engine.load_library("lapack")
    engine.add(emit_module(module))
    engine.finalize()
    _check_lapack(engine.address)
