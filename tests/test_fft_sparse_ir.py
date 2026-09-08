"""The fft and sparse dialects: transforms and sparse matrices that agree with NumPy."""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler
from ppy_compiler.ir import F64, I64, BufferType, Builder, IRModule, decode, encode, verify
from ppy_compiler.ir.dialects import core, fft, sparse, tensor
from ppy_compiler.ir.passes import PassContext, PassManager
from ppy_compiler.ir.transforms import DeadCodeElimination, LowerTensor, SimplifyCFG
from ppy_runtime.abi import STATUS_OK

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
requires_cc = pytest.mark.skipif(c_compiler() is None, reason="no C compiler on PATH")

F64S = BufferType(F64)
I64S = BufferType(I64)
CSR = sparse.sparse_type("csr", F64, I64, 3, 4)


def _module() -> IRModule:
    module = IRModule("spectral")
    module.require("tensor", 1)
    module.require("fft", 1)
    module.require("sparse", 1)
    complex8 = tensor.tensor_type(F64, (8, 2))
    real8 = tensor.tensor_type(F64, (8,))
    grid = tensor.tensor_type(F64, (2, 4, 2))

    f = module.add_function("roundtrip", [("x", F64S), ("spectrum", F64S), ("back", F64S)], [I64])
    b = Builder(f.add_entry_block())
    x, spectrum, back = f.entry.arguments[0], f.entry.arguments[1], f.entry.arguments[2]
    transformed = fft.fft(b, tensor.load(b, x, complex8))
    tensor.store(b, transformed, spectrum)
    tensor.store(b, fft.ifft(b, transformed), back)
    core.ret(b, core.const(b, 0, I64))

    g = module.add_function("real", [("x", F64S), ("spectrum", F64S), ("back", F64S)], [I64])
    b = Builder(g.add_entry_block())
    x, spectrum, back = g.entry.arguments[0], g.entry.arguments[1], g.entry.arguments[2]
    half = fft.rfft(b, tensor.load(b, x, real8))
    tensor.store(b, half, spectrum)
    tensor.store(b, fft.irfft(b, half, 8), back)
    core.ret(b, core.const(b, 0, I64))

    h = module.add_function("planar", [("x", F64S), ("out", F64S)], [I64])
    b = Builder(h.add_entry_block())
    x, out = h.entry.arguments[0], h.entry.arguments[1]
    tensor.store(b, fft.ifftn(b, fft.fftn(b, tensor.load(b, x, grid))), out)
    core.ret(b, core.const(b, 0, I64))

    # A 3x4 CSR matrix: values, column indices, row pointers; a 4x2 dense matrix.
    k = module.add_function(
        "sparse_ops",
        [
            ("values", F64S),
            ("cols", I64S),
            ("rows", I64S),
            ("dense", F64S),
            ("full", F64S),
            ("product", F64S),
            ("sums", F64S),
            ("cols_sum", F64S),
            ("doubled", F64S),
            ("again", F64S),
        ],
        [I64],
    )
    b = Builder(k.add_entry_block())
    args = k.entry.arguments
    matrix = sparse.from_parts(b, CSR, args[0], args[1], args[2])
    tensor.store(b, sparse.to_dense(b, matrix), args[4])
    tensor.store(
        b,
        sparse.matmul(b, matrix, tensor.load(b, args[3], tensor.tensor_type(F64, (4, 2)))),
        args[5],
    )
    tensor.store(b, sparse.reduce(b, matrix, 1), args[6])
    tensor.store(b, sparse.reduce(b, sparse.transpose(b, matrix), 1), args[7])
    twice = sparse.add(b, matrix, matrix)
    tensor.store(b, sparse.to_dense(b, twice), args[8])
    coo = sparse.convert(b, matrix, "coo")
    csc = sparse.convert(b, coo, "csc")
    back = sparse.convert(b, csc, "csr")
    tensor.store(b, sparse.to_dense(b, back), args[9])
    core.ret(b, core.const(b, 0, I64))
    problems = verify(module)
    assert not problems, problems
    return module


def _lower(module: IRModule) -> IRModule:
    manager = PassManager(PassContext())
    manager.add(LowerTensor())
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
        pointer_type = ctypes.POINTER(
            ctypes.c_double if array.dtype == np.float64 else ctypes.c_int64
        )
        atoms.extend([array.ctypes.data_as(pointer_type), array.size])
        kinds.extend([pointer_type, ctypes.c_int64])
    out = ctypes.c_int64(0)
    function = ctypes.CFUNCTYPE(ctypes.c_int32, *kinds, ctypes.POINTER(ctypes.c_int64))(
        address(name)
    )
    return function(*atoms, ctypes.byref(out))


def _check(address) -> None:  # type: ignore[no-untyped-def]
    signal = np.array([1.0, 2.0, 0.5, -1.0, 3.0, 0.0, -2.0, 1.5])
    x = np.stack([signal, np.zeros(8)], axis=1).copy()
    spectrum = np.zeros((8, 2))
    back = np.zeros((8, 2))
    assert _call(address, "roundtrip", x, spectrum, back) == STATUS_OK
    expected = np.fft.fft(signal)
    np.testing.assert_allclose(spectrum[:, 0] + 1j * spectrum[:, 1], expected, atol=1e-12)
    np.testing.assert_allclose(back[:, 0], signal, atol=1e-12)
    np.testing.assert_allclose(back[:, 1], 0.0, atol=1e-12)
    half = np.zeros((5, 2))
    real_back = np.zeros(8)
    assert _call(address, "real", signal.copy(), half, real_back) == STATUS_OK
    np.testing.assert_allclose(half[:, 0] + 1j * half[:, 1], np.fft.rfft(signal), atol=1e-12)
    np.testing.assert_allclose(real_back, signal, atol=1e-12)
    grid = np.arange(8, dtype=np.float64).reshape(2, 4)
    planar_in = np.stack([grid, np.zeros((2, 4))], axis=2).copy()
    planar_out = np.zeros((2, 4, 2))
    assert _call(address, "planar", planar_in, planar_out) == STATUS_OK
    np.testing.assert_allclose(planar_out[:, :, 0], grid, atol=1e-12)
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cols = np.array([0, 3, 1, 2, 3], dtype=np.int64)
    rows = np.array([0, 2, 3, 5], dtype=np.int64)
    dense = np.arange(8, dtype=np.float64).reshape(4, 2) + 1.0
    full = np.zeros((3, 4))
    product = np.zeros((3, 2))
    sums = np.zeros(3)
    cols_sum = np.zeros(4)
    doubled = np.zeros((3, 4))
    again = np.zeros((3, 4))
    assert (
        _call(
            address,
            "sparse_ops",
            values,
            cols,
            rows,
            dense,
            full,
            product,
            sums,
            cols_sum,
            doubled,
            again,
        )
        == STATUS_OK
    )
    reference = np.zeros((3, 4))
    for r in range(3):
        for p in range(rows[r], rows[r + 1]):
            reference[r, cols[p]] = values[p]
    np.testing.assert_array_equal(full, reference)
    np.testing.assert_array_equal(product, reference @ dense)
    np.testing.assert_array_equal(sums, reference.sum(axis=1))
    np.testing.assert_array_equal(cols_sum, reference.sum(axis=0))
    np.testing.assert_array_equal(doubled, 2 * reference)
    np.testing.assert_array_equal(again, reference, "CSR to COO to CSC to CSR is the same matrix")


def test_the_dialects_print_and_read_back():
    module = _module()
    text = encode(module)
    again = decode(text)
    assert not verify(again) and encode(again) == text
    assert "sparse.csr<f64, i64, 3, 4>" in text and "fft.irfft" in text and "n = 8" in text


@requires_llvm
def test_transforms_and_sparse_matrices_agree_with_numpy_on_llvm():
    from ppy_compiler.backend.llvm.from_ir import emit_module
    from ppy_compiler.backend.llvm.jit import JitEngine

    engine = JitEngine(opt_level=2).open()
    engine.add(emit_module(_lower(_module())))
    engine.finalize()
    _check(engine.address)


@requires_cc
def test_transforms_and_sparse_matrices_agree_with_numpy_in_c(tmp_path: Path):
    from ppy_compiler.backend.c import Language, emit_module

    source = tmp_path / "spectral.c"
    source.write_text(emit_module(_lower(_module()), Language.C), encoding="utf-8")
    library = tmp_path / "libspectral.so"
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
    _check(lambda name: ctypes.cast(getattr(handle, name), ctypes.c_void_p).value or 0)
