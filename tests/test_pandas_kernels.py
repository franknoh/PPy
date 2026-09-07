"""pandas Series converge onto the columnar kernels; Arrow crosses the boundary in place."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.from_ir import emit_module
from ppy_compiler.backend.llvm.fused_runtime import bind_fused
from ppy_compiler.backend.llvm.fusion import FusedLoop, kernel_module
from ppy_compiler.backend.llvm.ir_pipeline import optimize
from ppy_compiler.backend.llvm.jit import JitEngine
from ppy_compiler.ir import F64, I64, U8, BufferType, Builder, IRModule, PtrType, verify
from ppy_compiler.ir.dialects import arrow, columnar, core
from ppy_compiler.ir.passes import PassContext, PassManager
from ppy_compiler.ir.transforms import LowerTensor
from ppy_runtime.abi import STATUS_OK

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")

BLEND = FusedLoop(
    "k_series",
    ("s", "t"),
    (),
    expression="(add (mul a0 a1) (fill_null a0 c0.0))",
    storage="pandas",
    kinds=("f64", "f64"),
)
GREATER = FusedLoop(
    "k_greater",
    ("s", "t"),
    (),
    expression="(greater a0 a1)",
    storage="pandas",
    kinds=("f64", "f64"),
    result="bool",
)


def _engine(*loops: FusedLoop):  # type: ignore[no-untyped-def]
    module = kernel_module(loops, "series_kernels")
    optimize(module, 2)
    engine = JitEngine(opt_level=2).open()
    engine.add(emit_module(module))
    engine.finalize()
    return engine


@requires_llvm
def test_series_expressions_fuse_onto_columnar_kernels(write, analyze):
    from ppy_compiler.backend.llvm import _collect

    path = write(
        "frames.ppy",
        """
        import pandas as pd

        import ppy


        @ppy.pure
        def blend(s: pd.Series, t: pd.Series) -> pd.Series:
            return s * t + s.fillna(0.0)


        @ppy.pure
        def above(s: pd.Series, t: pd.Series) -> pd.Series:
            return (s > t) & t.notna()
        """,
    )
    module = _collect(analyze(path, backend="llvm"))["frames"]
    assert len(module.fused) == 2, module.fusion_notes
    above, blend = sorted(module.fused.values(), key=lambda loop: loop.symbol)
    assert blend.storage == "pandas" and blend.expression == "(add (mul a0 a1) (fill_null a0 c0.0))"
    assert above.expression == "(and (greater a0 a1) (is_valid a1))"
    assert above.kinds == ("f64", "f64") and above.result == "bool" and above.nullable
    assert "pandas expression fused" in " ".join(note for _line, note in module.fusion_notes)


@requires_llvm
def test_numpy_backed_series_keep_their_index_and_nans():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    engine = _engine(BLEND, GREATER)

    def blend(s, t):  # type: ignore[no-untyped-def]
        return s * t + s.fillna(0.0)

    binding = bind_fused(BLEND, engine.address("k_series"), blend)
    s = pd.Series([1.0, np.nan, 3.0, 4.0], name="s")
    t = pd.Series([2.0, 2.0, np.nan, 0.5], name="s")
    result = binding.wrapper(s, t)
    pd.testing.assert_series_equal(result, blend(s, t))
    assert binding.calls == 1 and binding.fallbacks == 0
    assert result.name == "s" and result.index is s.index
    other = pd.Series([2.0, 2.0, 2.0, 2.0], index=[3, 2, 1, 0])
    pd.testing.assert_series_equal(binding.wrapper(s, other), blend(s, other))
    assert binding.fallbacks == 1, "different indexes align in pandas, not in the kernel"
    ranged = pd.Series([2.0, 2.0, 2.0, 2.0], index=pd.RangeIndex(4), name="t")
    result = binding.wrapper(s, ranged)
    pd.testing.assert_series_equal(result, blend(s, ranged))
    assert result.name is None and binding.calls == 2
    above = bind_fused(GREATER, engine.address("k_greater"), lambda s, t: s > t)
    pd.testing.assert_series_equal(above.wrapper(s, t), s > t)
    assert above.fallbacks == 1, "a bool answer over NumPy storage is bytes, not bits"


@requires_llvm
def test_arrow_backed_series_keep_their_nulls():
    pd = pytest.importorskip("pandas")
    pa = pytest.importorskip("pyarrow")
    engine = _engine(BLEND, GREATER)
    kind = pd.ArrowDtype(pa.float64())
    s = pd.Series([1.0, None, 3.0, 4.0], dtype=kind)
    t = pd.Series([2.0, 2.0, None, 0.5], dtype=kind)

    def blend(s, t):  # type: ignore[no-untyped-def]
        return s * t + s.fillna(0.0)

    binding = bind_fused(BLEND, engine.address("k_series"), blend)
    result = binding.wrapper(s, t)
    pd.testing.assert_series_equal(result, blend(s, t))
    assert result.dtype == kind and binding.calls == 1 and binding.fallbacks == 0
    above = bind_fused(GREATER, engine.address("k_greater"), lambda s, t: s > t)
    flags = above.wrapper(s, t)
    pd.testing.assert_series_equal(flags, s > t)
    assert flags.dtype == pd.ArrowDtype(pa.bool_()) and above.calls == 1
    mixed = pd.Series([2.0, 2.0, 2.0, 2.0])
    pd.testing.assert_series_equal(binding.wrapper(s, mixed), blend(s, mixed))
    assert binding.fallbacks == 1, "one Arrow-backed and one NumPy-backed Series is pandas' call"


def _from_arrow_module() -> IRModule:
    module = IRModule("boundary")
    module.require("columnar", 1)
    module.require("arrow", 1)
    f = module.add_function(
        "negated", [("p", PtrType(U8)), ("out", BufferType(F64)), ("outv", BufferType(U8))], [I64]
    )
    b = Builder(f.add_entry_block())
    p, out, outv = f.entry.arguments
    imported = arrow.import_(b, p, F64)
    columnar.store(b, columnar.unary(b, "negate", arrow.to_column(b, imported)), out, outv)
    core.ret(b, arrow.count(b, "null_count", imported))
    assert not verify(module)
    PassManager(PassContext()).add(LowerTensor()).run(module)
    return module


@requires_llvm
def test_an_arrow_array_crosses_the_boundary_through_the_c_data_interface():
    pa = pytest.importorskip("pyarrow")
    from ppy_runtime.arrow import ArrowArray, exported

    module = _from_arrow_module()
    engine = JitEngine(opt_level=2).open()
    engine.add(emit_module(module))
    engine.finalize()
    double = ctypes.POINTER(ctypes.c_double)
    byte = ctypes.POINTER(ctypes.c_uint8)
    negated = ctypes.CFUNCTYPE(
        ctypes.c_int32,
        ctypes.c_void_p,
        double,
        ctypes.c_int64,
        byte,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64),
    )(engine.address("negated"))
    array = pa.array([1.5, None, -2.0, 4.0, None, 6.0], type=pa.float64()).slice(1)
    out = np.zeros(5)
    outv = np.zeros(1, dtype=np.uint8)
    nulls = ctypes.c_int64(0)
    with exported(array) as struct:
        header = ArrowArray.from_address(struct)
        assert header.length == 5 and header.offset == 1, "a slice is an offset, not a copy"
        status = negated(
            struct, out.ctypes.data_as(double), 5, outv.ctypes.data_as(byte), 1, ctypes.byref(nulls)
        )
        assert status == STATUS_OK and nulls.value == 2
        assert bool(header.release), "the producer still owns the array inside the block"
    assert not bool(header.release), "released once the borrow ended"
    valid = np.unpackbits(outv, bitorder="little")[:5].astype(bool)
    np.testing.assert_array_equal(valid, [False, True, True, False, True])
    np.testing.assert_array_equal(out[valid], [2.0, -4.0, -6.0])
