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
    assert above.expression == "(and_kleene (greater a0 a1) (is_valid a1))", (
        "pandas' `&` is Kleene's"
    )
    assert above.kinds == ("f64", "f64") and above.result == "bool" and above.nullable
    assert "pandas expression fused" in " ".join(note for _line, note in module.fusion_notes)


def _bound(engine, loop: FusedLoop, fallback):  # type: ignore[no-untyped-def]
    """A binding with both of the kernel's null models: Arrow's bits and NumPy's NaN."""
    return bind_fused(
        loop,
        engine.address(loop.symbol),
        fallback,
        nan_address=engine.address(loop.nan_symbol),
    )


@requires_llvm
def test_numpy_backed_series_keep_their_index_and_nans():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    engine = _engine(BLEND, GREATER)

    def blend(s, t):  # type: ignore[no-untyped-def]
        return s * t + s.fillna(0.0)

    binding = _bound(engine, BLEND, blend)
    s = pd.Series([1.0, np.nan, 3.0, 4.0], name="s")
    t = pd.Series([2.0, 2.0, np.nan, 0.5], name="s")
    result = binding.wrapper(s, t)
    pd.testing.assert_series_equal(result, blend(s, t))
    assert binding.calls == 1 and binding.fallbacks == 0
    assert result.name == "s" and result.index is s.index
    assert str(result.dtype) == "float64", "NumPy-backed in, NumPy-backed out"
    other = pd.Series([2.0, 2.0, 2.0, 2.0], index=[3, 2, 1, 0])
    pd.testing.assert_series_equal(binding.wrapper(s, other), blend(s, other))
    assert binding.fallbacks == 1, "different indexes align in pandas, not in the kernel"
    ranged = pd.Series([2.0, 2.0, 2.0, 2.0], index=pd.RangeIndex(4), name="t")
    result = binding.wrapper(s, ranged)
    pd.testing.assert_series_equal(result, blend(s, ranged))
    assert result.name is None and binding.calls == 2
    above = _bound(engine, GREATER, lambda s, t: s > t)
    flags = above.wrapper(s, t)
    pd.testing.assert_series_equal(flags, s > t)
    assert above.calls == 1 and str(flags.dtype) == "bool", "a bool answer is a byte per row"
    without = bind_fused(GREATER, engine.address("k_greater"), lambda s, t: s > t)
    pd.testing.assert_series_equal(without.wrapper(s, t), s > t)
    assert without.fallbacks == 1, "with no NaN-model twin, NumPy-backed Series run pandas"


@requires_llvm
def test_the_nan_model_is_pandas_null_convention_for_numpy_storage():
    """A NaN is the null `fillna` fills and `isna` finds; `!=` is IEEE's; masks are bytes."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    fill = FusedLoop(
        "k_fill", ("s",), (), expression="(fill_null a0 c0.0)", storage="pandas", kinds=("f64",)
    )
    missing = FusedLoop(
        "k_missing",
        ("s",),
        (),
        expression="(is_null a0)",
        storage="pandas",
        kinds=("f64",),
        result="bool",
        nullable=False,
    )
    differs = FusedLoop(
        "k_differs",
        ("s", "t"),
        (),
        expression="(not_equal a0 a1)",
        storage="pandas",
        kinds=("f64", "f64"),
        result="bool",
    )
    masked = FusedLoop(
        "k_masked",
        ("s", "t", "m"),
        (),
        expression="(and (greater a0 a1) (invert a2))",
        storage="pandas",
        kinds=("f64", "f64", "bool"),
        result="bool",
    )
    engine = _engine(fill, missing, differs, masked)
    s = pd.Series([1.0, np.nan, 3.0, np.nan], name="s")
    t = pd.Series([1.0, 2.0, np.nan, np.nan], name="t")
    m = pd.Series([False, False, True, False], name="m")
    cases = [
        (fill, lambda s: s.fillna(0.0), (s,)),
        (missing, lambda s: s.isna(), (s,)),
        (differs, lambda s, t: s != t, (s, t)),
        (masked, lambda s, t, m: (s > t) & ~m, (s, t, m)),
    ]
    for loop, function, arguments in cases:
        binding = _bound(engine, loop, function)
        pd.testing.assert_series_equal(binding.wrapper(*arguments), function(*arguments))
        assert binding.calls == 1 and binding.fallbacks == 0, loop.symbol
    assert (s != t).tolist() == [False, True, True, True], "NaN differs from everything"


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

    binding = _bound(engine, BLEND, blend)
    result = binding.wrapper(s, t)
    pd.testing.assert_series_equal(result, blend(s, t))
    assert result.dtype == kind and binding.calls == 1 and binding.fallbacks == 0
    above = _bound(engine, GREATER, lambda s, t: s > t)
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


COLUMNAR_PROGRAM = """
import numpy as np
import pandas as pd
import pyarrow as pa

import ppy


@ppy.pure
def blend(s: pd.Series, t: pd.Series) -> pd.Series:
    return s * t + s.fillna(0.0)


@ppy.pure
def above(s: pd.Series, t: pd.Series) -> pd.Series:
    return (s > t) & t.notna()


@ppy.pure
def filled(s: pd.Series) -> pd.Series:
    return s.fillna(-1.0)


@ppy.pure
def missing(s: pd.Series) -> pd.Series:
    return s.isna()


@ppy.pure
def nested(s: pd.Series, t: pd.Series, scale: float) -> pd.Series:
    return (s.fillna(0.0) * scale - t.fillna(1.0)) / (t.fillna(2.0) + s.fillna(1.0)) + s * t


@ppy.pure
def differs(s: pd.Series, t: pd.Series) -> pd.Series:
    return (s != t) | (s.isna() & t.isna())


def show(name: str, result: pd.Series) -> None:
    print(name, result.dtype, [None if pd.isna(x) else round(float(x), 6) for x in result.tolist()])


def main() -> None:
    left = pd.Series([1.0, float("nan"), 3.0, 4.0, 2.5, float("nan"), -7.25], name="s")
    right = pd.Series([2.0, 2.0, float("nan"), 0.5, 2.5, float("nan"), 8.0], name="t")
    for s, t in ((left, right), (left.astype(pd.ArrowDtype(pa.float64())), right.astype(pd.ArrowDtype(pa.float64())))):
        show("blend", blend(s, t))
        show("above", above(s, t))
        show("filled", filled(s))
        show("missing", missing(s))
        show("nested", nested(s, t, 1.5))
        show("differs", differs(s, t))
        print("sums", round(float(blend(s, t).sum()), 6), int(above(s, t).sum()), int(missing(s).sum()))


main()
"""


@requires_llvm
def test_columnar_programs_answer_as_pandas_does_on_every_path(tmp_path):
    """Finite values, NaN, `fillna`, nested trees, arithmetic with null handling, the sums after:
    what `python` prints -- pandas itself -- `ppy run` prints, NumPy-backed and Arrow-backed."""
    import subprocess
    import sys
    import textwrap

    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    path = tmp_path / "frames.ppy"
    path.write_text(textwrap.dedent(COLUMNAR_PROGRAM).lstrip("\n"), encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
    plain = subprocess.run(
        [sys.executable, "frames.ppy"], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    native = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "run", "frames.ppy"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert plain.returncode == 0, plain.stderr
    assert native.returncode == 0, native.stderr
    assert native.stdout == plain.stdout
    lines = native.stdout.splitlines()
    assert lines[0] == "blend float64 [3.0, None, None, 6.0, 8.75, None, -65.25]"
    assert lines[2] == "filled float64 [1.0, -1.0, 3.0, 4.0, 2.5, -1.0, -7.25]"
    assert lines[3] == "missing bool [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]"
