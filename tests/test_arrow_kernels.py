"""PyArrow compute fused into columnar kernels over Arrow storage."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from ppy_compiler.analysis import types as T
from ppy_compiler.analysis.refinements import Facts
from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.from_ir import emit_module
from ppy_compiler.backend.llvm.fused_runtime import bind_fused
from ppy_compiler.backend.llvm.fusion import FusedLoop, kernel_module
from ppy_compiler.backend.llvm.ir_pipeline import optimize
from ppy_compiler.backend.llvm.jit import JitEngine
from ppy_compiler.ir import encode, verify
from ppy_compiler.plugins.base import DialectOperationSpec
from ppy_compiler.plugins.pandas_plugin import PandasPlugin
from ppy_compiler.plugins.pyarrow_plugin import PyArrowPlugin
from ppy_runtime.abi import STATUS_OK

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")

ARRAY = (T.Instance("pyarrow.Array", (), ("pyarrow.Array", "object")), Facts())
SERIES = (T.Instance("pandas.Series", (), ("pandas.Series", "object")), Facts())

BLEND = FusedLoop(
    "k_blend",
    ("a", "b"),
    ("s",),
    expression="(add (mul a0 a1) (fill_null a0 s0))",
    storage="pyarrow",
    kinds=("f64", "f64"),
)
MASK = FusedLoop(
    "k_mask",
    ("a", "b", "m"),
    (),
    expression="(and (greater a0 a1) (invert a2))",
    storage="pyarrow",
    kinds=("f64", "f64", "bool"),
    result="bool",
)
CHOOSE = FusedLoop(
    "k_choose",
    ("m", "a"),
    (),
    expression="(select a0 (negate a1) (fill_null a1 c0.0))",
    storage="pyarrow",
    kinds=("bool", "f64"),
)
SCALED = FusedLoop("k_scaled", ("a",), ("s",), expression="(mul a0 s0)", storage="pyarrow")


def _pack(valid) -> np.ndarray:  # type: ignore[no-untyped-def]
    return np.packbits(np.asarray(valid, dtype=bool), bitorder="little")


def _unpack(bits: np.ndarray, n: int) -> np.ndarray:
    return np.unpackbits(np.asarray(bits, dtype=np.uint8), bitorder="little")[:n].astype(bool)


def test_plugins_spell_the_columnar_dialect():
    plugin = PyArrowPlugin()
    assert plugin.call("pyarrow.compute.multiply", [ARRAY, ARRAY], {}).spec == DialectOperationSpec(
        "columnar", "mul"
    )
    assert plugin.call("pyarrow.compute.if_else", [ARRAY, ARRAY, ARRAY], {}).spec == (
        DialectOperationSpec("columnar", "select")
    )
    assert plugin.call("pyarrow.compute.greater", [ARRAY, ARRAY], {}).spec == DialectOperationSpec(
        "columnar", "greater"
    )
    pandas = PandasPlugin()
    assert pandas.call("pandas.truediv", [SERIES, SERIES], {}).spec == DialectOperationSpec(
        "columnar", "div"
    )
    assert pandas.call("pandas.ge", [SERIES, SERIES], {}).spec == DialectOperationSpec(
        "columnar", "greater_equal"
    )


def test_columnar_kernels_are_columnar_ir():
    module = kernel_module([BLEND, MASK, CHOOSE, SCALED], "kernels")
    problems = verify(module)
    assert not problems, problems
    text = encode(module)
    assert "func @k_blend(%out: ptr<f64>, %outv: ptr<u8>, %a0: ptr<f64>, %v0: ptr<u8>" in text
    assert "func @k_mask(%out: ptr<u8>, %outv: ptr<u8>" in text, "a bool result is a bitmap"
    assert "columnar.from_parts" in text
    assert (
        'columnar.map %10, %11, %6, %9, %s0 {expression = "(add (mul a0 a1) (fill_null a0 s0))", '
        'gives = "f64", model = "bits"}'
    ) in text, "one map evaluates the whole tree; a scalar operand is its own leaf"
    assert 'expression = "(select a0 (negate a1) (fill_null a1 c0.0))", gives = "f64"' in text
    assert "func @k_blend__nan(%out: ptr<f64>, %outv: ptr<u8>, %a0: ptr<f64>, %v0: ptr<u8>" in text
    assert 'model = "nan"' in text, "every columnar kernel has a twin under NumPy's null model"
    assert "columnar.mul" not in text and "columnar.fill_null" not in text, "nothing materialized"


def _run(engine, name: str, out, outv, *arrays, scalars=(), n: int):  # type: ignore[no-untyped-def]
    byte_pointer = ctypes.POINTER(ctypes.c_uint8)
    atoms: list = [out.ctypes.data_as(ctypes.c_void_p), outv.ctypes.data_as(byte_pointer)]
    kinds: list = [ctypes.c_void_p, byte_pointer]
    for values, valid in arrays:
        atoms.extend(
            [values.ctypes.data_as(ctypes.c_void_p), _pack(valid).ctypes.data_as(byte_pointer)]
        )
        kinds.extend([ctypes.c_void_p, byte_pointer])
    atoms.extend(float(s) for s in scalars)
    kinds.extend(ctypes.c_double for _ in scalars)
    placeholder = ctypes.c_int64(0)
    signature = ctypes.CFUNCTYPE(
        ctypes.c_int32, *kinds, ctypes.c_int64, ctypes.POINTER(ctypes.c_int64)
    )
    return signature(engine.address(name))(*atoms, n, ctypes.byref(placeholder))


@requires_llvm
def test_columnar_kernels_keep_arrow_null_semantics():
    module = kernel_module([BLEND, MASK, CHOOSE, SCALED], "kernels")
    optimize(module, 2)
    engine = JitEngine(opt_level=2).open()
    engine.add(emit_module(module))
    engine.finalize()
    n = 11
    a = np.linspace(-2.0, 3.0, n)
    a_valid = np.array([True, True, False, True, True, True, False, True, True, True, False])
    b = np.linspace(4.0, -1.0, n)
    b_valid = np.array([True, False, True, True, True, False, True, True, True, True, True])
    out = np.zeros(n)
    outv = np.zeros(2, dtype=np.uint8)
    assert (
        _run(engine, "k_blend", out, outv, (a, a_valid), (b, b_valid), scalars=(0.5,), n=n)
        == STATUS_OK
    )
    expected_valid = a_valid & b_valid
    np.testing.assert_array_equal(_unpack(outv, n), expected_valid)
    np.testing.assert_allclose(out[expected_valid], (a * b + a)[expected_valid])
    m = np.array([True, False, True, False, True, False, True, False, True, False, True])
    m_valid = np.ones(n, dtype=bool)
    bits = np.zeros(2, dtype=np.uint8)
    maskv = np.zeros(2, dtype=np.uint8)
    packed_mask = _pack(m)
    assert (
        _run(
            engine,
            "k_mask",
            bits,
            maskv,
            (a, a_valid),
            (b, b_valid),
            (packed_mask, m_valid),
            n=n,
        )
        == STATUS_OK
    )
    np.testing.assert_array_equal(_unpack(maskv, n), a_valid & b_valid)
    result = _unpack(bits, n)
    keep = a_valid & b_valid
    np.testing.assert_array_equal(result[keep], ((a > b) & ~m)[keep])
    chosen = np.zeros(n)
    chosenv = np.zeros(2, dtype=np.uint8)
    assert (
        _run(engine, "k_choose", chosen, chosenv, (packed_mask, m_valid), (a, a_valid), n=n)
        == STATUS_OK
    )
    # Mask true: the negation, null where a is null. Mask false: a with its nulls filled by 0.
    np.testing.assert_array_equal(_unpack(chosenv, n), np.where(m, a_valid, True))
    expected = np.where(m, -a, np.where(a_valid, a, 0.0))
    valid = _unpack(chosenv, n)
    np.testing.assert_allclose(chosen[valid], expected[valid])
    scaled = np.zeros(n)
    scaledv = np.zeros(2, dtype=np.uint8)
    assert _run(engine, "k_scaled", scaled, scaledv, (a, a_valid), scalars=(2.5,), n=n) == STATUS_OK
    np.testing.assert_array_equal(_unpack(scaledv, n), a_valid)
    np.testing.assert_allclose(scaled[a_valid], (a * 2.5)[a_valid])


@requires_llvm
def test_kleene_logic_is_arrows_three_valued_and_and_or():
    """`and_kleene`/`or_kleene` decide on one valid side, as `pc.and_kleene` and `pc.or_kleene` do;
    `and`/`or` are null wherever a side is, as `pc.and_` and `pc.or_` are."""
    pa = pytest.importorskip("pyarrow")
    pc = pytest.importorskip("pyarrow.compute")
    loops = [
        FusedLoop(
            f"k_{name}",
            ("a", "b"),
            (),
            expression=f"({name} a0 a1)",
            storage="pyarrow",
            kinds=("bool", "bool"),
            result="bool",
        )
        for name in ("and", "or", "and_kleene", "or_kleene")
    ]
    module = kernel_module(loops, "kernels")
    optimize(module, 2)
    engine = JitEngine(opt_level=2).open()
    engine.add(emit_module(module))
    engine.finalize()
    values = [True, False, None]
    a = pa.array([x for x in values for _ in values], type=pa.bool_())
    b = pa.array([y for _ in values for y in values], type=pa.bool_())
    for loop, reference in zip(loops, (pc.and_, pc.or_, pc.and_kleene, pc.or_kleene), strict=True):
        binding = bind_fused(loop, engine.address(loop.symbol), reference)
        result = binding.wrapper(a, b)
        assert binding.calls == 1, loop.symbol
        assert result.equals(reference(a, b)), (loop.symbol, result, reference(a, b))


@requires_llvm
def test_pyarrow_compute_expressions_fuse(write, analyze):
    from ppy_compiler.backend.llvm import _collect

    path = write(
        "arrows.ppy",
        """
        import pyarrow as pa
        import pyarrow.compute as pc

        import ppy


        @ppy.pure
        def blend(a: pa.Array, b: pa.Array, s: float) -> pa.Array:
            return pc.add(pc.multiply(a, b), pc.fill_null(a, s))


        @ppy.pure
        def flags(a: pa.Array, b: pa.Array, m: pa.Array) -> pa.Array:
            return pc.and_(pc.greater(a, b), pc.invert(m))
        """,
    )
    module = _collect(analyze(path, backend="llvm"))["arrows"]
    assert len(module.fused) == 2
    blend, flags = sorted(module.fused.values(), key=lambda loop: loop.symbol)
    assert blend.storage == "pyarrow" and blend.expression == "(add (mul a0 a1) (fill_null a0 s0))"
    assert blend.kinds == ("f64", "f64") and blend.result == "f64" and blend.nullable
    assert flags.expression == "(and (greater a0 a1) (invert a2))"
    assert flags.kinds == ("f64", "f64", "bool") and flags.result == "bool"
    notes = " ".join(note for _line, note in module.fusion_notes)
    assert "PyArrow expression fused" in notes
    assert "columnar" not in module.ir, "the kernels are lowered to loops"


@requires_llvm
def test_the_kernel_runs_over_arrow_arrays_in_place():
    pa = pytest.importorskip("pyarrow")
    pc = pytest.importorskip("pyarrow.compute")
    module = kernel_module([BLEND, MASK], "kernels")
    optimize(module, 2)
    engine = JitEngine(opt_level=2).open()
    engine.add(emit_module(module))
    engine.finalize()
    a = pa.array([1.0, None, 3.0, 4.0, None, 6.0, 7.5], type=pa.float64())
    b = pa.array([2.0, 2.0, None, 0.5, 1.0, 1.0, 2.0], type=pa.float64())

    def blend(x, y, s):  # type: ignore[no-untyped-def]
        return pc.add(pc.multiply(x, y), pc.fill_null(x, s))

    binding = bind_fused(BLEND, engine.address("k_blend"), blend)
    result = binding.wrapper(a, b, 0.5)
    assert isinstance(result, pa.Array) and result.equals(blend(a, b, 0.5)), result
    assert binding.calls == 1 and binding.fallbacks == 0
    plain = pa.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert binding.wrapper(plain, plain, 1.0).equals(blend(plain, plain, 1.0)), "no nulls at all"
    assert binding.wrapper(pa.array([1, 2, 3]), pa.array([1, 2, 3]), 1.0).equals(
        blend(pa.array([1, 2, 3]), pa.array([1, 2, 3]), 1.0)
    )
    assert binding.fallbacks == 1, "an integer array is Arrow's to compute"
    mask = pa.array([True, False, True, False, True, False, True])
    flags = bind_fused(
        MASK, engine.address("k_mask"), lambda x, y, m: pc.and_(pc.greater(x, y), pc.invert(m))
    )
    expected = pc.and_(pc.greater(a, b), pc.invert(mask))
    assert flags.wrapper(a, b, mask).equals(expected) and flags.calls == 1
    shifted = mask.slice(1)
    shorter = a.slice(1), b.slice(1)
    assert flags.wrapper(*shorter, shifted).equals(
        pc.and_(pc.greater(*shorter), pc.invert(shifted))
    )
    assert flags.fallbacks == 1, "a bitmap sliced inside a byte is Arrow's to compute"
