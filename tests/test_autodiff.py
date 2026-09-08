"""Autodiff: `ppy.grad` under CPython, the IR transform natively, and their agreement."""

from __future__ import annotations

import ctypes
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import ppy
from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.from_ir import emit_module
from ppy_compiler.backend.llvm.ir_pipeline import optimize
from ppy_compiler.backend.llvm.jit import JitEngine
from ppy_compiler.ir import F64, I64, BufferType, Builder, IRModule, encode, verify
from ppy_compiler.ir.dialects import core, tensor
from ppy_compiler.ir.dialects import math as math_dialect
from ppy_compiler.ir.model import Successor
from ppy_compiler.ir.transforms import AutodiffError, differentiate, promote_slots
from ppy_runtime.abi import STATUS_OK

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")

F64S = BufferType(F64)


# -- the reference implementation ----------------------------------------------------------------


def f(x: float, y: float) -> float:
    return math.sin(x) * y + x * x


def g(x: float) -> float:
    a = x * 2.0
    b = math.exp(a) / (1.0 + x)
    b += math.sqrt(abs(a)) - math.log(b)
    return b**3.0 - x / a


def _finite(function, index, *args, step=1e-6):  # type: ignore[no-untyped-def]
    forward = list(args)
    backward = list(args)
    forward[index] += step
    backward[index] -= step
    return (function(*forward) - function(*backward)) / (2 * step)


def test_grad_matches_the_analytic_derivative_and_finite_differences():
    df = ppy.grad(f)
    assert df(0.3, 2.0) == math.cos(0.3) * 2.0 + (0.3 + 0.3)
    dy = ppy.grad(f, argnums=1)
    assert dy(0.3, 2.0) == math.sin(0.3)
    both = ppy.grad(f, argnums=(0, 1))
    assert both(0.3, 2.0) == (df(0.3, 2.0), dy(0.3, 2.0))
    value, gradient = ppy.value_and_grad(f)(0.3, 2.0)
    assert value == f(0.3, 2.0) and gradient == df(0.3, 2.0)
    dg = ppy.grad(g)
    for x in (0.4, 1.7, 3.2):
        assert dg(x) == pytest.approx(_finite(g, 0, x), rel=1e-5)


def test_grad_over_numpy_arrays_follows_the_same_rules():
    def loss(w: np.ndarray, x: np.ndarray) -> float:
        p = x @ w
        return np.sum(p * p) / 2.0 + np.mean(np.exp(w))

    rng = np.random.default_rng(0)
    w = rng.normal(size=(3, 2))
    x = rng.normal(size=(4, 3))
    gradient = ppy.grad(loss)(w, x)
    expected = x.T @ (x @ w) + np.exp(w) / w.size
    np.testing.assert_allclose(gradient, expected)
    for i in range(3):
        for j in range(2):
            bump = np.zeros_like(w)
            bump[i, j] = 1e-6
            finite = (loss(w + bump, x) - loss(w - bump, x)) / 2e-6
            assert gradient[i, j] == pytest.approx(finite, rel=1e-5)


def test_what_the_rules_do_not_cover_is_refused_with_the_reason():
    def branchy(x: float) -> float:
        if x > 0:
            return x
        return -x

    def loopy(x: float) -> float:
        for _ in range(3):
            x = x * x
        return x

    def calls(x: float) -> float:
        return f(x, x)

    with pytest.raises(TypeError, match="`If` is not differentiated"):
        ppy.grad(branchy)(1.0)
    with pytest.raises(TypeError, match="`For` is not differentiated"):
        ppy.grad(loopy)(1.0)
    with pytest.raises(TypeError, match="a call to `f` is not differentiated"):
        ppy.grad(calls)(1.0)
    with pytest.raises(TypeError, match="`argnums`"):
        ppy.grad(f, argnums=-1)
    with pytest.raises(TypeError, match="names no parameter"):
        ppy.grad(f, argnums=2)(1.0, 2.0)


# -- the IR transform ---------------------------------------------------------------------------


def _scalar_module() -> IRModule:
    """`f(x, y) = sin(x) * y + x * x`, spelled with the frontend's stack slots."""
    module = IRModule("scalars")
    module.require("math", 1)
    function = module.add_function("f", [("x", F64), ("y", F64)], [F64])
    b = Builder(function.add_entry_block())
    x, y = function.entry.arguments
    x_slot = core.alloca(b, F64, name="x.addr")
    core.store(b, x, x_slot)
    t_slot = core.alloca(b, F64, name="t.addr")
    x1 = core.load(b, x_slot)
    core.store(b, core.mul(b, math_dialect.call(b, "sin", x1), y), t_slot)
    x2 = core.load(b, x_slot)
    total = core.add(b, core.load(b, t_slot), core.mul(b, x2, x2))
    core.ret(b, total)
    return module


def test_slots_are_promoted_and_the_derivative_is_taken_in_reverse():
    module = _scalar_module()
    function = module.functions["f"]
    assert promote_slots(function) == 2
    names = [op.name for op in function.body.blocks[0].operations]
    assert "core.alloca" not in names and "core.load" not in names and "core.store" not in names
    derived = differentiate(module, function, wrt=(0, 1), value=True)
    assert derived.name == "f__value_and_grad_0_1"
    assert [str(t) for t in derived.results] == ["f64", "f64", "f64"]
    text = encode(module)
    assert "func @f__value_and_grad_0_1(%x: f64, %y: f64) -> (f64, f64, f64)" in text
    assert "math.cos" in text, "the adjoint of sin is cos"
    problems = verify(module)
    assert not problems, problems
    assert differentiate(module, function, wrt=(0, 1), value=True) is derived, "made once"


def _call(engine, name: str, *arguments, results: int = 1):  # type: ignore[no-untyped-def]
    atoms: list = []
    kinds: list = []
    double_pointer = ctypes.POINTER(ctypes.c_double)
    for argument in arguments:
        if isinstance(argument, np.ndarray):
            atoms.extend([argument.ctypes.data_as(double_pointer), argument.size])
            kinds.extend([double_pointer, ctypes.c_int64])
        else:
            atoms.append(float(argument))
            kinds.append(ctypes.c_double)
    outs = [ctypes.c_double(0.0) for _ in range(results)]
    signature = ctypes.CFUNCTYPE(ctypes.c_int32, *kinds, *([double_pointer] * results))
    status = signature(engine.address(name))(*atoms, *[ctypes.byref(o) for o in outs])
    return status, [o.value for o in outs]


@requires_llvm
def test_the_native_derivative_agrees_with_the_reference_bit_for_bit():
    module = _scalar_module()
    derived = differentiate(module, module.functions["f"], wrt=(0, 1), value=True)
    optimize(module, 2)
    engine = JitEngine(opt_level=2).open()
    engine.add(emit_module(module))
    engine.finalize()
    for x, y in ((0.3, 2.0), (-1.25, 0.5), (7.0, -3.0)):
        status, (value, dx, dy) = _call(engine, derived.name, x, y, results=3)
        assert status == STATUS_OK
        expected_value, expected_dx = ppy.value_and_grad(f)(x, y)
        assert value == expected_value and dx == expected_dx
        assert dy == ppy.grad(f, argnums=1)(x, y)


def _tensor_module() -> IRModule:
    """`loss(a, w) = sum((a @ w)^2)` over a 4x3 `a` and a 3x2 `w` read from buffers."""
    module = IRModule("tensors")
    module.require("tensor", 1)
    function = module.add_function(
        "loss", [("a", F64S), ("w", F64S)], [tensor.tensor_type(F64, ())]
    )
    b = Builder(function.add_entry_block())
    a, w = function.entry.arguments
    ta = tensor.load(b, a, tensor.tensor_type(F64, (4, 3)), "a")
    tw = tensor.load(b, w, tensor.tensor_type(F64, (3, 2)), "w")
    p = tensor.matmul(b, ta, tw)
    squared = tensor.elementwise(b, "mul", p, p)
    half = tensor.fill(b, core.const(b, 0.5, F64), tensor.tensor_type(F64, (4, 2)))
    core.ret(b, tensor.reduce(b, tensor.elementwise(b, "mul", squared, half), (0, 1), "add"))
    return module


@requires_llvm
def test_tensor_gradients_are_written_to_buffers():
    module = _tensor_module()
    derived = differentiate(module, module.functions["loss"], wrt=(1,), value=True)
    assert [name for name, _t in derived.params] == ["a", "w", "grad_w"]
    assert [str(t) for t in derived.results] == ["f64"], "a rank-0 value comes out as a scalar"
    problems = verify(module)
    assert not problems, problems
    # The source returns a tensor, which no call boundary carries; the derivative stands alone.
    del module.functions["loss"]
    optimize(module, 2)
    engine = JitEngine(opt_level=2).open()
    engine.add(emit_module(module))
    engine.finalize()
    rng = np.random.default_rng(1)
    a = rng.normal(size=(4, 3))
    w = rng.normal(size=(3, 2))
    grad_w = np.zeros((3, 2))
    status, (value,) = _call(engine, derived.name, a, w, grad_w, results=1)
    assert status == STATUS_OK
    p = a @ w
    assert value == pytest.approx(0.5 * np.sum(p * p))
    np.testing.assert_allclose(grad_w, a.T @ p)


def test_the_transform_refuses_what_it_cannot_follow():
    module = IRModule("refused")
    module.require("tensor", 1)
    two_blocks = module.add_function("branchy", [("x", F64)], [F64])
    entry = two_blocks.add_entry_block()
    other = two_blocks.body.add_block("other")
    b = Builder(entry)
    core.br(b, Successor(other))
    core.ret(Builder(other), entry.arguments[0])
    with pytest.raises(AutodiffError, match="control flow"):
        differentiate(module, two_blocks)
    ints = module.add_function("ints", [("n", I64)], [F64])
    b = Builder(ints.add_entry_block())
    core.ret(b, core.cast(b, ints.entry.arguments[0], F64))
    with pytest.raises(AutodiffError, match="a float or a buffer of floats"):
        differentiate(module, ints)
    top = module.add_function("top", [("a", F64S)], [tensor.tensor_type(F64, ())])
    b = Builder(top.add_entry_block())
    ta = tensor.load(b, top.entry.arguments[0], tensor.tensor_type(F64, (3,)))
    core.ret(b, tensor.reduce(b, ta, (0,), "max"))
    with pytest.raises(AutodiffError, match=r"tensor\.reduce max has no derivative rule"):
        differentiate(module, top)


# -- the language ----------------------------------------------------------------------------

PROGRAM = """
    import math

    import ppy


    def f(x: float, y: float) -> float:
        return math.sin(x) * y + x * x


    df = ppy.grad(f)
    both = ppy.value_and_grad(f, argnums=1)


    def slope(x: float, y: float) -> float:
        return df(x, y)


    def pair(x: float, y: float) -> float:
        v, g = both(x, y)
        return v * 2.0 + g


    def direct(x: float) -> float:
        return ppy.grad(f)(x, 1.5) + ppy.grad(f, argnums=1)(x, 1.5)
    """


IR_ROAD = '[tool.ppy]\nstrict = true\n\n[tool.ppy.llvm]\npipeline = "ir"\n'


def test_derivatives_type_check_and_lower_natively(write, analyze):
    from ppy_compiler.backend.llvm import _collect
    from ppy_compiler.backend.llvm.ir_pipeline import ir_modules

    write("pyproject.toml", IR_ROAD)
    path = write("deriv.ppy", PROGRAM)
    bundle = analyze(path, backend="llvm")
    assert not bundle.diagnostics.has_errors(), [d.message for d in bundle.diagnostics.errors]
    module = _collect(bundle)["deriv"]
    assert {"deriv.slope", "deriv.pair", "deriv.direct"} <= set(module.functions), module.rejected
    text = encode(ir_modules(bundle)["deriv"])
    assert "func @deriv_f__grad(" in text and "func @deriv_f__value_and_grad_1(" in text
    assert text.count("math.cos") == 1, "d/dy needs no cos; d/dx does"


def test_misuse_is_named(codes, write):
    path = write(
        "bad_grad.ppy",
        """
        import ppy


        def whole(n: int) -> int:
            return n * 2


        def loud(x: float) -> float:
            print(x)
            return x


        def fine(x: float, y: float) -> float:
            return x * y


        a = ppy.grad(whole)
        b = ppy.grad(fine, argnums=3)
        c = ppy.grad(3.0)
        d = ppy.grad(loud)
        e = ppy.grad(fine, step=2)
        """,
    )
    found = codes(path)
    assert found.count("E1661") == 3, found
    assert "E1660" in found and "E1662" in found


@requires_llvm
def test_the_three_paths_print_the_same_derivatives(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(IR_ROAD, encoding="utf-8")
    entry = tmp_path / "grad_run.ppy"
    entry.write_text(
        textwrap.dedent(PROGRAM).lstrip("\n")
        + textwrap.dedent(
            """

            def main() -> None:
                for x in (0.3, -1.25, 7.0):
                    print(repr(slope(x, 2.0)), repr(pair(x, 0.5)), repr(direct(x)))


            main()
            """
        ),
        encoding="utf-8",
    )
    plain = subprocess.run(
        [sys.executable, entry.name], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    native = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "run", entry.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert plain.returncode == 0, plain.stderr
    assert native.returncode == 0, native.stderr
    assert native.stdout == plain.stdout
    assert "0." in plain.stdout
