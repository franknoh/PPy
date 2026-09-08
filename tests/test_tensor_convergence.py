"""Numeric plugins converge onto the tensor IR, and fused kernels are built from it."""

from __future__ import annotations

import ctypes
import math

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
from ppy_compiler.ir import F64, I64, BufferType, Builder, IRModule, encode, verify
from ppy_compiler.ir import shape as shapes
from ppy_compiler.ir.dialects import core, tensor
from ppy_compiler.ir.passes import PassContext, PassManager
from ppy_compiler.ir.transforms import LoweringError, LowerTensor
from ppy_compiler.plugins.base import DialectOperationSpec, Lowering
from ppy_compiler.plugins.jax_plugin import JaxPlugin
from ppy_compiler.plugins.numpy_plugin import NumPyPlugin
from ppy_compiler.plugins.scipy_plugin import SciPyPlugin
from ppy_compiler.plugins.torch_plugin import TorchPlugin
from ppy_runtime.abi import STATUS_OK

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")

F64S = BufferType(F64)
N = shapes.Symbol("N")
ARRAY = (T.Instance("numpy.ndarray", (), ("numpy.ndarray", "object")), Facts())
TENSOR = (T.Instance("torch.Tensor", (), ("torch.Tensor", "object")), Facts())
JAX_ARRAY = (T.Instance("jax.Array", (), ("jax.Array", "object")), Facts())


def _lowered(module: IRModule) -> IRModule:
    problems = verify(module)
    assert not problems, problems
    PassManager(PassContext()).add(LowerTensor()).run(module)
    problems = verify(module)
    assert not problems, problems
    return module


def _jit(module: IRModule):  # type: ignore[no-untyped-def]
    engine = JitEngine(opt_level=2).open()
    engine.add(emit_module(module))
    engine.finalize()
    return engine


def _call(engine, name: str, *arguments):  # type: ignore[no-untyped-def]
    """Call an IR function: buffers as (pointer, length), floats as doubles, status back."""
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
    out = ctypes.c_int64(0)
    function = ctypes.CFUNCTYPE(ctypes.c_int32, *kinds, ctypes.POINTER(ctypes.c_int64))(
        engine.address(name)
    )
    return function(*atoms, ctypes.byref(out))


# -- symbolic shapes -----------------------------------------------------------------------


def _symbolic() -> IRModule:
    module = IRModule("symbolic")
    module.require("tensor", 1)
    vector = tensor.tensor_type(F64, (N,))
    f = module.add_function("scale", [("a", F64S), ("out", F64S), ("s", F64)], [I64])
    b = Builder(f.add_entry_block())
    a, out, s = f.entry.arguments[0], f.entry.arguments[1], f.entry.arguments[2]
    t = tensor.load(b, a, vector, "a")
    scaled = tensor.elementwise(b, "mul", tensor.unary(b, "sqrt", t), tensor.fill(b, s, vector))
    tensor.store(b, scaled, out)
    core.ret(b, core.const(b, 0, I64))
    g = module.add_function("rows", [("a", F64S), ("out", F64S)], [I64])
    b = Builder(g.add_entry_block())
    a, out = g.entry.arguments[0], g.entry.arguments[1]
    t = tensor.load(b, a, tensor.tensor_type(F64, (N, 3)), "a")
    tensor.store(b, tensor.reduce(b, t, (1,), "add"), out)
    core.ret(b, core.const(b, 0, I64))
    return module


def test_a_symbolic_dimension_is_bound_from_the_buffer_length():
    text = encode(_lowered(_symbolic()))
    assert "tensor." not in text.replace("tensor.tensor", "").replace("module @symbolic", "")
    assert "malloc" in text and "free" in text, "a tensor of unknown size lives on the heap"
    assert "not a multiple of 3" in text, "the length must divide by the static extent"


@requires_llvm
def test_symbolic_kernels_run_over_any_length():
    engine = _jit(_lowered(_symbolic()))
    for length in (1, 5, 8):
        a = np.linspace(1.0, 4.0, length)
        out = np.zeros(length)
        assert _call(engine, "scale", a, out, 1.5) == STATUS_OK
        np.testing.assert_allclose(out, np.sqrt(a) * 1.5)
    matrix = np.arange(9, dtype=np.float64)
    sums = np.zeros(3)
    assert _call(engine, "rows", matrix, sums) == STATUS_OK
    np.testing.assert_array_equal(sums, matrix.reshape(3, 3).sum(axis=1))
    assert _call(engine, "rows", np.zeros(7), np.zeros(3)) != STATUS_OK, "7 is not rows of 3"


def test_shapes_one_length_cannot_bind_are_refused():
    def function(shape: shapes.Shape, first: shapes.Shape | None = None) -> IRModule:
        module = IRModule("refused")
        module.require("tensor", 1)
        f = module.add_function("f", [("a", F64S), ("out", F64S)], [I64])
        b = Builder(f.add_entry_block())
        a, out = f.entry.arguments[0], f.entry.arguments[1]
        if first is not None:
            t = tensor.empty(b, tensor.tensor_type(F64, first))
        else:
            t = tensor.load(b, a, tensor.tensor_type(F64, shape))
        tensor.store(b, t, out)
        core.ret(b, core.const(b, 0, I64))
        return module

    with pytest.raises(LoweringError, match="2 unknown dimensions"):
        _lowered(function((N, shapes.Symbol("M"))))
    with pytest.raises(LoweringError, match="on its own, once"):
        _lowered(function((N, N)))
    with pytest.raises(LoweringError, match="is not bound"):
        _lowered(function((), first=(shapes.Symbol("K"),)))


# -- unary, fill, and the extrema --------------------------------------------------------


def _elementwise() -> IRModule:
    module = IRModule("elementwise")
    module.require("tensor", 1)
    six = tensor.tensor_type(F64, (6,))
    f = module.add_function("go", [("a", F64S), ("b", F64S), ("out", F64S), ("top", F64S)], [I64])
    b = Builder(f.add_entry_block())
    a, c, out, top = (f.entry.arguments[i] for i in range(4))
    ta = tensor.load(b, a, six, "a")
    tb = tensor.load(b, c, six, "b")
    two = tensor.fill(b, core.const(b, 2.0, F64), six)
    powered = tensor.elementwise(b, "pow", tensor.unary(b, "abs", ta), two)
    clipped = tensor.elementwise(b, "max", tensor.unary(b, "exp", tb), two)
    tensor.store(b, tensor.elementwise(b, "min", powered, clipped), out)
    tensor.store(b, tensor.reduce(b, ta, (0,), "max"), top)
    core.ret(b, core.const(b, 0, I64))
    return module


@requires_llvm
def test_unary_fill_and_extrema_agree_with_numpy():
    engine = _jit(_lowered(_elementwise()))
    a = np.array([-3.0, -1.5, 0.0, 1.5, 3.0, np.nan])
    c = np.array([0.0, 1.0, -2.0, 3.0, np.nan, 0.5])
    out = np.zeros(6)
    top = np.zeros(1)
    assert _call(engine, "go", a, c, out, top) == STATUS_OK
    expected = np.minimum(np.power(np.abs(a), 2.0), np.maximum(np.exp(c), 2.0))
    np.testing.assert_array_equal(out, expected)
    assert math.isnan(top[0]), "a NaN wins a max, as NumPy has it"


def test_the_verifier_names_bad_unary_fill_and_pow():
    module = IRModule("bad")
    module.require("tensor", 1)
    floats = tensor.tensor_type(F64, (2,))
    ints = tensor.tensor_type(I64, (2,))
    f = module.add_function("f", [("a", F64S), ("i", BufferType(I64))], [I64])
    b = Builder(f.add_entry_block())
    ta = tensor.load(b, f.entry.arguments[0], floats)
    ti = tensor.load(b, f.entry.arguments[1], ints)
    b.create("tensor.unary", (ta,), (floats,), {"op": "nope"})
    b.create("tensor.unary", (ti,), (ints,), {"op": "sqrt"})
    b.create("tensor.fill", (core.const(b, 1.5, F64),), (ints,))
    b.create("tensor.pow", (ti, ti), (ints,))
    core.ret(b, core.const(b, 0, I64))
    messages = [e.message for e in verify(module)]
    assert any("`op` is one of" in m for m in messages)
    assert any("sqrt takes a floating-point tensor" in m for m in messages)
    assert any("fill of f64 gives a tensor of f64" in m for m in messages)
    assert any("tensor.pow takes floating-point tensors" in m for m in messages)


# -- fused kernels are tensor IR ----------------------------------------------------------

LOOPS = [
    FusedLoop("k_map", ("a", "b"), ("s",), expression="(add (mul (sin a0) s0) (cos a1))"),
    FusedLoop("k_sum", ("a",), (), reduction="add", expression="(mul a0 a0)"),
    FusedLoop("k_mean", ("a",), (), reduction="mean", expression="a0"),
    FusedLoop("k_max", ("a",), (), reduction="max", expression="(neg a0)"),
]


def test_kernel_module_is_tensor_ir():
    module = kernel_module(LOOPS, "kernels")
    problems = verify(module)
    assert not problems, problems
    text = encode(module)
    assert "tensor.load" in text and "tensor.unary" in text and "tensor.reduce" in text
    assert "tensor.fill" in text, "a scalar argument is a filled tensor"
    assert "tensor.tensor<f64, N>" in text, "the length is a symbol the call binds"
    assert "func @k_sum(%a0: ptr<f64>, %n: i64) -> f64" in text


@requires_llvm
def test_kernels_built_from_tensor_ir_match_numpy():
    module = kernel_module(LOOPS, "kernels")
    optimize(module, 2)
    engine = _jit(module)
    values = np.linspace(-2.0, 2.0, 33)
    others = np.arange(33, dtype=np.float64)

    def bind(name: str, fallback):  # type: ignore[no-untyped-def]
        loop = next(loop for loop in LOOPS if loop.symbol == name)
        return bind_fused(loop, engine.address(name), fallback)

    blend = bind("k_map", lambda a, b, s: np.sin(a) * s + np.cos(b))
    np.testing.assert_allclose(
        blend.wrapper(values, others, 0.5), np.sin(values) * 0.5 + np.cos(others)
    )
    total = bind("k_sum", lambda a: float(np.sum(a * a)))
    assert total.wrapper(values) == pytest.approx(float(np.sum(values * values)))
    mean = bind("k_mean", lambda a: float(np.mean(a)))
    assert mean.wrapper(others) == pytest.approx(16.0)
    top = bind("k_max", lambda a: float(np.max(-a)))
    assert top.wrapper(values) == 2.0
    with np.errstate(all="ignore"):
        # Errors ignored, so the kernel's own answer stands: a NaN wins the max.
        assert math.isnan(top.wrapper(np.array([1.0, np.nan, 3.0])))
    assert all(b.calls >= 1 and b.fallbacks == 0 for b in (blend, total, mean, top))
    assert blend.wrapper(values[::2], others[::2], 0.5) is not None and blend.fallbacks == 1


# -- the plugins' shared vocabulary ---------------------------------------------------------


def test_plugins_name_the_shared_tensor_operation():
    unary = DialectOperationSpec("tensor", "unary", (("op", "neg"),))
    total = DialectOperationSpec("tensor", "reduce", (("op", "add"),))
    numpy_plugin = NumPyPlugin()
    assert numpy_plugin.tensor_operation("numpy.multiply") == DialectOperationSpec("tensor", "mul")
    assert numpy_plugin.tensor_operation("numpy.negative") == unary
    assert numpy_plugin.tensor_operation("numpy.sum") == total
    assert numpy_plugin.tensor_operation("numpy.tanh") is None
    torch_plugin = TorchPlugin()
    assert torch_plugin.tensor_operation("torch.add") == DialectOperationSpec("tensor", "add")
    assert torch_plugin.tensor_operation("torch.neg") == unary
    assert torch_plugin.tensor_operation("torch.sum") == total
    assert torch_plugin.tensor_operation("torch.matmul") is None, "the dispatcher owns it"
    jax_plugin = JaxPlugin()
    assert jax_plugin.tensor_operation("jax.numpy.add") == DialectOperationSpec("tensor", "add")
    assert jax_plugin.tensor_operation("jax.random.normal") is None
    scipy_plugin = SciPyPlugin()
    erf = DialectOperationSpec("tensor", "unary", (("op", "erf"),))
    assert scipy_plugin.tensor_operation("scipy.special.erf") == erf
    assert scipy_plugin.tensor_operation("scipy.special.j0") == DialectOperationSpec(
        "tensor", "unary", (("op", "bessel_j0"),)
    )
    assert scipy_plugin.tensor_operation("scipy.special.gammainc") is None


def test_call_verdicts_follow_the_convergence():
    added = TorchPlugin().call("torch.add", [TENSOR, TENSOR], {})
    assert added is not None and added.kind is Lowering.DIALECT_OPERATION
    assert added.spec == DialectOperationSpec("tensor", "add")
    assert "CPU float64" in " ".join(added.guards)
    with_alpha = TorchPlugin().call("torch.add", [TENSOR, TENSOR], {"alpha": (T.FLOAT, Facts())})
    assert with_alpha is not None and with_alpha.lowering is Lowering.DIRECT_NATIVE_CALL
    matmul = TorchPlugin().call("torch.matmul", [TENSOR, TENSOR], {})
    assert matmul is not None and matmul.lowering is Lowering.DIRECT_NATIVE_CALL
    summed = NumPyPlugin().call("numpy.sum", [ARRAY], {})
    assert summed is not None
    assert summed.spec == DialectOperationSpec("tensor", "reduce", (("op", "add"),))
    scaled = NumPyPlugin().call("numpy.multiply", [ARRAY, (T.FLOAT, Facts())], {})
    assert scaled is not None and scaled.spec == DialectOperationSpec("tensor", "mul")
    assert "tensor.mul" in scaled.reason
    eager = JaxPlugin().call("jax.numpy.add", [JAX_ARRAY, JAX_ARRAY], {})
    assert eager is not None and eager.lowering is Lowering.PYTHON_FALLBACK
    bessel = SciPyPlugin().call("scipy.special.j0", [ARRAY], {})
    assert bessel is not None
    assert bessel.spec == DialectOperationSpec("special", "bessel_j0", (("elementwise", True),))


@requires_llvm
def test_torch_and_scipy_expressions_fuse_through_the_shared_vocabulary(write, analyze):
    from ppy_compiler.backend.llvm import _collect

    path = write(
        "converge.ppy",
        """
        import numpy as np
        import torch
        from scipy import special

        import ppy


        @ppy.pure
        def torch_blend(a: torch.Tensor, b: torch.Tensor) -> float:
            return float(torch.sum(torch.abs(a) * b + 1.0))


        def torch_region(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.add(torch.mul(torch.abs(a), b), b)


        @ppy.pure
        def erf_scaled(x: np.ndarray) -> np.ndarray:
            return special.erf(x) * 2.0
        """,
    )
    bundle = analyze(path, backend="llvm")
    notes = bundle.analysis.modules["converge"].lowerings.values()
    operations = {(n.qualname, n.operation, n.attributes) for n in notes}
    assert ("torch.abs", "tensor.unary", (("op", "abs"),)) in operations
    assert ("scipy.special.erf", "tensor.unary", (("op", "erf"),)) in operations
    assert ("numpy.multiply", "tensor.mul", ()) in operations
    module = _collect(bundle)["converge"]
    by_storage = {loop.storage: loop for loop in module.fused.values()}
    assert set(by_storage) == {"numpy", "torch"} and len(module.fused) == 2, (
        "the whole-function region keeps its expressions; only torch_blend fuses"
    )
    assert by_storage["torch"].expression == "(add (mul (abs a0) a1) c1.0)"
    assert by_storage["numpy"].expression == "(mul (erf a0) c2.0)"
    fused_notes = " ".join(note for _line, note in module.fusion_notes)
    assert "torch expression fused" in fused_notes and "NumPy expression fused" in fused_notes
    assert "llvm.fabs.f64" in module.ir and '@"erf"' in module.ir


@requires_llvm
def test_the_same_kernel_runs_over_torch_storage():
    torch = pytest.importorskip("torch")
    loops = [
        FusedLoop(
            "k_torch", ("a", "b"), (), expression="(add (mul (abs a0) a1) c1.0)", storage="torch"
        ),
        FusedLoop("k_torch_sum", ("a",), (), reduction="add", expression="a0", storage="torch"),
    ]
    module = kernel_module(loops, "torch_kernels")
    optimize(module, 2)
    engine = _jit(module)
    blend = bind_fused(loops[0], engine.address("k_torch"), lambda a, b: torch.abs(a) * b + 1.0)
    a = torch.linspace(-1.0, 1.0, 16, dtype=torch.float64)
    b = torch.arange(16, dtype=torch.float64)
    out = blend.wrapper(a, b)
    assert isinstance(out, torch.Tensor) and torch.equal(out, torch.abs(a) * b + 1.0)
    assert blend.calls == 1 and blend.fallbacks == 0
    tracked = a.clone().requires_grad_(True)
    assert torch.equal(blend.wrapper(tracked, b).detach(), out), "autograd runs the torch path"
    blend.wrapper(a.float(), b.float())
    assert blend.fallbacks == 2, "float32 storage is not the kernel's"
    total = bind_fused(loops[1], engine.address("k_torch_sum"), lambda a: a.sum())
    result = total.wrapper(b)
    assert isinstance(result, torch.Tensor) and result.dim() == 0
    assert result.item() == b.sum().item() and total.calls == 1
