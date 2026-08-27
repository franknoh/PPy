"""Native buffers and fused library kernels (spec 13.3, 16, 19.4)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import _collect, available as llvm_available, emit_ir
from ppy_compiler.backend.llvm.jit import JitEngine
from ppy_compiler.backend.llvm.runtime import bind

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
has_numpy = importlib.util.find_spec("numpy") is not None
requires_numpy = pytest.mark.skipif(not has_numpy, reason="numpy is not installed")

pytestmark = requires_llvm


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


def _ppy(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return _run([sys.executable, "-m", "ppy_compiler", *args], cwd)


if has_numpy:
    import numpy as _numpy

    class _Subclass(_numpy.ndarray):
        """An ndarray subclass: dispatch may be overridden, so guards must reject it."""
else:  # pragma: no cover - exercised only where numpy is absent
    _Subclass = None


def _jit(module):  # type: ignore[no-untyped-def]
    engine = JitEngine(opt_level=2).open()
    engine.add(module.ir)
    engine.finalize()
    return engine


BUFFERS = """
    import ppy


    @ppy.pure
    def dot(a: list[float], b: list[float]) -> float:
        total: float = 0.0
        for i in range(len(a)):
            total += a[i] * b[i]
        return total


    @ppy.pure
    def total(xs: list[int]) -> int:
        result: int = 0
        for x in xs:
            result += x
        return result


    @ppy.pure
    def peak(xs: list[float]) -> float:
        return max(xs)


    @ppy.pure
    def add_up(xs: list[float]) -> float:
        return sum(xs)
    """


def test_list_parameters_are_lowered_to_native_buffers(write, analyze):
    path = write("buffers.ppy", BUFFERS)
    module = _collect(analyze(path, backend="llvm"))["buffers"]
    assert set(module.functions) >= {
        "buffers.dot", "buffers.total", "buffers.peak", "buffers.add_up"
    }
    signature = module.functions["buffers.dot"].signature
    assert signature.params == ("double*", "i64", "double*", "i64")
    assert "double* a, i64 a_len" in str(signature)


def test_buffer_loops_become_real_native_loops(write, analyze):
    path = write("loops.ppy", BUFFERS)
    ir = emit_ir(analyze(path, backend="llvm"))["loops"]
    assert "getelementptr double" in ir
    assert "load double" in ir
    assert "fmul" in ir or "fadd" in ir


def test_a_mutating_function_keeps_its_buffer_boxed(write, analyze):
    path = write(
        "mutates.ppy",
        """
        def push(xs: list[int]) -> int:
            xs.append(1)
            return len(xs)
        """,
    )
    module = _collect(analyze(path, backend="llvm"))["mutates"]
    assert "mutates.push" in module.rejected
    assert "mutates" in module.rejected["mutates.push"]


def test_native_buffer_results_match_python(write, analyze):
    path = write("exec.ppy", BUFFERS)
    module = _collect(analyze(path, backend="llvm"))["exec"]
    engine = _jit(module)

    def fallback_dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    lowered = module.functions["exec.dot"]
    binding = bind(lowered.signature, engine.address(lowered.signature.symbol), fallback_dot)
    assert binding.wrapper([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == 32.0
    assert binding.calls == 1 and binding.fallbacks == 0


def test_out_of_range_integers_fall_back_instead_of_truncating(write, analyze):
    """`array.array` rejects a too-wide value; a ctypes copy would truncate."""
    path = write("wide.ppy", BUFFERS)
    module = _collect(analyze(path, backend="llvm"))["wide"]
    engine = _jit(module)

    lowered = module.functions["wide.total"]
    binding = bind(lowered.signature, engine.address(lowered.signature.symbol), sum)
    assert binding.wrapper([1, 2, 3]) == 6
    assert binding.calls == 1

    assert binding.wrapper([10**30, 1]) == 10**30 + 1
    assert binding.fallbacks == 1


def test_non_list_arguments_fall_back(write, analyze):
    path = write("guard.ppy", BUFFERS)
    module = _collect(analyze(path, backend="llvm"))["guard"]
    engine = _jit(module)
    lowered = module.functions["guard.total"]
    binding = bind(lowered.signature, engine.address(lowered.signature.symbol), sum)

    assert binding.wrapper((1, 2, 3)) == 6
    assert binding.fallbacks == 1


def test_empty_reduction_falls_back_so_python_raises(write, analyze):
    path = write("empty.ppy", BUFFERS)
    module = _collect(analyze(path, backend="llvm"))["empty"]
    engine = _jit(module)
    lowered = module.functions["empty.peak"]
    binding = bind(lowered.signature, engine.address(lowered.signature.symbol), max)

    with pytest.raises(ValueError):
        binding.wrapper([])
    assert binding.fallbacks == 1


def test_buffer_programs_match_plain_cpython(tmp_path: Path):
    entry = tmp_path / "buffers_run.ppy"
    entry.write_text(
        textwrap.dedent(BUFFERS).lstrip("\n")
        + textwrap.dedent(
            """

            print(dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]))
            print(total([1, 2, 3, 4, 5]), total([10 ** 30, 1]))
            print(peak([3.0, 9.0, 1.0]), add_up([0.1, 0.2, 0.3]))
            try:
                peak([])
            except ValueError as error:
                print("caught", type(error).__name__)
            """
        ),
        encoding="utf-8",
    )
    plain = _run([sys.executable, entry.name], tmp_path)
    native = _ppy(["run", entry.name], tmp_path)
    assert plain.returncode == 0, plain.stderr
    assert native.returncode == 0, native.stderr
    assert native.stdout == plain.stdout


def test_float_reduction_keeps_source_order(write, analyze):
    """A fused sum accumulates sequentially, so no reassociation occurs."""
    path = write("order.ppy", BUFFERS)
    ir = emit_ir(analyze(path, backend="llvm"))["order"]
    assert "fadd fast" not in ir
    assert "reassoc" not in ir


# -- NumPy fusion ---------------------------------------------------------

FUSED = """
    import numpy as np

    import ppy


    @ppy.pure
    @ppy.opt(3)
    def blend(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.sin(a) * 2.0 + np.cos(b)


    @ppy.pure
    def energy(a: np.ndarray) -> float:
        return np.sum(a * a)
    """


def test_elementwise_expressions_are_fused_into_one_kernel(write, analyze):
    path = write("fused.ppy", FUSED)
    module = _collect(analyze(path, backend="llvm"))["fused"]
    assert len(module.fused) == 2
    blend = next(loop for loop in module.fused.values() if not loop.returns_scalar)
    assert blend.arrays == ("a", "b")
    assert not blend.scalars

    notes = " ".join(note for _line, note in module.fusion_notes)
    assert "sin" in notes and "cos" in notes and "multiply" in notes


def test_a_fused_reduction_returns_a_scalar(write, analyze):
    path = write("reduce.ppy", FUSED)
    module = _collect(analyze(path, backend="llvm"))["reduce"]
    energy = next(loop for loop in module.fused.values() if loop.returns_scalar)
    assert energy.reduction == "sum"
    assert energy.arrays == ("a",)


def test_fused_kernel_has_one_loop_and_no_temporaries(write, analyze):
    path = write("kernel.ppy", FUSED)
    ir = emit_ir(analyze(path, backend="llvm"))["kernel"]
    assert "@llvm.sin.f64" in ir
    assert "@llvm.cos.f64" in ir
    assert "fadd fast" not in ir


def test_fused_expression_is_replaced_with_its_kernel_call(write, analyze):
    from ppy_compiler.driver.pipeline import build_python

    path = write("rewrite.ppy", FUSED)
    bundle = analyze(path, backend="llvm")
    natives = _collect(bundle)
    output = build_python(
        bundle, target="llvm", fusion={n: m.fusion_plan for n, m in natives.items()}
    )
    code = output.generated["rewrite"].code
    assert "__ppy_bind_fused__" in code
    assert "return _ppy_fused_0(a, b)" in code
    # The original expression survives as the fallback the guards fall back to.
    assert "lambda a, b: np.sin(a) * 2.0 + np.cos(b)" in code


def test_the_python_backend_leaves_fusion_alone(write, analyze):
    from ppy_compiler.driver.pipeline import build_python

    path = write("plain.ppy", FUSED)
    code = build_python(analyze(path)).generated["plain"].code
    assert "__ppy_bind_fused__" not in code
    assert "np.sin(a) * 2.0 + np.cos(b)" in code


@requires_numpy
def test_fused_kernels_run_and_match_numpy(write, analyze):
    import numpy

    from ppy_compiler.backend.llvm.fused_runtime import bind_fused

    path = write("run_fused.ppy", FUSED)
    module = _collect(analyze(path, backend="llvm"))["run_fused"]
    engine = _jit(module)

    values = numpy.arange(64, dtype=numpy.float64)
    for symbol, loop in module.fused.items():
        address = engine.address(symbol)
        if loop.returns_scalar:
            binding = bind_fused(loop, address, lambda a: numpy.sum(a * a))
            assert binding.wrapper(values) == pytest.approx(float(numpy.sum(values * values)))
        else:
            expected = numpy.sin(values) * 2.0 + numpy.cos(values)
            binding = bind_fused(loop, address, lambda a, b: numpy.sin(a) * 2.0 + numpy.cos(b))
            numpy.testing.assert_allclose(binding.wrapper(values, values), expected)
        assert binding.calls == 1 and binding.fallbacks == 0


@requires_numpy
@pytest.mark.parametrize(
    ("make", "why"),
    [
        (lambda np: np.arange(8, dtype=np.float32), "dtype is not float64"),
        (lambda np: np.arange(16, dtype=np.float64)[::2], "not C-contiguous"),
        (lambda np: np.arange(8, dtype=np.float64).view(_Subclass), "not an exact ndarray"),
    ],
)
def test_guards_reject_values_outside_the_fast_path(write, analyze, make, why):
    import numpy

    from ppy_compiler.backend.llvm.fused_runtime import bind_fused

    path = write("guarded.ppy", FUSED)
    module = _collect(analyze(path, backend="llvm"))["guarded"]
    engine = _jit(module)
    loop = next(loop for loop in module.fused.values() if loop.returns_scalar)

    binding = bind_fused(loop, engine.address(loop.symbol), lambda a: float(numpy.sum(a * a)))
    value = make(numpy)
    assert binding.wrapper(value) == pytest.approx(float(numpy.sum(numpy.asarray(value) * numpy.asarray(value))))
    assert binding.fallbacks == 1, why


@requires_numpy
def test_mismatched_shapes_fall_back_to_numpy_broadcasting(write, analyze):
    import numpy

    from ppy_compiler.backend.llvm.fused_runtime import bind_fused

    path = write("broadcast.ppy", FUSED)
    module = _collect(analyze(path, backend="llvm"))["broadcast"]
    engine = _jit(module)
    loop = next(loop for loop in module.fused.values() if not loop.returns_scalar)

    def fallback(a, b):
        return numpy.sin(a) * 2.0 + numpy.cos(b)

    binding = bind_fused(loop, engine.address(loop.symbol), fallback)
    a = numpy.arange(6, dtype=numpy.float64).reshape(2, 3)
    b = numpy.arange(3, dtype=numpy.float64)
    numpy.testing.assert_allclose(binding.wrapper(a, b), fallback(a, b))
    assert binding.fallbacks == 1


@requires_numpy
def test_a_floating_point_condition_falls_back_to_numpy(write, analyze):
    """A generated loop raises no NumPy warning, so it must not hide one."""
    import numpy

    from ppy_compiler.backend.llvm.fused_runtime import bind_fused

    path = write("fperr.ppy", FUSED)
    module = _collect(analyze(path, backend="llvm"))["fperr"]
    engine = _jit(module)
    loop = next(loop for loop in module.fused.values() if loop.returns_scalar)

    binding = bind_fused(loop, engine.address(loop.symbol), lambda a: float(numpy.sum(a * a)))
    overflowing = numpy.array([1e200, 1e200], dtype=numpy.float64)
    with numpy.errstate(over="warn"):
        binding.wrapper(overflowing)
    assert binding.fallbacks == 1


@requires_numpy
def test_fused_program_matches_plain_cpython(tmp_path: Path):
    entry = tmp_path / "fused_run.ppy"
    entry.write_text(
        textwrap.dedent(FUSED).lstrip("\n")
        + textwrap.dedent(
            """

            def main() -> None:
                x = np.arange(64, dtype=np.float64)
                print(np.round(blend(x, x), 8).tolist())
                print(round(energy(x), 8))


            main()
            """
        ),
        encoding="utf-8",
    )
    plain = _run([sys.executable, entry.name], tmp_path)
    optimized = _ppy([entry.name], tmp_path)
    native = _ppy(["run", entry.name], tmp_path)
    assert plain.returncode == 0, plain.stderr
    assert optimized.returncode == 0, optimized.stderr
    assert native.returncode == 0, native.stderr
    assert optimized.stdout == plain.stdout
    assert native.stdout == plain.stdout


@requires_numpy
def test_a_nested_reduction_is_not_folded_into_an_elementwise_tree(write, analyze):
    """`sqrt(sum(x*x))` changes shape at the reduction, so only `sum` fuses."""
    path = write(
        "nested.ppy",
        """
        import numpy as np

        import ppy


        @ppy.pure
        def normalize(x: np.ndarray) -> np.ndarray:
            scale: float = np.sqrt(np.sum(x * x))
            return x / scale
        """,
    )
    module = _collect(analyze(path, backend="llvm"))["nested"]
    reductions = [loop for loop in module.fused.values() if loop.returns_scalar]
    elementwise = [loop for loop in module.fused.values() if not loop.returns_scalar]

    assert len(reductions) == 1
    assert reductions[0].reduction == "sum"
    # `x / scale` fuses one array against one scalar operand.
    assert len(elementwise) == 1
    assert elementwise[0].arrays == ("x",) and elementwise[0].scalars == ("scale",)


def test_a_scalar_only_call_is_not_claimed_as_fusible():
    from ppy_compiler.analysis.refinements import Facts
    from ppy_compiler.plugins.base import Lowering
    from ppy_compiler.plugins.numpy_plugin import NumPyPlugin

    from ppy_compiler.analysis import types as T

    result = NumPyPlugin().call("numpy.sqrt", [(T.FLOAT, Facts())], {})
    assert result.lowering is Lowering.DIRECT_NATIVE_CALL
    assert "no array operand" in result.reason
