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
    @ppy.fastmath
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
    """`sqrt(sum(x*x))` changes shape at the reduction, so it cannot be one tree."""
    path = write(
        "nested.ppy",
        """
        import numpy as np

        import ppy


        @ppy.pure
        @ppy.fastmath
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


@requires_numpy
def test_a_reassociating_reduction_needs_explicit_permission(write, analyze):
    """NumPy sums pairwise, so a sequential kernel would reassociate."""
    path = write(
        "strictsum.ppy",
        """
        import numpy as np

        import ppy


        @ppy.pure
        def strict(a: np.ndarray) -> float:
            return np.sum(a * a)


        @ppy.pure
        @ppy.fastmath
        def relaxed(a: np.ndarray) -> float:
            return np.sum(a * a)
        """,
    )
    module = _collect(analyze(path, backend="llvm"))["strictsum"]
    reductions = {loop.reduction for loop in module.fused.values()}
    # The elementwise `a * a` still fuses in both; only `relaxed` fuses the sum.
    assert reductions == {"", "sum"}


@requires_numpy
def test_strict_reduction_stays_bit_identical_to_numpy(tmp_path: Path):
    entry = tmp_path / "bits.ppy"
    entry.write_text(
        textwrap.dedent(
            """
            import numpy as np

            import ppy


            @ppy.pure
            def strict(a: np.ndarray) -> float:
                return np.sum(np.sqrt(a) * 2.0 + a)


            x = np.linspace(1.0, 1e6, 200000)
            print(float(strict(x)) == float(np.sum(np.sqrt(x) * 2.0 + x)))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    native = _ppy(["run", entry.name], tmp_path)
    assert native.returncode == 0, native.stderr
    assert native.stdout.strip() == "True"


@requires_numpy
def test_fused_min_max_propagate_nan_like_numpy(write, analyze):
    import numpy

    from ppy_compiler.backend.llvm.fused_runtime import bind_fused

    path = write(
        "peak.ppy",
        """
        import numpy as np

        import ppy


        @ppy.pure
        def peak(a: np.ndarray) -> float:
            return np.max(a * a)
        """,
    )
    module = _collect(analyze(path, backend="llvm"))["peak"]
    engine = _jit(module)
    loop = next(loop for loop in module.fused.values() if loop.returns_scalar)
    assert loop.reduction == "max"

    binding = bind_fused(loop, engine.address(loop.symbol), lambda a: float(numpy.max(a * a)))
    with numpy.errstate(all="ignore"):
        values = numpy.array([1.0, numpy.nan, 2.0])
        assert numpy.isnan(binding.wrapper(values))
        assert binding.wrapper(numpy.array([1.0, 3.0, 2.0])) == 9.0


@requires_numpy
def test_an_empty_fused_min_max_falls_back_so_numpy_raises(write, analyze):
    import numpy

    from ppy_compiler.backend.llvm.fused_runtime import bind_fused

    path = write(
        "emptypeak.ppy",
        """
        import numpy as np

        import ppy


        @ppy.pure
        def peak(a: np.ndarray) -> float:
            return np.max(a * a)
        """,
    )
    module = _collect(analyze(path, backend="llvm"))["emptypeak"]
    engine = _jit(module)
    loop = next(loop for loop in module.fused.values() if loop.returns_scalar)
    binding = bind_fused(loop, engine.address(loop.symbol), lambda a: float(numpy.max(a * a)))

    with pytest.raises(ValueError):
        binding.wrapper(numpy.array([], dtype=numpy.float64))
    assert binding.fallbacks == 1


# -- parallelization ------------------------------------------------------


def test_chunk_bounds_cover_the_whole_range():
    from ppy_compiler.backend.llvm.parallel import MIN_PARALLEL_ELEMENTS, chunk_bounds

    assert chunk_bounds(1000, 8) == [(0, 1000)], "a small range stays on one thread"
    assert chunk_bounds(MIN_PARALLEL_ELEMENTS * 8, 1) == [(0, MIN_PARALLEL_ELEMENTS * 8)]

    length = MIN_PARALLEL_ELEMENTS * 8
    bounds = chunk_bounds(length, 4)
    assert len(bounds) > 1
    assert bounds[0][0] == 0 and bounds[-1][1] == length
    for (_, previous_stop), (start, _) in zip(bounds, bounds[1:]):
        assert previous_stop == start


def test_only_the_parallel_directive_marks_a_kernel_splittable(write, analyze):
    path = write(
        "marked.ppy",
        """
        import numpy as np

        import ppy


        @ppy.pure
        @ppy.parallel
        def wide(a: np.ndarray) -> np.ndarray:
            return np.sin(a) + 1.0


        @ppy.pure
        def narrow(a: np.ndarray) -> np.ndarray:
            return np.sin(a) + 1.0
        """,
    )
    module = _collect(analyze(path, backend="llvm"))["marked"]
    marks = {loop.symbol.split("_marked_")[1].split("_")[0]: loop.parallel for loop in module.fused.values()}
    assert marks == {"wide": True, "narrow": False}


def test_a_reassociating_reduction_is_not_split_without_permission():
    from ppy_compiler.backend.llvm.fused_runtime import _splittable
    from ppy_compiler.backend.llvm.fusion import FusedLoop

    elementwise = FusedLoop(symbol="s", arrays=("a",), scalars=())
    assert _splittable(elementwise)
    assert _splittable(FusedLoop(symbol="s", arrays=("a",), scalars=(), reduction="max"))
    assert _splittable(FusedLoop(symbol="s", arrays=("a",), scalars=(), reduction="sum"))
    # Per-chunk means cannot be merged without weighting.
    assert not _splittable(FusedLoop(symbol="s", arrays=("a",), scalars=(), reduction="mean"))


@requires_numpy
def test_a_parallel_kernel_produces_the_same_array(write, analyze):
    import numpy

    from ppy_compiler.backend.llvm.fused_runtime import bind_fused
    from ppy_compiler.backend.llvm.parallel import MIN_PARALLEL_ELEMENTS

    path = write(
        "split.ppy",
        """
        import numpy as np

        import ppy


        @ppy.pure
        @ppy.parallel
        def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return a * b + a - b * 0.5
        """,
    )
    module = _collect(analyze(path, backend="llvm"))["split"]
    engine = _jit(module)
    loop = next(iter(module.fused.values()))
    assert loop.parallel

    def fallback(a, b):
        return a * b + a - b * 0.5

    binding = bind_fused(loop, engine.address(loop.symbol), fallback, parallel=True)
    size = MIN_PARALLEL_ELEMENTS * 4
    a = numpy.linspace(0.0, 10.0, size)
    b = numpy.linspace(1.0, 5.0, size)

    result = binding.wrapper(a, b)
    # Bit-identical, not merely close: each element is computed the same way.
    assert numpy.array_equal(result, fallback(a, b))
    assert binding.parallel_calls == 1 and binding.fallbacks == 0


@requires_numpy
def test_a_split_reduction_matches_the_serial_kernel(write, analyze):
    import numpy

    from ppy_compiler.backend.llvm.fused_runtime import bind_fused
    from ppy_compiler.backend.llvm.parallel import MIN_PARALLEL_ELEMENTS

    path = write(
        "splitmax.ppy",
        """
        import numpy as np

        import ppy


        @ppy.pure
        @ppy.parallel
        def peak(a: np.ndarray) -> float:
            return np.max(a * a)
        """,
    )
    module = _collect(analyze(path, backend="llvm"))["splitmax"]
    engine = _jit(module)
    loop = next(loop for loop in module.fused.values() if loop.returns_scalar)

    fallback = lambda a: float(numpy.max(a * a))  # noqa: E731
    values = numpy.linspace(-5.0, 5.0, MIN_PARALLEL_ELEMENTS * 4)

    parallel = bind_fused(loop, engine.address(loop.symbol), fallback, parallel=True)
    serial = bind_fused(loop, engine.address(loop.symbol), fallback, parallel=False)
    assert parallel.wrapper(values) == serial.wrapper(values) == fallback(values)
    assert parallel.parallel_calls == 1


def test_required_parallel_is_accepted_for_a_fused_region_on_llvm(write, analyze):
    path = write(
        "reqpar.ppy",
        """
        import numpy as np

        import ppy


        @ppy.pure
        @ppy.parallel(require=True)
        def kernel(a: np.ndarray) -> np.ndarray:
            return np.sin(a) + 1.0
        """,
    )
    on_llvm = analyze(path, backend="llvm")
    assert not on_llvm.diagnostics.has_errors()
    assert on_llvm.reports["reqpar.kernel"].parallel_ok

    on_python = analyze(path, backend="python")
    codes = [d.code for d in on_python.diagnostics]
    assert "E1701" in codes


# -- fixed-size tuples ----------------------------------------------------

TUPLES = """
    import ppy
    from ppy import Array


    @ppy.pure
    def norm2(point: tuple[float, float, float]) -> float:
        return point[0] * point[0] + point[1] * point[1] + point[2] * point[2]


    @ppy.pure
    def midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
        return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


    @ppy.pure
    def swap(pair: tuple[int, int]) -> tuple[int, int]:
        first, second = pair
        return (second, first)


    @ppy.pure
    def rgb_sum(colour: Array[int, 3]) -> int:
        return colour[0] + colour[1] + colour[2]
    """


def test_a_fixed_tuple_flattens_into_scalar_abi_atoms(write, analyze):
    path = write("tuples.ppy", TUPLES)
    module = _collect(analyze(path, backend="llvm"))["tuples"]
    assert set(module.functions) >= {
        "tuples.norm2", "tuples.midpoint", "tuples.swap", "tuples.rgb_sum"
    }

    norm2 = module.functions["tuples.norm2"].signature
    assert norm2.params == ("double", "double", "double")
    assert not norm2.returns_tuple

    midpoint = module.functions["tuples.midpoint"].signature
    assert midpoint.params == ("double",) * 4
    assert midpoint.returns == ("double", "double") and midpoint.returns_tuple


def test_a_ppy_array_marker_lowers_like_a_fixed_tuple(write, analyze):
    path = write("arr.ppy", TUPLES)
    module = _collect(analyze(path, backend="llvm"))["arr"]
    assert module.functions["arr.rgb_sum"].signature.params == ("i64", "i64", "i64")


def test_a_non_escaping_tuple_allocates_nothing(write, analyze):
    path = write("noalloc.ppy", TUPLES)
    ir = emit_ir(analyze(path, backend="llvm"))["noalloc"]
    body = ir.split("@ppy_noalloc_norm2")[1].split("}")[0]
    assert "alloca" not in body
    assert "getelementptr" not in body
    assert body.count("fmul") == 3


def test_a_tuple_wider_than_the_abi_limit_stays_boxed(write, analyze):
    path = write(
        "wide_tuple.ppy",
        """
        import ppy


        @ppy.pure
        def wide(t: tuple[int, int, int, int, int, int, int, int, int, int]) -> int:
            return t[0]
        """,
    )
    module = _collect(analyze(path, backend="llvm"))["wide_tuple"]
    assert "wide_tuple.wide" in module.rejected


def test_native_tuple_results_match_python(write, analyze):
    path = write("tupexec.ppy", TUPLES)
    module = _collect(analyze(path, backend="llvm"))["tupexec"]
    engine = _jit(module)

    def fallback_midpoint(a, b):
        return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)

    lowered = module.functions["tupexec.midpoint"]
    binding = bind(lowered.signature, engine.address(lowered.signature.symbol), fallback_midpoint)
    assert binding.wrapper((0.0, 0.0), (4.0, 6.0)) == (2.0, 3.0)
    assert binding.calls == 1 and binding.fallbacks == 0


def test_a_tuple_element_outside_the_machine_range_falls_back(write, analyze):
    path = write("tupguard.ppy", TUPLES)
    module = _collect(analyze(path, backend="llvm"))["tupguard"]
    engine = _jit(module)

    def fallback_swap(pair):
        return (pair[1], pair[0])

    lowered = module.functions["tupguard.swap"]
    binding = bind(lowered.signature, engine.address(lowered.signature.symbol), fallback_swap)
    assert binding.wrapper((1, 2)) == (2, 1)
    assert binding.calls == 1

    assert binding.wrapper((10**30, 1)) == (1, 10**30)
    assert binding.fallbacks == 1


def test_a_non_tuple_argument_falls_back(write, analyze):
    path = write("tuptype.ppy", TUPLES)
    module = _collect(analyze(path, backend="llvm"))["tuptype"]
    engine = _jit(module)
    lowered = module.functions["tuptype.swap"]
    binding = bind(lowered.signature, engine.address(lowered.signature.symbol), lambda p: (p[1], p[0]))

    assert binding.wrapper([1, 2]) == (2, 1)
    assert binding.fallbacks == 1


def test_tuple_programs_match_plain_cpython(tmp_path: Path):
    entry = tmp_path / "tuples_run.ppy"
    entry.write_text(
        textwrap.dedent(TUPLES).lstrip("\n")
        + textwrap.dedent(
            """

            print(norm2((1.0, 2.0, 3.0)))
            print(midpoint((0.0, 0.0), (4.0, 6.0)))
            print(swap((1, 2)), swap((10 ** 30, 1)))
            print(rgb_sum((255, 128, 0)), len((1, 2, 3)))
            """
        ),
        encoding="utf-8",
    )
    plain = _run([sys.executable, entry.name], tmp_path)
    native = _ppy(["run", entry.name], tmp_path)
    assert plain.returncode == 0, plain.stderr
    assert native.returncode == 0, native.stderr
    assert native.stdout == plain.stdout


def test_the_fusion_tables_match_what_the_plugin_claims():
    """A plugin must not advertise a kernel the backend cannot generate."""
    from ppy_compiler.backend.llvm.fusion import BINARY, REDUCTIONS, UNARY
    from ppy_compiler.plugins.numpy_plugin import (
        FUSIBLE_BINARY,
        FUSIBLE_REDUCTIONS,
        FUSIBLE_UNARY,
    )

    assert set(UNARY) == set(FUSIBLE_UNARY)
    assert BINARY == set(FUSIBLE_BINARY)
    assert REDUCTIONS == set(FUSIBLE_REDUCTIONS)


# -- borrowed buffers -----------------------------------------------------

BUFFERS_VIEW = """
    import ppy
    from ppy import Buffer


    @ppy.pure
    @ppy.opt(3)
    def total(values: Buffer[float]) -> float:
        result: float = 0.0
        for i in range(len(values)):
            result += values[i]
        return result


    @ppy.pure
    def counts(values: Buffer[int]) -> int:
        result: int = 0
        for value in values:
            result += value
        return result
    """


def test_a_buffer_parameter_is_borrowed_not_copied(write, analyze):
    path = write("view.ppy", BUFFERS_VIEW)
    module = _collect(analyze(path, backend="llvm"))["view"]
    signature = module.functions["view.total"].signature
    assert signature.parameters[0].is_buffer
    assert signature.parameters[0].is_borrowed
    assert "borrowed" in str(signature)


def test_a_list_parameter_is_copied_not_borrowed(write, analyze):
    path = write("copied.ppy", BUFFERS)
    module = _collect(analyze(path, backend="llvm"))["copied"]
    parameter = module.functions["copied.total"].signature.parameters[0]
    assert parameter.is_buffer and not parameter.is_borrowed


def test_the_borrowed_pointer_is_the_callers_own_memory(write, analyze):
    """No copy: the native code reads the array the caller already had."""
    import array
    import ctypes

    from ppy_compiler.backend.llvm.runtime import _expander_for

    path = write("borrow.ppy", BUFFERS_VIEW)
    module = _collect(analyze(path, backend="llvm"))["borrow"]
    parameter = module.functions["borrow.total"].signature.parameters[0]

    values = array.array("d", [1.0, 2.0, 3.0])
    atoms: list = []
    borrowed: list = []
    _expander_for(parameter)(values, atoms, borrowed)

    address = ctypes.cast(atoms[0], ctypes.c_void_p).value
    assert address == values.buffer_info()[0]
    assert atoms[1] == 3


def test_borrowed_buffer_guards_reject_what_cannot_be_pointed_at(write, analyze):
    import array

    from ppy_compiler.backend.llvm.runtime import GuardFailed, _expander_for

    path = write("guardview.ppy", BUFFERS_VIEW)
    module = _collect(analyze(path, backend="llvm"))["guardview"]
    expand = _expander_for(module.functions["guardview.total"].signature.parameters[0])

    for value in (
        [1.0, 2.0],                       # a list has no contiguous buffer
        array.array("q", [1, 2]),         # wrong element format
        b"\x00" * 16,                     # read-only
        memoryview(array.array("d", [1.0, 2.0, 3.0, 4.0]))[::2],  # not contiguous
    ):
        with pytest.raises(GuardFailed):
            expand(value, [], [])


def test_borrowed_buffer_results_match_python(write, analyze):
    import array

    path = write("viewexec.ppy", BUFFERS_VIEW)
    module = _collect(analyze(path, backend="llvm"))["viewexec"]
    engine = _jit(module)

    lowered = module.functions["viewexec.total"]
    binding = bind(lowered.signature, engine.address(lowered.signature.symbol), sum)
    values = array.array("d", [1.5, 2.5, 3.0])
    assert binding.wrapper(values) == 7.0
    assert binding.calls == 1 and binding.fallbacks == 0

    # Borrowed, so a later mutation is visible without rebinding.
    values[0] = 10.5
    assert binding.wrapper(values) == 16.0


def test_an_empty_borrowed_buffer_is_accepted(write, analyze):
    import array

    path = write("emptyview.ppy", BUFFERS_VIEW)
    module = _collect(analyze(path, backend="llvm"))["emptyview"]
    engine = _jit(module)
    lowered = module.functions["emptyview.total"]
    binding = bind(lowered.signature, engine.address(lowered.signature.symbol), sum)
    assert binding.wrapper(array.array("d", [])) == 0.0
    assert binding.fallbacks == 0


# -- explicit floating-point relaxation -----------------------------------

FP = """
    import ppy
    from ppy import Buffer


    @ppy.pure
    @ppy.opt(3)
    def strict(a: Buffer[float], b: Buffer[float]) -> float:
        total: float = 0.0
        for i in range(len(a)):
            total += a[i] * b[i]
        return total


    @ppy.pure
    @ppy.fastmath
    @ppy.opt(3)
    def relaxed(a: Buffer[float], b: Buffer[float]) -> float:
        total: float = 0.0
        for i in range(len(a)):
            total += a[i] * b[i]
        return total
    """


def test_only_the_fastmath_function_gets_relaxed_arithmetic(write, analyze):
    path = write("fp.ppy", FP)
    ir = emit_ir(analyze(path, backend="llvm", opt_level=3))["fp"]

    strict_body = ir.split("@ppy_fp_strict")[1].split("\ndefine")[0]
    relaxed_body = ir.split("@ppy_fp_relaxed")[1].split("\ndefine")[0]

    assert "fast" not in strict_body and "reassoc" not in strict_body
    assert "fast" in relaxed_body


def test_optimization_level_alone_never_relaxes_arithmetic(write, analyze):
    """O3 is not permission: only the directive is (spec 3.4, 12.5, 14.5)."""
    path = write(
        "o3only.ppy",
        """
        import ppy
        from ppy import Buffer


        @ppy.pure
        @ppy.opt(3)
        def total(a: Buffer[float]) -> float:
            result: float = 0.0
            for i in range(len(a)):
                result += a[i] * a[i]
            return result
        """,
    )
    for level in (0, 1, 2, 3):
        ir = emit_ir(analyze(path, backend="llvm", opt_level=level))["o3only"]
        assert "fast" not in ir and "reassoc" not in ir, f"O{level} relaxed the arithmetic"


def test_relaxed_arithmetic_vectorizes_the_reduction(write, analyze):
    path = write("vec.ppy", FP)
    ir = emit_ir(analyze(path, backend="llvm", opt_level=3))["vec"]
    relaxed_body = ir.split("@ppy_vec_relaxed")[1].split("\ndefine")[0]
    assert "x double>" in relaxed_body, "a permitted reassociation should vectorize"


def test_both_orderings_agree_to_within_rounding(write, analyze):
    import array

    path = write("fpexec.ppy", FP)
    module = _collect(analyze(path, backend="llvm"))["fpexec"]
    engine = _jit(module)

    def reference(a, b):
        return sum(x * y for x, y in zip(a, b))

    x = array.array("d", [i * 0.001 for i in range(2048)])
    y = array.array("d", [i * 0.002 for i in range(2048)])

    results = {}
    for name in ("strict", "relaxed"):
        lowered = module.functions[f"fpexec.{name}"]
        binding = bind(lowered.signature, engine.address(lowered.signature.symbol), reference)
        results[name] = binding.wrapper(x, y)
        assert binding.calls == 1

    assert results["strict"] == pytest.approx(results["relaxed"], rel=1e-9)


# -- JIT specialization ---------------------------------------------------

JIT = """
    import ppy
    from ppy import Buffer


    @ppy.jit(threshold=3, max_specializations=2)
    @ppy.pure
    @ppy.opt(3)
    def digest(values: Buffer[int], modulus: int) -> int:
        total: int = 0
        for i in range(len(values)):
            total += values[i] % modulus
        return total
    """


def _digest(values, modulus):
    return sum(v % modulus for v in values)


def _jit_binding(write, analyze, name="jitmod"):
    import array

    from ppy_compiler.backend.llvm.specialize import SpecializationPolicy, Specializer

    path = write(f"{name}.ppy", JIT)
    bundle = analyze(path, backend="llvm")
    module = _collect(bundle)[name]
    engine = _jit(module)

    specializer = Specializer(engine=engine, module_analysis=bundle.analysis.modules[name])
    for qualname, (info, node) in module.sources.items():
        specializer.register(info, node)

    lowered = module.functions[f"{name}.digest"]
    binding = bind(
        lowered.signature,
        engine.address(lowered.signature.symbol),
        _digest,
        specializer=specializer,
        policy=SpecializationPolicy.of(lowered.info),
        info=lowered.info,
    )
    return binding, array.array("q", [i * 7919 for i in range(64)])


def test_the_jit_directive_is_what_enables_specialization(write, analyze):
    from ppy_compiler.analysis.symbols import Directive, FunctionInfo
    from ppy_compiler.backend.llvm.specialize import SpecializationPolicy

    class _Info:
        def __init__(self, directives):
            self._directives = directives

        def directive(self, name):
            return self._directives.get(name)

    assert not SpecializationPolicy.of(_Info({})).enabled

    policy = SpecializationPolicy.of(_Info({"jit": Directive("jit", {"max_specializations": 2})}))
    assert policy.enabled and policy.maximum == 2

    # `@ppy.specialize` asks for it immediately rather than after a warm-up.
    eager = SpecializationPolicy.of(_Info({"specialize": Directive("specialize", {})}))
    assert eager.enabled and eager.threshold == 1


def test_a_specialization_appears_only_after_the_shape_repeats(write, analyze):
    binding, values = _jit_binding(write, analyze)

    for _ in range(2):
        assert binding.wrapper(values, 1000003) == _digest(values, 1000003)
    assert binding.specialization_count == 0, "two calls is below the threshold of three"

    assert binding.wrapper(values, 1000003) == _digest(values, 1000003)
    assert binding.specialization_count == 1
    assert binding.specialized_calls >= 1


def test_a_specialization_is_reused_for_the_shape_it_was_built_for(write, analyze):
    binding, values = _jit_binding(write, analyze, "jitreuse")

    for _ in range(6):
        binding.wrapper(values, 1000003)
    assert binding.specialization_count == 1
    before = binding.specialized_calls

    for _ in range(4):
        assert binding.wrapper(values, 1000003) == _digest(values, 1000003)
    assert binding.specialized_calls == before + 4


def test_another_shape_uses_the_generic_code(write, analyze):
    binding, values = _jit_binding(write, analyze, "jitother")

    for _ in range(4):
        binding.wrapper(values, 1000003)
    assert binding.specialization_count == 1
    specialized = binding.specialized_calls

    # A different constant does not match the guard, so the generic code runs.
    assert binding.wrapper(values, 97) == _digest(values, 97)
    assert binding.specialized_calls == specialized
    assert binding.fallbacks == 0, "the generic native code, not the Python path"


def test_specializations_are_capped(write, analyze):
    binding, values = _jit_binding(write, analyze, "jitcap")

    for modulus in (7, 11, 13, 17, 19):
        for _ in range(4):
            assert binding.wrapper(values, modulus) == _digest(values, modulus)
    assert binding.specialization_count <= 2, "max_specializations must bound code growth"


def test_watching_stops_when_the_arguments_never_settle(write, analyze):
    binding, values = _jit_binding(write, analyze, "jitbudget")

    for modulus in range(3, 3 + 200):
        binding.wrapper(values, modulus)
    assert not binding.observing, "a function whose shape never repeats stops paying"


def test_a_specialization_pins_the_values_it_guards_on(write, analyze):
    from ppy_compiler.backend.llvm.specialize import SpecializationKey

    key = SpecializationKey(((1, "modulus", "value", 97), (0, "values", "length", 64)))
    assert key.constants == {"modulus": 97, "len(values)": 64}

    matches = key.matcher()
    assert matches(([0] * 64, 97))
    assert not matches(([0] * 64, 98))
    assert not matches(([0] * 63, 97))


def test_a_buffer_selects_a_native_vector_representation():
    from ppy_compiler.analysis import types as T
    from ppy_compiler.analysis.refinements import Facts
    from ppy_compiler.analysis.representation import Repr, select

    chosen = select(T.instance("Buffer", T.FLOAT), Facts(contiguous=True))
    assert chosen.repr is Repr.NATIVE_VECTOR
    assert "borrowed" in chosen.reason

    # A buffer whose element type is unknown has no native layout.
    opaque = select(T.instance("Buffer", T.ANY), Facts())
    assert opaque.repr is Repr.PY_OBJECT
