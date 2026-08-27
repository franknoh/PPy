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
