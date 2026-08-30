"""Native buffers and fused library kernels (spec 13.3, 16, 19.4)."""

from __future__ import annotations

import array
import importlib.util
import itertools
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import _collect, emit_ir
from ppy_compiler.backend.llvm import available as llvm_available
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


def test_floor_division_by_a_power_of_two_lowers_to_shifts(write, analyze):
    """`n // 2` is `ashr` and `n % 2` is `and` -- floor semantics make them exact.

    No `sdiv`/`srem` may survive: the correction chain they drag along is
    what kept the hot loop twice as slow as the same loop in C.
    """
    path = write(
        "parity.ppy",
        """
        import ppy


        @ppy.pure
        def halve(n: int) -> int:
            return n // 2


        @ppy.pure
        def parity(n: int) -> int:
            return n % 8


        print(halve(-7), parity(-9))
        """,
    )
    bundle = analyze(path)
    native = _collect(bundle, 2)["parity"]
    assert "sdiv" not in native.ir and "srem" not in native.ir
    assert "ashr" in native.ir


def test_multiplied_index_guards_hoist_out_of_the_loop(write, analyze):
    """`i * n + k` proves its extremes once per loop entry, not per iteration.

    The body keeps a plain `mul nsw`; the corner checks live in the loop's
    guard block. `safeguards = "inline"` restores the per-operation guards.
    """
    path = write(
        "kernel.ppy",
        """
        import ppy
        from ppy import Buffer


        @ppy.opt(3)
        def total(a: Buffer[float], n: int) -> float:
            out: float = 0.0
            for i in range(n):
                for k in range(n):
                    out += a[i * n + k]
            return out
        """,
    )
    bundle = analyze(path)
    native = _collect(bundle, 2)["kernel"]
    assert "for.guards" in native.ir
    assert "mul nsw" in native.ir

    bundle.project.config.llvm.safeguards = "inline"
    inline = _collect(bundle, 2)["kernel"]
    assert "for.guards" not in inline.ir
    assert "mul nsw" not in inline.ir


def test_profitability_keeps_tiny_functions_off_the_boundary(write, analyze):
    """`can_lower_native` and `should_lower_native` answer different questions."""
    from ppy_compiler.backend.llvm.lowering import should_lower_native

    path = write(
        "sizes.ppy",
        """
        import ppy
        from ppy import Buffer


        @ppy.pure
        def add(x: int, y: int) -> int:
            return x + y


        @ppy.native
        def demanded(x: int) -> int:
            return x + 1


        @ppy.pure
        def looped(n: int) -> int:
            out: int = 0
            for i in range(n):
                out += i
            return out


        @ppy.pure
        def summed(xs: Buffer[int]) -> int:
            return sum(xs)


        print(add(1, 2), demanded(1), looped(3))
        """,
    )
    bundle = analyze(path)
    module = bundle.analysis.modules["sizes"]
    infos = bundle.symbols.modules["sizes"].functions

    def verdict(name):
        info = infos[name]
        return should_lower_native(info, module.functions[info.qualname])[0]

    assert not verdict("add")
    assert verdict("demanded")
    assert verdict("looped")
    assert verdict("summed")


def test_unexposed_helpers_are_still_called_natively(tmp_path: Path):
    """Off the boundary is not out of the binary: native callers go direct."""
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    (tmp_path / "tiny.ppy").write_text(
        textwrap.dedent(
            """
            import ppy


            @ppy.pure
            def add(x: int, y: int) -> int:
                return x + y


            @ppy.pure
            @ppy.opt(3)
            def total(n: int) -> int:
                out: int = 0
                for i in range(n):
                    out += add(i, i)
                return out


            print(total(100000), add(2, 3))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    plain = _run([sys.executable, "tiny.ppy"], tmp_path)
    native = _ppy(["run", "tiny.ppy"], tmp_path)
    assert native.returncode == 0, native.stderr
    assert native.stdout == plain.stdout


WRAPPING = """
import ppy


@ppy.pure
def spin(n: int) -> int:
    value: int = 3
    for _i in range(n):
        value = value * 2654435761 + 1
    return value


print(spin(41))
"""


def test_safe_mode_keeps_python_integers_and_unsafe_mode_wraps(tmp_path: Path):
    """The `--unsafe` contract, pinned: 64-bit wrap on data arithmetic.

    Safe mode overflows into CPython's arbitrary precision, bit for bit;
    unsafe mode is the deterministic machine result every wrap-semantics
    compiler produces.
    """
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    (tmp_path / "spin.ppy").write_text(textwrap.dedent(WRAPPING).lstrip("\n"), encoding="utf-8")
    plain = _run([sys.executable, "spin.ppy"], tmp_path)
    safe = _ppy(["run", "spin.ppy"], tmp_path)
    unsafe = _ppy(["run", "--unsafe", "spin.ppy"], tmp_path)
    assert safe.stdout == plain.stdout
    assert unsafe.stdout.strip() == "8606135309836935036"


def test_unsafe_mode_keeps_bounds_semantics(tmp_path: Path):
    """Unsafe drops overflow guards, never memory safety."""
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    (tmp_path / "oob.ppy").write_text(
        textwrap.dedent(
            """
            import array

            import ppy
            from ppy import Buffer


            @ppy.opt(3)
            def pick(xs: Buffer[int], index: int) -> int:
                return xs[index]


            xs = array.array("q", [1, 2, 3])
            print(pick(xs, 1))
            try:
                pick(xs, 7)
            except IndexError as error:
                print("IndexError")
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    plain = _run([sys.executable, "oob.ppy"], tmp_path)
    unsafe = _ppy(["run", "--unsafe", "oob.ppy"], tmp_path)
    assert unsafe.returncode == 0, unsafe.stderr
    assert unsafe.stdout == plain.stdout


def test_hoisted_guards_keep_python_semantics_at_the_extremes(tmp_path: Path):
    """A bailed preflight is CPython, bit for bit -- IndexError included."""
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    (tmp_path / "extremes.ppy").write_text(
        textwrap.dedent(
            """
            import array

            import ppy
            from ppy import Buffer


            @ppy.opt(3)
            def gather(xs: Buffer[int], n: int, scale: int) -> int:
                total: int = 0
                for i in range(n):
                    total += xs[i * scale]
                return total


            def main() -> None:
                xs = array.array("q", list(range(16)))
                print(gather(xs, 4, 3))
                for scale in (4611686018427387904, -3):
                    try:
                        gather(xs, 4, scale)
                    except IndexError as error:
                        print(type(error).__name__, scale < 0)


            main()
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    plain = _run([sys.executable, "extremes.ppy"], tmp_path)
    native = _ppy(["run", "extremes.ppy"], tmp_path)
    assert native.returncode == 0, native.stderr
    assert native.stdout == plain.stdout


def test_power_of_two_floor_semantics_match_python_on_every_sign(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    (tmp_path / "negdiv.ppy").write_text(
        textwrap.dedent(
            """
            import ppy


            @ppy.pure
            @ppy.opt(3)
            def probe(values: list[int]) -> list[int]:
                out: list[int] = []
                for v in values:
                    out.append(v // 2)
                    out.append(v % 2)
                    out.append(v // 8)
                    out.append(v % 8)
                    out.append(v // 1)
                    out.append(v % 1)
                    out.append(v // 4096)
                    out.append(v % 4096)
                return out


            def main() -> None:
                edges: list[int] = [
                    -9223372036854775808, -4097, -4096, -9, -8, -7, -2, -1,
                    0, 1, 2, 7, 8, 9, 4095, 4096, 9223372036854775807,
                ]
                print(probe(edges))


            main()
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    plain = _run([sys.executable, "negdiv.ppy"], tmp_path)
    native = _ppy(["run", "negdiv.ppy"], tmp_path)
    assert native.returncode == 0, native.stderr
    assert native.stdout == plain.stdout


def test_list_parameters_are_lowered_to_native_buffers(write, analyze):
    path = write("buffers.ppy", BUFFERS)
    module = _collect(analyze(path, backend="llvm"))["buffers"]
    assert set(module.functions) >= {
        "buffers.dot",
        "buffers.total",
        "buffers.peak",
        "buffers.add_up",
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
        return sum(x * y for x, y in zip(a, b, strict=False))

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
    assert binding.wrapper(value) == pytest.approx(
        float(numpy.sum(numpy.asarray(value) * numpy.asarray(value)))
    )
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
    from ppy_compiler.analysis import types as T
    from ppy_compiler.analysis.refinements import Facts
    from ppy_compiler.plugins.base import Lowering
    from ppy_compiler.plugins.numpy_plugin import NumPyPlugin

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
    for (_, previous_stop), (start, _) in itertools.pairwise(bounds):
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
    marks = {
        loop.symbol.split("_marked_")[1].split("_")[0]: loop.parallel
        for loop in module.fused.values()
    }
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

    def fallback(a):
        return float(numpy.max(a * a))

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
        "tuples.norm2",
        "tuples.midpoint",
        "tuples.swap",
        "tuples.rgb_sum",
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
    binding = bind(
        lowered.signature, engine.address(lowered.signature.symbol), lambda p: (p[1], p[0])
    )

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
    assert set(FUSIBLE_BINARY) == BINARY
    assert set(FUSIBLE_REDUCTIONS) == REDUCTIONS


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
    import ctypes

    from ppy_runtime.binding import _expander_for

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
    from ppy_runtime.binding import GuardFailed, _expander_for

    path = write("guardview.ppy", BUFFERS_VIEW)
    module = _collect(analyze(path, backend="llvm"))["guardview"]
    expand = _expander_for(module.functions["guardview.total"].signature.parameters[0])

    for value in (
        [1.0, 2.0],  # a list has no contiguous buffer
        array.array("q", [1, 2]),  # wrong element format
        b"\x00" * 16,  # read-only
        memoryview(array.array("d", [1.0, 2.0, 3.0, 4.0]))[::2],  # not contiguous
    ):
        with pytest.raises(GuardFailed):
            expand(value, [], [])


def test_borrowed_buffer_results_match_python(write, analyze):
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
    path = write("fpexec.ppy", FP)
    module = _collect(analyze(path, backend="llvm"))["fpexec"]
    engine = _jit(module)

    def reference(a, b):
        return sum(x * y for x, y in zip(a, b, strict=False))

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
    from ppy_compiler.backend.llvm.specialize import SpecializationPolicy, Specializer

    path = write(f"{name}.ppy", JIT)
    bundle = analyze(path, backend="llvm")
    module = _collect(bundle)[name]
    engine = _jit(module)

    specializer = Specializer(engine=engine, module_analysis=bundle.analysis.modules[name])
    for info, node in module.sources.values():
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
    from ppy_compiler.analysis.symbols import Directive
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


# -- value classes --------------------------------------------------------

VALUE_CLASSES = """
    from dataclasses import dataclass

    import ppy


    @dataclass
    class Point:
        x: float
        y: float
        z: float

        @ppy.pure
        @ppy.opt(3)
        def norm2(self) -> float:
            return self.x * self.x + self.y * self.y + self.z * self.z


    class Counter:
        total: int
        step: int

        def __init__(self, total: int, step: int) -> None:
            self.total = total
            self.step = step


    @ppy.pure
    @ppy.opt(3)
    def distance2(a: Point, b: Point) -> float:
        dx: float = a.x - b.x
        dy: float = a.y - b.y
        dz: float = a.z - b.z
        return dx * dx + dy * dy + dz * dz


    @ppy.pure
    def advance(c: Counter, times: int) -> int:
        return c.total + c.step * times
    """


def _layouts(bundle):  # type: ignore[no-untyped-def]
    from ppy_compiler.backend.llvm import _value_class_layouts

    return _value_class_layouts(bundle)


def test_a_value_class_flattens_into_scalar_arguments(write, analyze):
    path = write("values.ppy", VALUE_CLASSES)
    bundle = analyze(path, backend="llvm")
    assert _layouts(bundle)["values.Point"] == (("x", "float"), ("y", "float"), ("z", "float"))
    assert _layouts(bundle)["values.Counter"] == (("total", "int"), ("step", "int"))

    module = _collect(bundle)["values"]
    signature = module.functions["values.distance2"].signature
    assert signature.params == ("double",) * 6
    assert signature.parameters[0].is_object
    assert signature.parameters[0].class_name == "values.Point"
    assert "double a_x" in str(signature)


def test_a_method_receives_its_receiver_flattened(write, analyze):
    path = write("methods.ppy", VALUE_CLASSES)
    module = _collect(analyze(path, backend="llvm"))["methods"]
    assert "methods.Point.norm2" in module.functions
    assert module.functions["methods.Point.norm2"].signature.params == ("double",) * 3


def test_a_class_with_a_non_scalar_field_stays_boxed(write, analyze):
    path = write(
        "boxed.ppy",
        """
        import ppy


        class Holder:
            name: str
            size: int

            def __init__(self, name: str, size: int) -> None:
                self.name = name
                self.size = size


        @ppy.pure
        def size_of(h: Holder) -> int:
            return h.size
        """,
    )
    bundle = analyze(path, backend="llvm")
    assert "boxed.Holder" not in _layouts(bundle)
    assert "boxed.size_of" in _collect(bundle)["boxed"].rejected


def test_a_subclassed_value_class_is_not_flattened(write, analyze):
    path = write(
        "derived.ppy",
        """
        from dataclasses import dataclass


        @dataclass
        class Base:
            x: float


        @dataclass
        class Derived(Base):
            y: float
        """,
    )
    layouts = _layouts(analyze(path, backend="llvm"))
    assert "derived.Base" in layouts
    # A subclass may add state or override attribute access, so it is left alone.
    assert "derived.Derived" not in layouts


def test_writing_a_field_keeps_the_function_boxed(write, analyze):
    path = write(
        "mutating.ppy",
        """
        from dataclasses import dataclass


        @dataclass
        class Cell:
            value: float


        def bump(c: Cell, by: float) -> float:
            c.value = c.value + by
            return c.value
        """,
    )
    module = _collect(analyze(path, backend="llvm"))["mutating"]
    assert "mutating.bump" in module.rejected


def test_value_class_results_match_python(write, analyze):
    path = write("valexec.ppy", VALUE_CLASSES)
    module = _collect(analyze(path, backend="llvm"))["valexec"]
    engine = _jit(module)

    class Point:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    def fallback(a, b):
        return (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2

    lowered = module.functions["valexec.distance2"]
    binding = bind(lowered.signature, engine.address(lowered.signature.symbol), fallback)
    # The class the layout was built for lives in the fallback's module, which
    # this stand-in is not, so the guard sends the call to Python.
    assert binding.wrapper(Point(1.0, 2.0, 3.0), Point(4.0, 6.0, 3.0)) == 25.0
    assert binding.fallbacks == 1


def test_value_class_programs_match_plain_cpython(tmp_path: Path):
    entry = tmp_path / "values_run.ppy"
    entry.write_text(
        textwrap.dedent(VALUE_CLASSES).lstrip("\n")
        + textwrap.dedent(
            """

            a = Point(1.0, 2.0, 3.0)
            b = Point(4.0, 6.0, 3.0)
            print(a.norm2(), distance2(a, b), advance(Counter(10, 3), 4))
            """
        ),
        encoding="utf-8",
    )
    plain = _run([sys.executable, entry.name], tmp_path)
    native = _ppy(["run", entry.name], tmp_path)
    assert plain.returncode == 0, plain.stderr
    assert native.returncode == 0, native.stderr
    assert native.stdout == plain.stdout


def test_a_flattened_value_class_selects_an_aggregate_representation():
    from ppy_compiler.analysis import types as T
    from ppy_compiler.analysis.refinements import Facts
    from ppy_compiler.analysis.representation import Repr, select

    layouts = {"m.Point": (("x", "float"), ("y", "float"))}
    chosen = select(T.Instance("m.Point", (), ("m.Point", "object")), Facts(), layouts=layouts)
    assert chosen.repr is Repr.AGGREGATE
    assert "2 scalar(s)" in chosen.reason

    escaping = select(
        T.Instance("m.Point", (), ("m.Point", "object")),
        Facts(),
        escapes=True,
        layouts=layouts,
    )
    assert escaping.repr is Repr.PY_OBJECT


# -- the generated CPython-ABI boundary -----------------------------------


def test_the_wrapper_toolchain_is_reported():
    from ppy_compiler.backend.llvm.wrapper_build import wrapper_toolchain

    ready, detail = wrapper_toolchain()
    assert isinstance(ready, bool) and detail


def test_the_generator_emits_one_entry_point_per_function(write, analyze):
    from ppy_compiler.backend.llvm.wrapper import generate

    path = write("wrapgen.ppy", BUFFERS)
    module = _collect(analyze(path, backend="llvm"))["wrapgen"]
    signatures = {q: lowered.signature for q, lowered in module.functions.items()}

    built = generate("ppy_wrappers_test", signatures)
    assert set(built.entries) == set(signatures)
    for index in built.entries.values():
        assert f"ppy_call_{index}" in built.source
        assert f"ppy_bind_{index}" in built.source
    assert "METH_FASTCALL" in built.source
    assert "Py_RETURN_NOTIMPLEMENTED" in built.source
    assert "PyInit_ppy_wrappers_test" in built.source


def test_the_generator_covers_every_parameter_shape(write, analyze):
    from ppy_compiler.backend.llvm.wrapper import generate

    path = write(
        "shapes.ppy",
        """
        from dataclasses import dataclass

        import ppy
        from ppy import Buffer


        @dataclass
        class Point:
            x: float
            y: float


        @ppy.pure
        def mixed(
            scalar: int,
            pair: tuple[float, float],
            point: Point,
            copied: list[float],
            borrowed: Buffer[float],
        ) -> float:
            return scalar + pair[0] + point.x + copied[0] + borrowed[0]
        """,
    )
    module = _collect(analyze(path, backend="llvm"))["shapes"]
    assert "shapes.mixed" in module.functions, module.rejected

    source = generate(
        "ppy_wrappers_shapes", {"shapes.mixed": module.functions["shapes.mixed"].signature}
    ).source
    assert "PyLong_AsLongLong" in source  # a scalar int
    assert "PyTuple_GET_ITEM" in source  # a fixed tuple
    assert "PyObject_GetAttr(" in source  # a value class field
    assert "PyUnicode_InternFromString" in source  # its name, interned once
    assert "PyMem_Malloc" in source  # a copied list
    assert "PyObject_GetBuffer" in source  # a borrowed buffer
    assert "PyBuffer_Release" in source


def _wrapped(write, analyze, name, source, function):
    """Build a module, compile its wrapper, and bind one entry point."""
    from ppy_compiler.backend.llvm.wrapper_build import build_wrappers, wrapper_toolchain

    ready, detail = wrapper_toolchain()
    if not ready:
        pytest.skip(detail)

    path = write(f"{name}.ppy", source)
    module = _collect(analyze(path, backend="llvm"))[name]
    qualname = f"{name}.{function}"
    assert qualname in module.functions, module.rejected
    engine = _jit(module)

    built = build_wrappers(
        name,
        {q: lowered.signature for q, lowered in module.functions.items()},
        path.parent / ".ppy-cache",
    )
    if not built.ok:
        pytest.skip(built.reason)

    lowered = module.functions[qualname]
    entry = built.bind(qualname, engine.address(lowered.signature.symbol), ())
    assert entry is not None
    # The engine owns the compiled code: dropping it would free what the
    # wrapper points at, so it is handed back with the entry.
    return entry, (engine, built)


SCALARS = """
    import ppy


    @ppy.pure
    @ppy.opt(3)
    def poly(x: float, k: int) -> float:
        return x * float(k) + 1.0


    @ppy.pure
    def swap(pair: tuple[int, int]) -> tuple[int, int]:
        first, second = pair
        return (second, first)


    @ppy.pure
    def widen(x: int) -> int:
        return x * x
    """


def test_the_wrapper_computes_the_same_value(write, analyze):
    entry, _owner = _wrapped(write, analyze, "wrapcall", SCALARS, "poly")
    assert entry(2.0, 3) == 7.0
    # An exact int is accepted where a float is expected, as Python does.
    assert entry(2, 3) == 7.0


def test_the_wrapper_returns_a_tuple_result(write, analyze):
    entry, _owner = _wrapped(write, analyze, "wraptuple", SCALARS, "swap")
    assert entry((1, 2)) == (2, 1)


def test_a_wrong_argument_type_asks_for_the_python_path(write, analyze):
    entry, _owner = _wrapped(write, analyze, "wrapguard", SCALARS, "poly")
    assert entry("two", 3) is NotImplemented
    assert entry(2.0, 3.5) is NotImplemented
    assert entry(2.0) is NotImplemented, "the wrong arity is a guard failure too"


def test_an_out_of_range_integer_asks_for_the_python_path(write, analyze):
    entry, _owner = _wrapped(write, analyze, "wrapwide", SCALARS, "widen")
    assert entry(7) == 49
    assert entry(10**9) == 10**18, "still inside the machine range"
    assert entry(10**30) is NotImplemented, "the argument is beyond the machine range"
    assert entry(10**10) is NotImplemented, "the product overflows, so the native code bails"


def test_a_guard_failure_leaves_no_exception_set(write, analyze):
    entry, _owner = _wrapped(write, analyze, "wrapclean", SCALARS, "widen")
    assert entry(10**30) is NotImplemented
    # A stale exception here would surface at some unrelated later point.
    assert sys.exc_info() == (None, None, None)
    assert entry(3) == 9


def test_the_wrapper_reads_a_borrowed_buffer_without_copying(write, analyze):
    entry, _owner = _wrapped(write, analyze, "wrapview", BUFFERS_VIEW, "total")
    values = array.array("d", [1.5, 2.5, 3.0])
    assert entry(values) == 7.0

    values[0] = 10.5
    assert entry(values) == 16.0, "the wrapper points at the caller's memory"
    assert entry([1.0, 2.0]) is NotImplemented, "a list has no buffer to point at"


def test_the_wrapper_copies_a_list_buffer(write, analyze):
    entry, _owner = _wrapped(write, analyze, "wraplist", BUFFERS, "total")
    assert entry([1, 2, 3]) == 6
    assert entry((1, 2, 3)) is NotImplemented
    assert entry([1, 10**30]) is NotImplemented


def test_the_binding_prefers_the_generated_wrapper(write, analyze):
    from ppy_compiler.backend.llvm.wrapper_build import build_wrappers, wrapper_toolchain

    ready, detail = wrapper_toolchain()
    if not ready:
        pytest.skip(detail)

    path = write("prefer.ppy", SCALARS)
    module = _collect(analyze(path, backend="llvm"))["prefer"]
    engine = _jit(module)
    built = build_wrappers(
        "prefer",
        {q: lowered.signature for q, lowered in module.functions.items()},
        path.parent / ".ppy-cache",
    )
    if not built.ok:
        pytest.skip(built.reason)

    lowered = module.functions["prefer.widen"]
    entry = built.bind("prefer.widen", engine.address(lowered.signature.symbol), ())

    def fallback(x):
        return x * x

    binding = bind(
        lowered.signature,
        engine.address(lowered.signature.symbol),
        fallback,
        fast_entry=entry,
    )
    assert binding.fast_entry is entry
    assert binding.wrapper(7) == 49
    assert binding.calls == 1

    # Beyond the machine range the guard fails and the Python body answers.
    assert binding.wrapper(10**30) == 10**60
    assert binding.fallbacks == 1


MIXED = """
    from dataclasses import dataclass

    import ppy
    from ppy import Buffer


    @dataclass
    class Point:
        x: float
        y: float


    @ppy.pure
    def mixed(
        scalar: int,
        pair: tuple[float, float],
        point: Point,
        copied: list[float],
        borrowed: Buffer[float],
    ) -> float:
        return float(scalar) + pair[0] + point.x + copied[0] + borrowed[0]
    """


def test_a_guard_failing_on_an_early_argument_cleans_up_safely(write, analyze):
    """A jump to the cleanup must not read a declaration it skipped over."""
    from ppy_compiler.backend.llvm.wrapper_build import build_wrappers, wrapper_toolchain

    ready, detail = wrapper_toolchain()
    if not ready:
        pytest.skip(detail)

    path = write("mixedwrap.ppy", MIXED)
    bundle = analyze(path, backend="llvm")
    module = _collect(bundle)["mixedwrap"]
    assert "mixedwrap.mixed" in module.functions, module.rejected
    engine = _jit(module)

    built = build_wrappers(
        "mixedwrap",
        {q: lowered.signature for q, lowered in module.functions.items()},
        path.parent / ".ppy-cache",
    )
    if not built.ok:
        pytest.skip(built.reason)

    namespace: dict = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    point_type = namespace["Point"]

    lowered = module.functions["mixedwrap.mixed"]
    entry = built.bind("mixedwrap.mixed", engine.address(lowered.signature.symbol), (point_type,))
    assert entry is not None

    point = point_type(3.0, 4.0)
    values = array.array("d", [5.0, 6.0])
    assert entry(1, (2.0, 0.0), point, [10.0], values) == 21.0

    # The first argument fails, so the jump skips every later declaration.
    for _ in range(50):
        assert entry("no", (2.0, 0.0), point, [10.0], values) is NotImplemented
        assert entry(1, (2.0, 0.0), "not a point", [10.0], values) is NotImplemented
        assert entry(1, (2.0, 0.0), point, "not a list", values) is NotImplemented
        assert entry(1, (2.0, 0.0), point, [10.0], [1.0]) is NotImplemented
    assert entry(1, (2.0, 0.0), point, [10.0], values) == 21.0
    del engine, built


def test_a_class_that_intercepts_attribute_reads_stays_boxed(write, analyze):
    """Flattening is only invisible if reading a field is a plain read."""
    path = write(
        "intercept.ppy",
        """
        from dataclasses import dataclass


        @dataclass
        class Plain:
            x: float


        class Watching:
            x: float

            def __init__(self, x: float) -> None:
                self.x = x

            def __getattr__(self, name: str) -> float:
                return 0.0


        class Shadowed:
            x: float

            def __init__(self, x: float) -> None:
                self.x = x

            def x(self) -> float:
                return 1.0
        """,
    )
    layouts = _layouts(analyze(path, backend="llvm"))
    assert "intercept.Plain" in layouts
    assert "intercept.Watching" not in layouts
    assert "intercept.Shadowed" not in layouts


def test_a_specialization_key_describes_its_pins_for_the_wrapper():
    from ppy_compiler.backend.llvm.specialize import SpecializationKey

    key = SpecializationKey(
        ((1, "modulus", "value", 97), (0, "values", "length", 64), (2, "scale", "value", 0.5))
    )
    assert key.pins() == ((1, 1, 97), (3, 0, 64), (2, 2, 0.5))


def test_the_wrapper_selects_a_registered_specialization(write, analyze):
    """The specialization guard lives in C, so a specialized function keeps
    the cheap boundary (spec 16.9)."""
    from ppy_compiler.backend.llvm.wrapper_build import build_wrappers, wrapper_toolchain

    ready, detail = wrapper_toolchain()
    if not ready:
        pytest.skip(detail)

    path = write(
        "specwrap.ppy",
        """
        import ppy


        @ppy.pure
        @ppy.opt(3)
        def digest(value: int, modulus: int) -> int:
            return value % modulus
        """,
    )
    bundle = analyze(path, backend="llvm")
    module = _collect(bundle)["specwrap"]
    engine = _jit(module)
    built = build_wrappers(
        "specwrap",
        {q: lowered.signature for q, lowered in module.functions.items()},
        path.parent / ".ppy-cache",
    )
    if not built.ok:
        pytest.skip(built.reason)

    lowered = module.functions["specwrap.digest"]
    address = engine.address(lowered.signature.symbol)
    entry = built.bind("specwrap.digest", address, ())
    register = built.registrar("specwrap.digest")
    assert entry is not None and register is not None

    assert entry(17, 5) == 2

    # Register a deliberately wrong "specialization" so its selection is
    # observable: pinning modulus == 5 must route only calls with modulus 5.
    from ppy_compiler.backend.llvm.specialize import Specializer

    specializer = Specializer(engine=engine, module_analysis=bundle.analysis.modules["specwrap"])
    for info, node in module.sources.values():
        specializer.register(info, node)

    from ppy_compiler.backend.llvm.specialize import SpecializationKey

    key = SpecializationKey(((1, "modulus", "value", 5),))
    specialization = specializer.specialize(lowered.info, key)
    assert specialization is not None and specialization.ok
    assert register(specialization.address, key.pins())

    assert entry(17, 5) == 2, "the specialization computes the same answer"
    assert entry(17, 4) == 1, "another shape falls through to the generic code"
    del engine, built


def test_registering_beyond_the_wrapper_capacity_is_refused(write, analyze):
    from ppy_compiler.backend.llvm.wrapper_build import build_wrappers, wrapper_toolchain

    ready, detail = wrapper_toolchain()
    if not ready:
        pytest.skip(detail)

    path = write("capwrap.ppy", SCALARS)
    module = _collect(analyze(path, backend="llvm"))["capwrap"]
    engine = _jit(module)
    built = build_wrappers(
        "capwrap",
        {q: lowered.signature for q, lowered in module.functions.items()},
        path.parent / ".ppy-cache",
    )
    if not built.ok:
        pytest.skip(built.reason)

    lowered = module.functions["capwrap.widen"]
    address = engine.address(lowered.signature.symbol)
    built.bind("capwrap.widen", address, ())
    register = built.registrar("capwrap.widen")

    accepted = sum(register(address, ((1, 0, value),)) for value in range(32))
    assert 0 < accepted <= 8, "the wrapper bounds how many it will hold"
    del engine, built


SIEVE = """
    import ppy
    from ppy import Buffer


    @ppy.opt(3)
    def sieve(flags: Buffer[int], limit: int) -> int:
        for i in range(limit):
            flags[i] = 1
        flags[0] = 0
        flags[1] = 0
        p: int = 2
        while p * p < limit:
            if flags[p] == 1:
                multiple: int = p * p
                while multiple < limit:
                    flags[multiple] = 0
                    multiple += p
            p += 1
        total: int = 0
        for i in range(limit):
            total += flags[i]
        return total
"""


def test_a_borrowed_buffer_may_be_written_through(write, analyze):
    path = write("sieve.ppy", SIEVE)
    module = _collect(analyze(path, backend="llvm"))["sieve"]
    assert "sieve.sieve" in module.functions
    assert "sieve.sieve" not in module.rejected
    ir = emit_ir(analyze(path, backend="llvm"))["sieve"]
    assert "store i64" in ir


def test_writes_through_a_borrowed_buffer_reach_the_caller(write, analyze):
    path = write("visible.ppy", SIEVE)
    module = _collect(analyze(path, backend="llvm"))["visible"]
    engine = _jit(module)
    lowered = module.functions["visible.sieve"]

    def fallback(flags, limit):  # pragma: no cover - the native path is taken
        raise AssertionError("fell back")

    binding = bind(lowered.signature, engine.address(lowered.signature.symbol), fallback)
    flags = array.array("q", [0] * 50)
    assert binding.wrapper(flags, 50) == 15
    assert binding.calls == 1 and binding.fallbacks == 0
    assert [i for i, flag in enumerate(flags) if flag] == [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
    ]


def test_writing_to_a_copied_list_parameter_stays_boxed(write, analyze):
    path = write(
        "copied.ppy",
        """
        def zero(xs: list[int]) -> int:
            xs[0] = 0
            return len(xs)
        """,
    )
    module = _collect(analyze(path, backend="llvm"))["copied"]
    assert "copied.zero" in module.rejected
    assert "borrowed buffer" in module.rejected["copied.zero"]


def test_a_mutating_program_matches_plain_cpython(tmp_path: Path):
    entry = tmp_path / "sieve_run.ppy"
    entry.write_text(
        textwrap.dedent(SIEVE).lstrip("\n")
        + textwrap.dedent(
            """

            import array

            flags = array.array("q", [0] * 100000)
            print(sieve(flags, 100000))
            print(sum(flags[:100]))
            """
        ),
        encoding="utf-8",
    )
    plain = _run([sys.executable, entry.name], tmp_path)
    native = _ppy(["run", entry.name], tmp_path)
    assert plain.returncode == 0, plain.stderr
    assert native.returncode == 0, native.stderr
    assert native.stdout == plain.stdout


MINMAX = """
    import ppy


    @ppy.pure
    @ppy.opt(3)
    def widest(a: int, b: int) -> int:
        return max(a, b)


    @ppy.pure
    @ppy.opt(3)
    def narrowest(a: float, b: float, c: float) -> float:
        return min(a, b, c)
"""


def test_min_and_max_over_scalars_lower_natively(write, analyze):
    path = write("minmax.ppy", MINMAX)
    module = _collect(analyze(path, backend="llvm"))["minmax"]
    assert "minmax.widest" in module.functions
    assert "minmax.narrowest" in module.functions
    ir = emit_ir(analyze(path, backend="llvm"))["minmax"]
    # The pairwise selects are what is emitted; LLVM folds them into its own
    # `smax`/`minnum` intrinsics, and either form is a real machine operation
    # rather than a call back into CPython.
    assert "select" in ir or "llvm.smax" in ir or "llvm.minnum" in ir
    assert "PyObject" not in ir


def test_a_mixed_extremum_is_refused_rather_than_coerced(write, analyze):
    """CPython returns the winning object, so `max(3, 2.5)` is the int `3`."""
    path = write(
        "mixed.ppy",
        """
        import ppy

        @ppy.pure
        @ppy.opt(3)
        def either(a: int, b: float) -> float:
            return max(a, b)
        """,
    )
    module = _collect(analyze(path, backend="llvm"))["mixed"]
    assert "mixed.either" not in module.functions
    assert "would change the result type" in module.rejected["mixed.either"]


def test_extremum_results_match_plain_cpython(tmp_path: Path):
    entry = tmp_path / "extremum.ppy"
    entry.write_text(
        textwrap.dedent(MINMAX).lstrip("\n")
        + textwrap.dedent(
            """

            nan: float = float("nan")
            print(widest(3, 7), widest(7, 3), widest(4, 4), widest(-5, -9))
            print(narrowest(3.0, 1.0, 2.0), narrowest(1.0, 1.0, 1.0))
            print(narrowest(nan, 1.0, 2.0), narrowest(1.0, nan, 2.0))
            print(narrowest(-0.0, 0.0, 0.0))
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nopt-level = 3\n", encoding="utf-8")
    plain = _run([sys.executable, entry.name], tmp_path)
    native = _ppy(["run", entry.name], tmp_path)
    assert plain.returncode == 0, plain.stderr
    assert native.returncode == 0, native.stderr
    assert native.stdout == plain.stdout


def test_a_gil_free_body_is_marked_and_wrapped(write, analyze):
    """Spec 16.6: a region that cannot reach the interpreter drops the GIL."""
    from ppy_compiler.backend.llvm.wrapper import generate

    path = write(
        "gilfree.ppy",
        """
        import ppy

        @ppy.pure
        @ppy.opt(3)
        def busy(rounds: int) -> int:
            total: int = 0
            for i in range(rounds):
                total += i % 7
            return total
        """,
    )
    module = _collect(analyze(path, backend="llvm"))["gilfree"]
    lowered = module.functions["gilfree.busy"]
    assert lowered.signature.releases_gil

    source = generate("gilfree", {"gilfree.busy": lowered.signature}).source
    assert "Py_BEGIN_ALLOW_THREADS" in source
    assert "Py_END_ALLOW_THREADS" in source
    # The guards read Python objects, so the release has to come after them.
    assert source.index("PPY_GUARD_FAIL") < source.index("Py_BEGIN_ALLOW_THREADS")
    # And the result is built once the GIL is back.
    assert source.index("Py_END_ALLOW_THREADS") < source.index("PyLong_FromLongLong")


def test_a_body_that_can_reach_the_interpreter_keeps_the_gil():
    from ppy_compiler.analysis.effects import Effect, EffectSet
    from ppy_compiler.backend.llvm.lowering import _releases_gil

    class _Analysis:
        def __init__(self, effects):
            self.effects = effects

    assert _releases_gil(_Analysis(EffectSet.of(Effect.ALLOC)))
    for effect in (Effect.PYTHON_CALLBACK, Effect.EXTERNAL_UNKNOWN, Effect.IO):
        assert not _releases_gil(_Analysis(EffectSet.of(effect))), effect


def test_releasing_the_gil_keeps_the_answer(tmp_path: Path):
    entry = tmp_path / "gil_run.ppy"
    entry.write_text(
        textwrap.dedent(
            """
            import ppy


            @ppy.pure
            @ppy.opt(3)
            def busy(rounds: int) -> int:
                total: int = 0
                for i in range(rounds):
                    total += i % 7
                return total


            print(busy(100000))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nopt-level = 3\n", encoding="utf-8")
    plain = _run([sys.executable, entry.name], tmp_path)
    native = _ppy(["run", entry.name], tmp_path)
    assert plain.returncode == 0, plain.stderr
    assert native.stdout.splitlines()[-1] == plain.stdout.strip()


SEQUENCE = """
    from collections.abc import Sequence

    import ppy


    @ppy.pure
    @ppy.opt(3)
    def total(values: Sequence[float]) -> float:
        out: float = 0.0
        for value in values:
            out += value
        return out
"""


def test_a_read_only_sequence_still_lowers_natively(write, analyze):
    """Widening a parameter to `Sequence` must not cost the native path."""
    path = write("seq.ppy", SEQUENCE)
    module = _collect(analyze(path, backend="llvm"))["seq"]
    assert "seq.total" in module.functions
    assert "seq.total" not in module.rejected


def test_a_sequence_parameter_accepts_more_than_a_list(tmp_path: Path):
    entry = tmp_path / "seq_run.ppy"
    entry.write_text(
        textwrap.dedent(SEQUENCE).lstrip("\n")
        + textwrap.dedent(
            """

            print(total([1.0, 2.0, 3.0]))
            print(total((1.0, 2.0, 3.0)))
            print(total([]))
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nopt-level = 3\n", encoding="utf-8")
    plain = _run([sys.executable, entry.name], tmp_path)
    native = _ppy(["run", entry.name], tmp_path)
    assert plain.returncode == 0, plain.stderr
    assert native.returncode == 0, native.stderr
    assert native.stdout == plain.stdout


def test_a_concrete_container_satisfies_the_abstract_one():
    from ppy_compiler.analysis import types as T

    sequence = T.Instance("Sequence", (T.FLOAT,), ("Sequence", "Iterable", "object"))
    iterable = T.Instance("Iterable", (T.FLOAT,), ("Iterable", "object"))
    for concrete in (T.list_of(T.FLOAT), T.Tuple_((T.FLOAT,), homogeneous=True)):
        assert T.is_assignable(concrete, sequence), concrete
        assert T.is_assignable(concrete, iterable), concrete
    assert T.is_assignable(T.instance("set", T.FLOAT), iterable)
    # A set is not a sequence, and the widening does not run backwards.
    assert not T.is_assignable(T.instance("set", T.FLOAT), sequence)
    assert not T.is_assignable(sequence, T.list_of(T.FLOAT))
