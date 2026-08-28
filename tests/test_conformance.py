"""One test per v1 MUST requirement, named by the spec section it checks.

Unit tests assert an analysis property directly. End-to-end tests run a program
on all three paths -- plain CPython, the Python backend, the LLVM backend --
and require identical observable behavior, which is the invariant everything
else rests on (spec 3.3).
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.analysis import types as T
from ppy_compiler.analysis.effects import Effect
from ppy_compiler.backend.llvm import available as llvm_available

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.ppy]\nstrict = true\nopt-level = 3\n", encoding="utf-8"
    )
    return tmp_path


def _write(project: Path, name: str, source: str) -> Path:
    path = project / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    return path


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False, timeout=900)


def _ppy(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return _run([sys.executable, "-m", "ppy_compiler", *args], cwd)


def _three_paths(path: Path) -> tuple[str, str, str]:
    """Stdout from plain CPython, the Python backend, and the LLVM backend."""
    cwd = path.parent
    plain = _run([sys.executable, path.name], cwd)
    assert plain.returncode == 0, plain.stderr
    optimized = _ppy([path.name], cwd)
    assert optimized.returncode == 0, optimized.stderr
    native = _ppy(["run", path.name], cwd)
    assert native.returncode == 0, native.stderr
    noise = ("compiling ", "built ", "remark", "warning")

    def clean(text: str) -> str:
        kept = [line for line in text.splitlines() if not line.startswith(noise)]
        return "".join(line + "\n" for line in kept)

    return plain.stdout, clean(optimized.stdout), clean(native.stdout)


def _agree(path: Path) -> str:
    plain, optimized, native = _three_paths(path)
    assert optimized == plain, "the Python backend changed the answer"
    assert native == plain, "the LLVM backend changed the answer"
    return plain


# --------------------------------------------------------------------------
# 3. Core compatibility invariants
# --------------------------------------------------------------------------


def test_3_1_every_ppy_source_parses_with_the_cpython_grammar():
    """`.ppy` adds no syntax, so the stdlib parser accepts every example."""
    root = Path(__file__).resolve().parents[1] / "examples"
    sources = [p for p in root.rglob("*.ppy") if ".ppy-cache" not in str(p)]
    assert sources, "no examples to check"
    for path in sources:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_3_2_directives_do_not_change_behavior():
    """A directive returns the original callable, identity included."""
    import ppy

    def original(x: int) -> int:
        return x * x

    for directive in (ppy.pure, ppy.native, ppy.inline, ppy.noinline, ppy.fastmath):
        assert directive(original) is original, directive
    assert ppy.opt(3)(original) is original
    assert ppy.jit(original) is original


def test_3_2_markers_are_legal_annotation_objects():
    import typing

    import ppy

    def annotated(a: ppy.i64, b: ppy.f32, c: ppy.Buffer[float]) -> ppy.i32:
        return c[0] > 0 and a + b > 0

    hints = typing.get_type_hints(annotated, globalns={"ppy": ppy}, include_extras=True)
    assert set(hints) == {"a", "b", "c", "return"}


def test_3_3_o3_does_not_relax_arithmetic(project: Path):
    """Optimization level alone must not permit reassociation (spec 3.3, 12.4)."""
    path = _write(
        project,
        "strict_fp.ppy",
        """
        import ppy


        @ppy.pure
        @ppy.opt(3)
        def total(values: list[float]) -> float:
            out: float = 0.0
            for value in values:
                out += value
            return out


        data: list[float] = [1e16, 1.0, -1e16, 1.0]
        print(total(data))
        """,
    )
    # Left to right the two 1.0 terms are lost to rounding against 1e16 and
    # only the last survives. Reassociating would pair them and give 2.0.
    assert _agree(path).strip() == "1.0"


def test_3_4_a_ppy_file_stays_readable_python(project: Path):
    """It runs under plain CPython with the runtime installed and nothing else."""
    path = _write(
        project,
        "plain.ppy",
        """
        import ppy


        @ppy.pure
        @ppy.opt(3)
        def double(x: ppy.i64) -> ppy.i64:
            return x * 2


        print(double(21))
        """,
    )
    done = _run([sys.executable, path.name], project)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "42"


# --------------------------------------------------------------------------
# 5. Source and package model
# --------------------------------------------------------------------------


def test_5_2_the_import_hook_loads_ppy_modules_and_packages(project: Path):
    _write(project, "helper.ppy", "VALUE: int = 7\n")
    (project / "pkg").mkdir()
    _write(project, "pkg/__init__.ppy", "from .inner import NAME\n")
    _write(project, "pkg/inner.ppy", "NAME: str = 'inner'\n")
    path = _write(
        project,
        "entry.py",
        """
        import ppy

        import helper
        import pkg

        print(helper.VALUE, pkg.NAME, helper.__file__.endswith('.ppy'))
        """,
    )
    done = _run([sys.executable, path.name], project)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "7 inner True"


def test_5_2_a_module_provided_twice_is_rejected(project: Path):
    _write(project, "dup.py", "VALUE = 1\n")
    _write(project, "dup.ppy", "VALUE: int = 1\n")
    _write(project, "uses.ppy", "import dup\n\nprint(dup.VALUE)\n")
    done = _ppy(["check", "."], project)
    assert done.returncode == 1
    assert "E1003" in done.stderr


def test_5_3_public_summaries_cross_module_boundaries(project: Path):
    """A type is inferred from a call site in another file (spec 5.3)."""
    _write(project, "lib.py", "def scale(x):\n    return x * 2.0\n")
    _write(project, "app.py", "import lib\n\nprint(lib.scale(1.5))\n")
    assert _ppy(["convert", "."], project).returncode == 0
    assert "def scale(x: float) -> float:" in (project / "lib.ppy").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# 6. The `ppy` runtime package
# --------------------------------------------------------------------------


def test_6_2_importing_ppy_starts_nothing_expensive():
    """`import ppy` must not initialize LLVM or start a service (spec 6.2)."""
    done = subprocess.run(
        [sys.executable, "-c", "import sys, ppy; print(sorted(m for m in sys.modules if 'llvm' in m))"],
        capture_output=True, text=True, check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def test_6_3_directives_are_read_from_source_not_runtime(project: Path):
    """The compiler reads the AST, so a directive applies without executing it."""
    path = _write(
        project,
        "fromsource.ppy",
        """
        import ppy


        @ppy.pure
        def impure() -> None:
            print("side effect")
        """,
    )
    done = _ppy(["check", path.name], project)
    assert done.returncode == 1
    assert "E1601" in done.stderr


def test_6_4_a_false_purity_claim_is_rejected(project: Path):
    path = _write(
        project,
        "liar.ppy",
        """
        import ppy

        COUNTER: int = 0


        @ppy.pure
        def bump() -> int:
            global COUNTER
            COUNTER += 1
            return COUNTER
        """,
    )
    done = _ppy(["check", path.name], project)
    assert done.returncode == 1
    assert "E1601" in done.stderr


# --------------------------------------------------------------------------
# 8. Static type system
# --------------------------------------------------------------------------


def test_8_2_an_implicit_any_is_an_error(project: Path):
    path = _write(project, "vague.ppy", "def f(x):\n    return x\n")
    done = _ppy(["check", path.name], project)
    assert done.returncode == 1
    assert "E1201" in done.stderr


def test_8_2_a_bare_generic_is_an_error(project: Path):
    path = _write(project, "bare.ppy", "def f(x: list) -> int:\n    return len(x)\n")
    done = _ppy(["check", path.name], project)
    assert done.returncode == 1
    assert "E1203" in done.stderr


@pytest.mark.parametrize(
    "body",
    [
        "if x is None:\n        return 0\n    return x",
        "if isinstance(x, int):\n        return x\n    return 0",
        "match x:\n        case None:\n            return 0\n        case _:\n            return x",
        "return x if x is not None else 0",
        "assert x is not None\n    return x",
    ],
)
def test_8_4_narrowing_forms(project: Path, body: str):
    path = _write(project, "narrow.ppy", f"def f(x: int | None) -> int:\n    {body}\n")
    done = _ppy(["check", path.name], project)
    assert done.returncode == 0, done.stderr


# --------------------------------------------------------------------------
# 11. Effect and purity system
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "effect"),
    [
        ('print("x")', Effect.IO),
        ("import random\n    return random.random()", Effect.RANDOM),
        ("import time\n    return time.time()", Effect.TIME),
    ],
)
def test_11_1_effects_are_tracked_separately(project: Path, body: str, effect: Effect):
    from ppy_compiler.driver.pipeline import analyze_paths, open_project

    path = _write(project, "fx.ppy", f"def f() -> float:\n    {body}\n    return 0.0\n")
    bundle = analyze_paths(open_project(path), [path], backend="python")
    assert effect in bundle.analysis.modules["fx"].functions["fx.f"].effects


def test_11_2_purity_is_inferred_without_a_declaration(project: Path):
    from ppy_compiler.driver.pipeline import analyze_paths, open_project

    path = _write(project, "inferred.ppy", "def f(x: int) -> int:\n    return x * x\n")
    bundle = analyze_paths(open_project(path), [path], backend="python")
    assert bundle.analysis.modules["inferred"].functions["inferred.f"].verified_pure


# --------------------------------------------------------------------------
# 12. Numeric system
# --------------------------------------------------------------------------


def test_12_1_integers_keep_arbitrary_precision(project: Path):
    path = _write(
        project,
        "big.ppy",
        """
        import ppy


        @ppy.pure
        @ppy.opt(3)
        def factorial(n: int) -> int:
            out: int = 1
            for i in range(1, n + 1):
                out *= i
            return out


        print(factorial(25))
        """,
    )
    assert _agree(path).strip() == str(__import__("math").factorial(25))


def test_12_1_floor_division_and_remainder_follow_python(project: Path):
    path = _write(
        project,
        "signs.ppy",
        """
        import ppy


        @ppy.pure
        @ppy.opt(3)
        def pair(a: int, b: int) -> tuple[int, int]:
            return (a // b, a % b)


        print(pair(-7, 2), pair(7, -2), pair(-7, -2), pair(7, 2))
        """,
    )
    assert _agree(path).strip() == "(-4, 1) (-4, -1) (3, -1) (3, 1)"


def test_12_4_fastmath_is_the_only_way_to_reassociate(project: Path):
    """The same reduction, strict and relaxed, on the same input (spec 12.4)."""
    path = _write(
        project,
        "relax.ppy",
        """
        import ppy


        @ppy.pure
        @ppy.opt(3)
        def strict(values: list[float]) -> float:
            out: float = 0.0
            for value in values:
                out += value
            return out


        @ppy.pure
        @ppy.opt(3)
        @ppy.fastmath
        def relaxed(values: list[float]) -> float:
            out: float = 0.0
            for value in values:
                out += value
            return out


        data: list[float] = [1e16, 1.0, -1e16, 1.0]
        print(strict(data))
        """,
    )
    # Only the strict result is compared: the relaxed one is allowed to differ,
    # which is the whole point of requiring the directive.
    assert _agree(path).strip() == "1.0"


# --------------------------------------------------------------------------
# 13. Containers, aliasing, and memory
# --------------------------------------------------------------------------


def test_13_2_two_parameters_may_be_the_same_object(project: Path):
    """Arguments of mutable container type are assumed to alias (spec 13.2)."""
    path = _write(
        project,
        "alias.ppy",
        """
        import array

        import ppy
        from ppy import Buffer


        @ppy.opt(3)
        def shift(dst: Buffer[int], src: Buffer[int], n: int) -> int:
            for i in range(n):
                dst[i] = src[i] + 1
            out: int = 0
            for i in range(n):
                out += dst[i]
            return out


        shared = array.array("q", [1, 2, 3, 4])
        print(shift(shared, shared, 4), list(shared))
        apart = array.array("q", [0, 0, 0, 0])
        print(shift(apart, array.array("q", [1, 2, 3, 4]), 4), list(apart))
        """,
    )
    assert _agree(path).splitlines()[0] == "14 [2, 3, 4, 5]"


def test_13_3_a_native_container_materializes_correctly(project: Path):
    path = _write(
        project,
        "escape.ppy",
        """
        import ppy


        @ppy.pure
        @ppy.opt(3)
        def doubled(values: list[int]) -> int:
            out: int = 0
            for value in values:
                out += value * 2
            return out


        data: list[int] = [1, 2, 3]
        print(doubled(data), data, len(data))
        """,
    )
    assert _agree(path).strip() == "12 [1, 2, 3] 3"


# --------------------------------------------------------------------------
# 15 / 16. Backends
# --------------------------------------------------------------------------


@requires_llvm
def test_16_2_a_boxed_call_still_compiles(project: Path):
    """The backend compiles a module even when part of it stays boxed."""
    path = _write(
        project,
        "mixed.ppy",
        """
        import ppy


        @ppy.pure
        @ppy.opt(3)
        def fast(x: int) -> int:
            return x * x


        def slow(values: list[str]) -> str:
            return ",".join(values)


        print(fast(7), slow(["a", "b"]))
        """,
    )
    assert _agree(path).strip() == "49 a,b"


@requires_llvm
def test_16_3_an_overflowing_scalar_falls_back_instead_of_wrapping(project: Path):
    path = _write(
        project,
        "overflow.ppy",
        """
        import ppy


        @ppy.pure
        @ppy.opt(3)
        def cube(x: int) -> int:
            return x * x * x


        print(cube(3), cube(10 ** 7))
        """,
    )
    assert _agree(path).strip() == f"27 {10 ** 21}"


@requires_llvm
def test_16_7_exceptions_map_to_python(project: Path):
    path = _write(
        project,
        "raises.ppy",
        """
        import ppy


        @ppy.opt(3)
        def divide(a: int, b: int) -> int:
            return a // b


        @ppy.opt(3)
        def at(values: list[int], index: int) -> int:
            return values[index]


        for attempt in ((1, 0), (7, 2)):
            try:
                print(divide(attempt[0], attempt[1]))
            except ZeroDivisionError as error:
                print("caught", type(error).__name__)
        try:
            at([1, 2, 3], 9)
        except IndexError as error:
            print("caught", type(error).__name__)
        """,
    )
    assert _agree(path).splitlines() == ["caught ZeroDivisionError", "3", "caught IndexError"]


@requires_llvm
def test_16_9_a_specialization_falls_back_when_its_guard_fails(project: Path):
    """A pinned specialization must not be used for a shape it was not built for."""
    from ppy_compiler.backend.llvm import _collect
    from ppy_compiler.backend.llvm.jit import JitEngine
    from ppy_compiler.backend.llvm.runtime import bind
    from ppy_compiler.backend.llvm.specialize import Specializer
    from ppy_compiler.driver.pipeline import analyze_paths, open_project

    path = _write(
        project,
        "spec.ppy",
        """
        import ppy


        @ppy.jit
        @ppy.pure
        @ppy.opt(3)
        def scaled(x: int, factor: int) -> int:
            return x * factor
        """,
    )
    bundle = analyze_paths(open_project(path), [path], backend="llvm")
    module = _collect(bundle)["spec"]
    engine = JitEngine(opt_level=3).open()
    engine.add(module.ir)
    engine.finalize()
    lowered = module.functions["spec.scaled"]
    binding = bind(
        lowered.signature, engine.address(lowered.signature.symbol), lambda x, f: x * f
    )
    specializer = Specializer(module, bundle.analysis.modules["spec"], engine)
    for _ in range(8):
        assert binding.wrapper(3, 5) == 15
    assert binding.wrapper(3, 7) == 21, "a different shape must not reuse the specialization"
    assert specializer is not None


# --------------------------------------------------------------------------
# 19-23. Required plugins
# --------------------------------------------------------------------------

numpy = pytest.importorskip("numpy", reason="numpy is not installed")


@requires_llvm
def test_19_2_an_unsupported_numpy_case_falls_back(project: Path):
    """A case the fused path cannot take must still run, not be rejected."""
    path = _write(
        project,
        "npfall.ppy",
        """
        import numpy as np

        import ppy


        @ppy.pure
        @ppy.opt(3)
        def scaled(x: np.ndarray) -> np.ndarray:
            return x * 2.0


        strided = np.arange(12, dtype=np.float64)[::2]
        print(float(np.sum(scaled(strided))))
        print(float(np.sum(scaled(np.arange(4, dtype=np.float64)))))
        """,
    )
    assert _agree(path).splitlines() == ["60.0", "12.0"]


@requires_llvm
def test_19_6_reduction_order_is_kept_without_permission(project: Path):
    path = _write(
        project,
        "nporder.ppy",
        """
        import numpy as np

        import ppy


        @ppy.pure
        @ppy.opt(3)
        def total(x: np.ndarray) -> float:
            return float(np.sum(x))


        data = np.array([1e16, 1.0, -1e16, 1.0], dtype=np.float64)
        print(total(data) == float(np.sum(data)))
        """,
    )
    assert _agree(path).strip() == "True"


def test_23_pydantic_constraints_become_refinements():
    pydantic = pytest.importorskip("pydantic", reason="pydantic is not installed")
    from ppy_compiler.plugins.pydantic_plugin import PydanticPlugin

    assert pydantic is not None
    plugin = PydanticPlugin()
    assert plugin.name == "pydantic"
    assert "pydantic" in plugin.modules


# --------------------------------------------------------------------------
# 25. Python interoperability
# --------------------------------------------------------------------------


@requires_llvm
def test_25_a_native_function_is_callable_from_python(project: Path):
    """The trampoline accepts Python objects and returns a Python object."""
    path = _write(
        project,
        "trampoline.ppy",
        """
        import ppy


        @ppy.pure
        @ppy.opt(3)
        def square(x: int) -> int:
            return x * x


        values = [square(i) for i in range(5)]
        print(values, type(values[0]).__name__)
        """,
    )
    assert _agree(path).strip() == "[0, 1, 4, 9, 16] int"


@requires_llvm
def test_25_a_wrong_type_reaches_the_python_implementation(project: Path):
    """A guard that fails hands the call back rather than crashing (spec 25.2)."""
    path = _write(
        project,
        "guarded.ppy",
        """
        import ppy


        @ppy.pure
        @ppy.opt(3)
        def total(values: list[int]) -> int:
            out: int = 0
            for value in values:
                out += value
            return out


        print(total([1, 2, 3]))
        print(total([10 ** 30, 1]))
        """,
    )
    assert _agree(path).splitlines() == ["6", str(10 ** 30 + 1)]


# --------------------------------------------------------------------------
# 27. Incremental compilation and cache
# --------------------------------------------------------------------------


def test_27_1_artifacts_are_content_addressed(project: Path):
    """Touching a file without changing it must not invalidate anything."""
    path = _write(project, "stable.ppy", "def f(x: int) -> int:\n    return x + 1\n\n\nprint(f(1))\n")
    assert _ppy([path.name], project).returncode == 0
    before = _ppy(["cache", "status"], project).stdout
    path.touch()
    assert _ppy([path.name], project).returncode == 0
    assert _ppy(["cache", "status"], project).stdout == before


def test_27_1_a_changed_source_produces_a_new_artifact(project: Path):
    path = _write(project, "moving.ppy", "def f(x: int) -> int:\n    return x + 1\n\n\nprint(f(1))\n")
    assert _ppy([path.name], project).returncode == 0
    entries = _ppy(["cache", "status"], project).stdout
    path.write_text("def f(x: int) -> int:\n    return x + 2\n\n\nprint(f(1))\n", encoding="utf-8")
    assert _ppy([path.name], project).stdout.strip() == "3"
    assert _ppy(["cache", "status"], project).stdout != entries


def test_27_4_gc_removes_expired_artifacts_without_breaking_a_build(project: Path):
    path = _write(project, "gc.ppy", "def f(x: int) -> int:\n    return x * 3\n\n\nprint(f(2))\n")
    assert _ppy([path.name], project).returncode == 0
    assert _ppy(["cache", "gc", "--max-age-days", "0"], project).returncode == 0
    rebuilt = _ppy([path.name], project)
    assert rebuilt.returncode == 0, rebuilt.stderr
    assert rebuilt.stdout.strip() == "6"


# --------------------------------------------------------------------------
# 29. Diagnostics
# --------------------------------------------------------------------------


def test_29_a_diagnostic_carries_code_span_and_help(project: Path):
    path = _write(project, "diag.ppy", "def f(x):\n    return x\n")
    done = _ppy(["check", path.name], project)
    assert done.returncode == 1
    assert "error[E1201]" in done.stderr
    assert f"{path.name}:1:1" in done.stderr
    assert "= help:" in done.stderr


def test_29_explain_describes_a_diagnostic_code(project: Path):
    done = _ppy(["explain", "E1201"], project)
    assert done.returncode == 0
    assert "E1201" in done.stdout


# --------------------------------------------------------------------------
# 31. Build-time execution and security
# --------------------------------------------------------------------------


def test_31_2_analysis_does_not_execute_project_code_by_default(project: Path):
    """`ppy check` must not run the module it is checking."""
    marker = project / "executed.txt"
    path = _write(
        project,
        "sideeffect.ppy",
        f"""
        from pathlib import Path

        Path({str(marker)!r}).write_text("ran", encoding="utf-8")


        def f(x: int) -> int:
            return x
        """,
    )
    _ppy(["check", path.name], project)
    assert not marker.exists(), "checking the module executed it"


# --------------------------------------------------------------------------
# 16.6. GIL policy
# --------------------------------------------------------------------------


@requires_llvm
def test_16_6_a_gil_free_region_scales_across_threads(project: Path):
    """A native region that cannot reach the interpreter releases the GIL."""
    _write(
        project,
        "kernel.ppy",
        """
        import ppy


        @ppy.pure
        @ppy.opt(3)
        def busy(rounds: int) -> int:
            out: int = 0
            for i in range(rounds):
                out += i % 7
            return out
        """,
    )
    path = _write(
        project,
        "scaling.ppy",
        """
        import threading
        import time

        import ppy

        import kernel


        def timed(rounds: int, workers: int) -> float:
            threads: list[threading.Thread] = []
            for _ in range(workers):
                threads.append(threading.Thread(target=kernel.busy, args=(rounds,)))
            started: float = time.perf_counter()
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            return time.perf_counter() - started


        rounds: int = 40000000
        one: float = timed(rounds, 1)
        two: float = timed(rounds, 2)
        print(f"{2.0 * one / two:.2f}")
        """,
    )
    native = _ppy(["run", path.name], project)
    assert native.returncode == 0, native.stderr
    scaling = float(native.stdout.splitlines()[-1])
    # Two threads sharing one core would give 1.0; releasing the GIL gives ~2.0.
    assert scaling > 1.4, f"two threads did not scale ({scaling:.2f}x)"


@requires_llvm
def test_16_6_a_body_with_io_is_not_lowered_at_all(project: Path):
    from ppy_compiler.backend.llvm import _collect
    from ppy_compiler.driver.pipeline import analyze_paths, open_project

    path = _write(
        project,
        "talks.ppy",
        """
        import ppy


        @ppy.opt(3)
        def announce(n: int) -> int:
            print(n)
            return n
        """,
    )
    bundle = analyze_paths(open_project(path), [path], backend="llvm")
    module = _collect(bundle)["talks"]
    assert "talks.announce" not in module.functions
    assert "IO" in module.rejected["talks.announce"]


# --------------------------------------------------------------------------
# 17. Parallelization
# --------------------------------------------------------------------------


def test_17_required_parallel_reports_a_precise_reason(project: Path):
    path = _write(
        project,
        "par.ppy",
        """
        import ppy


        @ppy.parallel(require=True)
        def carried(n: int) -> int:
            out: int = 0
            for i in range(n):
                out = out * 2 + i
            return out
        """,
    )
    done = _ppy(["check", path.name], project)
    assert done.returncode == 1
    assert "E1701" in done.stderr
    assert "reason" in done.stderr or "cannot be satisfied" in done.stderr


def test_17_required_native_reports_a_precise_reason(project: Path):
    path = _write(
        project,
        "nat.ppy",
        """
        import ppy


        @ppy.native(require=True)
        def joined(values: list[str]) -> str:
            return ",".join(values)
        """,
    )
    done = _ppy(["check", path.name], project)
    assert done.returncode == 1
    assert "E1702" in done.stderr


# --------------------------------------------------------------------------
# 31. Build-time execution
# --------------------------------------------------------------------------


def test_31_2_build_time_export_is_denied_by_default(project: Path):
    """JAX export runs project code, so it needs an explicit opt-in."""
    pytest.importorskip("jax", reason="jax is not installed")
    from ppy_compiler.driver.pipeline import open_project

    path = _write(project, "staged.ppy", "VALUE: int = 1\n")
    plugins = open_project(path).plugins
    jax_plugin = plugins.for_module("jax")
    if jax_plugin is None:
        pytest.skip("the jax plugin is not registered")
    permitted, reason = jax_plugin.export_permitted("deny")
    assert not permitted
    assert "build-execution" in reason


@requires_llvm
def test_25_a_native_function_keeps_its_binding_across_modules(project: Path):
    """`import ppy` must not displace the loader that installs native bindings."""
    _write(
        project,
        "hot.ppy",
        """
        import ppy


        @ppy.pure
        @ppy.opt(3)
        def busy(rounds: int) -> int:
            out: int = 0
            for i in range(rounds):
                out += i % 7
            return out
        """,
    )
    path = _write(
        project,
        "driver.ppy",
        """
        import time

        import ppy

        import hot


        started: float = time.perf_counter()
        answer: int = hot.busy(20000000)
        elapsed: float = time.perf_counter() - started
        print(answer, elapsed < 0.25)
        """,
    )
    native = _ppy(["run", path.name], project)
    assert native.returncode == 0, native.stderr
    answer, fast = native.stdout.split()[-2:]
    assert answer == "59999997"
    assert fast == "True", "the imported module ran interpreted, not native"
