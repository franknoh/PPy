"""Python and LLVM backend behavior (spec 15, 16, 34)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.driver.pipeline import build_python

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")


def _generated(bundle, name: str) -> str:
    return build_python(bundle).generated[name].code


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


def _ppy(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return _run([sys.executable, "-m", "ppy_compiler", *args], cwd)


def test_directives_are_stripped_from_generated_python(write, analyze):
    path = write(
        "strip.ppy",
        """
        import ppy

        @ppy.pure
        @ppy.opt(3)
        def f(x: int) -> int:
            return x + 1
        """,
    )
    code = _generated(analyze(path), "strip")
    assert "@ppy" not in code
    assert "def f(x: int) -> int" in code


def test_constants_are_folded(write, analyze):
    path = write(
        "fold.ppy",
        """
        def f() -> int:
            a: int = 2
            b: int = 3
            return a * b + 1
        """,
    )
    code = _generated(analyze(path), "fold")
    assert "return 7" in code


def test_constant_branches_are_folded(write, analyze):
    path = write(
        "branch.ppy",
        """
        DEBUG: bool = False

        def f(x: int) -> int:
            if DEBUG:
                return x * 100
            return x
        """,
    )
    code = _generated(analyze(path), "branch")
    assert "100" not in code


def test_unreachable_code_is_removed(write, analyze):
    path = write(
        "dead.ppy",
        """
        def f(x: int) -> int:
            return x
            return x * 2
        """,
    )
    code = _generated(analyze(path), "dead")
    assert code.count("return") == 1


def test_loops_unroll_only_at_o3(write, analyze):
    source = """
        import ppy

        @ppy.opt(3)
        def hot() -> int:
            total: int = 0
            for i in range(3):
                total += i
            return total
        """
    path = write("unroll.ppy", source)
    code = _generated(analyze(path, opt_level=3), "unroll")
    assert "for i in range" not in code

    path2 = path.with_name("nounroll.ppy")
    path2.write_text(textwrap.dedent(source).replace("ppy.opt(3)", "ppy.opt(1)").lstrip("\n"), encoding="utf-8")
    code2 = _generated(analyze(path2, opt_level=1), "nounroll")
    assert "for i in range" in code2


def test_pure_functions_are_inlined_at_o2(write, analyze):
    path = write(
        "inline.ppy",
        """
        import ppy

        @ppy.pure
        def double(x: int) -> int:
            return x * 2

        def use(y: int) -> int:
            return double(y) + 1
        """,
    )
    code = _generated(analyze(path, opt_level=2), "inline")
    assert "return y * 2 + 1" in code


def test_generated_artifacts_never_overwrite_sources(write, analyze, project_dir):
    path = write("keep.ppy", "def f() -> int:\n    return 1\n")
    original = path.read_text(encoding="utf-8")
    output = build_python(analyze(path))
    artifact = output.generated["keep"].artifact
    assert path.read_text(encoding="utf-8") == original
    assert str(project_dir / ".ppy-cache") in str(artifact)


def test_generated_module_compiles_against_the_ppy_filename(write, analyze):
    path = write("trace.ppy", "def f() -> int:\n    return 1\n")
    generated = build_python(analyze(path)).generated["trace"]
    code = generated.compile()
    assert code.co_filename == str(path)


def test_traceback_points_at_the_ppy_source(tmp_path: Path):
    entry = tmp_path / "boom.ppy"
    entry.write_text(
        textwrap.dedent(
            """
            def explode(x: int, y: int) -> int:
                scaled: int = x * 2
                print(scaled)
                return scaled // y

            explode(1, 0)
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    result = _ppy(["boom.ppy"], tmp_path)
    assert result.returncode == 1
    assert "ZeroDivisionError" in result.stderr
    assert f'File "{entry}", line 4, in explode' in result.stderr
    assert f'File "{entry}", line 6, in <module>' in result.stderr


def test_inlining_preserves_the_exception_and_its_location(tmp_path: Path):
    """A frame may be elided by inlining, but never the class or the line."""
    entry = tmp_path / "inlined_boom.ppy"
    entry.write_text(
        textwrap.dedent(
            """
            def explode(x: int) -> int:
                return x // 0

            explode(1)
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    plain = _run([sys.executable, entry.name], tmp_path)
    optimized = _ppy([entry.name], tmp_path)
    assert plain.returncode == optimized.returncode == 1
    assert "ZeroDivisionError: division by zero" in optimized.stderr
    assert f'File "{entry}", line 2' in optimized.stderr


def test_cache_hit_avoids_recompilation(write, analyze):
    path = write("cached.ppy", "def f() -> int:\n    return 1\n")
    first = build_python(analyze(path))
    assert first.stats.get("cache_misses") == 1
    second = build_python(analyze(path))
    assert second.stats.get("cache_hits") == 1


def test_editing_a_source_invalidates_its_cache_entry(write, analyze):
    path = write("edit.ppy", "def f() -> int:\n    return 1\n")
    build_python(analyze(path))
    path.write_text("def f() -> int:\n    return 2\n", encoding="utf-8")
    output = build_python(analyze(path))
    assert output.stats.get("cache_misses") == 1
    assert "return 2" in output.generated["edit"].code


def test_python_backend_rejects_required_parallel(write, codes):
    path = write(
        "par.ppy",
        """
        import ppy

        @ppy.parallel(require=True)
        def total(xs: list[int]) -> int:
            result: int = 0
            for x in xs:
                result += x
            return result
        """,
    )
    assert "E1701" in codes(path, backend="python")


def test_required_native_reports_a_precise_reason(write, analyze):
    path = write(
        "nat.ppy",
        """
        import ppy

        @ppy.native(require=True)
        def show(x: int) -> int:
            print(x)
            return x
        """,
    )
    bundle = analyze(path, backend="llvm")
    errors = [d for d in bundle.diagnostics if d.code == "E1702"]
    assert errors and errors[0].help


# -- differential execution -----------------------------------------------

PROGRAMS = {
    "arithmetic": """
        import ppy
        from ppy import i64

        @ppy.pure
        def poly(x: i64) -> i64:
            return 3 * x * x + 2 * x + 1

        print(poly(4), poly(-4), poly(0))
        """,
    "arbitrary_precision": """
        import ppy

        @ppy.pure
        def cube(x: int) -> int:
            return x * x * x

        print(cube(3), cube(10 ** 7), cube(-(10 ** 7)))
        """,
    "floor_semantics": """
        import ppy

        @ppy.pure
        def pair(a: int, b: int) -> int:
            return a // b + a % b

        print(pair(-7, 2), pair(7, -2), pair(-7, -2), pair(7, 2))
        """,
    "exceptions": """
        import ppy

        @ppy.pure
        def divide(a: int, b: int) -> float:
            return a / b

        try:
            divide(1, 0)
        except ZeroDivisionError as exc:
            print("caught", type(exc).__name__)
        print(divide(1, 4))
        """,
    "control_flow": """
        import ppy

        @ppy.pure
        def collatz(n: int) -> int:
            steps: int = 0
            while n != 1:
                if n % 2 == 0:
                    n = n // 2
                else:
                    n = 3 * n + 1
                steps += 1
            return steps

        print(collatz(27), collatz(1), collatz(97))
        """,
    "floating_point": """
        import ppy
        import math

        @ppy.pure
        def norm(x: float, y: float) -> float:
            return math.sqrt(x * x + y * y)

        print(norm(3.0, 4.0), norm(0.1, 0.2))
        """,
    "aliasing": """
        def swap_first(a: list[int], b: list[int]) -> int:
            a[0] = 1
            return b[0]

        shared: list[int] = [0, 0]
        print(swap_first(shared, shared), swap_first([0], [9]))
        """,
    "classes": """
        class Point:
            x: int
            y: int

            def __init__(self, x: int, y: int) -> None:
                self.x = x
                self.y = y

            def norm2(self) -> int:
                return self.x * self.x + self.y * self.y

        p = Point(3, 4)
        print(p.norm2(), p.x, p.y)
        """,
    "bool_is_an_int": """
        def f(flag: bool) -> int:
            return flag + 1

        print(f(True), f(False), True == 1, isinstance(True, int))
        """,
}


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_python_backend_matches_plain_cpython(tmp_path: Path, name: str):
    entry = tmp_path / f"{name}.ppy"
    entry.write_text(textwrap.dedent(PROGRAMS[name]).lstrip("\n"), encoding="utf-8")

    plain = _run([sys.executable, entry.name], tmp_path)
    optimized = _ppy([entry.name], tmp_path)
    assert plain.returncode == 0, plain.stderr
    assert optimized.returncode == 0, optimized.stderr
    assert optimized.stdout == plain.stdout


@requires_llvm
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_llvm_backend_matches_plain_cpython(tmp_path: Path, name: str):
    entry = tmp_path / f"{name}.ppy"
    entry.write_text(textwrap.dedent(PROGRAMS[name]).lstrip("\n"), encoding="utf-8")

    plain = _run([sys.executable, entry.name], tmp_path)
    native = _ppy(["run", entry.name], tmp_path)
    assert plain.returncode == 0, plain.stderr
    assert native.returncode == 0, native.stderr
    assert native.stdout == plain.stdout


@requires_llvm
def test_scalar_functions_are_lowered_natively(write, analyze):
    from ppy_compiler.backend.llvm import _collect

    path = write(
        "native.ppy",
        """
        import ppy

        @ppy.pure
        def poly(x: int) -> int:
            return 3 * x * x + 2 * x + 1

        @ppy.pure
        def shout(x: int) -> int:
            print(x)
            return x
        """,
    )
    natives = _collect(analyze(path, backend="llvm"))
    module = natives["native"]
    assert "native.poly" in module.functions
    assert "native.shout" in module.rejected
    assert "IO" in module.rejected["native.shout"]


@requires_llvm
def test_native_ir_keeps_overflow_guards(write, analyze):
    from ppy_compiler.backend.llvm import emit_ir

    path = write(
        "guard.ppy",
        """
        import ppy

        @ppy.pure
        def square(x: int) -> int:
            return x * x
        """,
    )
    ir = emit_ir(analyze(path, backend="llvm"))["guard"]
    assert "smul.with.overflow" in ir
    assert "fast" not in ir


@requires_llvm
def test_o3_does_not_enable_fast_math(write, analyze):
    from ppy_compiler.backend.llvm import emit_ir

    path = write(
        "fp.ppy",
        """
        import ppy

        @ppy.pure
        @ppy.opt(3)
        def mix(a: float, b: float, c: float) -> float:
            return a * b + c
        """,
    )
    ir = emit_ir(analyze(path, backend="llvm", opt_level=3))["fp"]
    assert "fmul " in ir
    assert "fmul fast" not in ir and "reassoc" not in ir


@requires_llvm
def test_native_wrapper_falls_back_on_overflow(write, analyze):
    from ppy_compiler.backend.llvm import _collect
    from ppy_compiler.backend.llvm.jit import JitEngine
    from ppy_compiler.backend.llvm.runtime import bind

    path = write(
        "ovf.ppy",
        """
        import ppy

        @ppy.pure
        def square(x: int) -> int:
            return x * x
        """,
    )
    module = _collect(analyze(path, backend="llvm"))["ovf"]
    lowered = module.functions["ovf.square"]

    engine = JitEngine(opt_level=2).open()
    engine.add(module.ir)
    engine.finalize()

    def fallback(x):
        return x * x

    binding = bind(lowered.signature, engine.address(lowered.signature.symbol), fallback)
    assert binding.wrapper(7) == 49
    assert binding.calls == 1
    assert binding.wrapper(10**20) == 10**40
    assert binding.fallbacks == 1


def test_inlining_never_leaves_an_identity_test_against_a_literal(write, analyze):
    """CPython warns about `'x' is None`; the original source had no warning."""
    path = write(
        "identity.ppy",
        """
        import ppy

        @ppy.pure
        @ppy.opt(3)
        def present(name: str | None) -> bool:
            return name is not None and len(name) > 0

        answer: bool = present("ppy")
        """,
    )
    code = _generated(analyze(path, opt_level=3), "identity")
    assert "'ppy' is not None" not in code
    assert "is not None" not in code.split("def present")[-1].split("answer")[-1]


def test_a_settled_boolean_operand_is_dropped(write, analyze):
    path = write(
        "settled.ppy",
        """
        import ppy

        @ppy.pure
        @ppy.opt(3)
        def check(flag: bool, value: int) -> bool:
            return flag and value > 0

        answer: bool = check(True, 5)
        """,
    )
    code = _generated(analyze(path, opt_level=3), "settled")
    assert "True and" not in code


def test_folded_identity_keeps_the_same_answers(tmp_path: Path):
    entry = tmp_path / "fold_run.ppy"
    entry.write_text(
        textwrap.dedent(
            """
            import ppy


            @ppy.pure
            @ppy.opt(3)
            def present(name: str | None) -> bool:
                return name is not None and len(name) > 0


            print(present("ppy"), present(""), present(None))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nopt-level = 3\n", encoding="utf-8")
    plain = _run([sys.executable, entry.name], tmp_path)
    built = _ppy([entry.name], tmp_path)
    assert plain.returncode == 0 and built.returncode == 0, built.stderr
    assert built.stdout == plain.stdout == "True False False\n"
    assert "SyntaxWarning" not in built.stderr


def test_a_loop_carried_reset_is_not_hoisted(write, analyze):
    """`s = 0.0` inside a loop that also does `s += ...` is a reset, not an invariant."""
    path = write(
        "reset.ppy",
        """
        def rows(values: list[float], n: int, d: int) -> float:
            total: float = 0.0
            for i in range(n):
                s = 0.0
                for j in range(d):
                    s += values[i * d + j]
                total += s / float(d)
            return total
        """,
    )
    code = _generated(analyze(path, opt_level=3), "reset")
    body = code.split("def rows")[1]
    header, _, loop = body.partition("for i in range(n):")
    assert "s = 0.0" not in header, "the per-row reset was hoisted out of the loop"
    assert "s = 0.0" in loop


def test_reset_semantics_survive_optimization(tmp_path: Path):
    entry = tmp_path / "reset_run.ppy"
    entry.write_text(
        textwrap.dedent(
            """
            def rows(values: list[float], n: int, d: int) -> float:
                total: float = 0.0
                for i in range(n):
                    s = 0.0
                    for j in range(d):
                        s += values[i * d + j]
                    total += s / float(d)
                return total


            data: list[float] = [float(i % 7) for i in range(60)]
            print(f"{rows(data, 10, 6):.6f}")
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nopt-level = 3\n", encoding="utf-8")
    plain = _run([sys.executable, entry.name], tmp_path)
    built = _ppy([entry.name], tmp_path)
    native = _ppy(["run", entry.name], tmp_path)
    assert plain.returncode == 0, plain.stderr
    assert built.stdout == plain.stdout, "the Python backend changed the answer"
    assert native.stdout.splitlines()[-1] == plain.stdout.strip()


def test_an_invariant_computation_still_moves_out(write, analyze):
    path = write(
        "invariant.ppy",
        """
        def scaled(values: list[float], n: int, width: float) -> float:
            total: float = 0.0
            for i in range(n):
                factor = width * 2.0 + 1.0
                total += values[i] * factor
            return total
        """,
    )
    code = _generated(analyze(path, opt_level=3), "invariant")
    header = code.split("def scaled")[1].partition("for i in range(n):")[0]
    assert "_ppy_licm" in header, "the invariant computation was not hoisted"
