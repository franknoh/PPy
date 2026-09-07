"""`ppy.parallel` v2: `parallel.range` loops, on every path and every backend."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import c_compiler

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")

PROGRAM = """
    import ppy
    from ppy import Buffer, native, parallel


    def squares(out: native.ptr[int], n: int) -> int:
        for i in parallel.range(n):
            native.store(native.offset(out, i), i * i)
        return native.load(native.offset(out, n - 1))


    def dot(a: Buffer[float], b: Buffer[float]) -> float:
        total = 0.0
        for i in parallel.range(len(a)):
            total += a[i] * b[i]
        return total


    @ppy.fastmath
    def dot_relaxed(a: Buffer[float], b: Buffer[float]) -> float:
        total = 0.0
        for i in parallel.range(len(a)):
            total += a[i] * b[i]
        return total


    def count_odd(xs: Buffer[int], scale: int) -> int:
        hits = 0
        for i in parallel.range(0, len(xs)):
            if xs[i] % 2 == 1:
                hits += scale
        return hits


    @ppy.parallel
    def fill(out: native.ptr[float], n: int, base: float) -> float:
        for i in range(n):
            native.store(native.offset(out, i), base + i)
        acc = 0.0
        for j in range(n):
            acc += native.load(native.offset(out, j))
        return acc


    def main() -> None:
        n = 20000
        out = native.stack_alloc[int](n)
        print(squares(out, n), native.load(native.offset(out, 777)))
        a = ppy.buffer[float](n)
        b = ppy.buffer[float](n)
        for i in range(n):
            a[i] = 0.5 * i
            b[i] = 1.0 / (i + 1)
        print(dot(a, b), abs(dot_relaxed(a, b) - dot(a, b)) < 1e-6)
        xs = ppy.buffer[int](n)
        for i in range(n):
            xs[i] = i
        print(count_odd(xs, 3))
        floats = native.stack_alloc[float](n)
        print(fill(floats, n, 0.25))


    main()
    """


def _paths(tmp_path: Path, source: str, backend: str = "threads") -> tuple[str, str, str]:
    (tmp_path / "pyproject.toml").write_text(
        f'[tool.ppy]\nstrict = true\n\n[tool.ppy.llvm]\npipeline = "ir"\n\n[tool.ppy.parallel]\nbackend = "{backend}"\nthreads = 4\n',
        encoding="utf-8",
    )
    (tmp_path / "prog.ppy").write_text(textwrap.dedent(source).lstrip("\n"), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "PPY_LOWERING"}
    outputs = []
    for args in (
        ["prog.ppy"],
        ["-m", "ppy_compiler", "prog.ppy"],
        ["-m", "ppy_compiler", "run", "prog.ppy"],
    ):
        done = subprocess.run(
            [sys.executable, *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        assert done.returncode == 0, done.stderr
        outputs.append(done.stdout)
    return outputs[0], outputs[1], outputs[2]


def _emit(tmp_path: Path, *args: str) -> str:
    done = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "emit", *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


@requires_llvm
@pytest.mark.parametrize("backend", ["threads", "serial", "simd"])
def test_parallel_loops_answer_the_same_on_every_path_and_backend(tmp_path: Path, backend: str):
    plain, python, native = _paths(tmp_path, PROGRAM, backend)
    assert plain == python == native
    lines = plain.splitlines()
    assert lines[0] == f"{19999 * 19999} {777 * 777}"
    assert lines[2] == str(3 * 10000)
    assert lines[3] == str(sum(0.25 + i for i in range(20000)))


@requires_llvm
def test_the_frontend_outlines_the_body_and_the_pass_lowers_it_for_the_backend(tmp_path: Path):
    _paths(tmp_path, PROGRAM, "threads")
    (tmp_path / "pyproject.toml").write_text(
        '[tool.ppy]\nstrict = true\n\n[tool.ppy.llvm]\npipeline = "ir"\n\n[tool.ppy.parallel]\nbackend = "openmp"\n',
        encoding="utf-8",
    )
    text = _emit(tmp_path, "ir", "prog.ppy")
    assert "dialect parallel 1" in text
    assert "parallel.for %" in text and "callee = @prog_squares__par1" in text
    assert 'op = "add", reassociate = false' in text, "dot keeps its order"
    assert "reassociate = true" in text, "dot_relaxed and count_odd may split"
    assert "func @prog_dot__par" in text and 'ppy.synthesized = "parallel"' in text
    assert text.count("parallel.for") == 2, "fill's two loops: one for, one reduce"
    c = _emit(tmp_path, "c", "prog.ppy")
    assert "#pragma omp parallel" in c and "/* compile with: -fopenmp */" in c
    if c_compiler() is not None:
        (tmp_path / "prog.c").write_text(c, encoding="utf-8")
        built = subprocess.run(
            [
                c_compiler(),
                "-std=c11",
                "-Wall",
                "-fopenmp",
                "-c",
                str(tmp_path / "prog.c"),
                "-o",
                str(tmp_path / "prog.o"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert built.returncode == 0, built.stderr
        assert "warning:" not in built.stderr, built.stderr


@requires_llvm
def test_remarks_say_what_became_parallel(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.ppy]\nstrict = true\n\n[tool.ppy.llvm]\npipeline = "ir"\n\n'
        "[tool.ppy.diagnostics]\noptimization-remarks = true\n\n[tool.ppy.parallel]\nthreads = 3\n",
        encoding="utf-8",
    )
    (tmp_path / "prog.ppy").write_text(textwrap.dedent(PROGRAM).lstrip("\n"), encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "build", "prog.ppy", "-o", "dist"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "the loop over `i` is a parallel loop" in done.stderr
    assert "parallel add reduction into `total` (kept in order" in done.stderr
    assert "lowered to 3 chunks" in done.stderr and "lowered to one chunk" in done.stderr


def test_the_checker_and_the_frontend_refuse_what_cannot_run_at_once(write, codes, analyze):
    path = write(
        "bad.ppy",
        """
        from ppy import native, parallel


        def f(out: native.ptr[int], n: int) -> int:
            last = 0
            for i in parallel.range(n, 2.5):
                native.store(native.offset(out, i), i)
                last = i
            for j in parallel.range(n, step=2):
                pass
            return last + parallel.other(n)
        """,
    )
    found = codes(path)
    assert found.count("E1650") == 1, found
    assert "E1301" in found and "E1306" in found, "a float bound, and a name the namespace lacks"
    from ppy_compiler.backend.llvm import _definitions, _value_class_layouts
    from ppy_compiler.lowering import lower_module_to_ir

    good = write(
        "escapes.ppy",
        """
        from ppy import native, parallel


        def f(out: native.ptr[int], n: int) -> int:
            last = 0
            for i in parallel.range(n):
                native.store(native.offset(out, i), i)
                last = i
            return last


        def g(xs: list[int]) -> int:
            total = 0
            for i in parallel.range(len(xs)):
                if total > 100:
                    break
                total += xs[i]
            return total
        """,
    )
    bundle = analyze(good, backend="llvm")
    symbols = bundle.symbols.modules["escapes"]
    analysis = bundle.analysis.modules["escapes"]
    candidates = {}
    for owner, node in _definitions(symbols.module.tree):
        info = symbols.functions.get(node.name) if not owner else None
        if info is not None and info.qualname in analysis.functions:
            candidates[info.qualname] = (info, analysis.functions[info.qualname], node)
    lowered = lower_module_to_ir(analysis, candidates, _value_class_layouts(bundle))
    assert "assigns `last`, which lives outside the loop" in lowered.rejected["escapes.f"]
    assert "cannot `break`" in lowered.rejected["escapes.g"]
