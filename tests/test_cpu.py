"""`ppy.cpu`: what the machine has, folded and honoured at the boundary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_runtime import _cpu
from ppy_runtime.abi import NativeParam, NativeSignature
from ppy_runtime.binding import bind

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")

PROGRAM = """
    from ppy import cpu, native


    @cpu.target({feature!r})
    def wide_sum(xs: list[float]) -> float:
        total = 0.0
        for x in xs:
            total += x
        return total


    def warm(p: native.const_ptr[float], n: int) -> float:
        for i in range(n):
            cpu.prefetch(native.offset(p, i), locality=2)
        return native.load(native.offset(p, n - 1))


    def lanes() -> int:
        cpu.pause()
        return cpu.vector_width[float]()


    def fast() -> bool:
        return {feature!r} in cpu.features()


    def main() -> None:
        values = native.stack_alloc[float](8)
        for i in range(8):
            native.store(native.offset(values, i), 0.5 * i)
        print(wide_sum([0.5 * i for i in range(8)]), warm(values, 8), lanes(), fast())


    main()
    """


def _three_paths(tmp_path: Path, source: str) -> tuple[str, str, str]:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.ppy]\nstrict = true\n\n[tool.ppy.llvm]\npipeline = "ir"\n', encoding="utf-8"
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


def _present_feature() -> str:
    have = _cpu.features()
    for candidate in ("avx2", "sse2", "neon", "fma"):
        if candidate in have:
            return candidate
    return have[0] if have else "sse2"


@requires_llvm
def test_features_fold_and_a_targeted_function_runs_natively(tmp_path: Path):
    feature = _present_feature()
    plain, python, native = _three_paths(tmp_path, PROGRAM.format(feature=feature))
    assert plain == python == native
    assert plain == f"14.0 3.5 {_cpu.vector_width(64)} True\n"
    emitted = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "emit", "ir", "prog.ppy"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert emitted.returncode == 0, emitted.stderr
    assert f'cpu.features = ["{feature}"]' in emitted.stdout
    assert "cpu.prefetch" in emitted.stdout and "cpu.pause" in emitted.stdout
    assert f"core.const {_cpu.vector_width(64)}" in emitted.stdout, "vector_width folds"
    assert "core.const true" in emitted.stdout, "the membership test folds"
    llvm = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "emit", "llvm-ir", "prog.ppy"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert llvm.returncode == 0, llvm.stderr
    assert f'"target-features"="+{feature}"' in llvm.stdout
    built = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "build", "prog.ppy", "-o", "dist"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    manifest = json.loads((tmp_path / "dist" / "ppy-bindings.json").read_text(encoding="utf-8"))
    entries = {e["python_qualname"]: e for e in manifest["entries"]}
    assert entries["prog.wide_sum"]["abi"]["cpu_features"] == [feature]


def test_a_function_compiled_for_more_than_this_machine_runs_its_python():
    signature = NativeSignature(
        "m.f", "ppy_m_f", (NativeParam("n", "int"),), ("i64",), cpu_features=("no-such-feature",)
    )
    calls: list[int] = []

    def fallback(n: int) -> int:
        calls.append(n)
        return n * 2

    binding = bind(signature, 0, fallback)
    assert binding.wrapper is fallback
    assert binding.wrapper(21) == 42 and calls == [21]
    present = NativeSignature("m.g", "ppy_m_g", (), ("i64",), cpu_features=_cpu.features()[:1])
    assert bind(present, 0, lambda: 0).wrapper is not fallback


def test_the_machine_describes_itself_once_for_everyone():
    from ppy import cpu

    assert cpu.features() == _cpu.features()
    assert cpu.vector_width[float]() == _cpu.vector_width(64)
    assert (
        cpu.vector_width[__import__("ppy").i8]()
        == _cpu.vector_width(8)
        >= cpu.vector_width[float]()
    )
    assert cpu.prefetch(None) is None and cpu.pause() is None
    with pytest.raises(TypeError):
        cpu.vector_width[str]()
    with pytest.raises(TypeError):
        cpu.target()
    assert all(f == f.lower() for f in _cpu.features())
    assert _cpu.vector_bits() in {64, 128, 256, 512}


def test_the_checker_names_what_cpu_refuses(write, codes):
    path = write(
        "bad.ppy",
        """
        from ppy import cpu, native


        def f(p: native.ptr[float]) -> int:
            cpu.prefetch(3)
            cpu.prefetch(p, locality=7)
            cpu.vector_width[str]()
            cpu.nothing()
            return cpu.vector_width[float]()
        """,
    )
    found = codes(path)
    assert found.count("E1643") == 4, found
