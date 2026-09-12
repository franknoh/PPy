"""`ppy.native.compiled(f)` says whether calling `f` runs natively here, for whichever
object stands in the namespace, and that object bears the function's name."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")

PROGRAM = """
    import array

    import ppy
    from ppy import Buffer


    def total(xs: Buffer[float], n: int) -> float:
        s: float = 0.0
        for i in range(n):
            s += xs[i]
        return s


    def main() -> None:
        xs = array.array("d", [0.0, 1.0, 2.0, 3.0])
        print(total(xs, 4))
        print(ppy.native.compiled(total), total.__name__)


    main()
"""


@requires_llvm
def test_the_probe_answers_for_the_object_in_the_namespace(tmp_path: Path):
    """Under `ppy run` the namespace may hold the C entry point itself, which no
    attribute can be hung on; the probe knows it anyway, and it bears the qualified name."""
    entry = tmp_path / "probe.ppy"
    entry.write_text(textwrap.dedent(PROGRAM).lstrip("\n"), encoding="utf-8")
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
    assert plain.stdout == "6.0\nFalse total\n"
    assert native.stdout == "6.0\nTrue probe.total\n", native.stderr


def test_the_probe_is_false_for_a_plain_function_and_for_no_runtime():
    import ppy  # pylint: disable=import-outside-toplevel

    assert ppy.native.compiled(len) is False
    assert ppy.native.compiled(lambda: None) is False
