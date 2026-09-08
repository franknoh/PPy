"""`ppy.simd`: lanes on every path, and what the checker refuses."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")

PROGRAM = """
    import ppy
    from ppy import native, simd


    def dot(a: native.const_ptr[float], b: native.const_ptr[float]) -> float:
        return simd.reduce_add(simd.load[float, 4](a) * simd.load[float, 4](b))


    def ramp(p: native.ptr[int], start: int) -> int:
        v = simd.splat[int, 4](start)
        v = simd.insert(v, 1, start + 1)
        v = simd.insert(v, 2, start + 2)
        v = simd.insert(v, 3, start + 3)
        simd.store(v + v, p)
        return simd.reduce_max(v) - simd.reduce_min(v)


    def mix(p: native.ptr[float]) -> float:
        v = simd.load[float, 4](p)
        w = simd.shuffle(v, v, (3, 2, 1, 0))
        chosen = simd.select(v > w, v, w)
        simd.store(chosen / simd.splat[float, 4](2.0), p)
        return simd.extract(chosen, 0) + simd.reduce_add(-w)


    def bytes_sum(p: native.const_ptr[ppy.u8]) -> int:
        v = simd.load[ppy.u8, 8](p)
        return simd.reduce_add(v & simd.splat[ppy.u8, 8](15))


    def main() -> None:
        a = native.stack_alloc[float](4)
        b = native.stack_alloc[float](4)
        for i in range(4):
            native.store(native.offset(a, i), 1.0 + i)
            native.store(native.offset(b, i), 0.5 * i)
        print(dot(a, b))
        ints = native.stack_alloc[int](4)
        print(ramp(ints, 9223372036854775805), native.load(native.offset(ints, 3)))
        print(mix(a), native.load(native.offset(a, 0)), native.load(native.offset(a, 3)))
        raw = native.stack_alloc[ppy.u8](8)
        for i in range(8):
            native.store(native.offset(raw, i), (250 + i) & 255)
        print(bytes_sum(raw))


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


@requires_llvm
def test_vectors_answer_the_same_on_every_path(tmp_path: Path):
    plain, python, native = _three_paths(tmp_path, PROGRAM)
    assert plain == python == native
    lines = plain.splitlines()
    assert lines[0] == "10.0"
    assert lines[1] == f"{2**64 - 1} 0", (
        "the lanes wrap; the scalar difference falls back to Python"
    )
    assert lines[2] == "-6.0 2.0 2.0"
    assert lines[3] == str(sum((250 + i) & 15 for i in range(8)))
    emitted = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "emit", "ir", "prog.ppy"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert emitted.returncode == 0, emitted.stderr
    assert "dialect simd 1" in emitted.stdout
    assert "simd.reduce_add" in emitted.stdout and "vector<f64, 4>" in emitted.stdout
    assert "simd.shuffle" in emitted.stdout and "mask = [3, 2, 1, 0]" in emitted.stdout
    for name in ("prog_dot", "prog_ramp", "prog_mix", "prog_bytes_sum"):
        assert f"func @{name}" in emitted.stdout, f"{name} lowered natively"


def test_the_reference_implementation_keeps_lane_semantics():
    import ppy
    from ppy import native, simd

    v = simd.splat[int, 4](2**63 - 1)
    w = v + simd.splat[int, 4](1)
    assert w.lanes == (-(2**63),) * 4, "integer lanes wrap"
    assert (v > w).lanes == (True,) * 4
    assert simd.select(v > w, w, v).lanes == w.lanes
    assert simd.reduce_add(simd.splat[float, 4](0.1)) == ((0.1 + 0.1) + 0.1) + 0.1
    nan = float("nan")
    lanes = simd.insert(simd.splat[float, 3](1.0), 1, nan)
    assert simd.reduce_min(lanes) == 1.0 and simd.reduce_max(lanes) == 1.0
    first_nan = simd.insert(simd.splat[float, 2](1.0), 0, nan)
    assert simd.reduce_min(first_nan) != simd.reduce_min(first_nan)
    memory = native.stack_alloc[ppy.u8](4)
    simd.store(simd.splat[ppy.u8, 4](300), memory)
    assert native.load(native.offset(memory, 3)) == 300 - 256
    with pytest.raises(TypeError):
        _ = simd.splat[float, 4](1.0) + simd.splat[float, 2](1.0)
    with pytest.raises(TypeError):
        _ = simd.splat[int, 4](1) / simd.splat[int, 4](2)
    with pytest.raises(IndexError):
        simd.shuffle(v, v, (0, 8))
    assert simd.shuffle(v, w, (0, 4)).lanes == (2**63 - 1, -(2**63))
    assert repr(simd.Vector[float, 2]).startswith("typing.Annotated")


def test_the_checker_refuses_what_lanes_cannot_do(tmp_path: Path, write, codes):
    path = write(
        "bad.ppy",
        """
        from ppy import native, simd


        def f(p: native.ptr[float], q: native.ptr[int]) -> float:
            v = simd.load[float, 4](p)
            w = simd.load[float, 2](p)
            _ = v + w
            _ = v / simd.load[int, 4](q)
            _ = simd.load[int, 4](q) / simd.load[int, 4](q)
            _ = simd.shuffle(v, v, (0, 9))
            _ = simd.reduce_add(v > v)
            _ = simd.load[str, 4](p)
            simd.store(v, q)
            return simd.extract(v, 1.5)
        """,
    )
    found = codes(path)
    assert found.count("E1640") >= 7, found
    assert "E1301" not in found or found.count("E1640") >= 7
