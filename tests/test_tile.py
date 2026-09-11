"""`ppy.tile`: kernels over tiles -- the reference launch, the checker, the lowering to a
block of threads, and the same answers under `ppy run` (spec 74)."""

from __future__ import annotations

import array
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy import native, tile
from ppy._native_api import Pointer
from ppy_compiler.backend.llvm import available as llvm_available

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")

PROGRAM = """
    from ppy import native, tile


    @tile.kernel
    def saxpy(n: int, a: float, x: native.const_ptr[float], y: native.ptr[float]) -> None:
        offsets = tile.program_id() * 256 + tile.arange(256)
        mask = offsets < n
        xs = tile.load(x, offsets, mask)
        ys = tile.load(y, offsets, mask)
        tile.store(y, offsets, a * xs + ys, mask)


    @tile.kernel
    def stats(n: int, x: native.const_ptr[float], out: native.ptr[float]) -> None:
        pid = tile.program_id()
        offsets = pid * 1024 + tile.arange(1024)
        mask = offsets < n
        xs = tile.load(x, offsets, mask, -1.0)
        tile.store(out, pid * 4, tile.sum(tile.where(mask, xs, 0.0)))
        tile.store(out, pid * 4 + 1, tile.min(tile.where(mask, xs, 1e300)))
        tile.store(out, pid * 4 + 2, tile.max(xs))
        tile.store(out, pid * 4 + 3, tile.sum(mask))


    @tile.kernel
    def digest(n: int, x: native.const_ptr[int], y: native.ptr[int]) -> None:
        offsets = tile.program_id() * 64 + tile.arange(64)
        mask = offsets < n
        xs = tile.load(x, offsets, mask)
        tile.store(y, offsets, (xs * 3 + offsets) % 7 - xs // 2, mask)
        tile.store(y, tile.program_id() + n, tile.sum(xs) + tile.max(xs))


    def run(n: int, a: float, x: native.const_ptr[float], y: native.ptr[float]) -> None:
        tile.launch(saxpy, (n + 255) // 256, n, a, x, y)


    def main() -> None:
        n = 1000
        x = native.stack_alloc[float](n)
        y = native.stack_alloc[float](n)
        for i in range(n):
            native.store(native.offset(x, i), float(i))
            native.store(native.offset(y, i), 1.0)
        run(n, 2.0, x, y)
        total = 0.0
        for i in range(n):
            total += native.load(native.offset(y, i))
        print(total, native.load(native.offset(y, 999)))
        values = native.stack_alloc[float](3000)
        for i in range(3000):
            native.store(native.offset(values, i), float((i * 37) % 101) - 50.0)
        out = native.stack_alloc[float](12)
        tile.launch(stats, 3, 3000, values, out)
        print([native.load(native.offset(out, i)) for i in range(12)])
        a = native.stack_alloc[int](200)
        b = native.stack_alloc[int](204)
        for i in range(200):
            native.store(native.offset(a, i), (i * 7919) % 1000)
        tile.launch(digest, 4, 200, a, b)
        print(sum(native.load(native.offset(b, i)) for i in range(204)))


    main()
    """

EXPECTED = (
    "1000000.0 1999.0\n"
    "[-60.0, -50.0, 50.0, 1024.0, 21.0, -50.0, 50.0, 1024.0, 59.0, -50.0, 50.0, 952.0]\n"
    "54105\n"
)


def _filled(count: int, values, element=float):  # type: ignore[no-untyped-def]
    pointer = native.stack_alloc[element](count)
    for i, value in enumerate(values):
        native.store(native.offset(pointer, i), element(value))
    return pointer


def test_the_reference_launch_runs_the_programs_in_order():
    @tile.kernel
    def saxpy(n: int, a: float, x: native.const_ptr[float], y: native.ptr[float]) -> None:
        offsets = tile.program_id() * 256 + tile.arange(256)
        mask = offsets < n
        tile.store(y, offsets, a * tile.load(x, offsets, mask) + tile.load(y, offsets, mask), mask)

    x = _filled(300, range(300))
    y = _filled(300, [1.0] * 300)
    tile.launch(saxpy, (300 + 255) // 256, 300, 2.0, x, y)
    assert [native.load(native.offset(y, i)) for i in (0, 1, 255, 299)] == [1.0, 3.0, 511.0, 599.0]
    assert not tile.compiled(saxpy)

    seen: list[tuple[int, int]] = []

    @tile.kernel
    def probe(out: native.ptr[int]) -> None:
        seen.append((tile.program_id(), tile.num_programs()))
        lanes = tile.arange(64)
        picked = tile.where(lanes < 3, lanes * 10, -1)
        tile.store(out, tile.program_id(), tile.sum(picked) + tile.max(lanes) - tile.min(lanes))

    out = _filled(3, [0, 0, 0], int)
    tile.launch(probe, 3, out)
    assert seen == [(0, 3), (1, 3), (2, 3)]
    assert [native.load(native.offset(out, i)) for i in range(3)] == [30 - 61 + 63] * 3
    with pytest.raises(RuntimeError, match="inside a launched kernel"):
        tile.program_id()
    with pytest.raises(ValueError, match="positive int"):
        tile.launch(probe, 0, out)
    with pytest.raises(TypeError, match="kernel function"):
        tile.launch(3, 1)


def test_tiles_are_lane_by_lane_values():
    lanes = tile.arange(32)
    assert len(lanes) == 32 and list(lanes)[:3] == [0, 1, 2]
    scaled = lanes * 2.5 + 1
    assert scaled.kind is float and list(scaled)[:2] == [1.0, 3.5]
    assert list(lanes % 4 == 0)[:5] == [True, False, False, False, True]
    assert list(~(lanes < 2))[:3] == [False, False, True]
    assert list(3 - lanes)[:2] == [3, 2] and list(-lanes)[1] == -1
    assert tile.sum(lanes) == 496 and tile.max(lanes) == 31 and tile.min(lanes) == 0
    assert list(tile.where(lanes < 1, 7, lanes))[:2] == [7, 1]
    with pytest.raises(ValueError, match="no lanes in common"):
        _ = lanes + tile.arange(64)
    frozen = Pointer(array.array("d", [0.0] * 32), 0, float, mutable=False)
    with pytest.raises(TypeError, match="const_ptr"):
        tile.store(frozen, lanes, 1.0)
    with pytest.raises(TypeError, match="tile of bools"):
        tile.where(lanes, 1, 2)


def test_the_checker_types_the_vocabulary_and_names_a_misuse(write, codes):
    path = write(
        "tiles.ppy",
        """
        from ppy import native, tile


        @tile.kernel
        def fine(n: int, x: native.const_ptr[float], y: native.ptr[float]) -> None:
            offsets = tile.program_id() * 256 + tile.arange(256)
            mask = offsets < n
            xs = tile.load(x, offsets, mask, 0.0)
            tile.store(y, offsets, xs * 2.0 + 1, mask)
            tile.store(y, tile.program_id(), tile.sum(xs) + tile.max(offsets))


        @tile.kernel
        def wrong(x: native.const_ptr[float], y: native.ptr[float]) -> None:
            offsets = tile.arange(48)
            xs = tile.load(x, offsets, offsets)
            tile.store(x, offsets, xs)
            tile.store(y, 0, xs)
            tile.sum(3)
            tile.where(xs, 1, 2)
            tile.launch(fine, 1.5, 1, x, y)
            tile.load(x, offsets) ** 2


        def host(n: int, x: native.const_ptr[float], y: native.ptr[float]) -> None:
            tile.launch(fine, (n + 255) // 256, n, x, y)
            tile.launch(fine, 1, n)
        """,
    )
    assert codes(path) == [
        "E1644",  # arange(48) is not a power of two
        "E1644",  # the mask is not a tile of bools
        "E1631",  # a store through a const_ptr
        "E1644",  # one element takes a scalar
        "E1644",  # sum of a scalar
        "E1644",  # where takes a tile of bools first
        "E1644",  # programs is an int
        "E1644",  # ** has no tile form
        "E1644",  # the launch's argument count
    ]


@requires_llvm
def test_a_tile_kernel_lowers_to_a_block_of_threads(tmp_path: Path):
    source = tmp_path / "tiles.ppy"
    source.write_text(textwrap.dedent(PROGRAM).lstrip("\n"), encoding="utf-8")
    emitted = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "emit", "ir", "tiles.ppy"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert emitted.returncode == 0, emitted.stderr
    text = emitted.stdout
    kernel = text[text.index("@tiles_stats") : text.index("@tiles_digest")]
    assert "gpu.thread_id" in kernel and "gpu.block_id" in kernel
    assert "simd.reduce_add" in kernel and "gpu.subgroup_shuffle" in kernel
    assert "gpu.shared_alloc" in kernel and "gpu.barrier" in kernel
    assert "vector<f64, 4>" in kernel, "1024 lanes over 256 threads: four per thread"
    launch = text[text.index("@tiles_run") :]
    assert "gpu.launch" in launch and "core.const 256" in launch, "256 threads share a program"


@requires_llvm
def test_the_program_runs_the_same_under_ppy_run(tmp_path: Path):
    source = tmp_path / "tiles.ppy"
    source.write_text(textwrap.dedent(PROGRAM).lstrip("\n"), encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = false\n", encoding="utf-8")
    plain = subprocess.run(
        [sys.executable, "tiles.ppy"], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert plain.returncode == 0 and plain.stdout == EXPECTED, plain.stderr
    ran = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "run", "tiles.ppy"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == EXPECTED, "on the device or through the reference launch, the same"
    assert "W2004" not in ran.stderr, ran.stderr


NVCC = shutil.which("nvcc") or (
    "/usr/local/cuda/bin/nvcc" if Path("/usr/local/cuda/bin/nvcc").is_file() else None
)


@requires_llvm
@pytest.mark.skipif(NVCC is None, reason="nvcc is neither on PATH nor in /usr/local/cuda")
def test_the_cuda_source_backend_writes_a_unit_nvcc_compiles(tmp_path: Path):
    source = tmp_path / "tiles.ppy"
    source.write_text(textwrap.dedent(PROGRAM).lstrip("\n"), encoding="utf-8")
    emitted = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "emit", "cuda", "tiles.ppy", "-o", "unit.cu"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert emitted.returncode == 0, emitted.stderr
    compiled = subprocess.run(
        [str(NVCC), "-c", "-o", "unit.o", "unit.cu"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
