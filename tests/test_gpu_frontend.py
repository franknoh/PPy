"""`ppy.cuda` and `ppy.hip`: the reference launch, the checker, the gpu IR, and the source."""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy import cuda, hip, native
from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.ir import decode, verify

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")

PROGRAM = """
    from ppy import cuda, native


    @cuda.device
    def fma(a: float, x: float, y: float) -> float:
        return a * x + y


    @cuda.kernel
    def saxpy(n: int, a: float, x: native.const_ptr[float], y: native.ptr[float]) -> None:
        i = cuda.global_id()
        if i < n:
            slot = native.offset(y, i)
            native.store(slot, fma(a, native.load(native.offset(x, i)), native.load(slot)))


    @cuda.kernel
    def block_max(x: native.const_ptr[float], out: native.ptr[float]) -> None:
        parked = cuda.shared[float, 64]()
        tid = cuda.thread_id()
        native.store(native.offset(parked, tid), native.load(native.offset(x, cuda.global_id())))
        cuda.syncthreads()
        mine = native.load(native.offset(parked, tid))
        other = cuda.shfl_xor(mine, 1)
        if other > mine:
            mine = other
        native.store(native.offset(parked, tid), mine)
        cuda.syncthreads()
        if tid == 0:
            best = native.load(parked)
            for k in range(1, cuda.block_dim()):
                candidate = native.load(native.offset(parked, k))
                if candidate > best:
                    best = candidate
            native.store(native.offset(out, cuda.block_id()), best)


    def run(n: int, a: float, x: native.const_ptr[float], y: native.ptr[float]) -> None:
        cuda.launch(saxpy, (n + 255) // 256, 256, n, a, x, y)


    def main() -> None:
        n = 300
        x = native.stack_alloc[float](n)
        y = native.stack_alloc[float](n)
        for i in range(n):
            native.store(native.offset(x, i), float(i))
            native.store(native.offset(y, i), 1.0)
        run(n, 2.0, x, y)
        total = 0.0
        for i in range(n):
            total += native.load(native.offset(y, i))
        print(total)
        values = native.stack_alloc[float](128)
        for i in range(128):
            native.store(native.offset(values, i), float((i * 37) % 101))
        out = native.stack_alloc[float](2)
        cuda.launch(block_max, 2, 64, values, out)
        print(native.load(out), native.load(native.offset(out, 1)))
    """


def _program() -> dict:  # type: ignore[type-arg]
    namespace: dict = {}
    exec(textwrap.dedent(PROGRAM), namespace)  # the program under test
    return namespace


def _filled(count: int, values) -> native.ptr:  # type: ignore[no-untyped-def]
    pointer = native.stack_alloc[float](count)
    for i, value in enumerate(values):
        native.store(native.offset(pointer, i), float(value))
    return pointer


def test_the_reference_launch_runs_the_grid_on_threads():
    program = _program()
    x = _filled(300, range(300))
    y = _filled(300, [1.0] * 300)
    program["run"](300, 2.0, x, y)
    assert [native.load(native.offset(y, i)) for i in (0, 1, 255, 299)] == [1.0, 3.0, 511.0, 599.0]
    values = _filled(128, ((i * 37) % 101 for i in range(128)))
    out = native.stack_alloc[float](2)
    cuda.launch(program["block_max"], 2, 64, values, out)
    expected = [max((i * 37) % 101 for i in range(start, start + 64)) for start in (0, 64)]
    assert [native.load(out), native.load(native.offset(out, 1))] == expected
    hip.launch(program["block_max"], (2,), (64, 1), values, out)
    assert [native.load(out), native.load(native.offset(out, 1))] == expected
    assert cuda.warp_size() == 32 and hip.warp_size() == 64


def test_positions_shuffles_and_memory_follow_the_device_rules():
    seen: dict[tuple, tuple] = {}

    @cuda.kernel
    def probe(count: native.ptr[int]) -> None:
        lane = cuda.thread_id()
        up = cuda.shfl_up(lane, 1)
        down = cuda.shfl_down(lane, 1)
        picked = cuda.shfl(lane * 10, 3)
        mine = cuda.local[int, 2]()
        native.store(mine, lane)
        cuda.syncwarp()
        seen[(cuda.block_id("x"), cuda.block_id("y"), lane)] = (
            up,
            down,
            picked,
            cuda.global_id("y"),
            cuda.block_dim(),
            cuda.grid_dim("y"),
            native.load(mine),
        )
        native.store(count, native.load(count) + 1)

    count = native.stack_alloc[int](1)
    cuda.launch(probe, (1, 2), 40, count)
    assert native.load(count) == 80, "every thread of every block ran, one at a time or not"
    assert seen[(0, 1, 0)] == (0, 1, 30, 1, 40, 2, 0), "lane 0 keeps its own value on shfl_up"
    assert seen[(0, 0, 31)] == (30, 31, 30, 0, 40, 2, 31), "the last lane of a full warp"
    assert seen[(0, 0, 32)] == (32, 33, 350, 0, 40, 2, 32), "a partial warp of 8: its lane 3 is 35"
    assert seen[(0, 0, 39)] == (38, 39, 350, 0, 40, 2, 39)
    with pytest.raises(RuntimeError, match="inside a kernel"):
        cuda.thread_id()
    with pytest.raises(ValueError, match="positive int or a tuple"):
        cuda.launch(probe, 0, 1, count)
    with pytest.raises(TypeError, match="kernel function"):
        cuda.launch(3, 1, 1)

    @hip.kernel
    def failing(p: native.ptr[int]) -> None:
        if hip.thread_id() == 5:
            raise ZeroDivisionError("lane five")
        hip.syncthreads()

    with pytest.raises(ZeroDivisionError, match="lane five"):
        hip.launch(failing, 1, 64, count)


def test_the_checker_types_the_vocabulary_and_names_a_misuse(write, codes):
    assert codes(write("gpu_ok.ppy", PROGRAM), backend="llvm") == []
    misused = write(
        "gpu_bad.ppy",
        """
        from ppy import cuda, native


        @cuda.kernel
        def bad(p: native.ptr[float]) -> None:
            i = cuda.thread_id("w")
            s = cuda.shared[float]()
            v = cuda.shfl(p, 1)
            cuda.syncthreads(1)
            w = cuda.frobnicate()


        def host(p: native.ptr[float], n: int) -> None:
            cuda.launch(bad, (1, 2, 3, 4), 8, p)
            cuda.launch(bad, n, 8, p, n)
            q = cuda.device_alloc[float]()
            r = cuda.device_alloc[str](n)
            t = cuda.device_alloc[float, 4](n)
        """,
    )
    assert codes(misused, backend="llvm") == ["E1644"] * 10


def test_device_memory_is_a_pointer_the_host_reads_and_writes_through(monkeypatch):
    """Without a device, `device_alloc` is its mirror; every path sees one memory."""
    from ppy_runtime import cuda as runtime

    monkeypatch.setenv("PPY_NO_CUDA", "1")
    monkeypatch.setattr(runtime, "_driver", None)
    program = _program()
    x = cuda.device_alloc[float](300)
    y = hip.device_alloc[float](300)
    assert isinstance(x, native.Pointer) and "on the device" in repr(x)
    for i in range(300):
        native.store(native.offset(x, i), float(i))
        native.store(native.offset(y, i), 1.0)
    assert isinstance(native.offset(y, 7), type(y)), "an offset keeps the kind"
    program["run"](300, 2.0, x, y)
    cuda.launch(program["saxpy"], 2, 256, 300, 2.0, x, y)
    assert [native.load(native.offset(y, i)) for i in (0, 1, 299)] == [1.0, 5.0, 1197.0]
    with pytest.raises(ValueError, match="how many"):
        cuda.device_alloc[int](-1)
    with pytest.raises(TypeError):
        cuda.device_alloc[str](1)


@requires_llvm
def test_the_ir_carries_the_kinds_and_the_source_backends_write_them(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    (tmp_path / "gpu_prog.ppy").write_text(textwrap.dedent(PROGRAM).lstrip("\n"), encoding="utf-8")
    ir = _ppy(tmp_path, "emit", "ir", "gpu_prog.ppy")
    assert ir.returncode == 0, ir.stderr
    text = ir.stdout
    assert 'attrs {effects = [], gpu.kind = "device"' in text or 'gpu.kind = "device"' in text
    assert 'gpu.kind = "kernel"' in text and "gpu.thread_id.x : index" in text
    assert "gpu.shared_alloc {count = 64} : ptr<f64, shared>" in text and "gpu.barrier" in text
    assert "gpu.subgroup_shuffle.xor" in text and "gpu.launch %" in text
    assert "core.guard" not in text.split("func @gpu_prog_saxpy")[1].split("func @gpu_prog_run")[0]
    assert not verify(decode(text))
    cuda_text = _ppy(tmp_path, "emit", "cuda", "gpu_prog.ppy")
    assert cuda_text.returncode == 0, cuda_text.stderr
    source = cuda_text.stdout
    assert source.startswith("/* gpu_prog: generated by ppy, CUDA C++ */\n")
    assert "#include <cuda_runtime.h>" in source and "#include <cstdint>" in source
    assert (
        "__global__ void ppy_gpu_prog_saxpy(int64_t n, double a, const double *x, double *y)"
        in source
    )
    assert "static __device__ double ppy_gpu_prog_fma(double a, double x, double y)" in source
    assert "(int64_t)threadIdx.x" in source and "(int64_t)blockIdx.x" in source
    assert "__syncthreads();" in source and "__shared__ double" in source
    assert "__shfl_xor_sync(0xffffffffu, " in source
    assert (
        "ppy_gpu_prog_saxpy<<<dim3(" in source
        and "cudaDeviceSynchronize() != cudaSuccess) return 1;" in source
    )
    assert "goto" not in source
    assert _ppy(tmp_path, "emit", "cuda", "gpu_prog.ppy").stdout == source, "deterministic"
    hip_text = _ppy(tmp_path, "emit", "hip", "gpu_prog.ppy")
    assert hip_text.returncode == 0, hip_text.stderr
    assert "#include <hip/hip_runtime.h>" in hip_text.stdout and "HIP C++" in hip_text.stdout
    assert (
        "__shfl_xor(" in hip_text.stdout
        and "hipDeviceSynchronize() != hipSuccess) return 1;" in hip_text.stdout
    )
    plain_c = _ppy(tmp_path, "emit", "c", "gpu_prog.ppy")
    assert "__global__" not in plain_c.stdout and "saxpy" not in plain_c.stdout, (
        "C leaves device code"
    )
    _compile_if_present(tmp_path, "nvcc", "unit.cu", source)
    _compile_if_present(tmp_path, "hipcc", "unit.hip", hip_text.stdout)


def _compile_if_present(directory: Path, compiler: str, name: str, source: str) -> None:
    found = shutil.which(compiler)
    if found is None:
        return
    path = directory / name
    path.write_text(source, encoding="utf-8")
    done = subprocess.run(
        [found, "-c", str(path), "-o", str(directory / (name + ".o"))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr


@requires_llvm
def test_the_program_runs_the_same_under_ppy_run(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    entry = tmp_path / "gpu_run.ppy"
    entry.write_text(textwrap.dedent(PROGRAM).lstrip("\n") + "\n\nmain()\n", encoding="utf-8")
    plain = subprocess.run(
        [sys.executable, entry.name], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    native_run = _ppy(tmp_path, "run", entry.name)
    assert plain.returncode == 0, plain.stderr
    assert native_run.returncode == 0, native_run.stderr
    assert native_run.stdout == plain.stdout == "90000.0\n100.0 98.0\n"


def _ppy(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ppy_compiler", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
