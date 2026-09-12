"""The NVVM backend and the launch runtime: device code as NVPTX IR and PTX, run by the driver."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from test_gpu_frontend import PROGRAM, _filled, _ppy, _program

from ppy import cuda, native
from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.nvvm import available as nvptx_available
from ppy_compiler.backend.nvvm import emit_module, libdevice_path, ptx_from_ir
from ppy_runtime import cuda as runtime

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
requires_nvptx = pytest.mark.skipif(
    not (llvm_available() and nvptx_available()), reason="this LLVM has no NVPTX backend"
)
requires_device = pytest.mark.skipif(runtime.driver() is None, reason="no CUDA driver or device")
requires_libdevice = pytest.mark.skipif(
    libdevice_path() is None, reason="libdevice (the CUDA toolkit) is not installed"
)

MATH_PROGRAM = """
    import math

    from ppy import cuda, native


    @cuda.kernel
    def wave(x: native.ptr[float]) -> None:
        slot = native.offset(x, cuda.global_id())
        native.store(slot, math.sin(native.load(slot)) + math.exp(0.5))
    """


def _ir(write, analyze, name: str, source: str):  # type: ignore[no-untyped-def]
    from ppy_compiler.backend.llvm.ir_pipeline import ir_modules

    path = write(f"{name}.ppy", source)
    bundle = analyze(path, backend="llvm")
    assert not bundle.diagnostics.has_errors(), [d.message for d in bundle.diagnostics.errors]
    return bundle, ir_modules(bundle)[name]


@requires_llvm
def test_device_code_is_written_as_nvptx_ir(write, analyze):
    _bundle, module = _ir(write, analyze, "gpu_prog", PROGRAM)
    text = emit_module(module)
    for marker in (
        'target triple = "nvptx64-nvidia-cuda"',
        'define ptx_kernel void @"ppy_gpu_prog_saxpy"(',
        'define internal double @"ppy_gpu_prog_fma"(',
        "llvm.nvvm.read.ptx.sreg.tid.x",
        "llvm.nvvm.read.ptx.sreg.ctaid.x",
        "llvm.nvvm.read.ptx.sreg.ntid.x",
        "addrspace(3)",
        "llvm.nvvm.barrier0",
        "llvm.nvvm.shfl.sync.bfly.i32",
    ):
        assert marker in text, marker
    assert "ppy_gpu_prog_run" not in text and "fallback" not in text
    _bundle, math_module = _ir(write, analyze, "gpu_math", MATH_PROGRAM)
    math_text = emit_module(math_module)
    assert 'declare double @"__nv_sin"(double' in math_text
    assert "@__nv_exp" not in math_text, "a constant fold leaves only what the kernel computes"
    assert (
        emit_module(_ir(write, analyze, "plain", "def f(x: int) -> int:\n    return x + 1\n")[1])
        == ""
    )


@requires_nvptx
@requires_libdevice
def test_ptx_is_written_for_the_kernels_and_the_cli_emits_it(write, analyze, tmp_path):
    _bundle, module = _ir(write, analyze, "gpu_prog", PROGRAM)
    ptx = ptx_from_ir(emit_module(module))
    for marker in (
        ".entry ppy_gpu_prog_saxpy(",
        ".entry ppy_gpu_prog_block_max(",
        ".shared .align 8 .b8",
        "bar.sync",
        "shfl.sync.bfly.b32",
    ):
        assert marker in ptx, marker
    assert ".entry ppy_gpu_prog_fma" not in ptx and "ppy_gpu_prog_run" not in ptx
    _bundle, math_module = _ir(write, analyze, "gpu_math", MATH_PROGRAM)
    math_ptx = ptx_from_ir(emit_module(math_module))
    assert ".entry ppy_gpu_math_wave(" in math_ptx and ".func" in math_ptx.split("__nv_sin")[0], (
        "libdevice is linked: its sine is defined in the PTX, not left to resolve"
    )
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    (tmp_path / "prog.ppy").write_text(textwrap.dedent(PROGRAM).lstrip("\n"), encoding="utf-8")
    nvvm = _ppy(tmp_path, "emit", "nvvm-ir", "prog.ppy")
    assert nvvm.returncode == 0, nvvm.stderr
    assert 'define ptx_kernel void @"ppy_prog_saxpy"(' in nvvm.stdout
    emitted = _ppy(tmp_path, "emit", "ptx", "prog.ppy")
    assert emitted.returncode == 0, emitted.stderr
    assert ".entry ppy_prog_saxpy(" in emitted.stdout


@requires_device
def test_a_staged_kernel_launches_on_the_device(write, analyze):
    from ppy_compiler.driver.staging import stage_project
    from ppy_runtime.exported import bind_exported

    bundle, _module = _ir(write, analyze, "gpu_prog", PROGRAM)
    staged = stage_project(bundle)
    assert staged.names("gpu_prog") == {"saxpy", "block_max"}, staged.skipped
    payload = staged.artifacts["gpu_prog"]["saxpy"].payload
    described = json.loads(payload)
    assert described["kind"] == "ppy.cuda" and described["symbol"] == "ppy_gpu_prog_saxpy"
    assert described["params"] == [
        {"kind": "int"},
        {"kind": "float"},
        {"kind": "const_ptr"},
        {"kind": "ptr"},
    ]
    program = _program()
    binding = bind_exported("saxpy", payload, program["saxpy"])
    assert binding.routed and cuda.compiled(binding.wrapper) and not cuda.compiled(program["saxpy"])
    x = _filled(300, range(300))
    y = _filled(300, [1.0] * 300)
    cuda.launch(binding.wrapper, (300 + 255) // 256, 256, 300, 2.0, x, y)
    assert [native.load(native.offset(y, i)) for i in (0, 1, 255, 299)] == [1.0, 3.0, 511.0, 599.0]
    assert (binding.calls, binding.fallbacks) == (1, 0)
    with pytest.raises(RuntimeError, match="inside a kernel"):
        binding.wrapper(300, 2.0, x, y)
    reduction = bind_exported(
        "block_max", staged.artifacts["gpu_prog"]["block_max"].payload, program["block_max"]
    )
    values = _filled(128, ((i * 37) % 101 for i in range(128)))
    out = native.stack_alloc[float](2)
    cuda.launch(reduction.wrapper, 2, 64, values, out)
    assert [native.load(out), native.load(native.offset(out, 1))] == [100.0, 98.0]
    assert stage_project(bundle).artifacts["gpu_prog"]["saxpy"].payload == payload, "cached"


@requires_device
def test_device_memory_stays_on_the_device_between_launches(monkeypatch, write, analyze):
    """`device_alloc` memory is uploaded once, launched over as often as asked,
    and downloaded when the host reads it."""
    from ppy_compiler.driver.staging import stage_project
    from ppy_runtime.exported import bind_exported

    counts = {"upload": 0, "download": 0}
    for name in counts:
        original = getattr(runtime.Driver, name)

        def counted(self, *args, _name=name, _original=original):  # type: ignore[no-untyped-def]
            counts[_name] += 1
            return _original(self, *args)

        monkeypatch.setattr(runtime.Driver, name, counted)
    bundle, _module = _ir(write, analyze, "gpu_prog", PROGRAM)
    payload = stage_project(bundle).artifacts["gpu_prog"]["saxpy"].payload
    binding = bind_exported("saxpy", payload, _program()["saxpy"])
    assert binding.routed
    n = 1000
    x = cuda.device_alloc[float](n)
    y = cuda.device_alloc[float](n)
    for i in range(n):
        native.store(native.offset(x, i), float(i))
        native.store(native.offset(y, i), 1.0)
    assert counts == {"upload": 0, "download": 0}, "host writes go to the mirror"
    for _ in range(6):
        cuda.launch(binding.wrapper, (n + 255) // 256, 256, n, 2.0, x, y)
    assert counts == {"upload": 2, "download": 0}, "one upload per array, then none"
    assert native.load(native.offset(y, 999)) == 1.0 + 2.0 * 999 * 6
    assert counts == {"upload": 2, "download": 1}, "the host read brought the result back"
    native.store(y, 5.0)
    cuda.launch(binding.wrapper, (n + 255) // 256, 256, n, 2.0, x, y)
    assert native.load(y) == 5.0 and counts == {"upload": 3, "download": 2}
    assert binding.fallbacks == 0


@requires_nvptx
def test_without_a_driver_the_reference_launch_runs(monkeypatch, write, analyze):
    from ppy_compiler.driver.staging import stage_project
    from ppy_runtime.exported import bind_exported

    bundle, _module = _ir(write, analyze, "gpu_prog", PROGRAM)
    payload = stage_project(bundle).artifacts["gpu_prog"]["saxpy"].payload
    monkeypatch.setenv("PPY_NO_CUDA", "1")
    monkeypatch.setattr(runtime, "_driver", None)
    assert runtime.driver() is None
    program = _program()
    binding = bind_exported("saxpy", payload, program["saxpy"])
    assert not binding.routed and "no CUDA driver" in binding.reason
    assert not cuda.compiled(binding.wrapper)
    y = _filled(4, [1.0] * 4)
    cuda.launch(binding.wrapper, 1, 4, 4, 3.0, _filled(4, range(4)), y)
    assert [native.load(native.offset(y, i)) for i in range(4)] == [1.0, 4.0, 7.0, 10.0]


@requires_device
def test_ppy_run_launches_its_kernels_on_the_device(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    entry = tmp_path / "gpu_run.ppy"
    entry.write_text(
        textwrap.dedent(PROGRAM).lstrip("\n")
        + "\n\nprint(cuda.compiled(saxpy), cuda.compiled(block_max))\nmain()\n",
        encoding="utf-8",
    )
    plain = subprocess.run(
        [sys.executable, entry.name], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    on_device = _ppy(tmp_path, "run", entry.name)
    assert plain.returncode == 0, plain.stderr
    assert on_device.returncode == 0, on_device.stderr
    assert plain.stdout == "False False\n90000.0\n100.0 98.0\n90000.0\n"
    assert on_device.stdout == "True True\n90000.0\n100.0 98.0\n90000.0\n"


@requires_device
def test_a_built_artifact_carries_its_kernels(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    entry = tmp_path / "gpu_run.ppy"
    entry.write_text(
        textwrap.dedent(PROGRAM).lstrip("\n") + "\n\nprint(cuda.compiled(saxpy))\nmain()\n",
        encoding="utf-8",
    )
    built = _ppy(tmp_path, "build", "gpu_run.ppy", "-o", "dist")
    assert built.returncode == 0, built.stderr
    manifest = json.loads((tmp_path / "dist" / "ppy-bindings.json").read_text(encoding="utf-8"))
    assert set(manifest["staged"]["gpu_run"]) == {"saxpy", "block_max"}
    assert (tmp_path / "dist" / manifest["staged"]["gpu_run"]["saxpy"]).is_file()
    ran = _ppy(tmp_path, "run", "--prebuilt", "dist/ppy-bindings.json", "gpu_run.ppy")
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == "True\n90000.0\n100.0 98.0\n90000.0\n"
