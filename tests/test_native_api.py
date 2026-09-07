"""`ppy.native`, `ppy.ffi`, the math dialect, and C exports."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import toolchain_status
from ppy_compiler.ir import decode, verify

requires_llvm = pytest.mark.skipif(not llvm_available(), reason="llvmlite is not installed")
_usable, _detail = toolchain_status()
requires_toolchain = pytest.mark.skipif(not _usable, reason=f"no native toolchain: {_detail}")

POINTERS = """\
import ppy
from ppy import native


@ppy.native
def fill(p: native.ptr[float], n: int, scale: float) -> float:
    total = 0.0
    for i in range(n):
        q = native.offset(p, i)
        native.store(q, scale * i)
        total += native.load(q)
    return total


def scratch(n: int) -> int:
    buf = native.stack_alloc[int](8)
    acc = 0
    for i in range(8):
        native.store(native.offset(buf, i), i * n)
    for i in range(8):
        acc += native.load(native.offset(buf, i))
    return acc + native.sizeof[float]() + native.alignof[ppy.i8]()


def bytes_of(p: native.ptr[ppy.u8], n: int) -> int:
    total = 0
    for i in range(n):
        total += native.load(native.offset(p, i))
    return total


memory = native.stack_alloc[float](4)
print(fill(memory, 4, 1.5), scratch(3), bytes_of(native.cast[ppy.u8](memory), 8))
"""


def _three_paths(tmp_path: Path, source: str, name: str = "prog.ppy") -> list[str]:
    """Plain, the Python backend, and the IR road natively, compared."""
    (tmp_path / "pyproject.toml").write_text(
        '[tool.ppy]\nstrict = true\n\n[tool.ppy.llvm]\npipeline = "ir"\n', encoding="utf-8"
    )
    (tmp_path / name).write_text(source, encoding="utf-8")
    outputs = []
    for args in ([name], ["-m", "ppy_compiler", name], ["-m", "ppy_compiler", "run", name]):
        done = subprocess.run(
            [sys.executable, *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
            env={k: v for k, v in os.environ.items() if k != "PPY_LOWERING"},
        )
        assert done.returncode == 0, done.stderr
        outputs.append(done.stdout.strip())
    return outputs


def test_the_reference_implementation_is_real_memory():
    import ppy
    from ppy import native

    p = native.stack_alloc[int](3)
    native.store(native.offset(p, 2), 41)
    assert native.load(native.offset(p, 2)) == 41
    assert native.sizeof[int]() == 8 and native.sizeof[ppy.i8]() == 1
    assert native.alignof[float]() == 8
    as_bytes = native.cast[ppy.u8](p)
    assert native.load(native.offset(as_bytes, 16)) == 41
    with pytest.raises(TypeError, match="not an element type"):
        native.stack_alloc[str](1)
    with pytest.raises(TypeError, match=r"reads through a native\.ptr"):
        native.load(3)
    assert native.ptr[int] is not None and native.const_ptr[float] is not None
    assert repr(native) == "ppy.native"


@requires_llvm
def test_pointer_programs_agree_on_every_path(tmp_path: Path):
    outputs = _three_paths(tmp_path, POINTERS)
    assert outputs[0] == outputs[1] == outputs[2]
    assert outputs[0].split()[1] == str(sum(i * 3 for i in range(8)) + 8 + 1)


@requires_llvm
def test_pointer_code_lowers_to_pointer_operations(write, analyze):
    from ppy_compiler.backend.llvm.ir_pipeline import ir_modules

    path = write("ptr.ppy", POINTERS)
    bundle = analyze(path, backend="llvm")
    assert [d.code for d in bundle.diagnostics.sorted() if d.severity.name == "ERROR"] == []
    module = ir_modules(bundle)["ptr"]
    fill = module.functions["ptr_fill"]
    assert str(fill.params[0][1]) == "ptr<f64>"
    names = {op.name for op in fill.operations()}
    assert {"core.ptr_offset", "core.store", "core.load"} <= names
    scratch = module.functions["ptr_scratch"]
    allocas = [op for op in scratch.operations() if op.name == "core.alloca"]
    assert any(op.attributes.get("count") == 8 for op in allocas)
    assert not verify(module)


def test_the_checker_types_the_native_namespace(write, analyze):
    path = write(
        "bad.ppy",
        """
        import ppy
        from ppy import native


        def wrong(p: native.const_ptr[int], q: native.ptr[float]) -> int:
            native.store(p, 1)
            native.store(q, "text")
            native.load(3)
            return native.sizeof[int]() + native.load(p)


        @native.extern("nothing")
        def stub(x, y: int): ...
        """,
    )
    bundle = analyze(path, strict=False)
    codes = [d.code for d in bundle.diagnostics.sorted()]
    assert "E1631" in codes and "E1301" in codes and "E1630" in codes
    assert codes.count("E1633") == 2
    wrong = bundle.symbols.modules["bad"].functions["wrong"]
    assert str(wrong.params[0].type) == "ppy.native.const_ptr[int]"
    from ppy_compiler.analysis.effects import Effect

    effects = bundle.analysis.modules["bad"].functions["bad.wrong"].effects
    assert Effect.WRITE_MEMORY in effects and Effect.READ_MEMORY in effects


LIBM = """\
import math

from ppy import ffi

libm = ffi.library("m")


@ffi.bind(libm, symbol="sin", pure=True)
def c_sin(x: float) -> float: ...


@ffi.bind(libm, symbol="hypot", pure=True)
def c_hypot(x: float, y: float) -> float: ...


def wave(n: int) -> float:
    total = 0.0
    for i in range(n):
        total += c_sin(i * 0.1) + c_hypot(i * 1.0, 3.0) - math.sin(i * 0.1)
    return total


print(f"{wave(50):.6f}")
"""


@requires_llvm
def test_a_c_binding_is_called_directly_in_native_code_and_through_ctypes_otherwise(tmp_path):
    outputs = _three_paths(tmp_path, LIBM)
    assert outputs[0] == outputs[1] == outputs[2]
    emitted = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "emit", "ir", "prog.ppy"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert emitted.returncode == 0, emitted.stderr
    module = decode(emitted.stdout)
    wave = module.functions["prog_wave"]
    externs = [op for op in wave.operations() if op.name == "core.call_extern"]
    assert {op.attributes["callee"] for op in externs} == {"sin", "hypot"}
    assert module.attributes.get("ppy.libraries") == ("m",)
    assert any(op.name == "math.sin" for op in wave.operations())
    assert module.dialects.get("math") == 1


@requires_llvm
def test_math_functions_are_dialect_operations_that_fold(write, analyze):
    from ppy_compiler.backend.llvm.ir_pipeline import ir_modules

    path = write(
        "m.ppy",
        """
        import math


        def area(r: float) -> float:
            return math.sqrt(16.0) * r * r + abs(-2.0) + math.floor(math.floor(r))


        def power(x: float) -> float:
            return x ** 2.5 + math.pow(x, 0.5)
        """,
    )
    bundle = analyze(path, backend="llvm")
    module = ir_modules(bundle)["m"]
    area = [op.name for op in module.functions["m_area"].operations()]
    assert "math.sqrt" not in area and "math.abs" not in area, "constants fold"
    assert area.count("math.floor") == 1, "floor of floor is floor"
    power = [op.name for op in module.functions["m_power"].operations()]
    assert power.count("math.pow") == 2
    from ppy_compiler.backend.llvm.from_ir import emit_module

    text = emit_module(module)
    assert "llvm.pow.f64" in text and "llvm.floor.f64" in text


@requires_toolchain
def test_an_export_is_a_public_c_symbol_with_a_header(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\nstrict = true\n", encoding="utf-8")
    (tmp_path / "lib.ppy").write_text(
        """\
from ppy import Buffer, native


@native.export(name="ppy_dot")
def dot(a: Buffer[float], b: Buffer[float], n: int) -> float:
    total = 0.0
    for i in range(n):
        total += a[i] * b[i]
    return total


@native.export()
def clamp(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else (hi if x > hi else x)
""",
        encoding="utf-8",
    )
    built = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "build", "lib.ppy", "-o", "dist"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env={"PPY_LOWERING": "ir", **os.environ},
    )
    assert built.returncode == 0, built.stderr
    header = (tmp_path / "dist" / f"{tmp_path.name}.h").read_text(encoding="utf-8")
    assert (
        "double ppy_dot(double *a, int64_t a_len, double *b, int64_t b_len, int64_t n);" in header
    )
    assert "int64_t clamp(int64_t x, int64_t lo, int64_t hi);" in header
    assert 'extern "C"' in header
    library = ctypes.CDLL(str(tmp_path / "dist" / f"libppy_{tmp_path.name}.so"))
    clamp = library.clamp
    clamp.argtypes = [ctypes.c_int64] * 3
    clamp.restype = ctypes.c_int64
    assert clamp(5, 0, 3) == 3 and clamp(-2, 0, 3) == 0
    dot = library.ppy_dot
    xs = (ctypes.c_double * 3)(1.0, 2.0, 3.0)
    ys = (ctypes.c_double * 3)(4.0, 5.0, 6.0)
    dot.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int64] * 2 + [ctypes.c_int64]
    dot.restype = ctypes.c_double
    assert dot(xs, 3, ys, 3, 3) == 32.0
    import json

    manifest = json.loads((tmp_path / "dist" / "ppy-bindings.json").read_text(encoding="utf-8"))
    assert set(manifest["exports"]) == {"ppy_dot", "clamp"}
