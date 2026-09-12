"""`ppy.xla`: StableHLO from the IR, compiled and run by XLA through PJRT."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from ppy_compiler.backend.stablehlo import StableHloError, emit_module, supports
from ppy_compiler.ir import F64, BufferType, Builder, IRModule
from ppy_compiler.ir.dialects import core, tensor
from ppy_compiler.ir.dialects import math as math_dialect
from ppy_compiler.ir.model import Successor
from ppy_compiler.ir.passes import PassContext, PassManager
from ppy_compiler.ir.transforms import FuseTensor, promote_slots

has_xla = (
    importlib.util.find_spec("jax") is not None and importlib.util.find_spec("jaxlib") is not None
)
requires_xla = pytest.mark.skipif(not has_xla, reason="jax and jaxlib are not installed")


def _scalar_module() -> IRModule:
    module = IRModule("scalars")
    module.require("math", 1)
    f = module.add_function("f", [("x", F64), ("y", F64)], [F64])
    b = Builder(f.add_entry_block())
    x, y = f.entry.arguments
    slot = core.alloca(b, F64, name="x.addr")
    core.store(b, x, slot)
    loaded = core.load(b, slot)
    product = core.mul(b, math_dialect.call(b, "sin", loaded), y)
    bigger = core.cmp(b, "gt", product, core.const(b, 0.0, F64))
    total = core.add(b, product, core.select(b, bigger, x, core.neg(b, x)))
    core.ret(b, core.div(b, total, core.const(b, 2.0, F64)))
    return module


def _tensor_module(fused: bool = True) -> IRModule:
    module = IRModule("tensors")
    module.require("tensor", 1)
    a_type = tensor.tensor_type(F64, (4, 3))
    w_type = tensor.tensor_type(F64, (3, 2))
    f = module.add_function(
        "loss",
        [("a", a_type), ("w", w_type), ("s", F64)],
        [tensor.tensor_type(F64, ()), tensor.tensor_type(F64, (4, 2))],
    )
    b = Builder(f.add_entry_block())
    a, w, s = f.entry.arguments
    p = tensor.matmul(b, a, w)
    scaled = tensor.elementwise(
        b, "mul", tensor.unary(b, "exp", p), tensor.fill(b, s, tensor.tensor_type(F64, (4, 2)))
    )
    row_sums = tensor.reduce(b, scaled, (1,), "add", keepdims=True)
    centered = tensor.elementwise(b, "sub", scaled, tensor.broadcast(b, row_sums, (4, 2)))
    flat = tensor.reshape(b, tensor.transpose(b, centered, (1, 0)), (8,))
    total = tensor.reduce(b, tensor.elementwise(b, "mul", flat, flat), (0,), "add")
    core.ret(b, total, centered)
    if fused:
        PassManager(PassContext()).add(FuseTensor()).run(module)
    return module


def test_scalar_functions_become_rank_zero_tensors():
    module = _scalar_module()
    function = module.functions["f"]
    assert "core.alloca" in supports(function), "slots are memory until they are promoted"
    promote_slots(function)
    assert supports(function) is None
    text = emit_module(module)
    assert "func.func public @f(%x: tensor<f64>, %y: tensor<f64>) -> (tensor<f64>)" in text
    assert (
        "stablehlo.sine" in text and "stablehlo.compare GT" in text and "stablehlo.select" in text
    )
    assert "stablehlo.constant dense<2.00000000000000000e+00> : tensor<f64>" in text
    assert text.count("return ") == 1


def test_tensor_operations_map_onto_stablehlo():
    text = emit_module(_tensor_module(fused=False), ("loss",))
    assert '"stablehlo.dot_general"' in text and "lhs_contracting_dimensions = [1]" in text
    assert "stablehlo.exponential" in text and "stablehlo.broadcast_in_dim" in text
    assert "stablehlo.reduce(" in text and "across dimensions = [1]" in text
    assert "stablehlo.transpose" in text and "dims = [1, 0]" in text
    assert "stablehlo.reshape" in text
    assert "tensor<4x2xf64>" in text and "tensor<f64>" in text


def test_what_xla_cannot_take_is_named():
    module = IRModule("refused")
    module.require("tensor", 1)
    branchy = module.add_function("branchy", [("x", F64)], [F64])
    entry = branchy.add_entry_block()
    other = branchy.body.add_block("other")
    core.br(Builder(entry), Successor(other))
    core.ret(Builder(other), entry.arguments[0])
    assert "control flow" in supports(branchy)
    reader = module.add_function("reader", [("a", BufferType(F64))], [F64])
    b = Builder(reader.add_entry_block())
    core.ret(b, core.const(b, 1.0, F64))
    assert "buffer<f64> is not a value XLA holds" in supports(reader)
    guarded = module.add_function("guarded", [("x", F64)], [F64])
    b = Builder(guarded.add_entry_block())
    core.guard(b, core.cmp(b, "gt", guarded.entry.arguments[0], core.const(b, 0.0, F64)), "bounds")
    core.ret(b, guarded.entry.arguments[0])
    assert "guard" in supports(guarded)
    with pytest.raises(StableHloError, match="control flow"):
        emit_module(module, ("branchy",))


@requires_xla
def test_stablehlo_runs_on_the_device_and_agrees_with_numpy():
    from ppy_runtime.xla import pjrt

    client = pjrt.client()
    assert client.devices(), "a PJRT client has a device"
    module = _tensor_module()
    executable = client.compile(emit_module(module, ("loss",), entry="loss"))
    rng = np.random.default_rng(3)
    a = rng.normal(size=(4, 3))
    w = rng.normal(size=(3, 2))
    total, centered = client.execute(executable, [a, w, np.array(0.5)])
    scaled = np.exp(a @ w) * 0.5
    expected_centered = scaled - scaled.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(centered, expected_centered)
    flat = expected_centered.T.reshape(8)
    assert float(total) == pytest.approx(float((flat * flat).sum()))
    assert client.compile(emit_module(module, ("loss",), entry="loss")) is executable, (
        "cached by its digest"
    )


@requires_xla
def test_the_bridge_compiles_for_one_device_of_a_machine_that_has_several(tmp_path: Path):
    """Two devices on the platform: the executable takes one buffer per argument, not one
    shard per device, which is what a two-GPU machine refused before."""
    program = textwrap.dedent(
        """
        import numpy as np
        from ppy_compiler.backend.stablehlo import emit_module
        from ppy_runtime.xla import pjrt
        from test_xla import _tensor_module

        client = pjrt.client()
        assert len(client.devices()) == 2, client.devices()
        executable = client.compile(emit_module(_tensor_module(), ("loss",), entry="loss"))
        rng = np.random.default_rng(3)
        a, w = rng.normal(size=(4, 3)), rng.normal(size=(3, 2))
        total, centered = client.execute(executable, [a, w, np.array(0.5)])
        scaled = np.exp(a @ w) * 0.5
        np.testing.assert_allclose(centered, scaled - scaled.sum(axis=1, keepdims=True))
        print("ran on", client.device(), "of", len(client.devices()))
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
            "XDG_CACHE_HOME": str(tmp_path),
        },
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ran on cpu:0 of 2", done.stdout


def _fake_jax(monkeypatch, backend: str | Exception):
    """A `jax` whose `default_backend()` answers `backend`, or raises it."""
    import types

    module = types.ModuleType("jax")

    def default_backend():
        if isinstance(backend, Exception):
            raise backend
        return backend

    module.default_backend = default_backend  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "jax", module)
    from ppy_runtime.xla import pjrt

    monkeypatch.setattr(pjrt, "available", lambda: True)
    monkeypatch.delenv("PPY_XLA_PLATFORM", raising=False)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    return pjrt


def test_the_bridge_follows_a_jax_that_came_up_on_the_cpu_with_nothing_else(monkeypatch):
    pjrt = _fake_jax(monkeypatch, "cpu")
    monkeypatch.setattr(pjrt, "accelerator_plugins", list)
    assert pjrt.default_platform() == "cpu"
    _fake_jax(monkeypatch, "gpu")
    assert pjrt.default_platform() == "gpu"


def test_a_jax_that_cannot_initialize_is_not_read_as_the_cpu(monkeypatch):
    pjrt = _fake_jax(monkeypatch, RuntimeError("Unable to initialize backend 'cuda': ..."))
    with pytest.raises(RuntimeError, match="Unable to initialize backend"):
        pjrt.default_platform()


def test_a_plugin_that_failed_to_initialize_is_not_a_machine_without_a_gpu(monkeypatch):
    pjrt = _fake_jax(monkeypatch, "cpu")
    monkeypatch.setattr(pjrt, "accelerator_plugins", lambda: ["jax-cuda12-plugin 0.11.1"])
    with pytest.raises(RuntimeError, match=r"jax-cuda12-plugin 0\.11\.1"):
        pjrt.default_platform()
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    assert pjrt.default_platform() == "cpu", "the CPU asked for is the CPU"
    monkeypatch.setenv("PPY_XLA_PLATFORM", "gpu")
    assert pjrt.default_platform() == "gpu", "the bridge's own override wins"


def test_without_jax_the_platform_is_the_cpu_and_nothing_is_imported(monkeypatch):
    from ppy_runtime.xla import pjrt

    monkeypatch.setattr(pjrt, "available", lambda: False)
    monkeypatch.delenv("PPY_XLA_PLATFORM", raising=False)
    monkeypatch.setitem(sys.modules, "jax", None)
    assert pjrt.default_platform() == "cpu"


@requires_xla
def test_a_staged_scalar_payload_runs_through_the_bridge():
    import json

    from ppy_runtime.xla import runtime_call

    module = _scalar_module()
    promote_slots(module.functions["f"])
    payload = json.dumps(
        {
            "kind": "ppy.xla",
            "function": "scalars.f",
            "stablehlo": emit_module(module, entry="f"),
            "params": ["float", "float"],
            "results": ["float"],
        }
    ).encode("utf-8")
    call = runtime_call(payload)
    import math

    def f(x: float, y: float) -> float:
        product = math.sin(x) * y
        return (product + (x if product > 0.0 else -x)) / 2.0

    for x, y in ((0.3, 2.0), (-1.25, 0.5), (7.0, -3.0)):
        assert call(x, y) == pytest.approx(f(x, y), rel=1e-12)
        assert isinstance(call(x, y), float)


PROGRAM = """
    import math

    import ppy
    from ppy import xla


    @xla.jit
    def f(x: float, y: float) -> float:
        product = math.sin(x) * y
        return (product + (x if product > 0.0 else -x)) / 2.0


    @xla.jit
    def branchy(x: float) -> float:
        if x > 0.0:
            return x
        return -x
    """


def test_emit_stablehlo_and_staging_report_what_xla_takes(write, analyze, tmp_path):
    from ppy_compiler.driver.staging import stage_project

    path = write("xla_prog.ppy", PROGRAM)
    bundle = analyze(path, backend="llvm")
    assert not bundle.diagnostics.has_errors(), [d.message for d in bundle.diagnostics.errors]
    staged = stage_project(bundle)
    assert "f" in staged.names("xla_prog")
    assert any("branchy" in q and "control flow" in reason for q, reason in staged.skipped), (
        staged.skipped
    )
    payload = staged.artifacts["xla_prog"]["f"].payload
    assert b'"kind": "ppy.xla"' in payload and b"stablehlo.sine" in payload
    emitted = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", "emit", "stablehlo", path.name],
        cwd=path.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert emitted.returncode == 0, emitted.stderr
    assert "func.func public @xla_prog_f" in emitted.stdout and "stablehlo.divide" in emitted.stdout


@requires_xla
def test_the_program_runs_its_jitted_function_on_xla(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.ppy]\nstrict = true\n\n[tool.ppy.llvm]\npipeline = "ir"\n', encoding="utf-8"
    )
    entry = tmp_path / "xla_run.ppy"
    entry.write_text(
        textwrap.dedent(PROGRAM).lstrip("\n")
        + textwrap.dedent(
            """

            def main() -> None:
                print("device" if xla.devices() else "none", round(f(0.3, 2.0), 12))
                print(round(f(-1.25, 0.5), 12), repr(branchy(-2.0)))


            main()
            """
        ),
        encoding="utf-8",
    )
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
    assert native.stdout == plain.stdout
    assert "device" in plain.stdout
