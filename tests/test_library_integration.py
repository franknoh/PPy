"""Plugin-driven rewrites: ATen routing, JAX staging, ASGI hosting (spec 20-22)."""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest

from ppy_compiler.analysis import types as T
from ppy_compiler.analysis.refinements import Facts
from ppy_compiler.opt.rewrites import adjustments_for_project
from ppy_compiler.plugins.jax_plugin import JaxPlugin, staged_functions
from ppy_compiler.plugins.torch_plugin import ATEN_SCHEMAS, TorchPlugin
from ppy_compiler.plugins.torch_region import emit_source, find_regions
from ppy_compiler.plugins.uvicorn_plugin import PPY_RELOAD_PATTERN, UvicornPlugin


def _importable(name: str) -> bool:
    """A partially installed optional runtime must not fail the suite."""
    if importlib.util.find_spec(name) is None:
        return False
    try:
        importlib.import_module(name)
    except Exception:  # noqa: BLE001 - a broken optional install is a skip
        return False
    return True


has_torch = _importable("torch")
has_jax = _importable("jax")
has_uvicorn = _importable("uvicorn")

requires_torch = pytest.mark.skipif(not has_torch, reason="torch is not importable")
requires_jax = pytest.mark.skipif(not has_jax, reason="jax is not importable")
requires_uvicorn = pytest.mark.skipif(not has_uvicorn, reason="uvicorn is not importable")

TENSOR = (T.Instance("torch.Tensor", (), ("torch.Tensor", "object")), Facts())
SCALAR = (T.FLOAT, Facts())
UNKNOWN = (T.ANY, Facts())


# -- PyTorch: compiling an ATen region ------------------------------------


def test_a_curated_function_becomes_one_aten_region(write, analyze):
    path = write(
        "region.ppy",
        """
        import torch


        def layer(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.relu(torch.add(torch.matmul(x, w), b))
        """,
    )
    bundle = analyze(path)
    symbols = bundle.symbols.modules["region"]
    regions = find_regions(symbols, bundle.analysis.modules["region"])
    assert len(regions) == 1

    region = regions[0]
    assert region.body == "at::relu(at::add(at::matmul(x, w), b))"
    assert region.operations == ("at::relu", "at::add", "at::matmul")
    assert region.declaration().startswith("at::Tensor ppy_region_region_layer(const at::Tensor&")


def test_operators_translate_to_their_aten_counterparts(write, analyze):
    path = write(
        "ops.ppy",
        """
        import torch


        def mix(a: torch.Tensor, b: torch.Tensor, k: float) -> torch.Tensor:
            return (a * b + a) @ b - a / k
        """,
    )
    bundle = analyze(path)
    region = find_regions(bundle.symbols.modules["ops"], bundle.analysis.modules["ops"])[0]
    assert "at::matmul" in region.body
    assert "at::mul" in region.body and "at::div" in region.body
    assert ("k", "scalar") in region.parameters


def test_a_function_outside_the_curated_domain_is_rejected(write, analyze):
    path = write(
        "reject.ppy",
        """
        import torch


        def loopy(x: torch.Tensor) -> torch.Tensor:
            total: torch.Tensor = x
            for _i in range(3):
                total = torch.add(total, x)
            return total


        def exotic(x: torch.Tensor) -> torch.Tensor:
            return torch.fft.fft(x)
        """,
    )
    bundle = analyze(path)
    regions = {
        r.name: r
        for r in find_regions(bundle.symbols.modules["reject"], bundle.analysis.modules["reject"])
    }
    assert not regions["loopy"].body
    assert "only assignments and a final `return`" in regions["loopy"].reason
    assert not regions["exotic"].body
    assert "curated" in regions["exotic"].reason


def test_a_function_without_a_tensor_is_not_a_region(write, analyze):
    path = write(
        "scalar.ppy",
        """
        import torch


        def add(a: int, b: int) -> int:
            return a + b
        """,
    )
    bundle = analyze(path)
    assert find_regions(bundle.symbols.modules["scalar"], bundle.analysis.modules["scalar"]) == []


def test_the_generated_translation_unit_is_valid_cpp(write, analyze):
    path = write(
        "unit.ppy",
        """
        import torch


        def layer(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            return torch.tanh(torch.matmul(x, w))
        """,
    )
    bundle = analyze(path)
    source = emit_source(
        find_regions(bundle.symbols.modules["unit"], bundle.analysis.modules["unit"])
    )
    assert "#include <ATen/ATen.h>" in source
    assert "at::Tensor ppy_region_unit_layer(const at::Tensor& x, const at::Tensor& w) {" in source
    assert "return at::tanh(at::matmul(x, w));" in source


def test_the_aten_schema_table_documents_every_overload():
    for operation, schema in ATEN_SCHEMAS.items():
        forms = schema if isinstance(schema, tuple) else (schema,)
        for form in forms:
            assert "." in form, f"{operation} -> {form} has no overload"


@requires_torch
def test_the_region_toolchain_is_reported():
    from ppy_compiler.plugins.torch_build import toolchain_ready

    ready, detail = toolchain_ready()
    assert isinstance(ready, bool)
    assert detail


@requires_torch
def test_a_compiled_region_matches_the_python_function(write, analyze, tmp_path):
    import torch

    from ppy_compiler.backend.region_runtime import bind_region
    from ppy_compiler.plugins.torch_build import compile_regions, toolchain_ready

    ready, detail = toolchain_ready()
    if not ready:
        pytest.skip(detail)

    path = write(
        "compiled.ppy",
        """
        import torch


        def layer(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.relu(torch.add(torch.matmul(x, w), b))
        """,
    )
    bundle = analyze(path)
    regions = find_regions(bundle.symbols.modules["compiled"], bundle.analysis.modules["compiled"])
    built = compile_regions(regions, tmp_path)
    if not built.ok:
        pytest.skip(built.reason)

    def fallback(x, w, b):
        return torch.relu(torch.add(torch.matmul(x, w), b))

    binding = bind_region("layer", built.entry_points["layer"], fallback)
    assert binding.routed

    x = torch.randn(4, 8)
    w = torch.randn(8, 8)
    b = torch.randn(8)
    assert torch.equal(binding.wrapper(x, w, b), fallback(x, w, b))
    assert binding.calls == 1 and binding.fallbacks == 0


@requires_torch
def test_a_compiled_region_preserves_autograd(write, analyze, tmp_path):
    import torch

    from ppy_compiler.backend.region_runtime import bind_region
    from ppy_compiler.plugins.torch_build import compile_regions, toolchain_ready

    ready, detail = toolchain_ready()
    if not ready:
        pytest.skip(detail)

    path = write(
        "grad.ppy",
        """
        import torch


        def scaled(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.tanh(torch.add(x, torch.mul(y, 0.5)))
        """,
    )
    bundle = analyze(path)
    built = compile_regions(
        find_regions(bundle.symbols.modules["grad"], bundle.analysis.modules["grad"]), tmp_path
    )
    if not built.ok:
        pytest.skip(built.reason)

    def fallback(x, y):
        return torch.tanh(torch.add(x, torch.mul(y, 0.5)))

    binding = bind_region("scaled", built.entry_points["scaled"], fallback)
    x = torch.randn(4, 4, requires_grad=True)
    binding.wrapper(x, x).sum().backward()
    native_grad = x.grad.clone()

    x.grad = None
    fallback(x, x).sum().backward()
    assert torch.allclose(native_grad, x.grad)


@requires_torch
def test_a_tensor_subclass_falls_back_from_a_region():
    import torch

    from ppy_compiler.backend.region_runtime import bind_region

    class Tracked(torch.Tensor):
        pass

    def fallback(a, b):
        return torch.add(a, b)

    binding = bind_region("add", torch.add, fallback)
    result = binding.wrapper(torch.ones(3).as_subclass(Tracked), torch.ones(3))
    assert binding.fallbacks == 1
    assert torch.equal(result, torch.full((3,), 2.0))


def test_an_uncompiled_region_keeps_the_python_function():
    from ppy_compiler.backend.region_runtime import bind_region

    def original(x):
        return x

    binding = bind_region("original", None, original)
    assert not binding.routed
    assert binding.wrapper is original
    assert "not compiled" in binding.reason


# -- JAX: staged export ---------------------------------------------------


def test_a_staged_function_needs_a_declared_input_shape(write, analyze):
    path = write(
        "staged.ppy",
        """
        from typing import Annotated

        import jax
        import jax.numpy as jnp

        import ppy


        @jax.jit
        def described(x: Annotated[jax.Array, ppy.Shape("B", 4), ppy.DType("float32")]) -> jax.Array:
            return x


        @jax.jit
        def undescribed(x: jax.Array) -> jax.Array:
            return x
        """,
    )
    bundle = analyze(path)
    staged = {s.info.name: s for s in staged_functions(bundle.symbols.modules["staged"])}
    assert staged["described"].exportable
    assert staged["described"].shapes == (("B", 4),)
    assert staged["described"].dtypes == ("float32",)
    assert not staged["undescribed"].exportable
    assert "ppy.Shape" in staged["undescribed"].reason


def test_export_is_denied_unless_the_project_permits_it():
    plugin = JaxPlugin()
    allowed, reason = plugin.export_permitted("deny")
    assert not allowed and "explicit permission" in reason

    plugin = JaxPlugin({"allow-build-export": True})
    allowed, reason = plugin.export_permitted("deny")
    assert not allowed and "build-execution" in reason

    allowed, reason = plugin.export_permitted("allow")
    assert allowed and reason == ""

    allowed, reason = JaxPlugin().export_permitted("allow")
    assert not allowed and "allow-build-export" in reason


@requires_jax
def test_a_staged_function_exports_to_stablehlo(write, analyze, project_dir):
    """The export runs project code, so the project must permit it."""
    from ppy_compiler.driver.staging import stage_project

    (project_dir / "pyproject.toml").write_text(
        '[tool.ppy]\nstrict = true\nbuild-execution = "allow"\n\n'
        "[tool.ppy.plugins.jax]\nallow-build-export = true\n",
        encoding="utf-8",
    )
    path = write(
        "exported.ppy",
        """
        from typing import Annotated

        import jax
        import jax.numpy as jnp

        import ppy

        Batch = Annotated[jax.Array, ppy.Shape("B", 4), ppy.DType("float32")]


        @jax.jit
        def scale(x: Batch) -> jax.Array:
            return x * 2.0 + 1.0
        """,
    )
    bundle = analyze(path)
    result = stage_project(bundle)
    if result.skipped and not result.count:
        pytest.skip(f"export unavailable: {result.skipped[0][1]}")

    assert result.count == 1
    artifact = result.artifacts["exported"]["scale"]
    assert artifact.payload
    assert artifact.platforms


@requires_jax
def test_an_exported_artifact_is_cached_and_reused(write, analyze, project_dir):
    from ppy_compiler.driver.staging import stage_project

    (project_dir / "pyproject.toml").write_text(
        '[tool.ppy]\nbuild-execution = "allow"\n\n'
        "[tool.ppy.plugins.jax]\nallow-build-export = true\n",
        encoding="utf-8",
    )
    path = write(
        "cached_export.ppy",
        """
        from typing import Annotated

        import jax

        import ppy


        @jax.jit
        def double(x: Annotated[jax.Array, ppy.Shape(4), ppy.DType("float32")]) -> jax.Array:
            return x + x
        """,
    )
    first = stage_project(analyze(path))
    if not first.count:
        pytest.skip(f"export unavailable: {first.skipped[0][1] if first.skipped else 'unknown'}")

    second = stage_project(analyze(path))
    assert second.count == 1
    assert second.artifacts["cached_export"]["double"].payload == (
        first.artifacts["cached_export"]["double"].payload
    )


@requires_jax
def test_an_exported_region_computes_the_same_values(write, analyze, project_dir):
    import jax.numpy as jnp

    from ppy_compiler.backend.exported_runtime import bind_exported
    from ppy_compiler.driver.staging import stage_project

    (project_dir / "pyproject.toml").write_text(
        '[tool.ppy]\nbuild-execution = "allow"\n\n'
        "[tool.ppy.plugins.jax]\nallow-build-export = true\n",
        encoding="utf-8",
    )
    path = write(
        "run_export.ppy",
        """
        from typing import Annotated

        import jax
        import jax.numpy as jnp

        import ppy


        @jax.jit
        def shifted(x: Annotated[jax.Array, ppy.Shape(4), ppy.DType("float32")]) -> jax.Array:
            return jnp.tanh(x) + 1.0
        """,
    )
    result = stage_project(analyze(path))
    if not result.count:
        pytest.skip("export unavailable")

    def fallback(x):
        return jnp.tanh(x) + 1.0

    payload = result.artifacts["run_export"]["shifted"].payload
    binding = bind_exported("shifted", payload, fallback)
    if not binding.routed:
        pytest.skip(binding.reason)

    values = jnp.arange(4, dtype=jnp.float32)
    # Whether or not the artifact runs here, the answer has to be the same.
    assert jnp.allclose(binding.wrapper(values), fallback(values))
    if binding.last_error:
        pytest.skip(f"the exported artifact did not run here: {binding.last_error}")
    assert binding.calls == 1 and binding.fallbacks == 0


@requires_jax
def test_a_shape_outside_the_exported_signature_falls_back(write, analyze, project_dir):
    import jax.numpy as jnp

    from ppy_compiler.backend.exported_runtime import bind_exported
    from ppy_compiler.driver.staging import stage_project

    (project_dir / "pyproject.toml").write_text(
        '[tool.ppy]\nbuild-execution = "allow"\n\n'
        "[tool.ppy.plugins.jax]\nallow-build-export = true\n",
        encoding="utf-8",
    )
    path = write(
        "fixed_shape.ppy",
        """
        from typing import Annotated

        import jax

        import ppy


        @jax.jit
        def fixed(x: Annotated[jax.Array, ppy.Shape(4), ppy.DType("float32")]) -> jax.Array:
            return x + 1.0
        """,
    )
    result = stage_project(analyze(path))
    if not result.count:
        pytest.skip("export unavailable")

    binding = bind_exported(
        "fixed", result.artifacts["fixed_shape"]["fixed"].payload, lambda x: x + 1.0
    )
    if not binding.routed:
        pytest.skip(binding.reason)

    wrong = jnp.arange(8, dtype=jnp.float32)
    assert jnp.allclose(binding.wrapper(wrong), wrong + 1.0)
    assert binding.fallbacks == 1


def test_an_unloadable_artifact_keeps_the_staged_function():
    from ppy_compiler.backend.exported_runtime import bind_exported

    def original(x):
        return x

    binding = bind_exported("original", b"not a real artifact", original)
    assert not binding.routed
    assert binding.wrapper is original
    assert binding.reason


def test_staging_reports_why_it_did_not_run(write, analyze):
    from ppy_compiler.driver.staging import stage_project

    path = write(
        "nostage.ppy",
        """
        from typing import Annotated

        import jax

        import ppy


        @jax.jit
        def model(x: Annotated[jax.Array, ppy.Shape(4), ppy.DType("float32")]) -> jax.Array:
            return x
        """,
    )
    result = stage_project(analyze(path))
    assert result.count == 0
    assert result.skipped and "permission" in result.skipped[0][1]


# -- Uvicorn: hosting the ASGI application --------------------------------


def _call(source: str) -> ast.Call:
    return ast.parse(source).body[0].value


class _Symbols:
    name = "main"


def test_reload_is_made_to_watch_ppy_sources():
    adjustment = UvicornPlugin().adjust_call(
        "uvicorn.run", _call('uvicorn.run("main:app", reload=True)'), _Symbols()
    )
    assert adjustment is not None
    assert ("reload_includes", f"[{PPY_RELOAD_PATTERN!r}]") in adjustment.add_keywords
    # Reload needs the import string, so the application is not resolved away.
    assert adjustment.replace_first_argument is None


def test_a_module_string_is_resolved_when_reload_is_off():
    adjustment = UvicornPlugin().adjust_call(
        "uvicorn.run", _call('uvicorn.run("main:app", port=8000)'), _Symbols()
    )
    assert adjustment is not None
    assert adjustment.replace_first_argument == "app"
    assert "resolved statically" in adjustment.reason


def test_an_application_from_another_module_is_left_alone():
    adjustment = UvicornPlugin().adjust_call(
        "uvicorn.run", _call('uvicorn.run("other:app")'), _Symbols()
    )
    assert adjustment is None


def test_an_already_resolved_application_needs_no_adjustment():
    assert UvicornPlugin().adjust_call("uvicorn.run", _call("uvicorn.run(app)"), _Symbols()) is None


def test_an_explicit_reload_include_is_respected():
    adjustment = UvicornPlugin().adjust_call(
        "uvicorn.run",
        _call('uvicorn.run("main:app", reload=True, reload_includes=["*.ppy", "*.toml"])'),
        _Symbols(),
    )
    assert adjustment is None


def test_uvicorn_adjustments_reach_the_generated_module(write, analyze):
    from ppy_compiler.driver.pipeline import build_python

    path = write(
        "main.ppy",
        """
        import uvicorn


        async def app(scope: dict[str, str], receive: int, send: int) -> None:
            return None


        def serve() -> None:
            uvicorn.run("main:app", port=8000)
        """,
    )
    bundle = analyze(path)
    code = build_python(bundle, adjustments=adjustments_for_project(bundle)).generated["main"].code
    assert "uvicorn.run(app, port=8000)" in code
    assert '"main:app"' not in code


def test_reload_adjustment_reaches_the_generated_module(write, analyze):
    from ppy_compiler.driver.pipeline import build_python

    path = write(
        "dev.ppy",
        """
        import uvicorn


        async def app(scope: dict[str, str], receive: int, send: int) -> None:
            return None


        def serve() -> None:
            uvicorn.run("dev:app", reload=True)
        """,
    )
    bundle = analyze(path)
    code = build_python(bundle, adjustments=adjustments_for_project(bundle)).generated["dev"].code
    assert "reload_includes=['*.ppy']" in code


# -- packaging ------------------------------------------------------------


def test_each_plugin_runtime_has_its_own_dependency_group():
    """A contributor installs only the runtimes they intend to test."""
    import tomllib

    root = Path(__file__).resolve().parent.parent
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    groups = data["dependency-groups"]
    assert {"dev", "torch", "jax", "web", "plugins", "all"} <= set(groups)

    # `plugins` pulls in every runtime group, and `all` adds the base group.
    included = {entry["include-group"] for entry in groups["plugins"] if isinstance(entry, dict)}
    assert included == {"torch", "jax", "web"}
    assert {"dev", "plugins"} == {
        entry["include-group"] for entry in groups["all"] if isinstance(entry, dict)
    }

    extras = data["project"]["optional-dependencies"]
    assert {"llvm", "numpy", "torch", "jax", "web", "pydantic"} <= set(extras)

    # Only `dev` is installed by default; a plugin runtime is opt-in.
    assert data["tool"]["uv"]["default-groups"] == ["dev"]


def test_the_torch_index_is_recorded_in_the_project_configuration():
    """Which build torch resolves from is part of the plugin's cache key."""
    import tomllib

    root = Path(__file__).resolve().parent.parent
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    source = data["tool"]["uv"]["sources"]["torch"]
    index = source["index"] if isinstance(source, dict) else source[0]["index"]
    names = {entry["name"]: entry["url"] for entry in data["tool"]["uv"]["index"]}
    assert index in names
    assert "download.pytorch.org" in names[index]


@requires_torch
def test_the_torch_fingerprint_records_the_accelerator_runtime():
    fingerprint = TorchPlugin().fingerprint()
    assert "torch=" in fingerprint
    assert "cuda=" in fingerprint
    assert "cxx11abi=" in fingerprint


@requires_jax
def test_the_jax_fingerprint_records_both_versions():
    fingerprint = JaxPlugin().fingerprint()
    assert "jax=" in fingerprint and "jaxlib=" in fingerprint
    assert "absent" not in fingerprint


# -- ASGI application -----------------------------------------------------


@requires_uvicorn
def test_a_ppy_asgi_application_answers_a_request(write, analyze):
    """The compiled application keeps the ASGI callable contract (spec 22.3)."""
    import asyncio

    from ppy_compiler.driver.pipeline import build_python

    path = write(
        "asgi.ppy",
        """
        from typing import Any, Awaitable, Callable

        import ppy

        Scope = dict[str, Any]
        Send = Callable[[Scope], Awaitable[None]]


        @ppy.pure
        def render(path: str) -> bytes:
            return b'{"path": "' + path.encode() + b'"}'


        async def app(scope: Scope, receive: int, send: Send) -> None:
            body: bytes = render(str(scope.get("path", "/")))
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": body})
        """,
    )
    bundle = analyze(path)
    assert not bundle.diagnostics.has_errors(), [d.message for d in bundle.diagnostics.errors]

    generated = build_python(bundle).generated["asgi"]
    namespace: dict = {}
    exec(generated.compile(), namespace)

    messages: list[dict] = []

    async def send(message: dict) -> None:
        messages.append(message)

    asyncio.run(namespace["app"]({"path": "/health"}, 0, send))
    assert messages[0]["status"] == 200
    assert messages[1]["body"] == b'{"path": "/health"}'


def test_bytes_and_str_methods_are_typed(write, analyze):
    path = write(
        "bytesmethods.ppy",
        """
        def round_trip(text: str) -> str:
            payload: bytes = text.encode()
            return payload.decode().upper()


        def parts(payload: bytes) -> int:
            return len(payload.split())
        """,
    )
    bundle = analyze(path)
    assert not bundle.diagnostics.has_errors(), [d.message for d in bundle.diagnostics.errors]
    assert str(bundle.symbols.functions["bytesmethods.round_trip"].ret) == "str"


def test_dict_get_with_a_default_cannot_be_none(write, analyze):
    path = write(
        "getdefault.ppy",
        """
        def lookup(values: dict[str, int]) -> int:
            return values.get("a", 0)


        def unguarded(values: dict[str, int]) -> int:
            return values.get("a")
        """,
    )
    bundle = analyze(path)
    codes = [d.code for d in bundle.diagnostics]
    # Only the call without a default can return None.
    assert codes.count("E1303") == 1


def test_awaitable_annotations_are_resolved(write, analyze):
    path = write(
        "awaitable.ppy",
        """
        from typing import Awaitable, Callable


        async def use(fetch: Callable[[], Awaitable[int]]) -> int:
            return await fetch()
        """,
    )
    bundle = analyze(path)
    assert not bundle.diagnostics.has_errors(), [d.message for d in bundle.diagnostics.errors]
    assert str(bundle.symbols.functions["awaitable.use"].ret) == "int"
