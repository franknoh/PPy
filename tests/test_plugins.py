"""Library plugins: NumPy, PyTorch, JAX, Uvicorn, Pydantic (spec 18-23)."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ppy_compiler.analysis import types as T
from ppy_compiler.analysis.effects import Effect
from ppy_compiler.analysis.refinements import Facts
from ppy_compiler.driver.config import Config, PluginConfig
from ppy_compiler.plugins.base import Lowering, PluginRegistry
from ppy_compiler.plugins.jax_plugin import JaxPlugin
from ppy_compiler.plugins.numpy_plugin import NumPyPlugin
from ppy_compiler.plugins.pydantic_plugin import PydanticPlugin
from ppy_compiler.plugins.registry import load_plugins
from ppy_compiler.plugins.torch_plugin import TorchPlugin
from ppy_compiler.plugins.uvicorn_plugin import UvicornPlugin, resolve_app

has_numpy = importlib.util.find_spec("numpy") is not None
requires_numpy = pytest.mark.skipif(not has_numpy, reason="numpy is not installed")

ARRAY = (T.Instance("numpy.ndarray", (), ("numpy.ndarray", "object")), Facts())
FLOAT = (T.FLOAT, Facts())


def test_all_plugins_load_and_fingerprint():
    registry = load_plugins(Config())
    assert {p.name for p in registry} == {"numpy", "torch", "jax", "uvicorn", "pydantic"}
    fingerprints = registry.fingerprints()
    assert len(set(fingerprints)) == 5
    assert all(":" in f for f in fingerprints)


def test_disabled_plugins_are_not_loaded():
    config = Config()
    config.plugins["numpy"] = PluginConfig(enabled=False)
    assert "numpy" not in {p.name for p in load_plugins(config)}


def test_registry_resolves_modules_and_qualnames():
    registry = PluginRegistry()
    registry.register(NumPyPlugin())
    assert registry.for_module("numpy") is not None
    assert registry.for_module("numpy.linalg") is not None
    assert registry.for_qualname("numpy.sqrt") is not None
    assert registry.for_qualname("os.path.join") is None


# -- NumPy ----------------------------------------------------------------


def test_numpy_fingerprint_tracks_the_installed_build():
    plugin = NumPyPlugin()
    assert plugin.fingerprint().startswith("v1:numpy=")
    assert NumPyPlugin({"internal-api": True}).fingerprint() != plugin.fingerprint()


def test_numpy_elementwise_fuses_into_one_loop():
    result = NumPyPlugin().call("numpy.sin", [ARRAY], {})
    assert result is not None
    assert result.lowering is Lowering.INTRINSIC
    assert "fused" in result.reason
    assert Effect.ALLOC in result.effects


def test_numpy_guards_cover_the_documented_fast_path_domain():
    result = NumPyPlugin().call("numpy.add", [ARRAY, ARRAY], {})
    guards = " ".join(result.guards)
    assert "__array_ufunc__" in guards
    assert "dtype" in guards
    assert "byte order" in guards
    assert "identical shapes" in guards
    assert "floating-point error state" in guards


def test_numpy_claims_intrinsic_only_where_a_kernel_exists():
    """A plugin must not report a lowering the backend cannot perform."""
    from ppy_compiler.plugins.numpy_plugin import FUSIBLE

    plugin = NumPyPlugin()
    fused = plugin.call("numpy.sin", [ARRAY], {})
    unfused = plugin.call("numpy.tanh", [ARRAY], {})
    assert "sin" in FUSIBLE and "tanh" not in FUSIBLE
    assert fused.lowering is Lowering.INTRINSIC
    assert unfused.lowering is Lowering.DIRECT_NATIVE_CALL
    assert "no generated kernel" in unfused.reason


def test_numpy_falls_back_for_unsupported_dtype():
    result = NumPyPlugin().call(
        "numpy.add",
        [ARRAY, ARRAY],
        {"dtype": (T.STR, Facts(constant="complex128", has_constant=True))},
    )
    assert result.lowering is Lowering.PYTHON_FALLBACK
    assert "complex128" in result.reason


def test_numpy_falls_back_for_unknown_operand_types():
    result = NumPyPlugin().call("numpy.add", [(T.ANY, Facts())], {})
    assert result.lowering is Lowering.PYTHON_FALLBACK


def test_numpy_falls_back_for_unsupported_keywords():
    result = NumPyPlugin().call("numpy.sum", [ARRAY], {"where": (T.ANY, Facts())})
    assert result.lowering is Lowering.PYTHON_FALLBACK
    assert "where" in result.reason


def test_numpy_out_keyword_keeps_overlap_safe_semantics():
    result = NumPyPlugin().call("numpy.add", [ARRAY, ARRAY], {"out": ARRAY})
    assert result.lowering is Lowering.DIRECT_NATIVE_CALL
    assert Effect.WRITE_OBJECT in result.effects


def test_numpy_reduction_keeps_strict_float_order():
    result = NumPyPlugin().call("numpy.sum", [ARRAY], {})
    assert result.lowering is Lowering.INTRINSIC
    assert "strict floating-point order" in result.reason


def test_numpy_reduction_with_axis_returns_an_array():
    scalar = NumPyPlugin().call("numpy.sum", [ARRAY], {})
    axial = NumPyPlugin().call("numpy.sum", [ARRAY], {"axis": (T.INT, Facts())})
    assert scalar.type == T.FLOAT
    assert isinstance(axial.type, T.Instance) and axial.type.name == "numpy.ndarray"


def test_numpy_linalg_uses_blas_without_forcing_a_copy():
    result = NumPyPlugin().call("numpy.matmul", [ARRAY, ARRAY], {})
    assert result.lowering is Lowering.DIRECT_NATIVE_CALL
    assert "BLAS" in result.reason


def test_numpy_fusion_can_be_disabled():
    result = NumPyPlugin({"fusion": False}).call("numpy.sin", [ARRAY], {})
    assert result.lowering is Lowering.PYTHON_FALLBACK


def test_numpy_creation_records_shape(write, analyze):
    result = NumPyPlugin().call("numpy.zeros", [(T.INT, Facts(constant=8, has_constant=True))], {})
    assert result.facts.shape == (8,)
    assert result.facts.contiguous


def test_numpy_expressions_type_check(write, analyze):
    path = write(
        "np_use.ppy",
        """
        import numpy as np


        def normalize(x: np.ndarray) -> np.ndarray:
            scale: float = np.sqrt(np.sum(x * x))
            return x / scale


        def dims(x: np.ndarray) -> int:
            return x.ndim
        """,
    )
    bundle = analyze(path)
    assert not bundle.diagnostics.has_errors(), [d.message for d in bundle.diagnostics.errors]


def test_numpy_lowering_decisions_are_recorded(write, analyze):
    path = write(
        "np_low.ppy",
        """
        import numpy as np


        def f(x: np.ndarray) -> np.ndarray:
            return np.sin(x)
        """,
    )
    bundle = analyze(path)
    notes = list(bundle.analysis.modules["np_low"].lowerings.values())
    assert any(n.qualname == "numpy.sin" and n.lowering == "Intrinsic" for n in notes)


@requires_numpy
def test_numpy_program_matches_plain_cpython(tmp_path: Path):
    entry = tmp_path / "np_run.ppy"
    entry.write_text(
        textwrap.dedent(
            """
            import ppy
            import numpy as np


            def normalize(x: np.ndarray) -> np.ndarray:
                scale: float = np.sqrt(np.sum(x * x))
                return x / scale


            values = np.arange(8, dtype=np.float64)
            result = normalize(values)
            print(round(float(result[1]), 6), int(values.ndim))
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    plain = subprocess.run(
        [sys.executable, entry.name], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    optimized = subprocess.run(
        [sys.executable, "-m", "ppy_compiler", entry.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert plain.returncode == 0, plain.stderr
    assert optimized.returncode == 0, optimized.stderr
    assert optimized.stdout == plain.stdout


# -- PyTorch --------------------------------------------------------------


def test_torch_curated_ops_name_their_aten_counterpart():
    """The verdict says the operation has a C++ counterpart, and which one."""
    plugin = TorchPlugin()
    tensor = (T.Instance("torch.Tensor", (), ("torch.Tensor", "object")), Facts())
    result = plugin.call("torch.matmul", [tensor, tensor], {})
    assert result.lowering is Lowering.DIRECT_NATIVE_CALL
    assert "ATen C++ counterpart" in result.reason
    assert "autograd" in result.reason
    assert "aten::matmul.default" in result.reason


def test_the_named_overload_follows_the_operand_types():
    plugin = TorchPlugin()
    tensor = (T.Instance("torch.Tensor", (), ("torch.Tensor", "object")), Facts())
    scalar = (T.FLOAT, Facts())
    assert plugin.schema_for("add", [tensor[0], tensor[0]]) == "aten::add.Tensor"
    assert plugin.schema_for("add", [tensor[0], scalar[0]]) == "aten::add.Scalar"
    assert plugin.schema_for("relu", [tensor[0]]) == "aten::relu.default"
    assert plugin.schema_for("not_an_operation", [tensor[0]]) is None


def test_torch_guards_cover_override_mechanisms():
    tensor = (T.Instance("torch.Tensor", (), ("torch.Tensor", "object")), Facts())
    guards = " ".join(TorchPlugin().call("torch.add", [tensor, tensor], {}).guards)
    assert "__torch_function__" in guards
    assert "subclass" in guards
    assert "device" in guards


def test_torch_random_ops_carry_a_random_effect():
    result = TorchPlugin().call("torch.randn", [(T.INT, Facts())], {})
    assert Effect.RANDOM in result.effects


def test_torch_unknown_operator_is_not_claimed():
    assert TorchPlugin().call("torch.some_exotic_op", [], {}) is None


def test_torch_fingerprint_includes_abi_and_accelerator():
    fingerprint = TorchPlugin().fingerprint()
    assert "cxx11abi=" in fingerprint and "cuda=" in fingerprint


# -- JAX ------------------------------------------------------------------


def test_jax_eager_calls_stay_on_the_python_path():
    result = JaxPlugin().call("jax.numpy.sin", [(T.ANY, Facts())], {})
    assert result.lowering is Lowering.PYTHON_FALLBACK


def test_jax_staged_export_requires_explicit_permission():
    denied = JaxPlugin().staged_region(["jax.jit"])
    assert denied.lowering is Lowering.PYTHON_FALLBACK
    assert "explicit permission" in denied.reason

    allowed = JaxPlugin({"allow-build-export": True}).staged_region(["jax.jit"])
    assert allowed.lowering is Lowering.GRAPH_REGION
    assert "StableHLO" in allowed.reason and "PJRT" in allowed.reason


def test_jax_ignores_unstaged_functions():
    assert JaxPlugin().staged_region(["functools.cache"]) is None


def test_jax_fingerprint_covers_jaxlib():
    assert "jaxlib=" in JaxPlugin().fingerprint()


# -- Uvicorn --------------------------------------------------------------


def test_uvicorn_run_is_recognized_as_a_host_call():
    result = UvicornPlugin().call("uvicorn.run", [(T.ANY, Facts())], {})
    assert result.lowering is Lowering.PYTHON_FALLBACK
    assert Effect.IO in result.effects


def test_uvicorn_app_is_resolved_statically():
    call = ast.parse('uvicorn.run(app, host="0.0.0.0", port=8000)').body[0].value
    assert resolve_app(call) == "app"

    module_string = ast.parse('uvicorn.run("main:app")').body[0].value
    assert resolve_app(module_string) == "main:app"

    keyword_form = ast.parse("uvicorn.run(app=application)").body[0].value
    assert resolve_app(keyword_form) == "application"


def test_uvicorn_reload_watches_ppy_files():
    assert "*.ppy" in UvicornPlugin().reload_dirs_note()


# -- Pydantic -------------------------------------------------------------


def test_pydantic_field_constraints_become_refinements(write, analyze):
    path = write(
        "models.ppy",
        """
        from typing import Annotated

        import pydantic
        from pydantic import BaseModel, Field


        class Pixel(BaseModel):
            r: Annotated[int, Field(ge=0, le=255)]
        """,
    )
    bundle = analyze(path)
    info = bundle.symbols.classes["models.Pixel"]
    facts = info.field_facts["r"]
    assert facts.int_range.low == 0 and facts.int_range.high == 255
    assert info.is_pydantic


def test_pydantic_constructor_input_differs_from_validated_output(write, analyze):
    path = write(
        "coerce.ppy",
        """
        from pydantic import BaseModel


        class User(BaseModel):
            id: int


        def make() -> int:
            user = User(id="123")
            return user.id
        """,
    )
    bundle = analyze(path)
    assert not bundle.diagnostics.has_errors(), [d.message for d in bundle.diagnostics.errors]


def test_pydantic_validation_stays_a_python_call():
    result = PydanticPlugin().call("pydantic.BaseModel.model_validate", [], {})
    assert result.lowering is Lowering.PYTHON_FALLBACK
    assert "pydantic-core" in result.reason


def test_pydantic_schema_execution_is_denied_by_default():
    assert "schema-exec=deny" in PydanticPlugin().fingerprint()


def test_pydantic_validator_effects_are_summarized():
    assert "ValueError" in str(PydanticPlugin().validator_effects())


# -- Annotated protocol ---------------------------------------------------


def test_annotated_refinements_are_interpreted(write, analyze):
    path = write(
        "refine.ppy",
        """
        from typing import Annotated

        import ppy


        def f(
            x: Annotated[int, ppy.Range(0, 255)],
            xs: Annotated[list[float], ppy.Length(16), ppy.NoAlias()],
        ) -> int:
            return x
        """,
    )
    bundle = analyze(path)
    assert not bundle.diagnostics.has_errors()
    info = bundle.symbols.functions["refine.f"]
    assert info.params[0].facts.int_range.high == 255
    assert info.params[1].facts.length == 16
    assert info.params[1].facts.no_alias


def test_unknown_annotated_metadata_is_preserved_and_warned(write, analyze):
    path = write(
        "unknown_meta.ppy",
        """
        from typing import Annotated


        def marker(value: int) -> int:
            return value


        def f(x: Annotated[int, marker(1)]) -> int:
            return x
        """,
    )
    bundle = analyze(path)
    assert "W2003" in [d.code for d in bundle.diagnostics]
    assert not bundle.diagnostics.has_errors()


def test_tolist_follows_the_declared_dtype():
    """Without the receiver's refinements this could only guess an element type."""
    for plugin, owner in ((JaxPlugin(), "jax.Array"), (TorchPlugin(), "torch.Tensor")):
        signature, _ = plugin.instance_attribute(owner, "tolist", Facts(dtype="int32"))
        assert signature.ret == T.list_of(T.INT), owner
        signature, _ = plugin.instance_attribute(owner, "tolist", Facts(dtype="float32"))
        assert signature.ret == T.list_of(T.FLOAT), owner
        signature, _ = plugin.instance_attribute(owner, "tolist", Facts())
        assert signature.ret == T.list_of(T.FLOAT), owner


def test_a_field_constraint_becomes_a_refinement(tmp_path: Path):
    """Both ways of writing the bound state the same thing (spec 23.3)."""
    pytest.importorskip("pydantic")
    from ppy_compiler.driver.pipeline import analyze_paths, open_project

    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    path = tmp_path / "models.ppy"
    path.write_text(
        textwrap.dedent(
            """
            from typing import Annotated

            from pydantic import BaseModel, Field


            class ByDefault(BaseModel):
                count: int = Field(ge=0, le=100)
                offset: int = Field(gt=-5, lt=5)
                free: int = 0


            class ByAnnotation(BaseModel):
                count: Annotated[int, Field(ge=0, le=100)]
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    bundle = analyze_paths(open_project(path), [path], backend="python")
    classes = bundle.symbols.classes
    default = classes["models.ByDefault"].field_facts
    annotated = classes["models.ByAnnotation"].field_facts

    assert default["count"].int_range == annotated["count"].int_range
    assert (default["count"].int_range.low, default["count"].int_range.high) == (0, 100)
    # `gt`/`lt` are exclusive, so the range is the one inside them.
    assert (default["offset"].int_range.low, default["offset"].int_range.high) == (-4, 4)
    assert default["free"].int_range is None


def test_the_uvicorn_plugin_vouches_for_route_decorators():
    from ppy_compiler.plugins.uvicorn_plugin import UvicornPlugin

    plugin = UvicornPlugin()
    semantics = plugin.decorator_semantics("fastapi.FastAPI.get")
    assert semantics is not None
    assert semantics.reads_annotations
    assert not semantics.pure_at_definition
    assert plugin.decorator_semantics("fastapi.APIRouter.post") is not None
    assert plugin.decorator_semantics("fastapi.FastAPI.nonsense") is None


def test_the_uvicorn_plugin_covers_the_fastapi_surface(tmp_path: Path):
    """One plugin, one serving stack: a FastAPI service checks under strict.

    Route handlers stay under their unvouched decorator; what the plugin
    supplies is a signature for every call the module itself makes.
    """
    pytest.importorskip("fastapi")
    from ppy_compiler.driver.pipeline import analyze_paths, open_project

    (tmp_path / "pyproject.toml").write_text(
        "[tool.ppy]\nstrict = true\n\n[tool.ppy.plugins.uvicorn]\nenabled = true\n",
        encoding="utf-8",
    )
    path = tmp_path / "svc.ppy"
    path.write_text(
        textwrap.dedent(
            """
            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            app = FastAPI()


            @app.get("/")
            def read_root():
                return {"ok": True}


            def main() -> None:
                client = TestClient(app)
                response = client.get("/")
                print(response.status_code, response.json())


            main()
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    bundle = analyze_paths(open_project(path), [path], backend="python")
    errors = [d for d in bundle.diagnostics if d.severity.name == "ERROR"]
    assert not errors, [f"{d.code}: {d.message}" for d in errors]


def test_the_jax_plugin_covers_the_flax_surface(tmp_path: Path):
    """A Flax module class resolves `init`/`apply` through its external MRO,
    and layers, activations, and optax calls all have strict-mode signatures."""
    pytest.importorskip("flax")
    from ppy_compiler.driver.pipeline import analyze_paths, open_project

    (tmp_path / "pyproject.toml").write_text(
        "[tool.ppy]\nstrict = true\n\n[tool.ppy.plugins.jax]\nenabled = true\n",
        encoding="utf-8",
    )
    path = tmp_path / "net.ppy"
    path.write_text(
        textwrap.dedent(
            """
            from typing import Any

            import jax
            import optax
            from flax import linen as nn


            class Net(nn.Module):
                width: int

                @nn.compact
                def __call__(self, x: jax.Array) -> jax.Array:
                    return nn.relu(nn.Dense(self.width)(x))


            def build() -> None:
                model = Net(width=4)
                key = jax.random.PRNGKey(0)
                xs = jax.numpy.ones((2, 3))
                params: Any = model.init(key, xs)
                out = model.apply(params, xs)
                tx = optax.adam(1e-3)
                state: Any = tx.init(params)
                updates, state = tx.update(params, state)
                print(out, updates)
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    bundle = analyze_paths(open_project(path), [path], backend="python")
    errors = [d for d in bundle.diagnostics if d.severity.name == "ERROR"]
    assert not errors, [f"{d.code}: {d.message}" for d in errors]
