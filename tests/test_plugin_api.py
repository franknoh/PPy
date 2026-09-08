"""Plugin interface v2: the base, discovery, collisions, and the IR hooks."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ppy_compiler.analysis import types as T
from ppy_compiler.analysis.effects import Effect
from ppy_compiler.analysis.refinements import Facts
from ppy_compiler.driver.config import Config, PluginConfig
from ppy_compiler.ir import I64, Builder, Dialect, DialectRegistry, IRModule, Operation, OpSpec
from ppy_compiler.ir.dialects import core
from ppy_compiler.plugins.base import (
    PLUGIN_API_VERSION,
    CallResult,
    DialectOperationSpec,
    FallbackSpec,
    Lowering,
    Plugin,
    PluginError,
    PluginRegistry,
    RejectSpec,
)
from ppy_compiler.plugins.jax_plugin import JaxPlugin
from ppy_compiler.plugins.numpy_plugin import NumPyPlugin
from ppy_compiler.plugins.pandas_plugin import PandasPlugin
from ppy_compiler.plugins.pyarrow_plugin import PyArrowPlugin
from ppy_compiler.plugins.pydantic_plugin import PydanticPlugin
from ppy_compiler.plugins.registry import AVAILABLE, discover_external, load_plugins
from ppy_compiler.plugins.scipy_plugin import SciPyPlugin
from ppy_compiler.plugins.torch_plugin import TorchPlugin
from ppy_compiler.plugins.uvicorn_plugin import UvicornPlugin

BUILTINS = (
    NumPyPlugin,
    TorchPlugin,
    JaxPlugin,
    UvicornPlugin,
    PydanticPlugin,
    SciPyPlugin,
    PandasPlugin,
    PyArrowPlugin,
)
FLOAT = (T.FLOAT, Facts())
ARRAY = (T.Instance("numpy.ndarray", (), ("numpy.ndarray", "object")), Facts())
FRAME = (T.Instance("pandas.DataFrame", (), ("pandas.DataFrame", "object")), Facts())


# -- the interface ----------------------------------------------------------


def test_every_builtin_plugin_is_a_plugin_of_this_interface():
    for factory in BUILTINS:
        plugin = factory()
        assert isinstance(plugin, Plugin)
        assert plugin.api_version == PLUGIN_API_VERSION
        assert plugin.name in AVAILABLE
        assert plugin.modules
        assert f"api{PLUGIN_API_VERSION}" in plugin.fingerprint() or plugin.name in {
            "numpy",
            "torch",
            "jax",
            "uvicorn",
            "pydantic",
        }
        # Every hook exists and answers "nothing" for a name no plugin owns.
        assert plugin.call("nothing.nobody", [], {}) is None
        assert plugin.attribute_type("nothing.nobody") is None
        assert plugin.instance_attribute("nothing.Nobody", "x") is None
        assert plugin.subscript("nothing.Nobody", is_slice=False, tupled=False) is None
        assert plugin.operator("<=>") is None
        assert plugin.call_alias("nothing.Nobody") is None
        assert plugin.decorator_semantics("nothing.deco") is None
        assert plugin.stage(["nothing.deco"]) is None


def test_the_registry_refuses_the_wrong_interface_and_a_non_plugin():
    class Old(Plugin):
        name = "old"
        modules = ("old",)
        api_version = 1

    registry = PluginRegistry()
    with pytest.raises(
        PluginError, match="written against plugin interface 1; this compiler speaks 2"
    ):
        registry.register(Old())
    with pytest.raises(PluginError, match="is not a Plugin"):
        registry.register(object())  # type: ignore[arg-type]


def test_two_plugins_claiming_one_module_is_a_problem_not_a_winner():
    class A(Plugin):
        name = "a"
        modules = ("foo", "foo.bar")

    class B(Plugin):
        name = "b"
        modules = ("foo",)

    registry = PluginRegistry()
    registry.register(A())
    with pytest.raises(PluginError, match="plugin 'a' and plugin 'b' both claim module 'foo'"):
        registry.register(B())
    assert registry.for_module("foo").name == "a", "the first registration stands"


def test_lowering_specs_carry_their_kind():
    result = CallResult(T.FLOAT, lowering=DialectOperationSpec("special", "erf"))
    assert result.kind is Lowering.DIALECT_OPERATION
    assert result.spec == DialectOperationSpec("special", "erf")
    bare = CallResult(T.FLOAT, lowering=Lowering.REJECT, reason="no")
    assert bare.kind is Lowering.REJECT
    assert bare.spec == RejectSpec("no")
    assert CallResult(T.FLOAT).spec == FallbackSpec("")


# -- discovery ----------------------------------------------------------------


def _install_fake_plugin(tmp_path: Path, monkeypatch, source: str, name: str = "fakeplug") -> None:
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(source), encoding="utf-8")
    info = tmp_path / f"{name}-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n")
    (info / "entry_points.txt").write_text(f"[ppy.plugins]\n{name} = {name}:create_plugin\n")
    monkeypatch.syspath_prepend(str(tmp_path))


FAKE = """
    from ppy_compiler.plugins.base import Plugin

    LOADED = []


    class FakePlugin(Plugin):
        name = "fakeplug"
        modules = ("fakelib",)


    def create_plugin(options):
        LOADED.append(options)
        return FakePlugin(options)
    """


def test_an_installed_plugin_is_discovered_but_loaded_only_when_the_project_says_so(
    tmp_path: Path, monkeypatch
):
    _install_fake_plugin(tmp_path, monkeypatch, FAKE)
    assert "fakeplug" in discover_external()
    import sys

    sys.modules.pop("fakeplug", None)
    registry = load_plugins(Config())
    assert registry.for_module("fakelib") is None
    assert "fakeplug" not in sys.modules, "discovery does not import"
    config = Config()
    config.plugins["fakeplug"] = PluginConfig(options={"level": 3})
    registry = load_plugins(config)
    plugin = registry.for_module("fakelib")
    assert plugin is not None and plugin.name == "fakeplug"
    assert sys.modules["fakeplug"].LOADED == [{"level": 3}]
    config.plugins["fakeplug"] = PluginConfig(enabled=False)
    assert load_plugins(config).for_module("fakelib") is None


def test_a_plugin_that_collides_with_a_builtin_is_reported_not_registered(
    tmp_path: Path, monkeypatch
):
    _install_fake_plugin(
        tmp_path,
        monkeypatch,
        """
        from ppy_compiler.plugins.base import Plugin


        class Rival(Plugin):
            name = "rival"
            modules = ("numpy",)


        def create_plugin(options):
            return Rival(options)
        """,
        name="rival",
    )
    config = Config()
    config.plugins["rival"] = PluginConfig()
    registry = load_plugins(config)
    assert registry.problems == ["plugin 'numpy' and plugin 'rival' both claim module 'numpy'"]
    assert registry.for_module("numpy").name == "numpy"


def test_a_plugin_problem_is_a_diagnostic(write, analyze, tmp_path: Path, monkeypatch):
    _install_fake_plugin(
        tmp_path,
        monkeypatch,
        "def create_plugin(options):\n    raise RuntimeError('no such library')\n",
        name="broken",
    )
    (tmp_path / "pyproject.toml").write_text(
        "[tool.ppy]\nstrict = true\n\n[tool.ppy.plugins.broken]\nenabled = true\n", encoding="utf-8"
    )
    path = write("m.ppy", "def f() -> int:\n    return 1\n")
    bundle = analyze(path)
    codes = {d.code: d.message for d in bundle.diagnostics.sorted()}
    assert "E1901" in codes
    assert "plugin 'broken' could not be loaded: no such library" in codes["E1901"]


# -- the IR hooks --------------------------------------------------------------


def test_a_plugin_registers_a_dialect_a_pattern_and_a_pass_that_the_pipeline_runs():
    from ppy_compiler.ir import FunctionPass, Pattern, RewriteResult
    from ppy_compiler.ir.dialects.core import CoreDialect

    ran: list[str] = []

    class Twice(Pattern):
        root = "toy.twice"

        def match_and_rewrite(self, op: Operation, rewriter) -> RewriteResult:
            b = rewriter.builder(op)
            rewriter.replace_op(op, [core.add(b, op.operands[0], op.operands[0], overflow="wrap")])
            return RewriteResult.success()

    class ToyDialect(Dialect):
        name = "toy"

        def register_operations(self, registry: DialectRegistry) -> None:
            registry.add_op(OpSpec("toy.twice", pure=True, operands=1, results=1))

    class Note(FunctionPass):
        name = "note"
        preserves = "all"

        def run_on_function(self, function, ctx):
            ran.append(function.name)
            return False

    class ToyPlugin(Plugin):
        name = "toy"
        modules = ("toylib",)

        def register_dialects(self, registry):
            registry.register(ToyDialect())

        def register_patterns(self, registry):
            registry.add_pattern(Twice())

        def register_passes(self, manager):
            manager.register_stage_pass("after-canonicalization", Note)

    plugins = PluginRegistry()
    plugins.register(ToyPlugin())
    registry = plugins.dialect_registry()
    assert registry.dialect("toy") is not None and registry.dialect("core") is not None
    assert isinstance(registry.dialect("core"), CoreDialect)

    module = IRModule("m", {"core": 1, "toy": 1})
    function = module.add_function("f", [("x", I64)], [I64])
    entry = function.add_entry_block()
    b = Builder(entry)
    twice = b.create("toy.twice", (entry.arguments[0],), (I64,))
    core.ret(b, twice.result)

    from ppy_compiler.backend.llvm.ir_pipeline import optimize

    optimize(module, 1, plugins)
    assert [op.name for op in function.operations()] == ["core.add", "core.ret"]
    assert ran == ["f"]


def test_a_plugin_pass_that_breaks_the_ir_is_named():
    from ppy_compiler.backend.llvm.ir_pipeline import optimize
    from ppy_compiler.ir import Pass, PassVerificationError

    class Breaker(Pass):
        name = "toy-breaker"

        def run(self, module, ctx):
            module.functions["f"].entry.operations[-1].erase()
            return True

    class ToyPlugin(Plugin):
        name = "toy"
        modules = ("toylib",)

        def register_passes(self, manager):
            manager.register_stage_pass("before-backend", Breaker)

    plugins = PluginRegistry()
    plugins.register(ToyPlugin())
    module = IRModule("m")
    function = module.add_function("f", [("x", I64)], [I64])
    entry = function.add_entry_block()
    core.ret(Builder(entry), entry.arguments[0])
    with pytest.raises(PassVerificationError, match="pass 'toy-breaker' broke the IR"):
        optimize(module, 1, plugins)


# -- the new builtins -----------------------------------------------------------


def test_scipy_types_its_families_and_names_their_dialects():
    plugin = SciPyPlugin()
    erf = plugin.call("scipy.special.erf", [FLOAT], {})
    assert erf is not None and erf.type == T.FLOAT
    assert erf.spec == DialectOperationSpec("special", "erf")
    assert not erf.effects.violations()
    vector = plugin.call("scipy.special.erf", [ARRAY], {})
    assert vector is not None and vector.type == ARRAY[0]
    assert Effect.ALLOC in vector.effects
    fft = plugin.call("scipy.fft.rfft", [ARRAY], {})
    assert fft is not None and fft.spec == DialectOperationSpec("fft", "rfft")
    solve = plugin.call("scipy.linalg.solve", [ARRAY, ARRAY], {})
    assert solve is not None and solve.type == ARRAY[0]
    assert solve.spec == DialectOperationSpec("linalg", "solve")
    svd = plugin.call("scipy.linalg.svd", [ARRAY], {})
    assert svd is not None and isinstance(svd.type, T.Tuple_) and len(svd.type.items) == 3
    minimize = plugin.call("scipy.optimize.minimize", [(T.ANY, Facts()), ARRAY], {})
    assert minimize is not None and Effect.PYTHON_CALLBACK in minimize.effects
    assert minimize.kind is Lowering.PYTHON_FALLBACK
    csr = plugin.call("scipy.sparse.csr_matrix", [ARRAY], {})
    assert csr is not None and csr.type.name == "scipy.sparse.csr_matrix"
    assert csr.spec == DialectOperationSpec("sparse", "construct", (("format", "csr"),))
    attribute = plugin.instance_attribute("scipy.sparse.csr_matrix", "toarray")
    assert attribute is not None and attribute[0].ret == ARRAY[0]
    assert plugin.operator("@") == "scipy.sparse.matmul"
    assert plugin.call("scipy.special.nope", [FLOAT], {}) is None
    assert plugin.fingerprint().startswith("v1:api2:scipy=")


def test_pandas_keeps_frames_frames_and_says_what_it_does_not_model():
    plugin = PandasPlugin()
    frame = plugin.call("pandas.DataFrame", [(T.ANY, Facts())], {})
    assert frame is not None and frame.type.name == "pandas.DataFrame"
    read = plugin.call("pandas.read_csv", [(T.STR, Facts())], {})
    assert read is not None and Effect.IO in read.effects
    total = plugin.call("pandas.DataFrame.sum", [FRAME], {})
    assert total is not None and total.type.name == "pandas.Series"
    assert total.spec == DialectOperationSpec("columnar", "aggregate")
    filled = plugin.call(
        "pandas.Series.fillna", [(T.Instance("pandas.Series", (), ()), Facts())], {}
    )
    assert filled is not None and filled.spec == DialectOperationSpec("columnar", "fill_null")
    applied = plugin.call("pandas.DataFrame.apply", [FRAME], {})
    assert applied is not None and Effect.PYTHON_CALLBACK in applied.effects
    assert applied.kind is Lowering.PYTHON_FALLBACK
    shape = plugin.instance_attribute("pandas.DataFrame", "shape")
    assert shape is not None and isinstance(shape[0], T.Tuple_)
    column = plugin.subscript("pandas.DataFrame", is_slice=False, tupled=False)
    assert column is not None and column[0] == T.ANY, "a key's type is not known here"
    assert plugin.operator("+") == "pandas.add"
    added = plugin.call("pandas.add", [FRAME, FRAME], {})
    assert added is not None and added.type.name == "pandas.DataFrame"
    assert plugin.call("pandas.DataFrame.nothing", [FRAME], {}) is None


def test_pyarrow_types_arrow_as_arrow_and_names_compute_as_columnar():
    plugin = PyArrowPlugin()
    array = plugin.call("pyarrow.array", [(T.list_of(T.INT), Facts())], {})
    assert array is not None and array.type.name == "pyarrow.Array"
    chunked = (T.Instance("pyarrow.ChunkedArray", (), ()), Facts())
    added = plugin.call("pyarrow.compute.add", [chunked, chunked], {})
    assert added is not None and added.type.name == "pyarrow.ChunkedArray"
    assert added.spec == DialectOperationSpec("columnar", "add")
    total = plugin.call("pyarrow.compute.sum", [chunked], {})
    assert total is not None and total.type.name == "pyarrow.Scalar"
    table = (T.Instance("pyarrow.Table", (), ()), Facts())
    filtered = plugin.call("pyarrow.compute.filter", [table, chunked], {})
    assert filtered is not None and filtered.type.name == "pyarrow.Table"
    read = plugin.call("pyarrow.parquet.read_table", [(T.STR, Facts())], {})
    assert read is not None and Effect.IO in read.effects and read.type.name == "pyarrow.Table"
    column = plugin.subscript("pyarrow.Table", is_slice=False, tupled=False)
    assert column is not None and column[0].name == "pyarrow.ChunkedArray"
    rows = plugin.instance_attribute("pyarrow.Table", "num_rows")
    assert rows is not None and rows[0] == T.INT
    to_numpy = plugin.call(
        "pyarrow.Array.to_numpy", [(T.Instance("pyarrow.Array", (), ()), Facts())], {}
    )
    assert to_numpy is not None and "zero-copy" in to_numpy.spec.reason
    assert plugin.call("pyarrow.int64", [], {}).type.name == "pyarrow.DataType"
    assert plugin.call("pyarrow.compute.nothing", [chunked], {}) is None


def test_the_new_plugins_type_a_program_without_the_libraries(write, analyze):
    path = write(
        "frames.ppy",
        """
        import pandas as pd
        import pyarrow as pa
        import scipy.special


        def total(frame: pd.DataFrame) -> pd.Series:
            return frame.fillna(0).sum()


        def rows(table: pa.Table) -> int:
            return table.num_rows


        def smooth(x: float) -> float:
            return scipy.special.erf(x)
        """,
    )
    bundle = analyze(path)
    assert [d.code for d in bundle.diagnostics.sorted() if d.severity.name == "ERROR"] == []
    functions = bundle.symbols.modules["frames"].functions
    assert str(functions["total"].ret) == "pandas.Series"
    assert str(functions["rows"].ret) == "int"
    notes = bundle.analysis.modules["frames"].lowerings
    assert any(
        n.lowering == "DialectOperation" and n.qualname.endswith("erf") for n in notes.values()
    )
