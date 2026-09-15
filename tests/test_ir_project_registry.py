from __future__ import annotations

import argparse
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from ppy_compiler.backend import Backend, BuildResult
from ppy_compiler.backend.llvm import available as llvm_available
from ppy_compiler.backend.llvm.link import toolchain_status
from ppy_compiler.driver import commands
from ppy_compiler.driver import emit as driver_emit
from ppy_compiler.driver.config import Config
from ppy_compiler.driver.reporting import Reporter
from ppy_compiler.ir import (
    Builder,
    Dialect,
    DialectRegistry,
    FunctionPass,
    IRModule,
    Operation,
    OpSpec,
    PassContext,
    PassManager,
    Successor,
    read,
    verify,
    write,
)
from ppy_compiler.ir.analysis import ANALYSES, Dominators, register_analysis
from ppy_compiler.ir.dialects.core import CoreDialect
from ppy_compiler.ir.transforms import DeadCodeElimination, SimplifyCFG
from ppy_compiler.plugins import Plugin, PluginRegistry

_toolchain_available, _toolchain_detail = toolchain_status()
requires_native_toolchain = pytest.mark.skipif(
    not llvm_available() or not _toolchain_available,
    reason=f"native LLVM toolchain unavailable: {_toolchain_detail}",
)


class ProjectDialect(Dialect):
    name = "project"

    def register_operations(self, registry: DialectRegistry) -> None:
        registry.add_op(OpSpec("project.jump", terminator=True, successors=1))
        registry.add_op(OpSpec("project.stop", terminator=True, successors=0))
        registry.add_op(OpSpec("project.observe", pure=True, results=0))


def _registry() -> DialectRegistry:
    registry = DialectRegistry()
    registry.register(CoreDialect())
    registry.register(ProjectDialect())
    return registry


def _project_cfg() -> tuple[IRModule, object, object, object]:
    module = IRModule("custom", {"core": 1, "project": 1})
    function = module.add_function("f", (), ())
    entry = function.add_entry_block()
    reached = function.body.add_block("reached")
    Builder(entry).create("project.jump", successors=(Successor(reached),))
    Builder(reached).create("project.stop")
    return module, function, entry, reached


def test_model_and_verifier_use_the_supplied_registry_for_control_flow() -> None:
    registry = _registry()
    module, _function, entry, reached = _project_cfg()

    jump = entry.operations[-1]
    assert jump.spec_is_terminator(registry)
    assert entry.terminator_for(registry) is jump
    assert entry.successors_for(registry) == [reached]
    assert not verify(module, registry)


def test_cfg_analysis_and_dce_keep_project_dialect_successors() -> None:
    registry = _registry()
    module, function, entry, reached = _project_cfg()
    dead = function.body.add_block("dead")
    Builder(dead).create("project.stop")
    ctx = PassContext(registry, verify_after_each=True)

    dominators = ctx.analysis("dominators", function)
    assert isinstance(dominators, Dominators)
    assert dominators.dominates(entry, reached)
    report = PassManager(ctx).add(SimplifyCFG()).add(DeadCodeElimination()).run(module)

    assert report.changed
    assert [block.name for block in function.body.blocks] == ["entry", "reached"]


def test_a_replaced_analysis_keeps_the_public_one_argument_contract() -> None:
    registry = _registry()
    _module, function, _entry, _reached = _project_cfg()
    original = ANALYSES["dominators"]
    register_analysis("dominators", lambda analyzed: analyzed.name)
    try:
        assert PassContext(registry).analysis("dominators", function) == "f"
    finally:
        ANALYSES["dominators"] = original


def test_textual_ir_round_trips_with_an_isolated_project_registry(tmp_path: Path) -> None:
    registry = _registry()
    module, _function, _entry, _reached = _project_cfg()
    path = tmp_path / "custom.ppyir"

    write(module, path, registry)
    decoded = read(path, registry)

    assert not verify(decoded, registry)
    entry = decoded.functions["f"].entry
    assert entry is not None
    assert entry.terminator_for(registry).name == "project.jump"


def test_backend_build_reads_textual_ir_with_the_project_registry(
    tmp_path: Path, monkeypatch
) -> None:
    registry = _registry()
    module, _function, _entry, _reached = _project_cfg()
    path = tmp_path / "custom.ppyir"
    write(module, path, registry)
    built: list[str] = []

    class ProjectBackend(Backend):
        name = "project-backend"
        api_version = 1

        def build(self, modules, output, context):  # type: ignore[no-untyped-def]
            assert context.registry is registry
            assert not verify(modules["custom"], context.registry)
            built.extend(modules)
            return BuildResult()

    backend = ProjectBackend()
    monkeypatch.setattr("ppy_compiler.backend.load_backend", lambda _name, _options: backend)
    config = Config(root=tmp_path)
    project = SimpleNamespace(
        config=config,
        root=tmp_path,
        plugins=SimpleNamespace(
            dialect_registry=lambda: registry,
            fingerprints=lambda: (),
        ),
    )
    options = argparse.Namespace(output=tmp_path / "out", opt_level=None)

    result = commands._build_ir_file_with_backend(
        backend.name, path, options, Reporter(stream=io.StringIO()), project
    )

    assert result == 0
    assert built == ["custom"]


@requires_native_toolchain
def test_builtin_ir_build_reads_and_lowers_with_the_project_plugins(
    tmp_path: Path, monkeypatch
) -> None:
    registry = _registry()
    module, _function, _entry, _reached = _project_cfg()
    path = tmp_path / "custom.ppyir"
    write(module, path, registry)
    lowered: list[str] = []

    class LowerProjectControlFlow(FunctionPass):
        name = "lower-project-control-flow"

        def run_on_function(self, function, ctx):  # type: ignore[no-untyped-def]
            changed = False
            for block in function.body.blocks:
                for op in list(block.operations):
                    index = block.operations.index(op)
                    if op.name == "project.jump":
                        target = Successor(op.successors[0].block, op.successors[0].arguments)
                        lowered.append(op.name)
                        op.erase()
                        block.insert(index, Operation("core.br", successors=(target,)))
                        changed = True
                    elif op.name == "project.stop":
                        lowered.append(op.name)
                        op.erase()
                        block.insert(index, Operation("core.ret"))
                        changed = True
            return changed

    class ProjectPlugin(Plugin):
        name = "project-control-flow"

        def register_dialects(self, dialects: DialectRegistry) -> None:
            dialects.register(ProjectDialect())

        def register_passes(self, manager) -> None:  # type: ignore[no-untyped-def]
            manager.register_stage_pass("after-ir-generation", LowerProjectControlFlow)

    plugins = PluginRegistry()
    plugins.register(ProjectPlugin())
    config = Config(root=tmp_path)
    project = SimpleNamespace(config=config, plugins=plugins)
    monkeypatch.setattr(driver_emit, "open_project", lambda _path: project)
    output = tmp_path / "native"

    result = driver_emit.build_ir_file(
        path,
        argparse.Namespace(output=output, opt_level=1),
        Reporter(stream=io.StringIO()),
    )

    assert result == 0
    assert lowered == ["project.jump", "project.stop"]
    assert (output / "custom.o").is_file()
    assert list(output.glob("libppy_custom.*"))
    assert (output / "ppy-bindings.json").is_file()


def test_dce_keeps_dynamic_effects_on_an_operation_declared_pure() -> None:
    registry = _registry()
    module = IRModule("effects", {"core": 1, "project": 1})
    function = module.add_function("f", (), ())
    entry = function.add_entry_block()
    entry.append(Operation("project.observe", attributes={"effects": ("write_memory",)}))
    entry.append(Operation("project.stop"))

    PassManager(PassContext(registry, verify_after_each=True)).add(DeadCodeElimination()).run(
        module
    )

    assert [op.name for op in entry.operations] == ["project.observe", "project.stop"]
