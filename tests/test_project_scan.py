"""The shared project scan, and the work it exists to stop repeating.

A conversion used to walk, parse, and lexically scan the project once for the
write index and twice for the reflection index, and the reflection index
walked each module's tree once more per function it held. The answers must
be the same ones; only the work is meant to change.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from ppy_compiler.analysis.global_writes import build_write_index
from ppy_compiler.analysis.lexical import scan_module
from ppy_compiler.analysis.project_scan import scan_project
from ppy_compiler.analysis.reflection import build_reflection_index
from ppy_compiler.migration.pipeline import apply_passes, passes_for


def _project(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[tool.ppy]\n", encoding="utf-8")
    src = tmp_path / "src"
    src.mkdir()
    (src / "store.py").write_text(
        textwrap.dedent(
            """
            LIMIT = 5
            NAME = "x"

            def read(f):
                return f.__annotations__
            """
        ),
        encoding="utf-8",
    )
    (src / "user.py").write_text(
        textwrap.dedent(
            """
            import inspect
            import store
            from store import read as peek

            store.LIMIT = 6

            def shown(g):
                return inspect.signature(g)

            @peek
            def decorated(x: int) -> int:
                return x
            """
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_both_indexes_read_the_project_from_one_scan(tmp_path: Path):
    """One walk, one parse, one lexical scan per file -- and the same answers."""
    root = _project(tmp_path)
    scan = scan_project(root, ("src",))
    assert sorted(m.module for m in scan.modules) == ["store", "user"]
    assert not scan.tainted

    shared_writes = build_write_index(root, ("src",), scan=scan)
    shared_reads = build_reflection_index(root, ("src",), scan=scan)
    alone_writes = build_write_index(root, ("src",))
    alone_reads = build_reflection_index(root, ("src",))

    assert shared_writes.writes == alone_writes.writes == {"store": {"LIMIT"}}
    assert shared_reads.observed == alone_reads.observed
    assert shared_reads.dynamic is alone_reads.dynamic is False
    assert not shared_writes.can_emit_final("store", "LIMIT"), "another file assigns it"
    assert shared_writes.can_emit_final("store", "NAME")
    # `read` and `shown` look at their own parameter's annotations, so what
    # is handed to them is observed: the decorated function, by name.
    assert "user.decorated" in shared_reads.observed


def test_an_unparseable_file_taints_every_index_built_from_the_scan(tmp_path: Path):
    root = _project(tmp_path)
    (root / "src" / "broken.py").write_text("def (:\n", encoding="utf-8")
    scan = scan_project(root, ("src",))
    assert scan.tainted
    assert build_write_index(root, ("src",), scan=scan).tainted
    assert build_reflection_index(root, ("src",), scan=scan).dynamic


def test_shared_context_nodes_are_not_snapshotted():
    """`ast.Load()` is one object under every name; it is not a place in the file."""
    tree = ast.parse("import os\nx = os.sep\ny = os.name\nprint(x, y)\n")
    bindings = scan_module(tree, "m")
    contexts = {id(n) for n in ast.walk(tree) if isinstance(n, ast.expr_context)}
    assert contexts
    assert not contexts & bindings._at.keys(), "no context node was snapshotted"
    assert not any(isinstance(v, list) for v in bindings._at.values()), "nothing pending"
    assert bindings.targets_at(tree.body[1].value) == frozenset({"os.sep"})


def test_a_pending_join_is_resolved_when_asked_and_kept():
    """Two environments recorded at one node meet the first time a reader asks."""
    tree = ast.parse("import os\nos.sep\n")
    bindings = scan_module(tree, "m")
    attribute = tree.body[1].value
    first = bindings._at[id(attribute)]
    assert isinstance(first, dict)
    bindings._at[id(attribute)] = [first, {"os": frozenset({"sys"})}]
    assert bindings.targets_at(attribute) == frozenset({"os.sep", "sys.sep"})
    assert isinstance(bindings._at[id(attribute)], dict), "resolved once, kept resolved"
    assert bindings.targets_at(attribute) == frozenset({"os.sep", "sys.sep"})


def test_a_pass_that_cannot_fire_is_not_walked():
    """A module that never spells `getattr` has no `getattr` call to rewrite."""
    assert passes_for("x = 1\nprint(x)\n") == []
    names = [type(p).__name__ for p in passes_for("setattr(o, 'a', 1)\n")]
    assert names == ["LiteralAttributes"]
    names = [type(p).__name__ for p in passes_for("from importlib import import_module as imp\n")]
    assert names == ["StaticImports"], "an alias still spells the name where it is made"
    names = [type(p).__name__ for p in passes_for('globals()["A"] = 1\n')]
    assert names == ["ModuleNamespaceWrites"]

    untouched = "def f(x):\n    return x\n"
    assert apply_passes(untouched) == (untouched, [])
    rewritten, found = apply_passes('class O:\n    pass\n\no = O()\nsetattr(o, "a", 1)\n')
    assert "o.a = 1" in rewritten and found
