"""Who in the project looks at annotations at runtime.

Materializing an inferred annotation is invisible -- until someone reads
`f.__annotations__`, calls `inspect.signature(f)`, or prints the module's
`__annotations__`, at which point the conversion has changed the program's
output. This scan finds those readers across the whole project so the
converter can leave the observed objects exactly as their author wrote them.

Resolution is best-effort and failure is conservative: a reflective call
whose target cannot be named blocks materialization everywhere, because the
target could be anything.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from ..frontend.modules import resolve_module_name

__all__ = ["ReflectionIndex", "build_reflection_index"]

_SKIP = frozenset(
    {".venv", "venv", ".git", "__pycache__", "build", "dist", ".ppy-cache", ".tox", "node_modules"}
)

#: Callables whose argument's annotations become observable.
_READERS = frozenset(
    {
        "inspect.signature",
        "signature",
        "inspect.get_annotations",
        "get_annotations",
        "typing.get_type_hints",
        "get_type_hints",
    }
)


@dataclass(slots=True)
class ReflectionIndex:
    """Names whose annotations the project observes at runtime."""

    #: Dotted spellings whose `__annotations__`/signature someone reads.
    observed: set[str] = field(default_factory=set)
    #: Modules whose own `__annotations__` mapping is read.
    module_annotations: set[str] = field(default_factory=set)
    #: A reflective read whose target could not be named: everything may be
    #: observed, so nothing may be materialized.
    dynamic: bool = False

    def blocks_function(self, name: str, qualname: str) -> bool:
        if self.dynamic:
            return True
        for spelling in self.observed:
            tail = spelling.rpartition(".")[2]
            if tail == name or qualname == spelling or qualname.endswith("." + spelling):
                return True
        return False

    def blocks_module_globals(self, module: str) -> bool:
        if self.dynamic:
            return True
        return any(
            module == seen or module.endswith("." + seen) or seen.endswith("." + module)
            for seen in self.module_annotations
        )


def build_reflection_index(
    root: Path, source_roots: tuple[str, ...] = ("src", ".")
) -> ReflectionIndex:
    index = ReflectionIndex()
    search_paths = [root / entry for entry in source_roots if (root / entry).is_dir()]
    if not search_paths:
        search_paths = [root]
    for path in _sources(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            index.dynamic = True
            continue
        _scan(tree, resolve_module_name(path, search_paths), index)
    return index


def _sources(root: Path):  # type: ignore[no-untyped-def]
    for suffix in ("*.py", "*.ppy"):
        for path in root.rglob(suffix):
            if not any(part in _SKIP for part in path.parts):
                yield path


def _dotted(node: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _scan(tree: ast.Module, module: str, index: ReflectionIndex) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "__annotations__":
            spelling = _dotted(node.value)
            if spelling is None:
                index.dynamic = True
            else:
                # `store.__annotations__` observes a module as easily as a
                # function; both records are cheap and both are honest.
                index.observed.add(spelling)
                index.module_annotations.add(spelling)
        elif isinstance(node, ast.Name) and node.id == "__annotations__":
            # A bare read reaches the module's own mapping, wherever it
            # appears in the file.
            index.module_annotations.add(module)
        elif isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name not in _READERS or not node.args:
                continue
            spelling = _dotted(node.args[0])
            if spelling is None:
                index.dynamic = True
            else:
                index.observed.add(spelling)
                index.module_annotations.add(spelling)
