"""Which module-level names the rest of the project assigns to.

`Final` is a promise about the whole program, so proving it takes the whole
program: `config.LIMIT = 5` in a file that was not being converted rebinds the
name just as surely as a second assignment at home would. This index is built
once over every source under the project root -- including files outside the
current conversion -- and answers the one question the converter asks.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["GlobalWriteIndex", "build_write_index"]

#: Directories whose sources are not the project's own.
_SKIP = frozenset(
    {".venv", "venv", ".git", "__pycache__", "build", "dist", ".ppy-cache", ".tox", "node_modules"}
)


@dataclass(slots=True)
class GlobalWriteIndex:
    """`module-as-imported -> names` for every cross-module attribute write."""

    writes: dict[str, set[str]] = field(default_factory=dict)
    #: Modules hit by a `setattr` whose name is not a literal, or by writes
    #: through an alias the scan could not resolve. No name in them is
    #: provably unbound.
    dynamic: set[str] = field(default_factory=set)

    def can_emit_final(self, module: str, name: str) -> bool:
        """Is `module.name` free of assignments anywhere else in the project?

        `module` is the converter's qualified name for the file; imports may
        reach it under a shorter spelling, so both directions of suffix match
        count as the same module. Missing evidence is a no: a module under a
        dynamic `setattr` proves nothing about any of its names.
        """
        for target, names in self.writes.items():
            if _same_module(module, target) and name in names:
                return False
        return not any(_same_module(module, target) for target in self.dynamic)


def _same_module(module: str, target: str) -> bool:
    return module == target or module.endswith("." + target) or target.endswith("." + module)


def build_write_index(root: Path) -> GlobalWriteIndex:
    index = GlobalWriteIndex()
    for path in _sources(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        _scan(tree, index)
    return index


def _sources(root: Path):  # type: ignore[no-untyped-def]
    for suffix in ("*.py", "*.ppy"):
        for path in root.rglob(suffix):
            if not any(part in _SKIP for part in path.parts):
                yield path


def _scan(tree: ast.Module, index: GlobalWriteIndex) -> None:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.partition(".")[0]] = (
                    alias.name if alias.asname else alias.name.partition(".")[0]
                )
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                # `from package import foo` may bind the module `package.foo`;
                # recording it costs nothing when it turns out to be a value.
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def resolve(value: ast.expr) -> str | None:
        """The imported module a dotted expression names, if any."""
        parts: list[str] = []
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if not isinstance(value, ast.Name) or value.id not in aliases:
            return None
        parts.append(aliases[value.id])
        return ".".join(reversed(parts))

    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = list(node.targets)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"setattr", "delattr"}
            and node.args
        ):
            module = resolve(node.args[0])
            if module is None:
                continue
            written = node.args[1] if len(node.args) > 1 else None
            if isinstance(written, ast.Constant) and isinstance(written.value, str):
                index.writes.setdefault(module, set()).add(written.value)
            else:
                index.dynamic.add(module)
            continue
        for target in targets:
            if isinstance(target, ast.Attribute):
                module = resolve(target.value)
                if module is not None:
                    index.writes.setdefault(module, set()).add(target.attr)
